# region imports
# Standard library imports
import os
import time
from datetime import datetime
os.environ["GST_PLUGIN_FEATURE_RANK"] = "vaapidecodebin:NONE"

# Third-party imports
import gi

gi.require_version("Gst", "1.0")
import cv2

# Local application-specific imports
import hailo
from gi.repository import Gst

from hailo_apps.python.pipeline_apps.pose_estimation.pose_estimation_pipeline import (
    GStreamerPoseEstimationApp,
)
from hailo_apps.python.core.common.buffer_utils import (
    get_caps_from_pad,
    get_numpy_from_buffer,
)

from hailo_apps.python.core.common.hailo_logger import get_logger
from hailo_apps.python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.python.core.common.core import get_pipeline_parser

hailo_logger = get_logger(__name__)
# endregion imports


# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.last_fall_time = {}
        # Define Exclusion Zones (normalized coords: x_min, y_min, x_max, y_max)
        # Bounding box centers within these zones are treated as safe resting.
        # Format is roughly: top-left x, top-left y, bottom-right x, bottom-right y.
        # Example: the bottom-right quadrant of the camera's view (0.5 to 1.0)
        self.safe_zones = [
            (0.5, 0.5, 1.0, 1.0)
        ]


# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------
def app_callback(element, buffer, user_data):
    hailo_logger.debug("Callback triggered. Current frame count=%d", user_data.get_count())

    # Note: Frame counting is handled automatically by the framework wrapper
    if buffer is None:
        hailo_logger.warning("Received None buffer.")
        return

    hailo_logger.debug("Processing frame %d", user_data.get_count())
    string_to_print = f"Frame count: {user_data.get_count()}\n"

    pad = element.get_static_pad("src")
    format, width, height = get_caps_from_pad(pad)

    frame = None
    if user_data.use_frame and format and width and height:
        frame = get_numpy_from_buffer(buffer, format, width, height)

    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    keypoints = get_keypoints()

    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()

        if label == "person":
            track_id = 0
            track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            if len(track) == 1:
                track_id = track[0].get_id()

            string_to_print += (
                f"Detection: ID: {track_id} Label: {label} Confidence: {confidence:.2f}\n"
            )

            landmarks = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
            if landmarks:
                points = landmarks[0].get_points()
                
                check_fall_detection(user_data, track_id, bbox, points, keypoints)

                for eye in ["left_eye", "right_eye"]:
                    keypoint_index = keypoints[eye]
                    point = points[keypoint_index]
                    x = int((point.x() * bbox.width() + bbox.xmin()) * width)
                    y = int((point.y() * bbox.height() + bbox.ymin()) * height)
                    string_to_print += f"{eye}: x: {x:.2f} y: {y:.2f}\n"
                    if user_data.use_frame:
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    if user_data.use_frame:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)

    # print(string_to_print)
    return


def check_fall_detection(user_data, track_id, bbox, points, keypoints):
    # 1. Bounding Box Aspect Ratio Check
    aspect_ratio = bbox.width() / (bbox.height() + 1e-6)
    is_horizontal_shape = aspect_ratio > 1.0

    # 2. Torso Angle Check
    left_shoulder = points[keypoints["left_shoulder"]]
    right_shoulder = points[keypoints["right_shoulder"]]
    left_hip = points[keypoints["left_hip"]]
    right_hip = points[keypoints["right_hip"]]

    shoulder_mid_x = (left_shoulder.x() + right_shoulder.x()) / 2
    shoulder_mid_y = (left_shoulder.y() + right_shoulder.y()) / 2
    hip_mid_x = (left_hip.x() + right_hip.x()) / 2
    hip_mid_y = (left_hip.y() + right_hip.y()) / 2

    dx = hip_mid_x - shoulder_mid_x
    dy = hip_mid_y - shoulder_mid_y

    is_torso_horizontal = abs(dx) > abs(dy)

    # 3. Head Position Check
    nose = points[keypoints["nose"]]
    
    # Is the nose below the hips?
    is_head_low = nose.y() > left_hip.y() and nose.y() > right_hip.y()

    # Combine heuristics
    if is_torso_horizontal and is_head_low and is_horizontal_shape:
        in_safe_zone = False
        
        # Compute center of the person's bounding box
        mid_x = bbox.xmin() + (bbox.width() / 2.0)
        mid_y = bbox.ymin() + (bbox.height() / 2.0)
        
        # Check if center falls inside any predefined exclusion zone
        for (z_xmin, z_ymin, z_xmax, z_ymax) in user_data.safe_zones:
            if z_xmin <= mid_x <= z_xmax and z_ymin <= mid_y <= z_ymax:
                in_safe_zone = True
                break
        
        if not in_safe_zone:
            current_time = time.time()
            last_time = user_data.last_fall_time.get(track_id, 0.0)
            if current_time - last_time > 10.0:  # 10 seconds debounce
                user_data.last_fall_time[track_id] = current_time
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"FALL DETECTED: Person ID {track_id} at {timestamp}")
        else:
            hailo_logger.debug(f"Horizontal pose detected in SAFE ZONE (Person ID {track_id}). Interpreting as rest/sleep.")


def get_keypoints():
    return {
        "nose": 0,
        "left_eye": 1,
        "right_eye": 2,
        "left_ear": 3,
        "right_ear": 4,
        "left_shoulder": 5,
        "right_shoulder": 6,
        "left_elbow": 7,
        "right_elbow": 8,
        "left_wrist": 9,
        "right_wrist": 10,
        "left_hip": 11,
        "right_hip": 12,
        "left_knee": 13,
        "right_knee": 14,
        "left_ankle": 15,
        "right_ankle": 16,
    }


def main():
    hailo_logger.info("Starting Pose Estimation App.")
    user_data = user_app_callback_class()
    
    parser = get_pipeline_parser()
    parser.add_argument(
        "--show-time", 
        action="store_true", 
        help="Overlay current date and time on the screen"
    )

    app = GStreamerPoseEstimationApp(app_callback, user_data, parser=parser)    
    app.run()


if __name__ == "__main__":
    main()
