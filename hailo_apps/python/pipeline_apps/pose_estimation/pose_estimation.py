# region imports
# Standard library imports
import os
import time
import argparse
import subprocess
import threading
import urllib.request
import urllib.parse
import json
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


# COCO pose keypoint indices
KEYPOINTS = {
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

# Number of frames to keep recording after the last fall detection
TRAILING_FRAMES = 150

# Minimum seconds between fall alert logs for a single tracked person
INITIAL_FALL_DEBOUNCE_SECONDS = 5.0
MAX_FALL_DEBOUNCE_SECONDS = 3600.0

# Seconds a person must remain untracked as falling to reset the alarm backoff
FALL_RESET_SECONDS = 10.0


# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.last_alert_time = {}
        self.last_seen_fallen_time = {}
        self.fall_backoff = {}
        # Safe zones (normalized coords: x_min, y_min, x_max, y_max).
        # Persons whose bounding box center falls inside a safe zone are treated as resting.
        self.safe_zones = [
            (0.5, 0.5, 1.0, 1.0)  # bottom-right quadrant
        ]
        self.event_storage_path = None
        self.show_time = False
        self.telegram_token = None
        self.telegram_chat_id = None
        self.video_writer = None
        self.frames_since_last_fall = 0


# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------
def app_callback(element, buffer, user_data):
    if buffer is None:
        hailo_logger.warning("Received None buffer.")
        return

    hailo_logger.debug("Processing frame %d", user_data.get_count())

    pad = element.get_static_pad("src")
    format, width, height = get_caps_from_pad(pad)

    # Decode frame once; used for both display and video recording
    frame_bgr = None
    if (user_data.use_frame or user_data.event_storage_path) and format and width and height:
        frame = get_numpy_from_buffer(buffer, format, width, height)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    fall_detected = False

    for detection in detections:
        if detection.get_label() != "person":
            continue

        bbox = detection.get_bbox()
        track_id = 0
        track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        if len(track) == 1:
            track_id = track[0].get_id()

        hailo_logger.debug("Detection: ID=%d Confidence=%.2f", track_id, detection.get_confidence())

        landmarks = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
        if landmarks:
            points = landmarks[0].get_points()
            if check_fall_detection(user_data, track_id, bbox, points, frame_bgr):
                fall_detected = True

            if user_data.use_frame and frame_bgr is not None:
                for eye in ["left_eye", "right_eye"]:
                    point = points[KEYPOINTS[eye]]
                    x = int((point.x() * bbox.width() + bbox.xmin()) * width)
                    y = int((point.y() * bbox.height() + bbox.ymin()) * height)
                    cv2.circle(frame_bgr, (x, y), 5, (0, 255, 0), -1)

    if user_data.use_frame and frame_bgr is not None:
        user_data.set_frame(frame_bgr)

    if user_data.event_storage_path and frame_bgr is not None:
        _write_video_frame(user_data, frame_bgr, width, height, fall_detected)


def _write_video_frame(user_data, frame_bgr, width, height, fall_detected):
    """Start/stop event recording and write the current frame."""
    
    # Are we in an active recording state?
    is_writing = fall_detected or (
        user_data.video_writer is not None and user_data.frames_since_last_fall < TRAILING_FRAMES
    )

    if not is_writing:
        # Time to close out an active recording?
        if user_data.video_writer is not None:
            hailo_logger.info("Fall resolved, finishing event video.")
            user_data.video_writer.release()
            user_data.video_writer = None
        return

    # We are writing! Prepare the frame safely.
    frame_to_write = frame_bgr.copy()

    if user_data.show_time:
        cv2.putText(
            frame_to_write,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if fall_detected:
        user_data.frames_since_last_fall = 0
        if user_data.video_writer is None:
            os.makedirs(user_data.event_storage_path, exist_ok=True)
            filename = os.path.join(
                user_data.event_storage_path,
                f"fall_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
            )
            user_data.video_writer = cv2.VideoWriter(
                filename, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height)
            )
            hailo_logger.info("Started recording fall event video to: %s", filename)
        user_data.video_writer.write(frame_to_write)
    else:
        user_data.frames_since_last_fall += 1
        user_data.video_writer.write(frame_to_write)


def send_telegram_alert(token, chat_id, message, frame_bgr=None, track_id=0, current_time=0):
    if not token or not chat_id:
        return
        
    try:
        if frame_bgr is not None:
            snapshot_path = f"/tmp/fall_snapshot_{track_id}_{int(current_time)}.jpg"
            cv2.imwrite(snapshot_path, frame_bgr)
            subprocess.Popen(
                [
                    "curl", "-s",
                    "-F", f"chat_id={chat_id}",
                    "-F", f"photo=@{snapshot_path}",
                    "-F", f"caption={message}",
                    f"https://api.telegram.org/bot{token}/sendPhoto"
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        else:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            payload = urllib.parse.urlencode({
                "chat_id": chat_id,
                "text": message,
            }).encode("utf-8")
            
            req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/x-www-form-urlencoded'})
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status != 200:
                    hailo_logger.error("Telegram API returned status %d", response.status)
    except urllib.error.HTTPError as e:
        error_msg = e.read().decode('utf-8')
        hailo_logger.error("Telegram API HTTPError: %s - %s", e, error_msg)
    except Exception as e:
        hailo_logger.error("Failed to send Telegram alert: %s", e)


def check_fall_detection(user_data, track_id, bbox, points, frame_bgr):
    # 1. Bounding box wider than tall → person is horizontal
    is_horizontal_shape = bbox.width() / (bbox.height() + 1e-6) > 1.0

    # 2. Torso angle: horizontal when hip-to-shoulder dx exceeds dy
    left_shoulder = points[KEYPOINTS["left_shoulder"]]
    right_shoulder = points[KEYPOINTS["right_shoulder"]]
    left_hip = points[KEYPOINTS["left_hip"]]
    right_hip = points[KEYPOINTS["right_hip"]]

    shoulder_mid_x = (left_shoulder.x() + right_shoulder.x()) / 2
    shoulder_mid_y = (left_shoulder.y() + right_shoulder.y()) / 2
    hip_mid_x = (left_hip.x() + right_hip.x()) / 2
    hip_mid_y = (left_hip.y() + right_hip.y()) / 2

    is_torso_horizontal = abs(hip_mid_x - shoulder_mid_x) > abs(hip_mid_y - shoulder_mid_y)

    # 3. Nose below hips → head is near the ground
    nose = points[KEYPOINTS["nose"]]
    is_head_low = nose.y() > left_hip.y() and nose.y() > right_hip.y()

    if not (is_horizontal_shape and is_torso_horizontal and is_head_low):
        return False

    # Check whether the person is inside a predefined safe zone (e.g. a bed)
    mid_x = bbox.xmin() + bbox.width() / 2.0
    mid_y = bbox.ymin() + bbox.height() / 2.0

    for z_xmin, z_ymin, z_xmax, z_ymax in user_data.safe_zones:
        if z_xmin <= mid_x <= z_xmax and z_ymin <= mid_y <= z_ymax:
            hailo_logger.debug("Horizontal pose in safe zone (Person ID %d) — resting.", track_id)
            return False

    current_time = time.time()

    # If the person has not been seen falling for over FALL_RESET_SECONDS seconds, reset their backoff
    if current_time - user_data.last_seen_fallen_time.get(track_id, 0.0) > FALL_RESET_SECONDS:
        user_data.fall_backoff[track_id] = INITIAL_FALL_DEBOUNCE_SECONDS

    user_data.last_seen_fallen_time[track_id] = current_time

    # Exponential backoff: alert at most once every current_backoff seconds per tracked person
    current_backoff = user_data.fall_backoff.get(track_id, INITIAL_FALL_DEBOUNCE_SECONDS)
    
    if current_time - user_data.last_alert_time.get(track_id, 0.0) > current_backoff:
        user_data.last_alert_time[track_id] = current_time
        user_data.fall_backoff[track_id] = min(current_backoff * 2, MAX_FALL_DEBOUNCE_SECONDS)
        
        print(f"FALL DETECTED: Person ID {track_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (Next alert in {user_data.fall_backoff[track_id]}s)")

        if user_data.telegram_token and user_data.telegram_chat_id:
            alert_msg = f"⚠️ FALL DETECTED!\nPerson ID: {track_id}\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            image_to_send = frame_bgr.copy() if frame_bgr is not None else None
            t = threading.Thread(
                target=send_telegram_alert,
                args=(user_data.telegram_token, user_data.telegram_chat_id, alert_msg, image_to_send, track_id, current_time),
                daemon=True
            )
            t.start()

    return True


def main():
    hailo_logger.info("Starting Pose Estimation App.")
    user_data = user_app_callback_class()

    parser = get_pipeline_parser()
    parser.add_argument(
        "--show-time",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay current date and time on the video",
    )
    parser.add_argument(
        "--event-storage-path",
        type=str,
        default="/tmp",
        help="Directory for saving event videos when a fall is detected",
    )

    parser.add_argument(
        "--telegram-token",
        type=str,
        default=None,
        help="Telegram Bot Token for sending remote fall alerts",
    )
    parser.add_argument(
        "--telegram-chat-id",
        type=str,
        default=None,
        help="Telegram Chat ID to receive remote fall alerts",
    )

    app = GStreamerPoseEstimationApp(app_callback, user_data, parser=parser)
    user_data.event_storage_path = getattr(app.options_menu, "event_storage_path", "/tmp")
    user_data.show_time = getattr(app.options_menu, "show_time", True)
    user_data.telegram_token = getattr(app.options_menu, "telegram_token", None)
    user_data.telegram_chat_id = getattr(app.options_menu, "telegram_chat_id", None)

    app.run()

    if user_data.video_writer is not None:
        user_data.video_writer.release()
        user_data.video_writer = None


if __name__ == "__main__":
    main()
