# region imports
# Standard library imports
import os
import datetime
os.environ["GST_PLUGIN_FEATURE_RANK"] = "vaapidecodebin:NONE"

# Third-party imports
import gi

gi.require_version("Gst", "1.0")
import cv2

# Local application-specific imports
import hailo
from gi.repository import Gst

from hailo_apps.python.pipeline_apps.detection.detection_pipeline import GStreamerDetectionApp
from hailo_apps.python.core.common.buffer_utils import (
    get_caps_from_pad,
    get_numpy_from_buffer,
)

from hailo_apps.python.core.common.hailo_logger import get_logger
from hailo_apps.python.core.gstreamer.gstreamer_app import app_callback_class

hailo_logger = get_logger(__name__)
# endregion imports


# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.new_variable = 42
        self.writer = None
        self.recording = False
        self.cooldown_counter = 0
        self.cooldown_limit = 30  # 30 frames (about 1 second at 30fps)
        self.record_clips = False

    def new_function(self):
        return "The meaning of life is: "


# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------


def app_callback(element, buffer, user_data):
    if buffer is None:
        hailo_logger.warning("Received None buffer.")
        return

    # Note: Frame counting is handled automatically by the framework wrapper
    frame_idx = user_data.get_count()
    string_to_print = f"Frame count: {frame_idx}\n"

    pad = element.get_static_pad("src")
    format_cap, width, height = get_caps_from_pad(pad)

    # Check if recording or display is requested
    record_clips = getattr(user_data, "record_clips", False)

    frame = None
    if (record_clips or user_data.use_frame) and format_cap is not None and width is not None and height is not None:
        frame = get_numpy_from_buffer(buffer, format_cap, width, height)

    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    # Filter detections to target mouse, rat, and person
    target_labels = {"person", "mouse", "rat"}
    small_animals = []
    for detection in detections:
        label = detection.get_label()
        if label in target_labels:
            small_animals.append(detection)
        else:
            roi.remove_object(detection)

    detection_count = len(small_animals)
    for detection in small_animals:
        # Get track ID
        track_id = 0
        track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        if len(track) == 1:
            track_id = track[0].get_id()
        string_to_print += (
            f"Detection: ID: {track_id} Label: {detection.get_label()} Confidence: {detection.get_confidence():.2f}\n"
        )

    # Dynamic clip recording logic
    if (record_clips or user_data.use_frame) and frame is not None:
        # Convert RGB → BGR for OpenCV processing and saving
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Draw bounding boxes and track IDs directly on the video clip frames
        for detection in small_animals:
            bbox = detection.get_bbox()
            ymin = int(bbox.ymin() * height)
            xmin = int(bbox.xmin() * width)
            ymax = int(bbox.ymax() * height)
            xmax = int(bbox.xmax() * width)
            
            # Clip bounds to frame dimensions
            ymin, ymax = max(0, ymin), min(height, ymax)
            xmin, xmax = max(0, xmin), min(width, xmax)

            track_id = 0
            track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            if len(track) == 1:
                track_id = track[0].get_id()

            label_text = f"{detection.get_label()} ID:{track_id} ({detection.get_confidence():.2f})"
            cv2.rectangle(frame_bgr, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame_bgr, label_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if record_clips:
            object_present = len(small_animals) > 0
            if object_present:
                user_data.cooldown_counter = user_data.cooldown_limit
                
                if not user_data.recording:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{small_animals[0].get_label()}_{timestamp}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    fps = 30.0
                    user_data.writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
                    user_data.recording = True
                    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Target entered. Started clip recording: {filename}")
                    
                if user_data.writer is not None:
                    user_data.writer.write(frame_bgr)
            else:
                if user_data.recording:
                    if user_data.cooldown_counter > 0:
                        user_data.cooldown_counter -= 1
                        if user_data.writer is not None:
                            user_data.writer.write(frame_bgr)
                    else:
                        if user_data.writer is not None:
                            user_data.writer.release()
                            user_data.writer = None
                        user_data.recording = False
                        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Target left. Dynamic clip recording stopped.")

        # Overlay text on visual screen
        cv2.putText(
            frame,
            f"Detections: {detection_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        if record_clips and user_data.recording:
            cv2.putText(
                frame,
                "RECORDING",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
        if user_data.use_frame:
            # Convert RGB → BGR for display window
            frame_display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            user_data.set_frame(frame_display)

    if detection_count > 0:
        print(string_to_print)
    return


def main():
    hailo_logger.info("Starting Detection App.")
    user_data = user_app_callback_class()
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()


if __name__ == "__main__":
    main()
