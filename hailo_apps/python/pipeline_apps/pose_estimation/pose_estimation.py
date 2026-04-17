# region imports
# Standard library imports
import os
import time
import argparse
import subprocess
import threading
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
from hailo_apps.python.pipeline_apps.pose_estimation.fall_detector import FallDetector, KEYPOINTS
from hailo_apps.python.pipeline_apps.pose_estimation.alert_manager import AlertManager, TelegramAlertProvider

hailo_logger = get_logger(__name__)
# endregion imports


# Number of frames to keep recording after the last fall detection
TRAILING_FRAMES = 150


# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.fall_detector = FallDetector()
        self.alert_manager = AlertManager()
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
    active_track_id = 0

    for detection in detections:
        if detection.get_label() != "person":
            continue

        bbox = detection.get_bbox()
        track_id = 0
        track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        if len(track) == 1:
            track_id = track[0].get_id()
            
        active_track_id = track_id

        hailo_logger.debug("Detection: ID=%d Confidence=%.2f", track_id, detection.get_confidence())

        landmarks = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
        if landmarks:
            points = landmarks[0].get_points()
            if user_data.fall_detector.is_fall_detected(track_id, bbox, points):
                fall_detected = True
                should_alert, next_backoff = user_data.fall_detector.should_trigger_alert(track_id)
                if should_alert:
                    print(f"Fall detected: Person ID {track_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (Next alert in {next_backoff}s)")
                    alert_msg = f"⚠️ Fall detected!\nPerson ID: {track_id}\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    user_data.alert_manager.send_alert(alert_msg, image=frame_bgr)

            if user_data.use_frame and frame_bgr is not None:
                for eye in ["left_eye", "right_eye"]:
                    point = points[KEYPOINTS[eye]]
                    x = int((point.x() * bbox.width() + bbox.xmin()) * width)
                    y = int((point.y() * bbox.height() + bbox.ymin()) * height)
                    cv2.circle(frame_bgr, (x, y), 5, (0, 255, 0), -1)

    if user_data.use_frame and frame_bgr is not None:
        user_data.set_frame(frame_bgr)

    if user_data.event_storage_path and frame_bgr is not None:
        write_video_frame(user_data, frame_bgr, width, height, fall_detected, active_track_id)


def write_video_frame(user_data, frame_bgr, width, height, fall_detected, track_id):
    if fall_detected:
        user_data.frames_since_last_fall = 0
    elif user_data.video_writer is not None and user_data.frames_since_last_fall < TRAILING_FRAMES:
        user_data.frames_since_last_fall += 1
    else:
        if user_data.video_writer is not None:
            hailo_logger.info("Fall resolved, finishing event video.")
            user_data.video_writer.release()
            user_data.video_writer = None
            
            if user_data.alert_manager.providers:
                resolve_msg = f"✅ Fall event resolved\nPerson ID: {track_id}\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                user_data.alert_manager.send_alert(resolve_msg, image=frame_bgr)
        return

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

    if user_data.telegram_token and user_data.telegram_chat_id:
        user_data.alert_manager.add_provider(
            TelegramAlertProvider(user_data.telegram_token, user_data.telegram_chat_id)
        )

    app.run()

    if user_data.video_writer is not None:
        user_data.video_writer.release()
        user_data.video_writer = None


if __name__ == "__main__":
    main()
