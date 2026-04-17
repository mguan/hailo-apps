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
from hailo_apps.python.pipeline_apps.pose_estimation.video_recorder import EventVideoRecorder
from hailo_apps.python.pipeline_apps.pose_estimation.presence_detector import PresenceDetector

hailo_logger = get_logger(__name__)
# endregion imports


# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.fall_detector = FallDetector(safe_zones=[(0.5, 0.5, 1.0, 1.0)])
        self.presence_detector = PresenceDetector(door_zones=[(0.25, 0.0, 0.75, 0.5)])
        self.alert_manager = AlertManager()
        self.video_recorder = EventVideoRecorder()


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
    if (user_data.use_frame or user_data.video_recorder.is_enabled()) and format and width and height:
        frame = get_numpy_from_buffer(buffer, format, width, height)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    fall_detected = False
    current_tracks = {}

    for detection in detections:
        if detection.get_label() != "person":
            continue

        bbox = detection.get_bbox()
        track_id = 0
        track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        if len(track) == 1:
            track_id = track[0].get_id()

        if track_id > 0:
            current_tracks[track_id] = bbox

        hailo_logger.debug("Detection: ID=%d Confidence=%.2f", track_id, detection.get_confidence())

        landmarks = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
        if landmarks:
            points = landmarks[0].get_points()
            if user_data.fall_detector.is_fall_detected(track_id, bbox, points):
                fall_detected = True
                if user_data.fall_detector.check_alert_throttle(track_id):
                    alert_msg = f"⚠️ Fall detected!\nPerson ID: {track_id}\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    print(alert_msg)
                    user_data.alert_manager.send_alert(alert_msg, image=frame_bgr)
            elif user_data.fall_detector.is_fall_resolved(track_id):
                resolve_msg = f"✅ Fall event resolved\nPerson ID: {track_id}\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                user_data.alert_manager.send_alert(resolve_msg, image=frame_bgr)

            if user_data.use_frame and frame_bgr is not None:
                for eye in ["left_eye", "right_eye"]:
                    point = points[KEYPOINTS[eye]]
                    x = int((point.x() * bbox.width() + bbox.xmin()) * width)
                    y = int((point.y() * bbox.height() + bbox.ymin()) * height)
                    cv2.circle(frame_bgr, (x, y), 5, (0, 255, 0), -1)

    if user_data.use_frame and frame_bgr is not None:
        # Draw safe zones (Green)
        for z_xmin, z_ymin, z_xmax, z_ymax in user_data.fall_detector.safe_zones:
            cv2.rectangle(frame_bgr, 
                          (int(z_xmin * width), int(z_ymin * height)),
                          (int(z_xmax * width), int(z_ymax * height)),
                          (0, 255, 0), 2)
                          
        # Draw door zones (Blue)
        for z_xmin, z_ymin, z_xmax, z_ymax in user_data.presence_detector.door_zones:
            cv2.rectangle(frame_bgr, 
                          (int(z_xmin * width), int(z_ymin * height)),
                          (int(z_xmax * width), int(z_ymax * height)),
                          (255, 0, 0), 2)

        user_data.set_frame(frame_bgr)

    presence_events = user_data.presence_detector.update(current_tracks)
    for event in presence_events:
        event_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if event["type"] == "entry":
            msg = f"🚪 Person Entered!\nPerson ID: {event['track_id']}\nTime: {event_time}"
            user_data.alert_manager.send_alert(msg, image=frame_bgr)
            print(msg)
        elif event["type"] == "exit":
            msg = f"🚪 Person Exited.\nPerson ID: {event['track_id']}\nTime: {event_time}"
            user_data.alert_manager.send_alert(msg, image=frame_bgr)
            print(msg)

    presence_active = len(user_data.presence_detector.active_presences) > 0
    event_active = fall_detected or presence_active
    user_data.video_recorder.write_frame(frame_bgr, width, height, event_active)



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
    user_data.video_recorder.event_storage_path = getattr(app.options_menu, "event_storage_path", "/tmp")
    user_data.video_recorder.show_time = getattr(app.options_menu, "show_time", True)
    telegram_token = getattr(app.options_menu, "telegram_token", None)
    telegram_chat_id = getattr(app.options_menu, "telegram_chat_id", None)

    if telegram_token and telegram_chat_id:
        user_data.alert_manager.add_provider(
            TelegramAlertProvider(telegram_token, telegram_chat_id)
        )

    app.run()

    user_data.video_recorder.release()


if __name__ == "__main__":
    main()
