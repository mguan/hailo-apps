# region imports
# Standard library imports
import datetime
import threading
import os

os.environ["GST_PLUGIN_FEATURE_RANK"] = "vaapidecodebin:NONE"

# Third-party imports
import gi

gi.require_version("Gst", "1.0")
import cv2

# Local application-specific imports
from hailo_apps.python.pipeline_apps.detection.motion_recorder_pipeline import GStreamerMotionRecorderApp
from hailo_apps.python.pipeline_apps.detection.motion_detector import MotionDetector
from hailo_apps.python.pipeline_apps.detection.clip_recorder import ClipRecorder
from hailo_apps.python.core.common.buffer_utils import (
    get_caps_from_pad,
    get_numpy_from_buffer,
)
from hailo_apps.python.core.common.hailo_logger import get_logger
from hailo_apps.python.core.gstreamer.gstreamer_app import app_callback_class
import hailo_apps.python.pipeline_apps.detection.web_server as web_server

hailo_logger = get_logger(__name__)
# endregion imports


# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
class user_app_callback_class(app_callback_class):
    motion_detector: "MotionDetector"
    recorder: "ClipRecorder"
    web_app_port: int


# -----------------------------------------------------------------------------------------------
# Drawing helpers
# -----------------------------------------------------------------------------------------------
def _draw_motion_overlay(frame_bgr, boxes):
    for (xmin, ymin, xmax, ymax) in boxes:
        cv2.rectangle(frame_bgr, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
        cv2.putText(
            frame_bgr, "Motion", (xmin, ymin - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2,
        )


def _draw_hud(frame_bgr, num_zones: int, recording: bool):
    cv2.putText(
        frame_bgr, f"Motion Zones: {num_zones}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
    )
    if recording:
        cv2.putText(
            frame_bgr, "RECORDING", (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
        )


def _draw_timestamp(frame_bgr):
    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    color = (255, 255, 255)
    
    text_size, _ = cv2.getTextSize(timestamp_str, font, scale, thickness)
    text_w, text_h = text_size
    frame_h, frame_w = frame_bgr.shape[:2]
    
    x = frame_w - text_w - 20
    y = text_h + 20
    
    # Draw a 50% transparent background rectangle behind the text
    x1, y1 = max(0, x - 5), max(0, y - text_h - 5)
    x2, y2 = min(frame_w, x + text_w + 5), min(frame_h, y + 5)
    
    overlay = frame_bgr[y1:y2, x1:x2].copy()
    cv2.rectangle(overlay, (0, 0), (x2 - x1, y2 - y1), (0, 0, 0), -1)
    frame_bgr[y1:y2, x1:x2] = cv2.addWeighted(overlay, 0.5, frame_bgr[y1:y2, x1:x2], 0.5, 0)
    
    cv2.putText(frame_bgr, timestamp_str, (x, y), font, scale, color, thickness, cv2.LINE_AA)


# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------
def app_callback(element, buffer, user_data):
    if buffer is None:
        hailo_logger.warning("Received None buffer.")
        return

    pad = element.get_static_pad("src")
    format_cap, width, height = get_caps_from_pad(pad)
    if format_cap is None or width is None or height is None:
        return

    frame = get_numpy_from_buffer(buffer, format_cap, width, height)
    if frame is None:
        return

    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    motion_boxes = user_data.motion_detector.detect(frame, width, height)
    motion_detected = bool(motion_boxes)

    # Burned-in overlays — drawn before feed() so they are saved into the clip.
    _draw_motion_overlay(frame_bgr, motion_boxes)
    _draw_timestamp(frame_bgr)

    user_data.recorder.feed(frame_bgr, motion_detected, width, height)

    # Display-only overlay — drawn after feed() so it never lands in the clip.
    _draw_hud(frame_bgr, len(motion_boxes), user_data.recorder.recording)

    if user_data.use_frame:
        user_data.set_frame(frame_bgr)

    web_server.set_shared_frame(frame_bgr)


def main():

    hailo_logger.info("Starting Motion Recorder App.")
    user_data = user_app_callback_class()
    app = GStreamerMotionRecorderApp(app_callback, user_data)

    flask_thread = threading.Thread(
        target=web_server.start_server,
        kwargs={
            'host': '0.0.0.0',
            'port': user_data.web_app_port,
            'clips_dir': user_data.output_dir,
        },
        daemon=True,
    )
    flask_thread.start()
    hailo_logger.info(
        f"Web dashboard started on http://0.0.0.0:{user_data.web_app_port}"
    )

    try:
        app.run()
    finally:
        user_data.recorder.close()


if __name__ == "__main__":
    main()
