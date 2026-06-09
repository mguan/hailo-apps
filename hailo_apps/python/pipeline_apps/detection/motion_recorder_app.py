# region imports
# Standard library imports
import datetime
import threading
import os

os.environ["GST_PLUGIN_FEATURE_RANK"] = "vaapidecodebin:NONE"

# Third-party imports
import gi
import numpy as np

gi.require_version("Gst", "1.0")
import cv2

# Local application-specific imports
from hailo_apps.python.pipeline_apps.detection.motion_recorder_pipeline import GStreamerMotionRecorderApp
from hailo_apps.python.pipeline_apps.detection.motion_detector import MotionDetector
from hailo_apps.python.pipeline_apps.detection.clip_recorder import ClipRecorder
from hailo_apps.python.core.common.buffer_utils import (
    get_caps_from_pad,
    get_numpy_from_buffer_efficient,
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
    # Cached (format, width, height) from pad caps — populated on first frame.
    # Caps are constant for the lifetime of a running pipeline.
    cached_caps: tuple
    cached_caps = None


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


# -----------------------------------------------------------------------------------------------
# Timestamp overlay state — cached across frames to avoid per-frame allocations.
# -----------------------------------------------------------------------------------------------
class _TsState:
    """Holds the cached geometry and background buffer used by _draw_timestamp."""
    __slots__ = ('last_int_sec', 'ts_str', 'geom', 'black_bg')
    def __init__(self):
        self.last_int_sec = -1   # epoch-second of the last rendered timestamp
        self.ts_str = ""
        self.geom = None         # (x, y, x1, y1, x2, y2) — computed once per resolution
        self.black_bg = None     # pre-allocated zero array for the semi-transparent bg

_ts_state = _TsState()
_TS_FONT = cv2.FONT_HERSHEY_SIMPLEX
_TS_SCALE = 0.8
_TS_THICKNESS = 2
_TS_COLOR = (255, 255, 255)


def _draw_timestamp(frame_bgr):
    now = datetime.datetime.now()
    int_sec = int(now.timestamp())  # changes once per second
    if int_sec != _ts_state.last_int_sec:
        _ts_state.ts_str = now.strftime("%Y-%m-%d %H:%M:%S")
        _ts_state.last_int_sec = int_sec
        _ts_state.geom = None  # re-measure text if needed (safe guard)

    if _ts_state.geom is None:
        (tw, th), _ = cv2.getTextSize(_ts_state.ts_str, _TS_FONT, _TS_SCALE, _TS_THICKNESS)
        fh, fw = frame_bgr.shape[:2]
        x = fw - tw - 20
        y = th + 20
        x1, y1 = max(0, x - 5), max(0, y - th - 5)
        x2, y2 = min(fw, x + tw + 5), min(fh, y + 5)
        _ts_state.geom = (x, y, x1, y1, x2, y2)
        # Pre-allocate zeros once — reused every frame as the "black" blend source.
        _ts_state.black_bg = np.zeros((y2 - y1, x2 - x1, 3), dtype=np.uint8)

    x, y, x1, y1, x2, y2 = _ts_state.geom
    # Blend 50% black with 50% of the live ROI in-place (no extra .copy() allocation).
    roi = frame_bgr[y1:y2, x1:x2]
    cv2.addWeighted(_ts_state.black_bg, 0.5, roi, 0.5, 0, dst=roi)
    cv2.putText(frame_bgr, _ts_state.ts_str, (x, y), _TS_FONT, _TS_SCALE, _TS_COLOR, _TS_THICKNESS, cv2.LINE_AA)


# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------
def app_callback(element, buffer, user_data):
    if buffer is None:
        hailo_logger.warning("Received None buffer.")
        return

    # Cache caps on first frame — they are constant for a running pipeline.
    if user_data.cached_caps is None:
        pad = element.get_static_pad("src")
        user_data.cached_caps = get_caps_from_pad(pad)
    format_cap, width, height = user_data.cached_caps
    if format_cap is None or width is None or height is None:
        return

    frame = get_numpy_from_buffer_efficient(buffer, format_cap, width, height)
    if frame is None:
        return

    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # Pass BGR frame — MotionDetector now uses COLOR_BGR2GRAY, saving one conversion.
    motion_boxes = user_data.motion_detector.detect(frame_bgr, width, height)
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
