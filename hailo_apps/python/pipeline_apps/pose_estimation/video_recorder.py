import os
import cv2
from datetime import datetime

from hailo_apps.python.core.common.hailo_logger import get_logger

hailo_logger = get_logger(__name__)

# Number of frames to keep recording after the last event detection
TRAILING_FRAMES = 150

class EventVideoRecorder:
    def __init__(self, event_storage_path="/tmp", show_time=True):
        self.event_storage_path = event_storage_path
        self.show_time = show_time
        self.video_writer = None
        self.frames_since_last_event = 0

    def is_enabled(self):
        return bool(self.event_storage_path)

    def write_frame(self, frame_bgr, width, height, event_active):
        if not self.is_enabled() or frame_bgr is None:
            return

        if event_active:
            self.frames_since_last_event = 0
        elif self.video_writer is not None and self.frames_since_last_event < TRAILING_FRAMES:
            self.frames_since_last_event += 1
        else:
            if self.video_writer is not None:
                hailo_logger.info("Event resolved, finishing event video.")
                self.video_writer.release()
                self.video_writer = None
            return

        frame_to_write = frame_bgr.copy()
        if self.show_time:
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

        if self.video_writer is None:
            os.makedirs(self.event_storage_path, exist_ok=True)
            filename = os.path.join(
                self.event_storage_path,
                f"event_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
            )
            self.video_writer = cv2.VideoWriter(
                filename, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height)
            )
            hailo_logger.info("Started recording event video to: %s", filename)

        self.video_writer.write(frame_to_write)

    def release(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
