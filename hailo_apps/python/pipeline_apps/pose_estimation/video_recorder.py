import os
import cv2
from datetime import datetime

from hailo_apps.python.core.common.hailo_logger import get_logger

hailo_logger = get_logger(__name__)

# Number of frames to keep recording after the last fall detection
TRAILING_FRAMES = 150

class EventVideoRecorder:
    def __init__(self, event_storage_path="/tmp", show_time=True):
        self.event_storage_path = event_storage_path
        self.show_time = show_time
        self.video_writer = None
        self.frames_since_last_fall = 0

    def is_enabled(self):
        return bool(self.event_storage_path)

    def write_frame(self, frame_bgr, width, height, fall_detected) -> bool:
        """Write a frame to the active recording. Returns True when a recording just finished."""
        if not self.is_enabled() or frame_bgr is None:
            return False

        if fall_detected:
            self.frames_since_last_fall = 0
        elif self.video_writer is not None and self.frames_since_last_fall < TRAILING_FRAMES:
            self.frames_since_last_fall += 1
        else:
            if self.video_writer is not None:
                hailo_logger.info("Fall resolved, finishing event video.")
                self.video_writer.release()
                self.video_writer = None
                return True
            return False

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
                f"fall_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
            )
            self.video_writer = cv2.VideoWriter(
                filename, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height)
            )
            hailo_logger.info("Started recording fall event video to: %s", filename)

        self.video_writer.write(frame_to_write)
        return False

    def release(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
