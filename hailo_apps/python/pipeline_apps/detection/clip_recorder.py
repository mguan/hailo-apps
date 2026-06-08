# region imports
# Standard library imports
import datetime
import os
import subprocess

# Local application-specific imports
from hailo_apps.python.core.common.hailo_logger import get_logger

hailo_logger = get_logger(__name__)
# endregion imports


# -----------------------------------------------------------------------------------------------
# Fallback VideoWriter using FFmpeg subprocess for efficient browser-compatible MP4 encoding
# -----------------------------------------------------------------------------------------------
class FFmpegVideoWriter:
    def __init__(self, filename, fps, frame_size):
        self.filename = filename
        self.fps = fps
        self.width, self.height = frame_size
        self._process = None
        self._opened = False

        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{self.width}x{self.height}',
            '-r', str(self.fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-b:v', '6000k',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            self.filename
        ]
        try:
            self._process = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            self._opened = True
        except Exception as e:
            hailo_logger.error("Failed to start FFmpeg subprocess: %s", e)
            self._opened = False

    def isOpened(self) -> bool:
        return self._opened and self._process is not None and self._process.poll() is None

    def write(self, frame):
        if self.isOpened():
            try:
                self._process.stdin.write(frame.tobytes())
            except Exception as e:
                hailo_logger.error("Failed to write frame to FFmpeg: %s", e)

    def release(self):
        if self._process:
            try:
                if self._process.stdin:
                    self._process.stdin.close()
                self._process.wait()
            except Exception as e:
                hailo_logger.error("Error releasing FFmpeg subprocess for %s: %s", self.filename, e)
            self._process = None
        self._opened = False


# -----------------------------------------------------------------------------------------------
# Clip recording
# -----------------------------------------------------------------------------------------------
class ClipRecorder:
    COOLDOWN_SECONDS = 3.0

    def __init__(self, output_dir: str, fps: float, debounce_seconds: float = 4.0):
        self.output_dir = output_dir
        self.fps = fps
        self.debounce_seconds = debounce_seconds
        self._cooldown_limit = max(1, int(round(self.COOLDOWN_SECONDS * fps)))
        self._writer = None
        self._cooldown = 0
        self._current_path = None
        self._frames_written = 0
        self._motion_start_frame = 0
        self._motion_last_frame = 0

    @property
    def recording(self) -> bool:
        return self._writer is not None

    def feed(self, frame_bgr, motion: bool, width: int, height: int):
        if motion:
            self._cooldown = self._cooldown_limit
            if not self.recording:
                self._start(width, height)
                self._motion_start_frame = self._frames_written
            self._motion_last_frame = self._frames_written
        elif self.recording:
            if self._cooldown > 0:
                self._cooldown -= 1
            else:
                self._stop()
                return
        if self.recording:
            self._writer.write(frame_bgr)
            self._frames_written += 1

    def close(self):
        """Flush any in-progress recording. Safe to call multiple times."""
        if self.recording:
            self._stop()

    def _start(self, width: int, height: int):
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        target_dir = os.path.join(
            self.output_dir,
            now.strftime("%Y"),
            now.strftime("%m"),
            now.strftime("%d")
        )
        os.makedirs(target_dir, exist_ok=True)
        filename = os.path.join(target_dir, f"motion_{timestamp}.mp4")

        writer = FFmpegVideoWriter(filename, self.fps, (width, height))

        if not writer.isOpened():
            writer.release()
            hailo_logger.error(
                "Failed to open VideoWriter for %s (fps=%s, size=%dx%d). "
                "Skipping this motion event.",
                filename, self.fps, width, height,
            )
            return
        self._writer = writer
        self._current_path = filename
        self._frames_written = 0
        self._motion_start_frame = 0
        self._motion_last_frame = 0
        hailo_logger.info("Motion entered. Started clip recording: %s", filename)

    def _stop(self):
        self._writer.release()
        duration = self._frames_written / self.fps if self.fps else 0.0
        motion_duration = (self._motion_last_frame - self._motion_start_frame + 1) / self.fps if self.fps else 0.0

        if motion_duration <= self.debounce_seconds:
            hailo_logger.info(
                "Motion lasted %.2fs (<= %.2fs debounce). Discarding clip: %s",
                motion_duration, self.debounce_seconds, self._current_path
            )
            if self._current_path and os.path.exists(self._current_path):
                try:
                    os.remove(self._current_path)
                except Exception as e:
                    hailo_logger.error("Failed to delete debounced clip %s: %s", self._current_path, e)
        else:
            hailo_logger.info(
                "Motion left. Stopped clip recording: %s (%d frames, %.1fs, motion duration: %.1fs)",
                self._current_path, self._frames_written, duration, motion_duration
            )

        self._writer = None
        self._current_path = None
        self._frames_written = 0
        self._motion_start_frame = 0
        self._motion_last_frame = 0
