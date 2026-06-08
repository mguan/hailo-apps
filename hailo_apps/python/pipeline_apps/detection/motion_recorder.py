# region imports
# Standard library imports
import datetime
import os
import subprocess

os.environ["GST_PLUGIN_FEATURE_RANK"] = "vaapidecodebin:NONE"

# Third-party imports
import gi

gi.require_version("Gst", "1.0")
import cv2
import numpy as np

# Local application-specific imports
from hailo_apps.python.pipeline_apps.detection.motion_recorder_pipeline import GStreamerMotionRecorderApp
from hailo_apps.python.core.common.buffer_utils import (
    get_caps_from_pad,
    get_numpy_from_buffer,
)
from hailo_apps.python.core.common.hailo_logger import get_logger
from hailo_apps.python.core.gstreamer.gstreamer_app import app_callback_class
import hailo_apps.python.pipeline_apps.detection.web_app as web_app

hailo_logger = get_logger(__name__)
# endregion imports


# -----------------------------------------------------------------------------------------------
# Motion detection
# -----------------------------------------------------------------------------------------------
class MotionDetector:
    ANALYSIS_WIDTH = 640
    # higher = background adapts faster (less sensitive to slow motion)
    BG_LEARNING_RATE = 0.02
    BLUR_KERNEL = (5, 5)

    def __init__(self, min_area: int, threshold: int):
        self.min_area = min_area
        self.threshold = threshold
        self._avg_frame = None

    def detect(self, rgb_frame, orig_w: int, orig_h: int):
        analysis_h = int(self.ANALYSIS_WIDTH * orig_h / orig_w)
        small = cv2.resize(rgb_frame, (self.ANALYSIS_WIDTH, analysis_h))
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, self.BLUR_KERNEL, 0)

        if self._avg_frame is None:
            self._avg_frame = gray.astype(np.float32)
            return []

        cv2.accumulateWeighted(gray, self._avg_frame, self.BG_LEARNING_RATE)
        delta = cv2.absdiff(gray, cv2.convertScaleAbs(self._avg_frame))
        thresh = cv2.threshold(delta, self.threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        scale = (self.ANALYSIS_WIDTH * analysis_h) / (orig_w * orig_h)
        scaled_min_area = self.min_area * scale
        sx = orig_w / self.ANALYSIS_WIDTH
        sy = orig_h / analysis_h

        boxes = []
        for c in contours:
            if cv2.contourArea(c) < scaled_min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            boxes.append((int(x * sx), int(y * sy), int((x + w) * sx), int((y + h) * sy)))
        return boxes


# -----------------------------------------------------------------------------------------------
# Fallback VideoWriter using FFmpeg subprocess for efficient browser-compatible MP4 encoding
# -----------------------------------------------------------------------------------------------
class FFmpegVideoWriter:
    def __init__(self, filename, fourcc, fps, frame_size):
        self.filename = filename
        self.fps = fps
        self.width, self.height = frame_size
        self._process = None
        self._opened = False
        
        # Start ffmpeg subprocess
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
            except Exception:
                pass
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
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        
        writer = FFmpegVideoWriter(filename, fourcc, self.fps, (width, height))

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
        
        is_debounced = motion_duration <= self.debounce_seconds
        
        if is_debounced:
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


# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.motion_min_area = 50
        self.motion_threshold = 15
        self.output_dir = "/home/pi/Videos"
        self.fps = 30.0
        self.web_app_port = 5000
        self.debounce_seconds = 4.0
        # Helper objects — built lazily on the first frame
        self.motion_detector = None
        self.recorder = None


# -----------------------------------------------------------------------------------------------
# Drawing helpers
# -----------------------------------------------------------------------------------------------
def _ensure_helpers(user_data):
    if user_data.motion_detector is None:
        user_data.motion_detector = MotionDetector(
            user_data.motion_min_area, user_data.motion_threshold
        )
    if user_data.recorder is None:
        user_data.recorder = ClipRecorder(
            user_data.output_dir,
            user_data.fps,
            getattr(user_data, "debounce_seconds", 4.0)
        )


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
    _, frame_w = frame_bgr.shape[:2]
    
    x = frame_w - text_w - 20
    y = text_h + 20
    
    cv2.rectangle(
        frame_bgr,
        (x - 5, y - text_h - 5),
        (x + text_w + 5, y + 5),
        (0, 0, 0),
        -1
    )
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

    _ensure_helpers(user_data)

    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    motion_boxes = user_data.motion_detector.detect(frame, width, height)
    motion_detected = bool(motion_boxes)

    # Motion boxes are drawn before recording so they end up burned into the
    # saved clip. HUD (zone count + RECORDING indicator) is display-only and
    # is drawn after recording.
    _draw_motion_overlay(frame_bgr, motion_boxes)

    _draw_timestamp(frame_bgr)

    user_data.recorder.feed(frame_bgr, motion_detected, width, height)

    # Always draw the HUD for the web dashboard stream
    recording = user_data.recorder.recording if user_data.recorder else False
    _draw_hud(frame_bgr, len(motion_boxes), recording)

    if user_data.use_frame:
        user_data.set_frame(frame_bgr)

    web_app.set_shared_frame(frame_bgr)




def main():
    hailo_logger.info("Starting Motion Recorder App.")
    user_data = user_app_callback_class()
    app = GStreamerMotionRecorderApp(app_callback, user_data)

    import threading
    # Point Flask's CLIPS_DIR to the configured motion recorder output directory
    web_app.CLIPS_DIR = user_data.output_dir
    os.makedirs(web_app.CLIPS_DIR, exist_ok=True)

    # Start the Flask web server in a background daemon thread
    flask_thread = threading.Thread(
        target=web_app.start_server,
        kwargs={'host': '0.0.0.0', 'port': user_data.web_app_port},
        daemon=True
    )
    flask_thread.start()
    hailo_logger.info(
        f"Web dashboard started on http://0.0.0.0:{user_data.web_app_port}"
    )

    try:
        app.run()
    finally:
        if user_data.recorder is not None:
            user_data.recorder.close()


if __name__ == "__main__":
    main()
