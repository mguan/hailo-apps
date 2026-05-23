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
        self.writer = None
        self.recording = False
        self.cooldown_counter = 0
        self.cooldown_limit = 150  # 150 frames (about 5 seconds at 30fps)
        self.record_clips = False
        self.motion_detect = True
        self.motion_min_area = 50
        self.motion_threshold = 25
        self.avg_frame = None
        self.output_dir = "/home/pi/Videos"

# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------


def app_callback(element, buffer, user_data):
    if buffer is None:
        hailo_logger.warning("Received None buffer.")
        return

    # Note: Frame counting is handled automatically by the framework wrapper
    frame_idx = user_data.get_count()

    pad = element.get_static_pad("src")
    format_cap, width, height = get_caps_from_pad(pad)

    # Check if recording or display is requested
    record_clips = getattr(user_data, "record_clips", False)

    frame = None
    if (record_clips or user_data.use_frame) and format_cap is not None and width is not None and height is not None:
        frame = get_numpy_from_buffer(buffer, format_cap, width, height)

    motion_detected = False
    motion_boxes = []

    # Process frame with OpenCV if available
    if frame is not None:
        # Convert RGB → BGR for OpenCV processing and saving
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Determine analysis resolution (fixed to 640px width for smooth 30 FPS processing)
        analysis_width = 640
        analysis_height = int(analysis_width * height / width)

        # Downscale for motion detection analysis
        small_frame = cv2.resize(frame, (analysis_width, analysis_height))
        gray = cv2.cvtColor(small_frame, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (11, 11), 0)

        # Scale the min area threshold based on the resolution reduction
        scale_factor = (analysis_width * analysis_height) / (width * height)
        scaled_min_area = user_data.motion_min_area * scale_factor

        if user_data.avg_frame is None:
            user_data.avg_frame = gray.copy().astype("float")
        else:
            # Accumulate background average slowly (0.02 weight) so slow-moving objects are detected
            cv2.accumulateWeighted(gray, user_data.avg_frame, 0.02)
            # Compute absolute difference
            frame_delta = cv2.absdiff(gray, cv2.convertScaleAbs(user_data.avg_frame))
            # Threshold the difference image
            thresh = cv2.threshold(frame_delta, user_data.motion_threshold, 255, cv2.THRESH_BINARY)[1]
            # Dilate
            thresh = cv2.dilate(thresh, None, iterations=2)
            # Find contours
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if cv2.contourArea(contour) >= scaled_min_area:
                    motion_detected = True
                    (x, y, w, h) = cv2.boundingRect(contour)
                    
                    # Map coordinates back to the original full resolution
                    orig_xmin = int(x * width / analysis_width)
                    orig_ymin = int(y * height / analysis_height)
                    orig_xmax = int((x + w) * width / analysis_width)
                    orig_ymax = int((y + h) * height / analysis_height)
                    motion_boxes.append((orig_xmin, orig_ymin, orig_xmax, orig_ymax))

        # Draw motion bounding boxes on BGR frame
        for (xmin, ymin, xmax, ymax) in motion_boxes:
            cv2.rectangle(frame_bgr, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            cv2.putText(frame_bgr, "Motion", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if record_clips:
            if motion_detected:
                user_data.cooldown_counter = user_data.cooldown_limit
                
                if not user_data.recording:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_dir = getattr(user_data, "output_dir", "/home/pi/Videos")
                    filename = os.path.join(output_dir, f"motion_{timestamp}.mp4")
                    # Ensure directory exists before recording
                    os.makedirs(os.path.dirname(filename), exist_ok=True)
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    fps = 30.0
                    user_data.writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
                    user_data.recording = True
                    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Motion entered. Started clip recording: {filename}")
                    
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
                        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Motion left. Dynamic clip recording stopped.")

        if user_data.use_frame:
            # Draw HUD overlays on display frame
            cv2.putText(
                frame_bgr,
                f"Motion Zones: {len(motion_boxes)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            if user_data.recording:
                cv2.putText(
                    frame_bgr,
                    "RECORDING",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
            user_data.set_frame(frame_bgr)

    if motion_detected:
        print(f"Frame count: {frame_idx}\nMotion detected: {len(motion_boxes)} zones\n")
    return


def main():
    hailo_logger.info("Starting Detection App.")
    user_data = user_app_callback_class()
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()


if __name__ == "__main__":
    main()
