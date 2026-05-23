# region imports
# Standard library imports
from pathlib import Path

import setproctitle

from hailo_apps.python.core.common.core import (
    get_pipeline_parser,
)
from hailo_apps.python.core.common.defines import (
    DETECTION_APP_TITLE,
)

from hailo_apps.python.core.common.hailo_logger import get_logger
from hailo_apps.python.core.gstreamer.gstreamer_app import (
    GStreamerApp,
    app_callback_class,
    dummy_callback,
)
from hailo_apps.python.core.gstreamer.gstreamer_helper_pipelines import (
    DISPLAY_PIPELINE,
    USER_CALLBACK_PIPELINE,
)

hailo_logger = get_logger(__name__)

# endregion imports

# -----------------------------------------------------------------------------------------------
# User Gstreamer Application
# -----------------------------------------------------------------------------------------------


class GStreamerDetectionApp(GStreamerApp):
    def __init__(self, app_callback, user_data, parser=None):
        if parser is None:
            parser = get_pipeline_parser()
        parser.add_argument(
            "--no-display",
            action="store_true",
            help="Disable the visual display (run headless, e.g. for SSH or remote recording)",
        )
        parser.add_argument(
            "--record-clips",
            action="store_true",
            help="Dynamically record video clips with timestamps when objects enter/leave the frame",
        )
        parser.add_argument(
            "--motion-detect",
            action="store_true",
            default=True,
            help="Enable OpenCV motion-based recording (always enabled for motion-only mode)",
        )
        parser.add_argument(
            "--motion-min-area",
            type=int,
            default=500,
            help="Minimum contour area to consider as motion (default: 500)",
        )
        parser.add_argument(
            "--motion-threshold",
            type=int,
            default=25,
            help="Threshold value for frame differencing (default: 25)",
        )
        
        hailo_logger.info("Initializing GStreamer Detection App...")

        super().__init__(parser, user_data)

        # Override video sink if headless
        if self.options_menu.no_display:
            self.video_sink = "fakesink"

        # Pass options to user_data
        self.user_data.record_clips = getattr(self.options_menu, "record_clips", False)
        self.user_data.motion_detect = getattr(self.options_menu, "motion_detect", True)
        self.user_data.motion_min_area = getattr(self.options_menu, "motion_min_area", 500)
        self.user_data.motion_threshold = getattr(self.options_menu, "motion_threshold", 25)

        hailo_logger.debug(
            "Parent GStreamerApp initialized | arch=%s | input=%s | fps=%s | sync=%s | show_fps=%s",
            self.arch,
            self.video_source,
            self.frame_rate,
            self.sync,
            self.show_fps,
        )

        self.app_callback = app_callback

        # Set the process title
        setproctitle.setproctitle(DETECTION_APP_TITLE)
        hailo_logger.debug("Process title set to %s", DETECTION_APP_TITLE)

        self.create_pipeline()
        hailo_logger.debug("Pipeline created")

    def get_pipeline_string(self):
        source_pipeline = self.get_source_pipeline()
        user_callback_pipeline = USER_CALLBACK_PIPELINE()
        display_pipeline = DISPLAY_PIPELINE(
            video_sink=self.video_sink, sync=self.sync, show_fps=self.show_fps
        )

        pipeline_string = (
            f"{source_pipeline} ! "
            f"{user_callback_pipeline} ! "
            f"{display_pipeline}"
        )
        hailo_logger.debug("Pipeline string: %s", pipeline_string)
        return pipeline_string


def main():
    # Create an instance of the user app callback class
    hailo_logger.info("Starting Hailo Detection App...")
    user_data = app_callback_class()
    app_callback = dummy_callback
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()


if __name__ == "__main__":
    main()
