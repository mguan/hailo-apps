# region imports
# Standard library imports
import os

# Third-party imports
import setproctitle

# Local application-specific imports
from hailo_apps.python.core.common.core import get_pipeline_parser
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

MOTION_RECORDER_APP_TITLE = "motion_recorder"


# -----------------------------------------------------------------------------------------------
# User Gstreamer Application
# -----------------------------------------------------------------------------------------------
class GStreamerMotionRecorderApp(GStreamerApp):
    def __init__(self, app_callback, user_data, parser=None):
        if parser is None:
            parser = get_pipeline_parser()
        self._add_arguments(parser)

        hailo_logger.info("Initializing GStreamer Motion Recorder App...")
        super().__init__(parser, user_data)

        if not self.options_menu.gui:
            self.video_sink = "fakesink"

        opts = self.options_menu
        self.user_data.record_clips = opts.record_clips
        self.user_data.motion_min_area = opts.motion_min_area
        self.user_data.motion_threshold = opts.motion_threshold
        self.user_data.output_dir = opts.output_dir
        self.user_data.fps = float(self.frame_rate) if self.frame_rate else 30.0
        
        self.user_data.web_app_host = opts.web_app_host
        self.user_data.web_app_port = opts.web_app_port

        if opts.record_clips:
            os.makedirs(opts.output_dir, exist_ok=True)

        hailo_logger.debug(
            "Parent GStreamerApp initialized | arch=%s | input=%s | fps=%s | sync=%s | show_fps=%s",
            self.arch,
            self.video_source,
            self.frame_rate,
            self.sync,
            self.show_fps,
        )

        self.app_callback = app_callback

        setproctitle.setproctitle(MOTION_RECORDER_APP_TITLE)
        hailo_logger.debug("Process title set to %s", MOTION_RECORDER_APP_TITLE)

        self.create_pipeline()
        hailo_logger.debug("Pipeline created")

    @staticmethod
    def _add_arguments(parser):
        parser.add_argument(
            "--gui",
            action="store_true",
            help="Enable the visual GUI display window (disabled by default)",
        )
        parser.add_argument(
            "--no-record-clips",
            dest="record_clips",
            action="store_false",
            help="Disable recording of motion clips (recording is on by default)",
        )
        parser.add_argument(
            "--motion-min-area",
            type=int,
            default=50,
            help="Minimum contour area to consider as motion (default: 50)",
        )
        parser.add_argument(
            "--motion-threshold",
            type=int,
            default=15,
            help="Threshold value for frame differencing (default: 15)",
        )
        parser.add_argument(
            "--output-dir",
            type=str,
            default="/home/pi/Videos",
            help="Root directory where dynamic video clips are saved (default: /home/pi/Videos)",
        )

        parser.add_argument(
            "--web-app-host",
            type=str,
            default="0.0.0.0",
            help="Host interface address to bind the web dashboard server to (default: 0.0.0.0)",
        )
        parser.add_argument(
            "--web-app-port",
            type=int,
            default=5000,
            help="Network port to run the web dashboard server on (default: 5000)",
        )

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
    hailo_logger.info("Starting Hailo Motion Recorder App...")
    user_data = app_callback_class()
    app = GStreamerMotionRecorderApp(dummy_callback, user_data)
    app.run()


if __name__ == "__main__":
    main()
