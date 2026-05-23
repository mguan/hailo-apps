#!/bin/bash

# Navigate to the base directory and set up the environment
cd "$HOME/hailo-apps/"
source ./setup_env.sh

# Navigate to the detection application folder
cd "$HOME/hailo-apps/hailo_apps/python/pipeline_apps/detection/"

# Execute the application (using exec to replace the shell process for clean exit handling)
exec python detection.py --horizontal-mirror --vertical-mirror --input rpi --height=480 --width=640 --record-clips --motion-detect "$@"