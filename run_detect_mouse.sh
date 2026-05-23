#!/bin/bash

# Navigate to the base directory and set up the environment
cd "$HOME/hailo-apps/"
source ./setup_env.sh

# Navigate to the detection_simple application folder
cd "$HOME/hailo-apps/hailo_apps/python/pipeline_apps/detection_simple/"

# Execute the application (using exec to replace the shell process for clean exit handling)
exec python detection_simple.py --horizontal-mirror --vertical-mirror --input rpi --height=864 --width=1536 --output-file=record.mkv