#!/bin/bash

# Navigate to the base directory and set up the environment
cd "$HOME/hailo-apps/"
source ./setup_env.sh

# Navigate to the pose estimation application folder
cd "$HOME/hailo-apps/hailo_apps/python/pipeline_apps/pose_estimation/"

# Execute the application (using exec to replace the shell process for clean exit handling)
exec python pose_estimation.py "$@"