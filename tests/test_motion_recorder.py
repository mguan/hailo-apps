import os
import shutil
import tempfile
import numpy as np
import pytest
import datetime
from hailo_apps.python.pipeline_apps.detection.motion_recorder import ClipRecorder

def test_clip_recorder_directory_structure():
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    try:
        recorder = ClipRecorder(output_dir=temp_dir, fps=30.0)
        
        # Create a mock frame
        width, height = 640, 480
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Feed frame with motion=True to start recording
        recorder.feed(frame, motion=True, width=width, height=height)
        
        assert recorder.recording
        
        # Stop recording by feeding motion=False for cooldown period
        for _ in range(recorder._cooldown_limit + 1):
            recorder.feed(frame, motion=False, width=width, height=height)
            
        assert not recorder.recording
        
        # Verify the file was created under the YYYY/MM/DD directory structure
        now = datetime.datetime.now()
        year = now.strftime("%Y")
        month = now.strftime("%m")
        day = now.strftime("%d")
        
        target_dir = os.path.join(temp_dir, year, month, day)
        assert os.path.isdir(target_dir), f"Directory {target_dir} was not created"
        
        # List files in the target directory
        files = os.listdir(target_dir)
        assert len(files) == 1, f"Expected 1 recording, found: {files}"
        assert files[0].startswith("motion_")
        assert files[0].endswith(".mp4")
        
    finally:
        shutil.rmtree(temp_dir)
