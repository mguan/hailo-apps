"""Tests for hailo_apps.python.pipeline_apps.detection.motion_recorder."""
import datetime
import os
import shutil
import tempfile
import time

import numpy as np
import pytest

from hailo_apps.python.pipeline_apps.detection.motion_recorder import (
    ClipRecorder,
    MotionDetector,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

WIDTH, HEIGHT = 640, 480
FPS = 30.0


def _gray_frame(value: int = 80):
    """Solid-gray RGB frame at the canonical test resolution."""
    return np.full((HEIGHT, WIDTH, 3), value, dtype=np.uint8)


def _frame_with_square(value: int, x: int, y: int, size: int, base: int = 80):
    """Solid-gray frame with a brighter square painted into it."""
    frame = _gray_frame(base)
    frame[y:y + size, x:x + size, :] = value
    return frame


@pytest.fixture
def tmp_output_dir():
    path = tempfile.mkdtemp(prefix="motion_recorder_test_")
    yield path
    shutil.rmtree(path, ignore_errors=True)


# -----------------------------------------------------------------------------
# ClipRecorder: directory + filename layout
# -----------------------------------------------------------------------------

def test_clip_recorder_directory_structure(tmp_output_dir):
    recorder = ClipRecorder(output_dir=tmp_output_dir, fps=FPS)
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    recorder.feed(frame, motion=True, width=WIDTH, height=HEIGHT)
    assert recorder.recording

    for _ in range(recorder._cooldown_limit + 1):
        recorder.feed(frame, motion=False, width=WIDTH, height=HEIGHT)
    assert not recorder.recording

    now = datetime.datetime.now()
    target_dir = os.path.join(
        tmp_output_dir, now.strftime("%Y"), now.strftime("%m"), now.strftime("%d"),
    )
    assert os.path.isdir(target_dir), f"Directory {target_dir} was not created"

    files = os.listdir(target_dir)
    assert len(files) == 1, f"Expected 1 recording, found: {files}"
    assert files[0].startswith("motion_")
    assert files[0].endswith(".mp4")


# -----------------------------------------------------------------------------
# ClipRecorder: cooldown behaviour
# -----------------------------------------------------------------------------

def test_cooldown_keeps_recording_until_limit(tmp_output_dir):
    """During cooldown, the recorder must continue writing frames."""
    recorder = ClipRecorder(output_dir=tmp_output_dir, fps=FPS)
    frame = _gray_frame()

    recorder.feed(frame, motion=True, width=WIDTH, height=HEIGHT)
    assert recorder.recording
    assert recorder._frames_written == 1

    # The cooldown decrements once per non-motion frame from cooldown_limit down
    # to 0; only the *next* non-motion frame after it hits 0 actually stops the
    # recording. So `cooldown_limit` non-motion frames must all still record.
    for i in range(recorder._cooldown_limit):
        recorder.feed(frame, motion=False, width=WIDTH, height=HEIGHT)
        assert recorder.recording, f"stopped early at non-motion frame {i}"
    assert recorder._cooldown == 0

    # The next non-motion frame triggers _stop and is NOT written.
    recorder.feed(frame, motion=False, width=WIDTH, height=HEIGHT)
    assert not recorder.recording
    # Total frames written = 1 motion + cooldown_limit during cooldown.
    assert recorder._frames_written == 0  # reset on _stop


def test_motion_during_cooldown_resets_it(tmp_output_dir):
    """A motion frame during cooldown must re-arm the cooldown timer."""
    recorder = ClipRecorder(output_dir=tmp_output_dir, fps=FPS)
    frame = _gray_frame()

    recorder.feed(frame, motion=True, width=WIDTH, height=HEIGHT)

    # Burn off most of the cooldown
    for _ in range(recorder._cooldown_limit - 2):
        recorder.feed(frame, motion=False, width=WIDTH, height=HEIGHT)
    assert recorder.recording

    # Re-trigger motion — cooldown should reset to full
    recorder.feed(frame, motion=True, width=WIDTH, height=HEIGHT)
    assert recorder._cooldown == recorder._cooldown_limit

    # Now we should survive another (cooldown_limit - 1) non-motion frames
    for _ in range(recorder._cooldown_limit - 1):
        recorder.feed(frame, motion=False, width=WIDTH, height=HEIGHT)
        assert recorder.recording


# -----------------------------------------------------------------------------
# ClipRecorder: multiple events, close(), failure path
# -----------------------------------------------------------------------------

def test_multiple_events_produce_distinct_files(tmp_output_dir):
    recorder = ClipRecorder(output_dir=tmp_output_dir, fps=FPS)
    frame = _gray_frame()

    def _one_event():
        recorder.feed(frame, motion=True, width=WIDTH, height=HEIGHT)
        for _ in range(recorder._cooldown_limit + 1):
            recorder.feed(frame, motion=False, width=WIDTH, height=HEIGHT)
        assert not recorder.recording

    _one_event()
    # Filename timestamps have 1-second resolution; sleep so the second event
    # lands in a new second and gets a distinct filename.
    time.sleep(1.1)
    _one_event()

    now = datetime.datetime.now()
    target_dir = os.path.join(
        tmp_output_dir, now.strftime("%Y"), now.strftime("%m"), now.strftime("%d"),
    )
    files = sorted(os.listdir(target_dir))
    assert len(files) == 2, f"Expected 2 recordings, found: {files}"
    assert files[0] != files[1]


def test_close_flushes_in_progress_recording(tmp_output_dir):
    """Simulates Ctrl+C mid-recording: close() must release the writer."""
    recorder = ClipRecorder(output_dir=tmp_output_dir, fps=FPS)
    frame = _gray_frame()

    recorder.feed(frame, motion=True, width=WIDTH, height=HEIGHT)
    assert recorder.recording

    recorder.close()
    assert not recorder.recording

    # close() is idempotent
    recorder.close()
    assert not recorder.recording

    now = datetime.datetime.now()
    target_dir = os.path.join(
        tmp_output_dir, now.strftime("%Y"), now.strftime("%m"), now.strftime("%d"),
    )
    files = os.listdir(target_dir)
    assert len(files) == 1
    # File should exist and be non-empty (writer released cleanly).
    assert os.path.getsize(os.path.join(target_dir, files[0])) > 0


def test_videowriter_open_failure_is_handled(tmp_output_dir, monkeypatch, caplog):
    """If VideoWriter fails to open, the recorder must not enter recording state."""
    class FakeWriter:
        def __init__(self, *args, **kwargs):
            pass

        def isOpened(self):
            return False

        def release(self):
            pass

        def write(self, frame):  # pragma: no cover - should never be called
            raise AssertionError("write() called after open failure")

    import hailo_apps.python.pipeline_apps.detection.motion_recorder as mr
    monkeypatch.setattr(mr, "FFmpegVideoWriter", FakeWriter)

    recorder = ClipRecorder(output_dir=tmp_output_dir, fps=FPS)
    frame = _gray_frame()
    recorder.feed(frame, motion=True, width=WIDTH, height=HEIGHT)

    assert not recorder.recording, "Recorder should not be in recording state when writer fails to open"


# -----------------------------------------------------------------------------
# MotionDetector
# -----------------------------------------------------------------------------

def test_first_frame_returns_no_boxes():
    detector = MotionDetector(min_area=10, threshold=15)
    boxes = detector.detect(_gray_frame(), orig_w=WIDTH, orig_h=HEIGHT)
    assert boxes == []


def test_synthetic_delta_is_detected():
    detector = MotionDetector(min_area=50, threshold=15)
    # Prime background with a uniform frame
    detector.detect(_gray_frame(), orig_w=WIDTH, orig_h=HEIGHT)

    # Now introduce a bright square — should produce at least one box.
    frame = _frame_with_square(value=240, x=200, y=150, size=120)
    boxes = detector.detect(frame, orig_w=WIDTH, orig_h=HEIGHT)

    assert len(boxes) >= 1
    # And at least one box should overlap the square's region.
    xmin, ymin, xmax, ymax = boxes[0]
    assert xmax > xmin and ymax > ymin
    # Box coordinates must be within original frame bounds
    assert 0 <= xmin < WIDTH and 0 <= xmax <= WIDTH
    assert 0 <= ymin < HEIGHT and 0 <= ymax <= HEIGHT


def test_below_min_area_is_ignored():
    """A single-pixel delta is below any sensible min_area and must be filtered."""
    detector = MotionDetector(min_area=10_000, threshold=15)
    detector.detect(_gray_frame(), orig_w=WIDTH, orig_h=HEIGHT)

    # 4x4 bright square — area = 16 px², way under min_area=10_000.
    frame = _frame_with_square(value=240, x=300, y=200, size=4)
    boxes = detector.detect(frame, orig_w=WIDTH, orig_h=HEIGHT)

    assert boxes == []


def test_below_threshold_is_ignored():
    """A delta smaller than the per-pixel threshold must be filtered."""
    detector = MotionDetector(min_area=10, threshold=50)
    base = 80
    detector.detect(_gray_frame(base), orig_w=WIDTH, orig_h=HEIGHT)

    # +5 brightness on a 200x200 patch: large area but well under threshold=50.
    frame = _frame_with_square(value=base + 5, x=100, y=100, size=200, base=base)
    boxes = detector.detect(frame, orig_w=WIDTH, orig_h=HEIGHT)

    assert boxes == []
