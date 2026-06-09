# Motion Recorder

A GStreamer-based motion-triggered video recorder. When motion is detected in the
camera feed, the app starts writing an `.mp4` clip; recording continues for a short
cooldown after motion stops, then the file is closed and the app waits for the next
event.

> Note: the original object-detection example that used to live here has been replaced
> by this motion recorder. Detection uses no Hailo inference path — motion is detected
> on the CPU via OpenCV background subtraction.

## Files

| File | Purpose |
|------|---------|
| `motion_recorder_app.py` | App entry point — `app_callback`, overlay drawing, `main()` |
| `motion_recorder_pipeline.py` | `GStreamerMotionRecorderApp` — pipeline assembly + CLI arguments |
| `motion_detector.py` | `MotionDetector` — background-subtraction motion detection |
| `clip_recorder.py` | `ClipRecorder`, `FFmpegVideoWriter` — clip recording engine |

## Running

From the repo root, with `setup_env.sh` already sourced:

```bash
# Convenience wrapper (writes to /media/pi/Backup/Events)
./run_motion_recorder.sh

# Direct invocation
python hailo_apps/python/pipeline_apps/detection/motion_recorder_app.py --input rpi
```

Press `Ctrl+C` to stop. The in-progress clip (if any) is flushed and closed cleanly
on shutdown.

## Output layout

Clips are organized by date under `--output-dir`:

```
<output-dir>/YYYY/MM/DD/motion_YYYYMMDD_HHMMSS.mp4
```

Example: `/media/pi/Backup/Events/2026/05/23/motion_20260523_142105.mp4`

The saved clip has motion-zone red boxes drawn on each frame so you can see at a
glance where the motion was. The on-screen HUD (zone count + `RECORDING` indicator)
is drawn only when `--use-frame` is set and appears only in the live preview
window, never in the saved file.

> Note on the live view: the main GStreamer window (the one that opens by default)
> shows the **raw camera buffer** — overlays drawn from Python in `app_callback`
> cannot affect it, because the buffer is mapped read-only and the numpy array is a
> copy. Pass `--use-frame` to open a second OpenCV window that does render the
> motion boxes + HUD; or just watch the saved clips, which contain the boxes.

## CLI arguments

### Motion-recorder specific

| Flag | Default | Description |
|---|---|---|
| `--motion-min-area N` | `50` | Minimum contour area (in *original-resolution* pixels²) to count as motion |
| `--motion-threshold N` | `15` | Threshold for per-pixel frame difference (0–255) |
| `--output-dir PATH` | `/home/pi/Videos` | Root directory for saved clips |
| `--debounce-seconds N` | `3.0` | Discard clips whose motion lasted ≤ N seconds |
| `--gui` | off | Open the GStreamer display window (otherwise runs headless with `fakesink`) |
| `--web-app-port N` | `5000` | Port for the web dashboard |

### Common pipeline arguments

Inherited from `get_pipeline_parser()`. Most useful ones:

| Flag | Description |
|---|---|
| `--input {usb,rpi,/dev/videoN,FILE}` | Input source — `rpi` for Pi camera, `usb` to auto-detect a webcam |
| `--width N`, `--height N` | Capture resolution (e.g. `--width 1280 --height 720`) |
| `--horizontal-mirror`, `--vertical-mirror` | Flip the camera image |
| `--use-frame` | Enable the live preview window |
| `--show-fps` | Overlay FPS on the display |

Run `python motion_recorder_app.py --help` for the full list.

## Tuning

- **Too many false triggers** → raise `--motion-threshold` (e.g. `25`) or `--motion-min-area` (e.g. `200`).
- **Missing real motion** → lower `--motion-threshold` (e.g. `10`) or `--motion-min-area` (e.g. `20`).
- **Clip cuts too soon after action stops** → `ClipRecorder.COOLDOWN_SECONDS` in
  `clip_recorder.py` (default `3.0s`). Not currently a CLI flag.
- **Background adapts too fast / too slow** → `MotionDetector.BG_LEARNING_RATE`
  (default `0.02`). Higher = adapts faster (better for changing lighting, worse for
  catching slow-moving subjects).

## Notes

- The recording fps is taken from the pipeline's configured `frame_rate`. If the
  camera can't sustain that rate, clips will play back faster than real time. Match
  `--width`/`--height`/`--input` to a combination the camera can actually deliver.
- `MotionDetector` runs on the CPU at a downscaled `640px`-wide analysis frame
  regardless of capture resolution, so cost is roughly constant.
- The Hailo accelerator is **not** used by this app — there is no NN inference in the
  pipeline. If you want object-class-aware triggering (e.g. "only record when a
  person is in frame"), look at the `pose_estimation` / `instance_segmentation`
  reference apps for how to add `hailonet` to the pipeline.

## Tests

```bash
pytest tests/test_motion_recorder.py
```
