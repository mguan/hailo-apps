import logging
import os
import threading
import time
import cv2
import shutil
from flask import Flask, render_template, Response, jsonify, send_from_directory, request

server = Flask(__name__)

_frame_cond = threading.Condition()
shared_frame = None
frame_seq = 0

def get_shared_frame():
    """Retrieve the latest frame and its sequence atomically."""
    with _frame_cond:
        return shared_frame, frame_seq

def set_shared_frame(frame):
    """Update the shared frame, bump the sequence, and wake stream consumers."""
    global shared_frame, frame_seq
    with _frame_cond:
        shared_frame = frame
        frame_seq += 1
        _frame_cond.notify_all()

# Cap the browser stream so JPEG encoding cannot monopolize the CPU and starve
# the inference loop. The detector itself is unaffected by this rate.
STREAM_FPS = 15

CLIPS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'clips')


def _safe_path(filename):
    """Resolve `filename` under CLIPS_DIR or return None if it would escape it."""
    clips_dir_abs = os.path.abspath(CLIPS_DIR)
    if not clips_dir_abs.endswith(os.path.sep):
        clips_dir_abs += os.path.sep
    filepath = os.path.abspath(os.path.join(CLIPS_DIR, filename))
    if not filepath.startswith(clips_dir_abs):
        return None
    return filepath


def _prune_empty_dirs_upward(start_dir):
    """Remove `start_dir` and its parents while they're empty, stopping at CLIPS_DIR."""
    clips_root = os.path.abspath(CLIPS_DIR)
    current = os.path.abspath(start_dir)
    while current != clips_root:
        rel = os.path.relpath(current, clips_root)
        if rel.startswith(".."):
            return
        try:
            if not os.listdir(current):
                os.rmdir(current)
                current = os.path.dirname(current)
            else:
                return
        except OSError:
            return


@server.route('/')
def index():
    """Serve the main dashboard."""
    return render_template('index.html')


def gen_frames():
    """Generator for MJPEG stream. Wakes on new frames; never busy-loops."""
    min_interval = 1.0 / STREAM_FPS
    last_seq = -1
    last_emit = 0.0
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 80]
    while True:
        with _frame_cond:
            # Wait until a newer frame arrives (or 1s max so we can check for shutdown).
            _frame_cond.wait_for(lambda: frame_seq != last_seq, timeout=1.0)
            frame = shared_frame
            seq = frame_seq
        if frame is None or seq == last_seq:
            continue

        # Throttle without busy-sleeping: skip frames that come in faster than STREAM_FPS.
        now = time.monotonic()
        if now - last_emit < min_interval:
            last_seq = seq
            continue

        last_seq = seq
        last_emit = now

        ret, buffer = cv2.imencode('.jpg', frame, encode_params)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@server.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@server.route('/api/clips')
def get_clips():
    """Return a JSON list of available video clips."""
    clips = []
    if os.path.exists(CLIPS_DIR):
        for root, _, files in os.walk(CLIPS_DIR):
            for filename in files:
                if filename.endswith(('.mp4', '.webm', '.avi')):
                    rel_path = os.path.relpath(os.path.join(root, filename), CLIPS_DIR)
                    clips.append(rel_path.replace('\\', '/'))
    clips.sort(reverse=True)
    return jsonify(clips)


@server.route('/clips/<path:filename>')
def serve_clip(filename):
    """Serve a specific video clip."""
    filepath = _safe_path(filename)
    if filepath is None or not os.path.isfile(filepath):
        return jsonify({"status": "error", "message": "Access denied."}), 403
    return send_from_directory(CLIPS_DIR, filename)


@server.route('/api/clips/<path:filename>', methods=['DELETE'])
def delete_clip(filename):
    """Delete a specific video clip."""
    filepath = _safe_path(filename)
    if filepath is None:
        return jsonify({"status": "error", "message": "Access denied."}), 403

    if not os.path.exists(filepath):
        return jsonify({"status": "error", "message": "File not found."}), 404

    try:
        os.remove(filepath)
        _prune_empty_dirs_upward(os.path.dirname(filepath))
        return jsonify({"status": "success", "message": f"Clip {filename} deleted."}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@server.route('/api/clips/delete', methods=['POST'])
def delete_clips_bulk():
    """Delete multiple clips or subdirectories."""
    data = request.get_json() or {}
    paths = data.get('paths', [])

    if not paths:
        return jsonify({"status": "error", "message": "No paths provided."}), 400

    deleted = []
    errors = []
    touched_parents = set()

    for path in paths:
        filepath = _safe_path(path)
        if filepath is None:
            errors.append({"path": path, "error": "Access denied."})
            continue

        if not os.path.exists(filepath):
            errors.append({"path": path, "error": "Not found."})
            continue

        try:
            if os.path.isdir(filepath):
                shutil.rmtree(filepath)
            else:
                os.remove(filepath)
            deleted.append(path)
            touched_parents.add(os.path.dirname(filepath))
        except Exception as e:
            errors.append({"path": path, "error": str(e)})

    for parent in touched_parents:
        _prune_empty_dirs_upward(parent)

    if errors and not deleted:
        return jsonify({"status": "error", "message": "Failed to delete items.", "errors": errors}), 500

    return jsonify({
        "status": "success",
        "message": f"Successfully deleted {len(deleted)} items.",
        "deleted": deleted,
        "errors": errors
    }), 200


def start_server(host='0.0.0.0', port=5000, clips_dir=None):
    """Start the Flask server. Designed to be run in a thread."""
    global CLIPS_DIR
    if clips_dir is not None:
        CLIPS_DIR = clips_dir
    os.makedirs(CLIPS_DIR, exist_ok=True)
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    server.run(host=host, port=port, debug=False, use_reloader=False)


if __name__ == '__main__':
    start_server()
