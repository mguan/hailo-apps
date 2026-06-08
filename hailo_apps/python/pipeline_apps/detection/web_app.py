import logging
import os
import time
import cv2
import json
import shutil
from flask import Flask, render_template, Response, jsonify, send_from_directory, request

app = Flask(__name__)

import threading

# Global variables to store the latest frame and tracking sequence
frame_lock = threading.Lock()
shared_frame = None
frame_seq = 0

def get_shared_frame():
    """Retrieve the latest frame and its sequence atomically."""
    with frame_lock:
        return shared_frame, frame_seq

def set_shared_frame(frame):
    """Update the shared frame and increment the sequence atomically."""
    global shared_frame, frame_seq
    with frame_lock:
        shared_frame = frame
        frame_seq += 1

# Cap the browser stream so JPEG encoding cannot monopolize the CPU and starve
# the inference loop. The detector itself is unaffected by this rate.
STREAM_FPS = 15

# Ensure clips directory exists
CLIPS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'clips')
os.makedirs(CLIPS_DIR, exist_ok=True)

@app.route('/')
def index():
    """Serve the main dashboard."""
    return render_template('index.html')

def gen_frames():
    """Generator for MJPEG stream."""
    min_interval = 1.0 / STREAM_FPS
    last_seq = -1
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 80]
    while True:
        frame, seq = get_shared_frame()
        if frame is None or seq == last_seq:
            # No frame yet, or nothing new since we last encoded one.
            time.sleep(min_interval)
            continue

        last_seq = seq
        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame, encode_params)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        # Yield the frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # Throttle so the stream never encodes faster than STREAM_FPS.
        time.sleep(min_interval)

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/clips')
def get_clips():
    """Return a JSON list of available video clips."""
    clips = []
    if os.path.exists(CLIPS_DIR):
        for root, dirs, files in os.walk(CLIPS_DIR):
            for filename in files:
                if filename.endswith(('.mp4', '.webm', '.avi')):
                    rel_path = os.path.relpath(os.path.join(root, filename), CLIPS_DIR)
                    clips.append(rel_path.replace('\\', '/'))
    clips.sort(reverse=True)
    return jsonify(clips)

@app.route('/clips/<path:filename>')
def serve_clip(filename):
    """Serve a specific video clip."""
    # Ensure CLIPS_DIR ends with a path separator to prevent partial prefix match bypasses
    clips_dir_abs = os.path.abspath(CLIPS_DIR)
    if not clips_dir_abs.endswith(os.path.sep):
        clips_dir_abs += os.path.sep

    # Resolve path safely and ensure it remains inside CLIPS_DIR
    filepath = os.path.abspath(os.path.join(CLIPS_DIR, filename))
    if not filepath.startswith(clips_dir_abs) or not os.path.isfile(filepath):
        return jsonify({"status": "error", "message": "Access denied."}), 403

    return send_from_directory(CLIPS_DIR, filename)

@app.route('/api/clips/<path:filename>', methods=['DELETE'])
def delete_clip(filename):
    """Delete a specific video clip."""
    # Ensure CLIPS_DIR ends with a path separator to prevent partial prefix match bypasses
    clips_dir_abs = os.path.abspath(CLIPS_DIR)
    if not clips_dir_abs.endswith(os.path.sep):
        clips_dir_abs += os.path.sep

    # Resolve path safely and ensure it remains inside CLIPS_DIR
    filepath = os.path.abspath(os.path.join(CLIPS_DIR, filename))
    if not filepath.startswith(clips_dir_abs):
        return jsonify({"status": "error", "message": "Access denied."}), 403

    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            # Clean up empty parent directories up to CLIPS_DIR
            parent = os.path.dirname(filepath)
            while parent != CLIPS_DIR and len(parent) > len(CLIPS_DIR):
                if not os.listdir(parent):
                    os.rmdir(parent)
                    parent = os.path.dirname(parent)
                else:
                    break
            return jsonify({"status": "success", "message": f"Clip {filename} deleted."}), 200
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    else:
        return jsonify({"status": "error", "message": "File not found."}), 404

@app.route('/api/clips/delete', methods=['POST'])
def delete_clips_bulk():
    """Delete multiple clips or subdirectories."""
    data = request.get_json() or {}
    paths = data.get('paths', [])
    
    if not paths:
        return jsonify({"status": "error", "message": "No paths provided."}), 400
        
    deleted = []
    errors = []
    
    # Ensure CLIPS_DIR ends with a path separator to prevent partial prefix match bypasses
    clips_dir_abs = os.path.abspath(CLIPS_DIR)
    if not clips_dir_abs.endswith(os.path.sep):
        clips_dir_abs += os.path.sep

    for path in paths:
        filepath = os.path.abspath(os.path.join(CLIPS_DIR, path))
        if not filepath.startswith(clips_dir_abs):
            errors.append({"path": path, "error": "Access denied."})
            continue
            
        if os.path.exists(filepath):
            try:
                if os.path.isdir(filepath):
                    shutil.rmtree(filepath)
                else:
                    os.remove(filepath)
                deleted.append(path)
            except Exception as e:
                errors.append({"path": path, "error": str(e)})
        else:
            errors.append({"path": path, "error": "Not found."})
            
    # Clean up empty parent directories
    if os.path.exists(CLIPS_DIR):
        for root, dirs, files in os.walk(CLIPS_DIR, topdown=False):
            for d in dirs:
                dir_path = os.path.join(root, d)
                if not os.listdir(dir_path):
                    try:
                        os.rmdir(dir_path)
                    except Exception:
                        pass
                        
    if errors and not deleted:
        return jsonify({"status": "error", "message": "Failed to delete items.", "errors": errors}), 500
        
    return jsonify({
        "status": "success", 
        "message": f"Successfully deleted {len(deleted)} items.",
        "deleted": deleted,
        "errors": errors
    }), 200

def start_server(host='0.0.0.0', port=5000):
    """Start the Flask server. Designed to be run in a thread."""
    # Suppress werkzeug request logging (GET, POST request logs)
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    # Run with reloader=False to avoid issues when running in a background thread
    app.run(host=host, port=port, debug=False, use_reloader=False)

if __name__ == '__main__':
    start_server()
