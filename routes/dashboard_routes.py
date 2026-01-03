from flask import Blueprint, render_template, Response, request, redirect, url_for
from database.database import Database
from camera.camera import CameraHandler
import cv2
import numpy as np

dashboard_bp = Blueprint('dashboard', __name__)
db = Database()
camera_source = 1  # Default to OBS

@dashboard_bp.route('/')
def dashboard():
    violations = db.get_all_violations()
    return render_template('dashboard.html', violations=violations)

@dashboard_bp.route('/logs')
def logs():
    violations = db.get_all_violations()
    return render_template('logs.html', violations=violations)

def generate_frames():
    camera = CameraHandler()
    try:
        camera.start()
    except ValueError as e:
        print(f"Camera error: {e}. Using placeholder.")
        # Create a placeholder frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "Camera not available", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        placeholder = buffer.tobytes()
        while True:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + placeholder + b'\r\n')
    while True:
        frame = camera.read_frame()
        if frame is None:
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@dashboard_bp.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
