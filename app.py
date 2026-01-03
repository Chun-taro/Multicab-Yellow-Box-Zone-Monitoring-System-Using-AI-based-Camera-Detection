from flask import Flask, request, jsonify
from routes.dashboard_routes import dashboard_bp
from routes.api_routes import api_bp
from utils.helpers import setup_logging, save_violation_image
import threading
from ai_model.detect import VehicleDetector, TORCH_AVAILABLE as AI_AVAILABLE
from ai_model.tracker import CentroidTracker
from ai_model.stop_timer import StopTimer
from camera.camera import CameraHandler
from database.database import Database
from utils.zone import is_vehicle_in_zone
from config.config import config

app = Flask(__name__)
app.register_blueprint(dashboard_bp)
app.register_blueprint(api_bp, url_prefix='/api')

setup_logging()

# Global variables for monitoring
if AI_AVAILABLE:
    detector = VehicleDetector(MODEL_PATH)
    tracker = CentroidTracker()
    stop_timer = StopTimer()
else:
    detector = None
    tracker = None
    stop_timer = None
camera = CameraHandler()

def monitoring_loop():
    if not AI_AVAILABLE:
        print("AI modules not available. Monitoring disabled.")
        return
    camera.start()
    db = Database()
    while True:
        frame = camera.read_frame()
        if frame is None:
            continue
        detections = detector.detect(frame)
        rects = [d['bbox'] for d in detections if d['class'] == 2]  # class 2 for car
        objects = tracker.update(rects)
        for object_id, centroid in objects.items():
            bbox = rects[list(objects.keys()).index(object_id)]
            in_zone = is_vehicle_in_zone(bbox)
            stop_timer.update(object_id, in_zone)
            if stop_timer.check_violation(object_id, STOP_TIME_LIMIT):
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                image_filename = f'violation_{object_id}_{timestamp.replace(" ", "_")}.jpg'
                save_violation_image(frame, image_filename)
                db.insert_violation(object_id, timestamp, stop_timer.get_stop_time(object_id), image_filename)
        time.sleep(0.1)

@app.route('/set_camera', methods=['POST'])
def set_camera():
    return jsonify({'error': 'camera selection disabled; using camera 1'}), 403

if __name__ == '__main__':
    monitoring_thread = threading.Thread(target=monitoring_loop)
    monitoring_thread.start()
    app.run(debug=True)
