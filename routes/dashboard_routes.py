from flask import Blueprint, render_template, Response, request, redirect, url_for, jsonify
from database.database import Database
from utils.camera import CameraHandler
from config.config import config
import cv2
import numpy as np
import torch
import os
import time
from datetime import datetime
import sys
import math
import warnings
import csv
import re
try:
    import easyocr
except ImportError:
    easyocr = None
    print("Warning: 'easyocr' not installed. Plate reading will be disabled.")

dashboard_bp = Blueprint('dashboard', __name__)
db = Database()
camera_source = 1  # Default to OBS

def process_violations(raw_data):
    """Helper to convert database tuples to dictionaries for templates."""
    processed = []
    if not raw_data:
        return processed
    for row in raw_data:
        item = {}
        # If row is already a dict-like object (e.g. sqlite3.Row), use it
        if hasattr(row, 'keys'):
            item = dict(row)
        # If row is a tuple/list, convert to dict
        elif isinstance(row, (list, tuple)):
            # Map tuple to dict assuming order: id, label, timestamp, image_path
            if len(row) > 0: item['id'] = row[0]
            if len(row) > 1: item['label'] = row[1]
            if len(row) > 2: item['timestamp'] = row[2]
            if len(row) > 3: item['image_path'] = row[3]
        
        # Extract plate number from label (Format: "car [ABC 123]")
        raw_label = item.get('label', '')
        plate_match = re.search(r'\[(.*?)\]', raw_label)
        if plate_match:
            item['plate_number'] = plate_match.group(1)
            item['label'] = raw_label.split('[')[0].strip()
        else:
            item['plate_number'] = 'Unreadable'
            
        processed.append(item)
    return processed

@dashboard_bp.route('/')
def dashboard():
    violations = process_violations(db.get_all_violations())
    return render_template('dashboard.html', violations=violations)

@dashboard_bp.route('/logs')
def logs():
    violations = process_violations(db.get_all_violations())
    return render_template('logs.html', violations=violations)

@dashboard_bp.route('/api/recent_violations')
def api_recent_violations():
    violations = process_violations(db.get_all_violations())
    return jsonify(violations[:10])

# Global model variable to avoid reloading on every request
model = None

def get_model():
    global model
    if model is None:
        # Suppress FutureWarning from YOLOv5 using deprecated torch.cuda.amp.autocast
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.autocast.*")

        # Fix for 'utils' name collision:
        # YOLOv5 uses a module named 'utils', which conflicts with the local 'utils' folder.
        # We need to remove the local 'utils' from sys.modules and sys.path temporarily.
        original_path = sys.path.copy()
        local_utils = sys.modules.pop('utils', None)
        
        # Remove current directory and dot from sys.path to prevent finding local utils
        sys.path = [p for p in sys.path if p != os.getcwd() and p != '.']
            
        # Fix for weights_only=True in newer PyTorch versions (2.6+)
        # We temporarily override torch.load to default weights_only=False to allow loading the model
        original_load = torch.load
        def safe_load(*args, **kwargs):
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        torch.load = safe_load

        try:
            # Load YOLOv5 model
            # Check for local copy first to avoid GitHub API "Authorization" (403) errors
            local_path = os.path.join(os.getcwd(), 'ai_model', 'yolov5')
            model = None
            if os.path.exists(local_path) and os.path.exists(os.path.join(local_path, 'hubconf.py')):
                try:
                    model = torch.hub.load(local_path, 'yolov5s', source='local', pretrained=True)
                except Exception as e:
                    print(f"Local model load failed: {e}. Falling back to GitHub.")
            
            if model is None:
                # Fallback to GitHub, removed force_reload=True to prevent rate limiting
                model = torch.hub.load('ultralytics/yolov5:v7.0', 'yolov5s', pretrained=True)
            
            # Filter classes: 0=person, 2=car, 3=motorcycle, 5=bus, 7=truck
            model.classes = [0, 2, 3, 5, 7]
        finally:
            torch.load = original_load
            sys.path = original_path
            if local_utils:
                sys.modules['utils'] = local_utils
    return model

# Global OCR reader
reader = None
def get_reader():
    global reader
    if reader is None and easyocr is not None:
        # Initialize EasyOCR reader (English)
        reader = easyocr.Reader(['en'])
    return reader

class Tracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        self.id_count = 0

    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h, label = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 50: # Distance threshold to match ID
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, label, id])
                    same_object_detected = True
                    break

            # New object detection
            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, label, self.id_count])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        for obj in objects_bbs_ids:
            _, _, _, _, _, object_id = obj
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return objects_bbs_ids

def generate_frames():
    camera = CameraHandler()
    model = None
    model_error = None
    try:
        model = get_model()
    except Exception as e:
        model_error = str(e)
        print(f"Warning: AI Model failed to load: {e}")

    # Check if camera opened successfully
    if camera.use_placeholder:
        print(f"Camera error: Unable to open source. Using placeholder.")
        # Create a placeholder frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "Camera not available", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        placeholder = buffer.tobytes()
        while True:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + placeholder + b'\r\n')

    # Define Yellow Box Zone (Polygon points: [x, y])
    # Adjust these coordinates to match your actual camera view
    yellow_zone = np.array([
        [645, 360],
        [822, 369],
        [885, 697],
        [496, 675]
    ], np.int32).reshape((-1, 1, 2))

    # Setup for saving violations
    save_dir = os.path.join("static", "violations")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Initialize Tracker
    tracker = Tracker()
    
    # Dictionaries to store state per vehicle ID
    vehicle_timers = {}   # {id: start_time}
    previous_positions = {} # {id: (cx, cy)}
    violated_ids = set()  # Set of IDs that have already triggered a violation
    vehicle_plates = {}   # {id: plate_text}
    ocr_processed_ids = set() # IDs that we have already attempted to read
    resolution_checked = False

    while True:
        frame = camera.read_frame()
        if frame is None:
            break
        
        if not resolution_checked:
            h_debug, w_debug = frame.shape[:2]
            print(f"DEBUG: Current Frame Resolution: {w_debug}x{h_debug}")
            max_x = np.max(yellow_zone[:, :, 0])
            max_y = np.max(yellow_zone[:, :, 1])
            if max_x > w_debug or max_y > h_debug:
                print(f"CRITICAL WARNING: Yellow Zone points (Max: {max_x},{max_y}) are OUTSIDE the frame ({w_debug}x{h_debug}).")
                print("The zone will NOT work. Please update config.py FRAME_WIDTH/HEIGHT to match your coordinates.")
            resolution_checked = True
        
        # AI Detection
        detections_for_tracker = [] # List of [x, y, w, h, label]

        if model:
            # Convert BGR (OpenCV) to RGB (YOLOv5)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(img)
            
            # Draw the Yellow Box Zone on the frame
            cv2.polylines(frame, [yellow_zone], isClosed=True, color=(0, 255, 255), thickness=2)

            # 1. Collect Detections
            detections = results.xyxy[0].cpu().numpy()
            person_count = 0
            vehicle_count = 0
            
            for *xyxy, conf, cls in detections:
                label = model.names[int(cls)]
                x1, y1, x2, y2 = map(int, xyxy)
                w, h = x2 - x1, y2 - y1
                
                if label == 'person':
                    person_count += 1
                    # Draw people immediately (no tracking needed for violation)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    cv2.putText(frame, f"{label}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                elif label in ['car', 'truck', 'bus', 'motorcycle']:
                    vehicle_count += 1
                    detections_for_tracker.append([x1, y1, w, h, label])

            # 2. Update Tracker
            tracked_objects = tracker.update(detections_for_tracker)
            current_frame_ids = set()
            
            # 3. Process Tracked Vehicles
            for obj in tracked_objects:
                x1, y1, w, h, label, obj_id = obj
                x2, y2 = x1 + w, y1 + h
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                current_frame_ids.add(obj_id)
                
                # Check if inside Yellow Zone
                is_in_zone = cv2.pointPolygonTest(yellow_zone, (cx, cy), False) >= 0
                
                # Check if Stopped (compare with previous position)
                is_stopped = False
                if obj_id in previous_positions:
                    prev_cx, prev_cy = previous_positions[obj_id]
                    dist = math.hypot(cx - prev_cx, cy - prev_cy)
                    if dist < 3: # Threshold: moved less than 3 pixels
                        is_stopped = True
                previous_positions[obj_id] = (cx, cy)

                color = (255, 0, 0) # Default Blue (Safe)
                
                # Violation Logic
                # Ensure we only flag vehicles (double check label)
                if is_in_zone and is_stopped and label != 'person':
                    if obj_id not in vehicle_timers:
                        vehicle_timers[obj_id] = time.time() # Start timer
                    
                    # Attempt OCR early (Warning Phase) to show on screen
                    if obj_id not in ocr_processed_ids:
                        ocr_processed_ids.add(obj_id)
                        try:
                            ocr_reader = get_reader()
                            if ocr_reader:
                                h_img, w_img = frame.shape[:2]
                                pad = 20
                                c_x1, c_y1 = max(0, x1 - pad), max(0, y1 - pad)
                                c_x2, c_y2 = min(w_img, x2 + pad), min(h_img, y2 + pad)
                                crop_ocr = frame[c_y1:c_y2, c_x1:c_x2]
                                results = ocr_reader.readtext(crop_ocr, detail=0)
                                if results:
                                    p_text = " ".join([r for r in results if len(r) > 2])
                                    if len(p_text) > 3:
                                        vehicle_plates[obj_id] = p_text
                        except Exception as e:
                            print(f"Early OCR Error: {e}")
                    
                    elapsed = time.time() - vehicle_timers[obj_id]
                    time_limit = getattr(config, 'STOP_TIME_LIMIT', 15)
                    
                    if elapsed >= time_limit:
                        color = (0, 0, 255) # Red (Violation)
                        
                        if obj_id not in violated_ids:
                            # Capture Violation
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            
                            # Save cropped image (closer view) for easier identification
                            h_img, w_img, _ = frame.shape
                            pad = 50  # Padding around the vehicle
                            crop_x1, crop_y1 = max(0, x1 - pad), max(0, y1 - pad)
                            crop_x2, crop_y2 = min(w_img, x2 + pad), min(h_img, y2 + pad)
                            cropped_img = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                            
                            # Get license plate (from cache or fresh read)
                            plate_text = vehicle_plates.get(obj_id, "")
                            if not plate_text:
                                try:
                                    ocr_reader = get_reader()
                                    if ocr_reader:
                                        results = ocr_reader.readtext(cropped_img, detail=0)
                                        if results:
                                            plate_text = " ".join([r for r in results if len(r) > 2])
                                            if plate_text:
                                                vehicle_plates[obj_id] = plate_text
                                except Exception as e:
                                    print(f"OCR Error: {e}")

                            filename = os.path.join(save_dir, f"violation_{timestamp}_{label}_{obj_id}.jpg")
                            cv2.imwrite(filename, cropped_img)
                            print(f"Violation saved: {filename}")
                            
                            # Save record to database
                            try:
                                db_image_path = f"violations/violation_{timestamp}_{label}_{obj_id}.jpg"
                                db_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                
                                # Append plate info to label if found
                                db_label = f"{label} [{plate_text}]" if plate_text else label

                                # Save to CSV
                                csv_path = os.path.join(save_dir, "violation_log.csv")
                                file_exists = os.path.isfile(csv_path)
                                try:
                                    with open(csv_path, mode='a', newline='') as f:
                                        writer = csv.writer(f)
                                        if not file_exists:
                                            writer.writerow(['Time', 'Type', 'Plate Number', 'Evidence'])
                                        writer.writerow([db_timestamp, label, plate_text if plate_text else 'Unreadable', db_image_path])
                                except Exception as e:
                                    print(f"CSV Error: {e}")

                                # Ensure your Database class has an insert_violation method
                                if hasattr(db, 'insert_violation'):
                                    # Prepare image blob
                                    ret, buffer = cv2.imencode('.jpg', cropped_img)
                                    image_blob = buffer.tobytes() if ret else None
                                    
                                    try:
                                        # Try passing image blob as 4th argument (fixes 'missing argument: image' error)
                                        db.insert_violation(db_label, db_timestamp, db_image_path, image_blob)
                                    except TypeError:
                                        # Fallback to 3 arguments if database doesn't support blob
                                        db.insert_violation(db_label, db_timestamp, db_image_path)
                            except Exception as e:
                                print(f"Database error: {e}")

                            violated_ids.add(obj_id)
                        
                        # Draw text AFTER cropping so OCR doesn't read "VIOLATION" as the plate
                        cv2.putText(frame, "VIOLATION", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    else:
                        # Warning (Counting down)
                        remaining = int(time_limit - elapsed)
                        cv2.putText(frame, f"{remaining}s", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
                        color = (0, 165, 255) # Orange (Warning)
                else:
                    # Reset timer if moving or outside zone
                    if obj_id in vehicle_timers:
                        del vehicle_timers[obj_id]
                    # If it was a violation but moved, we keep it in violated_ids so we don't spam save if it stops again? 
                    # Or we can reset violated_ids if it leaves the zone.
                    if not is_in_zone and obj_id in violated_ids:
                        violated_ids.remove(obj_id)

                # Draw Bounding Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                label_text = f"{label} ID:{obj_id}"
                if obj_id in vehicle_plates:
                    label_text += f" [{vehicle_plates[obj_id]}]"
                cv2.putText(frame, label_text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                cv2.circle(frame, (cx, cy), 2, color, -1)

            # Cleanup: Remove IDs that are no longer in the frame
            # This prevents memory leaks in the dictionaries
            for obj_id in list(vehicle_timers.keys()):
                if obj_id not in current_frame_ids:
                    del vehicle_timers[obj_id]
            
            for obj_id in list(previous_positions.keys()):
                if obj_id not in current_frame_ids:
                    del previous_positions[obj_id]
            
            for obj_id in list(vehicle_plates.keys()):
                if obj_id not in current_frame_ids:
                    del vehicle_plates[obj_id]
            
            for obj_id in list(ocr_processed_ids):
                if obj_id not in current_frame_ids:
                    ocr_processed_ids.remove(obj_id)
            
            # Display counts on screen
            cv2.putText(frame, f"People: {person_count} Vehicles: {vehicle_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        elif model_error:
            # Draw error message on frame if model failed
            cv2.putText(frame, "AI Error: " + model_error, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@dashboard_bp.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
