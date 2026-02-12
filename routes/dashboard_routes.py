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
import threading
import requests

dashboard_bp = Blueprint('dashboard', __name__)
db = Database()
camera_source = 1  # Default to OBS

# Global event to signal new violations for long polling
new_violation_event = threading.Event()

# Person tracking dictionaries (for detecting load/unload operations)
person_bboxes = {}  # {frame_num: [(x1, y1, x2, y2, centroid_x, centroid_y), ...]}
persons_in_vehicle = {}  # {obj_id: True/False} - tracks if a person is detected inside vehicle

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return math.hypot(point1[0] - point2[0], point1[1] - point2[1])

def is_person_inside_bbox(person_center, bbox):
    """Check if person is inside vehicle bounding box."""
    x1, y1, x2, y2 = bbox
    px, py = person_center
    return x1 <= px <= x2 and y1 <= py <= y2

def is_person_near_vehicle(person_center, vehicle_bbox, distance_threshold=100):
    """
    Check if person is near but not inside vehicle (loading/unloading).
    distance_threshold in pixels (100px ≈ 1 meter on typical camera view)
    """
    if is_person_inside_bbox(person_center, vehicle_bbox):
        return False  # Person is inside, not near
    
    # Calculate distance to vehicle bbox center
    x1, y1, x2, y2 = vehicle_bbox
    vehicle_center = ((x1 + x2) / 2, (y1 + y2) / 2)
    distance = calculate_distance(person_center, vehicle_center)
    
    return distance <= distance_threshold

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

@dashboard_bp.route('/api/wait_for_violation')
def wait_for_violation():
    """
    This is a long-polling endpoint. It holds the client's request open
    until a new violation is detected or a timeout occurs.
    """
    # Wait for the event to be set, with a timeout (e.g., 30 seconds)
    event_was_set = new_violation_event.wait(timeout=30)
    
    if event_was_set:
        new_violation_event.clear()  # Reset the event for the next violation
        return jsonify({'update': True})
    else:
        return jsonify({'update': False}) # Timed out, no new violation

# Global model variable to avoid reloading on every request
model = None

def get_model():
    global model
    if model is None:
        # Suppress FutureWarning from YOLOv5 using deprecated torch.cuda.amp.autocast
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.autocast.*")
        # Suppress UserWarning from YOLOv5 about pkg_resources being deprecated.
        # This is an issue for the library maintainers to fix and can be safely ignored.
        warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources.*")

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

# Import the more robust CentroidTracker
from ai_model.tracker import CentroidTracker

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
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        placeholder = buffer.tobytes()
        while True:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + placeholder + b'\r\n')

    # Load Yellow Box Zone from the central configuration
    yellow_zone = np.array(config.YELLOW_BOX_ZONE, np.int32).reshape((-1, 1, 2))

    # Optimization: Create a bounding box for the zone to crop the frame for detection.
    # This significantly speeds up processing by running the AI model on a smaller image.
    x_coords = yellow_zone[:, :, 0]
    y_coords = yellow_zone[:, :, 1]
    zone_x_min, zone_x_max = np.min(x_coords), np.max(x_coords)
    zone_y_min, zone_y_max = np.min(y_coords), np.max(y_coords)

    # Setup for saving violations
    save_dir = os.path.join("static", "violations")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Initialize Tracker
    # Use the more robust CentroidTracker for better performance and accuracy
    tracker = CentroidTracker(max_disappeared=30)
    
    # Dictionaries to store state per vehicle ID
    vehicle_timers = {}   # {id: start_time}
    movement_start_pos = {} # {id: (time, cx, cy)}
    is_stopped_map = {} # {id: bool}
    violated_ids = set()  # Set of IDs that have already triggered a violation
    vehicle_types = {}    # {id: vehicle_type}
    resolution_checked = False
    
    # FPS optimization parameters
    frame_skip = getattr(config, 'FRAME_SKIP', 2)  # Process every Nth frame (1=all, 2=every other, etc)
    frame_count = 0
    jpeg_quality = getattr(config, 'JPEG_QUALITY', 70)  # Lower = faster, higher = better quality
    fps_counter = {'frames': 0, 'last_time': time.time(), 'fps': 0}

    while True:
        frame = camera.read_frame()
        if frame is None:
            break
        
        frame_count += 1
        current_time = time.time()
        
        # Update FPS counter
        fps_counter['frames'] += 1
        if current_time - fps_counter['last_time'] >= 1.0:
            fps_counter['fps'] = fps_counter['frames']
            fps_counter['frames'] = 0
            fps_counter['last_time'] = current_time
        
        if not resolution_checked:
            h_debug, w_debug = frame.shape[:2]
            print(f"DEBUG: Current Frame Resolution: {w_debug}x{h_debug}")
            max_x = np.max(yellow_zone[:, :, 0])
            max_y = np.max(yellow_zone[:, :, 1])
            if max_x > w_debug or max_y > h_debug:
                print(f"CRITICAL WARNING: Yellow Zone points (Max: {max_x},{max_y}) are OUTSIDE the frame ({w_debug}x{h_debug}).")
                print("The zone will NOT work. Please update config.py FRAME_WIDTH/HEIGHT to match your coordinates.")
            resolution_checked = True
        
        # Draw the Yellow Box Zone on every frame
        cv2.polylines(frame, [yellow_zone], isClosed=True, color=(0, 255, 255), thickness=2)
        
        # --- FRAME SKIPPING OPTIMIZATION ---
        # Only run AI model every Nth frame to boost FPS
        # Tracking state carries over to skipped frames
        if model and (frame_count % frame_skip) == 0:
            # AI Detection (runs every frame_skip frames)
            detections_for_tracker = [] # List of [x, y, w, h, label]

            # --- PERFORMANCE OPTIMIZATION: Crop frame to Zone of Interest ---
            # Instead of processing the whole frame, we only run the AI model on the
            # area containing the yellow box, significantly improving FPS.
            pad = 20 # Add padding to catch vehicles entering the zone
            h_frame, w_frame = frame.shape[:2]
            crop_x1 = max(0, zone_x_min - pad)
            crop_y1 = max(0, zone_y_min - pad)
            crop_x2 = min(w_frame, zone_x_max + pad)
            crop_y2 = min(h_frame, zone_y_max + pad)
            
            zone_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

            # Convert BGR (OpenCV) to RGB (YOLOv5) and run detection on the smaller crop
            img_rgb = cv2.cvtColor(zone_crop, cv2.COLOR_BGR2RGB)
            results = model(img_rgb)

            # 1. Collect Detections
            detections = results.xyxy[0].cpu().numpy()
            
            # --- DEDUPLICATION: Remove duplicate/overlapping detections ---
            # Keep only highest confidence detection when multiple detections overlap
            detections = deduplicate_detections(
                detections, 
                iou_threshold=0.4,      # Remove if 40%+ overlap
                conf_threshold=0.5      # Minimum confidence 50%
            )
            
            bbox_to_label = {} # Helper to re-associate labels after tracking
            person_count = 0
            vehicle_count = 0
            current_persons = []  # Store person bboxes and centroids for this frame
            
            for *xyxy, conf, cls in detections:
                label = model.names[int(cls)]
                # The model returns coordinates relative to the `zone_crop`.
                # We must translate them back to the full frame's coordinate system
                # by adding the crop's top-left offset (crop_x1, crop_y1).
                x1_crop, y1_crop, x2_crop, y2_crop = map(int, xyxy)
                x1 = x1_crop + crop_x1
                y1 = y1_crop + crop_y1
                x2 = x2_crop + crop_x1
                y2 = y2_crop + crop_y1
                w, h = x2 - x1, y2 - y1
                
                # Calculate centroid for zone check
                obj_cx = (x1 + x2) / 2
                obj_cy = (y1 + y2) / 2
                
                # --- ZONE FILTER: Only process detections INSIDE the yellow box ---
                is_in_zone = cv2.pointPolygonTest(yellow_zone, (int(obj_cx), int(obj_cy)), False) >= 0
                if not is_in_zone:
                    continue  # Skip this detection, it's outside the zone
                
                if label == 'person':
                    person_count += 1
                    # Calculate person centroid
                    person_cx = (x1 + x2) / 2
                    person_cy = (y1 + y2) / 2
                    current_persons.append((x1, y1, x2, y2, person_cx, person_cy))
                    
                    # Draw people immediately (no tracking needed for violation)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    cv2.putText(frame, f"{label}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                elif label in ['car', 'truck', 'bus', 'motorcycle']:
                    vehicle_count += 1
                    # The CentroidTracker expects (startX, startY, endX, endY)
                    rect = (x1, y1, x2, y2)
                    detections_for_tracker.append(rect)
                    bbox_to_label[rect] = label

            # 2. Update Tracker
            tracked_objects_map = tracker.update(detections_for_tracker)
        else:
            # Use existing tracked objects when not processing frame
            tracked_objects_map = tracker.get_current_objects()
            person_count = 0
            vehicle_count = 0
        
        current_frame_ids = set()
        
        # 3. Process Tracked Vehicles
        for obj_id, (centroid, bbox) in tracked_objects_map.items():
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            # Re-associate the label using the bounding box
            label = bbox_to_label.get(tuple(bbox), 'vehicle') if (frame_count % frame_skip) == 0 else vehicle_types.get(obj_id, 'vehicle')

            # Calculate center point and explicitly cast to standard Python integers.
            # This prevents a cv2.error caused by passing numpy integer types to OpenCV functions.
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            current_frame_ids.add(obj_id)
            
            # Check if inside Yellow Zone
            is_in_zone = cv2.pointPolygonTest(yellow_zone, (cx, cy), False) >= 0
            
            # Check if Stopped (compare with position 1 second ago)
            current_time = time.time()
            if obj_id not in movement_start_pos:
                movement_start_pos[obj_id] = (current_time, cx, cy)
                is_stopped_map[obj_id] = False # Assume moving initially
            
            start_t, start_x, start_y = movement_start_pos[obj_id]
            if current_time - start_t >= 1.0:
                dist = math.hypot(cx - start_x, cy - start_y)
                # If moved less than 20 pixels in 1 second, consider stopped
                if dist < 20: 
                    is_stopped_map[obj_id] = True
                else:
                    is_stopped_map[obj_id] = False
                # Reset reference
                movement_start_pos[obj_id] = (current_time, cx, cy)
            
            is_stopped = is_stopped_map.get(obj_id, False)

            # --- Check for nearby persons (loading/unloading) ---
            person_nearby = False
            person_inside = False
            if current_persons:
                for px1, py1, px2, py2, pcx, pcy in current_persons:
                    # Check if person is near this vehicle
                    vehicle_bbox = (x1, y1, x2, y2)
                    
                    # Check if person is inside vehicle bbox (got in)
                    if is_person_inside_bbox((pcx, pcy), vehicle_bbox):
                        person_inside = True
                        persons_in_vehicle[obj_id] = True
                        break  # Found person inside
                    # Check if person is near vehicle (loading/unloading - 100px ≈ 1 meter)
                    elif is_person_near_vehicle((pcx, pcy), vehicle_bbox, distance_threshold=100):
                        person_nearby = True
            
            # Check if person was previously inside but is now gone (person got out)
            person_got_out = False
            if obj_id in persons_in_vehicle and persons_in_vehicle[obj_id] and not person_inside:
                person_got_out = True  # Person was inside, now they're not
            
            # Update person state for next frame
            if person_inside:
                persons_in_vehicle[obj_id] = True
            elif person_got_out:
                persons_in_vehicle[obj_id] = False

            # --- State Management & Violation Triggering ---
            time_limit = getattr(config, 'STOP_TIME_LIMIT', 15)

            # Condition to start/continue timer: in zone, stopped, and a vehicle.
            if is_in_zone and is_stopped and label != 'person':
                # Reset timer if person is loading/unloading OR got out of vehicle
                if person_nearby or person_got_out:
                    # Person is interacting with vehicle - reset timer to be fair
                    vehicle_timers[obj_id] = time.time()
                    vehicle_types[obj_id] = label
                    # Draw indicator that person is interacting with vehicle
                    cv2.putText(frame, "LOADING/UNLOADING", (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 255), 1)
                elif person_inside:
                    # Person is inside vehicle - don't reset timer
                    if obj_id not in vehicle_timers:
                        vehicle_timers[obj_id] = time.time()
                        vehicle_types[obj_id] = label
                elif obj_id not in vehicle_timers:
                    # No person interaction - start normal timer
                    vehicle_timers[obj_id] = time.time()
                    vehicle_types[obj_id] = label

                elapsed = time.time() - vehicle_timers[obj_id]

                # If time limit is exceeded, mark as violator (if not already marked)
                if elapsed >= time_limit and obj_id not in violated_ids:
                    violated_ids.add(obj_id)
                    # --- CAPTURE AND SAVE VIOLATION (RUNS ONCE) ---
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    # Save cropped image (closer view) for easier identification
                    h_img, w_img, _ = frame.shape
                    pad = 50  # Padding around the vehicle
                    crop_x1, crop_y1 = max(0, x1 - pad), max(0, y1 - pad)
                    crop_x2, crop_y2 = min(w_img, x2 + pad), min(h_img, y2 + pad)
                    cropped_img = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                    
                    # Get vehicle type
                    vehicle_type = vehicle_types.get(obj_id, label)

                    filename = os.path.join(save_dir, f"violation_{timestamp}_{label}_{obj_id}.jpg")
                    cv2.imwrite(filename, cropped_img)
                    print(f"Violation saved: {filename}")
                    
                    # Save record to database
                    try:
                        db_image_path = f"violations/violation_{timestamp}_{label}_{obj_id}.jpg"
                        db_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        # Use vehicle type as label
                        db_label = vehicle_type

                        # Save to CSV
                        csv_path = os.path.join(save_dir, "violation_log.csv")
                        file_exists = os.path.isfile(csv_path)
                        try:
                            with open(csv_path, mode='a', newline='') as f:
                                writer = csv.writer(f)
                                if not file_exists:
                                    writer.writerow(['Timestamp', 'Vehicle Type', 'Evidence'])
                                writer.writerow([db_timestamp, vehicle_type, db_image_path])
                        except Exception as e:
                            print(f"CSV Error: {e}")

                        # Ensure your Database class has an insert_violation method
                        if hasattr(db, 'insert_violation'):
                            # Prepare image blob
                            ret, buffer = cv2.imencode('.jpg', cropped_img)
                            image_blob = buffer.tobytes() if ret else None
                            
                            # Create detection ID from timestamp and tracking object ID
                            detection_id = f"{timestamp}_{obj_id}"
                            
                            try:
                                # Insert with enhanced parameters
                                db.insert_violation(
                                    vehicle_type=vehicle_type,
                                    timestamp=db_timestamp,
                                    image_path=db_image_path,
                                    image_blob=image_blob,
                                    detection_id=detection_id,
                                    stop_duration=elapsed,
                                    confidence=0.0,  # TODO: maintain confidence mapping
                                    notes=f"Object ID: {obj_id}, Stopped for {elapsed:.1f}s"
                                )
                            except TypeError as e:
                                # Fallback for old signature
                                print(f"Database method signature mismatch: {e}")
                                try:
                                    db.insert_violation(vehicle_type, db_timestamp, db_image_path, image_blob)
                                except Exception as e2:
                                    print(f"Insert violation failed: {e2}")
                        
                        # --- SYNC TO NODE.JS BACKEND (Fallback Logic) ---
                        try:
                            # Attempt to send data to the Node.js/MongoDB backend
                            # If this fails, the data is already saved in SQLite (above), so no data loss.
                            node_api_url = "http://localhost:5001/api/violations"
                            payload = {
                                "timestamp": db_timestamp,
                                "vehicle_type": vehicle_type,
                                "image_path": db_image_path
                            }
                            requests.post(node_api_url, json=payload, timeout=1)
                        except Exception as sync_error:
                            print(f"Warning: Could not sync to Node.js Backend ({sync_error}). Data saved locally only.")

                    except Exception as e:
                        print(f"Database error: {e}")
                    
                    # After the violation is processed, set the event to notify long-poll clients
                    new_violation_event.set()
            else:
                # Reset timer if vehicle is not stopped in the zone
                if obj_id in vehicle_timers:
                    del vehicle_timers[obj_id]

            # --- Only draw objects that are INSIDE the yellow zone ---
            if not is_in_zone:
                continue  # Skip drawing objects outside the zone

            # --- Drawing Logic ---
            # 1. Check if it's a confirmed, persistent violator
            if obj_id in violated_ids:
                color = (0, 0, 255) # Red for violation
                cv2.putText(frame, "VIOLATION", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            # 2. Else, check if it's in a warning state (stopped in zone, timer running)
            elif obj_id in vehicle_timers:
                elapsed = time.time() - vehicle_timers[obj_id]
                remaining = int(time_limit - elapsed)
                color = (0, 165, 255) # Orange for warning
                cv2.putText(frame, f"{remaining}s", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            # 3. Otherwise, it's a safe vehicle
            else:
                color = (255, 0, 0) # Blue for safe

            # Draw the bounding box with the determined color
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            label_text = f"{label} ID:{obj_id}"
            cv2.putText(frame, label_text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            cv2.circle(frame, (cx, cy), 2, color, -1)

        # Cleanup: Remove IDs that are no longer in the frame
        # This prevents memory leaks in the dictionaries
        for obj_id in list(vehicle_timers.keys()):
            if obj_id not in current_frame_ids:
                del vehicle_timers[obj_id]
        
        for obj_id in list(movement_start_pos.keys()):
            if obj_id not in current_frame_ids:
                del movement_start_pos[obj_id]
        
        for obj_id in list(is_stopped_map.keys()):
            if obj_id not in current_frame_ids:
                del is_stopped_map[obj_id]
        
        for obj_id in list(vehicle_types.keys()):
            if obj_id not in current_frame_ids:
                del vehicle_types[obj_id]
        
        for obj_id in list(persons_in_vehicle.keys()):
            if obj_id not in current_frame_ids:
                del persons_in_vehicle[obj_id]
        
        for obj_id in list(violated_ids):
            if obj_id not in current_frame_ids:
                violated_ids.remove(obj_id)
        
        # Display counts on screen
        cv2.putText(frame, f"People: {person_count} Vehicles: {vehicle_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        if model_error:
            # Draw error message on frame if model failed
            cv2.putText(frame, "AI Error: " + model_error, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display FPS on frame
        cv2.putText(frame, f"FPS: {fps_counter['fps']}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # --- OPTIMIZED JPEG ENCODING ---
        # Use quality setting to balance speed and image quality
        # Lower quality = faster encoding and smaller file size
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if not ret:
            continue  # Skip frame if encoding fails
        
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def deduplicate_detections(detections, iou_threshold=0.5, conf_threshold=0.5):
    """
    Remove duplicate/overlapping detections keeping only highest confidence ones.
    
    Args:
        detections: Array of detections from YOLO (xyxy format)
        iou_threshold: Intersection over Union threshold for considering detections duplicate (0.5 = 50% overlap)
        conf_threshold: Minimum confidence threshold
        
    Returns:
        Filtered detections array
    """
    if len(detections) == 0:
        return detections
    
    # Filter by confidence threshold first
    detections = detections[detections[:, 4] >= conf_threshold]
    
    if len(detections) == 0:
        return detections
    
    # Sort by confidence (descending)
    detections = detections[np.argsort(-detections[:, 4])]
    
    keep = []
    for i, det in enumerate(detections):
        x1, y1, x2, y2, conf, cls = det
        
        # Check against already kept detections
        is_duplicate = False
        for kept_det in keep:
            kx1, ky1, kx2, ky2 = kept_det[:4]
            
            # Calculate Intersection over Union (IoU)
            inter_x1 = max(x1, kx1)
            inter_y1 = max(y1, ky1)
            inter_x2 = min(x2, kx2)
            inter_y2 = min(y2, ky2)
            
            if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                box_area = (x2 - x1) * (y2 - y1)
                kept_area = (kx2 - kx1) * (ky2 - ky1)
                union_area = box_area + kept_area - inter_area
                iou = inter_area / union_area if union_area > 0 else 0
                
                # If IoU is high, it's likely the same object
                if iou > iou_threshold:
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            keep.append(det)
    
    return np.array(keep) if keep else detections[:0]  # Return empty array with correct shape if no detections

@dashboard_bp.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
