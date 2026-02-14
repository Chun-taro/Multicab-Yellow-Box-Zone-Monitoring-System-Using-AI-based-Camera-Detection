# System configuration settings
import os

# Define the absolute path to the project's root directory
BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

class Config:
    # Camera settings
    # Use environment variable for camera source if available, otherwise use default.
    # Example: export CAMERA_SOURCE=0 for a webcam
    camera_source = os.getenv('CAMERA_SOURCE', os.path.join(BASE_DIR, 'camera', 'test_video5.mp4'))
  
    FRAME_WIDTH = 1440
    FRAME_HEIGHT = 900
    FPS = 30

    # Detection settings
    CONFIDENCE_THRESHOLD = 0.25  # Higher threshold = more accurate labels, fewer false detections
    # To use the generic pretrained model, simply use its name.
    # Ultralytics will download it automatically on the first run.
    MODEL_PATH = 'yolov8s.pt'

    # Zone settings (example coordinates for yellow box zone)
    # For config/config.py:
    YELLOW_BOX_ZONE = [
        (865, 587),
        (1057, 617),
        (1152, 247),
        (1074, 230),
    ]

    # Time limits
    STOP_TIME_LIMIT = 15  # seconds vehicle can stop in zone before violation

    # Performance optimization settings
    FRAME_SKIP = 2  # Process AI model every Nth frame (1=all frames, 2=every other, 3=every 3rd, etc)
    # Tip: Higher frame skip = faster FPS but less frequent detections
    # Current: 2 = AI runs every other frame for balanced ~20 FPS performance
    
    JPEG_QUALITY = 70  # JPEG quality (1-100, lower=faster+smaller, higher=better quality)
    # Current: 70 = High quality streaming for clear vehicle detection
    # For ultra-high quality, increase to 80-90. For faster, use 50-60.

    # Database settings
    DATABASE_PATH = os.path.join(BASE_DIR, 'database', 'violations.db')

    # Flask settings
    DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() in ('true', '1', 't')
    HOST = os.getenv('HOST', '127.0.0.1')
    PORT = int(os.getenv('PORT', 5000))

    # Desktop App settings
    WINDOW_TITLE = 'Multicab Yellow Box Zone Monitoring'

config = Config()
