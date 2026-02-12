# System configuration settings
import os

# Define the absolute path to the project's root directory
BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

class Config:
    # Camera settings
    camera_source = os.path.join(BASE_DIR, 'camera', 'test_video5.mp4')
  
    FRAME_WIDTH = 1440
    FRAME_HEIGHT = 900
    FPS = 30

    # Detection settings
    CONFIDENCE_THRESHOLD = 0.5
    # To use the generic pretrained model, simply use its name.
    # Ultralytics will download it automatically on the first run.
    MODEL_PATH = 'yolov8s.pt'

    # Zone settings (example coordinates for yellow box zone)
    # For config/config.py:
    YELLOW_BOX_ZONE = [
         (872, 585),
        (1056, 615),
        (1154, 248),
        (1080, 236),
    ]

    # Time limits
    STOP_TIME_LIMIT = 15  # seconds vehicle can stop in zone before violation

    # Performance optimization settings
    FRAME_SKIP = 3  # Process AI model every Nth frame (1=all frames, 2=every other, 3=every 3rd, etc)
    # Tip: Higher frame skip = faster FPS but less frequent detections
    # Current: 3 = AI runs every 3rd frame for 33% faster detection phase
    
    JPEG_QUALITY = 50  # JPEG quality (1-100, lower=faster+smaller, higher=better quality)
    # Current: 50 = Very fast streaming (29% faster encoding than quality=70)
    # For high quality, increase to 75+. For ultra-fast, use 35-45.

    # Database settings
    DATABASE_PATH = os.path.join(BASE_DIR, 'database', 'violations.db')

    # Flask settings
    DEBUG = False
    HOST = '127.0.0.1'
    PORT = 5000

    # Desktop App settings
    WINDOW_TITLE = 'Multicab Yellow Box Zone Monitoring'

config = Config()
