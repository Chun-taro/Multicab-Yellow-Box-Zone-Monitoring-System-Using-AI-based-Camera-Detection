# System configuration settings
import os

# Define the absolute path to the project's root directory
BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

class Config:
    # Camera settings
    camera_source = os.path.join(BASE_DIR, 'camera', 'test_video2.mp4')
  
    FRAME_WIDTH = 1440
    FRAME_HEIGHT = 900
    FPS = 30

    # Detection settings
    CONFIDENCE_THRESHOLD = 0.5
    # To use the generic pretrained model, simply use its name.
    # Ultralytics will download it automatically on the first run.
    MODEL_PATH = 'yolov8s.pt'

    # Zone settings (example coordinates for yellow box zone)
    YELLOW_BOX_ZONE = [
        (645, 360),
    (822, 369),
    (885, 697),
    (496, 675),
    ]

    # Time limits
    STOP_TIME_LIMIT = 15  # seconds vehicle can stop in zone before violation

    # Database settings
    DATABASE_PATH = os.path.join(BASE_DIR, 'database', 'violations.db')

    # Flask settings
    DEBUG = False
    HOST = '127.0.0.1'
    PORT = 5000

    # Desktop App settings
    WINDOW_TITLE = 'Multicab Yellow Box Zone Monitoring'

config = Config()
