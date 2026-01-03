# System configuration settings

class Config:
    # Camera settings
    camera_source = 0  # locked camera index (always 0)
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    FPS = 30

    # Detection settings
    CONFIDENCE_THRESHOLD = 0.5
    MODEL_PATH = 'ai_model/yolov5/yolov5s.pt'  # Path to YOLOv5 model

    # Zone settings (example coordinates for yellow box zone)
    YELLOW_BOX_ZONE = [
        (100, 200),  # top-left
        (500, 200),  # top-right
        (500, 400),  # bottom-right
        (100, 400)   # bottom-left
    ]

    # Time limits
    STOP_TIME_LIMIT = 15  # seconds vehicle can stop in zone before violation

    # Database settings
    DATABASE_PATH = 'database/violations.db'

    # Flask settings
    DEBUG = True
    HOST = '127.0.0.1'
    PORT = 5000

config = Config()
