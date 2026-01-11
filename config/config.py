# System configuration settings

class Config:
    # Camera settings
    camera_source = 'camera/test_video2.mp4'
  
    FRAME_WIDTH = 1440
    FRAME_HEIGHT = 900
    FPS = 30

    # Detection settings
    CONFIDENCE_THRESHOLD = 0.5
    MODEL_PATH = 'ai_model/yolov5/yolov5s.pt'  # Path to YOLOv5 model

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
    DATABASE_PATH = 'database/violations.db'

    # Flask settings
    DEBUG = False
    HOST = '127.0.0.1'
    PORT = 5000

    # Desktop App settings
    WINDOW_TITLE = 'Multicab Yellow Box Zone Monitoring'

config = Config()
