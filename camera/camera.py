import cv2
from config.config import config

class CameraHandler:
    def __init__(self, source=None):
        if source is None:
            self.source = config.camera_source
        else:
            self.source = source
        self.cap = None

    def start(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise ValueError("Unable to open camera/video source")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, config.FPS)

    def read_frame(self):
        if self.cap is None:
            raise RuntimeError("Camera not started")
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def stop(self):
        if self.cap:
            self.cap.release()
            self.cap = None
