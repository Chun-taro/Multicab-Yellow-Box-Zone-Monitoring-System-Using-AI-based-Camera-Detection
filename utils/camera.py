import cv2
import numpy as np
import logging
from config.config import config

class CameraHandler:
    def __init__(self):
        self.source = config.camera_source
        self.cap = None
        self.use_placeholder = False
        self.open()

    def open(self):
        self.close()
        try:
            idx = int(self.source)
            # Attempt 1: use DirectShow backend on Windows
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                self.cap = cap
                self.use_placeholder = False
                logging.info("Opened camera index %s (DSHOW)", idx)
                return True
            cap.release()
            
            # Attempt 2: Fallback to default backend
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                self.cap = cap
                self.use_placeholder = False
                logging.info("Opened camera index %s (Default)", idx)
                return True
            cap.release()
        except (ValueError, TypeError):
            # Source is not an integer, try opening as file path
            cap = cv2.VideoCapture(self.source)
            if cap.isOpened():
                self.cap = cap
                self.use_placeholder = False
                logging.info("Opened video source %s", self.source)
                return True
            cap.release()

        logging.error("Unable to open camera source %s, using placeholder", self.source)
        self.cap = None
        self.use_placeholder = True
        return False

    def set_camera(self, source):
        # switching disabled; keep locked to config.camera_source
        logging.info("Camera switching disabled; locked to index %s", config.camera_source)
        return False

    def read_frame(self):
        if self.use_placeholder or self.cap is None:
            return self.read()[1]
        ok, frame = self.cap.read()
        if not ok:
            # If it's a video file, loop it
            if isinstance(self.source, str):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                logging.info("Video ended, looping back to start.")
                ok, frame = self.cap.read()
                if ok:
                    # Resize frame to match config
                    h = getattr(config, 'FRAME_HEIGHT', 480)
                    w = getattr(config, 'FRAME_WIDTH', 640)
                    return cv2.resize(frame, (w, h))
            
            self.use_placeholder = True
            return self.read()[1]
        
        # Resize frame to match config
        h = getattr(config, 'FRAME_HEIGHT', 480)
        w = getattr(config, 'FRAME_WIDTH', 640)
        return cv2.resize(frame, (w, h))

    def read(self):
        # If camera failed to open (or read fails), generate a placeholder frame
        if self.use_placeholder or self.cap is None:
            w = getattr(config, 'FRAME_WIDTH', 640)
            h = getattr(config, 'FRAME_HEIGHT', 480)
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            text = "Camera unavailable - using placeholder"
            cv2.putText(frame, text, (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            return True, frame

        ok, frame = self.cap.read()
        if not ok:
            logging.warning("Camera read failed; switching to placeholder")
            self.use_placeholder = True
            return self.read()
        return ok, frame

    def close(self):
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None