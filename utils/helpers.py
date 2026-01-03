import logging
import os

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_violation_image(frame, filename):
    import cv2
    ensure_dir('static/images/screenshots')
    cv2.imwrite(f'static/images/screenshots/{filename}', frame)
