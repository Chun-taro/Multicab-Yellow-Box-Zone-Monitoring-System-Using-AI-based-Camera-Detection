# Basic test for detection
from ai_model.detect import VehicleDetector
import cv2
import numpy as np

def test_detection():
    # Create a dummy frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a rectangle to simulate a vehicle
    cv2.rectangle(frame, (100, 100), (200, 200), (255, 0, 0), -1)
    detector = VehicleDetector('ai_model/yolov5/yolov5s.pt')
    detections = detector.detect(frame)
    print(f"Detections: {detections}")
    # In real test, check if detections are found

if __name__ == "__main__":
    test_detection()
