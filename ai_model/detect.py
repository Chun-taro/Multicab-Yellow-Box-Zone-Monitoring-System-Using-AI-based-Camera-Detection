try:
    from ultralytics import YOLO
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

class VehicleDetector:
    def __init__(self, model_path, conf_thres=0.5):
        if not TORCH_AVAILABLE:
            raise ImportError("Ultralytics not available. Install it via 'pip install ultralytics'")
        self.model = YOLO(model_path)
        self.conf_thres = conf_thres

    def detect(self, frame):
        # Inference with YOLOv8
        # The model handles preprocessing (resizing, normalization) and NMS internally.
        # 'verbose=False' prevents printing detection details to the console on every frame.
        results = self.model(frame, conf=self.conf_thres, verbose=False)

        detections = []
        # The result object contains detections for the single frame.
        for result in results:
            boxes = result.boxes  # Boxes object for bounding box outputs
            for i in range(len(boxes)):
                # Bounding box coordinates
                xyxy = boxes.xyxy[i].cpu().numpy()
                detections.append({
                    'bbox': [int(x) for x in xyxy],
                    'confidence': float(boxes.conf[i].cpu().numpy()),
                    'class': int(boxes.cls[i].cpu().numpy())
                })
        return detections
