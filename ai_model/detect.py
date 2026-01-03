try:
    import torch
    import cv2
    import numpy as np
    import sys
    sys.path.append('ai_model/yolov5')
    from yolov5.models.experimental import attempt_load
    from yolov5.utils.general import non_max_suppression
    from yolov5.utils.torch_utils import select_device
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

class VehicleDetector:
    def __init__(self, model_path, conf_thres=0.5):
        if not TORCH_AVAILABLE:
            raise ImportError("Torch not available. Install torch to use AI detection.")
        self.device = select_device('')
        self.model = attempt_load(model_path, map_location=self.device)
        self.conf_thres = conf_thres

    def detect(self, frame):
        # Preprocess frame
        img = cv2.resize(frame, (640, 640))
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = self.model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, 0.45)

        detections = []
        for det in pred:
            if len(det):
                for *xyxy, conf, cls in reversed(det):
                    detections.append({
                        'bbox': [int(x) for x in xyxy],
                        'confidence': float(conf),
                        'class': int(cls)
                    })
        return detections
