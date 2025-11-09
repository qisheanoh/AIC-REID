
from typing import List, Tuple
import numpy as np, cv2
from ultralytics import YOLO

class PersonDetector:
    def __init__(self, weights: str = "yolov8n.pt", imgsz: int = 960, conf: float = 0.35, pre_upscale: float = 1.0):
        self.model = YOLO(weights)
        self.imgsz = imgsz
        self.conf = conf
        self.pre_upscale = pre_upscale  # e.g., 2.5 = 2.5x enlarge before YOLO

    def __call__(self, frame: np.ndarray) -> List[Tuple[int,int,int,int,float]]:
        if self.pre_upscale and self.pre_upscale != 1.0:
            h, w = frame.shape[:2]
            frame = cv2.resize(
                frame, (int(w*self.pre_upscale), int(h*self.pre_upscale)),
                interpolation=cv2.INTER_CUBIC
            )
        results = self.model.predict(
            source=frame, imgsz=self.imgsz, conf=self.conf, classes=[0], verbose=False
        )
        out: List[Tuple[int,int,int,int,float]] = []
        for r in results:
            if r.boxes is None:
                continue
            boxes = r.boxes.xyxy.cpu().numpy().astype(int)
            scores = r.boxes.conf.cpu().numpy()
            for (x1,y1,x2,y2), s in zip(boxes, scores):
                out.append((int(x1), int(y1), int(x2), int(y2), float(s)))
        return out
