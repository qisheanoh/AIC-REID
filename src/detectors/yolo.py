# src/detectors/yolo.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import numpy as np
import cv2
from ultralytics import YOLO


class PersonDetector:
    """
    YOLOv8 person detector wrapper.

    Returns boxes in ORIGINAL frame coordinates:
      (x1, y1, x2, y2, conf)

    Notes:
    - If pre_upscale != 1.0, the frame is resized before inference to help tiny persons,
      then boxes are scaled back down to the original frame size.
    - classes=[0] means person class only.
    """

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        imgsz: int = 960,
        conf: float = 0.35,
        iou: float = 0.50,
        pre_upscale: float = 1.0,
        device: str | None = None,
    ):
        self.model = YOLO(self._resolve_weights(weights))
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.pre_upscale = float(pre_upscale) if pre_upscale else 1.0
        self.device = device  # "cuda", "cpu", or None (ultralytics auto)

    @staticmethod
    def _resolve_weights(weights: str) -> str:
        p = Path(weights)
        if p.is_absolute() and p.exists():
            return str(p)
        if p.exists():
            return str(p.resolve())

        root = Path(__file__).resolve().parents[2]
        candidate = (root / "models" / p.name).resolve()
        if candidate.exists():
            return str(candidate)
        return weights

    def __call__(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        if frame is None or frame.size == 0:
            return []

        # ---- optional pre-upscale ----
        scale = self.pre_upscale
        run_frame = frame
        if scale != 1.0:
            h, w = frame.shape[:2]
            run_frame = cv2.resize(
                frame,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_CUBIC,
            )

        # ---- YOLO inference ----
        results = self.model.predict(
            source=run_frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            classes=[0],  # person only
            device=self.device,
            verbose=False,
        )

        inv = 1.0 / scale if scale != 0 else 1.0
        H, W = frame.shape[:2]

        out: List[Tuple[int, int, int, int, float]] = []

        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue

            xyxy = r.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
            confs = r.boxes.conf.detach().cpu().numpy().astype(np.float32)

            for (x1, y1, x2, y2), s in zip(xyxy, confs):
                # scale back to ORIGINAL frame coords
                x1 *= inv
                y1 *= inv
                x2 *= inv
                y2 *= inv

                # clamp into frame bounds
                x1 = float(np.clip(x1, 0, max(0, W - 1)))
                y1 = float(np.clip(y1, 0, max(0, H - 1)))
                x2 = float(np.clip(x2, 0, max(0, W - 1)))
                y2 = float(np.clip(y2, 0, max(0, H - 1)))

                # sanity: ensure valid box
                if x2 <= x1 + 1 or y2 <= y1 + 1:
                    continue

                out.append((int(x1), int(y1), int(x2), int(y2), float(s)))

        return out
