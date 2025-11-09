import os, sys, time, uuid
from pathlib import Path
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# === ✅ Ensure src/ is importable no matter how you run it ===
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # should be project root: /Users/ohqishean/video-reid
SRC = ROOT / "src"
if SRC.exists() and str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# === ✅ Now you can safely import ===
from src.reid.extractor import ReidExtractor
from src.detectors.yolo import PersonDetector  # if used

# === Paths ===
VIDEO_PATH = ROOT / "data" / "raw" / "terrace1-c0.avi"
OUT_PATH = ROOT / "runs" / "track" / f"{VIDEO_PATH.stem}_reid.mp4"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# === Config ===
device = "cpu" if torch.cpu.is_available() else "cpu"
model = YOLO(ROOT / "yolov8n.pt")
model.to(device)
IMGSZ = 640
CONF = 0.3
IOU_THRESH = 0.45
CLASSES = [0]  # person

MAX_AGE = 30
REID_SIM_THRESH = 0.5

# === ReID extractor ===
reid = ReidExtractor(model_name="osnet_x1_0", device=device)

# === Tracker state ===
next_id = 1
tracks = {}  # tid -> {"bbox": (x1,y1,x2,y2), "feat": np.array, "age": int}

# === Utils ===
def iou(b1, b2):
    x1, y1, x2, y2 = b1
    xx1, yy1, xx2, yy2 = b2
    xi1, yi1 = max(x1, xx1), max(y1, yy1)
    xi2, yi2 = min(x2, xx2), min(y2, yy2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (xx2 - xx1) * (yy2 - yy1)
    return inter / (area1 + area2 - inter + 1e-6)

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)

def draw_tracks(img, tracks):
    for tid, t in tracks.items():
        x1, y1, x2, y2 = map(int, t["bbox"])
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(img, f"ID {tid}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

# === Main ===
def main():
    global next_id, tracks
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    assert cap.isOpened(), f"❌ Cannot open video: {VIDEO_PATH}"

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out = cv2.VideoWriter(str(OUT_PATH), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    print(f"▶ Processing {VIDEO_PATH.name} → {OUT_PATH}")
    t0, frame_count = time.time(), 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_count += 1

        results = model.predict(
            source=frame,
            imgsz=IMGSZ,
            conf=CONF,
            iou=IOU_THRESH,
            classes=CLASSES,
            device=device,
            verbose=False
        )[0]

        bboxes = results.boxes.xyxy.cpu().numpy().astype(int) if results.boxes else []
        crops, new_dets = [], []
        for box in bboxes:
            x1, y1, x2, y2 = np.clip(box[:4], 0, [w-1, h-1, w-1, h-1])
            if x2 <= x1 or y2 <= y1:
                continue
            crops.append(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
            new_dets.append((x1, y1, x2, y2))

        new_feats = reid(crops) if crops else []

        assigned = set()
        for i, (x1, y1, x2, y2) in enumerate(new_dets):
            best_tid, best_sim = None, -1
            for tid, track in tracks.items():
                if track["age"] > MAX_AGE:
                    continue
                iou_score = iou((x1,y1,x2,y2), track["bbox"])
                sim = cosine_sim(new_feats[i], track["feat"])
                if iou_score > 0.1 and sim > REID_SIM_THRESH and sim > best_sim:
                    best_tid, best_sim = tid, sim
            if best_tid is not None:
                tracks[best_tid]["bbox"] = (x1, y1, x2, y2)
                tracks[best_tid]["feat"] = new_feats[i]
                tracks[best_tid]["age"] = 0
                assigned.add(i)

        for tid in list(tracks):
            tracks[tid]["age"] += 1
            if tracks[tid]["age"] > MAX_AGE:
                del tracks[tid]

        for i, (x1, y1, x2, y2) in enumerate(new_dets):
            if i in assigned:
                continue
            tracks[next_id] = {
                "bbox": (x1, y1, x2, y2),
                "feat": new_feats[i],
                "age": 0
            }
            next_id += 1

        draw_tracks(frame, tracks)
        out.write(frame)

        if frame_count % 50 == 0:
            fps_now = frame_count / (time.time() - t0)
            print(f"   {frame_count:5d} frames  |  {fps_now:.1f} FPS")

    cap.release()
    out.release()
    print(f"✅ Done: {OUT_PATH.name} | frames={frame_count}")

if __name__ == "__main__":
    main()
