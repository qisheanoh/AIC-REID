# src/pipelines/end2end_demo.py

import argparse, os, sys, time
from pathlib import Path
from typing import List, Tuple, Dict
import cv2
import numpy as np

from src.detectors.yolo import PersonDetector
try:
    from src.reid.extractor import ReidExtractor
    HAVE_REID = True
except Exception:
    HAVE_REID = False

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1,bx1), max(ay1,by1), min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw * ih
    if inter <= 0: return 0.0
    aa = (ax2-ax1)*(ay2-ay1); ba = (bx2-bx1)*(by2-by1)
    return inter / (aa + ba - inter + 1e-9)

def cosine_similarity(a, b):
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return np.dot(a, b)

class ReIDTracker:
    def __init__(self, iou_thr=0.3, sim_thr=0.4, max_age=30):
        self.iou_thr = iou_thr
        self.sim_thr = sim_thr
        self.max_age = max_age
        self.next_id = 1
        self.tracks: Dict[int, Dict] = {}

    def update(self, dets: List[Tuple[int,int,int,int,float]], features: List[np.ndarray]):
        assigned = set()
        updated_tracks = {}

        for tid, track in self.tracks.items():
            track["age"] += 1

        for j, (bbox, feat) in enumerate(zip(dets, features)):
            best_tid, best_score = None, -1
            for tid, track in self.tracks.items():
                iou_score = iou(bbox[:4], track["bbox"])
                sim_score = cosine_similarity(feat, track["feat"])
                if iou_score >= self.iou_thr and sim_score >= self.sim_thr and sim_score > best_score:
                    best_score = sim_score
                    best_tid = tid
            if best_tid is not None:
                self.tracks[best_tid] = {"bbox": tuple(map(int, bbox[:4])), "feat": feat, "age": 0}
                updated_tracks[best_tid] = tuple(map(int, bbox[:4]))
                assigned.add(j)

        for j, bbox in enumerate(dets):
            if j in assigned: continue
            self.tracks[self.next_id] = {
                "bbox": tuple(map(int, bbox[:4])),
                "feat": features[j],
                "age": 0
            }
            updated_tracks[self.next_id] = tuple(map(int, bbox[:4]))
            self.next_id += 1

        self.tracks = {tid: t for tid, t in self.tracks.items() if t["age"] <= self.max_age}
        return list(updated_tracks.items())

def parse_args():
    p = argparse.ArgumentParser("End2End Video ReID demo")
    p.add_argument("--video", type=str, required=True)
    p.add_argument("--weights", type=str, default="yolov8n.pt")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--pre-upscale", type=float, default=1.0, dest="pre_upscale")
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--auto-scan", action="store_true")
    p.add_argument("--scan-step", type=int, default=3)
    p.add_argument("--scan-backpad", type=int, default=12)
    p.add_argument("--scan-max-probe", type=int, default=800)
    p.add_argument("--track-iou", type=float, default=0.3)
    p.add_argument("--track-max-age", type=int, default=30)
    p.add_argument("--display", action="store_true")
    p.add_argument("--save", type=str, default="")
    return p.parse_args()

def open_writer(path: str, w: int, h: int, fps: float):
    if not path:
        return None, ""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*("mp4v" if path.endswith(".mp4") else "MJPG"))
    out = cv2.VideoWriter(path, fourcc, fps, (w, h))
    if out.isOpened():
        return out, path
    fallback = str(Path(path).with_suffix(".avi"))
    out = cv2.VideoWriter(fallback, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    return (out, fallback) if out.isOpened() else (None, "")

def draw_tracks(frame, tracks):
    for tid, (x1,y1,x2,y2) in tracks:
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f"ID {tid}", (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"❌ Could not open video: {args.video}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"▶️  {args.video}  [{W}x{H}]  {fps:.2f} FPS  frames={total}")

    det = PersonDetector(weights=args.weights, imgsz=args.imgsz, conf=args.conf, pre_upscale=args.pre_upscale)

    reid = None
    have_reid = HAVE_REID
    if have_reid:
        try:
            reid = ReidExtractor(model_name="osnet_x1_0", device="cpu")
            print("🔗 ReID extractor loaded (osnet_x1_0).")
        except Exception as e:
            print(f"⚠️  ReID unavailable: {e}")
            have_reid = False

    trk = ReIDTracker(iou_thr=args.track_iou, max_age=args.track_max_age)

    start_frame = args.start_frame or 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    out, out_path = open_writer(args.save, W, H, fps)
    if out_path:
        print(f"💾 Saving to: {out_path}")
    elif args.save:
        print(f"⚠️  Failed to open writer for {args.save}; continuing without saving.")

    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    t0 = time.time()
    wrote = 0
    while True:
        ok, img = cap.read()
        if not ok: break
        frame_idx += 1

        dets = det(img)
        crops, features = [], []
        for x1,y1,x2,y2,_ in dets:
            x1,y1,x2,y2 = max(0,x1),max(0,y1),min(W-1,x2),min(H-1,y2)
            crop = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            crops.append(crop)
        if reid and crops:
            features = reid(crops)
        else:
            features = [np.zeros(512)] * len(dets)  # dummy features

        tracks = trk.update(dets, features)
        draw_tracks(img, tracks)

        if out:
            out.write(img); wrote += 1
        if args.display:
            cv2.imshow("end2end_demo", img)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    if args.display: cv2.destroyAllWindows()
    if out: out.release()

    print(f"✅ Done. Frames written: {wrote}  Elapsed: {time.time()-t0:.1f}s")
    if out_path and wrote > 0:
        print(f"✅ Output at: {out_path}")
    elif args.save:
        print("⚠️  No frames written.")

if __name__ == "__main__":
    main()
