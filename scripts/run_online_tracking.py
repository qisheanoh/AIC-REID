from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_demo import run_video_with_kpis
from src.trackers.bot_sort import BOTSORT


def main() -> None:
    p = argparse.ArgumentParser(description="Stage 1+2: online local tracking only (no global ID arbitration).")
    p.add_argument("--video", type=Path, default=Path("data/raw/retail-shop/CAM1.mp4"))
    p.add_argument("--out_csv", type=Path, default=Path("runs/kpi_batch/retail-shop_CAM1_tracks.csv"))
    p.add_argument("--det_weights", type=Path, default=Path("models/yolo_cam1_person.pt"))
    p.add_argument("--reid_weights", type=Path, default=Path("models/osnet_cam1.pth"))
    p.add_argument(
        "--reid_off",
        action="store_true",
        help="Disable online ReID to approximate ByteTrack-only behavior.",
    )
    args = p.parse_args()

    det_weights = str(args.det_weights) if args.det_weights.exists() else "yolov8m.pt"
    reid_weights = str(args.reid_weights) if args.reid_weights.exists() else None
    reid_enabled = not bool(args.reid_off)

    tracker = BOTSORT(
        id_bank=None,
        det_weights=det_weights,
        reid=reid_enabled,
        reid_weights_path=reid_weights,
        det_conf=0.24,
        track_thresh=0.30,
        match_feat_thresh=0.40,
        min_match_conf=0.20,
        strong_reid_thresh=0.78,
        long_lost_reid_thresh=0.82,
        alpha_active=0.42,
        alpha_lost=0.82,
        track_buffer=220,
        motion_max_center_dist=0.72,
        motion_max_gap=32,
        min_height_ratio=0.10,
        min_width_ratio=0.04,
    )
    run_video_with_kpis(
        video_path=args.video,
        out_tracks_csv=args.out_csv,
        tracker=tracker,
        draw=False,
    )
    if not reid_enabled:
        print("[INFO] Mode: ByteTrack-only approximation (reid=False, id_bank=None)")
    print(f"[OK] Online local tracks saved: {args.out_csv}")


if __name__ == "__main__":
    main()
