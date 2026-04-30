from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SELECTED = ROOT / "experiments" / "dense_mot_cam1" / "selected_frames.csv"
DEFAULT_OUT_DIR = ROOT / "experiments" / "dense_mot_cam1" / "contact_sheets"


def _safe_int(x: Any, default: int = -1) -> int:
    try:
        return int(str(x).strip())
    except Exception:
        return default


def _load_selected(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _fit_tile(img: np.ndarray, tile_w: int, tile_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(tile_w / max(1, w), tile_h / max(1, h))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((tile_h, tile_w, 3), 24, dtype=np.uint8)
    x0 = (tile_w - nw) // 2
    y0 = (tile_h - nh) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def make_contact_sheets(
    *,
    selected_frames_csv: Path,
    out_dir: Path,
    cols: int = 6,
    rows: int = 5,
    tile_w: int = 320,
    tile_h: int = 240,
) -> List[Path]:
    selected = _load_selected(selected_frames_csv)
    by_window: Dict[int, List[Dict[str, str]]] = defaultdict(list)
    for r in selected:
        wid = _safe_int(r.get("window_id"), -1)
        by_window[wid].append(r)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths: List[Path] = []

    for wid in sorted(by_window.keys()):
        rows30 = sorted(by_window[wid], key=lambda r: _safe_int(r.get("subset_frame_idx"), 0))
        grid_h = rows * tile_h
        grid_w = cols * tile_w
        grid = np.full((grid_h, grid_w, 3), 12, dtype=np.uint8)

        for i, r in enumerate(rows30[: cols * rows]):
            fr = _safe_int(r.get("original_frame_idx"), -1)
            img_path = Path(str(r.get("image_path", "")).strip())
            img = cv2.imread(str(img_path)) if img_path.exists() else None
            if img is None:
                img = np.full((tile_h, tile_w, 3), 32, dtype=np.uint8)
                cv2.putText(img, "MISSING", (10, tile_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            tile = _fit_tile(img, tile_w, tile_h)
            cv2.putText(
                tile,
                f"f={fr}",
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            rr = i // cols
            cc = i % cols
            y0 = rr * tile_h
            x0 = cc * tile_w
            grid[y0 : y0 + tile_h, x0 : x0 + tile_w] = tile

        cv2.putText(
            grid,
            f"CAM1 Dense MOT Contact Sheet - Window {wid:02d}",
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )

        out_path = out_dir / f"window_{wid:02d}.png"
        cv2.imwrite(str(out_path), grid)
        out_paths.append(out_path)

    return out_paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Create per-window contact sheets for dense MOT annotation")
    ap.add_argument("--selected_frames", type=Path, default=DEFAULT_SELECTED)
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    paths = make_contact_sheets(selected_frames_csv=args.selected_frames, out_dir=args.out_dir)
    print(f"[OK] Generated {len(paths)} contact sheets in: {args.out_dir}")
    for p in paths:
        print(f"- {p}")


if __name__ == "__main__":
    main()
