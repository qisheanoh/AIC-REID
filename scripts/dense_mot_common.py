from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment  # type: ignore
except Exception:  # pragma: no cover
    linear_sum_assignment = None


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT / "experiments" / "dense_mot_cam1"
RUN_ROOT = ROOT / "runs" / "dense_mot_cam1"
VIDEO_PATH = ROOT / "data" / "raw" / "retail-shop" / "CAM1.mp4"
SELECTED_FRAMES_CSV = EXPERIMENT_ROOT / "selected_frames.csv"
FRAMES_DIR = EXPERIMENT_ROOT / "frames"
DENSE_GT_TEMPLATE_CSV = EXPERIMENT_ROOT / "dense_gt_template.csv"
DENSE_GT_CSV = EXPERIMENT_ROOT / "dense_gt.csv"
ANNOTATION_GUIDE_MD = EXPERIMENT_ROOT / "ANNOTATION_GUIDE.md"
DENSE_README_MD = EXPERIMENT_ROOT / "README.md"
GT_VALIDATION_JSON = EXPERIMENT_ROOT / "gt_validation_report.json"
MOT_ROOT = EXPERIMENT_ROOT / "mot_gt"
MOT_GT_TXT = MOT_ROOT / "gt" / "gt.txt"
MOT_SEQINFO_INI = MOT_ROOT / "seqinfo.ini"

CAM1_FRAME_WIDTH = 2560
CAM1_FRAME_HEIGHT = 1944
CAM1_EXPECTED_FRAMES = 1800
CAM1_FPS = 12.0

DEFAULT_WINDOWS: Sequence[Tuple[int, int]] = (
    (0, 29),
    (180, 209),
    (360, 389),
    (540, 569),
    (720, 749),
    (900, 929),
    (1080, 1109),
    (1260, 1289),
    (1440, 1469),
    (1710, 1739),
)


@dataclass
class SelectedFrameRow:
    subset_frame_idx: int
    original_frame_idx: int
    window_id: int
    video_path: str
    image_path: str
    notes: str


def ensure_dense_dirs() -> None:
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)


def _window_note(window_id: int) -> str:
    notes = {
        0: "early_scene_entry_and_initial_tracks",
        1: "early_crowding_and_crossing_candidate",
        2: "mid_early_reentry_candidate",
        3: "mid_overlap_candidate",
        4: "mid_clip_identity_pressure",
        5: "late_mid_reentry_candidate",
        6: "late_clip_overlap_and_occlusion_candidate",
        7: "late_clip_group_motion_candidate",
        8: "near_end_density_and_visibility_shift",
        9: "end_segment_reentry_and_departure_candidate",
    }
    return notes.get(int(window_id), "window_sample")


def build_selected_rows(
    *,
    video_path: Path = VIDEO_PATH,
    windows: Sequence[Tuple[int, int]] = DEFAULT_WINDOWS,
) -> List[SelectedFrameRow]:
    rows: List[SelectedFrameRow] = []
    subset_idx = 0
    for win_id, (start_f, end_f) in enumerate(windows):
        if int(end_f) < int(start_f):
            raise ValueError(f"Invalid window {win_id}: start={start_f}, end={end_f}")
        for fr in range(int(start_f), int(end_f) + 1):
            img_name = f"cam1_f{fr:06d}.jpg"
            rows.append(
                SelectedFrameRow(
                    subset_frame_idx=int(subset_idx),
                    original_frame_idx=int(fr),
                    window_id=int(win_id),
                    video_path=str(video_path),
                    image_path=str(FRAMES_DIR / img_name),
                    notes=f"window_based_dense_subset;{_window_note(win_id)}",
                )
            )
            subset_idx += 1
    return rows


def write_selected_frames_csv(rows: Sequence[SelectedFrameRow], out_csv: Path = SELECTED_FRAMES_CSV) -> None:
    fieldnames = [
        "subset_frame_idx",
        "original_frame_idx",
        "window_id",
        "video_path",
        "image_path",
        "notes",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "subset_frame_idx": int(r.subset_frame_idx),
                    "original_frame_idx": int(r.original_frame_idx),
                    "window_id": int(r.window_id),
                    "video_path": str(r.video_path),
                    "image_path": str(r.image_path),
                    "notes": str(r.notes),
                }
            )


def extract_selected_frames(
    *,
    video_path: Path,
    rows: Sequence[SelectedFrameRow],
    overwrite: bool = True,
) -> Dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    extracted = 0
    missing = 0
    for row in sorted(rows, key=lambda x: int(x.original_frame_idx)):
        img_path = Path(row.image_path)
        if img_path.exists() and not overwrite:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(row.original_frame_idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            missing += 1
            continue

        img_path.parent.mkdir(parents=True, exist_ok=True)
        if cv2.imwrite(str(img_path), frame):
            extracted += 1
        else:
            missing += 1

    cap.release()
    return {
        "video_path": str(video_path),
        "requested_frames": int(len(rows)),
        "extracted_frames": int(extracted),
        "failed_frames": int(missing),
        "frames_dir": str(FRAMES_DIR),
    }


def write_dense_gt_template(path: Path = DENSE_GT_TEMPLATE_CSV) -> None:
    fieldnames = [
        "frame_idx",
        "person_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "visibility",
        "ignore",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_selected_frame_map(selected_csv: Path = SELECTED_FRAMES_CSV) -> Dict[int, Dict[str, Any]]:
    rows = _read_csv_rows(selected_csv)
    out: Dict[int, Dict[str, Any]] = {}
    for raw in rows:
        fr = int(float(raw["original_frame_idx"]))
        out[fr] = {
            "subset_frame_idx": int(float(raw["subset_frame_idx"])),
            "window_id": int(float(raw.get("window_id", 0) or 0)),
            "video_path": str(raw.get("video_path", "")),
            "image_path": str(raw.get("image_path", "")),
            "notes": str(raw.get("notes", "")),
        }
    return out


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    x_a = max(float(a[0]), float(b[0]))
    y_a = max(float(a[1]), float(b[1]))
    x_b = min(float(a[2]), float(b[2]))
    y_b = min(float(a[3]), float(b[3]))
    inter = max(0.0, x_b - x_a) * max(0.0, y_b - y_a)
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    denom = area_a + area_b - inter + 1e-9
    return float(inter / denom) if denom > 0 else 0.0


def one_to_one_match(
    gt_boxes: Sequence[Sequence[float]],
    pred_boxes: Sequence[Sequence[float]],
    iou_threshold: float,
) -> List[Tuple[int, int, float]]:
    if not gt_boxes or not pred_boxes:
        return []

    n_gt = len(gt_boxes)
    n_pr = len(pred_boxes)
    iou_mat = np.zeros((n_gt, n_pr), dtype=np.float32)
    for gi, g in enumerate(gt_boxes):
        for pi, p in enumerate(pred_boxes):
            iou_mat[gi, pi] = float(iou_xyxy(g, p))

    matches: List[Tuple[int, int, float]] = []
    if linear_sum_assignment is not None:
        cost = 1.0 - iou_mat
        rr, cc = linear_sum_assignment(cost)
        for gi, pi in zip(rr.tolist(), cc.tolist()):
            iou_v = float(iou_mat[int(gi), int(pi)])
            if iou_v >= float(iou_threshold):
                matches.append((int(gi), int(pi), iou_v))
        return matches

    # Greedy fallback if scipy is unavailable.
    edges: List[Tuple[float, int, int]] = []
    for gi in range(n_gt):
        for pi in range(n_pr):
            iou_v = float(iou_mat[gi, pi])
            if iou_v >= float(iou_threshold):
                edges.append((iou_v, gi, pi))
    edges.sort(key=lambda x: x[0], reverse=True)

    used_g: set[int] = set()
    used_p: set[int] = set()
    for iou_v, gi, pi in edges:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(int(gi))
        used_p.add(int(pi))
        matches.append((int(gi), int(pi), float(iou_v)))
    return matches


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_float(raw: Any, default: float = 0.0) -> float:
    try:
        if raw is None or str(raw).strip() == "":
            return float(default)
        return float(raw)
    except Exception:
        return float(default)


def parse_int(raw: Any, default: int = 0) -> int:
    try:
        if raw is None or str(raw).strip() == "":
            return int(default)
        return int(round(float(raw)))
    except Exception:
        return int(default)


def load_csv(path: Path) -> List[Dict[str, str]]:
    return _read_csv_rows(path)

