from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_dense_mot_gt import validate_dense_gt


DEFAULT_SELECTED = ROOT / "experiments" / "dense_mot_cam1" / "selected_frames.csv"
DEFAULT_GT = ROOT / "experiments" / "dense_mot_cam1" / "dense_gt.csv"
DEFAULT_PROGRESS = ROOT / "experiments" / "dense_mot_cam1" / "annotation_progress.csv"
DEFAULT_VALIDATION_JSON = ROOT / "experiments" / "dense_mot_cam1" / "gt_validation_report.json"


def _safe_int(x: Any, default: int = -1) -> int:
    try:
        return int(str(x).strip())
    except Exception:
        return default


def _load_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def build_progress(
    *,
    selected_frames_csv: Path,
    gt_csv: Path,
    progress_csv: Path,
    validation_json: Path,
) -> List[Dict[str, Any]]:
    selected_rows = _load_csv(selected_frames_csv)
    if not selected_rows:
        raise RuntimeError(f"selected_frames.csv not found or empty: {selected_frames_csv}")

    window_frames: Dict[int, List[int]] = defaultdict(list)
    frame_to_window: Dict[int, int] = {}
    for r in selected_rows:
        fr = _safe_int(r.get("original_frame_idx"), -1)
        wid = _safe_int(r.get("window_id"), -1)
        if fr < 0 or wid < 0:
            continue
        window_frames[wid].append(fr)
        frame_to_window[fr] = wid

    for wid in list(window_frames.keys()):
        window_frames[wid] = sorted(window_frames[wid])

    gt_rows = _load_csv(gt_csv)
    rows_count_by_window: Dict[int, int] = defaultdict(int)
    annotated_frames_by_window: Dict[int, Set[int]] = defaultdict(set)

    for row in gt_rows:
        fr = _safe_int(row.get("frame_idx"), -1)
        if fr not in frame_to_window:
            continue
        wid = frame_to_window[fr]
        rows_count_by_window[wid] += 1
        annotated_frames_by_window[wid].add(fr)

    strict_valid_all = False
    validation_exists = gt_csv.exists()
    if validation_exists:
        rep = validate_dense_gt(
            gt_csv=gt_csv,
            selected_frames_csv=selected_frames_csv,
            out_json=validation_json,
            strict_complete=True,
        )
        strict_valid_all = bool(rep.get("valid", False))

    out_rows: List[Dict[str, Any]] = []
    for wid in sorted(window_frames.keys()):
        frames = window_frames[wid]
        frame_set = set(frames)
        ann_set = annotated_frames_by_window.get(wid, set())
        missing = sorted(frame_set - ann_set)
        annotated_frames_count = len(ann_set)
        total = len(frames)
        gt_rows_count = int(rows_count_by_window.get(wid, 0))

        if annotated_frames_count == 0:
            annotation_status = "NOT_STARTED"
        elif annotated_frames_count < total:
            annotation_status = "PARTIAL"
        else:
            annotation_status = "COMPLETE"

        if not validation_exists:
            validation_status = "NOT_VALIDATED"
        elif annotation_status != "COMPLETE":
            validation_status = "NOT_VALIDATED"
        elif strict_valid_all:
            validation_status = "STRICT_PASS"
            annotation_status = "STRICT_VALID"
        else:
            validation_status = "STRICT_FAIL"

        notes = f"annotated_frames={annotated_frames_count}/{total}; missing_count={len(missing)}"
        if missing:
            notes += f"; missing_preview={missing[:10]}"

        out_rows.append(
            {
                "window_id": wid,
                "frame_start": frames[0],
                "frame_end": frames[-1],
                "frames_count": total,
                "annotation_status": annotation_status,
                "gt_rows_count": gt_rows_count,
                "validation_status": validation_status,
                "notes": notes,
            }
        )

    _write_csv(
        progress_csv,
        out_rows,
        [
            "window_id",
            "frame_start",
            "frame_end",
            "frames_count",
            "annotation_status",
            "gt_rows_count",
            "validation_status",
            "notes",
        ],
    )
    return out_rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Update dense MOT manual annotation progress by window")
    ap.add_argument("--selected_frames", type=Path, default=DEFAULT_SELECTED)
    ap.add_argument("--gt_csv", type=Path, default=DEFAULT_GT)
    ap.add_argument("--progress_csv", type=Path, default=DEFAULT_PROGRESS)
    ap.add_argument("--validation_json", type=Path, default=DEFAULT_VALIDATION_JSON)
    args = ap.parse_args()

    rows = build_progress(
        selected_frames_csv=args.selected_frames,
        gt_csv=args.gt_csv,
        progress_csv=args.progress_csv,
        validation_json=args.validation_json,
    )

    print("window_id | frames_annotated | missing_frames | gt_rows | status | validation")
    print("-" * 78)
    for r in rows:
        total = int(r["frames_count"])
        note = str(r.get("notes", ""))
        ann = 0
        missing = 0
        for chunk in note.split(";"):
            chunk = chunk.strip()
            if chunk.startswith("annotated_frames="):
                v = chunk.split("=", 1)[1]
                ann = _safe_int(v.split("/", 1)[0], 0)
            if chunk.startswith("missing_count="):
                missing = _safe_int(chunk.split("=", 1)[1], 0)
        print(
            f"{int(r['window_id']):02d} | {ann}/{total} | {missing} | {int(r['gt_rows_count'])} | "
            f"{r['annotation_status']} | {r['validation_status']}"
        )

    print(f"\n[OK] Progress tracker updated: {args.progress_csv}")


if __name__ == "__main__":
    main()
