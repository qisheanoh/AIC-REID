from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from dense_mot_common import (
    DENSE_GT_CSV,
    SELECTED_FRAMES_CSV,
    iou_xyxy,
    load_csv,
    one_to_one_match,
    parse_float,
    parse_int,
    write_json,
    write_text,
)


def _to_box(row: Dict[str, str]) -> Tuple[float, float, float, float]:
    return (
        parse_float(row.get("x1"), default=0.0),
        parse_float(row.get("y1"), default=0.0),
        parse_float(row.get("x2"), default=0.0),
        parse_float(row.get("y2"), default=0.0),
    )


def _load_gt_dense(
    *,
    gt_csv: Path,
    selected_frames: Path,
) -> Tuple[List[int], Dict[int, List[Dict[str, Any]]]]:
    selected_rows = load_csv(selected_frames)
    selected_frame_ids = [parse_int(r.get("original_frame_idx"), default=-1) for r in selected_rows]
    selected_frame_set = set(int(x) for x in selected_frame_ids)

    gt_rows = load_csv(gt_csv)
    by_frame: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for row in gt_rows:
        fr = parse_int(row.get("frame_idx"), default=-1)
        if fr not in selected_frame_set:
            continue

        ignore = parse_int(row.get("ignore"), default=0)
        pid_raw = str(row.get("person_id", "")).strip()
        pid = parse_int(pid_raw, default=-1) if pid_raw != "" else -1

        by_frame[int(fr)].append(
            {
                "row_data": dict(row),
                "frame_idx": int(fr),
                "person_id": int(pid),
                "ignore": int(ignore),
                "box": _to_box(row),
                "visibility": parse_float(row.get("visibility"), default=1.0),
                "notes": str(row.get("notes", "") or ""),
            }
        )

    return selected_frame_ids, by_frame


def _load_predictions(
    *,
    tracks_csv: Path,
    selected_frames: List[int],
    include_zero_gid: bool,
) -> Dict[int, List[Dict[str, Any]]]:
    selected_set = set(int(x) for x in selected_frames)
    rows = load_csv(tracks_csv)
    by_frame: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        fr = parse_int(row.get("frame_idx"), default=-1)
        if fr not in selected_set:
            continue
        gid = parse_int(row.get("global_id"), default=0)
        if not include_zero_gid and gid <= 0:
            continue
        by_frame[int(fr)].append(
            {
                "frame_idx": int(fr),
                "global_id": int(gid),
                "box": _to_box(row),
                "ts_sec": parse_float(row.get("ts_sec"), default=0.0),
            }
        )
    return by_frame


def _compute_mt_ml(gt_total: Dict[int, int], gt_matched: Dict[int, int]) -> Tuple[int, int]:
    mt = 0
    ml = 0
    for pid, total in gt_total.items():
        if int(total) <= 0:
            continue
        hit = int(gt_matched.get(int(pid), 0))
        ratio = float(hit / max(1, int(total)))
        if ratio >= 0.80:
            mt += 1
        if ratio <= 0.20:
            ml += 1
    return int(mt), int(ml)


def _validate_gt_person_id_or_raise(frame_idx: int, gt_row: Dict[str, Any]) -> int:
    """Enforce valid positive numeric person_id for ignore=0 GT rows."""
    ignore = int(gt_row.get("ignore", 0))
    if ignore == 1:
        return -1

    raw = str(gt_row.get("row_data", {}).get("person_id", "")).strip()
    if raw == "":
        raise ValueError(
            f"Invalid GT person_id at frame_idx={frame_idx}: "
            f"ignore=0 requires non-empty positive numeric person_id; row={gt_row.get('row_data')}"
        )
    try:
        pid = int(round(float(raw)))
    except Exception:
        raise ValueError(
            f"Invalid GT person_id at frame_idx={frame_idx}: non-numeric person_id={raw!r}; "
            f"row={gt_row.get('row_data')}"
        )
    if pid <= 0:
        raise ValueError(
            f"Invalid GT person_id at frame_idx={frame_idx}: person_id must be > 0 for ignore=0, got {pid}; "
            f"row={gt_row.get('row_data')}"
        )
    return int(pid)


def _suppress_predictions_on_ignore_regions(
    *,
    predictions: List[Dict[str, Any]],
    ignore_gt_rows: List[Dict[str, Any]],
    ignore_iou_threshold: float,
) -> Tuple[List[Dict[str, Any]], int]:
    if not predictions or not ignore_gt_rows:
        return predictions, 0

    kept: List[Dict[str, Any]] = []
    suppressed = 0
    ignore_boxes = [g["box"] for g in ignore_gt_rows]
    for p in predictions:
        p_box = p["box"]
        overlaps_ignore = any(
            float(iou_xyxy(p_box, g_box)) >= float(ignore_iou_threshold) for g_box in ignore_boxes
        )
        if overlaps_ignore:
            suppressed += 1
            continue
        kept.append(p)
    return kept, int(suppressed)


def _motmetrics_id_scores_if_available(
    *,
    selected_frames: List[int],
    gt_by_frame: Dict[int, List[Dict[str, Any]]],
    pred_by_frame: Dict[int, List[Dict[str, Any]]],
    iou_threshold: float,
) -> Tuple[Optional[float], Optional[float], Optional[float], str]:
    try:
        import motmetrics as mm  # type: ignore
    except Exception as e:
        return None, None, None, f"motmetrics_unavailable:{e!r}"

    acc = mm.MOTAccumulator(auto_id=True)

    for fr in selected_frames:
        gt_rows = [r for r in gt_by_frame.get(int(fr), []) if int(r.get("ignore", 0)) == 0]
        pr_rows = pred_by_frame.get(int(fr), [])
        gt_ids: List[int] = []
        for r in gt_rows:
            gt_ids.append(_validate_gt_person_id_or_raise(int(fr), r))
        pr_ids = [int(r["global_id"]) for r in pr_rows]

        if not gt_ids and not pr_ids:
            acc.update([], [], np.empty((0, 0)))
            continue

        dist = np.full((len(gt_rows), len(pr_rows)), np.nan, dtype=np.float64)
        for gi, g in enumerate(gt_rows):
            for pi, p in enumerate(pr_rows):
                iou_v = iou_xyxy(g["box"], p["box"])
                if iou_v >= float(iou_threshold):
                    dist[gi, pi] = 1.0 - float(iou_v)

        acc.update(gt_ids, pr_ids, dist)

    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=["idp", "idr", "idf1"],
        name="dense_cam1",
    )

    idp = float(summary.loc["dense_cam1", "idp"]) if "idp" in summary.columns else None
    idr = float(summary.loc["dense_cam1", "idr"]) if "idr" in summary.columns else None
    idf1 = float(summary.loc["dense_cam1", "idf1"]) if "idf1" in summary.columns else None
    return idp, idr, idf1, "motmetrics"


def evaluate_dense_mot(
    *,
    gt_csv: Path,
    tracks_csv: Path,
    selected_frames_csv: Path,
    out_json: Path,
    out_txt: Path,
    iou_threshold: float = 0.50,
    ignore_iou_threshold: float = 0.50,
    include_zero_gid: bool = False,
) -> Dict[str, Any]:
    selected_frames, gt_by_frame = _load_gt_dense(gt_csv=gt_csv, selected_frames=selected_frames_csv)
    pred_by_frame = _load_predictions(
        tracks_csv=tracks_csv,
        selected_frames=selected_frames,
        include_zero_gid=include_zero_gid,
    )

    FP = 0
    FN = 0
    IDSW = 0
    matched_boxes = 0
    matched_iou_sum = 0.0
    gt_boxes = 0
    pred_boxes = 0
    ignored_predictions_suppressed = 0

    gt_last_pred: Dict[int, int] = {}
    gt_total: Dict[int, int] = defaultdict(int)
    gt_matched: Dict[int, int] = defaultdict(int)
    pred_by_frame_eval: Dict[int, List[Dict[str, Any]]] = {}

    for fr in selected_frames:
        gt_rows_all = gt_by_frame.get(int(fr), [])
        gt_rows = [r for r in gt_rows_all if int(r.get("ignore", 0)) == 0]
        ignore_rows = [r for r in gt_rows_all if int(r.get("ignore", 0)) == 1]
        pr_rows_raw = pred_by_frame.get(int(fr), [])
        pr_rows, suppressed = _suppress_predictions_on_ignore_regions(
            predictions=pr_rows_raw,
            ignore_gt_rows=ignore_rows,
            ignore_iou_threshold=ignore_iou_threshold,
        )
        ignored_predictions_suppressed += int(suppressed)
        pred_by_frame_eval[int(fr)] = pr_rows

        valid_gt_rows: List[Dict[str, Any]] = []
        for g in gt_rows:
            pid = _validate_gt_person_id_or_raise(int(fr), g)
            g["person_id"] = int(pid)
            valid_gt_rows.append(g)
        gt_rows = valid_gt_rows

        gt_boxes += int(len(gt_rows))
        pred_boxes += int(len(pr_rows))

        for g in gt_rows:
            gt_total[int(g["person_id"])] += 1

        gt_boxes_frame = [g["box"] for g in gt_rows]
        pr_boxes_frame = [p["box"] for p in pr_rows]

        matches = one_to_one_match(gt_boxes_frame, pr_boxes_frame, iou_threshold=iou_threshold)

        matched_gt_idx = set()
        matched_pr_idx = set()
        for gi, pi, iou_v in matches:
            matched_gt_idx.add(int(gi))
            matched_pr_idx.add(int(pi))
            matched_boxes += 1
            matched_iou_sum += float(iou_v)

            gt_pid = int(gt_rows[gi]["person_id"])
            pr_gid = int(pr_rows[pi]["global_id"])
            gt_matched[gt_pid] += 1

            if gt_pid in gt_last_pred and int(gt_last_pred[gt_pid]) != int(pr_gid):
                IDSW += 1
            gt_last_pred[gt_pid] = int(pr_gid)

        FN += int(len(gt_rows) - len(matched_gt_idx))
        FP += int(len(pr_rows) - len(matched_pr_idx))

    mota = None
    if gt_boxes > 0:
        mota = 1.0 - float((FP + FN + IDSW) / float(gt_boxes))

    det_precision = float(matched_boxes / pred_boxes) if pred_boxes > 0 else 0.0
    det_recall = float(matched_boxes / gt_boxes) if gt_boxes > 0 else 0.0
    mean_iou = float(matched_iou_sum / matched_boxes) if matched_boxes > 0 else 0.0
    mostly_tracked, mostly_lost = _compute_mt_ml(gt_total, gt_matched)

    idp, idr, idf1, id_source = _motmetrics_id_scores_if_available(
        selected_frames=selected_frames,
        gt_by_frame=gt_by_frame,
        pred_by_frame=pred_by_frame_eval,
        iou_threshold=iou_threshold,
    )

    notes = []
    if id_source != "motmetrics":
        notes.append(
            "IDP/IDR/IDF1 not computed from a trusted library because motmetrics is unavailable."
        )
    notes.append(
        "HOTA not computed; requires TrackEval or an equivalent validated implementation."
    )

    result: Dict[str, Any] = {
        "tracks_csv": str(tracks_csv),
        "gt_csv": str(gt_csv),
        "selected_frames_csv": str(selected_frames_csv),
        "iou_threshold": float(iou_threshold),
        "ignore_iou_threshold": float(ignore_iou_threshold),
        "include_zero_gid": bool(include_zero_gid),
        "frames_evaluated": int(len(selected_frames)),
        "gt_boxes": int(gt_boxes),
        "pred_boxes": int(pred_boxes),
        "ignored_predictions_suppressed": int(ignored_predictions_suppressed),
        "matched_boxes": int(matched_boxes),
        "FP": int(FP),
        "FN": int(FN),
        "IDSW": int(IDSW),
        "MOTA": mota,
        "IDP": idp,
        "IDR": idr,
        "IDF1": idf1,
        "HOTA": None,
        "mostly_tracked": int(mostly_tracked),
        "mostly_lost": int(mostly_lost),
        "detection_precision": float(det_precision),
        "detection_recall": float(det_recall),
        "mean_iou": float(mean_iou),
        "id_metric_source": str(id_source),
        "notes": notes,
    }

    write_json(out_json, result)

    txt = [
        "Dense CAM1 MOT Evaluation",
        f"tracks_csv={tracks_csv}",
        f"gt_csv={gt_csv}",
        f"frames_evaluated={result['frames_evaluated']}",
        f"gt_boxes={gt_boxes}",
        f"pred_boxes={pred_boxes}",
        f"ignored_predictions_suppressed={ignored_predictions_suppressed}",
        f"ignore_iou_threshold={float(ignore_iou_threshold):.3f}",
        f"matched_boxes={matched_boxes}",
        f"FP={FP}",
        f"FN={FN}",
        f"IDSW={IDSW}",
        f"MOTA={mota}",
        f"IDP={idp}",
        f"IDR={idr}",
        f"IDF1={idf1}",
        "HOTA=None (not computed)",
        f"mostly_tracked={mostly_tracked}",
        f"mostly_lost={mostly_lost}",
        f"detection_precision={det_precision:.6f}",
        f"detection_recall={det_recall:.6f}",
        f"mean_iou={mean_iou:.6f}",
        "notes:",
    ]
    for n in notes:
        txt.append(f"- {n}")
    write_text(out_txt, "\n".join(txt) + "\n")

    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate dense CAM1 MOT metrics on selected frames")
    ap.add_argument("--gt_csv", type=Path, default=DENSE_GT_CSV)
    ap.add_argument("--tracks_csv", type=Path, required=True)
    ap.add_argument("--selected_frames", type=Path, default=SELECTED_FRAMES_CSV)
    ap.add_argument("--out_json", type=Path, required=True)
    ap.add_argument("--out_txt", type=Path, required=True)
    ap.add_argument("--iou_threshold", type=float, default=0.50)
    ap.add_argument("--ignore_iou_threshold", type=float, default=0.50)
    ap.add_argument("--include_zero_gid", action="store_true", default=False)
    args = ap.parse_args()

    result = evaluate_dense_mot(
        gt_csv=args.gt_csv,
        tracks_csv=args.tracks_csv,
        selected_frames_csv=args.selected_frames,
        out_json=args.out_json,
        out_txt=args.out_txt,
        iou_threshold=args.iou_threshold,
        ignore_iou_threshold=args.ignore_iou_threshold,
        include_zero_gid=bool(args.include_zero_gid),
    )

    print(f"[OK] Dense MOT metrics JSON -> {args.out_json}")
    print(f"[OK] Dense MOT metrics TXT  -> {args.out_txt}")
    print(
        f"[INFO] frames={result['frames_evaluated']} gt_boxes={result['gt_boxes']} "
        f"pred_boxes={result['pred_boxes']} matched={result['matched_boxes']} "
        f"MOTA={result['MOTA']} IDF1={result['IDF1']}"
    )


if __name__ == "__main__":
    main()
