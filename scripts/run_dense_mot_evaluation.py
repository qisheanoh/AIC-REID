from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from convert_dense_gt_to_mot import convert_dense_gt_to_mot
from dense_mot_common import (
    ANNOTATION_GUIDE_MD,
    CAM1_EXPECTED_FRAMES,
    CAM1_FRAME_HEIGHT,
    CAM1_FRAME_WIDTH,
    DENSE_GT_CSV,
    DENSE_GT_TEMPLATE_CSV,
    DENSE_README_MD,
    EXPERIMENT_ROOT,
    FRAMES_DIR,
    GT_VALIDATION_JSON,
    MOT_GT_TXT,
    MOT_SEQINFO_INI,
    RUN_ROOT,
    SELECTED_FRAMES_CSV,
    VIDEO_PATH,
    build_selected_rows,
    ensure_dense_dirs,
    extract_selected_frames,
    load_csv,
    parse_float,
    parse_int,
    write_dense_gt_template,
    write_selected_frames_csv,
    write_text,
)
from evaluate_dense_mot_metrics import evaluate_dense_mot
from validate_dense_mot_gt import validate_dense_gt


AUDIT_CSV = ROOT / "experiments" / "audit" / "cam1_manual_audit_sheet.csv"

BASELINE_TRACKS: List[Tuple[str, Path]] = [
    (
        "Full AIC-ReID (frozen)",
        ROOT / "runs" / "baseline_freeze_2026-04-28" / "retail-shop_CAM1_tracks.FROZEN.csv",
    ),
    (
        "ByteTrack-only baseline (reid_off)",
        ROOT / "runs" / "baselines" / "cam1_bytetrack_only_20260428_223244" / "retail-shop_CAM1_tracks.csv",
    ),
    (
        "BoT-SORT-style online baseline",
        ROOT / "runs" / "baselines" / "cam1_botsort_online_20260428_221824" / "retail-shop_CAM1_tracks.csv",
    ),
    (
        "AIC-ReID (no audit-guided offline repair)",
        ROOT / "runs" / "baselines" / "cam1_no_audit_offline_repair_20260428_220407" / "retail-shop_CAM1_tracks.csv",
    ),
]

SUMMARY_COLUMNS = [
    "method",
    "tracks_csv",
    "frames_evaluated",
    "gt_boxes",
    "pred_boxes",
    "ignored_predictions_suppressed",
    "ignore_iou_threshold",
    "matched_boxes",
    "FP",
    "FN",
    "IDSW",
    "MOTA",
    "IDP",
    "IDR",
    "IDF1",
    "HOTA",
    "mostly_tracked",
    "mostly_lost",
    "detection_precision",
    "detection_recall",
    "mean_iou",
    "notes",
]


def _write_annotation_guide() -> None:
    guide = """# Dense CAM1 Annotation Guide (for valid MOT metrics)

## Why dense annotation is required
Standard MOT metrics such as `MOTA`, `IDF1`, and `HOTA` assume dense frame-level ground truth for every evaluated frame: every visible person must be annotated with a bounding box and consistent identity where known.

The existing CAM1 audit sheet is sparse and KPI-oriented. It is suitable for deployment-oriented auditing, but **not** valid as dense MOT ground truth. Therefore, standard MOT metrics must only be computed from `dense_gt.csv` after dense manual labelling.

## Dense subset protocol
- Video: `data/raw/retail-shop/CAM1.mp4`
- Resolution: `2560 x 1944`
- Dense subset: `300` frames
- Sampling mode: `10 windows x 30 consecutive frames`
- Frame list source: `selected_frames.csv`

## Required CSV schema
File: `dense_gt.csv` (copy from `dense_gt_template.csv`)

Columns:
- `frame_idx`
- `person_id`
- `x1`
- `y1`
- `x2`
- `y2`
- `visibility`
- `ignore`
- `notes`

## Annotation rules
1. Label every visible person in every selected frame.
2. Use consistent `person_id` across contiguous frames and across windows when identity is visually known.
3. If identity across distant windows is uncertain, assign a new `person_id` and put `uncertain_cross_window` in `notes`.
4. Use `ignore=1` for non-person/ambiguous regions (reflection, poster, severe truncation, etc.) that should not count.
5. Visibility values:
- `1.0`: fully or mostly visible
- `0.5`: partially occluded but still identifiable
- `0.25`: heavily occluded
6. Bounding boxes use pixel coordinates in original CAM1 frame space (`2560x1944`).

## Bounding box quality
- Tight around the person silhouette/body extent.
- Avoid excessive background.
- Keep temporal consistency (avoid jittering box position/size unnecessarily).
- Ensure `x1 < x2` and `y1 < y2`.

## Identity consistency guidance
- Inside each 30-frame window, keep IDs strictly consistent.
- Across windows, reuse IDs only when identity is reasonably certain.
- If unsure, create new ID and mark `uncertain_cross_window`.
- Do not leave `person_id` empty for `ignore=0` rows.

## Valid vs invalid row examples
Valid:
```csv
frame_idx,person_id,x1,y1,x2,y2,visibility,ignore,notes
720,5,1020,410,1204,930,1.0,0,
721,5,1018,412,1201,931,0.5,0,partial_occlusion
722,,430,280,520,470,0.25,1,reflection_or_ambiguous
```

Invalid:
```csv
frame_idx,person_id,x1,y1,x2,y2,visibility,ignore,notes
720,,1020,410,1204,930,1.0,0,missing_person_id_for_ignore0
721,5,1201,412,1018,931,0.5,0,x1_greater_than_x2
722,9,-10,280,520,470,1.2,0,out_of_bounds_and_bad_visibility
```

## Validation before evaluation
Run:
```bash
python scripts/validate_dense_mot_gt.py --gt_csv experiments/dense_mot_cam1/dense_gt.csv
```

The report is written to `experiments/dense_mot_cam1/gt_validation_report.json`.
"""
    write_text(ANNOTATION_GUIDE_MD, guide)


def _write_dense_readme() -> None:
    readme = """# Dense MOT CAM1 Protocol

## Purpose
This folder defines a **valid dense subset protocol** for CAM1 so standard MOT metrics (`MOTA`, `IDF1`, `IDSW`, etc.) are computed only where methodologically valid.

## Selected frame protocol
- Window-based subset for temporal continuity.
- 10 windows x 30 consecutive frames = 300 frames total.
- Frame list: `selected_frames.csv`
- Extracted frames: `frames/`

## Annotation status
- Template: `dense_gt_template.csv`
- Manual annotation target: `dense_gt.csv`
- Guide: `ANNOTATION_GUIDE.md`

## Why dense GT is required
Sparse audit logs do not provide dense, frame-complete tracking supervision. Therefore sparse audit cannot be used to compute standard MOT metrics reliably.

## Commands
Prepare subset and templates:
```bash
python scripts/run_dense_mot_evaluation.py --prepare
```

Validate dense GT:
```bash
python scripts/validate_dense_mot_gt.py --gt_csv experiments/dense_mot_cam1/dense_gt.csv
```

Run evaluation pipeline:
```bash
python scripts/run_dense_mot_evaluation.py --evaluate
```

## Evaluation outputs
- `runs/dense_mot_cam1/dense_mot_summary.csv`
- `runs/dense_mot_cam1/dense_mot_summary.json`
- `runs/dense_mot_cam1/dense_mot_report.md`

## Metric separation policy
Dense MOT metrics and sparse audit deployment metrics are kept in **separate sections**. They must not be mixed into one score.
"""
    write_text(DENSE_README_MD, readme)


def prepare_dense_subset() -> Dict[str, Any]:
    ensure_dense_dirs()
    rows = build_selected_rows(video_path=VIDEO_PATH)

    # Validate frame cardinality for publication traceability.
    if len(rows) != 300:
        raise RuntimeError(f"Dense subset must be 300 frames, got {len(rows)}")
    frame_ids = [int(r.original_frame_idx) for r in rows]
    if min(frame_ids) < 0 or max(frame_ids) >= CAM1_EXPECTED_FRAMES:
        raise RuntimeError(
            f"Selected frame index out of expected CAM1 range [0, {CAM1_EXPECTED_FRAMES-1}]"
        )

    write_selected_frames_csv(rows, SELECTED_FRAMES_CSV)
    extract_stats = extract_selected_frames(video_path=VIDEO_PATH, rows=rows, overwrite=True)
    write_dense_gt_template(DENSE_GT_TEMPLATE_CSV)
    _write_annotation_guide()
    _write_dense_readme()

    return {
        "selected_frames_csv": str(SELECTED_FRAMES_CSV),
        "frames_dir": str(FRAMES_DIR),
        "dense_gt_template_csv": str(DENSE_GT_TEMPLATE_CSV),
        "annotation_guide": str(ANNOTATION_GUIDE_MD),
        "readme": str(DENSE_README_MD),
        "extract_stats": extract_stats,
    }


def _slug(s: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in s)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def _locate_reentry_stats(tracks_csv: Path) -> Optional[Path]:
    cands = [
        tracks_csv.parent / "reentry_debug" / "retail-shop_CAM1" / "reentry_stats.json",
        tracks_csv.parent.parent / "reentry_debug" / "retail-shop_CAM1" / "reentry_stats.json",
        tracks_csv.parent / "reentry_stats.json",
    ]
    for c in cands:
        if c.exists():
            return c
    return None


def _load_deployment_metrics(method: str, tracks_csv: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "method": method,
        "tracks_csv": str(tracks_csv),
        "pred_to_gt_purity": None,
        "positive_coverage": None,
        "same_frame_duplicate_positive_ids": None,
        "reentry_accepted": None,
        "reentry_rejected": None,
        "fragmentation": None,
        "notes": "",
    }
    if not tracks_csv.exists() or not AUDIT_CSV.exists():
        out["notes"] = "tracks_or_audit_missing"
        return out

    from src.reid.track_linker import evaluate_first_two_minute_audit_metrics

    audit = evaluate_first_two_minute_audit_metrics(
        tracks_csv_path=tracks_csv,
        audit_csv_path=AUDIT_CSV,
        iou_threshold=0.34,
        max_sec=150.0,
    )
    out["pred_to_gt_purity"] = audit.get("pred_to_gt_purity_macro")
    out["positive_coverage"] = audit.get("audit_row_coverage_positive_id")
    out["same_frame_duplicate_positive_ids"] = audit.get("same_frame_duplicate_positive_ids")
    out["fragmentation"] = audit.get("gt_people_fragmented_multi_pred_ids")

    reentry_path = _locate_reentry_stats(tracks_csv)
    if reentry_path is not None:
        try:
            stats = json.loads(reentry_path.read_text(encoding="utf-8"))
            accepted = parse_int(stats.get("accepted_reuses"), default=0)
            rejected = (
                parse_int(stats.get("rejected_weak"), default=0)
                + parse_int(stats.get("rejected_ambiguous"), default=0)
                + parse_int(stats.get("rejected_conflict"), default=0)
            )
            out["reentry_accepted"] = accepted
            out["reentry_rejected"] = rejected
            out["notes"] = f"reentry_stats={reentry_path}"
        except Exception as e:
            out["notes"] = f"reentry_stats_parse_failed:{e!r}"
    else:
        out["notes"] = "reentry_stats_not_found"

    return out


def _write_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in SUMMARY_COLUMNS})


def _try_make_graph(rows: List[Dict[str, Any]], out_path: Path) -> Tuple[bool, str]:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        return False, f"matplotlib_unavailable:{e!r}"

    valid = [r for r in rows if r.get("MOTA") is not None]
    if not valid:
        return False, "no_dense_metrics_rows"

    labels = [str(r.get("method", "")) for r in valid]
    mota = np.array([float(r.get("MOTA") or 0.0) for r in valid], dtype=np.float32)
    idsw = np.array([float(r.get("IDSW") or 0.0) for r in valid], dtype=np.float32)
    fp = np.array([float(r.get("FP") or 0.0) for r in valid], dtype=np.float32)
    fn = np.array([float(r.get("FN") or 0.0) for r in valid], dtype=np.float32)
    idf1_vals = [r.get("IDF1") for r in valid]
    has_idf1 = any(v is not None for v in idf1_vals)
    idf1 = np.array([float(v or 0.0) for v in idf1_vals], dtype=np.float32)

    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    axes[0, 0].bar(x, mota, color="#4C78A8")
    axes[0, 0].set_title("MOTA")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels, rotation=20, ha="right")

    if has_idf1:
        axes[0, 1].bar(x, idf1, color="#72B7B2")
        axes[0, 1].set_title("IDF1")
    else:
        axes[0, 1].text(0.5, 0.5, "IDF1 unavailable", ha="center", va="center")
        axes[0, 1].set_title("IDF1")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(labels, rotation=20, ha="right")

    axes[1, 0].bar(x, idsw, color="#E45756")
    axes[1, 0].set_title("ID Switches (IDSW)")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels, rotation=20, ha="right")

    axes[1, 1].bar(x, fp, color="#F58518", label="FP")
    axes[1, 1].bar(x, fn, bottom=fp, color="#54A24B", label="FN")
    axes[1, 1].set_title("FP/FN")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1, 1].legend(loc="upper right")

    fig.suptitle("Dense CAM1 MOT Metrics Comparison", fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return True, "ok"


def _format_pct(v: Any) -> str:
    if v is None:
        return "NA"
    return f"{100.0 * float(v):.2f}%"


def _format_num(v: Any, nd: int = 4) -> str:
    if v is None:
        return "NA"
    return f"{float(v):.{nd}f}"


def _build_report(
    *,
    dense_rows: List[Dict[str, Any]],
    deployment_rows: List[Dict[str, Any]],
    graph_path: Optional[Path],
    graph_note: str,
) -> str:
    lines: List[str] = []
    lines.append("# Dense CAM1 MOT Evaluation Report")
    lines.append("")
    lines.append(f"Generated at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")

    lines.append("## Section A: Dense MOT Metrics (valid only on dense_gt.csv)")
    lines.append(
        "| method | frames_evaluated | gt_boxes | pred_boxes | ignore_suppressed | ignore_iou_th | matched_boxes | FP | FN | IDSW | MOTA | IDP | IDR | IDF1 | HOTA | mostly_tracked | mostly_lost | det_precision | det_recall | mean_iou | notes |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in dense_rows:
        lines.append(
            "| {method} | {fe} | {gt} | {pb} | {isup} | {ith} | {mb} | {fp} | {fn} | {idsw} | {mota} | {idp} | {idr} | {idf1} | {hota} | {mt} | {ml} | {dp} | {dr} | {miou} | {notes} |".format(
                method=r.get("method", ""),
                fe=r.get("frames_evaluated", "NA"),
                gt=r.get("gt_boxes", "NA"),
                pb=r.get("pred_boxes", "NA"),
                isup=r.get("ignored_predictions_suppressed", "NA"),
                ith=_format_num(r.get("ignore_iou_threshold"), nd=3),
                mb=r.get("matched_boxes", "NA"),
                fp=r.get("FP", "NA"),
                fn=r.get("FN", "NA"),
                idsw=r.get("IDSW", "NA"),
                mota=_format_num(r.get("MOTA"), nd=6),
                idp=_format_num(r.get("IDP"), nd=6),
                idr=_format_num(r.get("IDR"), nd=6),
                idf1=_format_num(r.get("IDF1"), nd=6),
                hota="NA" if r.get("HOTA") is None else _format_num(r.get("HOTA"), nd=6),
                mt=r.get("mostly_tracked", "NA"),
                ml=r.get("mostly_lost", "NA"),
                dp=_format_num(r.get("detection_precision"), nd=6),
                dr=_format_num(r.get("detection_recall"), nd=6),
                miou=_format_num(r.get("mean_iou"), nd=6),
                notes=r.get("notes", ""),
            )
        )
    lines.append("")

    lines.append("## Section B: Deployment-Oriented Sparse Audit Metrics (separate from dense MOT)")
    lines.append(
        "| method | pred_to_gt_purity | positive_coverage | same_frame_duplicate_positive_ids | reentry_accepted | reentry_rejected | fragmentation | notes |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for r in deployment_rows:
        lines.append(
            "| {method} | {pur} | {cov} | {dup} | {ra} | {rr} | {frag} | {notes} |".format(
                method=r.get("method", ""),
                pur=_format_num(r.get("pred_to_gt_purity"), nd=6),
                cov=_format_num(r.get("positive_coverage"), nd=6),
                dup=r.get("same_frame_duplicate_positive_ids", "NA"),
                ra=r.get("reentry_accepted", "NA"),
                rr=r.get("reentry_rejected", "NA"),
                frag=r.get("fragmentation", "NA"),
                notes=r.get("notes", ""),
            )
        )
    lines.append("")

    lines.append("## Notes")
    lines.append("- Standard MOT metrics are computed only from dense GT (`dense_gt.csv`).")
    lines.append("- Sparse audit metrics are retained for deployment/KPI interpretation only.")
    lines.append("- HOTA is reported as `null` unless a validated TrackEval-equivalent implementation is integrated.")
    if graph_path is not None:
        lines.append(f"- Dense metrics graph: `{graph_path}`")
    else:
        lines.append(f"- Dense metrics graph not created: {graph_note}")

    return "\n".join(lines) + "\n"


def run_evaluation_flow() -> Dict[str, str]:
    ensure_dense_dirs()
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    if not DENSE_GT_CSV.exists():
        prep = prepare_dense_subset()
        msg = (
            "Dense GT template created. Please manually annotate dense_gt.csv before computing MOT metrics."
        )
        print(msg)
        print(f"- selected_frames.csv: {prep['selected_frames_csv']}")
        print(f"- frames directory: {prep['frames_dir']}")
        print(f"- dense_gt_template.csv: {prep['dense_gt_template_csv']}")
        print(f"- annotation guide: {prep['annotation_guide']}")
        return {
            "selected_frames_csv": prep["selected_frames_csv"],
            "frames_dir": prep["frames_dir"],
            "dense_gt_template_csv": prep["dense_gt_template_csv"],
            "annotation_guide": prep["annotation_guide"],
            "next_step": "Annotate experiments/dense_mot_cam1/dense_gt.csv, then run: python scripts/run_dense_mot_evaluation.py --evaluate",
        }

    # Validate dense GT.
    val = validate_dense_gt(
        gt_csv=DENSE_GT_CSV,
        selected_frames_csv=SELECTED_FRAMES_CSV,
        out_json=GT_VALIDATION_JSON,
        strict_complete=True,
    )
    if not bool(val.get("valid", False)):
        raise RuntimeError(
            "Dense GT validation failed. Fix dense_gt.csv first. "
            f"See report: {GT_VALIDATION_JSON}"
        )

    convert_dense_gt_to_mot(
        gt_csv=DENSE_GT_CSV,
        selected_frames_csv=SELECTED_FRAMES_CSV,
        out_gt_txt=MOT_GT_TXT,
        out_seqinfo=MOT_SEQINFO_INI,
        include_ignore_rows=True,
    )

    metrics_dir = RUN_ROOT / "per_method_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    dense_rows: List[Dict[str, Any]] = []
    deployment_rows: List[Dict[str, Any]] = []

    for method, tracks_csv in BASELINE_TRACKS:
        row: Dict[str, Any] = {"method": method, "tracks_csv": str(tracks_csv)}

        if not tracks_csv.exists():
            for k in SUMMARY_COLUMNS:
                row.setdefault(k, None)
            row["method"] = method
            row["tracks_csv"] = str(tracks_csv)
            row["notes"] = "tracks_csv_missing"
            dense_rows.append(row)
            deployment_rows.append(_load_deployment_metrics(method, tracks_csv))
            continue

        slug = _slug(method)
        out_json = metrics_dir / f"{slug}.json"
        out_txt = metrics_dir / f"{slug}.txt"

        result = evaluate_dense_mot(
            gt_csv=DENSE_GT_CSV,
            tracks_csv=tracks_csv,
            selected_frames_csv=SELECTED_FRAMES_CSV,
            out_json=out_json,
            out_txt=out_txt,
            iou_threshold=0.50,
            ignore_iou_threshold=0.50,
            include_zero_gid=False,
        )

        row.update(
            {
                "frames_evaluated": result.get("frames_evaluated"),
                "gt_boxes": result.get("gt_boxes"),
                "pred_boxes": result.get("pred_boxes"),
                "ignored_predictions_suppressed": result.get("ignored_predictions_suppressed"),
                "ignore_iou_threshold": result.get("ignore_iou_threshold"),
                "matched_boxes": result.get("matched_boxes"),
                "FP": result.get("FP"),
                "FN": result.get("FN"),
                "IDSW": result.get("IDSW"),
                "MOTA": result.get("MOTA"),
                "IDP": result.get("IDP"),
                "IDR": result.get("IDR"),
                "IDF1": result.get("IDF1"),
                "HOTA": result.get("HOTA"),
                "mostly_tracked": result.get("mostly_tracked"),
                "mostly_lost": result.get("mostly_lost"),
                "detection_precision": result.get("detection_precision"),
                "detection_recall": result.get("detection_recall"),
                "mean_iou": result.get("mean_iou"),
                "notes": "; ".join(result.get("notes", [])) if isinstance(result.get("notes"), list) else str(result.get("notes", "")),
            }
        )

        dense_rows.append(row)
        deployment_rows.append(_load_deployment_metrics(method, tracks_csv))

    summary_csv = RUN_ROOT / "dense_mot_summary.csv"
    summary_json = RUN_ROOT / "dense_mot_summary.json"
    summary_md = RUN_ROOT / "dense_mot_report.md"

    _write_summary_csv(summary_csv, dense_rows)
    summary_json.write_text(json.dumps(dense_rows, indent=2, ensure_ascii=True), encoding="utf-8")

    graph_path = ROOT / "report_assets" / "graphs" / "dense_mot_cam1_metrics_comparison.png"
    graph_ok, graph_note = _try_make_graph(dense_rows, graph_path)

    report_text = _build_report(
        dense_rows=dense_rows,
        deployment_rows=deployment_rows,
        graph_path=graph_path if graph_ok else None,
        graph_note=graph_note,
    )
    write_text(summary_md, report_text)

    return {
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
        "graph": str(graph_path) if graph_ok else "",
        "graph_note": graph_note,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare and evaluate dense CAM1 MOT subset")
    ap.add_argument("--prepare", action="store_true", help="Prepare selected frames, extracted images, GT template, and guide.")
    ap.add_argument("--evaluate", action="store_true", help="Run dense MOT evaluation (requires dense_gt.csv).")
    args = ap.parse_args()

    if not args.prepare and not args.evaluate:
        # Default behavior: enforce evaluation flow gate.
        args.evaluate = True

    if args.prepare:
        prep = prepare_dense_subset()
        print("[OK] Dense subset prepared.")
        print(f"- selected_frames.csv: {prep['selected_frames_csv']}")
        print(f"- frames directory: {prep['frames_dir']}")
        print(f"- dense_gt_template.csv: {prep['dense_gt_template_csv']}")
        print(f"- annotation guide: {prep['annotation_guide']}")

    if args.evaluate:
        result = run_evaluation_flow()
        if "summary_csv" in result:
            print("[OK] Dense MOT evaluation complete.")
            print(f"- metrics summary CSV: {result['summary_csv']}")
            print(f"- metrics summary JSON: {result['summary_json']}")
            print(f"- metrics report MD: {result['summary_md']}")
            if result.get("graph"):
                print(f"- graph: {result['graph']}")
            else:
                print(f"- graph not created: {result.get('graph_note', 'unknown_reason')}")
        else:
            print("[INFO] Dense MOT metrics were not computed yet.")
            print(f"- selected_frames.csv: {result['selected_frames_csv']}")
            print(f"- frames directory: {result['frames_dir']}")
            print(f"- dense_gt_template.csv: {result['dense_gt_template_csv']}")
            print(f"- annotation guide: {result['annotation_guide']}")
            print(f"- next manual step: {result['next_step']}")


if __name__ == "__main__":
    main()
