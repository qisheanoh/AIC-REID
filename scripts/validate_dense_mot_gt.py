from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from dense_mot_common import (
    CAM1_FRAME_HEIGHT,
    CAM1_FRAME_WIDTH,
    DENSE_GT_CSV,
    GT_VALIDATION_JSON,
    SELECTED_FRAMES_CSV,
    load_csv,
    parse_float,
    parse_int,
    write_json,
)


REQUIRED_COLUMNS = [
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


def validate_dense_gt(
    *,
    gt_csv: Path,
    selected_frames_csv: Path,
    out_json: Path,
    strict_complete: bool = False,
    allow_dup_note_token: str = "allow_duplicate_same_frame",
) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    if not gt_csv.exists():
        report = {
            "valid": False,
            "errors": [f"gt_csv_missing:{gt_csv}"],
            "warnings": [],
        }
        write_json(out_json, report)
        return report

    if not selected_frames_csv.exists():
        report = {
            "valid": False,
            "errors": [f"selected_frames_csv_missing:{selected_frames_csv}"],
            "warnings": [],
        }
        write_json(out_json, report)
        return report

    gt_rows = load_csv(gt_csv)
    sel_rows = load_csv(selected_frames_csv)

    if not sel_rows:
        errors.append("selected_frames_csv_empty")
        sel_frame_set = set()
    else:
        sel_frame_set = set()
        for r in sel_rows:
            sel_frame_set.add(parse_int(r.get("original_frame_idx"), default=-1))

    if gt_rows:
        first_cols = list(gt_rows[0].keys())
        missing_cols = [c for c in REQUIRED_COLUMNS if c not in first_cols]
        if missing_cols:
            errors.append(f"missing_required_columns:{missing_cols}")
    else:
        warnings.append("gt_csv_has_no_rows")
        # We still consider this structurally valid once columns are present.
        # Re-open to inspect header.
        import csv

        with gt_csv.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []
        missing_cols = [c for c in REQUIRED_COLUMNS if c not in header]
        if missing_cols:
            errors.append(f"missing_required_columns:{missing_cols}")

    frame_annotated: set[int] = set()
    nonignore_identity_set: set[int] = set()
    ignored_boxes = 0
    nonignored_boxes = 0
    duplicate_key_rows: Dict[tuple[int, int], List[int]] = defaultdict(list)

    for idx, row in enumerate(gt_rows, start=2):
        frame_idx = parse_int(row.get("frame_idx"), default=-1)
        person_raw = str(row.get("person_id", "")).strip()
        x1 = parse_float(row.get("x1"), default=float("nan"))
        y1 = parse_float(row.get("y1"), default=float("nan"))
        x2 = parse_float(row.get("x2"), default=float("nan"))
        y2 = parse_float(row.get("y2"), default=float("nan"))
        vis_raw = str(row.get("visibility", "")).strip()
        ignore = parse_int(row.get("ignore"), default=0)
        notes = str(row.get("notes", "") or "")

        frame_annotated.add(int(frame_idx))

        if frame_idx not in sel_frame_set:
            errors.append(f"row_{idx}:frame_idx_not_in_selected_frames:{frame_idx}")

        # Coordinate sanity.
        vals_ok = all(v == v for v in (x1, y1, x2, y2))  # NaN check
        if not vals_ok:
            errors.append(f"row_{idx}:invalid_coordinate_nan")
        else:
            if not (x1 < x2 and y1 < y2):
                errors.append(f"row_{idx}:invalid_box_order:x1={x1},y1={y1},x2={x2},y2={y2}")
            if x1 < 0 or x2 > CAM1_FRAME_WIDTH or y1 < 0 or y2 > CAM1_FRAME_HEIGHT:
                errors.append(
                    f"row_{idx}:box_out_of_bounds:x1={x1},y1={y1},x2={x2},y2={y2},"
                    f"bounds=0..{CAM1_FRAME_WIDTH},0..{CAM1_FRAME_HEIGHT}"
                )

        # Ignore semantics.
        if ignore not in (0, 1):
            errors.append(f"row_{idx}:ignore_must_be_0_or_1:got={ignore}")

        if ignore == 0:
            nonignored_boxes += 1
            if person_raw == "":
                errors.append(f"row_{idx}:person_id_empty_for_ignore0")
                person_id = -1
            else:
                try:
                    person_id = int(round(float(person_raw)))
                except Exception:
                    errors.append(f"row_{idx}:person_id_not_numeric:{person_raw}")
                    person_id = -1
                if person_id <= 0:
                    errors.append(f"row_{idx}:person_id_must_be_positive_for_ignore0:{person_id}")
            if person_id > 0:
                nonignore_identity_set.add(int(person_id))
                duplicate_key_rows[(int(frame_idx), int(person_id))].append(int(idx))
        else:
            ignored_boxes += 1

        # Visibility checks.
        try:
            vis = float(vis_raw)
        except Exception:
            errors.append(f"row_{idx}:visibility_not_numeric:{vis_raw}")
            vis = -1.0
        if vis not in (1.0, 0.5, 0.25):
            if not (0.0 <= vis <= 1.0):
                errors.append(f"row_{idx}:visibility_out_of_range:{vis}")

    # Duplicate same-frame same-person check.
    for (fr, pid), row_ids in duplicate_key_rows.items():
        if len(row_ids) <= 1:
            continue
        allow_dup = True
        for rid in row_ids:
            row = gt_rows[rid - 2]
            note = str(row.get("notes", "") or "")
            if allow_dup_note_token not in note:
                allow_dup = False
                break
        if not allow_dup:
            errors.append(
                f"duplicate_person_id_same_frame:frame={fr},person_id={pid},rows={row_ids},"
                f"allow_token={allow_dup_note_token}"
            )
        else:
            warnings.append(
                f"duplicate_person_id_same_frame_allowed:frame={fr},person_id={pid},rows={row_ids}"
            )

    missing_frames = sorted(int(f) for f in sel_frame_set if int(f) not in frame_annotated)
    if missing_frames:
        msg = f"missing_selected_frames_without_annotations:{len(missing_frames)}"
        if bool(strict_complete):
            errors.append(msg)
        else:
            warnings.append(msg)

    report: Dict[str, Any] = {
        "valid": len(errors) == 0,
        "gt_csv": str(gt_csv),
        "selected_frames_csv": str(selected_frames_csv),
        "strict_complete": bool(strict_complete),
        "rows_total": int(len(gt_rows)),
        "frames_selected_total": int(len(sel_frame_set)),
        "frames_annotated": int(len(frame_annotated)),
        "gt_boxes": int(nonignored_boxes),
        "ignored_boxes": int(ignored_boxes),
        "identities_count": int(len(nonignore_identity_set)),
        "missing_selected_frames_count": int(len(missing_frames)),
        "missing_selected_frames_preview": [int(f) for f in missing_frames[:20]],
        "missing_selected_frames": [int(f) for f in missing_frames],
        "errors": errors,
        "warnings": warnings,
        "value_counts": {
            "ignore": dict(Counter(parse_int(r.get("ignore"), default=0) for r in gt_rows)),
            "visibility": dict(Counter(str(r.get("visibility", "")) for r in gt_rows)),
        },
    }

    write_json(out_json, report)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate dense CAM1 MOT GT annotations")
    ap.add_argument("--gt_csv", type=Path, default=DENSE_GT_CSV)
    ap.add_argument("--selected_frames", type=Path, default=SELECTED_FRAMES_CSV)
    ap.add_argument("--out_json", type=Path, default=GT_VALIDATION_JSON)
    ap.add_argument(
        "--strict_complete",
        action="store_true",
        help="Require all selected frames to have at least one annotation row (hard fail if missing).",
    )
    args = ap.parse_args()

    report = validate_dense_gt(
        gt_csv=args.gt_csv,
        selected_frames_csv=args.selected_frames,
        out_json=args.out_json,
        strict_complete=bool(args.strict_complete),
    )

    print(f"[INFO] Validation report -> {args.out_json}")
    print(
        f"[INFO] valid={report['valid']} rows={report.get('rows_total', 0)} "
        f"frames_annotated={report.get('frames_annotated', 0)}/"
        f"{report.get('frames_selected_total', 0)} gt_boxes={report.get('gt_boxes', 0)} "
        f"ignored_boxes={report.get('ignored_boxes', 0)} identities={report.get('identities_count', 0)}"
    )
    if report.get("errors"):
        print("[ERROR] validation failed with errors:")
        for e in report["errors"]:
            print(f"- {e}")
        if bool(args.strict_complete) and int(report.get("missing_selected_frames_count", 0)) > 0:
            print(
                "[ERROR] strict_complete mode: selected frames missing annotations "
                f"(count={report.get('missing_selected_frames_count', 0)}, "
                f"preview={report.get('missing_selected_frames_preview', [])})"
            )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
