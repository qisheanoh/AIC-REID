from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from dense_mot_common import (
    CAM1_FPS,
    CAM1_FRAME_HEIGHT,
    CAM1_FRAME_WIDTH,
    DENSE_GT_CSV,
    DENSE_GT_TEMPLATE_CSV,
    MOT_GT_TXT,
    MOT_ROOT,
    MOT_SEQINFO_INI,
    SELECTED_FRAMES_CSV,
    load_csv,
    load_selected_frame_map,
    parse_float,
    parse_int,
)


def convert_dense_gt_to_mot(
    *,
    gt_csv: Path,
    selected_frames_csv: Path,
    out_gt_txt: Path,
    out_seqinfo: Path,
    include_ignore_rows: bool = True,
) -> Dict[str, int]:
    if not gt_csv.exists():
        raise FileNotFoundError(f"GT CSV not found: {gt_csv}")
    if not selected_frames_csv.exists():
        raise FileNotFoundError(f"Selected frames CSV not found: {selected_frames_csv}")

    rows = load_csv(gt_csv)
    frame_map = load_selected_frame_map(selected_frames_csv)

    out_gt_txt.parent.mkdir(parents=True, exist_ok=True)
    out_lines: List[str] = []

    kept = 0
    dropped = 0
    for row in rows:
        fr_orig = parse_int(row.get("frame_idx"), default=-1)
        if fr_orig not in frame_map:
            dropped += 1
            continue
        subset_idx = int(frame_map[fr_orig]["subset_frame_idx"])
        mot_frame = int(subset_idx + 1)  # MOTChallenge uses 1-based frame index.

        ignore = parse_int(row.get("ignore"), default=0)
        if ignore == 1 and not include_ignore_rows:
            dropped += 1
            continue

        pid_raw = str(row.get("person_id", "")).strip()
        if pid_raw == "":
            mot_id = -1
        else:
            mot_id = parse_int(pid_raw, default=-1)

        x1 = parse_float(row.get("x1"), default=0.0)
        y1 = parse_float(row.get("y1"), default=0.0)
        x2 = parse_float(row.get("x2"), default=0.0)
        y2 = parse_float(row.get("y2"), default=0.0)
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)

        conf = 0 if ignore == 1 else 1
        out_lines.append(
            f"{mot_frame},{mot_id},{x1:.3f},{y1:.3f},{w:.3f},{h:.3f},{conf},-1,-1,-1"
        )
        kept += 1

    out_gt_txt.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")

    seq_length = len(frame_map)
    seqinfo = "\n".join(
        [
            "[Sequence]",
            "name=CAM1_dense_subset",
            "imDir=../frames",
            f"frameRate={int(CAM1_FPS)}",
            f"seqLength={int(seq_length)}",
            f"imWidth={int(CAM1_FRAME_WIDTH)}",
            f"imHeight={int(CAM1_FRAME_HEIGHT)}",
            "imExt=.jpg",
            "",
            "# frame_index_mode=local_dense_subset_1_based",
            "# local_frame_1 maps to selected_frames.csv subset_frame_idx=0",
            "# selected_frames.csv retains original CAM1 frame indexes",
            "",
        ]
    )
    out_seqinfo.parent.mkdir(parents=True, exist_ok=True)
    out_seqinfo.write_text(seqinfo, encoding="utf-8")

    mapping_csv = out_seqinfo.parent / "frame_mapping.csv"
    mapping_lines = ["local_frame_1based,subset_frame_idx,original_frame_idx,window_id,image_path"]
    for fr_orig, info in sorted(frame_map.items(), key=lambda kv: int(kv[1]["subset_frame_idx"])):
        sf = int(info["subset_frame_idx"])
        mapping_lines.append(
            f"{sf + 1},{sf},{int(fr_orig)},{int(info.get('window_id', 0))},{info.get('image_path', '')}"
        )
    mapping_csv.write_text("\n".join(mapping_lines) + "\n", encoding="utf-8")

    return {
        "rows_input": int(len(rows)),
        "rows_written": int(kept),
        "rows_dropped": int(dropped),
        "seq_length": int(seq_length),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert dense CAM1 GT CSV into MOTChallenge gt.txt")
    ap.add_argument("--gt_csv", type=Path, default=DENSE_GT_CSV)
    ap.add_argument("--selected_frames", type=Path, default=SELECTED_FRAMES_CSV)
    ap.add_argument("--out_gt_txt", type=Path, default=MOT_GT_TXT)
    ap.add_argument("--out_seqinfo", type=Path, default=MOT_SEQINFO_INI)
    ap.add_argument(
        "--allow_template",
        action="store_true",
        help="If --gt_csv is missing, fallback to dense_gt_template.csv (header-only template usually yields empty gt.txt).",
    )
    ap.add_argument("--exclude_ignore_rows", action="store_true")
    args = ap.parse_args()

    gt_csv = args.gt_csv
    if not gt_csv.exists() and args.allow_template:
        gt_csv = DENSE_GT_TEMPLATE_CSV

    stats = convert_dense_gt_to_mot(
        gt_csv=gt_csv,
        selected_frames_csv=args.selected_frames,
        out_gt_txt=args.out_gt_txt,
        out_seqinfo=args.out_seqinfo,
        include_ignore_rows=not bool(args.exclude_ignore_rows),
    )

    print(f"[OK] MOT GT written: {args.out_gt_txt}")
    print(f"[OK] seqinfo written: {args.out_seqinfo}")
    print(f"[INFO] {stats}")


if __name__ == "__main__":
    main()
