from __future__ import annotations

from pathlib import Path
import sys
import csv
import cv2
import numpy as np
from typing import Optional, Dict, Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.trackers.bot_sort import BOTSORT
from src.trackers.id_bank import GlobalIDBank
from src.preprocessing.frame_quality import score_frame as _score_frame


def analyze_frame_quality(frame: np.ndarray) -> Dict[str, Any]:
    """Wrapper — delegates to src.preprocessing.frame_quality.score_frame."""
    q = _score_frame(frame, frame_idx=0)
    return {
        "is_bad": q.is_bad,
        "mean_gray": q.mean_gray,
        "std_gray": q.std_gray,
        "white_ratio": q.white_ratio,
    }


def run_video_with_kpis(
    video_path: Path,
    out_tracks_csv: Path,
    fps_override: Optional[float] = None,
    tracker: Optional[BOTSORT] = None,
    forced_global_id: Optional[int] = None,
    out_video_path: Optional[Path] = None,
    draw: bool = True,
):
    video_path = Path(video_path)
    out_tracks_csv = Path(out_tracks_csv)
    out_tracks_csv.parent.mkdir(parents=True, exist_ok=True)

    if out_video_path is not None:
        out_video_path = Path(out_video_path)
        out_video_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or np.isnan(fps) or fps <= 1e-6:
        fps = 25.0
    if fps_override is not None:
        fps = float(fps_override)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if tracker is None:
        id_bank = GlobalIDBank(verbose=True)
        tracker = BOTSORT(id_bank=id_bank)

    writer = None
    if out_video_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (w, h))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot open VideoWriter for: {out_video_path}")
        print(f"[INFO] Saving annotated video -> {out_video_path}")

    bad_count = 0

    with open(out_tracks_csv, "w", newline="") as f:
        csvw = csv.writer(f)
        csvw.writerow([
            "frame_idx", "ts_sec", "global_id", "x1", "y1", "x2", "y2",
            "zone_id", "frame_is_bad", "white_ratio", "mean_gray", "std_gray"
        ])

        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            ts_sec = frame_idx / fps
            q = analyze_frame_quality(frame)
            frame_is_bad = bool(q["is_bad"])
            if frame_is_bad:
                bad_count += 1

            outputs = tracker.update(
                frame,
                frame_id=frame_idx,
                ts_sec=ts_sec,
                return_raw_tracks=False,
                frame_is_bad=frame_is_bad,
            )

            # Safety guard: keep at most one box per positive ID in a frame.
            # This prevents downstream identity pollution when tracker emits duplicate IDs.
            if outputs:
                dedup_pos = {}
                non_pos = []
                for item in outputs:
                    x1, y1, x2, y2, gid = item
                    gid = int(gid)
                    area = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
                    if gid <= 0:
                        non_pos.append(item)
                        continue
                    prev = dedup_pos.get(gid)
                    if prev is None:
                        dedup_pos[gid] = item
                    else:
                        px1, py1, px2, py2, _ = prev
                        parea = max(0.0, float(px2 - px1)) * max(0.0, float(py2 - py1))
                        if area > parea:
                            dedup_pos[gid] = item
                outputs = non_pos + [dedup_pos[k] for k in sorted(dedup_pos.keys())]

            if draw:
                if frame_is_bad:
                    cv2.putText(
                        frame,
                        f"BAD FRAME  white={q['white_ratio']:.2f}  mean={q['mean_gray']:.1f}",
                        (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

                for (x1, y1, x2, y2, gid) in outputs:
                    gid2 = int(forced_global_id) if forced_global_id is not None else int(gid)

                    csvw.writerow([
                        frame_idx,
                        f"{ts_sec:.6f}",
                        gid2,
                        x1, y1, x2, y2,
                        "",
                        int(frame_is_bad),
                        f"{q['white_ratio']:.6f}",
                        f"{q['mean_gray']:.6f}",
                        f"{q['std_gray']:.6f}",
                    ])

                    x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
                    if gid2 <= 0:
                        color = (120, 120, 120)
                    else:
                        color = (0, 165, 255) if frame_is_bad else (0, 255, 0)

                    cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), color, 2)
                    if gid2 > 0:
                        cv2.putText(
                            frame,
                            f"ID {gid2}",
                            (x1i, max(0, y1i - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2,
                            cv2.LINE_AA,
                        )
            else:
                for (x1, y1, x2, y2, gid) in outputs:
                    gid2 = int(forced_global_id) if forced_global_id is not None else int(gid)
                    csvw.writerow([
                        frame_idx,
                        f"{ts_sec:.6f}",
                        gid2,
                        x1, y1, x2, y2,
                        "",
                        int(frame_is_bad),
                        f"{q['white_ratio']:.6f}",
                        f"{q['mean_gray']:.6f}",
                        f"{q['std_gray']:.6f}",
                    ])

            if writer is not None:
                writer.write(frame)

            frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    print(f"[OK] Saved tracks CSV -> {out_tracks_csv}")
    if out_video_path is not None:
        print(f"[OK] Saved video -> {out_video_path}")
    print(f"[INFO] Bad frames detected: {bad_count}")
