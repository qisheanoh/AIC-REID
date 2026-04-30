# src/preprocessing/frame_quality.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class QualityConfig:
    # Fraction of near-white pixels (>= 245 / 255) that marks a frame bad
    white_ratio_threshold: float = 0.16
    # Mean brightness above which a flat frame is considered bad
    mean_gray_hard: float = 225.0
    # Mean brightness + low-texture combination threshold
    mean_gray_soft: float = 205.0
    std_gray_soft_cutoff: float = 45.0
    # Texture floor: frames with std_gray below this are considered flat/blank
    std_gray_floor: float = 8.0

    # TODO: add blur detection (Laplacian variance) threshold when needed
    # laplacian_threshold: float = 50.0

    # TODO: add per-camera brightness calibration offsets when needed
    # brightness_offset: float = 0.0


@dataclass
class FrameQuality:
    frame_idx: int
    mean_gray: float
    std_gray: float
    white_ratio: float
    is_bad: bool
    # Human-readable reason string when is_bad=True, empty otherwise
    reason: str = ""

    # TODO: add blur_score (Laplacian variance) field when laplacian check is implemented
    # blur_score: float = 0.0


def score_frame(frame: Optional[np.ndarray], frame_idx: int, cfg: Optional[QualityConfig] = None) -> FrameQuality:
    """
    Score a single decoded frame against quality thresholds.

    Single source of truth for frame-quality scoring.  scripts/run_demo.py::
    analyze_frame_quality delegates to this function so both the preprocessing
    pre-scan and the live tracker loop use identical thresholds.

    Returns a FrameQuality with is_bad=True and reason set if the frame fails
    any threshold.  Frames are never discarded — callers decide what to do with
    the flag.
    """
    if cfg is None:
        cfg = QualityConfig()

    if frame is None or frame.size == 0:
        return FrameQuality(
            frame_idx=frame_idx,
            mean_gray=255.0,
            std_gray=0.0,
            white_ratio=1.0,
            is_bad=True,
            reason="null_or_empty",
        )

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_gray = float(gray.mean())
    std_gray = float(gray.std())
    white_ratio = float((gray >= 245).mean())

    reason = ""
    if white_ratio >= cfg.white_ratio_threshold:
        reason = f"white_ratio={white_ratio:.3f}>={cfg.white_ratio_threshold}"
    elif mean_gray >= cfg.mean_gray_hard:
        reason = f"mean_gray={mean_gray:.1f}>={cfg.mean_gray_hard}"
    elif mean_gray >= cfg.mean_gray_soft and std_gray <= cfg.std_gray_soft_cutoff:
        reason = f"mean_gray={mean_gray:.1f}>={cfg.mean_gray_soft} and std_gray={std_gray:.1f}<={cfg.std_gray_soft_cutoff}"
    elif std_gray <= cfg.std_gray_floor:
        reason = f"std_gray={std_gray:.1f}<={cfg.std_gray_floor}"

    return FrameQuality(
        frame_idx=frame_idx,
        mean_gray=mean_gray,
        std_gray=std_gray,
        white_ratio=white_ratio,
        is_bad=bool(reason),
        reason=reason,
    )


def scan_video_quality(
    video_path: Path,
    cfg: Optional[QualityConfig] = None,
    max_frames: Optional[int] = None,
) -> List[FrameQuality]:
    """
    Decode every frame of a video and return a FrameQuality record for each.

    This is an offline pre-scan — it does not run the tracker or detector.
    Use it to produce a quality report before the detection/tracking stage.

    max_frames: if set, stop after this many frames (useful for spot-checks).

    TODO: make this a generator for large videos when memory becomes a concern.
    TODO: parallelise with ThreadPoolExecutor when scan speed matters.
    """
    if cfg is None:
        cfg = QualityConfig()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    results: List[FrameQuality] = []
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            results.append(score_frame(frame, frame_idx, cfg))
            frame_idx += 1
            if max_frames is not None and frame_idx >= max_frames:
                break
    finally:
        cap.release()

    return results


def quality_summary(results: List[FrameQuality]) -> dict:
    """
    Aggregate a scan result into a compact report dict.

    Suitable for JSON serialisation and for display on the preprocessing UI page.
    """
    if not results:
        return {"total_frames": 0, "bad_frames": 0, "bad_pct": 0.0, "reasons": {}}

    bad = [r for r in results if r.is_bad]
    reason_counts: dict = {}
    for r in bad:
        key = r.reason.split("=")[0]  # group by condition name
        reason_counts[key] = reason_counts.get(key, 0) + 1

    return {
        "total_frames": len(results),
        "bad_frames": len(bad),
        "bad_pct": round(100.0 * len(bad) / len(results), 2),
        "mean_gray_avg": round(float(np.mean([r.mean_gray for r in results])), 2),
        "std_gray_avg": round(float(np.mean([r.std_gray for r in results])), 2),
        "reasons": reason_counts,
    }
