# src/preprocessing/pipeline.py
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from src.preprocessing.video_ingestor import VideoMetadata, IngestConfig, extract_metadata
from src.preprocessing.frame_quality import QualityConfig, scan_video_quality, quality_summary


@dataclass
class PreprocessingReport:
    video: str                  # filename only (not full path)
    resolution: tuple           # (width, height)
    fps: float
    frame_count: int
    duration_s: float
    codec: str
    quality: dict               # output of quality_summary()
    report_path: Optional[str]  # absolute path of written JSON, or None if not written


def preprocess_video(
    video_path: Path,
    out_dir: Optional[Path] = None,
    ingest_cfg: Optional[IngestConfig] = None,
    quality_cfg: Optional[QualityConfig] = None,
    max_scan_frames: Optional[int] = None,
    write_report: bool = True,
) -> PreprocessingReport:
    """
    Run the full Stage 1 preprocessing pass for a single video.

    Steps
    -----
    1. extract_metadata()     — reads header properties, no frame decode
    2. scan_video_quality()   — decodes every frame and scores quality
    3. quality_summary()      — aggregates per-frame results into report dict
    4. write JSON report      — {stem}_preprocess_report.json in out_dir
                                (skipped when write_report=False)

    Parameters
    ----------
    video_path      : path to the source video file
    out_dir         : directory for the JSON report; defaults to video_path.parent
    ingest_cfg      : IngestConfig options (normalisation hooks); None = defaults
    quality_cfg     : QualityConfig thresholds; None = defaults matching run_demo.py
    max_scan_frames : scan only the first N frames (None = full video)
    write_report    : set False to suppress disk write (e.g. in unit tests)

    Returns
    -------
    PreprocessingReport dataclass — all fields safe to serialise via asdict()
    """
    video_path = Path(video_path)

    meta: VideoMetadata = extract_metadata(video_path)
    frames = scan_video_quality(video_path, quality_cfg, max_frames=max_scan_frames)
    summary = quality_summary(frames)

    report_path: Optional[str] = None
    if write_report:
        dest_dir = Path(out_dir) if out_dir is not None else video_path.parent
        dest_dir.mkdir(parents=True, exist_ok=True)
        json_path = dest_dir / f"{video_path.stem}_preprocess_report.json"

        payload = {
            "video": video_path.name,
            "resolution": list(meta.resolution),
            "fps": round(meta.fps, 4),
            "frame_count": meta.frame_count,
            "duration_s": round(meta.duration_s, 2),
            "codec": meta.codec,
            "quality": summary,
        }
        json_path.write_text(json.dumps(payload, indent=2))
        report_path = str(json_path)

    return PreprocessingReport(
        video=video_path.name,
        resolution=meta.resolution,
        fps=round(meta.fps, 4),
        frame_count=meta.frame_count,
        duration_s=round(meta.duration_s, 2),
        codec=meta.codec,
        quality=summary,
        report_path=report_path,
    )
