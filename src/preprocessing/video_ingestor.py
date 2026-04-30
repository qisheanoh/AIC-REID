# src/preprocessing/video_ingestor.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import cv2


@dataclass
class VideoMetadata:
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float
    codec: str

    @property
    def resolution(self) -> Tuple[int, int]:
        return (self.width, self.height)


@dataclass
class IngestConfig:
    # Target FPS for normalisation; None = keep source FPS
    # TODO: implement frame-drop / duplication logic in iter_frames when set
    target_fps: Optional[float] = None

    # Target resolution (width, height); None = keep source resolution
    # TODO: implement resize in iter_frames when set
    target_resolution: Optional[Tuple[int, int]] = None

    # Whether to skip bad frames during iteration (True) or pass them through
    # with their is_bad flag intact (False, default — never discard evidence)
    skip_bad_frames: bool = False


def extract_metadata(video_path: Path) -> VideoMetadata:
    """Open the video, read header properties, close immediately."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_s = (frame_count / fps) if fps > 0 else 0.0

        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = (
            chr(fourcc_int & 0xFF)
            + chr((fourcc_int >> 8) & 0xFF)
            + chr((fourcc_int >> 16) & 0xFF)
            + chr((fourcc_int >> 24) & 0xFF)
        ).strip("\x00") or "unknown"
    finally:
        cap.release()

    return VideoMetadata(
        path=Path(video_path),
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_s=duration_s,
        codec=codec,
    )
