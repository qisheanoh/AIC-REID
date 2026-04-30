# src/privacy/redact.py
from __future__ import annotations

from typing import Iterable, Tuple
import cv2
import numpy as np

Box = Tuple[float, float, float, float]


def _make_odd(k: int) -> int:
    k = int(k)
    if k <= 1:
        return 1
    return k if (k % 2 == 1) else (k + 1)


def blur_box(frame: np.ndarray, box: Box, ksize: int = 31) -> None:
    """
    In-place blur of a rectangular ROI on frame.

    box: (x1, y1, x2, y2) in pixel coords (float ok)
    ksize: Gaussian kernel size (will be forced odd and adapted to ROI size)
    """
    if frame is None or frame.size == 0:
        return

    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]

    # clamp to valid slicing range (end-exclusive)
    x1i = int(max(0, min(w, round(x1))))
    y1i = int(max(0, min(h, round(y1))))
    x2i = int(max(0, min(w, round(x2))))
    y2i = int(max(0, min(h, round(y2))))

    if x2i <= x1i or y2i <= y1i:
        return

    roi = frame[y1i:y2i, x1i:x2i]
    rh, rw = roi.shape[:2]
    if rh <= 1 or rw <= 1:
        return

    # ensure odd kernel and not bigger than ROI
    k = _make_odd(ksize)
    k = min(k, _make_odd(rw), _make_odd(rh))
    if k <= 1:
        return

    frame[y1i:y2i, x1i:x2i] = cv2.GaussianBlur(roi, (k, k), 0)


def blur_boxes(frame: np.ndarray, boxes: Iterable[Box], ksize: int = 31) -> np.ndarray:
    """
    Returns the same frame object (mutated in-place), convenient for chaining.
    """
    for b in boxes:
        blur_box(frame, b, ksize=ksize)
    return frame
