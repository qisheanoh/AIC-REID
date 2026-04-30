from __future__ import annotations

import numpy as np
from typing import Dict, Tuple, Optional


def l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = v.astype(np.float32)
    n = np.linalg.norm(v) + eps
    return v / n


class Gallery:
    """
    Stores an EMA feature per (camera_id, local_track_id) key.
    You can later finalize() to get a dict of normalized vectors.

    Key type: (camera_id: str, track_id: int)
    """

    def __init__(self, alpha: float = 0.9):
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0,1]")
        self.alpha = float(alpha)
        self.vecs: Dict[Tuple[str, int], Optional[np.ndarray]] = {}

    def update(self, key: Tuple[str, int], feat: Optional[np.ndarray]) -> None:
        if feat is None:
            return
        feat = l2_normalize(feat)

        v = self.vecs.get(key)
        if v is None:
            self.vecs[key] = feat
        else:
            blended = self.alpha * v + (1.0 - self.alpha) * feat
            self.vecs[key] = l2_normalize(blended)

    def finalize(self) -> Dict[Tuple[str, int], np.ndarray]:
        out: Dict[Tuple[str, int], np.ndarray] = {}
        for k, v in self.vecs.items():
            if v is None:
                continue
            out[k] = l2_normalize(v)
        return out
