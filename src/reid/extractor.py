from __future__ import annotations

from typing import List, Optional
import numpy as np

FeatureExtractor = None
_torchreid_import_err: Exception | None = None
for _path in (
    "torchreid.utils",
    "torchreid.reid.utils",
    "torchreid.reid.utils.feature_extractor",
):
    try:
        mod = __import__(_path, fromlist=["FeatureExtractor"])
        FeatureExtractor = getattr(mod, "FeatureExtractor")
        break
    except Exception as e:
        _torchreid_import_err = e


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = x.astype(np.float32)
    if x.ndim == 1:
        n = np.linalg.norm(x) + eps
        return x / n
    n = np.linalg.norm(x, axis=1, keepdims=True) + eps
    return x / n


class ReidExtractor:
    """
    Wrapper around torchreid FeatureExtractor.

    Input: list of RGB crops as np.ndarray, each (H,W,3) uint8
    Output: np.ndarray (N, D) float32, L2-normalized
    """

    def __init__(
        self,
        model_name: str = "osnet_x1_0",
        device: str = "cpu",
        model_path: Optional[str] = None,
    ):
        if FeatureExtractor is None:
            raise ImportError(
                "torchreid FeatureExtractor import failed.\n"
                "Tried: torchreid.utils, torchreid.reid.utils, torchreid.reid.utils.feature_extractor\n"
                f"Last error: {_torchreid_import_err}\n"
                "Fix: ensure 'torchreid' is installed and compatible."
            )

        kwargs = {"model_name": model_name, "device": device}
        if model_path is not None:
            kwargs["model_path"] = model_path

        self.ext = FeatureExtractor(**kwargs)

    def __call__(self, rgb_crops: List[np.ndarray]) -> np.ndarray:
        if not rgb_crops:
            return np.zeros((0, 0), dtype=np.float32)

        valid = []
        for c in rgb_crops:
            if c is None or not isinstance(c, np.ndarray) or c.size == 0:
                continue
            if c.ndim != 3 or c.shape[2] != 3:
                continue
            h, w = c.shape[:2]
            if h < 8 or w < 4:
                continue
            valid.append(c)

        if not valid:
            return np.zeros((0, 0), dtype=np.float32)

        feats = self.ext(valid)

        if hasattr(feats, "detach"):
            feats = feats.detach().cpu().numpy()
        else:
            feats = np.asarray(feats)

        feats = feats.astype(np.float32)

        if feats.ndim == 1:
            feats = feats[None, :]

        return l2_normalize(feats)