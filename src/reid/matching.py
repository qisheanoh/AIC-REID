from __future__ import annotations

import numpy as np
from typing import Dict, Tuple, List


def _l2(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = v.astype(np.float32)
    n = np.linalg.norm(v) + eps
    return v / n


def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    # Normalize inside to avoid relying on upstream normalization
    a = _l2(a, eps=eps)
    b = _l2(b, eps=eps)
    return float(np.dot(a, b))


def topk_matches(
    query_key: Tuple[str, int],
    gallery: Dict[Tuple[str, int], np.ndarray],
    k: int = 5,
) -> List[Tuple[Tuple[str, int], float]]:
    if query_key not in gallery:
        raise KeyError(f"query_key {query_key} not found in gallery")

    if k <= 0:
        return []

    q = gallery[query_key]
    sims = [(k2, cosine(q, v)) for k2, v in gallery.items() if k2 != query_key]
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:k]
