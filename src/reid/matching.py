import numpy as np
from typing import Dict, Tuple, List

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))

def topk_matches(query_key: Tuple[str,int],
                 gallery: Dict[Tuple[str,int], np.ndarray],
                 k:int=5) -> List[Tuple[Tuple[str,int], float]]:
    q = gallery[query_key]
    sims = [(k2, cosine(q, v)) for k2, v in gallery.items() if k2 != query_key]
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:k]
