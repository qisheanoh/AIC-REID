import numpy as np
from typing import Dict, Tuple, Optional

class Gallery:
    def __init__(self, alpha: float = 0.9):
        self.alpha = alpha
        self.vecs: Dict[Tuple[str,int], Optional[np.ndarray]] = {}

    def update(self, key: Tuple[str,int], feat: np.ndarray):
        v = self.vecs.get(key)
        self.vecs[key] = feat if v is None else self.alpha*v + (1-self.alpha)*feat

    def finalize(self) -> Dict[Tuple[str,int], np.ndarray]:
        out = {}
        for k, v in self.vecs.items():
            if v is None: continue
            n = np.linalg.norm(v) + 1e-12
            out[k] = (v / n).astype(np.float32)
        return out
