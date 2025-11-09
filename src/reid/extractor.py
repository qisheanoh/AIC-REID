from typing import List
import numpy as np
import cv2
from torchreid.utils import FeatureExtractor

class ReidExtractor:
    def __init__(self, model_name: str = "osnet_x1_0", device: str = "cpu"):
        self.ext = FeatureExtractor(model_name=model_name, device=device)

    def __call__(self, rgb_crops: List[np.ndarray]) -> np.ndarray:
        if not rgb_crops:
            return np.zeros((0, 512), dtype=np.float32)
        feats = self.ext(rgb_crops)
        return feats.float().cpu().numpy()  # ✅ Fixed line


if __name__ == "__main__":
    img = np.zeros((256, 128, 3), dtype=np.uint8)
    extractor = ReidExtractor(model_name="osnet_x1_0", device="cpu")
    feats = extractor([img])
    print("Feature shape:", feats.shape)
