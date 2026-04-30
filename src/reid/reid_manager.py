from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class ReIDPolicy:
    min_conf_for_extract: float = 0.42
    min_area_ratio: float = 0.010
    border_margin: float = 0.015
    min_blur_var: float = 35.0
    min_quality_for_extract: float = 0.34
    far_y2_ratio: float = 0.42
    cautious_y2_ratio: float = 0.62


class ReIDManager:
    """
    Centralized online ReID policy:
    - quality gating (conf/area/blur/border)
    - region-aware mode selection (reject/tentative/strong)
    """

    def __init__(self, policy: ReIDPolicy):
        self.policy = policy

    def assess_detection(
        self,
        crop_bgr: np.ndarray,
        *,
        x1i: int,
        y1i: int,
        x2i: int,
        y2i: int,
        conf: float,
        frame_h: int,
        frame_w: int,
    ) -> tuple[float, int]:
        area_ratio = max(0.0, float((x2i - x1i) * (y2i - y1i))) / max(1.0, float(frame_h * frame_w))
        conf_q = np.clip(
            (float(conf) - self.policy.min_conf_for_extract)
            / max(1e-6, 1.0 - self.policy.min_conf_for_extract),
            0.0,
            1.0,
        )
        area_q = np.clip(area_ratio / max(1e-6, self.policy.min_area_ratio), 0.0, 1.0)
        border_ok = (
            x1i > int(self.policy.border_margin * frame_w)
            and y1i > int(self.policy.border_margin * frame_h)
            and x2i < int((1.0 - self.policy.border_margin) * frame_w)
            and y2i < int((1.0 - self.policy.border_margin) * frame_h)
        )
        border_q = 1.0 if border_ok else 0.0

        if crop_bgr is None or crop_bgr.size == 0:
            blur_q = 0.0
        else:
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            blur_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
            blur_q = np.clip(
                (blur_var - self.policy.min_blur_var) / max(1e-6, 2.0 * self.policy.min_blur_var),
                0.0,
                1.0,
            )

        quality = float(np.clip(0.36 * conf_q + 0.28 * area_q + 0.20 * blur_q + 0.16 * border_q, 0.0, 1.0))

        y2_ratio = float(y2i) / max(1.0, float(frame_h))
        if y2_ratio < self.policy.far_y2_ratio:
            mode = 0
        elif y2_ratio < self.policy.cautious_y2_ratio:
            mode = 1
        else:
            mode = 2

        if quality < self.policy.min_quality_for_extract:
            mode = 0

        return quality, int(mode)
