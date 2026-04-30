from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .byte_tracker import BYTETracker
from .id_bank import GlobalIDBank, MatchCandidate
from src.reid.extractor import ReidExtractor
from src.reid.reid_manager import ReIDManager, ReIDPolicy


@dataclass
class IdentityProfileState:
    gid: int
    first_feature: Optional[np.ndarray] = None
    recent_feature: Optional[np.ndarray] = None
    stable_ema_feature: Optional[np.ndarray] = None
    topk_prototypes: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=24))
    global_osnet: Optional[np.ndarray] = None
    upper_body_feature: Optional[np.ndarray] = None
    lower_body_feature: Optional[np.ndarray] = None
    whole_body_feature: Optional[np.ndarray] = None
    upper_body_hist: Optional[np.ndarray] = None
    lower_body_hist: Optional[np.ndarray] = None
    upper_hue_sat_hist: Optional[np.ndarray] = None
    lower_hue_sat_hist: Optional[np.ndarray] = None
    upper_dominant_color: Optional[np.ndarray] = None
    lower_dominant_color: Optional[np.ndarray] = None
    body_shape_descriptor: Optional[np.ndarray] = None
    accessory_descriptor: Optional[np.ndarray] = None
    first_box: Optional[tuple[float, float, float, float]] = None
    last_box: Optional[tuple[float, float, float, float]] = None
    last_frame: int = -1
    velocity: tuple[float, float] = (0.0, 0.0)
    crop_quality_history: deque[float] = field(default_factory=lambda: deque(maxlen=32))
    trust_score: float = 1.0
    lost_age: int = 0
    frozen_until: int = -1
    recovery_until: int = -1
    last_block_reason: str = ""


def l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = x.astype(np.float32)
    if x.ndim == 1:
        return x / (np.linalg.norm(x) + eps)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def _safe_hist(region_bgr: np.ndarray, bins=(12, 8, 8)) -> np.ndarray:
    if region_bgr is None or region_bgr.size == 0:
        num_bins = bins[0] * bins[1] * bins[2]
        return np.zeros((num_bins,), dtype=np.float32)

    hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
    return hist


def attire_descriptor(crop_bgr: np.ndarray, bins=(12, 8, 8)) -> np.ndarray:
    """
    Stronger attire descriptor for unstable retail CCTV:
    - upper body
    - middle body
    - lower body
    - extra global histogram
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return np.zeros((bins[0] * bins[1] * bins[2] * 4,), dtype=np.float32)

    h, w = crop_bgr.shape[:2]
    if h < 12 or w < 6:
        num_bins = bins[0] * bins[1] * bins[2]
        return np.zeros((num_bins * 4,), dtype=np.float32)

    y1 = int(h * 0.00)
    y2 = int(h * 0.38)
    y3 = int(h * 0.72)

    upper = crop_bgr[y1:y2, :]
    middle = crop_bgr[y2:y3, :]
    lower = crop_bgr[y3:, :]
    whole = crop_bgr

    h_upper = _safe_hist(upper, bins=bins)
    h_middle = _safe_hist(middle, bins=bins)
    h_lower = _safe_hist(lower, bins=bins)
    h_whole = _safe_hist(whole, bins=bins)

    # Weight upper + lower more strongly than middle
    desc = np.concatenate([
        1.25 * h_upper,
        0.85 * h_middle,
        1.15 * h_lower,
        0.60 * h_whole,
    ], axis=0).astype(np.float32)

    return l2_normalize_np(desc)


def _hue_sat_histogram_region(region_bgr: np.ndarray, bins: int = 18) -> np.ndarray:
    """Compact HSV hue-saturation descriptor for a body region.

    This is a stronger identity cue than grey-mean statistics and complements
    the full HSV histogram used by ``_safe_hist``. We keep only saturated
    pixels (S>=40) so grey walls / shelves / background noise do not pollute
    the shirt/pants color signature. Returned vector is L2-normalized.
    """
    if region_bgr is None or region_bgr.size == 0:
        return np.zeros((bins,), dtype=np.float32)
    hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]
    # Drop pixels likely to be background / low-saturation strips.
    mask = (s >= 40) & (v >= 30) & (v <= 245)
    if not mask.any():
        return np.zeros((bins,), dtype=np.float32)
    hues = h[mask].astype(np.float32)
    hist, _ = np.histogram(hues, bins=bins, range=(0.0, 180.0))
    hist = hist.astype(np.float32)
    n = float(np.linalg.norm(hist))
    if n <= 1e-6:
        return np.zeros((bins,), dtype=np.float32)
    return hist / n


def dominant_color_descriptor(region_bgr: np.ndarray) -> np.ndarray:
    """Low-dimensional dominant-color fingerprint for a body region.

    Produces a length-6 vector: [H_mean, H_std, S_mean, S_std, V_mean, V_std]
    computed only over saturated pixels. This is a very cheap, complementary
    signal that catches "same shirt color, different person" vs "same
    OSNet embedding, different clothing" ambiguities.
    """
    out = np.zeros((6,), dtype=np.float32)
    if region_bgr is None or region_bgr.size == 0:
        return out
    hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    s = hsv[..., 1]
    v = hsv[..., 2]
    mask = (s >= 40) & (v >= 30) & (v <= 245)
    if not mask.any():
        return out
    h = hsv[..., 0][mask]
    s_m = s[mask]
    v_m = v[mask]
    out[0] = float(h.mean()) / 180.0
    out[1] = float(h.std()) / 90.0
    out[2] = float(s_m.mean()) / 255.0
    out[3] = float(s_m.std()) / 128.0
    out[4] = float(v_m.mean()) / 255.0
    out[5] = float(v_m.std()) / 128.0
    n = float(np.linalg.norm(out))
    return out / (n + 1e-12)


def body_shape_descriptor(box_xyxy: np.ndarray, frame_h: int, frame_w: int) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    frame_h = max(1, int(frame_h))
    frame_w = max(1, int(frame_w))
    area = w * h

    desc = np.array(
        [
            w / h,                                   # body aspect
            h / float(frame_h),                      # relative height
            np.sqrt(area) / np.sqrt(frame_h * frame_w),  # relative scale
            (w + h) / float(frame_h + frame_w),      # perimeter ratio
        ],
        dtype=np.float32,
    )
    return l2_normalize_np(desc)


def _resolve_det_weights_path(det_weights: str) -> str:
    """
    Prefer local model files under repo `models/` to avoid network downloads.
    """
    p = Path(det_weights)
    if p.is_absolute() and p.exists():
        return str(p)

    if p.exists():
        return str(p.resolve())

    repo_root = Path(__file__).resolve().parents[2]
    model_in_repo = (repo_root / "models" / p.name).resolve()
    if model_in_repo.exists():
        return str(model_in_repo)

    return det_weights


class BOTSORT:
    """
    YOLO detector + BYTETracker + GlobalIDBank

    This version is attire-aware:
    - OSNet appearance feature
    - strong attire histogram feature
    - fused embedding for better same-person recovery in low-quality video
    """

    def __init__(
        self,
        det_weights: str = "yolov8m.pt",
        device: str | None = None,
        reid: bool = True,
        reid_model_name: str = "osnet_x1_0",
        reid_weights_path: str | None = None,
        det_imgsz: int = 960,
        det_conf: float = 0.35,
        det_iou: float = 0.50,
        det_classes: tuple[int, ...] | None = (0,),

        track_thresh: float = 0.35,
        active_match_iou_thresh: float = 0.22,
        lost_match_iou_thresh: float = 0.10,
        match_feat_thresh: float = 0.46,
        strong_reid_thresh: float = 0.80,
        long_lost_reid_thresh: float = 0.86,
        alpha_active: float = 0.35,
        alpha_lost: float = 0.90,
        track_buffer: int = 140,
        motion_max_center_dist: float = 0.70,
        motion_max_gap: int = 24,
        overlap_iou_thresh: float = 0.30,
        min_height_ratio_for_update: float = 0.72,
        min_match_conf: float | None = 0.30,
        feature_history: int = 40,
        feature_update_min_sim: float = 0.66,
        confirm_hits: int = 4,
        bad_frame_hold: int = 12,

        id_bank: GlobalIDBank | None = None,
        min_confirmed_hits_for_gid: int = 4,
        min_height_ratio: float = 0.17,
        min_width_ratio: float = 0.06,
        reid_min_conf_for_extract: float = 0.42,
        reid_min_area_ratio: float = 0.010,
        reid_border_margin: float = 0.015,
        reid_min_blur_var: float = 35.0,
        reid_min_quality_for_bank: float = 0.56,
        reid_far_y2_ratio: float = 0.42,
        reid_cautious_y2_ratio: float = 0.62,
        reid_min_lock_hits: int = 4,
        reid_new_id_min_hits: int = 6,
        reid_new_id_min_quality: float = 0.56,

        # feature fusion weights
        F_OS: float = 1.00,
        F_ATTIRE: float = 0.55,
        F_SHAPE: float = 0.22,
        gid_reuse_thresh: float = 0.69,
        gid_reuse_with_spatial_thresh: float = 0.62,
        gid_spatial_max_age: int = 180,
        reentry_gallery_size: int = 20,
        reentry_min_samples: int = 3,
        reentry_max_age: int = 420,
        reentry_sim_thresh: float = 0.70,
        reentry_margin: float = 0.05,
        reentry_min_zone_compat: float = 0.32,
        lock_confirm_frames: int = 3,
        new_id_confirm_frames: int = 5,
        lock_score_thresh: float = 0.66,
        new_id_score_thresh: float = 0.68,
        gid_owner_reserve_frames: int = 240,
        gid_owner_override_score: float = 0.90,
        overlap_hold_iou_thresh: float = 0.18,
        overlap_hold_frames: int = 72,
        drift_release_hits: int = 6,
        drift_guard_min_mode: int = 2,
        drift_guard_min_quality: float = 0.52,
        short_memory_max_age: int = 120,
        short_memory_reuse_thresh: float = 0.70,
        short_memory_margin: float = 0.05,
        active_similar_block_thresh: float = 0.64,
        profile_topk: int = 4,
        profile_min_quality: float = 0.56,
        bank_update_min_margin: float = 0.08,
        # Conservative identity-assignment thresholds (crowded retail defaults).
        normal_reuse_threshold: float = 0.68,
        overlap_reuse_threshold: float = 0.74,
        long_gap_reid_threshold: float = 0.78,
        best_vs_second_margin: float = 0.05,
        overlap_best_vs_second_margin: float = 0.08,
        min_crop_quality_for_reuse: float = 0.52,
        min_spatial_consistency: float = 0.30,
        min_profile_update_similarity: float = 0.76,
        trust_decay_rate: float = 0.015,
        lost_track_max_age: int = 520,
        prototype_bank_size: int = 24,
        overlap_iou_threshold: float = 0.18,
        freeze_duration_after_overlap: int = 72,
        zombie_id_protection_threshold: float = 0.90,
        # Human/ghost sanity gates (conservative defaults for retail shelves/toilet regions).
        min_person_conf_for_identity: float = 0.32,
        min_human_aspect_ratio: float = 0.18,
        max_human_aspect_ratio: float = 1.08,
        min_human_area_ratio: float = 0.0032,
        ghost_right_zone_xmin: float = 0.74,
        ghost_zone_ymin: float = 0.32,
        ghost_low_sat_max: float = 0.15,
        ghost_low_texture_max: float = 13.5,
        debug_reid_decisions: bool = False,
        enable_pairwise_swap_correction: bool = True,
        # Optional lightweight pose model for part alignment; falls back to ratio split.
        pose_weights_path: str | None = None,
        pose_min_conf: float = 0.25,
        pose_infer_stride: int = 2,
        right_aisle_xmin: float = 0.68,
        right_aisle_ymin: float = 0.26,
        right_aisle_hold_extra_frames: int = 28,
        overlap_recovery_extra_frames: int = 22,
        zero_debug_path: str | None = None,
    ):
        self.reid = bool(reid)
        self.det_imgsz = int(det_imgsz)
        self.det_conf = float(det_conf)
        self.det_iou = float(det_iou)
        self.det_classes = list(det_classes) if det_classes is not None else None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.motion_max_gap = int(motion_max_gap)

        self.min_height_ratio = float(min_height_ratio)
        self.min_width_ratio = float(min_width_ratio)
        self.min_confirmed_hits_for_gid = int(min_confirmed_hits_for_gid)
        self.reid_min_conf_for_extract = float(reid_min_conf_for_extract)
        self.reid_min_area_ratio = float(reid_min_area_ratio)
        self.reid_border_margin = float(reid_border_margin)
        self.reid_min_blur_var = float(reid_min_blur_var)
        self.reid_min_quality_for_bank = float(reid_min_quality_for_bank)
        self.reid_min_quality_for_extract = max(0.30, self.reid_min_quality_for_bank - 0.22)
        self.reid_far_y2_ratio = float(reid_far_y2_ratio)
        self.reid_cautious_y2_ratio = float(reid_cautious_y2_ratio)
        self.reid_min_lock_hits = int(reid_min_lock_hits)
        self.reid_new_id_min_hits = int(reid_new_id_min_hits)
        self.reid_new_id_min_quality = float(reid_new_id_min_quality)

        self.F_OS = float(F_OS)
        self.F_ATTIRE = float(F_ATTIRE)
        self.F_SHAPE = float(F_SHAPE)
        self.gid_reuse_thresh = float(gid_reuse_thresh)
        self.gid_reuse_with_spatial_thresh = float(gid_reuse_with_spatial_thresh)
        self.gid_spatial_max_age = int(gid_spatial_max_age)
        self.reentry_gallery_size = int(reentry_gallery_size)
        self.reentry_min_samples = int(reentry_min_samples)
        self.reentry_max_age = int(reentry_max_age)
        self.reentry_sim_thresh = float(reentry_sim_thresh)
        self.reentry_margin = float(reentry_margin)
        self.reentry_min_zone_compat = float(reentry_min_zone_compat)
        self.lock_confirm_frames = int(lock_confirm_frames)
        self.new_id_confirm_frames = int(new_id_confirm_frames)
        self.lock_score_thresh = float(lock_score_thresh)
        self.new_id_score_thresh = float(new_id_score_thresh)
        self.gid_owner_reserve_frames = int(gid_owner_reserve_frames)
        self.gid_owner_override_score = float(gid_owner_override_score)
        self.overlap_hold_iou_thresh = float(overlap_hold_iou_thresh)
        self.overlap_hold_frames = int(overlap_hold_frames)
        self.drift_release_hits = int(drift_release_hits)
        self.drift_guard_min_mode = int(drift_guard_min_mode)
        self.drift_guard_min_quality = float(drift_guard_min_quality)
        self.short_memory_max_age = int(short_memory_max_age)
        self.short_memory_reuse_thresh = float(short_memory_reuse_thresh)
        self.short_memory_margin = float(short_memory_margin)
        self.active_similar_block_thresh = float(active_similar_block_thresh)
        self.profile_topk = max(1, int(profile_topk))
        self.profile_min_quality = float(profile_min_quality)
        self.bank_update_min_margin = float(bank_update_min_margin)
        self.normal_reuse_threshold = float(normal_reuse_threshold)
        self.overlap_reuse_threshold = float(overlap_reuse_threshold)
        self.long_gap_reid_threshold = float(long_gap_reid_threshold)
        self.best_vs_second_margin = float(best_vs_second_margin)
        self.overlap_best_vs_second_margin = float(overlap_best_vs_second_margin)
        self.min_crop_quality_for_reuse = float(min_crop_quality_for_reuse)
        self.min_spatial_consistency = float(min_spatial_consistency)
        self.min_profile_update_similarity = float(min_profile_update_similarity)
        self.trust_decay_rate = float(trust_decay_rate)
        self.lost_track_max_age = int(lost_track_max_age)
        self.prototype_bank_size = max(4, int(prototype_bank_size))
        self.overlap_iou_threshold = float(overlap_iou_threshold)
        self.freeze_duration_after_overlap = int(freeze_duration_after_overlap)
        self.zombie_id_protection_threshold = float(zombie_id_protection_threshold)
        self.min_person_conf_for_identity = float(min_person_conf_for_identity)
        self.min_human_aspect_ratio = float(min_human_aspect_ratio)
        self.max_human_aspect_ratio = float(max_human_aspect_ratio)
        self.min_human_area_ratio = float(min_human_area_ratio)
        self.ghost_right_zone_xmin = float(ghost_right_zone_xmin)
        self.ghost_zone_ymin = float(ghost_zone_ymin)
        self.ghost_low_sat_max = float(ghost_low_sat_max)
        self.ghost_low_texture_max = float(ghost_low_texture_max)
        self.debug_reid_decisions = bool(debug_reid_decisions)
        self.enable_pairwise_swap_correction = bool(enable_pairwise_swap_correction)
        self.pose_min_conf = float(pose_min_conf)
        self.pose_infer_stride = max(1, int(pose_infer_stride))
        self.right_aisle_xmin = float(right_aisle_xmin)
        self.right_aisle_ymin = float(right_aisle_ymin)
        self.right_aisle_hold_extra_frames = int(right_aisle_hold_extra_frames)
        self.overlap_recovery_extra_frames = int(overlap_recovery_extra_frames)
        self._zero_debug_path: str | None = zero_debug_path
        self._zero_debug_rows: list[dict] = []
        self._zero_debug_written_header: bool = False
        self.gid_history: dict[int, deque[tuple[int, tuple[float, float, float, float]]]] = {}
        self.reentry_gallery: dict[int, deque[np.ndarray]] = {}
        self.identity_profiles: dict[int, deque[dict[str, object]]] = {}
        self.identity_profile_state: dict[int, IdentityProfileState] = {}
        # Canonical identity memory bank keyed by stable global person IDs.
        # This intentionally mirrors `identity_profile_state` and keeps:
        # first/recent/EMA features, top-k prototypes, parts/colors/shape/accessory cues,
        # trust/lost age, motion, and overlap freeze state.
        self.canonical_identity_memory: dict[int, IdentityProfileState] = {}
        self.reentry_last_frame: dict[int, int] = {}
        self.reentry_last_box: dict[int, tuple[float, float, float, float]] = {}
        self.reentry_last_zone: dict[int, str] = {}
        self.exited_memory_ids: set[int] = set()
        self.gid_owner_track: dict[int, int] = {}
        self.gid_owner_last_frame: dict[int, int] = {}
        self._frame_h = 1
        self._frame_w = 1
        self._pose_frame_cache: dict[str, Any] = {}
        # osnet(512) + attire(12*8*8*4=3072) + shape(4)
        self.reid_feat_dim = 3588

        self.id_bank: GlobalIDBank | None = id_bank
        if self.id_bank is not None and hasattr(self.id_bank, "max_prototypes"):
            try:
                self.id_bank.max_prototypes = max(int(getattr(self.id_bank, "max_prototypes", 1)), int(self.prototype_bank_size))
            except Exception:
                pass
        self.reid_manager = ReIDManager(
            ReIDPolicy(
                min_conf_for_extract=self.reid_min_conf_for_extract,
                min_area_ratio=self.reid_min_area_ratio,
                border_margin=self.reid_border_margin,
                min_blur_var=self.reid_min_blur_var,
                min_quality_for_extract=self.reid_min_quality_for_extract,
                far_y2_ratio=self.reid_far_y2_ratio,
                cautious_y2_ratio=self.reid_cautious_y2_ratio,
            )
        )
        self.det_model = YOLO(_resolve_det_weights_path(det_weights))
        self.pose_model: YOLO | None = None
        if pose_weights_path:
            pose_path_resolved = _resolve_det_weights_path(pose_weights_path)
            try:
                self.pose_model = YOLO(pose_path_resolved)
            except Exception:
                self.pose_model = None

        self.tracker = BYTETracker(
            track_thresh=track_thresh,
            active_match_iou_thresh=active_match_iou_thresh,
            lost_match_iou_thresh=lost_match_iou_thresh,
            match_feat_thresh=match_feat_thresh,
            strong_reid_thresh=strong_reid_thresh,
            long_lost_reid_thresh=long_lost_reid_thresh,
            alpha_active=alpha_active,
            alpha_lost=alpha_lost,
            track_buffer=track_buffer,
            motion_max_center_dist=motion_max_center_dist,
            motion_max_gap=motion_max_gap,
            overlap_iou_thresh=overlap_iou_thresh,
            min_height_ratio_for_update=min_height_ratio_for_update,
            min_match_conf=min_match_conf,
            feature_history=feature_history,
            feature_update_min_sim=feature_update_min_sim,
            confirm_hits=confirm_hits,
            bad_frame_hold=bad_frame_hold,
            reid_update_min_quality=self.reid_min_quality_for_bank,
        )

        self.extractor: ReidExtractor | None = None
        if self.reid:
            self.extractor = ReidExtractor(
                model_name=reid_model_name,
                device=self.device,
                model_path=reid_weights_path,
            )

    def reset_for_new_video(self, reset_ids: bool = True, reset_global_ids: bool = True) -> None:
        self.tracker.reset(reset_ids=reset_ids)
        self.gid_history = {}
        self.reentry_gallery = {}
        self.identity_profiles = {}
        self.identity_profile_state = {}
        self.canonical_identity_memory = {}
        self.reentry_last_frame = {}
        self.reentry_last_box = {}
        self.reentry_last_zone = {}
        self.exited_memory_ids = set()
        self.gid_owner_track = {}
        self.gid_owner_last_frame = {}
        self._pose_frame_cache = {}
        if reset_global_ids and self.id_bank is not None:
            self.id_bank.reset()

    def lock_gid_owner(self, gid: int, track_id: int, frame_id: int) -> None:
        """Permanently pin a gid to a given track id at ``frame_id``.

        Used to seed canonical identity ownership from an external anchor
        (e.g. the CAM1 profile anchor) before the video runs. After a call
        to ``lock_gid_owner(3, 12, 0)`` the bank's owner map treats ``gid=3``
        as owned by track 12 with the reservation renewed at frame 0, so
        other tracks must clear the strict ``gid_owner_override_score`` gate
        (and all zombie/overlap gates) before they can claim that gid. The
        lock is refreshed automatically as long as the owner keeps being
        observed; it only expires when the owner is absent for longer than
        ``max(reentry_max_age, gid_owner_reserve_frames * 2)``.
        """
        g = int(gid)
        tid = int(track_id)
        if g <= 0 or tid <= 0:
            return
        self.gid_owner_track[g] = tid
        self.gid_owner_last_frame[g] = int(frame_id)

    def clear_gid_owner(self, gid: int) -> None:
        """Release a previous ``lock_gid_owner`` reservation."""
        g = int(gid)
        self.gid_owner_track.pop(g, None)
        self.gid_owner_last_frame.pop(g, None)

    def _spatial_score_for_gid(self, gid: int, frame_id: int, box_xyxy) -> float:
        hist = self.gid_history.get(int(gid))
        if not hist:
            return 0.0

        last_frame, last_box = hist[-1]
        age = max(0, int(frame_id) - int(last_frame))
        if age > self.gid_spatial_max_age:
            return 0.0

        x1, y1, x2, y2 = [float(v) for v in box_xyxy]
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        h = max(1.0, y2 - y1)

        lx1, ly1, lx2, ly2 = [float(v) for v in last_box]
        pcx = 0.5 * (lx1 + lx2)
        pcy = 0.5 * (ly1 + ly2)
        ph = max(1.0, ly2 - ly1)

        # Lightweight velocity extrapolation from last two observations.
        if len(hist) >= 2:
            prev_frame, prev_box = hist[-2]
            dt = max(1, int(last_frame) - int(prev_frame))
            px1, py1, px2, py2 = [float(v) for v in prev_box]
            prev_cx = 0.5 * (px1 + px2)
            prev_cy = 0.5 * (py1 + py2)
            vx = (pcx - prev_cx) / dt
            vy = (pcy - prev_cy) / dt
            pcx = pcx + vx * age
            pcy = pcy + vy * age

        dist = float(np.hypot(cx - pcx, cy - pcy))
        dist_norm = dist / max(ph, h, 1.0)
        return float(np.exp(-dist_norm))

    def _update_gid_history(self, gid: int, frame_id: int, box_xyxy) -> None:
        g = int(gid)
        if g not in self.gid_history:
            self.gid_history[g] = deque(maxlen=8)
        x1, y1, x2, y2 = [float(v) for v in box_xyxy]
        self.gid_history[g].append((int(frame_id), (x1, y1, x2, y2)))

    def _zone_of_box(self, box_xyxy) -> str:
        x1, y1, x2, y2 = [float(v) for v in box_xyxy]
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        w = max(1.0, float(self._frame_w))
        h = max(1.0, float(self._frame_h))

        if cx <= 0.18 * w:
            return "left"
        if cx >= 0.82 * w:
            return "right"
        if cy <= 0.18 * h:
            return "top"
        if cy >= 0.84 * h:
            return "bottom"
        return "center"

    def _is_right_aisle_box(self, box_xyxy) -> bool:
        x1, y1, x2, y2 = [float(v) for v in box_xyxy]
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        w = max(1.0, float(self._frame_w))
        h = max(1.0, float(self._frame_h))
        return bool(
            cx >= self.right_aisle_xmin * w
            and cy >= self.right_aisle_ymin * h
        )

    def _clip_box(self, box_xyxy) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = [float(v) for v in box_xyxy]
        w = max(1, int(self._frame_w))
        h = max(1, int(self._frame_h))
        x1i = max(0, min(w - 1, int(x1)))
        y1i = max(0, min(h - 1, int(y1)))
        x2i = max(0, min(w - 1, int(x2)))
        y2i = max(0, min(h - 1, int(y2)))
        return x1i, y1i, x2i, y2i

    def _compute_box_quality_metrics(self, crop_bgr: np.ndarray, box_xyxy) -> dict[str, float]:
        x1, y1, x2, y2 = [float(v) for v in box_xyxy]
        fw = max(1.0, float(self._frame_w))
        fh = max(1.0, float(self._frame_h))
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        area_ratio = float((bw * bh) / max(1.0, fw * fh))
        border_dist = min(x1, y1, fw - x2, fh - y2)
        border_trunc = 1.0 - float(np.clip(border_dist / max(4.0, 0.08 * min(fw, fh)), 0.0, 1.0))
        blur_var = 0.0
        sat_mean = 0.0
        edge_density = 0.0
        if crop_bgr is not None and crop_bgr.size > 0:
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            blur_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
            hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
            sat_mean = float(hsv[..., 1].mean() / 255.0)
            edges = cv2.Canny(gray, 60, 150)
            edge_density = float((edges > 0).mean()) if edges is not None and edges.size > 0 else 0.0
        return {
            "area_ratio": float(area_ratio),
            "border_trunc": float(np.clip(border_trunc, 0.0, 1.0)),
            "blur_var": float(blur_var),
            "sat_mean": float(np.clip(sat_mean, 0.0, 1.0)),
            "edge_density": float(np.clip(edge_density, 0.0, 1.0)),
            "box_h_ratio": float(bh / fh),
            "box_w_ratio": float(bw / fw),
        }

    def _run_pose_for_frame(self, frame: np.ndarray, frame_id: int) -> None:
        """
        Optional lightweight pose pass.
        If pose model is unavailable or inference fails, part extraction falls back to ratio crops.
        """
        if self.pose_model is None:
            self._pose_frame_cache = {"frame_id": int(frame_id), "poses": []}
            return
        if self._pose_frame_cache.get("frame_id", -1) == int(frame_id):
            return
        if int(frame_id) % int(self.pose_infer_stride) != 0:
            # Reuse previous pose snapshot for nearby frames to reduce overhead.
            return
        poses: list[dict[str, Any]] = []
        try:
            res = self.pose_model.predict(
                source=frame,
                imgsz=max(320, min(640, int(self.det_imgsz))),
                conf=max(0.05, self.pose_min_conf * 0.60),
                device=self.device,
                verbose=False,
            )[0]
            kp = getattr(res, "keypoints", None)
            bx = getattr(res, "boxes", None)
            if kp is not None and getattr(kp, "xy", None) is not None and bx is not None and getattr(bx, "xyxy", None) is not None:
                kxy = kp.xy.detach().cpu().numpy()
                kcf = kp.conf.detach().cpu().numpy() if getattr(kp, "conf", None) is not None else None
                bxyxy = bx.xyxy.detach().cpu().numpy()
                n = min(len(kxy), len(bxyxy))
                for i in range(n):
                    conf = kcf[i] if kcf is not None and i < len(kcf) else None
                    if conf is not None and float(np.nanmean(conf)) < self.pose_min_conf:
                        continue
                    poses.append(
                        {
                            "box": np.array(bxyxy[i], dtype=np.float32),
                            "kxy": np.array(kxy[i], dtype=np.float32),
                            "kconf": np.array(conf, dtype=np.float32) if conf is not None else None,
                        }
                    )
        except Exception:
            poses = []
        self._pose_frame_cache = {"frame_id": int(frame_id), "poses": poses}

    def _pose_for_box(self, box_xyxy) -> Optional[dict[str, Any]]:
        poses = self._pose_frame_cache.get("poses", [])
        if not poses:
            return None
        b = np.array(box_xyxy, dtype=np.float32)
        best = None
        best_iou = 0.0
        for p in poses:
            pb = np.array(p["box"], dtype=np.float32)
            xx1 = max(float(b[0]), float(pb[0]))
            yy1 = max(float(b[1]), float(pb[1]))
            xx2 = min(float(b[2]), float(pb[2]))
            yy2 = min(float(b[3]), float(pb[3]))
            iw = max(0.0, xx2 - xx1)
            ih = max(0.0, yy2 - yy1)
            inter = iw * ih
            a = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
            pa = max(0.0, float(pb[2] - pb[0])) * max(0.0, float(pb[3] - pb[1]))
            iou = inter / (a + pa - inter + 1e-6)
            if iou > best_iou:
                best_iou = float(iou)
                best = p
        if best is None or best_iou < 0.20:
            return None
        return best

    def _pose_aligned_regions(
        self,
        crop_bgr: np.ndarray,
        box_xyxy,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, bool]:
        h, w = crop_bgr.shape[:2]
        if h <= 10 or w <= 6:
            return crop_bgr, crop_bgr, crop_bgr, crop_bgr, 0.0, False
        pose = self._pose_for_box(box_xyxy)
        if pose is None:
            y_b = int(0.42 * h)
            y_c = int(0.75 * h)
            upper = crop_bgr[:y_b, :]
            lower = crop_bgr[y_c:, :]
            side = crop_bgr[int(0.20 * h):int(0.85 * h), int(0.70 * w):]
            return upper, lower, crop_bgr, side, 0.0, False

        kxy = pose.get("kxy")
        kcf = pose.get("kconf")
        if kxy is None or len(kxy) < 17:
            y_b = int(0.42 * h)
            y_c = int(0.75 * h)
            upper = crop_bgr[:y_b, :]
            lower = crop_bgr[y_c:, :]
            side = crop_bgr[int(0.20 * h):int(0.85 * h), int(0.70 * w):]
            return upper, lower, crop_bgr, side, 0.0, False

        x1, y1, x2, y2 = [float(v) for v in box_xyxy]
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)

        def _ys(ids: list[int]) -> list[float]:
            out = []
            for idx in ids:
                if idx >= len(kxy):
                    continue
                if kcf is not None and idx < len(kcf) and float(kcf[idx]) < self.pose_min_conf:
                    continue
                ky = float(kxy[idx][1])
                if np.isfinite(ky):
                    out.append(ky - y1)
            return out

        shoulder_ys = _ys([5, 6])
        hip_ys = _ys([11, 12])
        knee_ys = _ys([13, 14])
        conf_score = 0.0
        if kcf is not None and len(kcf) > 0:
            conf_score = float(np.nanmean(kcf))

        if not shoulder_ys or not hip_ys:
            y_b = int(0.42 * h)
            y_c = int(0.75 * h)
            upper = crop_bgr[:y_b, :]
            lower = crop_bgr[y_c:, :]
            side = crop_bgr[int(0.20 * h):int(0.85 * h), int(0.70 * w):]
            return upper, lower, crop_bgr, side, conf_score, False

        shoulder_y = float(np.clip(np.mean(shoulder_ys), 0.0, bh - 1.0))
        hip_y = float(np.clip(np.mean(hip_ys), 0.0, bh - 1.0))
        knee_y = float(np.clip(np.mean(knee_ys), hip_y + 2.0, bh - 1.0)) if knee_ys else min(bh - 1.0, hip_y + 0.30 * bh)

        upper_end = int(np.clip(0.58 * shoulder_y + 0.42 * hip_y, 6.0, bh - 6.0))
        lower_start = int(np.clip(0.55 * hip_y + 0.45 * knee_y, 6.0, bh - 6.0))
        if lower_start <= upper_end + 4:
            lower_start = min(int(bh - 6), upper_end + 5)

        upper = crop_bgr[:upper_end, :]
        lower = crop_bgr[lower_start:, :]
        side_x1 = int(max(0, min(w - 2, 0.72 * w)))
        side = crop_bgr[int(0.20 * h):int(0.88 * h), side_x1:]
        return upper, lower, crop_bgr, side, conf_score, True

    def _accessory_descriptor(self, crop_bgr: np.ndarray, side_region: np.ndarray) -> np.ndarray:
        if crop_bgr is None or crop_bgr.size == 0:
            return np.zeros((48,), dtype=np.float32)
        h, w = crop_bgr.shape[:2]
        top = crop_bgr[: max(2, int(0.22 * h)), :]
        left = crop_bgr[int(0.20 * h):int(0.85 * h), : max(2, int(0.22 * w))]
        side = side_region if side_region is not None and side_region.size > 0 else crop_bgr[int(0.20 * h):int(0.85 * h), int(0.72 * w):]
        v = np.concatenate(
            [
                self._region_stats(top),
                self._region_stats(left),
                self._region_stats(side),
                self._region_stats(crop_bgr),
            ],
            axis=0,
        ).astype(np.float32)
        return l2_normalize_np(v)

    def _profile_state(self, gid: int) -> IdentityProfileState:
        g = int(gid)
        st = self.canonical_identity_memory.get(g)
        if st is None:
            st = IdentityProfileState(gid=g, topk_prototypes=deque(maxlen=self.prototype_bank_size))
            self.canonical_identity_memory[g] = st
        self.identity_profile_state[g] = st
        return st

    def _decay_identity_profile_trust(self, frame_id: int) -> None:
        for gid, st in self.identity_profile_state.items():
            last_seen = int(self.reentry_last_frame.get(int(gid), st.last_frame))
            if last_seen < 0:
                continue
            lost_age = max(0, int(frame_id) - int(last_seen))
            st.lost_age = int(lost_age)
            decay = np.exp(-self.trust_decay_rate * float(lost_age))
            if lost_age > self.motion_max_gap:
                # Extra decay for long-missing identities: reduce zombie-ID stealing risk.
                decay *= np.exp(-0.80 * self.trust_decay_rate * float(lost_age - self.motion_max_gap))
            st.trust_score = float(np.clip(decay, 0.02, 1.0))
            if lost_age > self.lost_track_max_age:
                st.trust_score = float(min(st.trust_score, 0.06))

    def _should_update_profile(
        self,
        *,
        st: IdentityProfileState,
        track,
        profile: Optional[dict[str, object]],
        overlap_ambiguous: bool,
        frame_id: int,
    ) -> tuple[bool, str]:
        if profile is None:
            return False, "profile-missing"
        if overlap_ambiguous:
            # Freeze updates in overlap to avoid contamination by mixed/partial crops.
            return False, "overlap-frozen"
        q = float(profile.get("quality", 0.0))
        if q < self.min_crop_quality_for_reuse:
            return False, "low-crop-quality"
        blur_var = float(profile.get("blur_var", 0.0))
        if blur_var < self.reid_min_blur_var:
            return False, "blur-too-high"
        border_trunc = float(profile.get("border_trunc", 1.0))
        if border_trunc > 0.58:
            return False, "border-truncation"
        area_ratio = float(profile.get("area_ratio", 0.0))
        if area_ratio < max(self.reid_min_area_ratio, 0.0032):
            return False, "box-too-small"
        cur_feat = getattr(track, "feature", None)
        if cur_feat is None:
            return False, "missing-feature"
        cur_feat = cur_feat.astype(np.float32)
        n = float(np.linalg.norm(cur_feat))
        if n <= 1e-6:
            return False, "feature-invalid"
        cur_feat = cur_feat / n
        if st.stable_ema_feature is not None:
            sim = self._safe_cos(cur_feat, st.stable_ema_feature)
            if sim < self.min_profile_update_similarity:
                # Block profile update when current crop disagrees with stable profile.
                return False, "profile-sim-too-low"
        if int(frame_id) <= int(st.recovery_until):
            return False, "recovery-lock"
        return True, "ok"

    def _overlap_counts(self, tracks) -> dict[int, int]:
        counts: dict[int, int] = {}
        if not tracks:
            return counts
        n = len(tracks)
        for i in range(n):
            ti = tracks[i]
            bi = np.array(getattr(ti, "tlbr", [0, 0, 0, 0]), dtype=np.float32)
            c = 0
            for j in range(n):
                if i == j:
                    continue
                tj = tracks[j]
                bj = np.array(getattr(tj, "tlbr", [0, 0, 0, 0]), dtype=np.float32)
                xx1 = max(float(bi[0]), float(bj[0]))
                yy1 = max(float(bi[1]), float(bj[1]))
                xx2 = min(float(bi[2]), float(bj[2]))
                yy2 = min(float(bi[3]), float(bj[3]))
                iw = max(0.0, xx2 - xx1)
                ih = max(0.0, yy2 - yy1)
                inter = iw * ih
                ai = max(0.0, float(bi[2] - bi[0])) * max(0.0, float(bi[3] - bi[1]))
                aj = max(0.0, float(bj[2] - bj[0])) * max(0.0, float(bj[3] - bj[1]))
                den = ai + aj - inter + 1e-6
                iou = inter / den if den > 0 else 0.0
                if iou >= max(self.overlap_hold_iou_thresh, self.overlap_iou_threshold):
                    c += 1
            counts[int(getattr(ti, "track_id", -1))] = int(c)
        return counts

    def _zone_compatibility(self, prev_zone: str, cur_zone: str, age: int) -> float:
        if prev_zone == cur_zone:
            z = 1.0
        elif prev_zone == "center" or cur_zone == "center":
            z = 0.58
        elif {prev_zone, cur_zone} == {"left", "right"}:
            z = 0.20
        elif {prev_zone, cur_zone} == {"top", "bottom"}:
            z = 0.20
        else:
            z = 0.34

        # As age grows, require stronger geometric plausibility.
        if age > self.motion_max_gap:
            z -= 0.10 * min(1.0, float(age - self.motion_max_gap) / max(1.0, float(self.reentry_max_age)))
        return float(np.clip(z, 0.0, 1.0))

    def _safe_cos(self, a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
        if a is None or b is None:
            return 0.0
        if a.shape != b.shape:
            return 0.0
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na <= 1e-6 or nb <= 1e-6:
            return 0.0
        return float(np.dot(a / na, b / nb))

    def _debug_block(
        self,
        *,
        frame_id: int,
        reason: str,
        track=None,
        gid: Optional[int] = None,
        src: str = "",
        extra: str = "",
    ) -> None:
        if not bool(self.debug_reid_decisions):
            return
        tid = int(getattr(track, "track_id", -1)) if track is not None else -1
        gtxt = f" gid={int(gid)}" if gid is not None else ""
        stxt = f" src={str(src)}" if src else ""
        etxt = f" {str(extra)}" if extra else ""
        print(f"[REID-BLOCK] frame={int(frame_id)} track={tid}{gtxt}{stxt} reason={str(reason)}{etxt}")

    def _is_detection_human_like(
        self,
        frame: np.ndarray,
        box_xyxy,
        conf: float,
    ) -> tuple[bool, str]:
        x1, y1, x2, y2 = [float(v) for v in box_xyxy]
        fw = max(1.0, float(self._frame_w))
        fh = max(1.0, float(self._frame_h))
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        ar = float(bw / max(1.0, bh))
        area_ratio = float((bw * bh) / max(1.0, fw * fh))
        bw_ratio = float(bw / fw)
        bh_ratio = float(bh / fh)
        cx = float((x1 + x2) * 0.5 / fw)
        cy = float((y1 + y2) * 0.5 / fh)

        if ar < self.min_human_aspect_ratio or ar > self.max_human_aspect_ratio:
            return False, "non-human-aspect-ratio"
        if area_ratio < self.min_human_area_ratio:
            return False, "non-human-small-area"
        if float(conf) < self.min_person_conf_for_identity and area_ratio < (self.min_human_area_ratio * 1.8):
            return False, "non-human-low-conf-small"

        x1i, y1i, x2i, y2i = self._clip_box(box_xyxy)
        if x2i <= x1i + 2 or y2i <= y1i + 2:
            return False, "non-human-invalid-box"
        crop = frame[y1i:y2i, x1i:x2i]
        if crop is None or crop.size == 0:
            return False, "non-human-empty-crop"

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        sat_mean = float(hsv[..., 1].mean() / 255.0)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        edge_den = float((cv2.Canny(gray, 60, 150) > 0).mean()) if gray.size > 0 else 0.0

        right_zone = cx >= float(self.ghost_right_zone_xmin)
        lower_zone = cy >= float(self.ghost_zone_ymin)
        near_border = (x1 <= 0.010 * fw) or (x2 >= 0.992 * fw) or (y1 <= 0.010 * fh) or (y2 >= 0.992 * fh)

        # Ghost strips from shelf/toilet regions: narrow, low-texture, low-saturation, edge-hugging.
        if (
            right_zone
            and lower_zone
            and ar <= 0.34
            and bw_ratio <= 0.090
            and bh_ratio >= 0.12
            and sat_mean <= self.ghost_low_sat_max
            and lap_var <= self.ghost_low_texture_max
            and float(conf) <= 0.80
        ):
            return False, "non-human-right-toilet-ghost"

        if (
            right_zone
            and lower_zone
            and area_ratio <= max(self.min_human_area_ratio * 1.20, 0.0040)
            and sat_mean <= max(self.ghost_low_sat_max + 0.02, 0.18)
            and lap_var <= (self.ghost_low_texture_max + 3.0)
            and edge_den <= 0.065
            and float(conf) <= 0.88
        ):
            return False, "non-human-right-zone-weak-human-evidence"

        if (
            near_border
            and ar <= 0.28
            and area_ratio <= 0.014
            and edge_den <= 0.05
            and sat_mean <= max(0.10, self.ghost_low_sat_max)
        ):
            return False, "non-human-shelf-edge-ghost"

        return True, "ok"

    def _is_track_human_candidate(
        self,
        track,
        profile: Optional[dict[str, object]],
    ) -> tuple[bool, str]:
        if profile is None:
            return False, "non-human-missing-profile"
        q = float(profile.get("quality", 0.0))
        conf = float(getattr(track, "last_det_conf", 0.0))
        area_ratio = float(profile.get("area_ratio", 0.0))
        bw_ratio = float(profile.get("box_w_ratio", 0.0))
        bh_ratio = float(profile.get("box_h_ratio", 0.0))
        sat_mean = float(profile.get("sat_mean", 0.0))
        blur_var = float(profile.get("blur_var", 0.0))
        edge_den = float(profile.get("edge_density", 0.0))
        x1, y1, x2, y2 = [float(v) for v in getattr(track, "tlbr", [0, 0, 0, 0])]
        ar = float(max(1.0, x2 - x1) / max(1.0, y2 - y1))
        cx = float((x1 + x2) * 0.5 / max(1.0, float(self._frame_w)))
        cy = float((y1 + y2) * 0.5 / max(1.0, float(self._frame_h)))

        if area_ratio < self.min_human_area_ratio:
            return False, "non-human-small-track-area"
        if ar < self.min_human_aspect_ratio or ar > self.max_human_aspect_ratio:
            return False, "non-human-track-aspect"
        if bw_ratio <= 0.025 and bh_ratio >= 0.18:
            return False, "non-human-thin-strip"
        if bh_ratio <= 0.055:
            return False, "non-human-too-short-height"
        if area_ratio <= max(self.min_human_area_ratio * 1.15, 0.0038) and conf < (self.min_person_conf_for_identity + 0.08):
            return False, "non-human-small-low-conf-track"
        if conf < self.min_person_conf_for_identity and q < max(0.56, self.min_crop_quality_for_reuse):
            return False, "non-human-low-conf-quality"
        if (
            cx >= float(self.ghost_right_zone_xmin)
            and cy >= float(self.ghost_zone_ymin)
            and ar <= 0.35
            and sat_mean <= self.ghost_low_sat_max
            and blur_var <= self.ghost_low_texture_max
            and edge_den <= 0.060
            and q <= 0.70
        ):
            return False, "non-human-right-toilet-track"
        near_border = (
            x1 <= 0.010 * max(1.0, float(self._frame_w))
            or x2 >= 0.992 * max(1.0, float(self._frame_w))
            or y1 <= 0.010 * max(1.0, float(self._frame_h))
            or y2 >= 0.992 * max(1.0, float(self._frame_h))
        )
        if (
            near_border
            and ar <= 0.30
            and area_ratio <= 0.013
            and edge_den <= 0.055
            and sat_mean <= max(0.12, self.ghost_low_sat_max + 0.01)
            and q <= 0.72
        ):
            return False, "non-human-border-strip-track"
        return True, "ok"

    def _region_stats(self, region_bgr: np.ndarray) -> np.ndarray:
        if region_bgr is None or region_bgr.size == 0:
            return np.zeros((6,), dtype=np.float32)
        hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
        mean = hsv.mean(axis=0)
        std = hsv.std(axis=0)
        # Normalize approximate ranges: H[0..180], S,V[0..255]
        out = np.array(
            [
                mean[0] / 180.0,
                mean[1] / 255.0,
                mean[2] / 255.0,
                std[0] / 90.0,
                std[1] / 128.0,
                std[2] / 128.0,
            ],
            dtype=np.float32,
        )
        n = float(np.linalg.norm(out))
        return out / (n + 1e-12)

    def _extract_track_profile(self, frame: np.ndarray, track) -> Optional[dict[str, object]]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in track.tlbr]
        x1i, y1i = max(0, int(x1)), max(0, int(y1))
        x2i, y2i = min(w - 1, int(x2)), min(h - 1, int(y2))
        if x2i <= x1i + 6 or y2i <= y1i + 10:
            return None

        crop = frame[y1i:y2i, x1i:x2i]
        if crop is None or crop.size == 0:
            return None

        upper, lower, whole, side_region, pose_conf, pose_used = self._pose_aligned_regions(
            crop,
            track.tlbr,
        )

        upper_v = self._region_stats(upper)          # Upper-body feature.
        lower_v = self._region_stats(lower)          # Lower-body feature.
        whole_v = self._region_stats(whole)          # Whole-body feature.
        upper_hist = _safe_hist(upper)
        lower_hist = _safe_hist(lower)
        # Saturated-pixel hue histograms and dominant color signatures
        # act as an extra CV confirmation cue: two people with similar OSNet
        # embeddings but clearly different shirt/pants color produce very
        # different hue distributions even when the global embedding drifts.
        upper_hue_sat = _hue_sat_histogram_region(upper)
        lower_hue_sat = _hue_sat_histogram_region(lower)
        upper_dom_color = dominant_color_descriptor(upper)
        lower_dom_color = dominant_color_descriptor(lower)
        shape_v = body_shape_descriptor(np.array([x1, y1, x2, y2], dtype=np.float32), frame_h=h, frame_w=w)
        accessory_v = self._accessory_descriptor(crop, side_region)

        motion = np.array([float(getattr(track, "vx", 0.0)), float(getattr(track, "vy", 0.0))], dtype=np.float32)
        mnorm = float(np.linalg.norm(motion))
        if mnorm > 1e-6:
            motion = motion / mnorm

        q_metrics = self._compute_box_quality_metrics(crop, track.tlbr)
        quality = float(getattr(track, "last_feat_quality", 0.0))
        return {
            "upper": upper_v,
            "lower": lower_v,
            "whole": whole_v,
            "upper_hist": l2_normalize_np(upper_hist.astype(np.float32)),
            "lower_hist": l2_normalize_np(lower_hist.astype(np.float32)),
            "upper_hue_sat": upper_hue_sat,
            "lower_hue_sat": lower_hue_sat,
            "upper_dom_color": upper_dom_color,
            "lower_dom_color": lower_dom_color,
            "shape": shape_v,
            "accessory": accessory_v,
            "h_ratio": float(max(1.0, y2 - y1) / max(1.0, float(h))),
            "zone": self._zone_of_box(track.tlbr),
            "quality": float(quality),
            "motion": motion,
            "pose_used": bool(pose_used),
            "pose_conf": float(pose_conf),
            "blur_var": float(q_metrics["blur_var"]),
            "sat_mean": float(q_metrics["sat_mean"]),
            "edge_density": float(q_metrics["edge_density"]),
            "border_trunc": float(q_metrics["border_trunc"]),
            "area_ratio": float(q_metrics["area_ratio"]),
            "box_h_ratio": float(q_metrics["box_h_ratio"]),
            "box_w_ratio": float(q_metrics["box_w_ratio"]),
        }

    def _record_profile_sample(
        self,
        gid: int,
        profile: Optional[dict[str, object]],
        track=None,
        frame_id: Optional[int] = None,
        *,
        overlap_ambiguous: bool = False,
    ) -> None:
        if gid <= 0:
            return
        g = int(gid)
        st = self._profile_state(g)
        if profile is None:
            if overlap_ambiguous and frame_id is not None:
                st.frozen_until = max(int(st.frozen_until), int(frame_id) + self.freeze_duration_after_overlap)
                st.recovery_until = max(
                    int(st.recovery_until),
                    int(frame_id)
                    + int(self.freeze_duration_after_overlap)
                    + int(self.overlap_recovery_extra_frames),
                )
            return
        if frame_id is None:
            frame_id = int(self.reentry_last_frame.get(g, 0))

        # BUGFIX: Previously frozen_until was incremented BEFORE the
        # _should_update_profile() gate. That caused the freeze window to keep
        # extending even on frames where we ultimately rejected the update
        # (e.g. due to low quality, not just overlap). Result: the gid was
        # frozen far longer than needed, blocking legitimate re-entry matches
        # for the SAME person ("same person must reuse same id" bug).
        # Additionally the profile was being appended to identity_profiles[g]
        # BEFORE the gate, so overlap-poisoned or low-quality samples ended up
        # in the reference bank and silently lowered future similarity scores.
        # We now run the gate FIRST, then decide both whether to append and
        # whether to extend the freeze window.
        allow_update, reason = self._should_update_profile(
            st=st,
            track=track,
            profile=profile,
            overlap_ambiguous=overlap_ambiguous,
            frame_id=int(frame_id),
        )
        if not allow_update:
            # Keep explicit reason for debugging why profile updates were frozen.
            st.last_block_reason = str(reason)
            self._debug_block(
                frame_id=int(frame_id),
                reason=f"profile-update-blocked:{str(reason)}",
                track=track,
                gid=int(g),
                src="profile",
            )
            return

        # Append profile to the reference bank only after passing the gate.
        if g not in self.identity_profiles:
            self.identity_profiles[g] = deque(maxlen=max(6, self.reentry_gallery_size))
        refs = self.identity_profiles[g]
        if refs:
            last = refs[-1]
            sim_up = self._safe_cos(profile.get("upper"), last.get("upper"))
            sim_lo = self._safe_cos(profile.get("lower"), last.get("lower"))
            q_new = float(profile.get("quality", 0.0))
            q_last = float(last.get("quality", 0.0))
            if sim_up > 0.992 and sim_lo > 0.992 and q_new <= q_last:
                # Skip near-duplicate frames; keep profile bank diverse.
                pass
            else:
                refs.append(profile)
        else:
            refs.append(profile)

        if overlap_ambiguous:
            st.frozen_until = max(int(st.frozen_until), int(frame_id) + self.freeze_duration_after_overlap)
            st.recovery_until = max(
                int(st.recovery_until),
                int(frame_id)
                + int(self.freeze_duration_after_overlap)
                + int(self.overlap_recovery_extra_frames),
            )

        feat = getattr(track, "feature", None) if track is not None else None
        if feat is not None:
            feat = feat.astype(np.float32)
            n = float(np.linalg.norm(feat))
            if n > 1e-6:
                feat = feat / n
            else:
                feat = None

        if feat is not None:
            if st.first_feature is None:
                st.first_feature = feat.copy()
            st.recent_feature = feat.copy()
            if st.stable_ema_feature is None:
                st.stable_ema_feature = feat.copy()
            else:
                st.stable_ema_feature = l2_normalize_np(0.92 * st.stable_ema_feature + 0.08 * feat)
            st.global_osnet = feat.copy()
            if not st.topk_prototypes:
                st.topk_prototypes.append(feat.copy())
            else:
                sim_to_bank = [self._safe_cos(feat, p) for p in st.topk_prototypes]
                if max(sim_to_bank) < 0.985:
                    st.topk_prototypes.append(feat.copy())
                elif len(st.topk_prototypes) < st.topk_prototypes.maxlen:
                    st.topk_prototypes.append(feat.copy())

        st.upper_body_feature = profile.get("upper")
        st.lower_body_feature = profile.get("lower")
        st.whole_body_feature = profile.get("whole")
        st.upper_body_hist = profile.get("upper_hist")
        st.lower_body_hist = profile.get("lower_hist")
        st.upper_hue_sat_hist = profile.get("upper_hue_sat")
        st.lower_hue_sat_hist = profile.get("lower_hue_sat")
        st.upper_dominant_color = profile.get("upper_dom_color")
        st.lower_dominant_color = profile.get("lower_dom_color")
        st.body_shape_descriptor = profile.get("shape")
        st.accessory_descriptor = profile.get("accessory")
        q = float(profile.get("quality", 0.0))
        st.crop_quality_history.append(q)
        if st.first_box is None and track is not None:
            x1, y1, x2, y2 = [float(v) for v in track.tlbr]
            st.first_box = (x1, y1, x2, y2)
        if track is not None:
            x1, y1, x2, y2 = [float(v) for v in track.tlbr]
            if st.last_box is not None and st.last_frame >= 0 and int(frame_id) > int(st.last_frame):
                dt = max(1, int(frame_id) - int(st.last_frame))
                lcx = 0.5 * (st.last_box[0] + st.last_box[2])
                lcy = 0.5 * (st.last_box[1] + st.last_box[3])
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)
                vx = (cx - lcx) / float(dt)
                vy = (cy - lcy) / float(dt)
                old_vx, old_vy = st.velocity
                st.velocity = (0.72 * float(old_vx) + 0.28 * float(vx), 0.72 * float(old_vy) + 0.28 * float(vy))
            st.last_box = (x1, y1, x2, y2)
        st.last_frame = int(frame_id)
        st.lost_age = 0
        st.trust_score = float(min(1.0, max(0.20, 0.70 * st.trust_score + 0.30)))
        st.last_block_reason = "ok"

    def _profile_similarity_to_gid(self, gid: int, profile: Optional[dict[str, object]], frame_id: int) -> tuple[float, float, float, float]:
        if gid <= 0 or profile is None:
            return 0.0, 0.0, 0.0, 0.0
        refs = self.identity_profiles.get(int(gid))
        st = self.identity_profile_state.get(int(gid))
        if not refs and st is None:
            return 0.0, 0.0, 0.0, 0.0

        upper_scores = []
        lower_scores = []
        whole_scores = []
        upper_hist_scores = []
        lower_hist_scores = []
        upper_hue_scores: list[float] = []
        lower_hue_scores: list[float] = []
        upper_dom_scores: list[float] = []
        lower_dom_scores: list[float] = []
        shape_scores = []
        size_scores = []
        motion_scores = []
        accessory_scores = []
        cur_upper = profile.get("upper")
        cur_lower = profile.get("lower")
        cur_whole = profile.get("whole")
        cur_up_hist = profile.get("upper_hist")
        cur_lo_hist = profile.get("lower_hist")
        cur_up_hue = profile.get("upper_hue_sat")
        cur_lo_hue = profile.get("lower_hue_sat")
        cur_up_dom = profile.get("upper_dom_color")
        cur_lo_dom = profile.get("lower_dom_color")
        cur_shape = profile.get("shape")
        cur_hr = float(profile.get("h_ratio", 0.0))
        cur_accessory = profile.get("accessory")
        cur_motion = profile.get("motion")
        cur_zone = str(profile.get("zone", "center"))
        age = int(frame_id) - int(self.reentry_last_frame.get(int(gid), frame_id))
        zone_score = self._zone_compatibility(self.reentry_last_zone.get(int(gid), "center"), cur_zone, age)

        if refs:
            for r in refs:
                up = r.get("upper")
                lo = r.get("lower")
                wh = r.get("whole")
                uph = r.get("upper_hist")
                loh = r.get("lower_hist")
                uphue = r.get("upper_hue_sat")
                lohue = r.get("lower_hue_sat")
                updom = r.get("upper_dom_color")
                lodom = r.get("lower_dom_color")
                sh = r.get("shape")
                hr = float(r.get("h_ratio", 0.0))
                rm = r.get("motion")
                ac = r.get("accessory")
                upper_scores.append(float(self._safe_cos(cur_upper, up)))
                lower_scores.append(float(self._safe_cos(cur_lower, lo)))
                whole_scores.append(float(self._safe_cos(cur_whole, wh)))
                upper_hist_scores.append(float(self._safe_cos(cur_up_hist, uph)))
                lower_hist_scores.append(float(self._safe_cos(cur_lo_hist, loh)))
                upper_hue_scores.append(float(self._safe_cos(cur_up_hue, uphue)))
                lower_hue_scores.append(float(self._safe_cos(cur_lo_hue, lohue)))
                upper_dom_scores.append(float(self._safe_cos(cur_up_dom, updom)))
                lower_dom_scores.append(float(self._safe_cos(cur_lo_dom, lodom)))
                shape_scores.append(float(self._safe_cos(cur_shape, sh)))
                accessory_scores.append(float(self._safe_cos(cur_accessory, ac)))
                if cur_hr > 1e-6 and hr > 1e-6:
                    size_scores.append(float(min(cur_hr, hr) / max(cur_hr, hr)))
                if cur_motion is not None and rm is not None:
                    motion_scores.append(float(max(-1.0, min(1.0, float(np.dot(cur_motion, rm))))))

        if st is not None:
            upper_scores.append(float(self._safe_cos(cur_upper, st.upper_body_feature)))
            lower_scores.append(float(self._safe_cos(cur_lower, st.lower_body_feature)))
            whole_scores.append(float(self._safe_cos(cur_whole, st.whole_body_feature)))
            upper_hist_scores.append(float(self._safe_cos(cur_up_hist, st.upper_body_hist)))
            lower_hist_scores.append(float(self._safe_cos(cur_lo_hist, st.lower_body_hist)))
            upper_hue_scores.append(float(self._safe_cos(cur_up_hue, st.upper_hue_sat_hist)))
            lower_hue_scores.append(float(self._safe_cos(cur_lo_hue, st.lower_hue_sat_hist)))
            upper_dom_scores.append(float(self._safe_cos(cur_up_dom, st.upper_dominant_color)))
            lower_dom_scores.append(float(self._safe_cos(cur_lo_dom, st.lower_dominant_color)))
            shape_scores.append(float(self._safe_cos(cur_shape, st.body_shape_descriptor)))
            accessory_scores.append(float(self._safe_cos(cur_accessory, st.accessory_descriptor)))
            if cur_hr > 1e-6 and st.last_box is not None:
                st_h = max(1e-6, float(st.last_box[3] - st.last_box[1])) / max(1.0, float(self._frame_h))
                size_scores.append(float(min(cur_hr, st_h) / max(cur_hr, st_h)))
            if cur_motion is not None:
                vx, vy = st.velocity
                mv = np.array([float(vx), float(vy)], dtype=np.float32)
                mv_n = float(np.linalg.norm(mv))
                if mv_n > 1e-6:
                    mv = mv / mv_n
                    motion_scores.append(float(max(-1.0, min(1.0, float(np.dot(cur_motion, mv))))))

        k = self.profile_topk
        upper_topk = float(np.mean(sorted(upper_scores, reverse=True)[: min(k, len(upper_scores))])) if upper_scores else 0.0
        lower_topk = float(np.mean(sorted(lower_scores, reverse=True)[: min(k, len(lower_scores))])) if lower_scores else 0.0
        whole_topk = float(np.mean(sorted(whole_scores, reverse=True)[: min(k, len(whole_scores))])) if whole_scores else 0.0
        up_hist_topk = float(np.mean(sorted(upper_hist_scores, reverse=True)[: min(k, len(upper_hist_scores))])) if upper_hist_scores else 0.0
        lo_hist_topk = float(np.mean(sorted(lower_hist_scores, reverse=True)[: min(k, len(lower_hist_scores))])) if lower_hist_scores else 0.0
        up_hue_topk = float(np.mean(sorted(upper_hue_scores, reverse=True)[: min(k, len(upper_hue_scores))])) if upper_hue_scores else 0.0
        lo_hue_topk = float(np.mean(sorted(lower_hue_scores, reverse=True)[: min(k, len(lower_hue_scores))])) if lower_hue_scores else 0.0
        up_dom_topk = float(np.mean(sorted(upper_dom_scores, reverse=True)[: min(k, len(upper_dom_scores))])) if upper_dom_scores else 0.0
        lo_dom_topk = float(np.mean(sorted(lower_dom_scores, reverse=True)[: min(k, len(lower_dom_scores))])) if lower_dom_scores else 0.0
        accessory_topk = float(np.mean(sorted(accessory_scores, reverse=True)[: min(k, len(accessory_scores))])) if accessory_scores else 0.0
        part_disagreement = abs(upper_topk - lower_topk)
        strong_part_conflict = bool(
            (upper_topk >= 0.62 and lower_topk <= 0.36)
            or (lower_topk >= 0.62 and upper_topk <= 0.36)
        )
        part_agreement = float(np.clip(1.0 - max(0.0, part_disagreement - 0.06) * 2.20, 0.05, 1.0))
        # Secondary evidence built from part-level color/hist/accessory cues.
        color_part_mean = float(
            np.clip(
                0.30 * up_hue_topk
                + 0.22 * lo_hue_topk
                + 0.28 * up_dom_topk
                + 0.20 * lo_dom_topk,
                0.0,
                1.0,
            )
        )
        secondary_consistency = float(
            np.clip(
                0.48 * color_part_mean
                + 0.22 * up_hist_topk
                + 0.14 * lo_hist_topk
                + 0.16 * max(0.0, accessory_topk),
                0.0,
                1.0,
            )
        )
        global_appearance = float(
            np.clip(
                0.55 * whole_topk + 0.25 * upper_topk + 0.20 * lower_topk,
                0.0,
                1.0,
            )
        )
        disagreement_penalty = 0.0
        if strong_part_conflict:
            disagreement_penalty += 0.22
        if global_appearance >= 0.64 and secondary_consistency < 0.52:
            disagreement_penalty += 0.20 + 0.40 * (0.52 - secondary_consistency)
        if upper_topk >= 0.62 and up_hue_topk < 0.40 and up_dom_topk < 0.45:
            disagreement_penalty += 0.16
        if lower_topk >= 0.58 and lo_hue_topk < 0.38 and lo_dom_topk < 0.42:
            disagreement_penalty += 0.12
        if abs(up_hue_topk - lo_hue_topk) > 0.42 and min(up_hue_topk, lo_hue_topk) < 0.32:
            disagreement_penalty += 0.10
        if whole_topk >= 0.70 and (upper_topk < 0.42 or lower_topk < 0.42):
            disagreement_penalty += 0.15
        cloth_raw = (
            0.28 * upper_topk
            + 0.20 * lower_topk
            + 0.14 * whole_topk
            + 0.10 * up_hist_topk
            + 0.08 * lo_hist_topk
            + 0.12 * up_hue_topk
            + 0.08 * lo_hue_topk
            + 0.08 * max(0.0, accessory_topk)
        )
        cloth_raw += 0.10 * max(0.0, color_part_mean - 0.52)
        cloth_raw -= float(disagreement_penalty)
        # Penalize mismatched upper/lower agreement to prevent shirt-only false matches.
        cloth_score = float(max(0.0, cloth_raw) * part_agreement)
        shape_score = float(np.mean(sorted(shape_scores, reverse=True)[: min(k, len(shape_scores))])) if shape_scores else 0.0
        size_score = float(np.mean(sorted(size_scores, reverse=True)[: min(k, len(size_scores))])) if size_scores else 0.0
        motion_score = float(np.mean(sorted(motion_scores, reverse=True)[: min(k, len(motion_scores))])) if motion_scores else 0.0
        zone_motion = float(0.64 * zone_score + 0.18 * max(0.0, motion_score) + 0.18 * max(0.0, accessory_topk))
        return cloth_score, shape_score, size_score, zone_motion

    def _record_reentry_sample(
        self,
        track,
        frame_id: int,
        profile: Optional[dict[str, object]] = None,
        *,
        overlap_ambiguous: bool = False,
    ) -> None:
        gid = int(getattr(track, "global_id", 0) or 0)
        if gid <= 0:
            return

        feat = getattr(track, "feature", None)
        if feat is None:
            return
        f = feat.astype(np.float32)
        n = float(np.linalg.norm(f))
        if n <= 1e-6:
            return
        f = f / n

        q = float(getattr(track, "last_feat_quality", 0.0))
        mode = int(getattr(track, "last_reid_mode", 0))
        det_c = float(getattr(track, "last_det_conf", 0.0))
        if overlap_ambiguous:
            return
        if mode < 2 or q < max(self.reid_min_quality_for_bank, self.profile_min_quality) or det_c < self.det_conf:
            return

        if gid not in self.reentry_gallery:
            self.reentry_gallery[gid] = deque(maxlen=max(4, self.reentry_gallery_size))
        if self.reentry_gallery[gid]:
            best_sim = max(float(np.dot(f, s)) for s in self.reentry_gallery[gid] if s is not None and s.shape == f.shape)
            if best_sim > 0.992:
                # Too redundant; keep clean diverse prototypes only.
                return
        self.reentry_gallery[gid].append(f)

        x1, y1, x2, y2 = [float(v) for v in track.tlbr]
        self.reentry_last_frame[gid] = int(frame_id)
        self.reentry_last_box[gid] = (x1, y1, x2, y2)
        self.reentry_last_zone[gid] = self._zone_of_box(track.tlbr)

    def _reentry_match(
        self,
        track,
        track_profile: Optional[dict[str, object]],
        frame_id: int,
        forbidden_gids: Optional[set[int]] = None,
    ) -> tuple[int, float]:
        if not self.reentry_gallery:
            return 0, -1.0
        feat = getattr(track, "feature", None)
        if feat is None:
            return 0, -1.0

        f = feat.astype(np.float32)
        n = float(np.linalg.norm(f))
        if n <= 1e-6:
            return 0, -1.0
        f = f / n

        forbidden = set(int(x) for x in forbidden_gids) if forbidden_gids is not None else set()
        cur_zone = self._zone_of_box(track.tlbr)
        overlap_n = int(getattr(track, "overlap_neighbors", 0))
        right_aisle = self._is_right_aisle_box(track.tlbr)
        crop_q = float(track_profile.get("quality", 0.0)) if track_profile is not None else 0.0
        min_q = self.min_crop_quality_for_reuse + (0.04 if overlap_n > 0 else 0.0) + (0.03 if right_aisle else 0.0)
        if crop_q < min_q:
            return 0, -1.0

        best_gid = 0
        best_score = -1.0
        second_score = -1.0

        for gid, samples in self.reentry_gallery.items():
            g = int(gid)
            if g in forbidden:
                continue
            last_f = int(self.reentry_last_frame.get(g, -1))
            if last_f < 0:
                continue
            age = int(frame_id) - last_f
            if age <= 0 or age > self.reentry_max_age:
                continue
            if overlap_n > 0 and age > self.motion_max_gap:
                # In crowded overlap windows, do not revive long-lost IDs.
                continue

            if len(samples) < self.reentry_min_samples and age > self.motion_max_gap:
                continue
            if age > self.motion_max_gap and len(samples) < (self.reentry_min_samples + 1):
                continue

            st = self.identity_profile_state.get(g, None)
            if st is not None:
                if int(frame_id) <= int(st.frozen_until):
                    # Overlap-frozen identities are protected from being stolen during recovery.
                    continue
                trust = float(st.trust_score)
            else:
                trust = 0.55

            sims = [float(np.dot(f, s)) for s in samples if s is not None and s.shape == f.shape]
            if not sims:
                continue
            sims.sort(reverse=True)
            k = min(max(3, self.profile_topk), len(sims))
            sim_gallery = float(0.72 * np.mean(sims[:k]) + 0.28 * sims[0])

            dyn_sim_req = self.reentry_sim_thresh + 0.05 * min(1.0, age / max(1.0, float(self.reentry_max_age)))
            if overlap_n > 0:
                dyn_sim_req += 0.02
            if right_aisle:
                dyn_sim_req += 0.02
            # Safety clamp: avoid impossible thresholds (>1.0) under strict runtime configs.
            dyn_sim_req = float(min(0.995, max(0.0, dyn_sim_req)))
            if sim_gallery < dyn_sim_req:
                continue
            if age > self.motion_max_gap and (sim_gallery < (self.zombie_id_protection_threshold + 0.06) or trust < 0.34):
                # Zombie-ID protection: old/lost IDs must have very strong appearance evidence.
                continue

            sim_bank = 0.0
            if self.id_bank is not None:
                sim_bank = max(0.0, float(self.id_bank.similarity_to_gid(f, g)))
            if age > self.motion_max_gap and sim_bank < max(self.normal_reuse_threshold + 0.03, 0.76):
                continue

            prev_zone = self.reentry_last_zone.get(g, "center")
            zone = self._zone_compatibility(prev_zone, cur_zone, age)
            if age > self.motion_max_gap and zone < self.reentry_min_zone_compat:
                continue

            cloth_score, shape_score, size_score, zone_motion = self._profile_similarity_to_gid(
                g, track_profile, frame_id=frame_id
            )
            shape_size = 0.60 * shape_score + 0.40 * size_score
            if age > self.motion_max_gap and (
                cloth_score < (0.62 if not right_aisle else 0.66)
                or shape_size < (0.34 if not right_aisle else 0.37)
                or zone_motion < (0.32 if not right_aisle else 0.36)
            ):
                continue

            # Weighted multi-feature decision:
            # Long-gap regime: prioritize appearance/prototypes over motion continuity.
            deep_score = 0.75 * sim_gallery + 0.25 * sim_bank
            zone_score = zone
            motion_score = max(0.0, min(1.0, zone_motion))
            q_bonus = float(track_profile.get("quality", 0.0)) if track_profile is not None else 0.0
            q_bonus = max(0.0, min(1.0, q_bonus))

            score = (
                0.55 * deep_score
                + 0.23 * max(0.0, min(1.0, cloth_score))
                + 0.08 * max(0.0, min(1.0, shape_size))
                + 0.05 * max(0.0, min(1.0, zone_score))
                + 0.04 * max(0.0, min(1.0, motion_score))
                + 0.05 * q_bonus
            )
            score -= 0.04 * min(1.0, age / max(1.0, float(self.reentry_max_age)))
            score += 0.03 * max(0.0, min(1.0, trust))
            if overlap_n > 0:
                score -= 0.03
            if right_aisle:
                score -= 0.03

            if score > best_score:
                second_score = best_score
                best_score = float(score)
                best_gid = g
            elif score > second_score:
                second_score = float(score)

        if best_gid <= 0:
            return 0, -1.0
        margin = best_score - max(-1.0, second_score)
        req_score = self.long_gap_reid_threshold + (0.05 if overlap_n > 0 else 0.02)
        if right_aisle:
            req_score += 0.02
        req_score = float(min(0.995, max(0.0, req_score)))
        req_margin = max(self.reentry_margin, self.best_vs_second_margin + 0.04)
        if overlap_n > 0:
            req_margin = max(req_margin, self.overlap_best_vs_second_margin + 0.05)
        if right_aisle:
            req_margin = max(req_margin, self.overlap_best_vs_second_margin + 0.07)
        if best_score < req_score:
            return 0, -1.0
        if margin < req_margin:
            return 0, -1.0
        return int(best_gid), float(best_score)

    def _short_memory_match(
        self,
        track,
        track_profile: Optional[dict[str, object]],
        frame_id: int,
        forbidden_gids: Optional[set[int]] = None,
    ) -> tuple[int, float]:
        if not self.reentry_gallery:
            return 0, -1.0
        feat = getattr(track, "feature", None)
        if feat is None:
            return 0, -1.0
        f = feat.astype(np.float32)
        n = float(np.linalg.norm(f))
        if n <= 1e-6:
            return 0, -1.0
        f = f / n

        forbidden = set(int(x) for x in forbidden_gids) if forbidden_gids is not None else set()
        overlap_n = int(getattr(track, "overlap_neighbors", 0))
        right_aisle = self._is_right_aisle_box(track.tlbr)
        crop_q = float(track_profile.get("quality", 0.0)) if track_profile is not None else 0.0
        min_q = self.min_crop_quality_for_reuse + (0.03 if overlap_n > 0 else 0.0) + (0.03 if right_aisle else 0.0)
        if crop_q < min_q:
            return 0, -1.0
        best_gid = 0
        best_score = -1.0
        second_score = -1.0

        for gid, samples in self.reentry_gallery.items():
            g = int(gid)
            if g in forbidden:
                continue
            last_f = int(self.reentry_last_frame.get(g, -1))
            if last_f < 0:
                continue
            age = int(frame_id) - last_f
            if age <= 0 or age > self.short_memory_max_age:
                continue
            if overlap_n > 0 and age > self.motion_max_gap:
                continue
            if right_aisle and age > self.motion_max_gap:
                continue
            if not samples:
                continue
            st = self.identity_profile_state.get(g, None)
            if st is not None:
                if int(frame_id) <= int(st.frozen_until):
                    continue
                trust = float(st.trust_score)
            else:
                trust = 0.55
            if age > self.motion_max_gap and trust < 0.30:
                continue

            sims = [float(np.dot(f, s)) for s in samples if s is not None and s.shape == f.shape]
            if not sims:
                continue
            sims.sort(reverse=True)
            k = min(max(3, self.profile_topk), len(sims))
            sim_gallery = float(0.72 * np.mean(sims[:k]) + 0.28 * sims[0])
            if age > self.motion_max_gap and sim_gallery < (self.zombie_id_protection_threshold + 0.04):
                continue
            sim_bank = 0.0
            if self.id_bank is not None:
                sim_bank = max(0.0, float(self.id_bank.similarity_to_gid(f, g)))
            spatial = self._spatial_score_for_gid(g, frame_id=frame_id, box_xyxy=track.tlbr)
            if spatial < self.min_spatial_consistency and sim_gallery < max(self.normal_reuse_threshold, 0.74):
                continue
            cloth, shape, size, zone_motion = self._profile_similarity_to_gid(g, track_profile, frame_id=frame_id)
            shape_size = 0.60 * shape + 0.40 * size
            if age > self.motion_max_gap and (
                cloth < (0.58 if not right_aisle else 0.62)
                or shape_size < (0.32 if not right_aisle else 0.36)
                or zone_motion < (0.30 if not right_aisle else 0.34)
            ):
                continue
            part_warn = abs(max(0.0, min(1.0, cloth)) - max(0.0, min(1.0, shape_size)))
            part_penalty = 0.16 * max(0.0, part_warn - 0.15)
            score = (
                0.46 * max(0.0, min(1.0, sim_gallery))
                + 0.24 * max(0.0, min(1.0, sim_bank))
                + 0.14 * max(0.0, min(1.0, cloth))
                + 0.06 * max(0.0, min(1.0, shape_size))
                + 0.06 * max(0.0, min(1.0, spatial))
                + 0.04 * max(0.0, min(1.0, zone_motion))
            )
            score -= 0.03 * min(1.0, age / max(1.0, float(self.short_memory_max_age)))
            score -= float(part_penalty)
            score += 0.04 * max(0.0, min(1.0, trust))
            if right_aisle:
                score -= 0.02

            if score > best_score:
                second_score = best_score
                best_score = float(score)
                best_gid = g
            elif score > second_score:
                second_score = float(score)

        if best_gid <= 0:
            return 0, -1.0
        margin = float(best_score - max(-1.0, second_score))
        req_score = self.overlap_reuse_threshold if overlap_n > 0 else self.normal_reuse_threshold
        req_score = max(req_score, self.short_memory_reuse_thresh)
        if overlap_n > 0:
            req_score += 0.02
        if right_aisle:
            req_score += 0.02
        req_score = float(min(0.995, max(0.0, req_score)))
        if best_score < req_score:
            return 0, -1.0
        req_margin = self.overlap_best_vs_second_margin if overlap_n > 0 else self.best_vs_second_margin
        req_margin = max(req_margin, self.short_memory_margin + (0.08 if overlap_n > 0 else 0.04))
        if right_aisle:
            req_margin += 0.02
        if margin < req_margin:
            return 0, -1.0
        return int(best_gid), float(best_score)

    def _detect(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        res = self.det_model.predict(
            source=frame,
            imgsz=self.det_imgsz,
            conf=self.det_conf,
            iou=self.det_iou,
            classes=self.det_classes,
            device=self.device,
            verbose=False,
        )[0]

        boxes = res.boxes
        if boxes is None or boxes.data is None or len(boxes) == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

        det_xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        det_conf = boxes.conf.detach().cpu().numpy().astype(np.float32)
        return det_xyxy, det_conf

    def _filter_whole_body(
        self,
        frame: np.ndarray,
        dets_xyxy: np.ndarray,
        det_confs: np.ndarray,
        *,
        frame_id: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if len(dets_xyxy) == 0:
            return dets_xyxy, det_confs

        H, W = frame.shape[:2]
        keep = []
        reject_counts: Dict[str, int] = {}
        for i, (x1, y1, x2, y2) in enumerate(dets_xyxy):
            h_box = max(0.0, float(y2) - float(y1))
            w_box = max(0.0, float(x2) - float(x1))
            if (h_box / max(H, 1)) < self.min_height_ratio or (w_box / max(W, 1)) < self.min_width_ratio:
                reject_counts["non-human-size-threshold"] = int(reject_counts.get("non-human-size-threshold", 0)) + 1
                continue
            ok_h, reason = self._is_detection_human_like(
                frame,
                dets_xyxy[i],
                conf=float(det_confs[i]),
            )
            if not ok_h:
                reject_counts[str(reason)] = int(reject_counts.get(str(reason), 0)) + 1
                continue
            keep.append(i)

        if bool(self.debug_reid_decisions) and reject_counts:
            rid = int(frame_id) if frame_id is not None else -1
            parts = ", ".join(f"{k}:{v}" for k, v in sorted(reject_counts.items(), key=lambda kv: (-kv[1], kv[0])))
            print(f"[REID-BLOCK] frame={rid} detect_filter_reject {{{parts}}}")

        if not keep:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

        return dets_xyxy[keep], det_confs[keep]

    def _extract_feats(
        self,
        frame: np.ndarray,
        dets_xyxy: np.ndarray,
        confs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if len(dets_xyxy) == 0:
            return (
                np.empty((0, 0), dtype=np.float32),
                dets_xyxy,
                confs,
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
            )
        if self.extractor is None:
            q = np.zeros((len(dets_xyxy),), dtype=np.float32)
            m = np.zeros((len(dets_xyxy),), dtype=np.int32)
            return np.zeros((len(dets_xyxy), self.reid_feat_dim), dtype=np.float32), dets_xyxy, confs, q, m

        H, W = frame.shape[:2]
        crops_rgb = []
        crops_bgr = []
        keep_idx = []
        quality = np.zeros((len(dets_xyxy),), dtype=np.float32)
        feat_mode = np.zeros((len(dets_xyxy),), dtype=np.int32)  # 0=reject,1=tentative,2=strong

        for i, (x1, y1, x2, y2) in enumerate(dets_xyxy):
            x1i, y1i = max(0, int(x1)), max(0, int(y1))
            x2i, y2i = min(W - 1, int(x2)), min(H - 1, int(y2))
            if x2i <= x1i + 4 or y2i <= y1i + 6:
                continue

            w = x2i - x1i
            h = y2i - y1i

            # tighter crop to reduce background noise
            shrink_x = int(0.08 * max(1, w))
            x1c = max(0, x1i + shrink_x)
            x2c = min(W - 1, x2i - shrink_x)

            # small lower padding for trousers/legs cues
            pad_down = int(0.06 * max(1, h))
            y2c = min(H - 1, y2i + pad_down)
            y1c = y1i

            if x2c <= x1c + 4 or y2c <= y1c + 6:
                continue

            crop_bgr = frame[y1c:y2c, x1c:x2c]
            q, mode = self.reid_manager.assess_detection(
                crop_bgr,
                x1i=x1i,
                y1i=y1i,
                x2i=x2i,
                y2i=y2i,
                conf=float(confs[i]),
                frame_h=H,
                frame_w=W,
            )
            quality[i] = float(q)
            feat_mode[i] = int(mode)

            if mode <= 0:
                continue

            crop_rgb = crop_bgr[:, :, ::-1]

            crops_bgr.append(crop_bgr)
            crops_rgb.append(crop_rgb)
            keep_idx.append(i)

        if not crops_rgb:
            return np.zeros((len(dets_xyxy), self.reid_feat_dim), dtype=np.float32), dets_xyxy, confs, quality, feat_mode

        os_feats = self.extractor(crops_rgb).astype(np.float32)
        if os_feats.ndim == 1:
            os_feats = os_feats[None, :]
        os_feats = l2_normalize_np(os_feats)

        attire_feats = np.stack([attire_descriptor(c) for c in crops_bgr], axis=0).astype(np.float32)
        attire_feats = l2_normalize_np(attire_feats)

        selected_boxes = dets_xyxy[keep_idx].astype(np.float32)
        shape_feats = np.stack(
            [body_shape_descriptor(b, frame_h=H, frame_w=W) for b in selected_boxes],
            axis=0,
        ).astype(np.float32)
        shape_feats = l2_normalize_np(shape_feats)

        fused = np.concatenate([
            self.F_OS * os_feats,
            self.F_ATTIRE * attire_feats,
            self.F_SHAPE * shape_feats,
        ], axis=1).astype(np.float32)
        fused = l2_normalize_np(fused)
        self.reid_feat_dim = int(fused.shape[1])

        full = np.zeros((len(dets_xyxy), fused.shape[1]), dtype=np.float32)
        for k, i in enumerate(keep_idx):
            full[int(i)] = fused[k]
        return full, dets_xyxy, confs, quality, feat_mode

    def _emit_zero_debug(
        self,
        frame_id: int,
        ts_sec: float,
        t,
        gid0_reason: str,
        online_tracks,
    ) -> None:
        if self._zero_debug_path is None:
            return
        x1, y1, x2, y2 = t.tlbr
        area_ratio = (x2 - x1) * (y2 - y1) / max(1.0, float(self._frame_h) * float(self._frame_w))
        y2_ratio = float(y2) / max(1.0, float(self._frame_h))
        det_conf = float(getattr(t, "last_det_conf", 0.0))
        mode = int(getattr(t, "last_reid_mode", 0))
        q = float(getattr(t, "last_feat_quality", 0.0))
        hits = int(getattr(t, "hits", 0))
        is_confirmed = int(bool(t.is_confirmed))

        best_iou = 0.0
        best_near_tid = -1
        best_near_gid = 0
        for ot in online_tracks:
            if ot is t:
                continue
            cur_gid = int(ot.global_id) if ot.global_id is not None else 0
            if cur_gid <= 0:
                continue
            ox1, oy1, ox2, oy2 = ot.tlbr
            ix1 = max(x1, ox1); iy1 = max(y1, oy1)
            ix2 = min(x2, ox2); iy2 = min(y2, oy2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter > 0:
                area_t = (x2 - x1) * (y2 - y1)
                area_o = (ox2 - ox1) * (oy2 - oy1)
                iou_v = inter / max(1e-9, area_t + area_o - inter)
                if iou_v > best_iou:
                    best_iou = iou_v
                    best_near_tid = int(getattr(ot, "track_id", -1))
                    best_near_gid = cur_gid

        self._zero_debug_rows.append({
            "frame_idx": int(frame_id),
            "ts_sec": round(float(ts_sec), 3),
            "track_id": int(getattr(t, "track_id", -1)),
            "x1": round(float(x1), 1),
            "y1": round(float(y1), 1),
            "x2": round(float(x2), 1),
            "y2": round(float(y2), 1),
            "det_conf": round(det_conf, 4),
            "area_ratio": round(area_ratio, 6),
            "y2_ratio": round(y2_ratio, 4),
            "reid_mode": mode,
            "feat_quality": round(q, 4),
            "hits": hits,
            "is_confirmed": is_confirmed,
            "nearest_pos_track_id": best_near_tid,
            "nearest_pos_gid": best_near_gid,
            "max_iou_with_pos_track": round(best_iou, 4),
            "gid0_reason": gid0_reason,
        })

    def flush_zero_debug(self) -> None:
        if self._zero_debug_path is None or not self._zero_debug_rows:
            return
        import csv as _csv
        import os as _os
        _os.makedirs(_os.path.dirname(_os.path.abspath(self._zero_debug_path)), exist_ok=True)
        fieldnames = [
            "frame_idx", "ts_sec", "track_id",
            "x1", "y1", "x2", "y2",
            "det_conf", "area_ratio", "y2_ratio",
            "reid_mode", "feat_quality", "hits", "is_confirmed",
            "nearest_pos_track_id", "nearest_pos_gid", "max_iou_with_pos_track",
            "gid0_reason",
        ]
        open_mode = "a" if self._zero_debug_written_header else "w"
        with open(self._zero_debug_path, open_mode, newline="") as f:
            w = _csv.DictWriter(f, fieldnames=fieldnames)
            if not self._zero_debug_written_header:
                w.writeheader()
                self._zero_debug_written_header = True
            w.writerows(self._zero_debug_rows)
        self._zero_debug_rows.clear()

    def update(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        ts_sec: Optional[float] = None,
        return_raw_tracks: bool = False,
        frame_is_bad: bool = False,
    ):
        self._frame_h, self._frame_w = frame.shape[:2]
        if ts_sec is None:
            ts_sec = float(frame_id)
        self._run_pose_for_frame(frame, frame_id=int(frame_id))

        if frame_is_bad:
            online_tracks = self.tracker.handle_bad_frame(frame_id)

            if return_raw_tracks:
                return online_tracks

            outputs = []
            for t in online_tracks:
                x1, y1, x2, y2 = t.tlbr
                gid = int(t.global_id) if t.global_id is not None else 0
                outputs.append([float(x1), float(y1), float(x2), float(y2), gid])
            return outputs

        dets_xyxy, det_confs = self._detect(frame)
        dets_xyxy, det_confs = self._filter_whole_body(
            frame,
            dets_xyxy,
            det_confs,
            frame_id=int(frame_id),
        )
        feats, dets_xyxy, det_confs, feat_quality, feat_mode = self._extract_feats(frame, dets_xyxy, det_confs)

        if len(dets_xyxy) > 0:
            dets_full = np.concatenate([dets_xyxy, det_confs[:, None]], axis=1).astype(np.float32)
        else:
            dets_full = np.empty((0, 5), dtype=np.float32)
            feats = np.empty((0, 0), dtype=np.float32)
            feat_quality = np.empty((0,), dtype=np.float32)
            feat_mode = np.empty((0,), dtype=np.int32)

        online_tracks = self.tracker.update(
            dets_full,
            feats,
            feat_quality=feat_quality,
            feat_mode=feat_mode,
            frame_id=frame_id,
        )

        if return_raw_tracks:
            return online_tracks

        outputs = []
        used_gids_this_frame: set[int] = set()
        resolved_tracks: list[tuple[object, int]] = []
        track_profile_cache: dict[int, Optional[dict[str, object]]] = {}
        reject_reason_counts: dict[str, int] = {}
        overlap_counts = self._overlap_counts(online_tracks)
        for t in online_tracks:
            tid = int(getattr(t, "track_id", -1))
            t.overlap_neighbors = int(overlap_counts.get(tid, 0))

        # Lost-track trust decay (zombie-ID prevention for long-missing identities).
        self._decay_identity_profile_trust(int(frame_id))

        # Prune stale ownership reservations.
        for gid, last_fr in list(self.gid_owner_last_frame.items()):
            if int(frame_id) - int(last_fr) > max(self.reentry_max_age, self.gid_owner_reserve_frames * 2):
                self.gid_owner_last_frame.pop(gid, None)
                self.gid_owner_track.pop(gid, None)

        def _get_profile(track) -> Optional[dict[str, object]]:
            tid = int(getattr(track, "track_id", -1))
            if tid not in track_profile_cache:
                track_profile_cache[tid] = self._extract_track_profile(frame, track)
            return track_profile_cache[tid]

        def _note_reject(reason: str, *, track=None, gid: Optional[int] = None, src: str = "", extra: str = "") -> None:
            key = str(reason)
            reject_reason_counts[key] = int(reject_reason_counts.get(key, 0)) + 1
            self._debug_block(
                frame_id=int(frame_id),
                reason=key,
                track=track,
                gid=gid,
                src=src,
                extra=extra,
            )

        def _force_unique_gid(track, gid_candidate: int) -> int:
            gid_candidate = int(gid_candidate)
            if gid_candidate <= 0:
                return gid_candidate
            if gid_candidate not in used_gids_this_frame:
                return gid_candidate
            # Hard same-frame uniqueness: keep strongest owner only.
            # Conflicting detections are forced to 0 (temporary) instead of reusing/stealing IDs.
            track.global_id = None
            track.global_id_locked = False
            track.id_state = "tentative"
            _note_reject(
                "same-frame-duplicate-suppress",
                track=track,
                gid=gid_candidate,
                src="conflict",
            )
            return 0

        def _pick_gid_with_compatibility(
            track,
            track_profile: Optional[dict[str, object]],
            forbidden_gids: Optional[set[int]] = None,
            allow_new_id: bool = False,
        ) -> tuple[int, float, str, bool]:
            if self.id_bank is None:
                return int(track.track_id), 1.0, "tracker", False
            overlap_n = int(getattr(track, "overlap_neighbors", 0))
            right_aisle = self._is_right_aisle_box(track.tlbr)
            crop_q = float(track_profile.get("quality", 0.0)) if track_profile is not None else 0.0
            if overlap_n > 0 and crop_q < max(self.min_crop_quality_for_reuse + 0.08 + (0.03 if right_aisle else 0.0), 0.68):
                _note_reject("reuse-rejected-overlap-freeze", track=track, src="overlap")
                if not allow_new_id:
                    return 0, -1.0, "none", False
            if overlap_n > 0 and track_profile is not None:
                if float(track_profile.get("border_trunc", 0.0)) > 0.52:
                    _note_reject("reuse-rejected-overlap-border-truncation", track=track, src="overlap")
                    if not allow_new_id:
                        return 0, -1.0, "none", False
                if float(track_profile.get("area_ratio", 0.0)) < max(self.min_human_area_ratio * 1.15, 0.0042):
                    _note_reject("reuse-rejected-overlap-small-crop", track=track, src="overlap")
                    if not allow_new_id:
                        return 0, -1.0, "none", False
            if right_aisle and track_profile is not None:
                if float(track_profile.get("border_trunc", 0.0)) > 0.45:
                    _note_reject("reuse-rejected-right-aisle-border", track=track, src="overlap")
                    if not allow_new_id:
                        return 0, -1.0, "none", False
                if float(track_profile.get("blur_var", 0.0)) < (self.reid_min_blur_var + 4.0):
                    _note_reject("reuse-rejected-right-aisle-blur", track=track, src="overlap")
                    if not allow_new_id:
                        return 0, -1.0, "none", False

            mem_gid, mem_score = self._short_memory_match(
                track,
                track_profile=track_profile,
                frame_id=frame_id,
                forbidden_gids=forbidden_gids,
            )
            if mem_gid > 0:
                return int(mem_gid), float(mem_score), "memory", False
            _note_reject("reuse-rejected-short-memory", track=track, src="memory")

            gallery_gid, gallery_score = self._reentry_match(
                track,
                track_profile=track_profile,
                frame_id=frame_id,
                forbidden_gids=forbidden_gids,
            )
            if gallery_gid > 0:
                return int(gallery_gid), float(gallery_score), "reentry", False
            _note_reject("reuse-rejected-long-gap", track=track, src="reentry")

            candidate: Optional[MatchCandidate] = self.id_bank.best_candidate(
                track.feature,
                forbidden_gids=forbidden_gids,
            )
            if candidate is not None:
                spatial = self._spatial_score_for_gid(candidate.gid, frame_id=frame_id, box_xyxy=track.tlbr)
                last_seen = int(self.reentry_last_frame.get(int(candidate.gid), int(frame_id)))
                age = max(0, int(frame_id) - last_seen)
                st = self.identity_profile_state.get(int(candidate.gid), None)
                trust = float(st.trust_score) if st is not None else 0.55
                frozen = bool(st is not None and int(frame_id) <= int(st.frozen_until))
                cloth, shape, size, zone_motion = self._profile_similarity_to_gid(
                    int(candidate.gid), track_profile, frame_id=frame_id
                )
                q = float(track_profile.get("quality", 0.0)) if track_profile is not None else 0.0
                shape_size = 0.60 * shape + 0.40 * size
                part_disagree = abs(max(0.0, min(1.0, cloth)) - max(0.0, min(1.0, shape_size)))
                strong_disagreement = bool(
                    candidate.sim_fused >= 0.84
                    and (
                        cloth < (0.56 if not right_aisle else 0.62)
                        or shape_size < (0.30 if not right_aisle else 0.34)
                        or zone_motion < (0.28 if not right_aisle else 0.32)
                    )
                )
                multi = (
                    0.35 * max(0.0, min(1.0, candidate.sim_fused))
                    + 0.30 * max(0.0, min(1.0, cloth))
                    + 0.13 * max(0.0, min(1.0, shape_size))
                    + 0.09 * max(0.0, min(1.0, zone_motion))
                    + 0.08 * max(0.0, min(1.0, spatial))
                    + 0.05 * max(0.0, min(1.0, q))
                )
                multi -= 0.22 * max(0.0, part_disagree - 0.13)
                if strong_disagreement:
                    multi -= 0.20
                multi += 0.03 * max(0.0, min(1.0, trust))
                if frozen:
                    multi -= 0.08
                if right_aisle:
                    multi -= 0.04
                req_margin = self.overlap_best_vs_second_margin if overlap_n > 0 else self.best_vs_second_margin
                if right_aisle:
                    req_margin = max(req_margin, self.overlap_best_vs_second_margin + 0.03)
                margin_ok = candidate.margin >= max(req_margin, getattr(self.id_bank, "margin", 0.06))
                profile_ok = (
                    (
                        cloth >= (0.62 if not right_aisle else 0.66)
                        and shape_size >= (0.32 if not right_aisle else 0.36)
                        and zone_motion >= (0.34 if not right_aisle else 0.38)
                    )
                    or (
                        candidate.sim_fused >= 0.93
                        and cloth >= (0.54 if not right_aisle else 0.58)
                        and zone_motion >= (0.32 if not right_aisle else 0.36)
                    )
                )
                age_ok = age <= self.reentry_max_age and (age <= self.gid_spatial_max_age or spatial >= self.min_spatial_consistency)
                overlap_boost = 0.09 if overlap_n > 0 else 0.0
                min_reuse = self.overlap_reuse_threshold if overlap_n > 0 else self.normal_reuse_threshold
                min_reuse = float(min(0.995, max(0.0, min_reuse)))
                spatial_ok = spatial >= self.min_spatial_consistency or (
                    age <= self.motion_max_gap and candidate.sim_fused >= (self.zombie_id_protection_threshold + 0.02)
                )
                if age > self.motion_max_gap and spatial < (self.min_spatial_consistency + 0.06) and cloth < 0.60:
                    spatial_ok = False
                quality_ok = q >= (
                    self.min_crop_quality_for_reuse
                    + (0.03 if overlap_n > 0 else 0.0)
                    + (0.03 if right_aisle else 0.0)
                )
                trust_ok = trust >= (0.28 if age > self.motion_max_gap else 0.24)
                zombie_ok = age <= self.motion_max_gap or (
                    candidate.sim_fused >= (self.zombie_id_protection_threshold + (0.02 if right_aisle else 0.01))
                    and cloth >= (0.56 if not right_aisle else 0.60)
                    and shape_size >= (0.32 if not right_aisle else 0.36)
                )
                part_ok = (
                    part_disagree <= (0.22 if not right_aisle else 0.18)
                    or (
                        candidate.sim_fused >= 0.93
                        and cloth >= (0.54 if not right_aisle else 0.58)
                        and shape_size >= (0.34 if not right_aisle else 0.38)
                    )
                )
                overlap_ambiguous_reuse = False
                if overlap_n > 0:
                    # In congestion, prefer fragmentation over ID theft.
                    # Reuse is allowed only under clearly strong and consistent evidence.
                    strict_overlap_margin = max(self.overlap_best_vs_second_margin + (0.12 if right_aisle else 0.10), 0.30)
                    strict_overlap_sim = min(0.995, max(self.overlap_reuse_threshold + (0.06 if right_aisle else 0.04), 0.94))
                    strict_overlap_multi = min(
                        0.995,
                        max(
                            self.overlap_reuse_threshold + (0.14 if right_aisle else 0.11),
                            self.lock_score_thresh + (0.18 if right_aisle else 0.15),
                        ),
                    )
                    strict_part_disagree = 0.14 if right_aisle else 0.16
                    overlap_ambiguous_reuse = bool(
                        candidate.margin < strict_overlap_margin
                        or candidate.sim_fused < strict_overlap_sim
                        or multi < strict_overlap_multi
                        or part_disagree > strict_part_disagree
                    )
                candidate_ok = (
                    (not frozen)
                    and (not overlap_ambiguous_reuse)
                    and (not strong_disagreement)
                    and multi >= max(
                        min_reuse + overlap_boost + (0.03 if right_aisle else 0.0),
                        self.lock_score_thresh + overlap_boost + (0.03 if right_aisle else 0.0),
                    )
                    and candidate.sim_fused >= max(
                        min_reuse - 0.01 + (0.03 if right_aisle else 0.0),
                        self.gid_reuse_with_spatial_thresh + (0.02 if right_aisle else 0.0) + 0.7 * overlap_boost,
                    )
                    and margin_ok
                    and profile_ok
                    and age_ok
                    and spatial_ok
                    and quality_ok
                    and trust_ok
                    and zombie_ok
                    and part_ok
                )
                if candidate_ok:
                    return int(candidate.gid), float(multi), "bank", False
                if frozen:
                    _note_reject("reuse-rejected-overlap-freeze", track=track, gid=int(candidate.gid), src="bank")
                if not margin_ok:
                    _note_reject("reuse-rejected-second-best-margin", track=track, gid=int(candidate.gid), src="bank")
                if not quality_ok:
                    _note_reject("reuse-rejected-low-crop-quality", track=track, gid=int(candidate.gid), src="bank")
                if not part_ok:
                    _note_reject("reuse-rejected-low-part-agreement", track=track, gid=int(candidate.gid), src="bank")
                if not trust_ok or not zombie_ok:
                    _note_reject("reuse-rejected-zombie-trust-decay", track=track, gid=int(candidate.gid), src="bank")
                if not spatial_ok:
                    _note_reject("reuse-rejected-spatial-inconsistent", track=track, gid=int(candidate.gid), src="bank")
                if not profile_ok:
                    _note_reject("reuse-rejected-profile-separation", track=track, gid=int(candidate.gid), src="bank")
                if overlap_ambiguous_reuse:
                    _note_reject("reuse-rejected-overlap-ambiguous", track=track, gid=int(candidate.gid), src="bank")
                if strong_disagreement:
                    _note_reject("reuse-rejected-feature-disagreement", track=track, gid=int(candidate.gid), src="bank")
                # Conservative policy: block unsafe reuse and prefer creating a new ID.
                # Reasons include low margin, weak spatial consistency, low trust, overlap freeze, and zombie-risk.

            if not allow_new_id:
                _note_reject("new-id-not-allowed-yet", track=track, src="new")
                return 0, -1.0, "none", False

            q = float(getattr(track, "last_feat_quality", 0.0))
            mode = int(getattr(track, "last_reid_mode", 0))
            hits = int(getattr(track, "hits", 0))
            min_new_q = self.reid_new_id_min_quality - (0.03 if overlap_n > 0 else 0.0)
            min_new_hits = self.reid_new_id_min_hits - (1 if overlap_n > 0 else 0)
            allow_mode2 = (mode >= 2 and q >= min_new_q and hits >= min_new_hits)
            allow_mode1 = (mode == 1 and q >= (min_new_q + 0.02) and hits >= (min_new_hits + 2))
            if not (allow_mode2 or allow_mode1):
                _note_reject("new-id-rejected-weak-crop", track=track, src="new")
                return 0, -1.0, "none", False

            # Keep one safety brake: if an existing ID is already very strong and unambiguous, avoid unnecessary fragmentation.
            clear_reuse_sim = 0.94 if overlap_n > 0 else 0.90
            clear_reuse_margin = self.best_vs_second_margin + (0.08 if overlap_n > 0 else 0.04)
            if candidate is not None and candidate.sim_fused >= clear_reuse_sim and candidate.margin >= clear_reuse_margin:
                _note_reject("new-id-rejected-clear-reuse-exists", track=track, gid=int(candidate.gid), src="new")
                return 0, -1.0, "none", False

            # Prevent duplicate/new IDs when track is likely a fragment of an already active person.
            if self.id_bank is not None:
                for ot in online_tracks:
                    if ot is track:
                        continue
                    og = int(getattr(ot, "global_id", 0) or 0)
                    if og <= 0:
                        continue
                    if not bool(getattr(ot, "global_id_locked", False)):
                        continue
                    sim_active = float(self.id_bank.similarity_to_gid(getattr(track, "feature", None), og))
                    if sim_active < self.active_similar_block_thresh:
                        continue
                    ax1, ay1, ax2, ay2 = [float(v) for v in getattr(track, "tlbr", [0, 0, 0, 0])]
                    bx1, by1, bx2, by2 = [float(v) for v in getattr(ot, "tlbr", [0, 0, 0, 0])]
                    xA = max(ax1, bx1)
                    yA = max(ay1, by1)
                    xB = min(ax2, bx2)
                    yB = min(ay2, by2)
                    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
                    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
                    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
                    iou_to_active = inter / (area_a + area_b - inter + 1e-6)
                    cloth_a, shape_a, size_a, zone_a = self._profile_similarity_to_gid(og, track_profile, frame_id=frame_id)
                    shape_size_a = 0.60 * shape_a + 0.40 * size_a
                    if overlap_n > 0:
                        # During overlap we only block new-ID creation when existing-owner evidence is very strong.
                        same_owner_like = (
                            iou_to_active >= 0.20
                            and sim_active >= (self.active_similar_block_thresh + 0.05)
                            and (cloth_a >= 0.60 or shape_size_a >= 0.44 or zone_a >= 0.56)
                        )
                    else:
                        same_owner_like = bool(iou_to_active >= 0.10 or cloth_a >= 0.52 or shape_size_a >= 0.36 or zone_a >= 0.48)
                    if same_owner_like:
                        _note_reject("new-id-rejected-active-owner-protection", track=track, gid=int(og), src="new")
                        return 0, -1.0, "none", False

            det_c = float(getattr(track, "last_det_conf", 0.0))
            new_score = 0.42 * q + 0.24 * max(0.0, min(1.0, det_c)) + 0.20 * min(1.0, hits / 10.0) + 0.14 * (1.0 if mode >= 2 else 0.8)
            # Prefer new temporary IDs over wrong reuse in overlap.
            # Lower the new-ID score threshold slightly during overlap windows.
            new_id_thresh = self.new_id_score_thresh - (0.02 if overlap_n > 0 else 0.0)
            if new_score < max(0.0, new_id_thresh):
                _note_reject("new-id-rejected-score-threshold", track=track, src="new")
                return 0, float(new_score), "new", False
            return 0, float(new_score), "new", True

        def _proposal_score_for_gid(track, gid: int) -> float:
            sim = -1.0
            spatial = 0.0
            if self.id_bank is not None:
                sim = self.id_bank.similarity_to_gid(getattr(track, "feature", None), int(gid))
                spatial = self._spatial_score_for_gid(int(gid), frame_id=frame_id, box_xyxy=track.tlbr)
            sim = max(sim, 0.0)
            lock_bonus = 0.22 if bool(getattr(track, "global_id_locked", False)) and int(getattr(track, "global_id", -1)) == int(gid) else 0.0
            hit_bonus = min(0.10, 0.012 * float(max(0, int(getattr(track, "hits", 0)) - 1)))
            return float(0.72 * sim + 0.24 * spatial + lock_bonus + hit_bonus)

        def _is_frozen(track) -> bool:
            return bool(getattr(track, "global_id_locked", False))

        def _allow_gid_for_track(track, gid: int, score: float, src: str) -> bool:
            gid = int(gid)
            if gid <= 0:
                return True
            st = self.identity_profile_state.get(int(gid), None)
            if st is not None and int(frame_id) <= int(st.frozen_until):
                # Owner protection during overlap/recovery window.
                _note_reject("reuse-rejected-overlap-freeze", track=track, gid=int(gid), src=src)
                return False
            trust = float(st.trust_score) if st is not None else 0.55
            if trust < 0.16 and float(score) < self.zombie_id_protection_threshold:
                # Reject zombie-ID takeovers unless evidence is extremely strong.
                _note_reject("reuse-rejected-zombie-trust-decay", track=track, gid=int(gid), src=src)
                return False
            owner_tid = self.gid_owner_track.get(gid, None)
            if owner_tid is None:
                return True
            tr_tid = int(getattr(track, "track_id", -1))
            if int(owner_tid) == tr_tid:
                return True
            age = int(frame_id) - int(self.gid_owner_last_frame.get(gid, -10**9))
            if age > self.gid_owner_reserve_frames:
                return True

            q = float(getattr(track, "last_feat_quality", 0.0))
            mode = int(getattr(track, "last_reid_mode", 0))
            hits = int(getattr(track, "hits", 0))
            overlap_n = int(getattr(track, "overlap_neighbors", 0))
            right_aisle = self._is_right_aisle_box(track.tlbr)

            strong_reentry = (
                src == "reentry"
                and float(score) >= (
                    self.gid_owner_override_score
                    + (0.20 if overlap_n > 0 else 0.10)
                    + (0.05 if right_aisle else 0.0)
                )
                and mode >= 2
                and q >= (self.reid_min_quality_for_bank + 0.06 + (0.03 if right_aisle else 0.0))
                and hits >= (self.reid_min_lock_hits + (4 if overlap_n > 0 else 3) + (1 if right_aisle else 0))
            )
            very_strong_bank = (
                src == "bank"
                and float(score) >= (
                    self.gid_owner_override_score
                    + (0.26 if overlap_n > 0 else 0.16)
                    + (0.05 if right_aisle else 0.0)
                )
                and mode >= 2
                and q >= (self.reid_min_quality_for_bank + 0.08 + (0.03 if right_aisle else 0.0))
                and hits >= (self.reid_min_lock_hits + (5 if overlap_n > 0 else 4) + (1 if right_aisle else 0))
            )
            strong_memory = (
                src == "memory"
                and float(score) >= (
                    self.gid_owner_override_score
                    + (0.16 if overlap_n > 0 else 0.07)
                    + (0.04 if right_aisle else 0.0)
                )
                and mode >= 2
                and q >= (self.reid_min_quality_for_bank + 0.05 + (0.03 if right_aisle else 0.0))
                and hits >= (self.reid_min_lock_hits + (4 if overlap_n > 0 else 3) + (1 if right_aisle else 0))
            )
            ok = bool(strong_reentry or very_strong_bank or strong_memory)
            if (overlap_n > 0 or right_aisle) and src != "memory":
                ok = bool(strong_reentry or very_strong_bank)
            if not ok:
                _note_reject("reuse-rejected-owner-protection", track=track, gid=int(gid), src=src)
            return ok

        def _resolve_gid_conflicts() -> None:
            if len(resolved_tracks) < 2:
                return

            for _ in range(3):
                changed = False
                gid_to_indices: dict[int, list[int]] = {}
                for idx, (_, gid) in enumerate(resolved_tracks):
                    gid_to_indices.setdefault(int(gid), []).append(idx)

                conflict_gids = [gid for gid, idxs in gid_to_indices.items() if gid > 0 and len(idxs) > 1]
                if not conflict_gids:
                    break

                for gid in conflict_gids:
                    idxs = gid_to_indices.get(gid, [])
                    if len(idxs) <= 1:
                        continue

                    locked_idxs = [
                        idx for idx in idxs
                        if bool(getattr(resolved_tracks[idx][0], "global_id_locked", False))
                        and int(getattr(resolved_tracks[idx][0], "global_id", -1)) == int(gid)
                    ]
                    best_pool = locked_idxs if locked_idxs else idxs
                    best_idx = max(best_pool, key=lambda idx: _proposal_score_for_gid(resolved_tracks[idx][0], gid))
                    all_now = {int(g) for _, g in resolved_tracks}

                    for idx in idxs:
                        if idx == best_idx:
                            continue
                        tr, old_gid = resolved_tracks[idx]
                        if bool(getattr(tr, "global_id_locked", False)) and int(getattr(tr, "global_id", -1)) == int(old_gid):
                            tr.global_id_locked = False
                        # Hard same-frame uniqueness: conflicting claimants are forced
                        # to temporary 0 and must be re-assigned safely in later frames.
                        tr.global_id = None
                        tr.id_state = "tentative"
                        tr.pending_gid = -1
                        tr.pending_hits = max(1, int(getattr(tr, "pending_hits", 0)))
                        resolved_tracks[idx] = (tr, 0)
                        _note_reject("same-frame-duplicate-suppress", track=tr, gid=int(gid), src="conflict")
                        changed = True
                        all_now.add(0)

                if not changed:
                    break

        def _correct_pairwise_swaps() -> None:
            if self.id_bank is None or len(resolved_tracks) < 2:
                return

            for _ in range(2):
                swapped_any = False
                n = len(resolved_tracks)
                for i in range(n):
                    ti, gi = resolved_tracks[i]
                    fi = getattr(ti, "feature", None)
                    if fi is None:
                        continue

                    for j in range(i + 1, n):
                        tj, gj = resolved_tracks[j]
                        if gi == gj:
                            continue
                        if gi <= 0 or gj <= 0:
                            continue
                        if _is_frozen(ti) or _is_frozen(tj):
                            continue

                        fj = getattr(tj, "feature", None)
                        if fj is None:
                            continue

                        s_ii = self.id_bank.similarity_to_gid(fi, gi)
                        s_jj = self.id_bank.similarity_to_gid(fj, gj)
                        s_ij = self.id_bank.similarity_to_gid(fi, gj)
                        s_ji = self.id_bank.similarity_to_gid(fj, gi)
                        if min(s_ii, s_jj, s_ij, s_ji) < -0.5:
                            continue

                        own = s_ii + s_jj
                        cross = s_ij + s_ji

                        # If cross assignment explains both tracks much better, revert likely overlap swap.
                        if cross > own + 0.12 and min(s_ij, s_ji) > 0.42:
                            resolved_tracks[i] = (ti, int(gj))
                            resolved_tracks[j] = (tj, int(gi))
                            ti.global_id = int(gj)
                            tj.global_id = int(gi)
                            swapped_any = True

                if not swapped_any:
                    break

        locked_active_gids = {
            int(getattr(tt, "global_id"))
            for tt in online_tracks
            if getattr(tt, "global_id", None) is not None
            and bool(getattr(tt, "global_id_locked", False))
            and int(getattr(tt, "global_id")) > 0
        }

        for t in online_tracks:
            profile = _get_profile(t)
            human_ok, human_reason = self._is_track_human_candidate(t, profile)
            if not human_ok:
                _note_reject("rejected-non-human-ghost-detection", track=t, src="human-gate", extra=str(human_reason))
                t.global_id = None
                t.global_id_locked = False
                t.id_state = "unassigned"
                resolved_tracks.append((t, 0))
                continue

            if self.id_bank is None:
                gid = int(t.track_id)
                resolved_tracks.append((t, gid))
                continue

            mode = int(getattr(t, "last_reid_mode", 0))
            q = float(getattr(t, "last_feat_quality", 0.0))
            can_assign_global = (
                t.is_confirmed
                and t.hits >= self.min_confirmed_hits_for_gid
                and mode >= 1
                and q >= self.reid_min_quality_for_bank
                and float(getattr(t, "last_det_conf", 0.0)) >= self.min_person_conf_for_identity
            )
            if can_assign_global and mode == 1:
                can_assign_global = t.hits >= (self.reid_min_lock_hits + 1)

            current_gid = int(t.global_id) if t.global_id is not None else 0
            if current_gid <= 0:
                gid = 0
                # Overlap hold: keep recently stable ID for a short window during crowded
                # interactions to reduce ID flips caused by temporary occlusion/mix-ups.
                hold_gid = int(getattr(t, "id_hold_gid", 0) or 0)
                hold_until = int(getattr(t, "id_hold_until", -1))
                if hold_gid > 0 and int(frame_id) <= hold_until and hold_gid not in used_gids_this_frame:
                    gid = int(hold_gid)
                    t.global_id = int(gid)
                    t.global_id_locked = True
                    t.id_state = "locked"
                    t.locked_since_frame = int(frame_id)

                if can_assign_global and gid <= 0:
                    forbidden = set(locked_active_gids)
                    cand_gid, cand_score, cand_src, can_new = _pick_gid_with_compatibility(
                        t,
                        track_profile=profile,
                        forbidden_gids=forbidden,
                        allow_new_id=False,
                    )
                    if cand_gid <= 0:
                        allow_new_id = (
                            (mode >= 2 and q >= self.reid_new_id_min_quality and t.hits >= self.reid_new_id_min_hits)
                            or (mode == 1 and q >= (self.reid_new_id_min_quality + 0.02) and t.hits >= (self.reid_new_id_min_hits + 2))
                        )
                        cand_gid, cand_score, cand_src, can_new = _pick_gid_with_compatibility(
                            t,
                            track_profile=profile,
                            forbidden_gids=forbidden,
                            allow_new_id=allow_new_id,
                        )
                    if cand_gid > 0 and not _allow_gid_for_track(t, cand_gid, cand_score, cand_src):
                        _note_reject("reuse-rejected-owner-protection", track=t, gid=int(cand_gid), src=str(cand_src))
                        cand_gid, cand_score, cand_src = 0, -1.0, "none"

                    if cand_gid > 0:
                        if int(getattr(t, "pending_gid", 0) or 0) == int(cand_gid):
                            t.pending_hits = int(getattr(t, "pending_hits", 0)) + 1
                            t.pending_score = 0.72 * float(getattr(t, "pending_score", 0.0)) + 0.28 * float(cand_score)
                        else:
                            t.pending_gid = int(cand_gid)
                            t.pending_hits = 1
                            t.pending_score = float(cand_score)
                        t.id_state = "tentative"

                        if t.pending_hits >= self.lock_confirm_frames and t.pending_score >= self.lock_score_thresh:
                            gid = int(cand_gid)
                            t.pending_gid = None
                            t.pending_hits = 0
                            t.pending_score = 0.0
                    elif cand_src == "new":
                        if int(getattr(t, "pending_gid", 0) or 0) == -1:
                            t.pending_hits = int(getattr(t, "pending_hits", 0)) + 1
                            t.pending_score = 0.72 * float(getattr(t, "pending_score", 0.0)) + 0.28 * float(cand_score)
                        else:
                            t.pending_gid = -1
                            t.pending_hits = 1
                            t.pending_score = float(cand_score)
                        t.id_state = "tentative"

                        if can_new and t.pending_hits >= self.new_id_confirm_frames and t.pending_score >= self.new_id_score_thresh:
                            if hasattr(self.id_bank, "new_identity"):
                                gid = int(self.id_bank.new_identity(getattr(t, "feature", None), ts_sec=ts_sec))
                            else:
                                gid = int(self.id_bank.assign(getattr(t, "feature", None), ts_sec=ts_sec))
                            t.pending_gid = None
                            t.pending_hits = 0
                            t.pending_score = 0.0
                    else:
                        t.pending_hits = max(0, int(getattr(t, "pending_hits", 0)) - 1)
                        if t.pending_hits == 0:
                            t.pending_gid = None
                            t.pending_score = 0.0
                        t.id_state = "unassigned"

                    if gid > 0:
                        t.global_id = int(gid)
                        t.global_id_locked = True
                        t.id_state = "locked"
                        t.locked_since_frame = int(frame_id)
                else:
                    gid = 0
                    t.pending_hits = max(0, int(getattr(t, "pending_hits", 0)) - 1)
                    if t.pending_hits == 0:
                        t.pending_gid = None
                        t.pending_score = 0.0
                    t.id_state = "unassigned"
            else:
                gid = int(current_gid)
                if t.is_confirmed and t.hits >= max(self.min_confirmed_hits_for_gid, self.reid_min_lock_hits):
                    t.global_id_locked = True
                if bool(getattr(t, "global_id_locked", False)):
                    t.id_state = "locked"
                else:
                    t.id_state = "tentative"

                overlap_n = int(overlap_counts.get(int(getattr(t, "track_id", -1)), 0))
                right_aisle = self._is_right_aisle_box(t.tlbr)
                if overlap_n > 0:
                    t.id_hold_gid = int(gid)
                    hold_frames = int(self.overlap_hold_frames + (self.right_aisle_hold_extra_frames if right_aisle else 0))
                    t.id_hold_until = max(int(getattr(t, "id_hold_until", -1)), int(frame_id) + hold_frames)
                    t.id_drift_hits = 0
                    st = self._profile_state(int(gid))
                    # Overlap lock: protect current owner and freeze profile updates temporarily.
                    freeze_frames = int(self.freeze_duration_after_overlap + (self.right_aisle_hold_extra_frames if right_aisle else 0))
                    st.frozen_until = max(int(st.frozen_until), int(frame_id) + freeze_frames)
                    st.recovery_until = max(
                        int(st.recovery_until),
                        int(frame_id)
                        + freeze_frames
                        + int(self.overlap_recovery_extra_frames),
                    )
                else:
                    # Blur/partial-occlusion continuity: an active locked track
                    # that momentarily produces a low-quality or very blurry
                    # crop should not die. Extend id_hold_until conservatively
                    # so the tracker keeps the identity on the same person
                    # instead of handing it off once the crop recovers.
                    cur_prof = _get_profile(t)
                    low_quality_frame = False
                    if cur_prof is not None:
                        q_now = float(cur_prof.get("quality", 0.0))
                        blur_now = float(cur_prof.get("blur_var", 0.0))
                        border_trunc_now = float(cur_prof.get("border_trunc", 0.0))
                        if (
                            q_now < self.min_crop_quality_for_reuse
                            or blur_now < self.reid_min_blur_var
                            or border_trunc_now > 0.45
                        ):
                            low_quality_frame = True
                    if low_quality_frame and gid > 0 and bool(getattr(t, "global_id_locked", False)):
                        t.id_hold_gid = int(gid)
                        t.id_hold_until = max(
                            int(getattr(t, "id_hold_until", -1)),
                            int(frame_id)
                            + max(
                                12,
                                self.overlap_hold_frames // 2,
                            )
                            + (self.right_aisle_hold_extra_frames // 2 if right_aisle else 0),
                        )

                # Identity-drift guard: if current locked ID becomes repeatedly incompatible
                # with appearance/profile, release it and force tentative reassignment.
                if self.id_bank is not None and gid > 0 and t.is_confirmed:
                    mode_now = int(getattr(t, "last_reid_mode", 0))
                    q_now = float(getattr(t, "last_feat_quality", 0.0))
                    y2_ratio = float(max(0.0, min(1.0, float(t.tlbr[3]) / max(1.0, float(self._frame_h)))))
                    drift_guard_allowed = (
                        mode_now >= self.drift_guard_min_mode
                        and q_now >= self.drift_guard_min_quality
                        and y2_ratio >= self.reid_cautious_y2_ratio
                        and int(overlap_counts.get(int(getattr(t, "track_id", -1)), 0)) <= 0
                    )
                    if not drift_guard_allowed:
                        t.id_drift_hits = max(0, int(getattr(t, "id_drift_hits", 0)) - 1)
                        sim_gid = 0.0
                        cloth = shape = size = zone_motion = 0.0
                    else:
                        sim_gid = float(self.id_bank.similarity_to_gid(getattr(t, "feature", None), gid))
                        cloth, shape, size, zone_motion = self._profile_similarity_to_gid(
                            gid, profile, frame_id=frame_id
                        )
                        shape_size = 0.60 * shape + 0.40 * size
                        drift_score = (
                            0.45 * max(0.0, min(1.0, sim_gid))
                            + 0.35 * max(0.0, min(1.0, cloth))
                            + 0.10 * max(0.0, min(1.0, shape_size))
                            + 0.10 * max(0.0, min(1.0, zone_motion))
                        )
                        hard_bad = (
                            sim_gid < 0.36
                            and cloth < 0.26
                            and shape_size < 0.18
                            and zone_motion < 0.24
                        )
                        soft_bad = drift_score < 0.34
                        if hard_bad or soft_bad:
                            t.id_drift_hits = int(getattr(t, "id_drift_hits", 0)) + 1
                        else:
                            t.id_drift_hits = max(0, int(getattr(t, "id_drift_hits", 0)) - 1)

                        if int(getattr(t, "id_drift_hits", 0)) >= self.drift_release_hits:
                            hold_gid = int(getattr(t, "id_hold_gid", 0) or 0)
                            hold_until = int(getattr(t, "id_hold_until", -1))
                            if hold_gid > 0 and int(frame_id) <= hold_until and hold_gid == int(gid) and overlap_n > 0:
                                t.id_drift_hits = 0
                            else:
                                # Drift-release: the track has been mis-identified
                                # against its current gid. Release ownership AND
                                # clear all pending-state so next frame's matcher
                                # treats this track as fresh. Previously we left
                                # pending_gid/pending_hits/pending_score partially
                                # populated, which caused a "ghost lock" where the
                                # same wrong gid kept being proposed again.
                                released_gid = int(gid)
                                gid = 0
                                t.global_id = None
                                t.global_id_locked = False
                                t.id_state = "unassigned"
                                t.pending_gid = -1
                                t.pending_hits = 0
                                t.pending_score = 0.0
                                t.id_drift_hits = 0
                                # Release ownership bookkeeping for the released
                                # gid so a different track can legitimately claim
                                # it again via re-entry match. This prevents a
                                # drift-released track from continuing to "own"
                                # the gid through id_hold/owner_track metadata.
                                t.id_hold_gid = 0
                                t.id_hold_until = -1
                                if released_gid > 0 and self.gid_owner_track.get(released_gid) == int(getattr(t, "track_id", -1)):
                                    self.gid_owner_track.pop(released_gid, None)

            if self.id_bank is not None:
                t.global_id = int(gid) if int(gid) > 0 else None
                if int(gid) <= 0:
                    t.global_id_locked = False
                    if t.id_state == "locked":
                        t.id_state = "unassigned"

            if int(gid) == 0 and self._zero_debug_path is not None:
                _zmode = int(getattr(t, "last_reid_mode", 0))
                _zq    = float(getattr(t, "last_feat_quality", 0.0))
                _zdc   = float(getattr(t, "last_det_conf", 0.0))
                if not t.is_confirmed:
                    _zr = "gate:not_confirmed"
                elif t.hits < self.min_confirmed_hits_for_gid:
                    _zr = f"gate:hits_{t.hits}_lt_{self.min_confirmed_hits_for_gid}"
                elif _zmode == 0:
                    _zr = "gate:mode0_far_zone"
                elif _zq < self.reid_min_quality_for_bank:
                    _zr = f"gate:quality_{_zq:.3f}_lt_{self.reid_min_quality_for_bank}"
                elif _zdc < self.min_person_conf_for_identity:
                    _zr = f"gate:det_conf_{_zdc:.3f}_lt_{self.min_person_conf_for_identity}"
                elif str(getattr(t, "id_state", "")) == "tentative":
                    _zr = "pending:accumulating_confirmation"
                else:
                    _zr = "pending:no_match_or_ambiguous"
                self._emit_zero_debug(
                    frame_id=frame_id,
                    ts_sec=float(ts_sec),
                    t=t,
                    gid0_reason=_zr,
                    online_tracks=online_tracks,
                )

            resolved_tracks.append((t, int(gid)))

        if bool(getattr(self, "enable_pairwise_swap_correction", False)):
            _correct_pairwise_swaps()
        _resolve_gid_conflicts()

        # Finalize outputs and refresh memory bank with corrected IDs.
        used_gids_this_frame = set()
        resolved_tracks.sort(
            key=lambda p: (
                1 if bool(getattr(p[0], "global_id_locked", False)) else 0,
                1 if int(getattr(p[0], "last_reid_mode", 0)) <= 1 else 0,
                int(getattr(p[0], "hits", 0)),
            ),
            reverse=True,
        )
        for t, gid in resolved_tracks:
            x1, y1, x2, y2 = t.tlbr
            gid = _force_unique_gid(t, int(gid))
            if gid > 0:
                used_gids_this_frame.add(gid)
            overlap_n = int(overlap_counts.get(int(getattr(t, "track_id", -1)), 0))

            if self.id_bank is not None:
                t.global_id = int(gid)
                if gid > 0 and t.is_confirmed and t.feature is not None:
                    q = float(getattr(t, "last_feat_quality", 0.0))
                    det_c = float(getattr(t, "last_det_conf", 0.0))
                    mode = int(getattr(t, "last_reid_mode", 0))
                    y2_ratio = float(max(0.0, min(1.0, float(t.tlbr[3]) / max(1.0, float(self._frame_h)))))
                    if (
                        mode >= 2
                        and q >= self.reid_min_quality_for_bank
                        and det_c >= (self.det_conf + 0.06)
                        and overlap_n <= 0
                        and y2_ratio >= self.reid_cautious_y2_ratio
                    ):
                        cand = self.id_bank.best_candidate(t.feature, forbidden_gids=None)
                        if cand is not None and int(cand.gid) == int(gid):
                            sim_ok = cand.sim_fused >= max(0.74, self.gid_reuse_with_spatial_thresh + 0.06)
                            margin_ok = cand.margin >= max(self.bank_update_min_margin, getattr(self.id_bank, "margin", 0.06))
                            if sim_ok and margin_ok:
                                self.id_bank.observe(gid=gid, feat=t.feature, ts_sec=ts_sec, quality=q)
                if gid > 0:
                    t.id_hold_gid = int(gid)
                    hold_bonus = self.right_aisle_hold_extra_frames // 2 if self._is_right_aisle_box(t.tlbr) else 0
                    t.id_hold_until = max(
                        int(getattr(t, "id_hold_until", -1)),
                        int(frame_id) + max(10, self.overlap_hold_frames // 2) + int(hold_bonus),
                    )
                    self.gid_owner_track[int(gid)] = int(getattr(t, "track_id", -1))
                    self.gid_owner_last_frame[int(gid)] = int(frame_id)
                    self.reentry_last_frame[int(gid)] = int(frame_id)
                    self.reentry_last_box[int(gid)] = (float(x1), float(y1), float(x2), float(y2))
                    self.reentry_last_zone[int(gid)] = self._zone_of_box(t.tlbr)
                    self._update_gid_history(gid=gid, frame_id=frame_id, box_xyxy=t.tlbr)
                    profile_now = _get_profile(t)
                    self._record_profile_sample(
                        int(gid),
                        profile_now,
                        track=t,
                        frame_id=int(frame_id),
                        overlap_ambiguous=(overlap_n > 0),
                    )
                    self._record_reentry_sample(
                        t,
                        frame_id=frame_id,
                        profile=profile_now,
                        overlap_ambiguous=(overlap_n > 0),
                    )
                    t.id_state = "locked" if bool(getattr(t, "global_id_locked", False)) else str(getattr(t, "id_state", "tentative"))
                else:
                    t.global_id = None
                    if str(getattr(t, "id_state", "unassigned")) != "tentative":
                        t.id_state = "unassigned"

            outputs.append([float(x1), float(y1), float(x2), float(y2), int(gid)])

        # Track exited-memory IDs (locked IDs that are currently not active but still within re-entry window).
        active_positive = {int(g) for _, g in resolved_tracks if int(g) > 0}
        exited: set[int] = set()
        for gid, last_f in list(self.reentry_last_frame.items()):
            age = int(frame_id) - int(last_f)
            g = int(gid)
            if age <= 0:
                continue
            if age <= self.reentry_max_age and g not in active_positive:
                exited.add(g)
        self.exited_memory_ids = exited

        if bool(self.debug_reid_decisions) and reject_reason_counts:
            parts = ", ".join(
                f"{k}:{v}"
                for k, v in sorted(reject_reason_counts.items(), key=lambda kv: (-kv[1], kv[0]))
            )
            print(f"[REID-BLOCK] frame={int(frame_id)} summary {{{parts}}}")

        self.flush_zero_debug()
        return outputs
