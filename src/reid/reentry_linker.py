from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import csv
import json
import math

import cv2
import numpy as np

from src.reid.extractor import ReidExtractor
from src.trackers.bot_sort import attire_descriptor


def _l2(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = v.astype(np.float32)
    n = float(np.linalg.norm(v)) + eps
    return v / n


def _cos(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return -1.0
    if a.shape != b.shape:
        return -1.0
    aa = _l2(a)
    bb = _l2(b)
    return float(np.dot(aa, bb))


def _iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    xA = max(float(a[0]), float(b[0]))
    yA = max(float(a[1]), float(b[1]))
    xB = min(float(a[2]), float(b[2]))
    yB = min(float(a[3]), float(b[3]))
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    areaA = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    areaB = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    denom = areaA + areaB - inter + 1e-6
    return float(inter / denom) if denom > 0 else 0.0


def _zone_of_point(cx: float, cy: float, frame_w: int, frame_h: int) -> str:
    w = max(1.0, float(frame_w))
    h = max(1.0, float(frame_h))
    if cx <= 0.18 * w:
        return "left"
    if cx >= 0.82 * w:
        return "right"
    if cy <= 0.20 * h:
        return "top"
    if cy >= 0.82 * h:
        return "bottom"
    return "center"


def _side_score(exit_side: str, entry_side: str) -> float:
    if exit_side == entry_side:
        return 1.0
    if exit_side == "center" or entry_side == "center":
        return 0.62
    if {exit_side, entry_side} == {"left", "right"}:
        return 0.28
    if {exit_side, entry_side} == {"top", "bottom"}:
        return 0.28
    return 0.40


def _region_stats(region_bgr: np.ndarray) -> np.ndarray:
    if region_bgr is None or region_bgr.size == 0:
        return np.zeros((6,), dtype=np.float32)
    hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    mean = hsv.mean(axis=0)
    std = hsv.std(axis=0)
    vec = np.array(
        [mean[0] / 180.0, mean[1] / 255.0, mean[2] / 255.0, std[0] / 90.0, std[1] / 128.0, std[2] / 128.0],
        dtype=np.float32,
    )
    return _l2(vec)


def _upper_lower_support(crop_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if crop_bgr is None or crop_bgr.size == 0:
        z = np.zeros((6,), dtype=np.float32)
        return z, z
    h = crop_bgr.shape[0]
    if h < 12:
        z = np.zeros((6,), dtype=np.float32)
        return z, z
    y2 = int(0.45 * h)
    y3 = int(0.72 * h)
    upper = crop_bgr[:y2, :]
    lower = crop_bgr[y3:, :]
    return _region_stats(upper), _region_stats(lower)


@dataclass
class ReentryConfig:
    segment_gap_frames: int = 2
    max_reentry_gap_frames: int = 1600
    min_reentry_gap_frames: int = 4
    min_side_score: float = 0.22
    min_size_ratio: float = 0.32
    max_size_ratio: float = 2.20
    min_aspect_ratio: float = 0.30
    max_aspect_ratio: float = 2.20
    min_area_ratio: float = 0.008
    border_margin: float = 0.008
    min_blur_var: float = 24.0
    max_overlap_iou_for_memory: float = 0.55
    min_tracklet_quality: float = 0.10
    sample_stride: int = 3
    max_samples_per_tracklet: int = 24
    topk_embeds: int = 10
    topk_parts: int = 8
    strong_reuse_score: float = 0.70
    strong_reuse_margin: float = 0.03
    # If top-1 and top-2 point at different previous gids and their score
    # margin is below this threshold, treat as ambiguous and refuse reuse.
    cross_person_ambiguity_margin: float = 0.045
    # A "safe-same-candidate" margin: when top-1 and top-2 both point at the same
    # previous-tracklet/gid, they are NOT ambiguous — they agree on identity.
    # In that case we accept at a much lower margin (basically any margin).
    same_candidate_safe_accept: bool = True
    # Relaxed score / deep-sim thresholds ONLY for the same_candidate_safe_accept
    # path (top-1 and top-2 agree on the same prev gid).  When positive these
    # override strong_reuse_score / min_deep_sim_for_reuse on that path only.
    # Useful when a person's appearance changes mid-video (e.g. puts on a mask)
    # causing lower deep-similarity despite the tracker and both top candidates
    # consistently pointing at the same identity.
    # Default -1 → fall back to strong_reuse_score / min_deep_sim_for_reuse.
    same_candidate_safe_score: float = -1.0
    same_candidate_min_deep_relaxed: float = -1.0
    # Maximum gap (frames) allowed for the same_candidate_safe_accept path.
    # Prevents long-gap false merges where two different people both score well
    # against a small early tracklet in memory. -1 → no extra limit.
    same_candidate_max_gap_frames: float = -1.0
    # Source-GID continuity bias:
    # if the tracker's current source gid has a competitive historical candidate,
    # prefer preserving that source identity instead of switching to a nearby
    # competing gid. This reduces "same person, id keeps changing" cascades.
    same_source_gid_bias: bool = True
    same_source_allow_short_gap: bool = True
    same_source_min_gap_frames: int = 1
    same_source_max_gap_frames: int = 260
    same_source_min_score: float = 0.72
    same_source_min_deep: float = 0.62
    same_source_min_topk: float = 0.62
    same_source_min_part_topk: float = 0.58
    same_source_competitive_margin: float = 0.06
    # If cross-source top-1 is only slightly better than a viable same-source
    # candidate, block the cross-source reuse and force a new id (conservative).
    same_source_block_cross_source: bool = True
    same_source_block_min_score: float = 0.66
    same_source_block_margin: float = 0.05
    min_deep_sim_for_reuse: float = 0.66
    min_topk_sim_for_reuse: float = 0.68
    min_part_topk_for_reuse: float = 0.58
    min_part_mean_for_reuse: float = 0.64
    # When the deep/topk evidence is unusually strong, we can relax the
    # part-body requirements a little (helps when someone's upper/lower body
    # gets partially occluded during re-entry).
    strong_deep_relax_topk: float = 0.78
    strong_deep_relax_deep: float = 0.76
    w_deep: float = 0.42
    w_topk: float = 0.28
    w_upper_lower: float = 0.06
    w_part_topk: float = 0.12
    w_shape_size: float = 0.10
    w_time: float = 0.06
    w_side: float = 0.04
    w_motion: float = 0.02
    max_candidates_per_tracklet: int = 72
    min_candidate_prefilter_score: float = 0.30
    rerank_top_n: int = 6
    # Previously hardcoded inside _rerank_pool(); now exposed so retail_cam1.yaml
    # can tune them without editing source. Defaults were softened to reduce
    # false "ambiguous" rejections (was 0.08 / 0.05 / 0.05).
    rerank_part_imbalance_threshold: float = 0.46
    rerank_part_imbalance_penalty: float = 0.04
    rerank_part_asymmetry_threshold: float = 0.33
    rerank_part_asymmetry_penalty: float = 0.025
    rerank_deep_weak_threshold: float = 0.56
    rerank_deep_weak_penalty: float = 0.025
    enable_group_merge_pass: bool = True
    merge_max_gap_frames: int = 2200
    merge_min_score: float = 0.78
    merge_min_deep: float = 0.68
    merge_min_topk: float = 0.68
    merge_min_part_topk: float = 0.58
    enable_overlap_handoff_pass: bool = True
    overlap_handoff_max_gap_frames: int = 220
    overlap_handoff_min_score: float = 0.74
    overlap_handoff_min_deep: float = 0.62
    overlap_handoff_min_topk: float = 0.62
    overlap_handoff_min_part_topk: float = 0.55
    overlap_handoff_min_spatial: float = 0.46
    overlap_window_top_k: int = 5
    overlap_window_min_people: int = 4
    overlap_window_min_mean_iou: float = 0.13
    overlap_window_pad_frames: int = 28
    overlap_window_relax_delta: float = 0.05
    overlap_window_relaxed_max_gap_frames: int = 160
    enable_overlap_anti_switch_lock: bool = True
    anti_switch_max_gap_frames: int = 120
    anti_switch_min_score: float = 0.78
    anti_switch_min_deep: float = 0.64
    anti_switch_min_topk: float = 0.62
    anti_switch_min_part_topk: float = 0.56
    anti_switch_min_spatial: float = 0.48
    anti_switch_margin: float = 0.05
    anti_switch_tracklet_max_len_frames: int = 180
    anti_switch_target_gid_min_rows: int = 120
    anti_switch_reassign_margin_over_current: float = 0.08
    # Enabled by default: if multiple tracklets share the same local tracker id
    # and their appearance evidence agrees, they should share one global id.
    # Previously False — this was a primary cause of "same person, new id" bugs.
    enable_local_consistency_pass: bool = True
    local_consistency_min_score: float = 0.68
    local_consistency_min_deep: float = 0.58
    local_consistency_min_topk: float = 0.60
    # Stitch-trust path: when a same-source candidate exists within
    # max_reentry_gap_frames and its rerank score meets this threshold, accept
    # the reconnection without requiring cross-source appearance thresholds.
    # Fixes overhead-view cameras where OSNet similarity collapses for the same
    # person. Default -1.0 = disabled. Typical value: 0.60.
    same_source_stitch_trust_score: float = -1.0
    # Single-candidate spatial accept: when ONLY ONE candidate survives the gap
    # and quality filters (no cross-person ambiguity by construction), accept it
    # if spatial evidence (side_score, part_topk) passes conservative gates even
    # when the holistic appearance score is too weak for the normal cross-source
    # path. Designed for overhead-view cameras where OSNet collapses while
    # body-shape and entry-side evidence remain reliable. Default False.
    single_candidate_spatial_accept: bool = False
    single_candidate_min_score: float = 0.48
    single_candidate_min_part_topk: float = 0.80
    single_candidate_min_side_score: float = 0.60
    # Spatial-temporal reentry disambiguation (experiment: spatial_disambig_v1).
    # When cross_person_ambiguous fires, if the top candidate's last known position
    # is spatially implausible (jump > gap_frames * max_speed * top_factor) while
    # the second candidate is spatially plausible (jump <= gap_frames * max_speed *
    # sec_factor) and clears a score floor, redirect to the second candidate
    # instead of declaring ambiguous.  Disabled by default.
    spatial_disambig_enable: bool = False
    spatial_disambig_max_speed_px_frame: float = 180.0
    spatial_disambig_top_implausible_factor: float = 1.0
    spatial_disambig_sec_plausible_factor: float = 0.50
    spatial_disambig_min_sec_score: float = 0.68


@dataclass
class TrackRow:
    idx: int
    frame_idx: int
    gid: int
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def box(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def center(self) -> Tuple[float, float]:
        return 0.5 * (self.x1 + self.x2), 0.5 * (self.y1 + self.y2)

    @property
    def w(self) -> float:
        return max(1.0, self.x2 - self.x1)

    @property
    def h(self) -> float:
        return max(1.0, self.y2 - self.y1)


@dataclass
class Tracklet:
    tracklet_id: int
    local_track_id: int
    global_id: int
    start_frame: int
    end_frame: int
    rows: List[TrackRow]
    start_center: Tuple[float, float]
    end_center: Tuple[float, float]
    entry_side: str
    exit_side: str
    motion_dir: Tuple[float, float]
    median_h_ratio: float
    median_aspect: float
    embed_mean: Optional[np.ndarray] = None
    embeds_topk: List[np.ndarray] = None
    upper_mean: Optional[np.ndarray] = None
    lower_mean: Optional[np.ndarray] = None
    upper_topk: List[np.ndarray] = None
    lower_topk: List[np.ndarray] = None
    quality_score: float = 0.0
    clean_embed_count: int = 0
    final_gid: int = 0

    def __post_init__(self) -> None:
        if self.embeds_topk is None:
            self.embeds_topk = []
        if self.upper_topk is None:
            self.upper_topk = []
        if self.lower_topk is None:
            self.lower_topk = []


def _load_rows(csv_path: Path) -> Tuple[List[str], List[TrackRow]]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows: List[TrackRow] = []
        for idx, raw in enumerate(reader):
            try:
                rows.append(
                    TrackRow(
                        idx=int(idx),
                        frame_idx=int(raw["frame_idx"]),
                        gid=int(raw.get("global_id", 0) or 0),
                        x1=float(raw["x1"]),
                        y1=float(raw["y1"]),
                        x2=float(raw["x2"]),
                        y2=float(raw["y2"]),
                    )
                )
            except Exception:
                continue
    rows.sort(key=lambda r: (r.frame_idx, r.gid, r.idx))
    return fieldnames, rows


def _build_tracklets(rows: List[TrackRow], frame_w: int, frame_h: int, cfg: ReentryConfig) -> Tuple[List[Tracklet], Dict[int, int]]:
    by_gid: Dict[int, List[TrackRow]] = defaultdict(list)
    for r in rows:
        if int(r.gid) > 0:
            by_gid[int(r.gid)].append(r)

    tracklets: List[Tracklet] = []
    row_to_tracklet: Dict[int, int] = {}
    tid = 1
    for gid, rs in by_gid.items():
        rs = sorted(rs, key=lambda x: (x.frame_idx, x.idx))
        segs: List[List[TrackRow]] = []
        cur: List[TrackRow] = []
        prev_f = None
        for r in rs:
            if prev_f is None or (int(r.frame_idx) - int(prev_f)) <= int(cfg.segment_gap_frames):
                cur.append(r)
            else:
                if cur:
                    segs.append(cur)
                cur = [r]
            prev_f = int(r.frame_idx)
        if cur:
            segs.append(cur)

        for seg in segs:
            seg = sorted(seg, key=lambda x: (x.frame_idx, x.idx))
            s = seg[0]
            e = seg[-1]
            s_cx, s_cy = s.center
            e_cx, e_cy = e.center
            dv = np.array([e_cx - s_cx, e_cy - s_cy], dtype=np.float32)
            n = float(np.linalg.norm(dv))
            if n > 1e-6:
                dv = dv / n
            hs = np.array([r.h for r in seg], dtype=np.float32)
            ws = np.array([r.w for r in seg], dtype=np.float32)
            median_h_ratio = float(np.median(hs) / max(1.0, float(frame_h)))
            median_aspect = float(np.median(ws / hs))
            trk = Tracklet(
                tracklet_id=int(tid),
                local_track_id=int(gid),
                global_id=int(gid),
                start_frame=int(s.frame_idx),
                end_frame=int(e.frame_idx),
                rows=seg,
                start_center=(float(s_cx), float(s_cy)),
                end_center=(float(e_cx), float(e_cy)),
                entry_side=_zone_of_point(s_cx, s_cy, frame_w=frame_w, frame_h=frame_h),
                exit_side=_zone_of_point(e_cx, e_cy, frame_w=frame_w, frame_h=frame_h),
                motion_dir=(float(dv[0]), float(dv[1])),
                median_h_ratio=float(median_h_ratio),
                median_aspect=float(median_aspect),
            )
            tracklets.append(trk)
            for r in seg:
                row_to_tracklet[int(r.idx)] = int(tid)
            tid += 1

    tracklets.sort(key=lambda t: (t.start_frame, t.tracklet_id))
    return tracklets, row_to_tracklet


def _row_overlap_index(rows: List[TrackRow]) -> Dict[int, float]:
    by_frame: Dict[int, List[TrackRow]] = defaultdict(list)
    for r in rows:
        by_frame[int(r.frame_idx)].append(r)
    out: Dict[int, float] = defaultdict(float)
    for fr_rows in by_frame.values():
        for i, a in enumerate(fr_rows):
            best = 0.0
            for j, b in enumerate(fr_rows):
                if i == j:
                    continue
                best = max(best, _iou_xyxy(a.box, b.box))
            out[int(a.idx)] = float(best)
    return out


def _find_top_overlap_windows(
    rows: List[TrackRow],
    overlap_idx: Dict[int, float],
    cfg: ReentryConfig,
) -> List[Tuple[int, int, float]]:
    by_frame: Dict[int, List[TrackRow]] = defaultdict(list)
    for r in rows:
        by_frame[int(r.frame_idx)].append(r)
    if not by_frame:
        return []

    stats: List[Tuple[int, int, float, float]] = []
    for fr in sorted(by_frame.keys()):
        fr_rows = by_frame[fr]
        n = int(len(fr_rows))
        if n <= 0:
            continue
        mean_iou = float(np.mean([float(overlap_idx.get(int(rr.idx), 0.0)) for rr in fr_rows]))
        pressure = float(max(0.0, mean_iou - float(cfg.overlap_window_min_mean_iou) + 0.03) * float(n))
        stats.append((int(fr), n, mean_iou, pressure))

    segments: List[Tuple[int, int, float]] = []
    seg_start = None
    seg_end = None
    seg_score = 0.0
    prev_fr = None
    for fr, n, mean_iou, pressure in stats:
        active = bool(n >= int(cfg.overlap_window_min_people) and mean_iou >= float(cfg.overlap_window_min_mean_iou))
        contiguous = (prev_fr is None) or (int(fr) - int(prev_fr) <= 1)
        if active:
            if seg_start is None or not contiguous:
                if seg_start is not None:
                    segments.append((int(seg_start), int(seg_end), float(seg_score)))
                seg_start = int(fr)
                seg_end = int(fr)
                seg_score = float(pressure)
            else:
                seg_end = int(fr)
                seg_score += float(pressure)
        else:
            if seg_start is not None:
                segments.append((int(seg_start), int(seg_end), float(seg_score)))
                seg_start = None
                seg_end = None
                seg_score = 0.0
        prev_fr = int(fr)
    if seg_start is not None:
        segments.append((int(seg_start), int(seg_end), float(seg_score)))

    if not segments:
        return []

    pad = int(cfg.overlap_window_pad_frames)
    segs = [(max(0, int(s) - pad), int(e) + pad, float(sc)) for s, e, sc in segments]
    segs.sort(key=lambda x: x[2], reverse=True)

    picked: List[Tuple[int, int, float]] = []
    for s, e, sc in segs:
        if any(not (e < ps or s > pe) for ps, pe, _ in picked):
            continue
        picked.append((int(s), int(e), float(sc)))
        if len(picked) >= int(cfg.overlap_window_top_k):
            break
    picked.sort(key=lambda x: x[0])
    return picked


def _select_sample_rows(rows: List[TrackRow], cfg: ReentryConfig) -> List[TrackRow]:
    if not rows:
        return []
    rows = sorted(rows, key=lambda x: (x.frame_idx, x.idx))
    if len(rows) <= cfg.max_samples_per_tracklet:
        return rows[:: max(1, cfg.sample_stride)]
    picks = np.linspace(0, len(rows) - 1, num=cfg.max_samples_per_tracklet, dtype=np.int32)
    return [rows[int(i)] for i in picks]


def _crop_quality(
    frame: np.ndarray,
    row: TrackRow,
    overlap_iou: float,
    cfg: ReentryConfig,
) -> float:
    h, w = frame.shape[:2]
    x1 = max(0, min(w - 1, int(row.x1)))
    y1 = max(0, min(h - 1, int(row.y1)))
    x2 = max(0, min(w - 1, int(row.x2)))
    y2 = max(0, min(h - 1, int(row.y2)))
    if x2 <= x1 + 6 or y2 <= y1 + 10:
        return 0.0

    bw = x2 - x1
    bh = y2 - y1
    area_ratio = float((bw * bh) / max(1.0, float(w * h)))
    if area_ratio < (0.55 * float(cfg.min_area_ratio)):
        return 0.0
    area_score = float(np.clip(area_ratio / max(1e-6, float(cfg.min_area_ratio) * 3.5), 0.0, 1.0))

    margin_x = int(cfg.border_margin * w)
    margin_y = int(cfg.border_margin * h)
    on_border = (
        x1 <= margin_x
        or y1 <= margin_y
        or x2 >= int((1.0 - cfg.border_margin) * w)
        or y2 >= int((1.0 - cfg.border_margin) * h)
    )
    border_score = 0.62 if on_border else 1.0
    if on_border and area_ratio < float(cfg.min_area_ratio):
        return 0.0

    if overlap_iou > max(0.72, float(cfg.max_overlap_iou_for_memory) + 0.10):
        return 0.0
    overlap_score = float(np.clip(1.0 - (overlap_iou / max(1e-6, float(cfg.max_overlap_iou_for_memory) * 1.35)), 0.0, 1.0))

    crop = frame[y1:y2, x1:x2]
    if crop is None or crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur < max(8.0, float(cfg.min_blur_var) * 0.25):
        return 0.0
    blur_score = float(np.clip((blur - float(cfg.min_blur_var) * 0.35) / max(1e-6, float(cfg.min_blur_var) * 1.8), 0.0, 1.0))

    q = 0.40 * area_score
    q += 0.35 * blur_score
    q += 0.25 * overlap_score
    q *= float(border_score)
    return float(np.clip(q, 0.0, 1.0))


def _extract_tracklet_features(
    *,
    video_path: Path,
    tracklets: List[Tracklet],
    overlap_idx: Dict[int, float],
    reid_weights_path: Optional[str],
    cfg: ReentryConfig,
) -> None:
    if not tracklets:
        return

    extractor: Optional[ReidExtractor] = None
    try:
        extractor = ReidExtractor(model_name="osnet_x1_0", device=None, model_path=reid_weights_path)
    except Exception:
        extractor = None

    asks_by_frame: Dict[int, List[Tuple[int, TrackRow]]] = defaultdict(list)
    for t in tracklets:
        for r in _select_sample_rows(t.rows, cfg):
            asks_by_frame[int(r.frame_idx)].append((int(t.tracklet_id), r))

    embeds_by_tracklet: Dict[int, List[np.ndarray]] = defaultdict(list)
    upper_by_tracklet: Dict[int, List[np.ndarray]] = defaultdict(list)
    lower_by_tracklet: Dict[int, List[np.ndarray]] = defaultdict(list)
    q_by_tracklet: Dict[int, List[float]] = defaultdict(list)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for re-entry linker: {video_path}")

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        asks = asks_by_frame.get(frame_idx, [])
        if not asks:
            frame_idx += 1
            continue

        valid: List[Tuple[int, TrackRow, np.ndarray, float, np.ndarray, np.ndarray]] = []
        crops_rgb: List[np.ndarray] = []
        for tid, r in asks:
            ov = float(overlap_idx.get(int(r.idx), 0.0))
            q = _crop_quality(frame, r, overlap_iou=ov, cfg=cfg)
            if q <= 0.0:
                continue
            h, w = frame.shape[:2]
            x1 = max(0, min(w - 1, int(r.x1)))
            y1 = max(0, min(h - 1, int(r.y1)))
            x2 = max(0, min(w - 1, int(r.x2)))
            y2 = max(0, min(h - 1, int(r.y2)))
            crop_bgr = frame[y1:y2, x1:x2]
            if crop_bgr is None or crop_bgr.size == 0:
                continue
            up, lo = _upper_lower_support(crop_bgr)
            valid.append((int(tid), r, crop_bgr, q, up, lo))
            crops_rgb.append(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))

        if not valid:
            frame_idx += 1
            continue

        os_feats = None
        if extractor is not None:
            try:
                os_feats = extractor(crops_rgb).astype(np.float32)
                if os_feats.ndim == 1:
                    os_feats = os_feats[None, :]
            except Exception:
                os_feats = None

        for i, (tid, _r, crop_bgr, q, up, lo) in enumerate(valid):
            if os_feats is not None and i < len(os_feats):
                d_os = _l2(os_feats[i])
            else:
                d_os = np.zeros((512,), dtype=np.float32)
            d_att = attire_descriptor(crop_bgr).astype(np.float32)
            fused = np.concatenate([0.86 * d_os, 0.24 * d_att], axis=0).astype(np.float32)
            fused = _l2(fused)
            embeds_by_tracklet[int(tid)].append(fused)
            upper_by_tracklet[int(tid)].append(up)
            lower_by_tracklet[int(tid)].append(lo)
            q_by_tracklet[int(tid)].append(float(q))

        frame_idx += 1

    cap.release()

    by_tid = {int(t.tracklet_id): t for t in tracklets}
    for tid, t in by_tid.items():
        embs = embeds_by_tracklet.get(tid, [])
        qs = q_by_tracklet.get(tid, [])
        ups = upper_by_tracklet.get(tid, [])
        los = lower_by_tracklet.get(tid, [])
        if not embs:
            t.embed_mean = None
            t.embeds_topk = []
            t.upper_mean = None
            t.lower_mean = None
            t.upper_topk = []
            t.lower_topk = []
            t.quality_score = 0.0
            t.clean_embed_count = 0
            continue

        order = sorted(range(len(embs)), key=lambda i: qs[i], reverse=True)
        keep_n = min(int(cfg.topk_embeds), len(order))
        keep_idx = order[:keep_n]
        keep_embs = [_l2(embs[i]) for i in keep_idx]
        t.embeds_topk = keep_embs
        t.embed_mean = _l2(np.mean(np.stack(keep_embs, axis=0), axis=0))

        if ups:
            t.upper_mean = _l2(np.mean(np.stack([ups[i] for i in keep_idx], axis=0), axis=0))
            keep_up_n = min(int(cfg.topk_parts), len(keep_idx))
            t.upper_topk = [_l2(ups[i]) for i in keep_idx[:keep_up_n]]
        else:
            t.upper_topk = []
        if los:
            t.lower_mean = _l2(np.mean(np.stack([los[i] for i in keep_idx], axis=0), axis=0))
            keep_lo_n = min(int(cfg.topk_parts), len(keep_idx))
            t.lower_topk = [_l2(los[i]) for i in keep_idx[:keep_lo_n]]
        else:
            t.lower_topk = []
        t.quality_score = float(np.mean([qs[i] for i in keep_idx])) if keep_idx else 0.0
        t.clean_embed_count = int(len(keep_idx))


def _topk_similarity(a: List[np.ndarray], b: List[np.ndarray], k: int = 4) -> float:
    if not a or not b:
        return -1.0
    sims: List[float] = []
    for ea in a:
        for eb in b:
            sims.append(_cos(ea, eb))
    if not sims:
        return -1.0
    sims.sort(reverse=True)
    kk = max(1, min(int(k), len(sims)))
    return float(np.mean(sims[:kk]))


def _part_topk_similarity(prev: Tracklet, cur: Tracklet, k: int = 3) -> Tuple[float, float, float]:
    up = _topk_similarity(prev.upper_topk, cur.upper_topk, k=k)
    lo = _topk_similarity(prev.lower_topk, cur.lower_topk, k=k)
    up_p = max(0.0, float(up))
    lo_p = max(0.0, float(lo))
    combined = float(0.56 * up_p + 0.44 * lo_p)
    return float(up), float(lo), combined


def _motion_plausibility(prev: Tracklet, cur: Tracklet) -> float:
    px, py = prev.end_center
    cx, cy = cur.start_center
    d = np.array([cx - px, cy - py], dtype=np.float32)
    dn = float(np.linalg.norm(d))
    if dn > 1e-6:
        d = d / dn
    pv = np.array(prev.motion_dir, dtype=np.float32)
    if float(np.linalg.norm(pv)) <= 1e-6:
        return 0.5
    align = float(np.dot(_l2(pv), d))
    return float(np.clip(0.5 * (align + 1.0), 0.0, 1.0))


def _shape_size_score(prev: Tracklet, cur: Tracklet) -> Tuple[float, float, float]:
    h_ratio = min(prev.median_h_ratio, cur.median_h_ratio) / max(prev.median_h_ratio, cur.median_h_ratio, 1e-6)
    h_ratio = float(np.clip(h_ratio, 0.0, 1.0))
    a_ratio = min(prev.median_aspect, cur.median_aspect) / max(prev.median_aspect, cur.median_aspect, 1e-6)
    a_ratio = float(np.clip(a_ratio, 0.0, 1.0))
    return h_ratio, a_ratio, float(0.6 * h_ratio + 0.4 * a_ratio)


def _time_score(gap: int, max_gap: int) -> float:
    g = max(0, int(gap))
    if max_gap <= 0:
        return 0.0
    return float(math.exp(-float(g) / float(max_gap)))


def _candidate_prefilter_score(prev: Tracklet, cur: Tracklet, cfg: ReentryConfig) -> Tuple[float, float]:
    gap = int(cur.start_frame) - int(prev.end_frame)
    side = _side_score(prev.exit_side, cur.entry_side)
    t_sc = _time_score(gap, int(cfg.max_reentry_gap_frames))
    q_prev = float(np.clip(prev.quality_score, 0.0, 1.0))
    q_cur = float(np.clip(cur.quality_score, 0.0, 1.0))
    q_sc = float(math.sqrt(max(0.0, q_prev * q_cur)))
    pre = 0.50 * side + 0.30 * t_sc + 0.20 * q_sc
    return float(pre), float(gap)


def _hard_gate(prev: Tracklet, cur: Tracklet, cfg: ReentryConfig) -> Tuple[bool, str, Dict[str, float]]:
    gap = int(cur.start_frame) - int(prev.end_frame)
    if gap <= 0:
        return False, "time_overlap", {"gap": float(gap)}
    if gap < int(cfg.min_reentry_gap_frames):
        return False, "gap_too_short", {"gap": float(gap)}
    if gap > int(cfg.max_reentry_gap_frames):
        return False, "gap_too_long", {"gap": float(gap)}

    side = _side_score(prev.exit_side, cur.entry_side)
    if side < float(cfg.min_side_score):
        return False, "entry_exit_side_incompatible", {"side_score": float(side), "gap": float(gap)}

    h_ratio, a_ratio, _ = _shape_size_score(prev, cur)
    if h_ratio < float(cfg.min_size_ratio):
        return False, "size_ratio_implausible", {"h_ratio": float(h_ratio), "gap": float(gap), "side_score": float(side)}
    if a_ratio < float(cfg.min_aspect_ratio):
        return False, "aspect_ratio_implausible", {"a_ratio": float(a_ratio), "gap": float(gap), "side_score": float(side)}

    if prev.embed_mean is None or len(prev.embeds_topk) <= 0:
        return False, "prev_tracklet_no_prototype", {"prev_quality": float(prev.quality_score), "gap": float(gap)}
    if cur.embed_mean is None or len(cur.embeds_topk) <= 0:
        return False, "cur_tracklet_no_prototype", {"cur_quality": float(cur.quality_score), "gap": float(gap)}
    if (
        max(float(prev.quality_score), float(cur.quality_score)) < float(cfg.min_tracklet_quality)
        and min(len(prev.embeds_topk), len(cur.embeds_topk)) < 2
    ):
        return False, "pair_low_quality", {
            "prev_quality": float(prev.quality_score),
            "cur_quality": float(cur.quality_score),
            "gap": float(gap),
        }

    return True, "ok", {
        "gap": float(gap),
        "side_score": float(side),
        "h_ratio": float(h_ratio),
        "a_ratio": float(a_ratio),
        "prev_quality": float(prev.quality_score),
        "cur_quality": float(cur.quality_score),
    }


def _score_reentry(prev: Tracklet, cur: Tracklet, gate_info: Dict[str, float], cfg: ReentryConfig) -> Dict[str, float]:
    gap = int(gate_info.get("gap", 0.0))
    side_sc = float(gate_info.get("side_score", 0.0))
    deep = _cos(prev.embed_mean, cur.embed_mean)
    topk = _topk_similarity(prev.embeds_topk, cur.embeds_topk, k=4)
    up = _cos(prev.upper_mean, cur.upper_mean)
    lo = _cos(prev.lower_mean, cur.lower_mean)
    ul = 0.55 * max(0.0, up) + 0.45 * max(0.0, lo)
    up_topk, lo_topk, part_topk = _part_topk_similarity(prev, cur, k=3)
    _, _, shape = _shape_size_score(prev, cur)
    t_sc = _time_score(gap, int(cfg.max_reentry_gap_frames))
    m_sc = _motion_plausibility(prev, cur)
    q_prev = float(np.clip(prev.quality_score, 0.0, 1.0))
    q_cur = float(np.clip(cur.quality_score, 0.0, 1.0))
    q_sc = float(math.sqrt(max(0.0, q_prev * q_cur)))

    fused = (
        cfg.w_deep * max(0.0, deep)
        + cfg.w_topk * max(0.0, topk)
        + cfg.w_upper_lower * max(0.0, ul)
        + cfg.w_part_topk * max(0.0, part_topk)
        + cfg.w_shape_size * max(0.0, shape)
        + cfg.w_time * max(0.0, t_sc)
        + cfg.w_side * max(0.0, side_sc)
        + cfg.w_motion * max(0.0, m_sc)
    )
    # Keep low-quality tracklets usable but slightly down-weight their final score.
    fused = float(fused * (0.82 + 0.18 * q_sc))
    return {
        "deep_sim": float(deep),
        "topk_sim": float(topk),
        "upper_lower": float(ul),
        "upper_topk": float(up_topk),
        "lower_topk": float(lo_topk),
        "part_topk": float(part_topk),
        "shape_size": float(shape),
        "time_score": float(t_sc),
        "side_score": float(side_sc),
        "motion_score": float(m_sc),
        "quality_score": float(q_sc),
        "score": float(fused),
    }


def _overlap_frames(a: Tracklet, b: Tracklet) -> int:
    return max(0, min(int(a.end_frame), int(b.end_frame)) - max(int(a.start_frame), int(b.start_frame)) + 1)


def _min_gap_between_groups(a: List[Tracklet], b: List[Tracklet]) -> int:
    best = 10**9
    for ta in a:
        for tb in b:
            if _overlap_frames(ta, tb) > 0:
                return -1
            if ta.end_frame <= tb.start_frame:
                g = int(tb.start_frame) - int(ta.end_frame)
            else:
                g = int(ta.start_frame) - int(tb.end_frame)
            if g < best:
                best = g
    return int(best if best < 10**9 else -1)


def _group_has_overlap(a: List[Tracklet], b: List[Tracklet]) -> bool:
    return _min_gap_between_groups(a, b) < 0


def _group_similarity(a: List[Tracklet], b: List[Tracklet]) -> Dict[str, float]:
    a_means = [t.embed_mean for t in a if t.embed_mean is not None]
    b_means = [t.embed_mean for t in b if t.embed_mean is not None]
    if not a_means or not b_means:
        return {"score": -1.0, "deep": -1.0, "topk": -1.0, "part_topk": -1.0, "part_mean": -1.0}

    a_mean = _l2(np.mean(np.stack(a_means, axis=0), axis=0))
    b_mean = _l2(np.mean(np.stack(b_means, axis=0), axis=0))
    deep = _cos(a_mean, b_mean)

    a_topk: List[np.ndarray] = []
    b_topk: List[np.ndarray] = []
    a_up: List[np.ndarray] = []
    b_up: List[np.ndarray] = []
    a_lo: List[np.ndarray] = []
    b_lo: List[np.ndarray] = []
    for t in a:
        a_topk.extend(t.embeds_topk[:8])
        a_up.extend(t.upper_topk[:6])
        a_lo.extend(t.lower_topk[:6])
    for t in b:
        b_topk.extend(t.embeds_topk[:8])
        b_up.extend(t.upper_topk[:6])
        b_lo.extend(t.lower_topk[:6])

    topk = _topk_similarity(a_topk, b_topk, k=6)
    upk = _topk_similarity(a_up, b_up, k=4)
    lok = _topk_similarity(a_lo, b_lo, k=4)
    part_topk = 0.56 * max(0.0, upk) + 0.44 * max(0.0, lok)

    a_um = [t.upper_mean for t in a if t.upper_mean is not None]
    b_um = [t.upper_mean for t in b if t.upper_mean is not None]
    a_lm = [t.lower_mean for t in a if t.lower_mean is not None]
    b_lm = [t.lower_mean for t in b if t.lower_mean is not None]
    up_m = -1.0
    lo_m = -1.0
    if a_um and b_um:
        up_m = _cos(_l2(np.mean(np.stack(a_um, axis=0), axis=0)), _l2(np.mean(np.stack(b_um, axis=0), axis=0)))
    if a_lm and b_lm:
        lo_m = _cos(_l2(np.mean(np.stack(a_lm, axis=0), axis=0)), _l2(np.mean(np.stack(b_lm, axis=0), axis=0)))
    part_mean = 0.55 * max(0.0, up_m) + 0.45 * max(0.0, lo_m)

    score = 0.44 * max(0.0, deep) + 0.30 * max(0.0, topk) + 0.16 * max(0.0, part_topk) + 0.10 * max(0.0, part_mean)
    return {
        "score": float(score),
        "deep": float(deep),
        "topk": float(topk),
        "part_topk": float(part_topk),
        "part_mean": float(part_mean),
    }


def _tracklet_handoff_similarity(early: Tracklet, late: Tracklet) -> Dict[str, float]:
    gap = int(late.start_frame) - int(early.end_frame)
    if gap <= 0:
        return {"score": -1.0}

    deep = _cos(early.embed_mean, late.embed_mean)
    topk = _topk_similarity(early.embeds_topk, late.embeds_topk, k=4)
    _, _, part_topk = _part_topk_similarity(early, late, k=3)
    _, _, shape = _shape_size_score(early, late)
    side = _side_score(early.exit_side, late.entry_side)
    motion = _motion_plausibility(early, late)

    ex, ey = early.end_center
    sx, sy = late.start_center
    h_ref = max(20.0, float(max(early.rows[-1].h, late.rows[0].h)))
    dist = float(np.hypot(sx - ex, sy - ey) / h_ref)
    spatial = float(np.exp(-dist))

    score = (
        0.32 * max(0.0, deep)
        + 0.24 * max(0.0, topk)
        + 0.14 * max(0.0, part_topk)
        + 0.14 * max(0.0, spatial)
        + 0.08 * max(0.0, shape)
        + 0.04 * max(0.0, side)
        + 0.04 * max(0.0, motion)
    )
    return {
        "score": float(score),
        "gap": float(gap),
        "deep": float(deep),
        "topk": float(topk),
        "part_topk": float(part_topk),
        "shape": float(shape),
        "side": float(side),
        "motion": float(motion),
        "spatial": float(spatial),
    }


def _frame_in_windows(frame_idx: int, windows: List[Tuple[int, int, float]]) -> bool:
    fr = int(frame_idx)
    for s, e, _ in windows:
        if int(s) <= fr <= int(e):
            return True
    return False


def _tracklet_in_windows(t: Tracklet, windows: List[Tuple[int, int, float]]) -> bool:
    if not windows:
        return False
    s = int(t.start_frame)
    e = int(t.end_frame)
    for ws, we, _ in windows:
        if not (e < int(ws) or s > int(we)):
            return True
    return False


def _best_handoff_between_groups(
    a: List[Tracklet],
    b: List[Tracklet],
    max_gap: int,
    windows: Optional[List[Tuple[int, int, float]]] = None,
) -> Dict[str, float]:
    best: Dict[str, float] = {"score": -1.0}
    windows = windows or []
    for ta in a:
        for tb in b:
            early: Optional[Tracklet] = None
            late: Optional[Tracklet] = None
            if int(ta.end_frame) < int(tb.start_frame):
                early, late = ta, tb
            elif int(tb.end_frame) < int(ta.start_frame):
                early, late = tb, ta
            else:
                continue
            gap = int(late.start_frame) - int(early.end_frame)
            if gap <= 0 or gap > int(max_gap):
                continue
            sim = _tracklet_handoff_similarity(early, late)
            mid = int(0.5 * (int(early.end_frame) + int(late.start_frame)))
            sim["in_overlap_window"] = 1.0 if _frame_in_windows(mid, windows) else 0.0
            sim["mid_frame"] = float(mid)
            if float(sim.get("score", -1.0)) > float(best.get("score", -1.0)):
                best = sim
    return best


def _write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def link_reentry_offline(
    *,
    video_path: Path,
    tracks_csv_path: Path,
    reid_weights_path: Optional[str] = None,
    debug_dir: Optional[Path] = None,
    config: Optional[ReentryConfig] = None,
) -> dict:
    cfg = config or ReentryConfig()
    video_path = Path(video_path)
    tracks_csv_path = Path(tracks_csv_path)
    fieldnames, rows = _load_rows(tracks_csv_path)
    if not rows:
        return {"applied": False, "reason": "empty_tracks"}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for re-entry linking: {video_path}")
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    tracklets, row_to_tracklet = _build_tracklets(rows, frame_w=frame_w, frame_h=frame_h, cfg=cfg)
    if not tracklets:
        return {"applied": False, "reason": "no_tracklets"}

    overlap_idx = _row_overlap_index(rows)
    overlap_windows = _find_top_overlap_windows(rows, overlap_idx, cfg)
    _extract_tracklet_features(
        video_path=video_path,
        tracklets=tracklets,
        overlap_idx=overlap_idx,
        reid_weights_path=reid_weights_path,
        cfg=cfg,
    )

    # Fallback quality for tracklets with no clean crops.
    for t in tracklets:
        if t.embed_mean is None:
            t.quality_score = 0.0

    max_gid = max((int(t.global_id) for t in tracklets), default=0)
    next_gid = int(max_gid) + 1
    by_tid = {int(t.tracklet_id): t for t in tracklets}
    assigned_by_tid: Dict[int, int] = {}
    assigned_tracklets_by_gid: Dict[int, List[int]] = defaultdict(list)

    candidates_log: List[Dict[str, object]] = []
    decisions_log: List[Dict[str, object]] = []
    reentry_attempts = 0
    accepted_reuses = 0
    rejected_weak = 0
    rejected_ambiguous = 0
    rejected_conflict = 0
    accepted_same_source_bias = 0
    accepted_same_source_stitch_trust = 0
    accepted_single_candidate = 0
    blocked_cross_source_by_same_source = 0
    same_source_candidate_seen = 0

    # Process in chronological order: new tracklets query exited memory.
    for cur in sorted(tracklets, key=lambda t: (t.start_frame, t.tracklet_id)):
        # ensure tracklet gets some final gid
        cur_best = None
        cur_second = None
        best_side_score = -1.0
        best_n_candidates = 0
        scored_candidates: List[Dict[str, object]] = []

        plausible_prev_scored: List[Tuple[float, Tracklet]] = []
        for tid in assigned_by_tid.keys():
            prev = by_tid[tid]
            if prev.end_frame >= cur.start_frame:
                continue
            gap = int(cur.start_frame) - int(prev.end_frame)
            same_source = bool(int(cur.global_id) > 0 and int(prev.global_id) == int(cur.global_id))
            min_gap_req = int(cfg.min_reentry_gap_frames)
            if bool(cfg.same_source_allow_short_gap) and same_source:
                min_gap_req = min(min_gap_req, int(cfg.same_source_min_gap_frames))
            if gap < int(min_gap_req) or gap > int(cfg.max_reentry_gap_frames):
                continue
            pre_sc, _ = _candidate_prefilter_score(prev, cur, cfg)
            if pre_sc < float(cfg.min_candidate_prefilter_score):
                continue
            plausible_prev_scored.append((float(pre_sc), prev))
        plausible_prev_scored.sort(key=lambda x: x[0], reverse=True)
        if int(cfg.max_candidates_per_tracklet) > 0:
            plausible_prev_scored = plausible_prev_scored[: int(cfg.max_candidates_per_tracklet)]
        plausible_prev: List[Tracklet] = [x[1] for x in plausible_prev_scored]

        if plausible_prev:
            reentry_attempts += 1

        for prev in plausible_prev:
            ok, reason, gate_info = _hard_gate(prev, cur, cfg)
            if not ok:
                candidates_log.append(
                    {
                        "new_tracklet_id": int(cur.tracklet_id),
                        "prev_tracklet_id": int(prev.tracklet_id),
                        "prev_final_gid": int(assigned_by_tid.get(int(prev.tracklet_id), 0)),
                        "prev_source_gid": int(prev.global_id),
                        "status": "rejected",
                        "reason": str(reason),
                        **{k: float(v) for k, v in gate_info.items()},
                    }
                )
                continue

            comp = _score_reentry(prev, cur, gate_info, cfg)
            record = {
                "new_tracklet_id": int(cur.tracklet_id),
                "prev_tracklet_id": int(prev.tracklet_id),
                "prev_final_gid": int(assigned_by_tid.get(int(prev.tracklet_id), 0)),
                "prev_source_gid": int(prev.global_id),
                "status": "candidate",
                "reason": "scored",
                **{k: float(v) for k, v in gate_info.items()},
                **{k: float(v) for k, v in comp.items()},
            }
            candidates_log.append(record)
            scored_candidates.append(record)

        if scored_candidates:
            scored_candidates.sort(key=lambda r: float(r.get("score", -1.0) or -1.0), reverse=True)
            rerank_pool = scored_candidates[: max(1, int(cfg.rerank_top_n))]
            for rec in rerank_pool:
                base = float(rec.get("score", -1.0) or -1.0)
                deep = max(0.0, float(rec.get("deep_sim", -1.0) or -1.0))
                topk = max(0.0, float(rec.get("topk_sim", -1.0) or -1.0))
                part_topk = max(0.0, float(rec.get("part_topk", -1.0) or -1.0))
                up_topk = max(0.0, float(rec.get("upper_topk", -1.0) or -1.0))
                lo_topk = max(0.0, float(rec.get("lower_topk", -1.0) or -1.0))
                part_min = min(up_topk, lo_topk)
                penalty = 0.0
                if part_min < float(cfg.rerank_part_imbalance_threshold):
                    penalty += float(cfg.rerank_part_imbalance_penalty)
                if abs(up_topk - lo_topk) > float(cfg.rerank_part_asymmetry_threshold):
                    penalty += float(cfg.rerank_part_asymmetry_penalty)
                if deep < float(cfg.rerank_deep_weak_threshold):
                    penalty += float(cfg.rerank_deep_weak_penalty)
                rerank_score = (
                    0.66 * base
                    + 0.16 * deep
                    + 0.10 * topk
                    + 0.08 * part_topk
                    - penalty
                )
                rec["rerank_score"] = float(rerank_score)
                rec["rerank_penalty"] = float(penalty)

            for rec in scored_candidates[int(cfg.rerank_top_n) :]:
                rec["rerank_score"] = float(rec.get("score", -1.0) or -1.0)
                rec["rerank_penalty"] = 0.0

            scored_candidates.sort(key=lambda r: float(r.get("rerank_score", -1.0) or -1.0), reverse=True)
            if scored_candidates:
                best = scored_candidates[0]
                cur_best = (
                    float(best.get("rerank_score", -1.0) or -1.0),
                    float(best.get("deep_sim", -1.0) or -1.0),
                    float(best.get("topk_sim", -1.0) or -1.0),
                    float(best.get("part_topk", -1.0) or -1.0),
                    float(best.get("upper_lower", -1.0) or -1.0),
                    int(best.get("prev_tracklet_id", -1) or -1),
                )
                best_side_score = float(best.get("side_score", -1.0) or -1.0)
                best_n_candidates = len(scored_candidates)
            if len(scored_candidates) > 1:
                sec = scored_candidates[1]
                cur_second = (
                    float(sec.get("rerank_score", -1.0) or -1.0),
                    float(sec.get("deep_sim", -1.0) or -1.0),
                    float(sec.get("topk_sim", -1.0) or -1.0),
                    float(sec.get("part_topk", -1.0) or -1.0),
                    float(sec.get("upper_lower", -1.0) or -1.0),
                    int(sec.get("prev_tracklet_id", -1) or -1),
                )

        assigned_gid = 0
        decision_reason = "new_id_no_candidate"
        best_prev_source_gid = 0
        same_source_best_score = -1.0
        same_source_best_prev_tid = 0
        spatial_disambig_path = False
        _sd_jump_top = -1.0
        _sd_jump_sec = -1.0
        same_source_bias_applied = False
        same_source_cross_block_fired = False

        if cur_best is not None:
            best_score, best_deep, best_topk, best_part_topk, best_part_mean, best_prev_tid = cur_best
            margin = float(best_score - (cur_second[0] if cur_second is not None else -1.0))
            prev_gid = int(assigned_by_tid.get(int(best_prev_tid), 0))
            best_prev_tracklet = by_tid.get(int(best_prev_tid))
            best_prev_source_gid = int(best_prev_tracklet.global_id) if best_prev_tracklet is not None else 0

            same_source_best = None
            same_prev_gid = 0
            same_gap = float(10 ** 9)
            if bool(cfg.same_source_gid_bias) and int(cur.global_id) > 0:
                for rec in scored_candidates:
                    if int(rec.get("prev_source_gid", 0) or 0) == int(cur.global_id):
                        same_source_best = rec
                        break
            if same_source_best is not None:
                same_source_candidate_seen += 1
                same_source_best_score = float(same_source_best.get("rerank_score", -1.0) or -1.0)
                same_source_best_prev_tid = int(same_source_best.get("prev_tracklet_id", 0) or 0)
                same_prev_gid = int(assigned_by_tid.get(int(same_source_best_prev_tid), 0))
                same_gap = float(same_source_best.get("gap", 10**9) or 10**9)
                same_deep = float(same_source_best.get("deep_sim", -1.0) or -1.0)
                same_topk = float(same_source_best.get("topk_sim", -1.0) or -1.0)
                same_part_topk = float(same_source_best.get("part_topk", -1.0) or -1.0)
                same_ok = (
                    same_prev_gid > 0
                    and same_source_best_score >= float(cfg.same_source_min_score)
                    and same_deep >= float(cfg.same_source_min_deep)
                    and same_topk >= float(cfg.same_source_min_topk)
                    and same_part_topk >= float(cfg.same_source_min_part_topk)
                    and same_gap <= float(cfg.same_source_max_gap_frames)
                )
                best_is_cross_source = bool(int(cur.global_id) > 0 and int(best_prev_source_gid) > 0 and int(best_prev_source_gid) != int(cur.global_id))
                best_score_before_same_source = float(best_score)
                if (
                    same_ok
                    and best_is_cross_source
                    and (same_source_best_score + float(cfg.same_source_competitive_margin) >= best_score)
                ):
                    best_score = float(same_source_best_score)
                    best_deep = float(same_deep)
                    best_topk = float(same_topk)
                    best_part_topk = float(same_part_topk)
                    best_part_mean = float(same_source_best.get("upper_lower", -1.0) or -1.0)
                    best_prev_tid = int(same_source_best_prev_tid)
                    prev_gid = int(same_prev_gid)
                    best_prev_source_gid = int(cur.global_id)
                    same_source_bias_applied = True
                    margin_second = -1.0
                    for rec in scored_candidates:
                        rec_tid = int(rec.get("prev_tracklet_id", -1) or -1)
                        if rec_tid == int(best_prev_tid):
                            continue
                        margin_second = float(rec.get("rerank_score", -1.0) or -1.0)
                        break
                    margin = float(best_score - margin_second)

                if (
                    bool(cfg.same_source_block_cross_source)
                    and same_ok
                    and best_is_cross_source
                    and (not same_source_bias_applied)
                    and same_source_best_score >= float(cfg.same_source_block_min_score)
                    and (best_score_before_same_source - same_source_best_score) <= float(cfg.same_source_block_margin)
                ):
                    same_source_cross_block_fired = True

            # Stitch-trust: accept same-source candidate at a permissive
            # threshold, bypassing normal cross-source appearance requirements.
            # Intended for overhead-view cameras where OSNet similarity collapses.
            same_source_stitch_trust_ok = (
                same_source_best is not None
                and float(cfg.same_source_stitch_trust_score) > 0
                and same_prev_gid > 0
                and same_source_best_score >= float(cfg.same_source_stitch_trust_score)
                and same_gap <= float(cfg.max_reentry_gap_frames)
                and not same_source_cross_block_fired
            )

            # Single-candidate spatial accept: when only one candidate survives
            # the gap/quality filters there is no cross-person ambiguity by
            # construction. Accept using spatial gates (entry side, body-shape)
            # rather than holistic appearance, which collapses in overhead view.
            single_candidate_ok = (
                bool(cfg.single_candidate_spatial_accept)
                and best_n_candidates == 1
                and prev_gid > 0
                and best_score >= float(cfg.single_candidate_min_score)
                and best_part_topk >= float(cfg.single_candidate_min_part_topk)
                and best_side_score >= float(cfg.single_candidate_min_side_score)
                and not same_source_cross_block_fired
            )

            # -------- Safe-accept #1: top-1 and top-2 agree on the SAME prev gid.
            # This is NOT ambiguity — both strong candidates point at the same
            # identity. Previously this case was rejected because margin was tiny
            # (top-1 and top-2 were both ~0.72).
            top_two_same_prev_gid = False
            if (
                bool(cfg.same_candidate_safe_accept)
                and cur_second is not None
                and prev_gid > 0
            ):
                second_prev_tid = int(cur_second[5])
                second_prev_gid = int(assigned_by_tid.get(second_prev_tid, 0))
                if second_prev_gid == prev_gid:
                    top_two_same_prev_gid = True

            # -------- Safe-accept #2: very strong deep+topk evidence, even with
            # narrower margin. A 0.80 deep match is reliable even if part-body
            # signal is partially occluded.
            # Was margin >= 0.0 — too lax. When two different people wear
            # similar-colored clothing (e.g. black-shirt boy vs. black-shirt
            # woman) their deep/topk both clear 0.76/0.78 and the margin
            # between them is near zero. We now require a real 0.04 margin
            # on this path to prevent wrong-reuse across persons.
            strong_deep_path = (
                best_deep >= float(cfg.strong_deep_relax_deep)
                and best_topk >= float(cfg.strong_deep_relax_topk)
                and margin >= 0.04
            )

            # Primary acceptance gate (original, but using relaxed thresholds).
            primary_ok = (
                best_score >= float(cfg.strong_reuse_score)
                and margin >= float(cfg.strong_reuse_margin)
                and best_deep >= float(cfg.min_deep_sim_for_reuse)
                and best_topk >= float(cfg.min_topk_sim_for_reuse)
                and best_part_topk >= float(cfg.min_part_topk_for_reuse)
                and best_part_mean >= float(cfg.min_part_mean_for_reuse)
            )

            # -------- Cross-person disambiguation gate.
            # If the runner-up candidate points to a DIFFERENT previous gid
            # than top-1, and the margin is tight (< 0.045), we CANNOT tell
            # two similar-looking persons apart from OSNet alone. Abstain
            # and let this tracklet become a fresh id — fragmentation is
            # strictly safer than wrong-reuse.
            #
            # This gate never fires when top-1 and top-2 agree on the same
            # prev_gid (that case is handled by `same_candidate_safe_accept`,
            # which is not an ambiguity — it is agreement).
            cross_person_ambiguous = False
            if cur_second is not None:
                second_prev_tid = int(cur_second[5])
                second_prev_gid_check = int(assigned_by_tid.get(second_prev_tid, 0))
                if (
                    second_prev_gid_check > 0
                    and prev_gid > 0
                    and int(second_prev_gid_check) != int(prev_gid)
                    and float(margin) < float(cfg.cross_person_ambiguity_margin)
                ):
                    cross_person_ambiguous = True
            if cross_person_ambiguous:
                # Invalidate both the primary and the strong-deep-relax paths.
                # `same_candidate_safe_accept` is unaffected because its
                # definition requires top-1 and top-2 to point at the SAME
                # prev_gid, which is incompatible with cross_person_ambiguous.
                primary_ok = False
                strong_deep_path = False

            # -------- Spatial-temporal disambiguation (spatial_disambig_v1).
            # When cross_person_ambiguous fires and the top candidate's last
            # known position is physically unreachable in the elapsed frames
            # (jump > gap * max_speed * top_factor), but the second candidate's
            # last known position is clearly reachable (jump <= gap * max_speed *
            # sec_factor) and its rerank score clears the floor, redirect all
            # "best" variables to the second candidate.  This converts tracklets
            # that were forced to gid=0 (due to false-tie ambiguity with a
            # spatially impossible candidate) into accepted reentries.
            spatial_disambig_path = False
            _sd_jump_top = -1.0
            _sd_jump_sec = -1.0
            if (
                cross_person_ambiguous
                and not same_source_cross_block_fired
                and bool(cfg.spatial_disambig_enable)
                and cur_second is not None
            ):
                _sec_prev_tid_sd = int(cur_second[5])
                _top_prev_t = by_tid.get(int(best_prev_tid))
                _sec_prev_t = by_tid.get(_sec_prev_tid_sd)
                _sec_gid_sd = int(assigned_by_tid.get(_sec_prev_tid_sd, 0))
                if _top_prev_t is not None and _sec_prev_t is not None and _sec_gid_sd > 0:
                    _cur_cx, _cur_cy = cur.start_center
                    _sd_jump_top = float(math.hypot(
                        _cur_cx - _top_prev_t.end_center[0],
                        _cur_cy - _top_prev_t.end_center[1],
                    ))
                    _sd_jump_sec = float(math.hypot(
                        _cur_cx - _sec_prev_t.end_center[0],
                        _cur_cy - _sec_prev_t.end_center[1],
                    ))
                    _gap_top = max(1, int(cur.start_frame) - int(_top_prev_t.end_frame))
                    _gap_sec = max(1, int(cur.start_frame) - int(_sec_prev_t.end_frame))
                    _speed = float(cfg.spatial_disambig_max_speed_px_frame)
                    _top_limit = _gap_top * _speed * float(cfg.spatial_disambig_top_implausible_factor)
                    _sec_limit = _gap_sec * _speed * float(cfg.spatial_disambig_sec_plausible_factor)
                    if (
                        _sd_jump_top > _top_limit
                        and _sd_jump_sec <= _sec_limit
                        and float(cur_second[0]) >= float(cfg.spatial_disambig_min_sec_score)
                    ):
                        prev_gid = _sec_gid_sd
                        best_prev_tid = _sec_prev_tid_sd
                        best_score = float(cur_second[0])
                        best_deep = float(cur_second[1])
                        best_topk = float(cur_second[2])
                        best_part_topk = float(cur_second[3])
                        best_part_mean = float(cur_second[4])
                        margin = best_score  # no tracked third candidate; conservative
                        cross_person_ambiguous = False
                        spatial_disambig_path = True
                        primary_ok = (
                            best_score >= float(cfg.strong_reuse_score)
                            and margin >= float(cfg.strong_reuse_margin)
                            and best_deep >= float(cfg.min_deep_sim_for_reuse)
                            and best_topk >= float(cfg.min_topk_sim_for_reuse)
                            and best_part_topk >= float(cfg.min_part_topk_for_reuse)
                            and best_part_mean >= float(cfg.min_part_mean_for_reuse)
                        )
                        strong_deep_path = (
                            best_deep >= float(cfg.strong_deep_relax_deep)
                            and best_topk >= float(cfg.strong_deep_relax_topk)
                            and margin >= 0.04
                        )

            accept_path = ""
            if same_source_cross_block_fired:
                accept_path = ""
            elif prev_gid > 0 and spatial_disambig_path:
                accept_path = "spatial_disambig_second_candidate"
            elif prev_gid > 0 and primary_ok:
                accept_path = "strong_reentry_reuse"
            elif prev_gid > 0 and top_two_same_prev_gid and best_score >= (float(cfg.same_candidate_safe_score) if float(cfg.same_candidate_safe_score) > 0 else float(cfg.strong_reuse_score)) and best_deep >= (float(cfg.same_candidate_min_deep_relaxed) if float(cfg.same_candidate_min_deep_relaxed) > 0 else float(cfg.min_deep_sim_for_reuse)) and (float(cfg.same_candidate_max_gap_frames) <= 0 or gap <= float(cfg.same_candidate_max_gap_frames)):
                accept_path = "same_candidate_safe_accept"
            elif prev_gid > 0 and strong_deep_path and best_score >= float(cfg.strong_reuse_score):
                accept_path = "strong_deep_relax_accept"
            elif (
                prev_gid > 0
                and same_source_bias_applied
                and best_score >= float(cfg.same_source_min_score)
                and best_deep >= float(cfg.same_source_min_deep)
                and best_topk >= float(cfg.same_source_min_topk)
                and best_part_topk >= float(cfg.same_source_min_part_topk)
                and margin >= 0.0
                and not cross_person_ambiguous
            ):
                accept_path = "same_source_gid_bias"
            elif same_source_stitch_trust_ok:
                # Redirect to the same-source candidate — the stitch's identity
                # grouping is more reliable than collapsed overhead-view OSNet scores.
                prev_gid = int(same_prev_gid)
                best_prev_tid = int(same_source_best_prev_tid)
                accept_path = "same_source_stitch_trust"
            elif single_candidate_ok:
                # Only one candidate in memory within the allowed gap window —
                # no cross-person confusion is possible. Spatial gates (entry
                # side, body shape) confirm plausibility without relying on the
                # holistic appearance score that collapses in overhead view.
                accept_path = "single_candidate_spatial_accept"

            if accept_path:
                # Conservative rule: never allow same final gid in overlapping time.
                conflict = False
                for otid in assigned_tracklets_by_gid.get(prev_gid, []):
                    if _overlap_frames(by_tid[otid], cur) > 0:
                        conflict = True
                        break
                if conflict:
                    rejected_conflict += 1
                    decision_reason = "rejected_overlap_conflict_new_id"
                else:
                    assigned_gid = int(prev_gid)
                    accepted_reuses += 1
                    if accept_path == "same_source_gid_bias":
                        accepted_same_source_bias += 1
                    elif accept_path == "same_source_stitch_trust":
                        accepted_same_source_stitch_trust += 1
                    elif accept_path == "single_candidate_spatial_accept":
                        accepted_single_candidate += 1
                    decision_reason = f"accepted_reentry_prev_tracklet_{best_prev_tid}_via_{accept_path}"
                    # Update candidate status for accepted one.
                    for rec in candidates_log:
                        if int(rec.get("new_tracklet_id", -1)) == int(cur.tracklet_id) and int(rec.get("prev_tracklet_id", -1)) == int(best_prev_tid):
                            rec["status"] = "accepted"
                            rec["reason"] = accept_path
            else:
                if cross_person_ambiguous:
                    # Top-1 and top-2 point at different prev_gids and the
                    # margin is too thin to tell the two persons apart.
                    # We assign a new id rather than risk wrong-reuse.
                    rejected_ambiguous += 1
                    decision_reason = "rejected_cross_person_ambiguous_new_id"
                elif same_source_cross_block_fired:
                    blocked_cross_source_by_same_source += 1
                    rejected_ambiguous += 1
                    decision_reason = "rejected_same_source_competitive_block_new_id"
                elif margin < float(cfg.strong_reuse_margin):
                    rejected_ambiguous += 1
                    decision_reason = "rejected_ambiguous_margin_new_id"
                elif best_part_topk < float(cfg.min_part_topk_for_reuse) or best_part_mean < float(cfg.min_part_mean_for_reuse):
                    rejected_weak += 1
                    decision_reason = "rejected_part_mismatch_new_id"
                else:
                    rejected_weak += 1
                    decision_reason = "rejected_weak_score_new_id"

        if assigned_gid <= 0:
            assigned_gid = int(next_gid)
            next_gid += 1

        assigned_by_tid[int(cur.tracklet_id)] = int(assigned_gid)
        assigned_tracklets_by_gid[int(assigned_gid)].append(int(cur.tracklet_id))
        cur.final_gid = int(assigned_gid)

        decisions_log.append(
            {
                "tracklet_id": int(cur.tracklet_id),
                "local_track_id": int(cur.local_track_id),
                "source_gid": int(cur.global_id),
                "assigned_final_gid": int(assigned_gid),
                "start_frame": int(cur.start_frame),
                "end_frame": int(cur.end_frame),
                "entry_side": str(cur.entry_side),
                "exit_side": str(cur.exit_side),
                "quality": float(cur.quality_score),
                "best_score": float(best_score) if cur_best is not None else -1.0,
                "best_prev_tracklet_id": int(best_prev_tid) if cur_best is not None else 0,
                "best_prev_source_gid": int(best_prev_source_gid) if cur_best is not None else 0,
                "same_source_best_score": float(same_source_best_score),
                "same_source_best_prev_tracklet_id": int(same_source_best_prev_tid),
                "same_source_bias_applied": bool(same_source_bias_applied),
                "same_source_cross_block_fired": bool(same_source_cross_block_fired),
                "spatial_disambig_applied": bool(spatial_disambig_path),
                "spatial_disambig_jump_top_px": float(_sd_jump_top),
                "spatial_disambig_jump_sec_px": float(_sd_jump_sec),
                "decision_reason": str(decision_reason),
            }
        )

    merged_gid_pairs = 0
    local_consistency_reassignments = 0
    if bool(cfg.enable_group_merge_pass):
        gid_to_tids: Dict[int, List[int]] = defaultdict(list)
        for tid, gid in assigned_by_tid.items():
            gid_to_tids[int(gid)].append(int(tid))

        gids = sorted([int(g) for g in gid_to_tids.keys() if int(g) > 0])
        merge_candidates: List[Tuple[float, int, int]] = []
        merge_debug: Dict[Tuple[int, int], Dict[str, float]] = {}
        for i in range(len(gids)):
            ga = int(gids[i])
            ta = [by_tid[t] for t in gid_to_tids.get(ga, [])]
            if not ta:
                continue
            for j in range(i + 1, len(gids)):
                gb = int(gids[j])
                tb = [by_tid[t] for t in gid_to_tids.get(gb, [])]
                if not tb:
                    continue
                gap = _min_gap_between_groups(ta, tb)
                if gap < 0 or gap > int(cfg.merge_max_gap_frames):
                    continue
                sim = _group_similarity(ta, tb)
                # Standard strict merge gate — OSNet-primary evidence.
                strict_ok = (
                    sim["score"] >= float(cfg.merge_min_score)
                    and sim["deep"] >= float(cfg.merge_min_deep)
                    and sim["topk"] >= float(cfg.merge_min_topk)
                    and sim["part_topk"] >= float(cfg.merge_min_part_topk)
                )
                # Pose-change-tolerant merge gate (color/part-led).
                #
                # Motivation: when a person sits down, bends, or has a
                # partial occlusion between two tracked fragments, their
                # full-body OSNet similarity can fall into the 0.70-0.78
                # band even though the garment colors (upper + lower means)
                # are unmistakably the same. The strict gate misses these,
                # producing fragmentation (pink-shirt girl two IDs, green-
                # shirt 1→6→9, beige-girl 3→11). We ONLY fire this path
                # when BOTH part channels (topk + mean) are very high
                # (>= 0.72) AND full-body OSNet is still moderate (>= 0.72)
                # AND deep mean is at least 0.64. This preserves purity:
                # two different persons who happen to wear similar colors
                # will not pass because their deep/topk will not also hit
                # 0.72/0.72 — color alone is never enough.
                pose_change_ok = (
                    sim["score"] >= 0.72
                    and sim["deep"] >= 0.64
                    and sim["topk"] >= 0.72
                    and sim["part_topk"] >= 0.72
                    and sim["part_mean"] >= 0.72
                )
                if strict_ok or pose_change_ok:
                    merge_candidates.append((float(sim["score"]), int(ga), int(gb)))
                    merge_debug[(int(ga), int(gb))] = sim

        parent: Dict[int, int] = {int(g): int(g) for g in gids}
        group_members: Dict[int, List[Tracklet]] = {
            int(g): [by_tid[t] for t in gid_to_tids.get(int(g), [])]
            for g in gids
        }

        def _find(x: int) -> int:
            y = int(x)
            while parent[y] != y:
                parent[y] = parent[parent[y]]
                y = parent[y]
            return int(y)

        def _union(a: int, b: int) -> None:
            ra = _find(a)
            rb = _find(b)
            if ra == rb:
                return
            keep = min(int(ra), int(rb))
            drop = max(int(ra), int(rb))
            parent[drop] = keep
            group_members[keep].extend(group_members.get(drop, []))
            group_members.pop(drop, None)

        for _sc, ga, gb in sorted(merge_candidates, key=lambda x: x[0], reverse=True):
            ra = _find(int(ga))
            rb = _find(int(gb))
            if ra == rb:
                continue
            grp_a = group_members.get(ra, [])
            grp_b = group_members.get(rb, [])
            if _group_has_overlap(grp_a, grp_b):
                continue
            _union(ra, rb)
            merged_gid_pairs += 1

        if merged_gid_pairs > 0:
            for tid, gid in list(assigned_by_tid.items()):
                assigned_by_tid[int(tid)] = int(_find(int(gid)))
            assigned_tracklets_by_gid = defaultdict(list)
            for tid, gid in assigned_by_tid.items():
                assigned_tracklets_by_gid[int(gid)].append(int(tid))
            for d in decisions_log:
                tid = int(d.get("tracklet_id", 0) or 0)
                if tid > 0:
                    d["assigned_final_gid"] = int(assigned_by_tid.get(tid, int(d.get("assigned_final_gid", 0) or 0)))
                    d["decision_reason"] = f"{d.get('decision_reason', '')}|group_merge_pass"

    overlap_handoff_merged_gid_pairs = 0
    if bool(cfg.enable_overlap_handoff_pass):
        gid_to_tids: Dict[int, List[int]] = defaultdict(list)
        for tid, gid in assigned_by_tid.items():
            gid_to_tids[int(gid)].append(int(tid))
        gids = sorted([int(g) for g in gid_to_tids.keys() if int(g) > 0])

        parent_h: Dict[int, int] = {int(g): int(g) for g in gids}
        members_h: Dict[int, List[Tracklet]] = {
            int(g): [by_tid[t] for t in gid_to_tids.get(int(g), [])]
            for g in gids
        }

        def _find_h(x: int) -> int:
            y = int(x)
            while parent_h[y] != y:
                parent_h[y] = parent_h[parent_h[y]]
                y = parent_h[y]
            return int(y)

        def _union_h(a: int, b: int) -> None:
            ra = _find_h(a)
            rb = _find_h(b)
            if ra == rb:
                return
            keep = min(int(ra), int(rb))
            drop = max(int(ra), int(rb))
            parent_h[drop] = keep
            members_h[keep].extend(members_h.get(drop, []))
            members_h.pop(drop, None)

        handoff_candidates: List[Tuple[float, int, int]] = []
        for i in range(len(gids)):
            ga = int(gids[i])
            grp_a = members_h.get(ga, [])
            if not grp_a:
                continue
            for j in range(i + 1, len(gids)):
                gb = int(gids[j])
                grp_b = members_h.get(gb, [])
                if not grp_b:
                    continue
                if _group_has_overlap(grp_a, grp_b):
                    continue
                sim = _best_handoff_between_groups(
                    grp_a,
                    grp_b,
                    int(cfg.overlap_handoff_max_gap_frames),
                    overlap_windows,
                )
                if float(sim.get("score", -1.0)) < 0.0:
                    continue
                in_overlap_window = bool(float(sim.get("in_overlap_window", 0.0)) > 0.5)
                relax = float(cfg.overlap_window_relax_delta) if in_overlap_window else 0.0
                max_gap_allowed = int(cfg.overlap_handoff_max_gap_frames)
                if in_overlap_window:
                    max_gap_allowed = min(max_gap_allowed, int(cfg.overlap_window_relaxed_max_gap_frames))
                if int(sim.get("gap", 0.0)) > int(max_gap_allowed):
                    continue
                if (
                    float(sim.get("score", -1.0)) >= float(cfg.overlap_handoff_min_score - relax)
                    and float(sim.get("deep", -1.0)) >= float(cfg.overlap_handoff_min_deep - relax)
                    and float(sim.get("topk", -1.0)) >= float(cfg.overlap_handoff_min_topk - relax)
                    and float(sim.get("part_topk", -1.0)) >= float(cfg.overlap_handoff_min_part_topk - relax)
                    and float(sim.get("spatial", -1.0)) >= float(cfg.overlap_handoff_min_spatial - 1.2 * relax)
                ):
                    handoff_candidates.append((float(sim["score"]), int(ga), int(gb)))

        for _sc, ga, gb in sorted(handoff_candidates, key=lambda x: x[0], reverse=True):
            ra = _find_h(int(ga))
            rb = _find_h(int(gb))
            if ra == rb:
                continue
            grp_a = members_h.get(ra, [])
            grp_b = members_h.get(rb, [])
            if _group_has_overlap(grp_a, grp_b):
                continue
            _union_h(ra, rb)
            overlap_handoff_merged_gid_pairs += 1

        if overlap_handoff_merged_gid_pairs > 0:
            for tid, gid in list(assigned_by_tid.items()):
                assigned_by_tid[int(tid)] = int(_find_h(int(gid)))
            for d in decisions_log:
                tid = int(d.get("tracklet_id", 0) or 0)
                if tid > 0:
                    d["assigned_final_gid"] = int(assigned_by_tid.get(tid, int(d.get("assigned_final_gid", 0) or 0)))
                    d["decision_reason"] = f"{d.get('decision_reason', '')}|overlap_handoff_pass"

    overlap_anti_switch_locks = 0
    if bool(cfg.enable_overlap_anti_switch_lock) and overlap_windows:
        # Focus only on dense-overlap windows: keep short-gap identity continuity
        # with very strict evidence + margin checks.
        gid_total_rows: Dict[int, int] = defaultdict(int)
        gid_tids_map: Dict[int, List[int]] = defaultdict(list)
        for tid, gid in assigned_by_tid.items():
            gid = int(gid)
            if gid <= 0:
                continue
            gid_total_rows[gid] += int(len(by_tid[int(tid)].rows))
            gid_tids_map[gid].append(int(tid))

        tids_in_windows = [
            int(t.tracklet_id)
            for t in sorted(tracklets, key=lambda tt: (tt.start_frame, tt.tracklet_id))
            if _tracklet_in_windows(t, overlap_windows)
        ]

        for cur_tid in tids_in_windows:
            cur = by_tid.get(int(cur_tid))
            if cur is None:
                continue
            if int(len(cur.rows)) > int(cfg.anti_switch_tracklet_max_len_frames):
                continue
            cur_gid = int(assigned_by_tid.get(int(cur_tid), 0))
            if cur_gid <= 0:
                continue

            scored_prev: List[Tuple[float, float, float, float, float, int]] = []
            for prev_tid, prev_gid in assigned_by_tid.items():
                prev_tid = int(prev_tid)
                prev_gid = int(prev_gid)
                if prev_tid == int(cur_tid) or prev_gid <= 0:
                    continue
                prev = by_tid.get(prev_tid)
                if prev is None:
                    continue
                if int(prev.end_frame) >= int(cur.start_frame):
                    continue
                gap = int(cur.start_frame) - int(prev.end_frame)
                if gap <= 0 or gap > int(cfg.anti_switch_max_gap_frames):
                    continue
                sim = _tracklet_handoff_similarity(prev, cur)
                sc = float(sim.get("score", -1.0))
                deep = float(sim.get("deep", -1.0))
                topk = float(sim.get("topk", -1.0))
                part_topk = float(sim.get("part_topk", -1.0))
                spatial = float(sim.get("spatial", -1.0))
                if (
                    sc >= float(cfg.anti_switch_min_score)
                    and deep >= float(cfg.anti_switch_min_deep)
                    and topk >= float(cfg.anti_switch_min_topk)
                    and part_topk >= float(cfg.anti_switch_min_part_topk)
                    and spatial >= float(cfg.anti_switch_min_spatial)
                ):
                    scored_prev.append((sc, deep, topk, part_topk, spatial, prev_tid))

            if not scored_prev:
                continue
            scored_prev.sort(key=lambda x: x[0], reverse=True)
            best = scored_prev[0]
            second_score = float(scored_prev[1][0]) if len(scored_prev) > 1 else -1.0
            margin = float(best[0] - second_score)
            if margin < float(cfg.anti_switch_margin):
                continue

            best_prev_tid = int(best[5])
            target_gid = int(assigned_by_tid.get(best_prev_tid, 0))
            if target_gid <= 0 or target_gid == cur_gid:
                continue
            if int(gid_total_rows.get(target_gid, 0)) < int(cfg.anti_switch_target_gid_min_rows):
                continue

            current_ref_score = -1.0
            current_ref_deep = -1.0
            current_ref_topk = -1.0
            current_ref_part = -1.0
            cur_gid_other_tids = [t for t in gid_tids_map.get(cur_gid, []) if int(t) != int(cur_tid)]
            if cur_gid_other_tids:
                cur_gid_group = [by_tid[t] for t in cur_gid_other_tids]
                sim_cur = _group_similarity([cur], cur_gid_group)
                current_ref_score = float(sim_cur.get("score", -1.0))
                current_ref_deep = float(sim_cur.get("deep", -1.0))
                current_ref_topk = float(sim_cur.get("topk", -1.0))
                current_ref_part = float(sim_cur.get("part_topk", -1.0))

            best_score = float(best[0])
            if best_score < float(current_ref_score + float(cfg.anti_switch_reassign_margin_over_current)):
                continue
            # Require target evidence to beat current-ID evidence on core appearance channels.
            if current_ref_deep > 0 and float(best[1]) < float(current_ref_deep + 0.03):
                continue
            if current_ref_topk > 0 and float(best[2]) < float(current_ref_topk + 0.03):
                continue
            if current_ref_part > 0 and float(best[3]) < float(current_ref_part + 0.03):
                continue

            # Never allow same final gid in overlapping time ranges.
            conflict = False
            for otid, ogid in assigned_by_tid.items():
                if int(ogid) != int(target_gid) or int(otid) == int(cur_tid):
                    continue
                if _overlap_frames(by_tid[int(otid)], cur) > 0:
                    conflict = True
                    break
            if conflict:
                continue

            old_gid = int(cur_gid)
            assigned_by_tid[int(cur_tid)] = int(target_gid)
            overlap_anti_switch_locks += 1
            gid_total_rows[old_gid] = max(0, int(gid_total_rows.get(old_gid, 0)) - int(len(cur.rows)))
            gid_total_rows[target_gid] = int(gid_total_rows.get(target_gid, 0)) + int(len(cur.rows))
            gid_tids_map[old_gid] = [t for t in gid_tids_map.get(old_gid, []) if int(t) != int(cur_tid)]
            gid_tids_map[target_gid].append(int(cur_tid))
            for d in decisions_log:
                if int(d.get("tracklet_id", 0) or 0) == int(cur_tid):
                    d["assigned_final_gid"] = int(target_gid)
                    d["decision_reason"] = f"{d.get('decision_reason', '')}|overlap_anti_switch_lock"
                    d["best_prev_tracklet_id"] = int(best_prev_tid)
                    d["best_score"] = float(best[0])
                    break

    # Keep identity coherent for fragmented segments under the same local tracker id
    # when appearance evidence supports the merge.
    if bool(cfg.enable_local_consistency_pass):
        local_to_tids: Dict[int, List[int]] = defaultdict(list)
        for tid, t in by_tid.items():
            local_to_tids[int(t.local_track_id)].append(int(tid))
        gid_to_tids_now: Dict[int, List[int]] = defaultdict(list)
        for tid, gid in assigned_by_tid.items():
            gid_to_tids_now[int(gid)].append(int(tid))

        for _local_id, tids in local_to_tids.items():
            if len(tids) <= 1:
                continue
            frames_per_gid: Dict[int, int] = defaultdict(int)
            for tid in tids:
                gid = int(assigned_by_tid.get(int(tid), 0))
                if gid <= 0:
                    continue
                frames_per_gid[gid] += int(len(by_tid[tid].rows))
            if not frames_per_gid:
                continue
            dom_gid = int(max(frames_per_gid.items(), key=lambda kv: kv[1])[0])
            dom_group = [by_tid[t] for t in gid_to_tids_now.get(dom_gid, [])]
            if not dom_group:
                continue

            for tid in tids:
                tid = int(tid)
                cur_gid = int(assigned_by_tid.get(tid, 0))
                if cur_gid <= 0 or cur_gid == dom_gid:
                    continue
                cur_track = by_tid[tid]
                if _group_has_overlap([cur_track], dom_group):
                    continue
                sim = _group_similarity([cur_track], dom_group)
                if (
                    sim["score"] >= float(cfg.local_consistency_min_score)
                    and sim["deep"] >= float(cfg.local_consistency_min_deep)
                    and sim["topk"] >= float(cfg.local_consistency_min_topk)
                ):
                    assigned_by_tid[tid] = int(dom_gid)
                    local_consistency_reassignments += 1
                    gid_to_tids_now[int(cur_gid)] = [x for x in gid_to_tids_now.get(int(cur_gid), []) if int(x) != tid]
                    gid_to_tids_now[int(dom_gid)].append(int(tid))
                    dom_group.append(cur_track)

        if local_consistency_reassignments > 0:
            for d in decisions_log:
                tid = int(d.get("tracklet_id", 0) or 0)
                if tid > 0:
                    d["assigned_final_gid"] = int(assigned_by_tid.get(tid, int(d.get("assigned_final_gid", 0) or 0)))
                    d["decision_reason"] = f"{d.get('decision_reason', '')}|local_consistency_pass"

    # Rewrite CSV IDs by tracklet decisions.
    id_by_idx: Dict[int, int] = {}
    for r in rows:
        tid = int(row_to_tracklet.get(int(r.idx), 0))
        gid = int(assigned_by_tid.get(tid, int(r.gid)))
        id_by_idx[int(r.idx)] = int(gid)

    with tracks_csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        out_rows = []
        for idx, raw in enumerate(reader):
            raw["global_id"] = str(int(id_by_idx.get(int(idx), int(raw.get("global_id", 0) or 0))))
            out_rows.append(raw)

    with tracks_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    # Final safety: avoid duplicate positive IDs in same frame.
    by_frame: Dict[int, List[TrackRow]] = defaultdict(list)
    _, rows_after = _load_rows(tracks_csv_path)
    for r in rows_after:
        by_frame[int(r.frame_idx)].append(r)
    row_gid_after = {int(r.idx): int(r.gid) for r in rows_after}
    dedup_changed = 0
    for fr_rows in by_frame.values():
        by_gid: Dict[int, List[TrackRow]] = defaultdict(list)
        for r in fr_rows:
            g = int(row_gid_after.get(int(r.idx), int(r.gid)))
            if g > 0:
                by_gid[g].append(r)
        for gid, rs in by_gid.items():
            if len(rs) <= 1:
                continue
            keep = max(rs, key=lambda rr: rr.w * rr.h)
            for rr in rs:
                if int(rr.idx) == int(keep.idx):
                    continue
                if row_gid_after.get(int(rr.idx), 0) > 0:
                    row_gid_after[int(rr.idx)] = 0
                    dedup_changed += 1

    if dedup_changed > 0:
        with tracks_csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            out_rows = []
            for idx, raw in enumerate(reader):
                raw["global_id"] = str(int(row_gid_after.get(int(idx), int(raw.get("global_id", 0) or 0))))
                out_rows.append(raw)
        with tracks_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(out_rows)

    # Debug exports.
    dbg = debug_dir or (tracks_csv_path.parent / "reentry_debug")
    dbg.mkdir(parents=True, exist_ok=True)

    exited_rows: List[Dict[str, object]] = []
    entered_rows: List[Dict[str, object]] = []
    for t in tracklets:
        base = {
            "tracklet_id": int(t.tracklet_id),
            "local_track_id": int(t.local_track_id),
            "source_gid": int(t.global_id),
            "final_gid": int(assigned_by_tid.get(int(t.tracklet_id), 0)),
            "start_frame": int(t.start_frame),
            "end_frame": int(t.end_frame),
            "entry_side": str(t.entry_side),
            "exit_side": str(t.exit_side),
            "motion_dx": float(t.motion_dir[0]),
            "motion_dy": float(t.motion_dir[1]),
            "median_h_ratio": float(t.median_h_ratio),
            "median_aspect": float(t.median_aspect),
            "quality": float(t.quality_score),
            "num_rows": int(len(t.rows)),
            "num_clean_embeds": int(len(t.embeds_topk)),
            "clean_embed_count": int(t.clean_embed_count),
            "upper_topk_count": int(len(t.upper_topk)),
            "lower_topk_count": int(len(t.lower_topk)),
        }
        entered_rows.append(base)
        exited_rows.append(base)

    _write_csv(
        dbg / "entered_tracklets.csv",
        entered_rows,
        [
            "tracklet_id",
            "local_track_id",
            "source_gid",
            "final_gid",
            "start_frame",
            "end_frame",
            "entry_side",
            "exit_side",
            "motion_dx",
            "motion_dy",
            "median_h_ratio",
            "median_aspect",
            "quality",
            "num_rows",
            "num_clean_embeds",
            "clean_embed_count",
            "upper_topk_count",
            "lower_topk_count",
        ],
    )
    _write_csv(
        dbg / "exited_tracklets.csv",
        exited_rows,
        [
            "tracklet_id",
            "local_track_id",
            "source_gid",
            "final_gid",
            "start_frame",
            "end_frame",
            "entry_side",
            "exit_side",
            "motion_dx",
            "motion_dy",
            "median_h_ratio",
            "median_aspect",
            "quality",
            "num_rows",
            "num_clean_embeds",
            "clean_embed_count",
            "upper_topk_count",
            "lower_topk_count",
        ],
    )

    cand_fields = sorted({k for r in candidates_log for k in r.keys()}) if candidates_log else [
        "new_tracklet_id",
        "prev_tracklet_id",
        "prev_final_gid",
        "status",
        "reason",
    ]
    _write_csv(dbg / "reentry_candidates.csv", candidates_log, cand_fields)
    _write_csv(
        dbg / "reentry_decisions.csv",
        decisions_log,
        [
            "tracklet_id",
            "local_track_id",
            "source_gid",
            "assigned_final_gid",
            "start_frame",
            "end_frame",
            "entry_side",
            "exit_side",
            "quality",
            "best_score",
            "best_prev_tracklet_id",
            "best_prev_source_gid",
            "same_source_best_score",
            "same_source_best_prev_tracklet_id",
            "same_source_bias_applied",
            "same_source_cross_block_fired",
            "spatial_disambig_applied",
            "spatial_disambig_jump_top_px",
            "spatial_disambig_jump_sec_px",
            "decision_reason",
        ],
    )

    tracklet_rows: List[Dict[str, object]] = []
    for r in rows:
        tid = int(row_to_tracklet.get(int(r.idx), 0))
        tracklet_rows.append(
            {
                "row_idx": int(r.idx),
                "frame_idx": int(r.frame_idx),
                "source_gid": int(r.gid),
                "tracklet_id": int(tid),
                "assigned_final_gid": int(assigned_by_tid.get(int(tid), 0)),
            }
        )
    _write_csv(
        dbg / "tracklet_rows.csv",
        tracklet_rows,
        ["row_idx", "frame_idx", "source_gid", "tracklet_id", "assigned_final_gid"],
    )

    overlap_rows = [
        {
            "window_idx": int(i + 1),
            "start_frame": int(s),
            "end_frame": int(e),
            "score": float(sc),
        }
        for i, (s, e, sc) in enumerate(overlap_windows)
    ]
    _write_csv(
        dbg / "overlap_windows.csv",
        overlap_rows,
        ["window_idx", "start_frame", "end_frame", "score"],
    )

    stats = {
        "tracklets_total": int(len(tracklets)),
        "reentry_attempts": int(reentry_attempts),
        "accepted_reuses": int(accepted_reuses),
        "accepted_same_source_bias": int(accepted_same_source_bias),
        "accepted_same_source_stitch_trust": int(accepted_same_source_stitch_trust),
        "accepted_single_candidate": int(accepted_single_candidate),
        "same_source_candidate_seen": int(same_source_candidate_seen),
        "blocked_cross_source_by_same_source": int(blocked_cross_source_by_same_source),
        "rejected_weak": int(rejected_weak),
        "rejected_ambiguous": int(rejected_ambiguous),
        "rejected_conflict": int(rejected_conflict),
        "new_ids_created": int(sum(1 for d in decisions_log if "new_id" in str(d.get("decision_reason", "")))),
        "group_merged_gid_pairs": int(merged_gid_pairs),
        "overlap_handoff_merged_gid_pairs": int(overlap_handoff_merged_gid_pairs),
        "overlap_windows_count": int(len(overlap_windows)),
        "overlap_anti_switch_locks": int(overlap_anti_switch_locks),
        "local_consistency_reassignments": int(local_consistency_reassignments),
        "dedup_changed_rows": int(dedup_changed),
    }
    (dbg / "reentry_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    return {
        "applied": True,
        "stats": stats,
        "debug_dir": str(dbg),
    }
