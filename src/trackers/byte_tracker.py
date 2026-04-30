from __future__ import annotations

import numpy as np
from collections import deque
from typing import List, Optional, Tuple
try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None


def iou_xyxy(a, b) -> float:
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    areaA = max(0.0, (a[2] - a[0])) * max(0.0, (a[3] - a[1]))
    areaB = max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))
    denom = areaA + areaB - inter + 1e-6
    return float(inter / denom) if denom > 0 else 0.0


def cos_sim(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    if a.shape != b.shape:
        return 0.0
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    a = a / na
    b = b / nb
    return float(np.dot(a, b))


def bbox_center_size(box) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    w = max(0.0, float(x2) - float(x1))
    h = max(0.0, float(y2) - float(y1))
    cx = float(x1) + 0.5 * w
    cy = float(y1) + 0.5 * h
    return cx, cy, w, h


class STrack:
    _next_id = 1

    @staticmethod
    def reset_id_counter() -> None:
        STrack._next_id = 1

    def __init__(
        self,
        tlbr,
        score: float,
        feature: Optional[np.ndarray] = None,
        track_id: Optional[int] = None,
        feature_history: int = 40,
        feature_update_min_sim: float = 0.68,
        confirm_hits: int = 4,
    ):
        self.track_id = int(track_id) if track_id is not None else STrack._next_id
        if track_id is None:
            STrack._next_id += 1

        self.tlbr = list(map(float, tlbr))
        self.score = float(score)
        self.feature = feature.astype(np.float32) if feature is not None else None
        self.anchor_feature = self.feature.copy() if self.feature is not None else None
        cx, cy, _, _ = bbox_center_size(self.tlbr)
        self.last_cx = float(cx)
        self.last_cy = float(cy)
        self.vx = 0.0
        self.vy = 0.0

        self.is_activated = True
        self.last_frame = 0
        self.age = 0
        self.time_since_update = 0

        self.hits = 1
        self.confirm_hits = int(confirm_hits)
        self.is_confirmed = self.hits >= self.confirm_hits

        self.feature_update_min_sim = float(feature_update_min_sim)
        self.features = deque(maxlen=int(feature_history))
        if self.feature is not None:
            self.features.append(self.feature)

        self.global_id: Optional[int] = None
        self.global_id_locked: bool = False
        self.last_feat_quality: float = 0.0
        self.last_det_conf: float = float(score)
        self.last_reid_mode: int = 0
        self.id_state: str = "unassigned"  # unassigned|tentative|locked
        self.pending_gid: Optional[int] = None
        self.pending_hits: int = 0
        self.pending_score: float = 0.0
        self.locked_since_frame: int = -1

    def mark_hit(self) -> None:
        self.hits += 1
        self.is_confirmed = self.hits >= self.confirm_hits

    def best_feature_similarity(self, feat: Optional[np.ndarray]) -> float:
        if feat is None:
            return 0.0
        best = 0.0
        if self.anchor_feature is not None:
            best = max(best, cos_sim(self.anchor_feature, feat))
        if self.feature is not None:
            best = max(best, cos_sim(self.feature, feat))
        for f in self.features:
            best = max(best, cos_sim(f, feat))
        return float(best)

    def anchor_similarity(self, feat: Optional[np.ndarray]) -> float:
        if feat is None or self.anchor_feature is None:
            return 0.0
        return float(cos_sim(self.anchor_feature, feat))

    def update_feature(self, new_feature: np.ndarray, ema: float = 0.92):
        nf = new_feature.astype(np.float32)

        if self.feature is None:
            self.feature = nf
            if self.anchor_feature is None:
                self.anchor_feature = nf.copy()
            self.features.append(nf)
            return

        sim = cos_sim(self.feature, nf)
        self.features.append(nf)

        if sim >= self.feature_update_min_sim:
            self.feature = (ema * self.feature + (1.0 - ema) * nf).astype(np.float32)

        # Keep first-ID anchor mostly stable while allowing tiny cleanup.
        if self.anchor_feature is None:
            self.anchor_feature = nf.copy()
        else:
            sim_anchor = cos_sim(self.anchor_feature, nf)
            if sim_anchor >= 0.90:
                self.anchor_feature = (0.97 * self.anchor_feature + 0.03 * nf).astype(np.float32)

    def update_tlbr(self, tlbr):
        old_cx, old_cy, _, _ = bbox_center_size(self.tlbr)
        self.tlbr = list(map(float, tlbr))
        new_cx, new_cy, _, _ = bbox_center_size(self.tlbr)

        dx = float(new_cx - old_cx)
        dy = float(new_cy - old_cy)

        # Keep lightweight motion memory to reduce ID swaps after overlap/short occlusion.
        self.vx = 0.7 * self.vx + 0.3 * dx
        self.vy = 0.7 * self.vy + 0.3 * dy
        self.last_cx = float(new_cx)
        self.last_cy = float(new_cy)


class BYTETracker:
    def __init__(
        self,
        track_thresh: float = 0.35,
        active_match_iou_thresh: float = 0.22,
        lost_match_iou_thresh: float = 0.10,
        match_feat_thresh: float = 0.50,
        strong_reid_thresh: float = 0.84,
        long_lost_reid_thresh: float = 0.90,
        alpha_active: float = 0.35,
        alpha_lost: float = 0.88,
        track_buffer: int = 140,
        motion_max_center_dist: float = 0.70,
        motion_max_gap: int = 24,
        overlap_iou_thresh: float = 0.30,
        min_height_ratio_for_update: float = 0.72,
        min_match_conf: Optional[float] = 0.30,
        feature_history: int = 40,
        feature_update_min_sim: float = 0.68,
        confirm_hits: int = 4,
        bad_frame_hold: int = 12,
        reid_update_min_quality: float = 0.55,
    ):
        self.track_thresh = float(track_thresh)
        self.active_match_iou_thresh = float(active_match_iou_thresh)
        self.lost_match_iou_thresh = float(lost_match_iou_thresh)
        self.match_feat_thresh = float(match_feat_thresh)
        self.strong_reid_thresh = float(strong_reid_thresh)
        self.long_lost_reid_thresh = float(long_lost_reid_thresh)
        self.alpha_active = float(alpha_active)
        self.alpha_lost = float(alpha_lost)
        self.track_buffer = int(track_buffer)
        self.motion_max_center_dist = float(motion_max_center_dist)
        self.motion_max_gap = int(motion_max_gap)
        self.overlap_iou_thresh = float(overlap_iou_thresh)
        self.min_height_ratio_for_update = float(min_height_ratio_for_update)
        self.min_match_conf = float(min_match_conf) if min_match_conf is not None else None
        self.feature_history = int(feature_history)
        self.feature_update_min_sim = float(feature_update_min_sim)
        self.confirm_hits = int(confirm_hits)
        self.bad_frame_hold = int(bad_frame_hold)
        self.reid_update_min_quality = float(reid_update_min_quality)

        self.tracks: List[STrack] = []
        self.lost_tracks: List[STrack] = []
        self.frame_id = 0
        self.bad_frame_streak = 0

    def _det_conf_ok(self, det) -> bool:
        if self.min_match_conf is None:
            return True
        return float(det[4]) >= self.min_match_conf

    def _should_update_feature(self, tr: STrack, new_box) -> bool:
        _, _, _, prev_h = bbox_center_size(tr.tlbr)
        _, _, _, new_h = bbox_center_size(new_box)
        if prev_h <= 1e-6:
            return True
        ratio = new_h / prev_h
        return ratio >= self.min_height_ratio_for_update

    def _is_overlap_ambiguous(self, tr: STrack, tracks: List[STrack], iou_thresh: Optional[float] = None) -> bool:
        if iou_thresh is None:
            iou_thresh = self.overlap_iou_thresh
        for other in tracks:
            if other is tr:
                continue
            if iou_xyxy(tr.tlbr, other.tlbr) >= iou_thresh:
                return True
        return False

    def _predict_center(self, tr: STrack) -> Tuple[float, float, float]:
        cx, cy, _, h = bbox_center_size(tr.tlbr)
        h = max(h, 1e-6)
        gap = max(1, int(tr.time_since_update))
        pred_cx = cx + tr.vx * gap
        pred_cy = cy + tr.vy * gap
        return float(pred_cx), float(pred_cy), float(h)

    def _motion_alignment(self, tr: STrack, box) -> float:
        prev_cx, prev_cy, _, _ = bbox_center_size(tr.tlbr)
        cx, cy, _, _ = bbox_center_size(box)
        dx = float(cx - prev_cx)
        dy = float(cy - prev_cy)
        vnorm = float(np.hypot(tr.vx, tr.vy))
        dnorm = float(np.hypot(dx, dy))
        if vnorm <= 1e-6 or dnorm <= 1e-6:
            return 0.0
        return float((dx * tr.vx + dy * tr.vy) / (vnorm * dnorm))

    def _select_non_conflicting(
        self,
        candidates: List[Tuple[float, int, int]],
        n_dets: int,
        min_score: float = 0.0,
    ) -> Tuple[List[Tuple[int, int]], List[int]]:
        valid = [(float(s), int(ti), int(di)) for s, ti, di in candidates if float(s) >= float(min_score)]
        if not valid:
            return [], list(range(n_dets))

        if linear_sum_assignment is not None:
            track_ids = sorted(set(ti for _, ti, _ in valid))
            det_ids = sorted(set(di for _, _, di in valid))
            ti_to_row = {ti: i for i, ti in enumerate(track_ids)}
            di_to_col = {di: j for j, di in enumerate(det_ids)}

            big = 1e6
            cost = np.full((len(track_ids), len(det_ids)), big, dtype=np.float32)
            for s, ti, di in valid:
                r = ti_to_row[ti]
                c = di_to_col[di]
                v = -float(s)
                if v < cost[r, c]:
                    cost[r, c] = v

            rows, cols = linear_sum_assignment(cost)
            assigned: List[Tuple[int, int]] = []
            used_dets = set()
            for r, c in zip(rows, cols):
                if cost[r, c] >= big * 0.5:
                    continue
                ti = track_ids[r]
                di = det_ids[c]
                assigned.append((ti, di))
                used_dets.add(di)

            unused_det = [i for i in range(n_dets) if i not in used_dets]
            return assigned, unused_det

        assigned: List[Tuple[int, int]] = []
        used_tracks = set()
        used_dets = set()

        for score, ti, di in sorted(valid, key=lambda x: x[0], reverse=True):
            if ti in used_tracks or di in used_dets:
                continue
            used_tracks.add(ti)
            used_dets.add(di)
            assigned.append((ti, di))

        unused_det = [i for i in range(n_dets) if i not in used_dets]
        return assigned, unused_det

    def _feat_ok(
        self,
        feats: Optional[np.ndarray],
        feat_quality: Optional[np.ndarray],
        feat_mode: Optional[np.ndarray],
        di: int,
    ) -> bool:
        if feats is None or len(feats) == 0 or di < 0 or di >= len(feats):
            return False
        if feat_mode is not None and di < len(feat_mode):
            if int(feat_mode[di]) <= 0:
                return False
        if feat_quality is not None and di < len(feat_quality):
            if float(feat_quality[di]) < self.reid_update_min_quality:
                return False
        f = feats[di]
        if f is None:
            return False
        return float(np.linalg.norm(f)) > 1e-6

    def _match_active_tracks(self, tracks: List[STrack], dets: np.ndarray, feats: Optional[np.ndarray]):
        candidates: List[Tuple[float, int, int]] = []

        for ti, tr in enumerate(tracks):
            pred_cx, pred_cy, prev_h = self._predict_center(tr)
            overlap_ambiguous = self._is_overlap_ambiguous(tr, tracks)
            alpha_feat = self.alpha_active if not overlap_ambiguous else 0.90

            for di, det in enumerate(dets):
                if not self._det_conf_ok(det):
                    continue

                x1, y1, x2, y2, _ = det
                box = [x1, y1, x2, y2]

                s_iou = iou_xyxy(tr.tlbr, box)
                s_best = tr.best_feature_similarity(feats[di]) if feats is not None and len(feats) > 0 else 0.0
                s_anchor = tr.anchor_similarity(feats[di]) if feats is not None and len(feats) > 0 else 0.0
                if tr.is_confirmed:
                    s_feat = 0.66 * s_best + 0.34 * s_anchor
                else:
                    s_feat = s_best

                cx, cy, _, _ = bbox_center_size(box)
                center_dist = np.hypot(cx - pred_cx, cy - pred_cy)
                center_dist_norm = center_dist / prev_h

                close_motion = (
                    center_dist_norm <= self.motion_max_center_dist
                    and tr.time_since_update <= self.motion_max_gap
                )

                if overlap_ambiguous and tr.is_confirmed and max(s_best, s_anchor) < max(0.58, self.match_feat_thresh + 0.10):
                    continue

                if not close_motion and s_iou < self.active_match_iou_thresh and s_feat < 0.58:
                    continue

                if close_motion:
                    score = alpha_feat * s_feat + (1.0 - alpha_feat) * s_iou + 0.03
                else:
                    score = 0.70 * s_feat + 0.30 * s_iou

                if overlap_ambiguous:
                    score += 0.08 * s_feat
                    score -= 0.04 * max(0.0, 0.55 - s_feat)

                score += 0.05 * s_anchor
                score -= 0.04 * min(center_dist_norm, 2.0)
                score += 0.03 * self._motion_alignment(tr, box)
                candidates.append((float(score), ti, di))

        assigned_idx, unused_det = self._select_non_conflicting(candidates, n_dets=len(dets), min_score=0.10)
        assigned = [(tracks[ti], di) for ti, di in assigned_idx]
        return assigned, unused_det

    def _match_lost_tracks(self, tracks: List[STrack], dets: np.ndarray, feats: Optional[np.ndarray]):
        candidates: List[Tuple[float, int, int]] = []

        for ti, tr in enumerate(tracks):
            pred_cx, pred_cy, prev_h = self._predict_center(tr)

            for di, det in enumerate(dets):
                if not self._det_conf_ok(det):
                    continue

                x1, y1, x2, y2, _ = det
                box = [x1, y1, x2, y2]

                s_iou = iou_xyxy(tr.tlbr, box)
                s_best = tr.best_feature_similarity(feats[di]) if feats is not None and len(feats) > 0 else 0.0
                s_anchor = tr.anchor_similarity(feats[di]) if feats is not None and len(feats) > 0 else 0.0
                s_feat = 0.64 * s_best + 0.36 * s_anchor if tr.is_confirmed else s_best

                cx, cy, _, _ = bbox_center_size(box)
                center_dist = np.hypot(cx - pred_cx, cy - pred_cy)
                center_dist_norm = center_dist / prev_h

                close_motion = (
                    center_dist_norm <= self.motion_max_center_dist * 1.25
                    and tr.time_since_update <= self.motion_max_gap
                )

                if max(s_best, s_anchor) < self.match_feat_thresh and s_iou < self.lost_match_iou_thresh:
                    continue
                speed = float(np.hypot(tr.vx, tr.vy))
                stationary = speed < 4.0
                strong_gate = max(self.strong_reid_thresh, 0.84)
                if stationary and center_dist_norm < 0.45:
                    strong_gate = max(self.match_feat_thresh + 0.05, strong_gate - 0.08)

                if not close_motion and s_feat < strong_gate:
                    continue

                score = self.alpha_lost * s_feat + (1.0 - self.alpha_lost) * s_iou
                if close_motion:
                    score += 0.02
                score += 0.05 * s_anchor
                score -= 0.05 * min(center_dist_norm, 2.5)
                candidates.append((float(score), ti, di))

        assigned_idx, unused_det = self._select_non_conflicting(candidates, n_dets=len(dets), min_score=0.12)
        assigned = [(tracks[ti], di) for ti, di in assigned_idx]
        return assigned, unused_det

    def _match_long_lost_tracks(self, tracks: List[STrack], dets: np.ndarray, feats: Optional[np.ndarray]):
        if feats is None or len(feats) == 0 or len(dets) == 0:
            return [], []

        candidates: List[Tuple[float, int, int]] = []
        for ti, tr in enumerate(tracks):
            for di in range(len(feats)):
                x1, y1, x2, y2, _ = dets[di]
                box = [x1, y1, x2, y2]
                pred_cx, pred_cy, prev_h = self._predict_center(tr)
                cx, cy, _, _ = bbox_center_size(box)
                center_dist = np.hypot(cx - pred_cx, cy - pred_cy)
                center_dist_norm = center_dist / max(prev_h, 1e-6)

                s_best = tr.best_feature_similarity(feats[di])
                s_anchor = tr.anchor_similarity(feats[di])
                s_feat = 0.62 * s_best + 0.38 * s_anchor if tr.is_confirmed else s_best
                speed = float(np.hypot(tr.vx, tr.vy))
                stationary = speed < 4.0

                req = self.long_lost_reid_thresh
                if center_dist_norm < 0.55:
                    req -= 0.08
                if stationary and center_dist_norm < 0.70:
                    req -= 0.05
                req = max(self.match_feat_thresh + 0.06, req)

                if s_feat < req:
                    continue

                # Prefer stronger appearance match and shorter time gap.
                age_penalty = 0.0025 * float(min(tr.time_since_update, self.track_buffer))
                score = s_feat + 0.05 * s_anchor - age_penalty - 0.03 * min(center_dist_norm, 2.0)
                candidates.append((float(score), ti, di))

        min_score = max(self.match_feat_thresh + 0.02, self.long_lost_reid_thresh - 0.12)
        assigned_idx, unused_det = self._select_non_conflicting(candidates, n_dets=len(feats), min_score=min_score)
        assigned = [(tracks[ti], di) for ti, di in assigned_idx]
        return assigned, unused_det

    def handle_bad_frame(self, frame_id: int) -> List[STrack]:
        """
        Bad frame mode:
        - do not run normal association
        - do not create new tracks
        - keep current active tracks alive for a short window
        """
        self.frame_id = int(frame_id)
        self.bad_frame_streak += 1

        for tr in self.tracks:
            tr.age += 1
            tr.time_since_update += 1

        for tr in list(self.lost_tracks):
            tr.age += 1
            tr.time_since_update += 1
            if tr.time_since_update > self.track_buffer:
                self.lost_tracks.remove(tr)

        # Keep active tracks alive across short corrupted windows
        kept = []
        for tr in self.tracks:
            if tr.time_since_update <= self.bad_frame_hold:
                kept.append(tr)
            else:
                tr.is_activated = False
                if tr not in self.lost_tracks:
                    self.lost_tracks.append(tr)

        self.tracks = kept
        return self.tracks

    def reset(self, reset_ids: bool = False) -> None:
        self.tracks = []
        self.lost_tracks = []
        self.frame_id = 0
        self.bad_frame_streak = 0
        if reset_ids:
            STrack.reset_id_counter()

    def update(
        self,
        dets_full: Optional[np.ndarray],
        feats: Optional[np.ndarray],
        feat_quality: Optional[np.ndarray] = None,
        feat_mode: Optional[np.ndarray] = None,
        frame_id: int = 0,
    ) -> List[STrack]:
        self.frame_id = int(frame_id)
        self.bad_frame_streak = 0

        dets = np.empty((0, 5), dtype=np.float32) if dets_full is None else dets_full.astype(np.float32)

        for tr in self.tracks:
            tr.age += 1
            tr.time_since_update += 1

        for tr in list(self.lost_tracks):
            tr.age += 1
            tr.time_since_update += 1
            if tr.time_since_update > self.track_buffer:
                self.lost_tracks.remove(tr)

        if len(dets) == 0:
            for tr in self.tracks:
                tr.is_activated = False
                if tr not in self.lost_tracks:
                    self.lost_tracks.append(tr)
            self.tracks = []
            return self.tracks

        new_active: List[STrack] = []
        used_tracks = set()
        used_dets = set()

        assigned_active, _ = self._match_active_tracks(self.tracks, dets, feats)
        for tr, di in assigned_active:
            x1, y1, x2, y2, _ = dets[di]
            new_box = [x1, y1, x2, y2]

            tr.update_tlbr(new_box)
            allow_feat_update = (
                feats is not None
                and len(feats) > 0
                and self._feat_ok(feats, feat_quality, feat_mode, di)
                and self._should_update_feature(tr, new_box)
                and (not self._is_overlap_ambiguous(tr, self.tracks))
            )
            if allow_feat_update:
                tr.update_feature(feats[di])

            tr.mark_hit()
            tr.last_frame = self.frame_id
            tr.time_since_update = 0
            tr.is_activated = True
            tr.last_feat_quality = float(feat_quality[di]) if feat_quality is not None and di < len(feat_quality) else 0.0
            tr.last_det_conf = float(dets[di][4])
            tr.last_reid_mode = int(feat_mode[di]) if feat_mode is not None and di < len(feat_mode) else 0

            new_active.append(tr)
            used_tracks.add(tr)
            used_dets.add(di)

        for tr in self.tracks:
            if tr not in used_tracks:
                tr.is_activated = False
                if tr not in self.lost_tracks:
                    self.lost_tracks.append(tr)

        recent_lost = [tr for tr in self.lost_tracks if tr.time_since_update <= self.motion_max_gap]
        det_indices_for_lost = [i for i in range(len(dets)) if i not in used_dets]

        if recent_lost and det_indices_for_lost:
            dets_lost = dets[det_indices_for_lost]
            feats_lost = feats[det_indices_for_lost] if feats is not None and len(feats) > 0 else feats

            assigned_lost, _ = self._match_lost_tracks(recent_lost, dets_lost, feats_lost)
            for tr, idx_local in assigned_lost:
                di = det_indices_for_lost[idx_local]
                x1, y1, x2, y2, _ = dets[di]
                new_box = [x1, y1, x2, y2]

                tr.update_tlbr(new_box)
                allow_feat_update = (
                    feats is not None
                    and len(feats) > 0
                    and self._feat_ok(feats, feat_quality, feat_mode, di)
                    and self._should_update_feature(tr, new_box)
                    and (not self._is_overlap_ambiguous(tr, recent_lost))
                )
                if allow_feat_update:
                    tr.update_feature(feats[di])

                tr.mark_hit()
                tr.last_frame = self.frame_id
                tr.time_since_update = 0
                tr.is_activated = True
                tr.last_feat_quality = float(feat_quality[di]) if feat_quality is not None and di < len(feat_quality) else 0.0
                tr.last_det_conf = float(dets[di][4])
                tr.last_reid_mode = int(feat_mode[di]) if feat_mode is not None and di < len(feat_mode) else 0

                if tr in self.lost_tracks:
                    self.lost_tracks.remove(tr)

                new_active.append(tr)
                used_dets.add(di)

        # Reconnect long-lost tracks by strong appearance only (for long occlusions / heavy overlap).
        long_lost = [
            tr for tr in self.lost_tracks
            if self.motion_max_gap < tr.time_since_update <= self.track_buffer
        ]
        det_indices_for_long = [i for i in range(len(dets)) if i not in used_dets]
        if long_lost and det_indices_for_long and feats is not None and len(feats) > 0:
            dets_long = dets[det_indices_for_long]
            feats_long = feats[det_indices_for_long]
            assigned_long, _ = self._match_long_lost_tracks(long_lost, dets_long, feats_long)
            for tr, idx_local in assigned_long:
                di = det_indices_for_long[idx_local]
                x1, y1, x2, y2, _ = dets[di]
                new_box = [x1, y1, x2, y2]

                tr.update_tlbr(new_box)
                if feats is not None and len(feats) > 0 and self._feat_ok(feats, feat_quality, feat_mode, di):
                    tr.update_feature(feats[di])

                tr.mark_hit()
                tr.last_frame = self.frame_id
                tr.time_since_update = 0
                tr.is_activated = True
                tr.last_feat_quality = float(feat_quality[di]) if feat_quality is not None and di < len(feat_quality) else 0.0
                tr.last_det_conf = float(dets[di][4])
                tr.last_reid_mode = int(feat_mode[di]) if feat_mode is not None and di < len(feat_mode) else 0

                if tr in self.lost_tracks:
                    self.lost_tracks.remove(tr)

                new_active.append(tr)
                used_dets.add(di)

        for di in range(len(dets)):
            if di in used_dets:
                continue

            x1, y1, x2, y2, s = dets[di]
            if float(s) < self.track_thresh:
                continue

            feat_i = feats[di] if self._feat_ok(feats, feat_quality, feat_mode, di) else None
            tr = STrack(
                [x1, y1, x2, y2],
                float(s),
                feature=feat_i,
                feature_history=self.feature_history,
                feature_update_min_sim=self.feature_update_min_sim,
                confirm_hits=self.confirm_hits,
            )
            tr.last_frame = self.frame_id
            tr.time_since_update = 0
            tr.is_activated = True
            tr.last_feat_quality = float(feat_quality[di]) if feat_quality is not None and di < len(feat_quality) else 0.0
            tr.last_det_conf = float(s)
            tr.last_reid_mode = int(feat_mode[di]) if feat_mode is not None and di < len(feat_mode) else 0
            new_active.append(tr)

        self.tracks = new_active

        active_set = set(self.tracks)
        self.lost_tracks = [tr for tr in self.lost_tracks if tr not in active_set]

        return self.tracks
