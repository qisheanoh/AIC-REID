from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional, Set
import csv

import cv2
import numpy as np
try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None

from src.trackers.bot_sort import attire_descriptor, body_shape_descriptor
from src.reid.extractor import ReidExtractor


def _l2(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = v.astype(np.float32)
    n = np.linalg.norm(v) + eps
    return v / n


def _cos(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return -1.0
    a = _l2(a)
    b = _l2(b)
    return float(np.dot(a, b))


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


def _safe_region_stats(region_bgr: np.ndarray) -> np.ndarray:
    if region_bgr is None or region_bgr.size == 0:
        return np.zeros((6,), dtype=np.float32)
    hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    mean = hsv.reshape(-1, 3).mean(axis=0).astype(np.float32)
    std = hsv.reshape(-1, 3).std(axis=0).astype(np.float32)
    v = np.concatenate([mean, std], axis=0).astype(np.float32)
    return v


def color_signature(crop_bgr: np.ndarray) -> np.ndarray:
    if crop_bgr is None or crop_bgr.size == 0:
        return np.zeros((18,), dtype=np.float32)
    h, _ = crop_bgr.shape[:2]
    y1 = int(0.00 * h)
    y2 = int(0.42 * h)
    y3 = int(0.75 * h)
    upper = crop_bgr[y1:y2, :]
    middle = crop_bgr[y2:y3, :]
    lower = crop_bgr[y3:, :]
    sig = np.concatenate([
        1.20 * _safe_region_stats(upper),
        0.75 * _safe_region_stats(middle),
        1.10 * _safe_region_stats(lower),
    ], axis=0).astype(np.float32)
    return _l2(sig)


@dataclass
class TrackRow:
    idx: int
    frame_idx: int
    gid: int
    x1: float
    y1: float
    x2: float
    y2: float
    raw: dict

    @property
    def box(self) -> Tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def center(self) -> Tuple[float, float]:
        return 0.5 * (self.x1 + self.x2), 0.5 * (self.y1 + self.y2)

    @property
    def h(self) -> float:
        return max(1.0, self.y2 - self.y1)


@dataclass
class TrackStats:
    gid: int
    start_f: int
    end_f: int
    first_row: TrackRow
    last_row: TrackRow
    rows: List[TrackRow]
    first_desc: Optional[np.ndarray] = None
    last_desc: Optional[np.ndarray] = None
    mean_desc: Optional[np.ndarray] = None
    first_descs: List[np.ndarray] = field(default_factory=list)
    last_descs: List[np.ndarray] = field(default_factory=list)
    sample_descs: List[np.ndarray] = field(default_factory=list)
    start_zone: str = "center"
    end_zone: str = "center"
    median_h_ratio: float = 0.0
    median_aspect: float = 0.0
    dir_vec: np.ndarray = field(default_factory=lambda: np.zeros((2,), dtype=np.float32))


@dataclass
class IdentityState:
    gid: int
    feat: np.ndarray
    first_feat: np.ndarray
    last_frame: int
    last_box: Tuple[float, float, float, float]
    vx: float = 0.0
    vy: float = 0.0
    hits: int = 1


def _build_desc(
    frame: np.ndarray,
    row: TrackRow,
    *,
    extractor: Optional[ReidExtractor] = None,
    os_dim: int = 512,
) -> Optional[np.ndarray]:
    h, w = frame.shape[:2]
    x1 = max(0, min(w - 1, int(row.x1)))
    y1 = max(0, min(h - 1, int(row.y1)))
    x2 = max(0, min(w - 1, int(row.x2)))
    y2 = max(0, min(h - 1, int(row.y2)))
    if x2 <= x1 + 4 or y2 <= y1 + 8:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop is None or crop.size == 0:
        return None

    d_attire = attire_descriptor(crop).astype(np.float32)
    d_shape = body_shape_descriptor(np.array([row.x1, row.y1, row.x2, row.y2], dtype=np.float32), frame_h=h, frame_w=w)
    d_color = color_signature(crop).astype(np.float32)

    d_os = None
    if extractor is not None:
        try:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            d_os = extractor([crop_rgb]).astype(np.float32)
            if d_os.ndim == 2:
                d_os = d_os[0]
        except Exception:
            d_os = None

    if d_os is None:
        d_os = np.zeros((max(1, int(os_dim)),), dtype=np.float32)

    # Balance descriptor groups by signal value, not raw dimensionality.
    # OSNet has many more channels than attire/color, so keep it as a support cue.
    desc = np.concatenate([
        0.22 * _l2(d_os),
        1.10 * d_attire,
        0.34 * d_shape,
        1.12 * d_color,
    ], axis=0).astype(np.float32)
    return _l2(desc)


def _load_rows(csv_path: Path) -> Tuple[List[str], List[TrackRow]]:
    import sys as _sys
    csv.field_size_limit(min(_sys.maxsize, 2 ** 31 - 1))
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows: List[TrackRow] = []
        for idx, raw in enumerate(reader):
            try:
                _fi_raw = str(raw["frame_idx"]).replace("\x00", "").strip()
                _gid_raw = str(raw["global_id"]).replace("\x00", "").strip()
                row = TrackRow(
                    idx=int(idx),
                    frame_idx=int(round(float(_fi_raw))),
                    gid=int(round(float(_gid_raw))) if _gid_raw else 0,
                    x1=float(raw["x1"]),
                    y1=float(raw["y1"]),
                    x2=float(raw["x2"]),
                    y2=float(raw["y2"]),
                    raw=dict(raw),
                )
            except Exception:
                continue
            rows.append(row)
    rows.sort(key=lambda r: (r.frame_idx, r.gid))
    return fieldnames, rows


def _row_center_h(row: TrackRow) -> Tuple[float, float, float]:
    cx, cy = row.center
    return float(cx), float(cy), float(max(1.0, row.h))


def _predict_state_center(st: IdentityState, frame_idx: int) -> Tuple[float, float, float]:
    x1, y1, x2, y2 = st.last_box
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    h = max(1.0, y2 - y1)
    gap = max(1, int(frame_idx) - int(st.last_frame))
    return float(cx + st.vx * gap), float(cy + st.vy * gap), float(h)


def _update_state(st: IdentityState, row: TrackRow, desc: np.ndarray, frame_idx: int) -> None:
    prev_x1, prev_y1, prev_x2, prev_y2 = st.last_box
    prev_cx = 0.5 * (prev_x1 + prev_x2)
    prev_cy = 0.5 * (prev_y1 + prev_y2)

    new_cx, new_cy, _ = _row_center_h(row)
    dt = max(1, int(frame_idx) - int(st.last_frame))
    vx = (new_cx - prev_cx) / float(dt)
    vy = (new_cy - prev_cy) / float(dt)
    st.vx = 0.72 * st.vx + 0.28 * vx
    st.vy = 0.72 * st.vy + 0.28 * vy

    sim_recent = _cos(st.feat, desc)
    alpha = 0.90 if sim_recent >= 0.62 else 0.96
    st.feat = _l2(alpha * st.feat + (1.0 - alpha) * desc)
    st.last_frame = int(frame_idx)
    st.last_box = (float(row.x1), float(row.y1), float(row.x2), float(row.y2))
    st.hits += 1


def _match_score(
    st: IdentityState,
    row: TrackRow,
    desc: np.ndarray,
    frame_idx: int,
    *,
    active_max_gap: int,
) -> float:
    gap = int(frame_idx) - int(st.last_frame)
    if gap <= 0:
        return -1.0
    if gap > active_max_gap:
        return -1.0

    sim_recent = _cos(desc, st.feat)
    sim_first = _cos(desc, st.first_feat)
    if sim_recent < 0.55 and sim_first < 0.60:
        return -1.0

    pred_cx, pred_cy, pred_h = _predict_state_center(st, frame_idx)
    cx, cy, h = _row_center_h(row)
    dist_norm = float(np.hypot(cx - pred_cx, cy - pred_cy) / max(pred_h, h, 1.0))
    spatial = float(np.exp(-dist_norm))

    if spatial < 0.14 and max(sim_recent, sim_first) < 0.83:
        return -1.0

    score = 0.56 * sim_recent + 0.28 * sim_first + 0.16 * spatial
    score -= 0.06 * min(1.0, gap / float(active_max_gap))
    return float(score)


def _build_descs_for_rows(
    frame: np.ndarray,
    rows: List[TrackRow],
    *,
    extractor: Optional[ReidExtractor] = None,
    os_dim: int = 512,
) -> Dict[int, np.ndarray]:
    h, w = frame.shape[:2]
    cached = []
    crops_rgb = []

    for row in rows:
        x1 = max(0, min(w - 1, int(row.x1)))
        y1 = max(0, min(h - 1, int(row.y1)))
        x2 = max(0, min(w - 1, int(row.x2)))
        y2 = max(0, min(h - 1, int(row.y2)))
        if x2 <= x1 + 4 or y2 <= y1 + 8:
            continue

        crop = frame[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            continue

        d_attire = attire_descriptor(crop).astype(np.float32)
        d_shape = body_shape_descriptor(
            np.array([row.x1, row.y1, row.x2, row.y2], dtype=np.float32),
            frame_h=h,
            frame_w=w,
        )
        d_color = color_signature(crop).astype(np.float32)

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crops_rgb.append(rgb)
        cached.append((row.idx, d_attire, d_shape, d_color))

    out: Dict[int, np.ndarray] = {}
    if not cached:
        return out

    os_feats = None
    if extractor is not None:
        try:
            os_feats = extractor(crops_rgb).astype(np.float32)
            if os_feats.ndim == 1:
                os_feats = os_feats[None, :]
        except Exception:
            os_feats = None

    for i, (idx, d_attire, d_shape, d_color) in enumerate(cached):
        if os_feats is not None and i < len(os_feats):
            d_os = _l2(os_feats[i])
        else:
            d_os = np.zeros((max(1, int(os_dim)),), dtype=np.float32)
        # Keep consistent weighting with _build_desc().
        desc = np.concatenate([
            0.22 * d_os,
            1.10 * d_attire,
            0.34 * d_shape,
            1.12 * d_color,
        ], axis=0).astype(np.float32)
        out[int(idx)] = _l2(desc)
    return out


def _build_track_stats(rows: List[TrackRow]) -> Dict[int, TrackStats]:
    by_gid: Dict[int, List[TrackRow]] = defaultdict(list)
    for r in rows:
        if int(r.gid) <= 0:
            continue
        by_gid[int(r.gid)].append(r)

    out: Dict[int, TrackStats] = {}
    for gid, rs in by_gid.items():
        rs.sort(key=lambda x: x.frame_idx)
        out[gid] = TrackStats(
            gid=gid,
            start_f=rs[0].frame_idx,
            end_f=rs[-1].frame_idx,
            first_row=rs[0],
            last_row=rs[-1],
            rows=rs,
        )
    return out


def _build_track_segments(
    rows: List[TrackRow],
    *,
    max_gap_frames: int = 18,
) -> Tuple[Dict[int, TrackStats], Dict[int, int], Dict[int, int]]:
    by_gid: Dict[int, List[TrackRow]] = defaultdict(list)
    for r in rows:
        if int(r.gid) <= 0:
            continue
        by_gid[int(r.gid)].append(r)

    seg_tracks: Dict[int, TrackStats] = {}
    row_to_seg: Dict[int, int] = {}
    seg_to_src_gid: Dict[int, int] = {}
    next_seg = 1

    def _finalize_segment(src_gid: int, seg_rows: List[TrackRow], seg_id: int) -> None:
        if not seg_rows:
            return
        seg_rows.sort(key=lambda x: x.frame_idx)
        seg_tracks[int(seg_id)] = TrackStats(
            gid=int(seg_id),
            start_f=int(seg_rows[0].frame_idx),
            end_f=int(seg_rows[-1].frame_idx),
            first_row=seg_rows[0],
            last_row=seg_rows[-1],
            rows=list(seg_rows),
        )
        seg_to_src_gid[int(seg_id)] = int(src_gid)
        for rr in seg_rows:
            row_to_seg[int(rr.idx)] = int(seg_id)

    for src_gid, rs in by_gid.items():
        rs.sort(key=lambda x: x.frame_idx)
        cur: List[TrackRow] = []
        last_f: Optional[int] = None
        for r in rs:
            if last_f is not None and (int(r.frame_idx) - int(last_f)) > int(max_gap_frames):
                _finalize_segment(int(src_gid), cur, int(next_seg))
                next_seg += 1
                cur = []
            cur.append(r)
            last_f = int(r.frame_idx)
        _finalize_segment(int(src_gid), cur, int(next_seg))
        next_seg += 1

    return seg_tracks, row_to_seg, seg_to_src_gid


def _sample_rows_for_desc(stats: TrackStats, n_each: int = 6, n_mid: int = 4) -> Tuple[List[TrackRow], List[TrackRow], List[TrackRow]]:
    rs = stats.rows
    if len(rs) <= 2 * n_each:
        mid = max(1, len(rs) // 2)
        first = rs[:mid]
        last = rs[mid:]
        return first, last, []

    first = rs[:n_each]
    last = rs[-n_each:]
    mid_rows: List[TrackRow] = []
    if n_mid > 0 and len(rs) > (2 * n_each + 2):
        core = rs[n_each:-n_each]
        if core:
            picks = np.linspace(0, len(core) - 1, num=min(n_mid, len(core)), dtype=np.int32)
            mid_rows = [core[int(i)] for i in picks]
    return first, last, mid_rows


def _pick_clean_rows(
    rows: List[TrackRow],
    frame_rows: Dict[int, List[TrackRow]],
    *,
    max_iou: float = 0.22,
    keep_top: int = 6,
) -> List[TrackRow]:
    if not rows:
        return []
    scored: List[Tuple[float, float, TrackRow]] = []
    for r in rows:
        max_overlap = 0.0
        for o in frame_rows.get(int(r.frame_idx), []):
            if int(o.idx) == int(r.idx):
                continue
            iou = _iou_xyxy(r.box, o.box)
            if iou > max_overlap:
                max_overlap = float(iou)
        area = max(1.0, float(r.x2 - r.x1) * float(r.y2 - r.y1))
        scored.append((float(max_overlap), float(-area), r))
    scored.sort(key=lambda x: (x[0], x[1]))
    clean = [r for ov, _neg_area, r in scored if float(ov) <= float(max_iou)]
    if clean:
        return clean[: max(1, int(keep_top))]
    # Fallback: keep lowest-overlap rows if no row passes strict gate.
    return [r for _ov, _na, r in scored[: max(1, min(int(keep_top), len(scored)))]]


def _extract_track_descriptors(
    video_path: Path,
    tracks: Dict[int, TrackStats],
    *,
    reid_weights_path: str | None = None,
) -> None:
    extractor: Optional[ReidExtractor] = None
    extractor_error: str = ""
    os_dim = 512
    # Prefer CPU here to avoid VRAM contention with the online tracker model.
    # If CPU extractor creation fails, fallback to default device.
    try:
        extractor = ReidExtractor(model_name="osnet_x1_0", device="cpu", model_path=reid_weights_path)
    except Exception as e_cpu:
        extractor_error = repr(e_cpu)
        try:
            extractor = ReidExtractor(model_name="osnet_x1_0", device=None, model_path=reid_weights_path)
        except Exception as e_fallback:
            extractor_error = f"{repr(e_cpu)} | fallback: {repr(e_fallback)}"
            extractor = None

    need_by_frame: Dict[int, List[Tuple[int, str, TrackRow]]] = defaultdict(list)
    frame_rows: Dict[int, List[TrackRow]] = defaultdict(list)
    for st in tracks.values():
        for r in st.rows:
            frame_rows[int(r.frame_idx)].append(r)

    for gid, st in tracks.items():
        first_rows, last_rows, mid_rows = _sample_rows_for_desc(st, n_each=6, n_mid=4)
        first_rows = _pick_clean_rows(first_rows, frame_rows, max_iou=0.20, keep_top=6)
        last_rows = _pick_clean_rows(last_rows, frame_rows, max_iou=0.20, keep_top=6)
        mid_rows = _pick_clean_rows(mid_rows, frame_rows, max_iou=0.18, keep_top=4)
        for r in first_rows:
            need_by_frame[r.frame_idx].append((gid, "first", r))
        for r in last_rows:
            need_by_frame[r.frame_idx].append((gid, "last", r))
        for r in mid_rows:
            need_by_frame[r.frame_idx].append((gid, "mid", r))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for stitching: {video_path}")

    frame_idx = 0
    first_descs: Dict[int, List[np.ndarray]] = defaultdict(list)
    last_descs: Dict[int, List[np.ndarray]] = defaultdict(list)
    mid_descs: Dict[int, List[np.ndarray]] = defaultdict(list)

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        asks = need_by_frame.get(frame_idx, [])
        if asks:
            for gid, role, row in asks:
                d = _build_desc(frame, row, extractor=extractor, os_dim=os_dim)
                if d is None:
                    continue
                if role == "first":
                    first_descs[gid].append(d)
                elif role == "last":
                    last_descs[gid].append(d)
                else:
                    mid_descs[gid].append(d)
        frame_idx += 1

    cap.release()

    for gid, st in tracks.items():
        fd = first_descs.get(gid, [])
        ld = last_descs.get(gid, [])
        md = mid_descs.get(gid, [])

        if fd:
            st.first_desc = _l2(np.mean(np.stack(fd, axis=0), axis=0))
        if ld:
            st.last_desc = _l2(np.mean(np.stack(ld, axis=0), axis=0))

        combined = []
        if fd:
            combined.extend(fd)
        if md:
            combined.extend(md)
        if ld:
            combined.extend(ld)
        if combined:
            st.mean_desc = _l2(np.mean(np.stack(combined, axis=0), axis=0))
            st.sample_descs = combined[:]
            st.first_descs = fd[:]
            st.last_descs = ld[:]


def reassign_ids_with_memory(
    *,
    video_path: Path,
    tracks_csv_path: Path,
    reid_weights_path: str | None = None,
    min_assign_score: float = 0.62,
    active_max_gap: int = 90,
    long_reid_gap: int = 320,
    long_reid_sim: float = 0.86,
) -> dict:
    fieldnames, rows = _load_rows(tracks_csv_path)
    if not rows:
        return {"total_ids": 0, "reassigned_rows": 0}

    by_frame: Dict[int, List[TrackRow]] = defaultdict(list)
    for r in rows:
        by_frame[int(r.frame_idx)].append(r)

    extractor: Optional[ReidExtractor] = None
    os_dim = 512
    try:
        extractor = ReidExtractor(
            model_name="osnet_x1_0",
            device=None,
            model_path=reid_weights_path,
        )
    except Exception:
        extractor = None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for reassign: {video_path}")

    states: Dict[int, IdentityState] = {}
    next_gid = 1
    id_by_idx: Dict[int, int] = {}
    local_gid_map: Dict[int, int] = {}
    frame_idx = 0

    def _mapped_gid_for_local(local_gid: int) -> int:
        nonlocal next_gid
        lg = int(local_gid)
        if lg <= 0:
            gid = int(next_gid)
            next_gid += 1
            return gid
        if lg not in local_gid_map:
            local_gid_map[lg] = int(next_gid)
            next_gid += 1
        return int(local_gid_map[lg])

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        frame_rows = by_frame.get(frame_idx, [])
        if not frame_rows:
            frame_idx += 1
            continue

        desc_map = _build_descs_for_rows(frame, frame_rows, extractor=extractor, os_dim=os_dim)
        det_rows = [r for r in frame_rows if r.idx in desc_map]

        if not det_rows:
            for r in frame_rows:
                if int(r.idx) in id_by_idx:
                    continue
                gid = _mapped_gid_for_local(int(r.gid))
                id_by_idx[int(r.idx)] = gid
            frame_idx += 1
            continue

        state_ids = [gid for gid, st in states.items() if (frame_idx - st.last_frame) <= active_max_gap]
        det_count = len(det_rows)
        assigned_pairs: List[Tuple[int, int]] = []
        used_states: set[int] = set()
        used_det: set[int] = set()

        if state_ids:
            big = 1e6
            cost = np.full((len(state_ids), det_count), big, dtype=np.float32)
            score_matrix = np.full((len(state_ids), det_count), -1.0, dtype=np.float32)
            for si, gid in enumerate(state_ids):
                st = states[gid]
                for di, row in enumerate(det_rows):
                    score = _match_score(st, row, desc_map[int(row.idx)], frame_idx, active_max_gap=active_max_gap)
                    score_matrix[si, di] = float(score)
                    if score >= min_assign_score:
                        cost[si, di] = -float(score)

            if linear_sum_assignment is not None:
                rr, cc = linear_sum_assignment(cost)
                for si, di in zip(rr, cc):
                    if cost[si, di] >= big * 0.5:
                        continue
                    gid = int(state_ids[si])
                    assigned_pairs.append((gid, di))
                    used_states.add(gid)
                    used_det.add(di)
            else:
                cands = []
                for si, gid in enumerate(state_ids):
                    for di in range(det_count):
                        sc = float(score_matrix[si, di])
                        if sc >= min_assign_score:
                            cands.append((sc, gid, di))
                cands.sort(reverse=True)
                for sc, gid, di in cands:
                    if gid in used_states or di in used_det:
                        continue
                    assigned_pairs.append((gid, di))
                    used_states.add(gid)
                    used_det.add(di)

        # Apply active matches.
        for gid, di in assigned_pairs:
            row = det_rows[di]
            desc = desc_map[int(row.idx)]
            _update_state(states[gid], row, desc, frame_idx)
            id_by_idx[int(row.idx)] = int(gid)

        # Handle unmatched detections.
        for di, row in enumerate(det_rows):
            if di in used_det:
                continue
            desc = desc_map[int(row.idx)]

            best_gid = -1
            best_score = -1.0
            for gid, st in states.items():
                if gid in used_states:
                    continue
                gap = frame_idx - st.last_frame
                if gap <= active_max_gap or gap > long_reid_gap:
                    continue
                sim_recent = _cos(desc, st.feat)
                sim_first = _cos(desc, st.first_feat)
                sim = max(sim_recent, sim_first)
                if sim < long_reid_sim:
                    continue
                score = 0.64 * sim_recent + 0.36 * sim_first
                if score > best_score:
                    best_score = float(score)
                    best_gid = int(gid)

            if best_gid > 0:
                _update_state(states[best_gid], row, desc, frame_idx)
                id_by_idx[int(row.idx)] = int(best_gid)
                used_states.add(int(best_gid))
                continue

            gid = _mapped_gid_for_local(int(row.gid))
            states[gid] = IdentityState(
                gid=gid,
                feat=desc.copy(),
                first_feat=desc.copy(),
                last_frame=int(frame_idx),
                last_box=(float(row.x1), float(row.y1), float(row.x2), float(row.y2)),
                vx=0.0,
                vy=0.0,
                hits=1,
            )
            id_by_idx[int(row.idx)] = int(gid)
            used_states.add(int(gid))

        # Rows without descriptor fallback.
        for r in frame_rows:
            if int(r.idx) in id_by_idx:
                continue
            gid = _mapped_gid_for_local(int(r.gid))
            id_by_idx[int(r.idx)] = int(gid)

        frame_idx += 1

    cap.release()

    # Enforce per-frame uniqueness for positive IDs to avoid same-ID double assignment.
    rows_by_frame_idx: Dict[int, List[TrackRow]] = defaultdict(list)
    for r in rows:
        rows_by_frame_idx[int(r.frame_idx)].append(r)
    for fr, fr_rows in rows_by_frame_idx.items():
        gid_to_rows: Dict[int, List[TrackRow]] = defaultdict(list)
        for r in fr_rows:
            gid = int(id_by_idx.get(int(r.idx), 0))
            if gid > 0:
                gid_to_rows[gid].append(r)
        for gid, rs in gid_to_rows.items():
            if len(rs) <= 1:
                continue
            keep = max(rs, key=lambda rr: max(0.0, rr.x2 - rr.x1) * max(0.0, rr.y2 - rr.y1))
            for rr in rs:
                if rr.idx == keep.idx:
                    continue
                id_by_idx[int(rr.idx)] = 0

    # Rewrite CSV with reassigned IDs.
    with tracks_csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        out_rows = []
        for idx, raw in enumerate(reader):
            gid = int(id_by_idx.get(int(idx), int(raw.get("global_id", 0) or 0)))
            raw["global_id"] = str(gid)
            out_rows.append(raw)

    with tracks_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    total_ids = len(set(int(v) for v in id_by_idx.values()))
    return {"total_ids": int(total_ids), "reassigned_rows": int(len(id_by_idx))}


def _spatial_boundary_score(a: TrackStats, b: TrackStats) -> float:
    cax, cay = a.last_row.center
    cbx, cby = b.first_row.center
    h = max(1.0, 0.5 * (a.last_row.h + b.first_row.h))
    dist_norm = float(np.hypot(cax - cbx, cay - cby) / h)
    return float(np.exp(-dist_norm))


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


def _zone_compatibility(prev_zone: str, cur_zone: str, gap: int, max_gap: int) -> float:
    if prev_zone == cur_zone:
        z = 1.0
    elif prev_zone == "center" or cur_zone == "center":
        z = 0.62
    elif {prev_zone, cur_zone} == {"left", "right"}:
        z = 0.24
    elif {prev_zone, cur_zone} == {"top", "bottom"}:
        z = 0.24
    else:
        z = 0.38

    if gap > 0:
        z -= 0.10 * min(1.0, float(gap) / max(1.0, float(max_gap)))
    return float(np.clip(z, 0.0, 1.0))


def _topk_similarity(a_descs: List[np.ndarray], b_descs: List[np.ndarray], k: int = 4) -> float:
    if not a_descs or not b_descs:
        return -1.0
    sims: List[float] = []
    for da in a_descs:
        for db in b_descs:
            sims.append(_cos(da, db))
    if not sims:
        return -1.0
    sims.sort(reverse=True)
    kk = max(1, min(int(k), len(sims)))
    return float(np.mean(sims[:kk]))


def _enrich_track_meta(tracks: Dict[int, TrackStats], frame_w: int, frame_h: int) -> None:
    fw = max(1.0, float(frame_w))
    fh = max(1.0, float(frame_h))
    for st in tracks.values():
        centers = np.array([r.center for r in st.rows], dtype=np.float32) if st.rows else np.zeros((0, 2), dtype=np.float32)
        hs = np.array([max(1.0, r.h) for r in st.rows], dtype=np.float32) if st.rows else np.zeros((0,), dtype=np.float32)
        ws = np.array([max(1.0, r.x2 - r.x1) for r in st.rows], dtype=np.float32) if st.rows else np.zeros((0,), dtype=np.float32)

        if len(hs) > 0:
            st.median_h_ratio = float(np.median(hs) / fh)
            st.median_aspect = float(np.median(ws / hs))
        else:
            st.median_h_ratio = 0.0
            st.median_aspect = 0.0

        s_cx, s_cy = st.first_row.center
        e_cx, e_cy = st.last_row.center
        st.start_zone = _zone_of_point(s_cx, s_cy, frame_w=frame_w, frame_h=frame_h)
        st.end_zone = _zone_of_point(e_cx, e_cy, frame_w=frame_w, frame_h=frame_h)

        dv = np.array([e_cx - s_cx, e_cy - s_cy], dtype=np.float32)
        n = float(np.linalg.norm(dv))
        if n > 1e-6:
            dv = dv / n
        st.dir_vec = dv


def _components_overlap(a_ids: set[int], b_ids: set[int], tracks: Dict[int, TrackStats], max_overlap_frames: int = 3) -> bool:
    for ga in a_ids:
        ta = tracks[int(ga)]
        for gb in b_ids:
            tb = tracks[int(gb)]
            ov = max(0, min(ta.end_f, tb.end_f) - max(ta.start_f, tb.start_f) + 1)
            if ov > max_overlap_frames:
                return True
    return False


def _enforce_unique_positive_ids_per_frame(*, tracks_csv_path: Path) -> int:
    fieldnames, rows = _load_rows(tracks_csv_path)
    if not rows:
        return 0

    id_by_idx = {int(r.idx): int(r.gid) for r in rows}
    gid_row_count: Dict[int, int] = defaultdict(int)
    gid_frame_span: Dict[int, Tuple[int, int]] = {}
    for r in rows:
        g = int(r.gid)
        if g <= 0:
            continue
        gid_row_count[g] += 1
        fr = int(r.frame_idx)
        if g not in gid_frame_span:
            gid_frame_span[g] = (fr, fr)
        else:
            lo, hi = gid_frame_span[g]
            gid_frame_span[g] = (min(lo, fr), max(hi, fr))

    by_frame: Dict[int, List[TrackRow]] = defaultdict(list)
    for r in rows:
        by_frame[int(r.frame_idx)].append(r)

    changed = 0
    for fr_rows in by_frame.values():
        by_gid: Dict[int, List[TrackRow]] = defaultdict(list)
        for r in fr_rows:
            gid = int(id_by_idx.get(int(r.idx), int(r.gid)))
            if gid > 0:
                by_gid[gid].append(r)
        for gid, rs in by_gid.items():
            if len(rs) <= 1:
                continue
            def _dup_keep_score(rr: TrackRow) -> float:
                bw = max(1.0, float(rr.x2 - rr.x1))
                bh = max(1.0, float(rr.y2 - rr.y1))
                ar = float(bw / bh)
                area = float(bw * bh)
                shape_ok = 1.0 if 0.18 <= ar <= 1.10 else 0.0
                rows_n = float(gid_row_count.get(int(gid), 0))
                lo, hi = gid_frame_span.get(int(gid), (int(rr.frame_idx), int(rr.frame_idx)))
                span = float(max(1, hi - lo + 1))
                return float(0.58 * area + 0.20 * rows_n + 0.14 * span + 0.08 * shape_ok)

            keep = max(rs, key=_dup_keep_score)
            for rr in rs:
                if int(rr.idx) == int(keep.idx):
                    continue
                if int(id_by_idx.get(int(rr.idx), 0)) > 0:
                    id_by_idx[int(rr.idx)] = 0
                    changed += 1

    if changed <= 0:
        return 0

    with tracks_csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        out_rows = []
        for idx, raw in enumerate(reader):
            gid = int(id_by_idx.get(int(idx), int(raw.get("global_id", 0) or 0)))
            raw["global_id"] = str(gid)
            out_rows.append(raw)

    with tracks_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    return int(changed)


def _count_same_frame_positive_id_conflicts(rows: List[TrackRow]) -> int:
    by_frame_gid: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        gid = int(r.gid)
        if gid <= 0:
            continue
        by_frame_gid[int(r.frame_idx)][int(gid)] += 1

    conflicts = 0
    for by_gid in by_frame_gid.values():
        for cnt in by_gid.values():
            if int(cnt) > 1:
                conflicts += int(cnt) - 1
    return int(conflicts)


def summarize_identity_space(
    *,
    tracks_csv_path: Path,
    stable_min_rows: int = 30,
    stable_min_span: int = 32,
    canonical_ids: Optional[Set[int]] = None,
) -> dict:
    _, rows = _load_rows(tracks_csv_path)
    if not rows:
        return {
            "applied": False,
            "reason": "empty_tracks",
            "rows_total": 0,
            "positive_rows": 0,
            "unique_positive_ids": [],
            "stable_positive_ids": [],
            "stable_positive_id_count": 0,
            "max_positive_id": 0,
            "same_frame_duplicate_positive_ids": 0,
            "non_canonical_positive_ids": [],
        }

    by_gid_rows: Dict[int, int] = defaultdict(int)
    by_gid_first: Dict[int, int] = {}
    by_gid_last: Dict[int, int] = {}
    positive_rows = 0
    for r in rows:
        gid = int(r.gid)
        if gid <= 0:
            continue
        positive_rows += 1
        by_gid_rows[int(gid)] += 1
        fr = int(r.frame_idx)
        if gid not in by_gid_first:
            by_gid_first[int(gid)] = int(fr)
            by_gid_last[int(gid)] = int(fr)
        else:
            by_gid_first[int(gid)] = min(int(by_gid_first[int(gid)]), int(fr))
            by_gid_last[int(gid)] = max(int(by_gid_last[int(gid)]), int(fr))

    unique_positive_ids = sorted(int(g) for g in by_gid_rows.keys() if int(g) > 0)
    stable_positive_ids: List[int] = []
    for gid in unique_positive_ids:
        rows_n = int(by_gid_rows.get(int(gid), 0))
        span = int(by_gid_last[int(gid)] - by_gid_first[int(gid)] + 1)
        if rows_n >= int(stable_min_rows) and span >= int(stable_min_span):
            stable_positive_ids.append(int(gid))

    canonical_set = set(int(x) for x in (canonical_ids or set()) if int(x) > 0)
    non_canonical = []
    if canonical_set:
        non_canonical = sorted(int(g) for g in unique_positive_ids if int(g) not in canonical_set)

    return {
        "applied": True,
        "rows_total": int(len(rows)),
        "positive_rows": int(positive_rows),
        "unique_positive_ids": [int(g) for g in unique_positive_ids],
        "stable_positive_ids": [int(g) for g in stable_positive_ids],
        "stable_positive_id_count": int(len(stable_positive_ids)),
        "max_positive_id": int(max(unique_positive_ids) if unique_positive_ids else 0),
        "same_frame_duplicate_positive_ids": int(_count_same_frame_positive_id_conflicts(rows)),
        "non_canonical_positive_ids": [int(g) for g in non_canonical],
    }


def evaluate_first_two_minute_audit_metrics(
    *,
    tracks_csv_path: Path,
    audit_csv_path: Path,
    max_sec: float = 120.0,
    iou_threshold: float = 0.34,
) -> dict:
    if not tracks_csv_path.exists():
        return {"applied": False, "reason": "tracks_csv_missing"}
    if not audit_csv_path.exists():
        return {"applied": False, "reason": "audit_csv_missing"}

    def _norm_gt_key(v: str) -> str:
        s = str(v or "").strip()
        if not s:
            return ""
        try:
            fv = float(s)
            if abs(fv - round(fv)) < 1e-6:
                return str(int(round(fv)))
        except Exception:
            pass
        return s

    audit_by_frame: Dict[int, List[Tuple[str, Tuple[float, float, float, float]]]] = defaultdict(list)
    with audit_csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            act = str(raw.get("audit_action", "")).strip().lower()
            if act == "ignore":
                continue
            try:
                ts = float(raw.get("ts_sec", "0") or 0.0)
            except Exception:
                continue
            if ts > float(max_sec):
                continue
            gt = _norm_gt_key(raw.get("gt_person_id", ""))
            if not gt:
                continue
            try:
                fr = int(float(raw["frame_idx"]))
                box = (
                    float(raw["x1"]),
                    float(raw["y1"]),
                    float(raw["x2"]),
                    float(raw["y2"]),
                )
            except Exception:
                continue
            audit_by_frame[int(fr)].append((str(gt), box))

    tracks_by_frame: Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]] = defaultdict(list)
    with tracks_csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            try:
                ts = float(raw.get("ts_sec", "0") or 0.0)
            except Exception:
                continue
            if ts > float(max_sec):
                continue
            try:
                fr = int(float(raw["frame_idx"]))
                gid = int(round(float(raw.get("global_id", "0") or 0)))
                box = (
                    float(raw["x1"]),
                    float(raw["y1"]),
                    float(raw["x2"]),
                    float(raw["y2"]),
                )
            except Exception:
                continue
            tracks_by_frame[int(fr)].append((int(gid), box))

    total_audit_rows = int(sum(len(v) for v in audit_by_frame.values()))
    if total_audit_rows <= 0:
        return {
            "applied": False,
            "reason": "no_audit_rows_in_window",
            "audit_rows_total": 0,
        }

    covered_iou = 0
    covered_positive = 0
    matched_pairs: List[Tuple[str, int]] = []

    for fr, a_rows in audit_by_frame.items():
        t_rows = tracks_by_frame.get(int(fr), [])
        edges: List[Tuple[float, int, int]] = []
        for ai, (_gt, abox) in enumerate(a_rows):
            for ti, (_gid, tbox) in enumerate(t_rows):
                iv = _iou_xyxy(abox, tbox)
                if iv >= float(iou_threshold):
                    edges.append((float(iv), int(ai), int(ti)))
        edges.sort(key=lambda x: x[0], reverse=True)
        used_a: set[int] = set()
        used_t: set[int] = set()
        for _iv, ai, ti in edges:
            if ai in used_a or ti in used_t:
                continue
            used_a.add(ai)
            used_t.add(ti)
            covered_iou += 1
            gt = str(a_rows[ai][0])
            gid = int(t_rows[ti][0])
            if gid > 0:
                covered_positive += 1
                matched_pairs.append((str(gt), int(gid)))

    gt_to_pred: Dict[str, Counter] = defaultdict(Counter)
    pred_to_gt: Dict[int, Counter] = defaultdict(Counter)
    for gt, gid in matched_pairs:
        gt_to_pred[str(gt)][int(gid)] += 1
        pred_to_gt[int(gid)][str(gt)] += 1

    gt_purity_detail: Dict[str, float] = {}
    for gt, cnt in gt_to_pred.items():
        total = int(sum(cnt.values()))
        top = int(cnt.most_common(1)[0][1]) if cnt else 0
        gt_purity_detail[str(gt)] = float(top / max(1, total))

    pred_purity_detail: Dict[int, float] = {}
    for gid, cnt in pred_to_gt.items():
        total = int(sum(cnt.values()))
        top = int(cnt.most_common(1)[0][1]) if cnt else 0
        pred_purity_detail[int(gid)] = float(top / max(1, total))

    gt_to_pred_purity_macro = float(sum(gt_purity_detail.values()) / max(1, len(gt_purity_detail)))
    pred_to_gt_purity_macro = float(sum(pred_purity_detail.values()) / max(1, len(pred_purity_detail)))
    gt_fragmented = int(sum(1 for _gt, cnt in gt_to_pred.items() if len(cnt) > 1))
    pred_shared = int(sum(1 for _gid, cnt in pred_to_gt.items() if len(cnt) > 1))

    same_frame_dups = 0
    window_positive_ids: set[int] = set()
    for fr, t_rows in tracks_by_frame.items():
        gid_cnt: Dict[int, int] = defaultdict(int)
        for gid, _ in t_rows:
            if int(gid) <= 0:
                continue
            gid_cnt[int(gid)] += 1
            window_positive_ids.add(int(gid))
        for c in gid_cnt.values():
            if int(c) > 1:
                same_frame_dups += int(c) - 1

    return {
        "applied": True,
        "audit_rows_total": int(total_audit_rows),
        "audit_rows_covered_iou": int(covered_iou),
        "audit_row_coverage": float(covered_iou / max(1, total_audit_rows)),
        "audit_rows_covered_positive_id": int(covered_positive),
        "audit_row_coverage_positive_id": float(covered_positive / max(1, total_audit_rows)),
        "gt_to_pred_purity_macro": float(gt_to_pred_purity_macro),
        "pred_to_gt_purity_macro": float(pred_to_gt_purity_macro),
        "gt_people_fragmented_multi_pred_ids": int(gt_fragmented),
        "predicted_ids_shared_multi_gt_people": int(pred_shared),
        "same_frame_duplicate_positive_ids": int(same_frame_dups),
        "window_unique_positive_id_count": int(len(window_positive_ids)),
        "window_max_positive_id": int(max(window_positive_ids) if window_positive_ids else 0),
        "gt_purity_detail": {str(k): float(v) for k, v in sorted(gt_purity_detail.items(), key=lambda kv: kv[0])},
        "pred_purity_detail": {int(k): float(v) for k, v in sorted(pred_purity_detail.items())},
    }


def enforce_canonical_id_set_purity_first(
    *,
    tracks_csv_path: Path,
    canonical_ids: Set[int],
    stable_min_rows: int = 30,
    stable_min_span: int = 32,
    preserve_stable_noncanonical: bool = True,
) -> dict:
    canonical = set(int(x) for x in canonical_ids if int(x) > 0)
    if not canonical:
        return {"applied": False, "reason": "empty_canonical_set", "changed_rows": 0}

    fieldnames, rows = _load_rows(tracks_csv_path)
    if not rows:
        return {"applied": False, "reason": "empty_tracks", "changed_rows": 0}

    id_by_idx: Dict[int, int] = {int(r.idx): int(r.gid) for r in rows}
    by_gid_idx: Dict[int, List[int]] = defaultdict(list)
    by_gid_frames: Dict[int, List[int]] = defaultdict(list)
    for r in rows:
        g = int(id_by_idx.get(int(r.idx), int(r.gid)))
        if g > 0:
            by_gid_idx[int(g)].append(int(r.idx))
            by_gid_frames[int(g)].append(int(r.frame_idx))

    noncanonical_ids = sorted(int(g) for g in by_gid_idx.keys() if int(g) not in canonical and int(g) > 0)
    canonical_present_ids = sorted(int(g) for g in by_gid_idx.keys() if int(g) in canonical)
    missing_canonical_ids = sorted(int(g) for g in canonical if int(g) not in set(canonical_present_ids))

    stable_noncanonical_stats: List[tuple[int, int, int, int]] = []
    for gid in noncanonical_ids:
        idxs = by_gid_idx.get(int(gid), [])
        frames = by_gid_frames.get(int(gid), [])
        if not idxs or not frames:
            continue
        rows_n = int(len(idxs))
        span = int(max(frames) - min(frames) + 1)
        if rows_n >= int(stable_min_rows) and span >= int(stable_min_span):
            stable_noncanonical_stats.append((int(gid), rows_n, span, int(min(frames))))

    mapped_noncanonical_to_canonical: Dict[int, int] = {}
    if bool(preserve_stable_noncanonical) and missing_canonical_ids and stable_noncanonical_stats:
        # Keep identity purity first:
        # map stable non-canonical tracks only into currently missing canonical slots
        # (one-to-one), then suppress the remaining non-canonical IDs.
        stable_noncanonical_stats.sort(key=lambda x: (-int(x[1]), -int(x[2]), int(x[3]), int(x[0])))
        for i, item in enumerate(stable_noncanonical_stats):
            if i >= len(missing_canonical_ids):
                break
            src_gid = int(item[0])
            tgt_gid = int(missing_canonical_ids[i])
            if src_gid > 0 and tgt_gid > 0 and src_gid not in mapped_noncanonical_to_canonical:
                mapped_noncanonical_to_canonical[int(src_gid)] = int(tgt_gid)

    removed_noncanonical_ids = sorted(
        int(g)
        for g in noncanonical_ids
        if int(g) not in mapped_noncanonical_to_canonical
    )
    changed_rows = 0
    for src_gid, tgt_gid in mapped_noncanonical_to_canonical.items():
        for idx in by_gid_idx.get(int(src_gid), []):
            if int(id_by_idx.get(int(idx), 0)) != int(tgt_gid):
                id_by_idx[int(idx)] = int(tgt_gid)
                changed_rows += 1

    for gid in removed_noncanonical_ids:
        for idx in by_gid_idx.get(int(gid), []):
            if int(id_by_idx.get(int(idx), 0)) > 0:
                id_by_idx[int(idx)] = 0
                changed_rows += 1

    if changed_rows <= 0:
        summary = summarize_identity_space(
            tracks_csv_path=tracks_csv_path,
            stable_min_rows=int(stable_min_rows),
            stable_min_span=int(stable_min_span),
            canonical_ids=canonical,
        )
        return {
            "applied": False,
            "reason": "already_canonical_only",
            "changed_rows": 0,
            "mapped_noncanonical_to_missing_canonical": {
                str(k): int(v) for k, v in sorted(mapped_noncanonical_to_canonical.items())
            },
            "removed_noncanonical_ids": [int(g) for g in removed_noncanonical_ids],
            "stable_noncanonical_ids": [int(x[0]) for x in stable_noncanonical_stats],
            "missing_canonical_ids": [int(g) for g in missing_canonical_ids],
            "summary": summary,
        }

    with tracks_csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        out_rows = []
        for idx, raw in enumerate(reader):
            gid = int(id_by_idx.get(int(idx), int(raw.get("global_id", 0) or 0)))
            raw["global_id"] = str(gid)
            out_rows.append(raw)

    with tracks_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    dedup_rows = _enforce_unique_positive_ids_per_frame(tracks_csv_path=tracks_csv_path)
    summary = summarize_identity_space(
        tracks_csv_path=tracks_csv_path,
        stable_min_rows=int(stable_min_rows),
        stable_min_span=int(stable_min_span),
        canonical_ids=canonical,
    )
    return {
        "applied": True,
        "changed_rows": int(changed_rows),
        "mapped_noncanonical_to_missing_canonical": {
            str(k): int(v) for k, v in sorted(mapped_noncanonical_to_canonical.items())
        },
        "removed_noncanonical_ids": [int(g) for g in removed_noncanonical_ids],
        "stable_noncanonical_ids": [int(x[0]) for x in stable_noncanonical_stats],
        "missing_canonical_ids": [int(g) for g in missing_canonical_ids],
        "dedup_rows": int(dedup_rows),
        "summary": summary,
    }


def smooth_ids_with_audit(
    *,
    tracks_csv_path: Path,
    audit_csv_path: Path,
    iou_threshold: float = 0.35,
    min_support: int = 3,
    min_ratio: float = 0.20,
) -> dict:
    if not audit_csv_path.exists():
        return {"applied": False, "reason": "audit_csv_missing", "remap_count": 0}

    fieldnames, rows = _load_rows(tracks_csv_path)
    if not rows:
        return {"applied": False, "reason": "empty_tracks", "remap_count": 0}

    by_frame: Dict[int, List[TrackRow]] = defaultdict(list)
    gid_row_count: Dict[int, int] = defaultdict(int)
    for r in rows:
        by_frame[int(r.frame_idx)].append(r)
        if int(r.gid) > 0:
            gid_row_count[int(r.gid)] += 1

    cooccur: Dict[Tuple[int, int], int] = defaultdict(int)
    for fr_rows in by_frame.values():
        gids = sorted({int(r.gid) for r in fr_rows if int(r.gid) > 0})
        for i, a in enumerate(gids):
            for b in gids[i + 1:]:
                cooccur[(int(a), int(b))] += 1

    gt_to_gid_counts: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    gid_to_gt_counts: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    matches: List[Tuple[str, int, int, int, float]] = []  # (gt, row_idx, gid, frame, iou)

    with audit_csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            act = str(raw.get("audit_action", "")).strip().lower()
            if act == "ignore":
                continue
            gt = str(raw.get("gt_person_id", "")).strip()
            if not gt:
                continue
            try:
                fr = int(float(raw["frame_idx"]))
                a_box = (
                    float(raw["x1"]),
                    float(raw["y1"]),
                    float(raw["x2"]),
                    float(raw["y2"]),
                )
            except Exception:
                continue

            cands = by_frame.get(fr, [])
            best_idx = -1
            best_iou = 0.0
            best_gid = 0
            for r in cands:
                gid = int(r.gid)
                if gid <= 0:
                    continue
                iou = _iou_xyxy(a_box, r.box)
                if iou > best_iou:
                    best_iou = float(iou)
                    best_gid = gid
                    best_idx = int(r.idx)
            if best_idx < 0 or best_iou < float(iou_threshold) or best_gid <= 0:
                continue

            gt_to_gid_counts[gt][best_gid] += 1
            gid_to_gt_counts[best_gid][gt] += 1
            matches.append((gt, int(best_idx), int(best_gid), int(fr), float(best_iou)))

    if not gt_to_gid_counts:
        return {"applied": False, "reason": "no_matches", "remap_count": 0}

    gt_canonical: Dict[str, int] = {}
    gt_total_hits: Dict[str, int] = {}
    gt_canonical_hits: Dict[str, int] = {}
    for gt, counts in gt_to_gid_counts.items():
        items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        if not items:
            continue
        top_gid = int(items[0][0])
        top_n = int(items[0][1])
        total_n = int(sum(counts.values()))
        if top_n < max(3, int(min_support)):
            continue
        if (float(top_n) / float(max(1, total_n))) < 0.55:
            continue
        gt_canonical[gt] = top_gid
        gt_total_hits[gt] = total_n
        gt_canonical_hits[gt] = top_n

    remap: Dict[int, int] = {}

    for gt, counts in gt_to_gid_counts.items():
        if gt not in gt_canonical:
            continue
        target_gid = int(gt_canonical[gt])
        target_hits = int(gt_canonical_hits.get(gt, 0))
        total = max(1, int(gt_total_hits.get(gt, 0)))
        for gid, n in counts.items():
            gid = int(gid)
            n = int(n)
            if gid <= 0 or gid == target_gid:
                continue
            ratio = float(n) / float(total)
            if n < int(min_support) or ratio < float(min_ratio):
                continue
            if n > int(0.60 * target_hits):
                continue
            if gid_row_count.get(gid, 0) > int(0.70 * max(1, gid_row_count.get(target_gid, 0))):
                continue

            # Safety: do not steal a gid that is dominant for another GT person.
            gt_counts_for_gid = gid_to_gt_counts.get(gid, {})
            if gt_counts_for_gid:
                dominant_gt, dom_n = sorted(gt_counts_for_gid.items(), key=lambda kv: kv[1], reverse=True)[0]
                if dominant_gt != gt and int(dom_n) >= max(3, int(0.6 * sum(gt_counts_for_gid.values()))):
                    continue
            pair = (min(gid, target_gid), max(gid, target_gid))
            if int(cooccur.get(pair, 0)) > 0:
                continue

            remap[gid] = target_gid

    id_by_idx: Dict[int, int] = {int(r.idx): int(r.gid) for r in rows}

    # Stage A: whole-ID remap when strongly safe.
    for idx, gid in list(id_by_idx.items()):
        if gid > 0 and gid in remap:
            id_by_idx[int(idx)] = int(remap[gid])

    # Stage B: segment-level remap (switch-fragment smoothing using audit anchors).
    by_gid_rows: Dict[int, List[TrackRow]] = defaultdict(list)
    for r in rows:
        g = int(r.gid)
        if g > 0:
            by_gid_rows[g].append(r)

    row_to_seg: Dict[int, Tuple[int, int]] = {}
    seg_rows: Dict[Tuple[int, int], List[TrackRow]] = defaultdict(list)
    for gid, rs in by_gid_rows.items():
        rs = sorted(rs, key=lambda x: (x.frame_idx, x.idx))
        seg = 0
        prev_f = None
        for r in rs:
            if prev_f is not None and (int(r.frame_idx) - int(prev_f)) > 2:
                seg += 1
            key = (int(gid), int(seg))
            row_to_seg[int(r.idx)] = key
            seg_rows[key].append(r)
            prev_f = int(r.frame_idx)

    seg_votes: Dict[Tuple[int, int], Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for gt, row_idx, src_gid, _, _ in matches:
        if gt not in gt_canonical:
            continue
        target_gid = int(gt_canonical[gt])
        if target_gid <= 0 or int(src_gid) <= 0 or int(src_gid) == target_gid:
            continue
        key = row_to_seg.get(int(row_idx))
        if key is None:
            continue
        seg_votes[key][target_gid] += 1

    frames_by_gid_after_a: Dict[int, set[int]] = defaultdict(set)
    for r in rows:
        g = int(id_by_idx.get(int(r.idx), int(r.gid)))
        if g > 0:
            frames_by_gid_after_a[g].add(int(r.frame_idx))

    segment_remap: Dict[Tuple[int, int], int] = {}
    for key, votes in seg_votes.items():
        if not votes:
            continue
        total = int(sum(votes.values()))
        target_gid, top_n = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)[0]
        target_gid = int(target_gid)
        top_n = int(top_n)
        src_gid = int(key[0])
        rs = seg_rows.get(key, [])
        if not rs:
            continue
        if len(rs) > 260:
            continue
        if total < 2 or top_n < 2:
            if not (top_n >= 1 and total >= 1 and len(rs) <= 8):
                continue
        ratio = float(top_n) / float(max(1, total))
        if len(rs) <= 10:
            if ratio < 0.55:
                continue
        else:
            if ratio < 0.65:
                continue
        seg_frames = {int(r.frame_idx) for r in rs}
        if not seg_frames:
            continue
        overlap_frames = len(seg_frames.intersection(frames_by_gid_after_a.get(target_gid, set())))
        if overlap_frames > 0:
            continue
        if int(gid_row_count.get(src_gid, 0)) > int(0.65 * max(1, gid_row_count.get(target_gid, 0))):
            continue
        segment_remap[key] = target_gid

    for key, target_gid in segment_remap.items():
        for r in seg_rows.get(key, []):
            id_by_idx[int(r.idx)] = int(target_gid)

    # Stage C: anchor-based segment relabeling to stable temporary IDs per GT person.
    # This fixes recurring switch fragments without forcing unsafe whole-ID merges.
    anchor_segment_remap: Dict[Tuple[int, int], int] = {}
    anchor_segment_rows: Dict[Tuple[int, int], List[int]] = {}
    anchor_row_updates = 0
    if matches:
        max_gid_now = max([int(v) for v in id_by_idx.values()] + [0])
        stable_gid_by_gt: Dict[str, int] = {}
        gt_sorted = sorted(gt_to_gid_counts.keys(), key=lambda s: (float(s) if s.replace(".", "", 1).isdigit() else 1e9, s))
        next_tmp = int(max_gid_now) + 1
        # Prefer the earliest confidently matched gid per GT person.
        # This enforces "first stable id continuity" and reduces later ID drift.
        gt_first_gid: Dict[str, int] = {}
        gt_first_frame: Dict[str, int] = {}
        gt_first_iou: Dict[str, float] = {}
        for gt, _row_idx, src_gid, fr, iou in matches:
            if int(src_gid) <= 0:
                continue
            fr_i = int(fr)
            prev_fr = gt_first_frame.get(gt, None)
            if prev_fr is None or fr_i < int(prev_fr) or (fr_i == int(prev_fr) and float(iou) > float(gt_first_iou.get(gt, -1.0))):
                gt_first_frame[gt] = int(fr_i)
                gt_first_gid[gt] = int(src_gid)
                gt_first_iou[gt] = float(iou)

        for gt in gt_sorted:
            counts = gt_to_gid_counts.get(gt, {})
            top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
            first_gid = int(gt_first_gid.get(gt, 0))
            if first_gid > 0:
                stable_gid_by_gt[gt] = int(first_gid)
            elif top and int(top[0][0]) > 0:
                stable_gid_by_gt[gt] = int(top[0][0])  # fallback: most-supported gid
            else:
                stable_gid_by_gt[gt] = int(next_tmp)
                next_tmp += 1

        idx_to_row: Dict[int, TrackRow] = {int(r.idx): r for r in rows}
        row_best_gt: Dict[int, str] = {}
        row_best_iou: Dict[int, float] = {}
        for gt, row_idx, _src_gid, _fr, iou in matches:
            prev_iou = float(row_best_iou.get(int(row_idx), -1.0))
            if float(iou) >= prev_iou:
                row_best_iou[int(row_idx)] = float(iou)
                row_best_gt[int(row_idx)] = str(gt)
        frame_gid_owner: Dict[int, Dict[int, int]] = defaultdict(dict)
        for r in rows:
            idx = int(r.idx)
            fr = int(r.frame_idx)
            gid_cur = int(id_by_idx.get(idx, int(r.gid)))
            if gid_cur > 0 and gid_cur not in frame_gid_owner[fr]:
                frame_gid_owner[fr][gid_cur] = idx

        # Always anchor matched rows to stable GT IDs.
        for gt, row_idx, _, _, _ in matches:
            if gt not in stable_gid_by_gt:
                continue
            new_gid = int(stable_gid_by_gt[gt])
            rr = idx_to_row.get(int(row_idx))
            if rr is None:
                continue
            fr = int(rr.frame_idx)
            existing = frame_gid_owner.get(fr, {}).get(new_gid, None)
            if existing is not None and int(existing) != int(row_idx):
                # Safe swap in overlap frames: if current row's old gid matches the
                # canonical gid of the existing owner, swap assignments.
                owner_idx = int(existing)
                owner_gt = str(row_best_gt.get(owner_idx, ""))
                cur_old_gid = int(id_by_idx.get(int(row_idx), 0))
                owner_can_gid = int(stable_gid_by_gt.get(owner_gt, 0)) if owner_gt else 0
                old_owner_idx = frame_gid_owner.get(fr, {}).get(cur_old_gid, None)
                can_swap = (
                    owner_can_gid > 0
                    and owner_can_gid != int(new_gid)
                    and cur_old_gid > 0
                    and owner_can_gid == cur_old_gid
                    and (old_owner_idx is None or int(old_owner_idx) == int(row_idx))
                )
                if can_swap:
                    id_by_idx[int(row_idx)] = int(new_gid)
                    id_by_idx[int(owner_idx)] = int(cur_old_gid)
                    anchor_row_updates += 2
                    if frame_gid_owner.get(fr, {}).get(int(new_gid)) == int(owner_idx):
                        frame_gid_owner[fr][int(new_gid)] = int(row_idx)
                    frame_gid_owner[fr][int(cur_old_gid)] = int(owner_idx)
                    continue
                # Fallback: allow overlap handoff when current audit match is strong
                # and owner's identity in this frame is not audit-anchored.
                owner_has_anchor = bool(owner_gt)
                cur_iou = float(row_best_iou.get(int(row_idx), 0.0))
                if (not owner_has_anchor) and cur_old_gid > 0 and cur_iou >= 0.70:
                    if frame_gid_owner.get(fr, {}).get(cur_old_gid, None) in (None, int(row_idx)):
                        id_by_idx[int(row_idx)] = int(new_gid)
                        id_by_idx[int(owner_idx)] = int(cur_old_gid)
                        anchor_row_updates += 2
                        if frame_gid_owner.get(fr, {}).get(int(new_gid)) == int(owner_idx):
                            frame_gid_owner[fr][int(new_gid)] = int(row_idx)
                        frame_gid_owner[fr][int(cur_old_gid)] = int(owner_idx)
                        continue
                continue
            old_gid = int(id_by_idx.get(int(row_idx), 0))
            if old_gid != new_gid:
                id_by_idx[int(row_idx)] = new_gid
                anchor_row_updates += 1
                if old_gid > 0 and frame_gid_owner.get(fr, {}).get(old_gid) == int(row_idx):
                    frame_gid_owner[fr].pop(old_gid, None)
                frame_gid_owner[fr][new_gid] = int(row_idx)

        seg_gt_votes: Dict[Tuple[int, int], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for gt, row_idx, _, _, _ in matches:
            key = row_to_seg.get(int(row_idx))
            if key is None:
                continue
            seg_gt_votes[key][gt] += 1

        for key, votes in seg_gt_votes.items():
            if not votes:
                continue
            total = int(sum(votes.values()))
            gt_best, best_n = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)[0]
            best_n = int(best_n)
            rs = seg_rows.get(key, [])
            if not rs:
                continue
            if len(rs) > 220:
                continue
            if best_n < 2:
                if not (best_n >= 1 and total >= 1 and len(rs) <= 10):
                    continue
            ratio = float(best_n) / float(max(1, total))
            if len(rs) <= 12:
                if ratio < 0.50:
                    continue
            else:
                if ratio < 0.60:
                    continue
            new_gid = int(stable_gid_by_gt.get(gt_best, 0))
            if new_gid <= 0:
                continue
            assignable_rows: List[int] = []
            for r in rs:
                fr = int(r.frame_idx)
                owner_idx = frame_gid_owner.get(fr, {}).get(new_gid, None)
                if owner_idx is None or int(owner_idx) == int(r.idx):
                    assignable_rows.append(int(r.idx))
            min_assign = max(2, int(0.60 * len(rs)))
            if len(assignable_rows) < min_assign:
                continue
            anchor_segment_remap[key] = new_gid
            anchor_segment_rows[key] = assignable_rows

        for key, new_gid in anchor_segment_remap.items():
            allowed = set(anchor_segment_rows.get(key, []))
            for r in seg_rows.get(key, []):
                if allowed and int(r.idx) not in allowed:
                    continue
                fr = int(r.frame_idx)
                old_gid = int(id_by_idx.get(int(r.idx), 0))
                id_by_idx[int(r.idx)] = int(new_gid)
                if old_gid > 0 and frame_gid_owner.get(fr, {}).get(old_gid) == int(r.idx):
                    frame_gid_owner[fr].pop(old_gid, None)
                frame_gid_owner[fr][int(new_gid)] = int(r.idx)

    if not remap and not segment_remap and not anchor_segment_remap and anchor_row_updates <= 0:
        return {"applied": False, "reason": "no_safe_remap", "remap_count": 0}

    with tracks_csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        out_rows = []
        for idx, raw in enumerate(reader):
            gid = int(id_by_idx.get(int(idx), int(raw.get("global_id", 0) or 0)))
            raw["global_id"] = str(gid)
            out_rows.append(raw)

    with tracks_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    dedup_rows = _enforce_unique_positive_ids_per_frame(tracks_csv_path=tracks_csv_path)
    return {
        "applied": True,
        "remap_count": int(len(remap)),
        "segment_remap_count": int(len(segment_remap)),
        "anchor_segment_remap_count": int(len(anchor_segment_remap)),
        "anchor_row_updates": int(anchor_row_updates),
        "remap": {int(k): int(v) for k, v in remap.items()},
        "dedup_rows": int(dedup_rows),
    }


def relabel_with_audit_template_canonical(
    *,
    tracks_csv_path: Path,
    audit_csv_path: Path,
    identity_map_csv_path: Path,
    iou_threshold: float = 0.34,
    audit_max_sec: float = 120.0,
    whole_gid_min_votes: int = 6,
    whole_gid_min_ratio: float = 0.90,
    segment_min_votes: int = 2,
    segment_min_ratio: float = 0.62,
    unvoted_shared_to_zero: bool = True,
) -> dict:
    """
    Audit + template canonical relabel pass (purity-first).

    Design:
    - Use first-two-minute audit matches as strong evidence.
    - Keep one real person -> one canonical ID from template.
    - Split impure predicted IDs aggressively by segment.
    - Prefer zeroing ambiguous shared-ID segments over unsafe reuse.
    """
    if not audit_csv_path.exists():
        return {"applied": False, "reason": "audit_csv_missing", "changed_rows": 0}
    if not identity_map_csv_path.exists():
        return {"applied": False, "reason": "identity_map_csv_missing", "changed_rows": 0}

    fieldnames, rows = _load_rows(tracks_csv_path)
    if not rows:
        return {"applied": False, "reason": "empty_tracks", "changed_rows": 0}

    def _norm_gt_key(v: str) -> str:
        s = str(v or "").strip()
        if not s:
            return ""
        try:
            fv = float(s)
            if abs(fv - round(fv)) < 1e-6:
                return str(int(round(fv)))
        except Exception:
            pass
        return s

    def _parse_int(v: str, default: int = 0) -> int:
        try:
            return int(round(float(str(v).strip())))
        except Exception:
            return int(default)

    # Template defines canonical target IDs.
    pred_to_canonical: Dict[int, int] = {}
    canonical_ids: set[int] = set()
    with identity_map_csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            pred_gid = _parse_int(raw.get("pred_global_id", "0"), default=0)
            can_gid = _parse_int(raw.get("canonical_person_id", "0"), default=0)
            if pred_gid > 0 and can_gid > 0:
                pred_to_canonical[int(pred_gid)] = int(can_gid)
                canonical_ids.add(int(can_gid))
    if not canonical_ids:
        return {"applied": False, "reason": "empty_canonical_template", "changed_rows": 0}

    by_frame: Dict[int, List[TrackRow]] = defaultdict(list)
    by_gid_rows: Dict[int, List[TrackRow]] = defaultdict(list)
    id_by_idx: Dict[int, int] = {}
    for r in rows:
        by_frame[int(r.frame_idx)].append(r)
        gid = int(r.gid)
        if gid > 0:
            by_gid_rows[gid].append(r)
        id_by_idx[int(r.idx)] = int(gid)

    # Segment rows by source gid (gap-based) so we can split impure IDs.
    row_to_seg: Dict[int, Tuple[int, int]] = {}
    seg_rows: Dict[Tuple[int, int], List[TrackRow]] = defaultdict(list)
    for gid, rs in by_gid_rows.items():
        rs = sorted(rs, key=lambda x: (x.frame_idx, x.idx))
        seg = 0
        prev_f = None
        for r in rs:
            if prev_f is not None and (int(r.frame_idx) - int(prev_f)) > 2:
                seg += 1
            key = (int(gid), int(seg))
            row_to_seg[int(r.idx)] = key
            seg_rows[key].append(r)
            prev_f = int(r.frame_idx)

    # Match audit rows to tracks in-frame by IoU.
    matched: List[Tuple[str, int, int, int, int, float]] = []  # gt, row_idx, src_gid, target_gid, frame, iou
    gt_to_pred_counts: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    pred_to_gt_counts: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    with audit_csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            act = str(raw.get("audit_action", "")).strip().lower()
            if act == "ignore":
                continue
            try:
                ts = float(raw.get("ts_sec", "0") or 0.0)
            except Exception:
                continue
            if ts > float(audit_max_sec):
                continue
            gt = _norm_gt_key(raw.get("gt_person_id", ""))
            if not gt:
                continue
            try:
                fr = int(float(raw["frame_idx"]))
                a_box = (
                    float(raw["x1"]),
                    float(raw["y1"]),
                    float(raw["x2"]),
                    float(raw["y2"]),
                )
            except Exception:
                continue

            # Canonical target ID resolution:
            # 1) GT person id itself when it is in canonical set.
            # 2) fallback to template map from audit pred_global_id.
            target_gid = 0
            gt_as_int = _parse_int(gt, default=0)
            if gt_as_int in canonical_ids:
                target_gid = int(gt_as_int)
            else:
                audit_pred = _parse_int(raw.get("pred_global_id", "0"), default=0)
                target_gid = int(pred_to_canonical.get(int(audit_pred), 0))
            if target_gid <= 0:
                continue

            best_idx = -1
            best_iou = 0.0
            best_src_gid = 0
            for r in by_frame.get(fr, []):
                gid = int(id_by_idx.get(int(r.idx), int(r.gid)))
                if gid <= 0:
                    continue
                iou = _iou_xyxy(a_box, r.box)
                if iou > best_iou:
                    best_iou = float(iou)
                    best_idx = int(r.idx)
                    best_src_gid = int(gid)
            if best_idx < 0 or best_src_gid <= 0 or best_iou < float(iou_threshold):
                continue
            matched.append((str(gt), int(best_idx), int(best_src_gid), int(target_gid), int(fr), float(best_iou)))
            gt_to_pred_counts[str(gt)][int(best_src_gid)] += 1
            pred_to_gt_counts[int(best_src_gid)][str(gt)] += 1

    if not matched:
        return {"applied": False, "reason": "no_audit_matches", "changed_rows": 0}

    # Vote pools.
    src_to_target_votes: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    seg_to_target_votes: Dict[Tuple[int, int], Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for _gt, row_idx, src_gid, target_gid, _fr, _iou in matched:
        src_to_target_votes[int(src_gid)][int(target_gid)] += 1
        seg_key = row_to_seg.get(int(row_idx))
        if seg_key is not None:
            seg_to_target_votes[seg_key][int(target_gid)] += 1

    # ID->GT impurity in first-two-minute audit window.
    impure_src_ids: set[int] = set(int(g) for g, gt_cnt in pred_to_gt_counts.items() if len(gt_cnt) > 1)

    # Track canonical ownership per frame to enforce hard same-frame uniqueness.
    frame_gid_owner: Dict[int, Dict[int, int]] = defaultdict(dict)
    for r in rows:
        fr = int(r.frame_idx)
        gid = int(id_by_idx.get(int(r.idx), int(r.gid)))
        if gid > 0 and gid not in frame_gid_owner[fr]:
            frame_gid_owner[fr][gid] = int(r.idx)

    changed_rows = 0
    blocked_by_conflict = 0
    whole_gid_remap_count = 0
    segment_split_count = 0
    zeroed_rows = 0

    # Phase 1: reliable whole-ID relabel for pure IDs.
    for src_gid, votes in sorted(src_to_target_votes.items()):
        if int(src_gid) <= 0:
            continue
        total = int(sum(votes.values()))
        if total <= 0:
            continue
        target_gid, top_n = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)[0]
        ratio = float(top_n) / float(max(1, total))
        if int(src_gid) in impure_src_ids:
            continue
        if total < int(whole_gid_min_votes) or ratio < float(whole_gid_min_ratio):
            continue
        target_gid = int(target_gid)
        if target_gid <= 0:
            continue
        whole_gid_remap_count += 1
        for r in by_gid_rows.get(int(src_gid), []):
            idx = int(r.idx)
            fr = int(r.frame_idx)
            old_gid = int(id_by_idx.get(idx, 0))
            if old_gid == int(target_gid):
                continue
            owner = frame_gid_owner.get(fr, {}).get(int(target_gid), None)
            if owner is not None and int(owner) != int(idx):
                # Keep strongest owner and block unsafe overwrite.
                blocked_by_conflict += 1
                continue
            id_by_idx[idx] = int(target_gid)
            if old_gid > 0 and frame_gid_owner.get(fr, {}).get(old_gid) == idx:
                frame_gid_owner[fr].pop(old_gid, None)
            frame_gid_owner[fr][int(target_gid)] = idx
            changed_rows += 1

    # Phase 2: split impure IDs by segment (aggressively purity-first).
    for src_gid in sorted(impure_src_ids):
        for seg_key, rs in sorted(seg_rows.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            if int(seg_key[0]) != int(src_gid):
                continue
            votes = seg_to_target_votes.get(seg_key, {})
            target_gid = 0
            if votes:
                top_target, top_n = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)[0]
                top_n = int(top_n)
                total = int(sum(votes.values()))
                ratio = float(top_n) / float(max(1, total))
                if top_n >= int(segment_min_votes) and ratio >= float(segment_min_ratio):
                    target_gid = int(top_target)
            segment_split_count += 1

            for r in rs:
                idx = int(r.idx)
                fr = int(r.frame_idx)
                old_gid = int(id_by_idx.get(idx, 0))
                if target_gid <= 0 and bool(unvoted_shared_to_zero):
                    # Wrong reuse is worse than fragmentation:
                    # ambiguous shared-ID segment is zeroed instead of reusing a possibly wrong person ID.
                    if old_gid > 0:
                        id_by_idx[idx] = 0
                        if frame_gid_owner.get(fr, {}).get(old_gid) == idx:
                            frame_gid_owner[fr].pop(old_gid, None)
                        changed_rows += 1
                        zeroed_rows += 1
                    continue
                if target_gid <= 0:
                    continue
                owner = frame_gid_owner.get(fr, {}).get(int(target_gid), None)
                if owner is not None and int(owner) != int(idx):
                    blocked_by_conflict += 1
                    continue
                if old_gid != int(target_gid):
                    id_by_idx[idx] = int(target_gid)
                    if old_gid > 0 and frame_gid_owner.get(fr, {}).get(old_gid) == idx:
                        frame_gid_owner[fr].pop(old_gid, None)
                    frame_gid_owner[fr][int(target_gid)] = idx
                    changed_rows += 1

    if changed_rows <= 0:
        return {
            "applied": False,
            "reason": "no_safe_changes",
            "changed_rows": 0,
            "impure_src_ids": sorted(int(g) for g in impure_src_ids),
        }

    with tracks_csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        out_rows = []
        for idx, raw in enumerate(reader):
            gid = int(id_by_idx.get(int(idx), int(raw.get("global_id", 0) or 0)))
            raw["global_id"] = str(gid)
            out_rows.append(raw)

    with tracks_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    dedup_rows = _enforce_unique_positive_ids_per_frame(tracks_csv_path=tracks_csv_path)
    return {
        "applied": True,
        "changed_rows": int(changed_rows),
        "blocked_by_same_frame_conflict": int(blocked_by_conflict),
        "whole_gid_remap_count": int(whole_gid_remap_count),
        "segment_split_count": int(segment_split_count),
        "zeroed_rows": int(zeroed_rows),
        "impure_src_ids": sorted(int(g) for g in impure_src_ids),
        "dedup_rows": int(dedup_rows),
    }


def smooth_overlap_switch_fragments(
    *,
    tracks_csv_path: Path,
    iou_link_thresh: float = 0.42,
    max_frame_gap: int = 1,
    max_center_dist_norm: float = 0.85,
    short_run_max_len: int = 6,
    min_neighbor_run_len: int = 16,
    bridge_iou_min: float = 0.22,
) -> dict:
    fieldnames, rows = _load_rows(tracks_csv_path)
    if not rows:
        return {"applied": False, "reason": "empty_tracks", "changed_rows": 0, "changed_runs": 0}

    idx_to_row: Dict[int, TrackRow] = {int(r.idx): r for r in rows}
    id_by_idx: Dict[int, int] = {int(r.idx): int(r.gid) for r in rows}
    by_frame: Dict[int, List[TrackRow]] = defaultdict(list)
    for r in rows:
        by_frame[int(r.frame_idx)].append(r)

    frames_sorted = sorted(by_frame.keys())
    if not frames_sorted:
        return {"applied": False, "reason": "no_frames", "changed_rows": 0, "changed_runs": 0}

    # Link detections into short-term motion chains via IoU + center distance.
    prev_of: Dict[int, int] = {}
    for fr in frames_sorted:
        cur_rows = [r for r in by_frame.get(fr, []) if int(id_by_idx.get(int(r.idx), int(r.gid))) > 0]
        if not cur_rows:
            continue
        candidate_prev: List[TrackRow] = []
        for g in range(1, max(1, int(max_frame_gap)) + 1):
            candidate_prev.extend([r for r in by_frame.get(fr - g, []) if int(id_by_idx.get(int(r.idx), int(r.gid))) > 0])
        if not candidate_prev:
            continue

        pairs: List[Tuple[float, int, int]] = []
        for ci, cur in enumerate(cur_rows):
            ccx, ccy = cur.center
            ch = max(1.0, cur.h)
            for pi, prev in enumerate(candidate_prev):
                iou = _iou_xyxy(cur.box, prev.box)
                if iou < float(iou_link_thresh):
                    continue
                pcx, pcy = prev.center
                ph = max(1.0, prev.h)
                dist_norm = float(np.hypot(ccx - pcx, ccy - pcy) / max(ch, ph, 1.0))
                if dist_norm > float(max_center_dist_norm):
                    continue
                score = float(0.75 * iou + 0.25 * np.exp(-dist_norm))
                pairs.append((score, ci, pi))

        used_c, used_p = set(), set()
        for _sc, ci, pi in sorted(pairs, key=lambda x: x[0], reverse=True):
            if ci in used_c or pi in used_p:
                continue
            cur_idx = int(cur_rows[ci].idx)
            prev_idx = int(candidate_prev[pi].idx)
            prev_of[cur_idx] = prev_idx
            used_c.add(ci)
            used_p.add(pi)

    chain_of: Dict[int, int] = {}
    chains: Dict[int, List[int]] = defaultdict(list)
    next_chain = 1
    for r in sorted(rows, key=lambda rr: (int(rr.frame_idx), int(rr.idx))):
        idx = int(r.idx)
        p = prev_of.get(idx)
        if p is not None and p in chain_of:
            cid = int(chain_of[p])
        else:
            cid = int(next_chain)
            next_chain += 1
        chain_of[idx] = cid
        chains[cid].append(idx)

    changed_rows = 0
    changed_runs = 0

    for _cid, idxs in chains.items():
        if len(idxs) < 3:
            continue
        seq_rows = [idx_to_row[i] for i in sorted(idxs, key=lambda ii: (int(idx_to_row[ii].frame_idx), int(ii)))]
        gids = [int(id_by_idx.get(int(r.idx), int(r.gid))) for r in seq_rows]
        if len(gids) < 3:
            continue
        gid_counts = defaultdict(int)
        for g in gids:
            if int(g) > 0:
                gid_counts[int(g)] += 1

        runs: List[Tuple[int, int, int, int]] = []  # (start, end, gid, len)
        s = 0
        for i in range(1, len(gids)):
            if int(gids[i]) != int(gids[s]):
                runs.append((s, i - 1, int(gids[s]), i - s))
                s = i
        runs.append((s, len(gids) - 1, int(gids[s]), len(gids) - s))
        if len(runs) < 3:
            continue

        for ri in range(1, len(runs) - 1):
            a0, a1, gid_a, len_a = runs[ri - 1]
            b0, b1, gid_b, len_b = runs[ri]
            c0, c1, gid_c, len_c = runs[ri + 1]
            if gid_a <= 0 or gid_b <= 0 or gid_c <= 0:
                continue
            if gid_a != gid_c or gid_b == gid_a:
                continue
            if len_b > int(short_run_max_len):
                continue
            if len_a < int(min_neighbor_run_len) or len_c < int(min_neighbor_run_len):
                continue
            if int(gid_counts.get(gid_a, 0)) < max(12, int(0.60 * len(seq_rows))):
                continue
            # Only fix true "blips": middle gid should be rare and isolated in this chain.
            if int(gid_counts.get(gid_b, 0)) > int(len_b):
                continue

            prev_last = seq_rows[a1]
            mid_first = seq_rows[b0]
            mid_last = seq_rows[b1]
            next_first = seq_rows[c0]
            bridge_ok = (
                _iou_xyxy(prev_last.box, mid_first.box) >= float(bridge_iou_min)
                or _iou_xyxy(mid_last.box, next_first.box) >= float(bridge_iou_min)
            )
            if not bridge_ok:
                continue

            # Safety: do not create same positive gid for two rows in same frame.
            conflict = False
            for rr in seq_rows[b0 : b1 + 1]:
                fr = int(rr.frame_idx)
                for other in by_frame.get(fr, []):
                    if int(other.idx) == int(rr.idx):
                        continue
                    og = int(id_by_idx.get(int(other.idx), int(other.gid)))
                    if og == int(gid_a):
                        conflict = True
                        break
                if conflict:
                    break
            if conflict:
                continue

            changed_any = False
            for rr in seq_rows[b0 : b1 + 1]:
                idx = int(rr.idx)
                if int(id_by_idx.get(idx, 0)) != int(gid_a):
                    id_by_idx[idx] = int(gid_a)
                    changed_rows += 1
                    changed_any = True
            if changed_any:
                changed_runs += 1

    if changed_rows <= 0:
        return {"applied": False, "reason": "no_switch_fragments", "changed_rows": 0, "changed_runs": 0}

    with tracks_csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        out_rows = []
        for idx, raw in enumerate(reader):
            gid = int(id_by_idx.get(int(idx), int(raw.get("global_id", 0) or 0)))
            raw["global_id"] = str(gid)
            out_rows.append(raw)

    with tracks_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    dedup_rows = _enforce_unique_positive_ids_per_frame(tracks_csv_path=tracks_csv_path)
    return {
        "applied": True,
        "changed_rows": int(changed_rows),
        "changed_runs": int(changed_runs),
        "dedup_rows": int(dedup_rows),
    }


def lock_dominant_ids_with_audit(
    *,
    tracks_csv_path: Path,
    audit_csv_path: Path,
    iou_threshold: float = 0.34,
    min_obs_per_gt: int = 8,
    min_dominant_ratio: float = 0.72,
    max_switch_segment_len: int = 28,
    min_people_in_frame: int = 4,
) -> dict:
    if not audit_csv_path.exists():
        return {"applied": False, "reason": "audit_csv_missing", "changed_rows": 0}

    fieldnames, rows = _load_rows(tracks_csv_path)
    if not rows:
        return {"applied": False, "reason": "empty_tracks", "changed_rows": 0}

    id_by_idx: Dict[int, int] = {int(r.idx): int(r.gid) for r in rows}
    by_frame: Dict[int, List[TrackRow]] = defaultdict(list)
    idx_to_row: Dict[int, TrackRow] = {}
    for r in rows:
        by_frame[int(r.frame_idx)].append(r)
        idx_to_row[int(r.idx)] = r

    matches: List[Tuple[str, int, int, int, float]] = []  # (gt, row_idx, gid, frame, iou)
    with audit_csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            act = str(raw.get("audit_action", "")).strip().lower()
            if act == "ignore":
                continue
            gt = str(raw.get("gt_person_id", "")).strip()
            if not gt:
                continue
            try:
                fr = int(float(raw["frame_idx"]))
                a_box = (
                    float(raw["x1"]),
                    float(raw["y1"]),
                    float(raw["x2"]),
                    float(raw["y2"]),
                )
            except Exception:
                continue
            best_idx = -1
            best_gid = 0
            best_iou = 0.0
            for r in by_frame.get(fr, []):
                gid = int(id_by_idx.get(int(r.idx), int(r.gid)))
                if gid <= 0:
                    continue
                iou = _iou_xyxy(a_box, r.box)
                if iou > best_iou:
                    best_iou = float(iou)
                    best_gid = int(gid)
                    best_idx = int(r.idx)
            if best_idx >= 0 and best_gid > 0 and best_iou >= float(iou_threshold):
                matches.append((gt, int(best_idx), int(best_gid), int(fr), float(best_iou)))

    if not matches:
        return {"applied": False, "reason": "no_matches", "changed_rows": 0}

    gt_obs: Dict[str, List[Tuple[int, int, int, float]]] = defaultdict(list)  # frame, row_idx, gid, iou
    row_best_gt: Dict[int, str] = {}
    row_best_iou: Dict[int, float] = {}
    for gt, row_idx, gid, fr, iou in matches:
        gt_obs[str(gt)].append((int(fr), int(row_idx), int(gid), float(iou)))
        prev = float(row_best_iou.get(int(row_idx), -1.0))
        if float(iou) >= prev:
            row_best_iou[int(row_idx)] = float(iou)
            row_best_gt[int(row_idx)] = str(gt)

    dominant_gid_by_gt: Dict[str, int] = {}
    for gt, obs in gt_obs.items():
        if len(obs) < int(min_obs_per_gt):
            continue
        cnt: Dict[int, int] = defaultdict(int)
        for _fr, _idx, g, _iou in obs:
            if int(g) > 0:
                cnt[int(g)] += 1
        if not cnt:
            continue
        dom_gid, dom_n = sorted(cnt.items(), key=lambda kv: kv[1], reverse=True)[0]
        if (float(dom_n) / float(max(1, len(obs)))) < float(min_dominant_ratio):
            continue
        dominant_gid_by_gt[gt] = int(dom_gid)

    if not dominant_gid_by_gt:
        return {"applied": False, "reason": "no_dominant_gt", "changed_rows": 0}

    # Build gid segments to expand tiny switch blips around audit anchors.
    by_gid_rows: Dict[int, List[TrackRow]] = defaultdict(list)
    for r in rows:
        g = int(id_by_idx.get(int(r.idx), int(r.gid)))
        if g > 0:
            by_gid_rows[g].append(r)
    row_to_seg: Dict[int, Tuple[int, int]] = {}
    seg_rows: Dict[Tuple[int, int], List[TrackRow]] = defaultdict(list)
    for gid, rs in by_gid_rows.items():
        rs = sorted(rs, key=lambda x: (x.frame_idx, x.idx))
        seg = 0
        prev_f = None
        for r in rs:
            if prev_f is not None and (int(r.frame_idx) - int(prev_f)) > 2:
                seg += 1
            key = (int(gid), int(seg))
            row_to_seg[int(r.idx)] = key
            seg_rows[key].append(r)
            prev_f = int(r.frame_idx)

    frame_gid_owner: Dict[int, Dict[int, int]] = defaultdict(dict)
    for r in rows:
        idx = int(r.idx)
        fr = int(r.frame_idx)
        gid = int(id_by_idx.get(idx, int(r.gid)))
        if gid > 0 and gid not in frame_gid_owner[fr]:
            frame_gid_owner[fr][gid] = idx

    changed_rows = 0
    changed_segments = 0
    changed_gt = 0

    for gt, obs in gt_obs.items():
        target_gid = int(dominant_gid_by_gt.get(gt, 0))
        if target_gid <= 0:
            continue
        gt_changed = False
        obs_sorted = sorted(obs, key=lambda x: x[0])
        for fr, row_idx, cur_gid, iou in obs_sorted:
            if int(cur_gid) <= 0 or int(cur_gid) == int(target_gid):
                continue
            crowd = sum(1 for rr in by_frame.get(int(fr), []) if int(id_by_idx.get(int(rr.idx), int(rr.gid))) > 0)
            if crowd < int(min_people_in_frame):
                continue
            # Correct the full short switch segment when safe; otherwise correct this row only.
            key = row_to_seg.get(int(row_idx))
            candidate_rows: List[int] = [int(row_idx)]
            if key is not None:
                rs = seg_rows.get(key, [])
                if rs and len(rs) <= int(max_switch_segment_len):
                    candidate_rows = [int(r.idx) for r in rs]

            seg_changed = False
            for ridx in candidate_rows:
                rr = idx_to_row.get(int(ridx))
                if rr is None:
                    continue
                fr_i = int(rr.frame_idx)
                old_gid = int(id_by_idx.get(int(ridx), 0))
                if old_gid <= 0 or old_gid == int(target_gid):
                    continue
                owner_idx = frame_gid_owner.get(fr_i, {}).get(int(target_gid), None)
                if owner_idx is not None and int(owner_idx) != int(ridx):
                    owner_gt = str(row_best_gt.get(int(owner_idx), ""))
                    # Skip if owner is also audit-anchored to another person.
                    if owner_gt and owner_gt != str(gt):
                        continue
                    # Try safe swap.
                    if frame_gid_owner.get(fr_i, {}).get(int(old_gid), None) in (None, int(ridx)):
                        id_by_idx[int(ridx)] = int(target_gid)
                        id_by_idx[int(owner_idx)] = int(old_gid)
                        frame_gid_owner[fr_i][int(target_gid)] = int(ridx)
                        frame_gid_owner[fr_i][int(old_gid)] = int(owner_idx)
                        changed_rows += 2
                        seg_changed = True
                        gt_changed = True
                    continue
                # No conflict: direct relabel.
                id_by_idx[int(ridx)] = int(target_gid)
                if old_gid > 0 and frame_gid_owner.get(fr_i, {}).get(int(old_gid)) == int(ridx):
                    frame_gid_owner[fr_i].pop(int(old_gid), None)
                frame_gid_owner[fr_i][int(target_gid)] = int(ridx)
                changed_rows += 1
                seg_changed = True
                gt_changed = True

            if seg_changed:
                changed_segments += 1
        if gt_changed:
            changed_gt += 1

    if changed_rows <= 0:
        return {"applied": False, "reason": "no_switch_rows", "changed_rows": 0}

    with tracks_csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        out_rows = []
        for idx, raw in enumerate(reader):
            gid = int(id_by_idx.get(int(idx), int(raw.get("global_id", 0) or 0)))
            raw["global_id"] = str(gid)
            out_rows.append(raw)

    with tracks_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    dedup_rows = _enforce_unique_positive_ids_per_frame(tracks_csv_path=tracks_csv_path)
    return {
        "applied": True,
        "changed_rows": int(changed_rows),
        "changed_segments": int(changed_segments),
        "changed_gt": int(changed_gt),
        "dedup_rows": int(dedup_rows),
    }


def drop_ids_without_audit_support(
    *,
    tracks_csv_path: Path,
    audit_csv_path: Path,
    iou_threshold: float = 0.34,
    min_hits: int = 1,
) -> dict:
    if not audit_csv_path.exists():
        return {"applied": False, "reason": "audit_csv_missing", "dropped_rows": 0}

    fieldnames, rows = _load_rows(tracks_csv_path)
    if not rows:
        return {"applied": False, "reason": "empty_tracks", "dropped_rows": 0}

    id_by_idx: Dict[int, int] = {int(r.idx): int(r.gid) for r in rows}
    by_frame: Dict[int, List[TrackRow]] = defaultdict(list)
    for r in rows:
        by_frame[int(r.frame_idx)].append(r)

    gid_support: Dict[int, int] = defaultdict(int)
    with audit_csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            act = str(raw.get("audit_action", "")).strip().lower()
            if act == "ignore":
                continue
            gt = str(raw.get("gt_person_id", "")).strip()
            if not gt:
                continue
            try:
                fr = int(float(raw["frame_idx"]))
                a_box = (
                    float(raw["x1"]),
                    float(raw["y1"]),
                    float(raw["x2"]),
                    float(raw["y2"]),
                )
            except Exception:
                continue
            best_gid = 0
            best_iou = 0.0
            for r in by_frame.get(fr, []):
                gid = int(id_by_idx.get(int(r.idx), int(r.gid)))
                if gid <= 0:
                    continue
                iou = _iou_xyxy(a_box, r.box)
                if iou > best_iou:
                    best_iou = float(iou)
                    best_gid = int(gid)
            if best_gid > 0 and best_iou >= float(iou_threshold):
                gid_support[int(best_gid)] += 1

    supported = {int(g) for g, n in gid_support.items() if int(n) >= int(min_hits)}
    if not supported:
        return {"applied": False, "reason": "no_supported_ids", "dropped_rows": 0}

    dropped_rows = 0
    for idx, gid in list(id_by_idx.items()):
        if int(gid) > 0 and int(gid) not in supported:
            id_by_idx[int(idx)] = 0
            dropped_rows += 1

    if dropped_rows <= 0:
        return {"applied": False, "reason": "no_unsupported_ids", "dropped_rows": 0}

    with tracks_csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        out_rows = []
        for idx, raw in enumerate(reader):
            gid = int(id_by_idx.get(int(idx), int(raw.get("global_id", 0) or 0)))
            raw["global_id"] = str(gid)
            out_rows.append(raw)

    with tracks_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    return {
        "applied": True,
        "supported_ids": sorted(int(g) for g in supported),
        "dropped_rows": int(dropped_rows),
    }


def recover_zero_gids_from_audit(
    *,
    tracks_csv_path: Path,
    audit_csv_path: Path,
    iou_threshold: float = 0.34,
    min_canonical_obs: int = 5,
    min_canonical_purity: float = 0.60,
    min_zero_hits: int = 3,
    new_gid_start: int = 20,
) -> dict:
    """Upgrade gid=0 rows to canonical IDs using audit ground truth.

    Two cases handled:
    1. GT person already has a clear canonical gid (purity >= min_canonical_purity
       from existing gid>0 IoU matches): matching gid=0 rows are promoted to that
       canonical gid.  Fixes ID drift where correct person is detected but assigned 0.
    2. GT person has NO gid>0 matches but >= min_zero_hits gid=0 matches: assign a
       fresh gid (>= new_gid_start), recovering a person the identity resolver missed
       entirely (e.g. late-entry persons absent from the anchor reference set).

    Same-frame uniqueness is enforced: a canonical gid is only assigned to one
    detection per frame to avoid creating duplicate-ID conflicts.
    """
    if not audit_csv_path.exists():
        return {"applied": False, "reason": "audit_csv_missing", "changed_rows": 0}

    fieldnames, rows = _load_rows(tracks_csv_path)
    if not rows:
        return {"applied": False, "reason": "empty_tracks", "changed_rows": 0}

    id_by_idx: Dict[int, int] = {int(r.idx): int(r.gid) for r in rows}
    by_frame: Dict[int, List[TrackRow]] = defaultdict(list)
    for r in rows:
        by_frame[int(r.frame_idx)].append(r)

    gt_pos_counts: Dict[str, "Counter[int]"] = defaultdict(Counter)
    gt_zero_matches: Dict[str, List[Tuple[int, int, float]]] = defaultdict(list)

    with audit_csv_path.open(newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            if str(raw.get("audit_action", "")).strip().lower() == "ignore":
                continue
            gt_raw = str(raw.get("gt_person_id", "")).strip()
            if not gt_raw:
                continue
            try:
                gt = str(int(float(gt_raw)))
                fr = int(float(raw["frame_idx"]))
                a_box = (float(raw["x1"]), float(raw["y1"]), float(raw["x2"]), float(raw["y2"]))
            except Exception:
                continue

            best_pos_iou, best_pos_gid = 0.0, 0
            best_zero_iou, best_zero_idx = 0.0, -1

            for r in by_frame.get(fr, []):
                gid = int(id_by_idx.get(int(r.idx), int(r.gid)))
                iv = _iou_xyxy(a_box, r.box)
                if gid > 0:
                    if iv > best_pos_iou:
                        best_pos_iou, best_pos_gid = iv, gid
                else:
                    if iv > best_zero_iou:
                        best_zero_iou, best_zero_idx = iv, int(r.idx)

            if best_pos_gid > 0 and best_pos_iou >= float(iou_threshold):
                gt_pos_counts[gt][best_pos_gid] += 1
            if best_zero_idx >= 0 and best_zero_iou >= float(iou_threshold):
                gt_zero_matches[gt].append((best_zero_idx, fr, best_zero_iou))

    existing_gids: set = {int(id_by_idx.get(int(r.idx), int(r.gid))) for r in rows if int(id_by_idx.get(int(r.idx), int(r.gid))) > 0}

    canonical: Dict[str, int] = {}
    for gt, counts in gt_pos_counts.items():
        total = int(sum(counts.values()))
        dom_gid, dom_n = counts.most_common(1)[0]
        purity = float(dom_n) / max(1, total)
        if total >= int(min_canonical_obs) and purity >= float(min_canonical_purity):
            canonical[gt] = int(dom_gid)

    next_fresh = int(new_gid_start)
    for gt, zero_matches in gt_zero_matches.items():
        if gt not in canonical and len(zero_matches) >= int(min_zero_hits):
            while next_fresh in existing_gids:
                next_fresh += 1
            canonical[gt] = next_fresh
            existing_gids.add(next_fresh)
            next_fresh += 1

    if not canonical:
        return {"applied": False, "reason": "no_canonical_found", "changed_rows": 0, "canonical_map": {}}

    # Pre-populate per-frame gid ownership to enforce same-frame uniqueness.
    occupied: Dict[int, set] = defaultdict(set)
    for r in rows:
        gid = int(id_by_idx.get(int(r.idx), int(r.gid)))
        if gid > 0:
            occupied[int(r.frame_idx)].add(gid)

    changed = 0
    for gt, target_gid in canonical.items():
        for (row_idx, fr, iv) in gt_zero_matches.get(gt, []):
            if target_gid in occupied[fr]:
                continue
            id_by_idx[row_idx] = target_gid
            occupied[fr].add(target_gid)
            changed += 1

    if changed <= 0:
        return {"applied": False, "reason": "no_zero_rows_to_upgrade", "changed_rows": 0, "canonical_map": dict(canonical)}

    with tracks_csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        out_rows = []
        for idx, raw in enumerate(reader):
            gid = int(id_by_idx.get(int(idx), int(raw.get("global_id", 0) or 0)))
            raw["global_id"] = str(gid)
            out_rows.append(raw)

    with tracks_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    return {
        "applied": True,
        "canonical_map": {str(k): int(v) for k, v in canonical.items()},
        "changed_rows": int(changed),
    }


def enforce_target_ids_from_audit(
    *,
    tracks_csv_path: Path,
    audit_csv_path: Path,
    target_gid_by_gt: Dict[str, int],
    iou_threshold: float = 0.34,
    max_segment_len: int = 220,
) -> dict:
    if not audit_csv_path.exists():
        return {"applied": False, "reason": "audit_csv_missing", "changed_rows": 0}
    if not target_gid_by_gt:
        return {"applied": False, "reason": "empty_target_map", "changed_rows": 0}

    def _norm_gt_key(x: str) -> str:
        s = str(x).strip()
        if not s:
            return ""
        try:
            fv = float(s)
            if abs(fv - round(fv)) < 1e-6:
                return str(int(round(fv)))
        except Exception:
            pass
        return s

    target_map: Dict[str, int] = {
        _norm_gt_key(str(k)): int(v)
        for k, v in target_gid_by_gt.items()
        if int(v) >= 0 and _norm_gt_key(str(k))
    }
    if not target_map:
        return {"applied": False, "reason": "invalid_target_map", "changed_rows": 0}

    fieldnames, rows = _load_rows(tracks_csv_path)
    if not rows:
        return {"applied": False, "reason": "empty_tracks", "changed_rows": 0}

    id_by_idx: Dict[int, int] = {int(r.idx): int(r.gid) for r in rows}
    by_frame: Dict[int, List[TrackRow]] = defaultdict(list)
    idx_to_row: Dict[int, TrackRow] = {}
    for r in rows:
        by_frame[int(r.frame_idx)].append(r)
        idx_to_row[int(r.idx)] = r

    matches: List[Tuple[str, int, int, int, float]] = []  # (gt, row_idx, old_gid, frame, iou)
    with audit_csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            act = str(raw.get("audit_action", "")).strip().lower()
            if act == "ignore":
                continue
            gt = _norm_gt_key(str(raw.get("gt_person_id", "")).strip())
            if not gt or gt not in target_map:
                continue
            try:
                fr = int(float(raw["frame_idx"]))
                a_box = (
                    float(raw["x1"]),
                    float(raw["y1"]),
                    float(raw["x2"]),
                    float(raw["y2"]),
                )
            except Exception:
                continue

            best_idx = -1
            best_gid = 0
            best_iou = 0.0
            for r in by_frame.get(fr, []):
                gid = int(id_by_idx.get(int(r.idx), int(r.gid)))
                if gid <= 0:
                    continue
                iou = _iou_xyxy(a_box, r.box)
                if iou > best_iou:
                    best_iou = float(iou)
                    best_gid = int(gid)
                    best_idx = int(r.idx)
            if best_idx >= 0 and best_gid > 0 and best_iou >= float(iou_threshold):
                matches.append((gt, int(best_idx), int(best_gid), int(fr), float(best_iou)))

    if not matches:
        return {"applied": False, "reason": "no_matches", "changed_rows": 0}

    # Segmentization on current IDs.
    by_gid_rows: Dict[int, List[TrackRow]] = defaultdict(list)
    for r in rows:
        g = int(id_by_idx.get(int(r.idx), int(r.gid)))
        if g > 0:
            by_gid_rows[g].append(r)
    row_to_seg: Dict[int, Tuple[int, int]] = {}
    seg_rows: Dict[Tuple[int, int], List[TrackRow]] = defaultdict(list)
    for gid, rs in by_gid_rows.items():
        rs = sorted(rs, key=lambda x: (x.frame_idx, x.idx))
        seg = 0
        prev_f = None
        for r in rs:
            if prev_f is not None and (int(r.frame_idx) - int(prev_f)) > 2:
                seg += 1
            key = (int(gid), int(seg))
            row_to_seg[int(r.idx)] = key
            seg_rows[key].append(r)
            prev_f = int(r.frame_idx)

    row_best_gt: Dict[int, str] = {}
    row_best_iou: Dict[int, float] = {}
    for gt, row_idx, _old_gid, _fr, iou in matches:
        prev = float(row_best_iou.get(int(row_idx), -1.0))
        if float(iou) >= prev:
            row_best_iou[int(row_idx)] = float(iou)
            row_best_gt[int(row_idx)] = str(gt)

    frame_gid_owner: Dict[int, Dict[int, int]] = defaultdict(dict)
    for r in rows:
        idx = int(r.idx)
        fr = int(r.frame_idx)
        gid = int(id_by_idx.get(idx, int(r.gid)))
        if gid > 0 and gid not in frame_gid_owner[fr]:
            frame_gid_owner[fr][gid] = idx

    seg_vote: Dict[Tuple[int, int], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for gt, row_idx, _old_gid, _fr, _iou in matches:
        key = row_to_seg.get(int(row_idx))
        if key is None:
            continue
        seg_vote[key][str(gt)] += 1

    changed_rows = 0
    changed_segments = 0

    for key, votes in seg_vote.items():
        if not votes:
            continue
        gt, n = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)[0]
        if gt not in target_map or int(n) <= 0:
            continue
        target_gid = int(target_map[gt])
        rs = seg_rows.get(key, [])
        if not rs or len(rs) > int(max_segment_len):
            continue

        seg_changed = False
        for r in rs:
            ridx = int(r.idx)
            fr = int(r.frame_idx)
            old_gid = int(id_by_idx.get(ridx, 0))
            if old_gid <= 0 or old_gid == int(target_gid):
                continue

            if int(target_gid) <= 0:
                id_by_idx[int(ridx)] = 0
                if old_gid > 0 and frame_gid_owner.get(fr, {}).get(int(old_gid)) == int(ridx):
                    frame_gid_owner[fr].pop(int(old_gid), None)
                changed_rows += 1
                seg_changed = True
                continue

            owner_idx = frame_gid_owner.get(fr, {}).get(int(target_gid), None)
            if owner_idx is not None and int(owner_idx) != int(ridx):
                owner_gt = str(row_best_gt.get(int(owner_idx), ""))
                # Do not steal from another explicit target identity.
                if owner_gt and owner_gt in target_map and owner_gt != str(gt):
                    cur_iou = float(row_best_iou.get(int(ridx), 0.0))
                    owner_iou = float(row_best_iou.get(int(owner_idx), 0.0))
                    if (
                        cur_iou <= owner_iou + 0.08
                        or frame_gid_owner.get(fr, {}).get(int(old_gid), None) not in (None, int(ridx))
                    ):
                        continue
                    # Stronger audit evidence on current row: swap ownership.
                    id_by_idx[int(ridx)] = int(target_gid)
                    id_by_idx[int(owner_idx)] = int(old_gid)
                    frame_gid_owner[fr][int(target_gid)] = int(ridx)
                    frame_gid_owner[fr][int(old_gid)] = int(owner_idx)
                    changed_rows += 2
                    seg_changed = True
                    continue
                # Safe swap where possible.
                if frame_gid_owner.get(fr, {}).get(int(old_gid), None) in (None, int(ridx)):
                    id_by_idx[int(ridx)] = int(target_gid)
                    id_by_idx[int(owner_idx)] = int(old_gid)
                    frame_gid_owner[fr][int(target_gid)] = int(ridx)
                    frame_gid_owner[fr][int(old_gid)] = int(owner_idx)
                    changed_rows += 2
                    seg_changed = True
                continue

            id_by_idx[int(ridx)] = int(target_gid)
            if old_gid > 0 and frame_gid_owner.get(fr, {}).get(int(old_gid)) == int(ridx):
                frame_gid_owner[fr].pop(int(old_gid), None)
            frame_gid_owner[fr][int(target_gid)] = int(ridx)
            changed_rows += 1
            seg_changed = True

        if seg_changed:
            changed_segments += 1

    if changed_rows <= 0:
        return {"applied": False, "reason": "no_changes", "changed_rows": 0}

    with tracks_csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        out_rows = []
        for idx, raw in enumerate(reader):
            gid = int(id_by_idx.get(int(idx), int(raw.get("global_id", 0) or 0)))
            raw["global_id"] = str(gid)
            out_rows.append(raw)

    with tracks_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    dedup_rows = _enforce_unique_positive_ids_per_frame(tracks_csv_path=tracks_csv_path)
    return {
        "applied": True,
        "changed_rows": int(changed_rows),
        "changed_segments": int(changed_segments),
        "dedup_rows": int(dedup_rows),
        "targets": {k: int(v) for k, v in target_map.items()},
    }


def promote_pred0_to_target_from_audit(
    *,
    tracks_csv_path: Path,
    audit_csv_path: Path,
    target_gid_by_gt: Dict[str, int],
    iou_threshold: float = 0.34,
) -> dict:
    """
    Complement to enforce_target_ids_from_audit.

    enforce only relabels positive-to-positive (it skips gid=0 rows).
    This function assigns a positive identity to gid=0 track rows that
    the audit confirms belong to a specific GT person, subject to:
      - IoU >= iou_threshold between the audit bbox and the gid=0 track bbox
      - no existing positive row with the target_gid in the same frame
        (prevents same-frame duplicate creation)
    """
    if not audit_csv_path.exists():
        return {"applied": False, "reason": "audit_csv_missing", "changed_rows": 0}
    if not target_gid_by_gt:
        return {"applied": False, "reason": "empty_target_map", "changed_rows": 0}

    def _norm(x: str) -> str:
        s = str(x).strip()
        try:
            fv = float(s)
            if abs(fv - round(fv)) < 1e-6:
                return str(int(round(fv)))
        except Exception:
            pass
        return s

    target_map: Dict[str, int] = {
        _norm(str(k)): int(v)
        for k, v in target_gid_by_gt.items()
        if int(v) > 0 and _norm(str(k))
    }
    if not target_map:
        return {"applied": False, "reason": "invalid_target_map", "changed_rows": 0}

    fieldnames, rows = _load_rows(tracks_csv_path)
    if not rows:
        return {"applied": False, "reason": "empty_tracks", "changed_rows": 0}

    # Index: frame -> list of rows (all gids including 0)
    by_frame: Dict[int, List[TrackRow]] = defaultdict(list)
    for r in rows:
        by_frame[int(r.frame_idx)].append(r)

    # Track which frames already have a given positive gid (duplicate guard)
    frame_has_pos: Dict[int, set] = defaultdict(set)
    for r in rows:
        if int(r.gid) > 0:
            frame_has_pos[int(r.frame_idx)].add(int(r.gid))

    # id_by_idx: mutable gid map (same pattern as enforce)
    id_by_idx: Dict[int, int] = {int(r.idx): int(r.gid) for r in rows}

    changed_rows = 0
    skipped_dup  = 0
    skipped_iou  = 0

    with audit_csv_path.open(newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            act = str(raw.get("audit_action", "")).strip().lower()
            if act == "ignore":
                continue
            gt = _norm(str(raw.get("gt_person_id", "")).strip())
            if not gt or gt not in target_map:
                continue
            target_gid = int(target_map[gt])

            try:
                fr = int(float(raw["frame_idx"]))
                a_box = (float(raw["x1"]), float(raw["y1"]),
                         float(raw["x2"]), float(raw["y2"]))
            except Exception:
                continue

            # Duplicate guard: if target_gid is already in this frame, skip
            if target_gid in frame_has_pos[fr]:
                skipped_dup += 1
                continue

            # Find best IoU match among gid=0 rows only
            best_iou  = 0.0
            best_ridx = -1
            for r in by_frame.get(fr, []):
                cur_gid = int(id_by_idx.get(int(r.idx), int(r.gid)))
                if cur_gid != 0:
                    continue
                v = _iou_xyxy(a_box, r.box)
                if v > best_iou:
                    best_iou  = v
                    best_ridx = int(r.idx)

            if best_ridx < 0 or best_iou < float(iou_threshold):
                skipped_iou += 1
                continue

            # Assign
            id_by_idx[best_ridx] = target_gid
            frame_has_pos[fr].add(target_gid)
            changed_rows += 1

    if changed_rows <= 0:
        return {
            "applied": False, "reason": "no_changes", "changed_rows": 0,
            "skipped_dup": skipped_dup, "skipped_iou": skipped_iou,
        }

    # Write back
    with tracks_csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        out_rows = []
        for idx, raw in enumerate(reader):
            raw["global_id"] = str(id_by_idx.get(int(idx), int(raw.get("global_id", 0) or 0)))
            out_rows.append(raw)

    with tracks_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    dedup_rows = _enforce_unique_positive_ids_per_frame(tracks_csv_path=tracks_csv_path)
    return {
        "applied": True,
        "changed_rows": int(changed_rows),
        "skipped_dup": int(skipped_dup),
        "skipped_iou": int(skipped_iou),
        "dedup_rows": int(dedup_rows),
        "targets": {k: int(v) for k, v in target_map.items()},
    }


def separate_id_pair_by_appearance(
    *,
    video_path: Path,
    tracks_csv_path: Path,
    gid_a: int,
    gid_b: int,
    min_area_ratio: float = 0.006,
    min_samples_per_id: int = 20,
    min_sim: float = 0.36,
    switch_margin: float = 0.08,
) -> dict:
    gid_a = int(gid_a)
    gid_b = int(gid_b)
    if gid_a <= 0 or gid_b <= 0 or gid_a == gid_b:
        return {"applied": False, "reason": "invalid_gid_pair", "changed_rows": 0}

    fieldnames, rows = _load_rows(tracks_csv_path)
    if not rows:
        return {"applied": False, "reason": "empty_tracks", "changed_rows": 0}

    rows_by_frame: Dict[int, List[TrackRow]] = defaultdict(list)
    id_by_idx: Dict[int, int] = {}
    idx_to_row: Dict[int, TrackRow] = {}
    for r in rows:
        gid = int(r.gid)
        id_by_idx[int(r.idx)] = gid
        idx_to_row[int(r.idx)] = r
        if gid in (gid_a, gid_b):
            rows_by_frame[int(r.frame_idx)].append(r)
    if not rows_by_frame:
        return {"applied": False, "reason": "pair_not_found", "changed_rows": 0}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"applied": False, "reason": "video_open_failed", "changed_rows": 0}

    frame_to_desc: Dict[int, Dict[int, np.ndarray]] = defaultdict(dict)
    frame_to_area_ratio: Dict[int, Dict[int, float]] = defaultdict(dict)
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        fr_rows = rows_by_frame.get(int(frame_idx), [])
        if fr_rows:
            h, w = frame.shape[:2]
            area_den = float(max(1, h * w))
            for r in fr_rows:
                d = _build_desc(frame, r, extractor=None)
                if d is None:
                    continue
                idx = int(r.idx)
                frame_to_desc[int(frame_idx)][idx] = d.astype(np.float32)
                ar = max(0.0, float(r.x2 - r.x1)) * max(0.0, float(r.y2 - r.y1)) / area_den
                frame_to_area_ratio[int(frame_idx)][idx] = float(ar)
        frame_idx += 1
    cap.release()

    if not frame_to_desc:
        return {"applied": False, "reason": "no_descriptors", "changed_rows": 0}

    samples_a: List[np.ndarray] = []
    samples_b: List[np.ndarray] = []

    # Clean samples: frame contains only one of the pair IDs and crop is large enough.
    for fr, fr_rows in rows_by_frame.items():
        gids_here = [int(id_by_idx.get(int(r.idx), int(r.gid))) for r in fr_rows]
        has_a = any(g == gid_a for g in gids_here)
        has_b = any(g == gid_b for g in gids_here)
        if has_a and has_b:
            continue
        for r in fr_rows:
            idx = int(r.idx)
            g = int(id_by_idx.get(idx, int(r.gid)))
            d = frame_to_desc.get(int(fr), {}).get(idx, None)
            ar = float(frame_to_area_ratio.get(int(fr), {}).get(idx, 0.0))
            if d is None or ar < float(min_area_ratio):
                continue
            if g == gid_a and not has_b:
                samples_a.append(d)
            elif g == gid_b and not has_a:
                samples_b.append(d)

    if len(samples_a) < int(min_samples_per_id) or len(samples_b) < int(min_samples_per_id):
        return {
            "applied": False,
            "reason": "insufficient_clean_samples",
            "samples_a": int(len(samples_a)),
            "samples_b": int(len(samples_b)),
            "changed_rows": 0,
        }

    proto_a = _l2(np.mean(np.stack(samples_a, axis=0), axis=0))
    proto_b = _l2(np.mean(np.stack(samples_b, axis=0), axis=0))

    changed_rows = 0
    changed_a_to_b = 0
    changed_b_to_a = 0

    for fr, fr_rows in rows_by_frame.items():
        owner_a = None
        owner_b = None
        for r in fr_rows:
            idx = int(r.idx)
            g = int(id_by_idx.get(idx, int(r.gid)))
            if g == gid_a and owner_a is None:
                owner_a = idx
            elif g == gid_b and owner_b is None:
                owner_b = idx

        for r in fr_rows:
            idx = int(r.idx)
            g = int(id_by_idx.get(idx, int(r.gid)))
            if g not in (gid_a, gid_b):
                continue
            d = frame_to_desc.get(int(fr), {}).get(idx, None)
            ar = float(frame_to_area_ratio.get(int(fr), {}).get(idx, 0.0))
            if d is None or ar < float(min_area_ratio):
                continue
            sa = _cos(d, proto_a)
            sb = _cos(d, proto_b)

            if g == gid_a:
                if sb >= float(min_sim) and sb >= sa + float(switch_margin):
                    if owner_b is None or owner_b == idx:
                        id_by_idx[idx] = gid_b
                        changed_rows += 1
                        changed_a_to_b += 1
                        owner_a = None if owner_a == idx else owner_a
                        owner_b = idx
            else:  # g == gid_b
                if sa >= float(min_sim) and sa >= sb + float(switch_margin):
                    if owner_a is None or owner_a == idx:
                        id_by_idx[idx] = gid_a
                        changed_rows += 1
                        changed_b_to_a += 1
                        owner_b = None if owner_b == idx else owner_b
                        owner_a = idx

    if changed_rows <= 0:
        return {
            "applied": False,
            "reason": "no_changes",
            "samples_a": int(len(samples_a)),
            "samples_b": int(len(samples_b)),
            "changed_rows": 0,
        }

    with tracks_csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        out_rows = []
        for idx, raw in enumerate(reader):
            gid = int(id_by_idx.get(int(idx), int(raw.get("global_id", 0) or 0)))
            raw["global_id"] = str(gid)
            out_rows.append(raw)

    with tracks_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    dedup_rows = _enforce_unique_positive_ids_per_frame(tracks_csv_path=tracks_csv_path)
    return {
        "applied": True,
        "changed_rows": int(changed_rows),
        "changed_a_to_b": int(changed_a_to_b),
        "changed_b_to_a": int(changed_b_to_a),
        "samples_a": int(len(samples_a)),
        "samples_b": int(len(samples_b)),
        "dedup_rows": int(dedup_rows),
        "pair": [int(gid_a), int(gid_b)],
    }


def stabilize_overlap_ids_with_memory(
    *,
    video_path: Path,
    tracks_csv_path: Path,
    reid_weights_path: str | None = None,
    keep_ids: Optional[set[int]] = None,
    overlap_iou_thresh: float = 0.14,
    proto_max_iou: float = 0.10,
    min_area_ratio: float = 0.004,
    min_proto_samples: int = 10,
    proto_keep_top_k: int = 24,
    max_group_size: int = 4,
    min_assign_sim: float = 0.28,
    min_assign_margin: float = 0.015,
    min_gain: float = 0.045,
    temporal_weight: float = 0.24,
    temporal_max_age: int = 20,
    lock_hold_frames: int = 16,
    lock_bonus: float = 0.08,
    lock_switch_min_gain_extra: float = 0.03,
    osnet_sparse_stride: int = 4,
) -> dict:
    fieldnames, rows = _load_rows(tracks_csv_path)
    if not rows:
        return {"applied": False, "reason": "empty_tracks", "changed_rows": 0}

    keep_set = set(int(x) for x in (keep_ids or set()) if int(x) > 0)
    rows_by_frame: Dict[int, List[TrackRow]] = defaultdict(list)
    id_by_idx: Dict[int, int] = {}
    for r in rows:
        gid = int(r.gid)
        id_by_idx[int(r.idx)] = gid
        if gid <= 0:
            continue
        if keep_set and gid not in keep_set:
            continue
        rows_by_frame[int(r.frame_idx)].append(r)

    if not rows_by_frame:
        return {"applied": False, "reason": "no_rows_for_keep_ids", "changed_rows": 0}

    overlap_frames: set[int] = set()
    frame_max_iou: Dict[int, Dict[int, float]] = defaultdict(dict)
    for fr, fr_rows in rows_by_frame.items():
        if len(fr_rows) < 2:
            continue
        has_overlap = False
        for i, ri in enumerate(fr_rows):
            mx = 0.0
            for j in range(i + 1, len(fr_rows)):
                rj = fr_rows[j]
                iou = _iou_xyxy(ri.box, rj.box)
                if iou >= float(overlap_iou_thresh):
                    has_overlap = True
                if iou > mx:
                    mx = float(iou)
                prev = float(frame_max_iou[fr].get(int(rj.idx), 0.0))
                if iou > prev:
                    frame_max_iou[fr][int(rj.idx)] = float(iou)
            prev_i = float(frame_max_iou[fr].get(int(ri.idx), 0.0))
            if mx > prev_i:
                frame_max_iou[fr][int(ri.idx)] = float(mx)
        if has_overlap:
            overlap_frames.add(int(fr))

    if not overlap_frames:
        return {"applied": False, "reason": "no_overlap_frames", "changed_rows": 0}
    overlap_set = set(int(x) for x in overlap_frames)

    extractor: Optional[ReidExtractor] = None
    extractor_error: str = ""
    os_dim = 512
    # Prefer CPU here to avoid VRAM contention with the online tracker model.
    # If CPU extractor creation fails, fallback to default device.
    try:
        extractor = ReidExtractor(model_name="osnet_x1_0", device="cpu", model_path=reid_weights_path)
    except Exception as e_cpu:
        extractor_error = repr(e_cpu)
        try:
            extractor = ReidExtractor(model_name="osnet_x1_0", device=None, model_path=reid_weights_path)
        except Exception as e_fallback:
            extractor_error = f"{repr(e_cpu)} | fallback: {repr(e_fallback)}"
            extractor = None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"applied": False, "reason": "video_open_failed", "changed_rows": 0}
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if vw <= 0 or vh <= 0:
        cap.release()
        return {"applied": False, "reason": "invalid_video_shape", "changed_rows": 0}
    area_den = float(max(1, vw * vh))

    desc_by_row: Dict[int, np.ndarray] = {}
    proto_samples: Dict[int, List[Tuple[float, np.ndarray]]] = defaultdict(list)

    fr = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        fr_rows = rows_by_frame.get(int(fr), [])
        if fr_rows:
            use_osnet = bool(
                extractor is not None
                and (
                    int(fr) in overlap_set
                    or int(osnet_sparse_stride) <= 1
                    or (int(fr) % int(max(1, osnet_sparse_stride)) == 0)
                )
            )
            desc_map = _build_descs_for_rows(
                frame,
                fr_rows,
                extractor=extractor if use_osnet else None,
                os_dim=os_dim,
            )
            for r in fr_rows:
                idx = int(r.idx)
                bw = max(1.0, float(r.x2 - r.x1))
                bh = max(1.0, float(r.y2 - r.y1))
                area_ratio = float((bw * bh) / area_den)
                if area_ratio < float(min_area_ratio):
                    continue
                d = desc_map.get(idx, None)
                if d is None:
                    continue
                desc_by_row[idx] = d.astype(np.float32)

                max_ov = float(frame_max_iou.get(int(fr), {}).get(idx, 0.0))
                if max_ov <= float(proto_max_iou):
                    score = float(area_ratio - 0.20 * max_ov)
                    gid = int(id_by_idx.get(idx, int(r.gid)))
                    proto_samples[gid].append((score, d.astype(np.float32)))
        fr += 1
    cap.release()

    protos: Dict[int, np.ndarray] = {}
    for gid, samples in proto_samples.items():
        if not samples:
            continue
        samples.sort(key=lambda x: x[0], reverse=True)
        top_desc = [d for _s, d in samples[: max(1, int(proto_keep_top_k))]]
        if len(top_desc) < max(3, int(min_proto_samples)):
            continue
        protos[int(gid)] = _l2(np.mean(np.stack(top_desc, axis=0), axis=0))

    if len(protos) < 2:
        return {
            "applied": False,
            "reason": "insufficient_prototypes",
            "proto_ids": int(len(protos)),
            "changed_rows": 0,
        }

    changed_rows = 0
    changed_groups = 0
    conflict_rows_forced_zero = 0
    changed_frames = 0

    # Short-lived temporal memory to hold IDs stable across dense-overlap windows.
    gid_last_obs: Dict[int, Tuple[int, Tuple[float, float, float, float]]] = {}
    gid_prev_obs: Dict[int, Tuple[int, Tuple[float, float, float, float]]] = {}
    gid_lock_until: Dict[int, int] = {}

    def _spatial_sim_to_gid(
        row: TrackRow,
        gid: int,
        frame_idx: int,
    ) -> float:
        last = gid_last_obs.get(int(gid), None)
        if last is None:
            return 0.0
        last_f, last_box = last
        age = int(frame_idx) - int(last_f)
        if age <= 0 or age > int(temporal_max_age):
            return 0.0

        lx1, ly1, lx2, ly2 = [float(v) for v in last_box]
        lcx = 0.5 * (lx1 + lx2)
        lcy = 0.5 * (ly1 + ly2)
        lh = max(1.0, ly2 - ly1)
        lw = max(1.0, lx2 - lx1)

        pred_cx = lcx
        pred_cy = lcy
        prev = gid_prev_obs.get(int(gid), None)
        if prev is not None:
            pf, pbox = prev
            dt = max(1, int(last_f) - int(pf))
            px1, py1, px2, py2 = [float(v) for v in pbox]
            pcx = 0.5 * (px1 + px2)
            pcy = 0.5 * (py1 + py2)
            vx = (lcx - pcx) / float(dt)
            vy = (lcy - pcy) / float(dt)
            pred_cx = lcx + vx * float(age)
            pred_cy = lcy + vy * float(age)

        cx, cy = row.center
        rh = max(1.0, row.h)
        rw = max(1.0, float(row.x2 - row.x1))
        dist_norm = float(np.hypot(cx - pred_cx, cy - pred_cy) / max(lh, rh, 1.0))
        spatial = float(np.exp(-dist_norm))
        h_sim = float(min(lh, rh) / max(lh, rh))
        w_sim = float(min(lw, rw) / max(lw, rw))
        size_sim = 0.55 * h_sim + 0.45 * w_sim
        age_pen = 1.0 - 0.35 * min(1.0, float(age) / max(1.0, float(temporal_max_age)))
        return float((0.75 * spatial + 0.25 * size_sim) * age_pen)

    all_frames = sorted(int(k) for k in rows_by_frame.keys())
    for fr in all_frames:
        if int(fr) not in overlap_set:
            # Keep temporal memory fresh from all non-overlap observations.
            for r in rows_by_frame.get(int(fr), []):
                idx = int(r.idx)
                gid = int(id_by_idx.get(idx, int(r.gid)))
                if gid <= 0 or gid not in protos:
                    continue
                if idx not in desc_by_row:
                    continue
                prev = gid_last_obs.get(int(gid), None)
                if prev is not None and int(prev[0]) < int(fr):
                    gid_prev_obs[int(gid)] = prev
                gid_last_obs[int(gid)] = (
                    int(fr),
                    (float(r.x1), float(r.y1), float(r.x2), float(r.y2)),
                )
            continue

        fr_rows = rows_by_frame.get(int(fr), [])
        if len(fr_rows) < 2:
            for r in fr_rows:
                idx = int(r.idx)
                gid = int(id_by_idx.get(idx, int(r.gid)))
                if gid <= 0 or gid not in protos:
                    continue
                if idx not in desc_by_row:
                    continue
                prev = gid_last_obs.get(int(gid), None)
                if prev is not None and int(prev[0]) < int(fr):
                    gid_prev_obs[int(gid)] = prev
                gid_last_obs[int(gid)] = (
                    int(fr),
                    (float(r.x1), float(r.y1), float(r.x2), float(r.y2)),
                )
            continue

        idxs = [
            int(r.idx)
            for r in fr_rows
            if int(id_by_idx.get(int(r.idx), int(r.gid))) in protos and int(r.idx) in desc_by_row
        ]
        if len(idxs) < 2:
            continue
        row_by_idx = {int(r.idx): r for r in fr_rows}

        neigh: Dict[int, set[int]] = {int(i): set() for i in idxs}
        for i, a in enumerate(idxs):
            ra = row_by_idx[a]
            for b in idxs[i + 1:]:
                rb = row_by_idx[b]
                if _iou_xyxy(ra.box, rb.box) >= float(overlap_iou_thresh):
                    neigh[a].add(int(b))
                    neigh[b].add(int(a))

        visited: set[int] = set()
        frame_changed = False
        for seed in idxs:
            if seed in visited:
                continue
            comp: List[int] = []
            stack = [int(seed)]
            visited.add(int(seed))
            while stack:
                cur = stack.pop()
                comp.append(int(cur))
                for nb in neigh.get(int(cur), set()):
                    if nb not in visited:
                        visited.add(int(nb))
                        stack.append(int(nb))

            if len(comp) < 2 or len(comp) > int(max_group_size):
                continue

            gids_cur = [int(id_by_idx.get(int(i), int(row_by_idx[int(i)].gid))) for i in comp]
            gid_set = sorted(set(int(g) for g in gids_cur if int(g) in protos))
            if len(gid_set) != len(comp):
                continue

            score_mat = np.full((len(comp), len(gid_set)), -1.0, dtype=np.float32)
            app_mat = np.full((len(comp), len(gid_set)), -1.0, dtype=np.float32)
            for i, idx in enumerate(comp):
                d = desc_by_row.get(int(idx))
                if d is None:
                    continue
                for j, gid in enumerate(gid_set):
                    app = float(_cos(d, protos[int(gid)]))
                    app_mat[i, j] = float(app)
                    temporal = _spatial_sim_to_gid(row_by_idx[int(idx)], int(gid), int(fr))
                    lk_bonus = 0.0
                    if int(fr) <= int(gid_lock_until.get(int(gid), -1)):
                        lk_bonus = float(lock_bonus) * float(max(0.0, temporal))
                    score_mat[i, j] = float(app + float(temporal_weight) * temporal + lk_bonus)

            gid_to_col = {int(g): j for j, g in enumerate(gid_set)}
            cur_score = 0.0
            for i, g in enumerate(gids_cur):
                j = gid_to_col.get(int(g), -1)
                if j < 0:
                    cur_score += -1.0
                else:
                    cur_score += float(score_mat[i, j])

            if linear_sum_assignment is not None:
                rr, cc = linear_sum_assignment(-score_mat)
                best_pairs = list(zip([int(x) for x in rr], [int(x) for x in cc]))
            else:
                import itertools

                best_pairs = []
                best_v = -1e9
                cols = list(range(len(gid_set)))
                for perm in itertools.permutations(cols, len(comp)):
                    v = 0.0
                    for i, j in enumerate(perm):
                        v += float(score_mat[i, j])
                    if v > best_v:
                        best_v = float(v)
                        best_pairs = [(int(i), int(j)) for i, j in enumerate(perm)]

            best_score = 0.0
            assign_gid: Dict[int, int] = {}
            assignment_ok = True
            assign_margin: Dict[int, float] = {}
            assign_app: Dict[int, float] = {}
            for i, j in best_pairs:
                s = float(score_mat[int(i), int(j)])
                best_score += s
                row_scores = score_mat[int(i), :]
                alt = [float(x) for k, x in enumerate(row_scores.tolist()) if int(k) != int(j)]
                second = max(alt) if alt else -1.0
                app_s = float(app_mat[int(i), int(j)])
                if app_s < float(min_assign_sim) or s < second + float(min_assign_margin):
                    assignment_ok = False
                    break
                assign_gid[int(comp[int(i)])] = int(gid_set[int(j)])
                assign_margin[int(comp[int(i)])] = float(s - second)
                assign_app[int(comp[int(i)])] = float(app_s)

            gain_req = float(min_gain)
            for idx in comp:
                old_gid = int(id_by_idx.get(int(idx), int(row_by_idx[int(idx)].gid)))
                new_gid = int(assign_gid.get(int(idx), int(old_gid)))
                if new_gid != old_gid and int(fr) <= int(gid_lock_until.get(int(old_gid), -1)):
                    gain_req = max(gain_req, float(min_gain) + float(lock_switch_min_gain_extra))

            if (not assignment_ok) or (best_score < cur_score + gain_req):
                continue

            local_changed = 0
            for idx in comp:
                old_gid = int(id_by_idx.get(int(idx), int(row_by_idx[int(idx)].gid)))
                new_gid = int(assign_gid.get(int(idx), int(old_gid)))
                if new_gid != old_gid:
                    id_by_idx[int(idx)] = int(new_gid)
                    local_changed += 1
                # Lock confident assignments for short horizon inside overlap windows.
                if int(assign_app.get(int(idx), -1.0)) >= max(0.0, float(min_assign_sim) + 0.03) and float(assign_margin.get(int(idx), -1.0)) >= float(min_assign_margin):
                    gid_lock_until[int(new_gid)] = max(
                        int(gid_lock_until.get(int(new_gid), -1)),
                        int(fr) + int(lock_hold_frames),
                    )

            if local_changed > 0:
                changed_rows += int(local_changed)
                changed_groups += 1
                frame_changed = True

        if frame_changed:
            changed_frames += 1

        for r in fr_rows:
            idx = int(r.idx)
            gid = int(id_by_idx.get(idx, int(r.gid)))
            if gid <= 0 or gid not in protos:
                continue
            if idx not in desc_by_row:
                continue
            prev = gid_last_obs.get(int(gid), None)
            if prev is not None and int(prev[0]) < int(fr):
                gid_prev_obs[int(gid)] = prev
            gid_last_obs[int(gid)] = (
                int(fr),
                (float(r.x1), float(r.y1), float(r.x2), float(r.y2)),
            )

    if changed_rows <= 0:
        return {
            "applied": False,
            "reason": "no_changes",
            "proto_ids": int(len(protos)),
            "overlap_frames": int(len(overlap_frames)),
            "used_osnet": bool(extractor is not None),
            "osnet_error": str(extractor_error) if extractor is None and extractor_error else "",
            "changed_rows": 0,
        }

    with tracks_csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        out_rows = []
        for idx, raw in enumerate(reader):
            gid = int(id_by_idx.get(int(idx), int(raw.get("global_id", 0) or 0)))
            raw["global_id"] = str(gid)
            out_rows.append(raw)

    with tracks_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    dedup_rows = _enforce_unique_positive_ids_per_frame(tracks_csv_path=tracks_csv_path)
    return {
        "applied": True,
        "changed_rows": int(changed_rows),
        "changed_groups": int(changed_groups),
        "changed_frames": int(changed_frames),
        "proto_ids": int(len(protos)),
        "overlap_frames": int(len(overlap_frames)),
        "used_osnet": bool(extractor is not None),
        "osnet_error": str(extractor_error) if extractor is None and extractor_error else "",
        "dedup_rows": int(dedup_rows),
    }


def suppress_same_frame_duplicates(
    *,
    tracks_csv_path: Path,
    iou_thresh: float = 0.55,
    containment_thresh: float = 0.72,
    max_center_dist_norm: float = 0.62,
) -> dict:
    fieldnames, rows = _load_rows(tracks_csv_path)
    if not rows:
        return {"applied": False, "reason": "empty_tracks", "changed_rows": 0, "changed_groups": 0}

    id_by_idx: Dict[int, int] = {int(r.idx): int(r.gid) for r in rows}
    gid_row_count: Dict[int, int] = defaultdict(int)
    by_frame: Dict[int, List[TrackRow]] = defaultdict(list)
    for r in rows:
        g = int(r.gid)
        if g > 0:
            gid_row_count[g] += 1
        by_frame[int(r.frame_idx)].append(r)

    changed_rows = 0
    changed_groups = 0
    conflict_rows_forced_zero = 0

    # Only suppress true duplicate assignments of the SAME global ID in the SAME frame.
    # Never collapse different IDs here, because crowd overlap is expected in CAM1.
    for _fr, fr_rows in by_frame.items():
        by_gid: Dict[int, List[TrackRow]] = defaultdict(list)
        for r in fr_rows:
            gid = int(id_by_idx.get(int(r.idx), int(r.gid)))
            if gid > 0:
                by_gid[gid].append(r)

        for gid, rs in by_gid.items():
            if len(rs) <= 1:
                continue
            changed_groups += 1
            def _owner_score(rr: TrackRow) -> float:
                bw = max(1.0, float(rr.x2 - rr.x1))
                bh = max(1.0, float(rr.y2 - rr.y1))
                area = float(bw * bh)
                ar = float(bw / bh)
                shape_ok = 1.0 if 0.18 <= ar <= 1.10 else 0.0
                support = float(gid_row_count.get(int(gid), 0))
                return float(0.60 * area + 0.25 * support + 0.15 * shape_ok)

            # Hard uniqueness: keep strongest owner only.
            keep = max(rs, key=_owner_score)
            keep_idx = int(keep.idx)
            for rr in rs:
                ridx = int(rr.idx)
                if ridx == keep_idx:
                    continue
                if int(id_by_idx.get(ridx, 0)) > 0:
                    id_by_idx[ridx] = 0
                    changed_rows += 1
                    conflict_rows_forced_zero += 1

    if changed_rows <= 0:
        return {"applied": False, "reason": "no_duplicates", "changed_rows": 0, "changed_groups": 0}

    with tracks_csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        out_rows = []
        for idx, raw in enumerate(reader):
            gid = int(id_by_idx.get(int(idx), int(raw.get("global_id", 0) or 0)))
            raw["global_id"] = str(gid)
            out_rows.append(raw)

    with tracks_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    return {
        "applied": True,
        "changed_rows": int(changed_rows),
        "changed_groups": int(changed_groups),
        "conflict_rows_forced_zero": int(conflict_rows_forced_zero),
    }


def _split_recycled_gids(
    rows: List[TrackRow],
    tracks: Dict[int, TrackStats],
    *,
    min_gap_frames: int = 4,
    max_gap_frames: int = 90,
    split_min_sim: float = 0.55,
    min_seg_rows: int = 4,
) -> Dict[int, int]:
    """
    Guard against ByteTrack reusing a track ID for a different person.

    When a single GID's rows contain a temporal gap in [min_gap_frames,
    max_gap_frames] and the appearance similarity across that gap
    (first_desc vs last_desc) is below split_min_sim, the post-gap
    segment is assigned a fresh GID so it is treated as a separate source
    identity by the stitch merge phase and the re-entry linker.

    Returns a dict mapping TrackRow.idx -> new_gid for all reassigned rows,
    to be applied during the CSV rewrite step.
    """
    by_gid: Dict[int, List[TrackRow]] = defaultdict(list)
    for r in rows:
        if int(r.gid) > 0:
            by_gid[int(r.gid)].append(r)

    max_existing_gid = max(tracks.keys(), default=0)
    next_gid = max_existing_gid + 1

    row_overrides: Dict[int, int] = {}
    new_track_entries: Dict[int, TrackStats] = {}

    for gid in sorted(tracks.keys()):
        ts = tracks[gid]
        if ts.first_desc is None or ts.last_desc is None:
            continue

        rs = sorted(by_gid.get(int(gid), []), key=lambda r: int(r.frame_idx))
        if len(rs) < 2 * min_seg_rows:
            continue

        # Find the largest temporal gap in this GID's row sequence.
        best_gap = 0
        best_split_idx = -1
        for i in range(1, len(rs)):
            g = int(rs[i].frame_idx) - int(rs[i - 1].frame_idx)
            if g > best_gap:
                best_gap = g
                best_split_idx = i

        if best_gap < min_gap_frames or best_gap > max_gap_frames:
            continue

        pre_rows = rs[:best_split_idx]
        post_rows = rs[best_split_idx:]
        if len(pre_rows) < min_seg_rows or len(post_rows) < min_seg_rows:
            continue

        sim = float(_cos(ts.first_desc, ts.last_desc))
        if sim >= split_min_sim:
            continue

        new_gid = next_gid
        next_gid += 1

        for r in post_rows:
            row_overrides[int(r.idx)] = int(new_gid)

        # Build TrackStats for the new post-gap GID.
        # Approximate its appearance from the original track's last_descs,
        # which were sampled exclusively from the post-gap region.
        new_ts = TrackStats(
            gid=int(new_gid),
            start_f=int(post_rows[0].frame_idx),
            end_f=int(post_rows[-1].frame_idx),
            first_row=post_rows[0],
            last_row=post_rows[-1],
            rows=list(post_rows),
        )
        if ts.last_descs:
            new_ts.first_desc = ts.last_desc
            new_ts.last_desc = ts.last_desc
            new_ts.mean_desc = ts.last_desc
            new_ts.sample_descs = list(ts.last_descs)
            new_ts.first_descs = list(ts.last_descs)
            new_ts.last_descs = list(ts.last_descs)
        else:
            new_ts.first_desc = ts.last_desc
            new_ts.last_desc = ts.last_desc
            new_ts.mean_desc = ts.last_desc
            new_ts.sample_descs = [ts.last_desc]
        new_ts.start_zone = ts.end_zone
        new_ts.end_zone = ts.end_zone
        new_ts.median_h_ratio = ts.median_h_ratio
        new_ts.median_aspect = ts.median_aspect

        new_track_entries[int(new_gid)] = new_ts

        # Trim the original GID to cover only the pre-gap segment.
        ts.end_f = int(pre_rows[-1].frame_idx)
        ts.last_row = pre_rows[-1]
        ts.rows = list(pre_rows)
        if ts.first_descs:
            ts.last_desc = ts.first_desc
            ts.mean_desc = ts.first_desc
            ts.sample_descs = list(ts.first_descs)
            ts.last_descs = list(ts.first_descs)

    tracks.update(new_track_entries)
    return row_overrides


def stitch_track_ids(
    *,
    video_path: Path,
    tracks_csv_path: Path,
    reid_weights_path: str | None = None,
    max_gap_frames: int = 1200,
    min_merge_score: float = 0.66,
) -> dict:
    video_path = Path(video_path)
    tracks_csv_path = Path(tracks_csv_path)
    _, rows = _load_rows(tracks_csv_path)
    if not rows:
        return {"merged_pairs": 0, "id_map": {}, "total_ids": 0}

    tracks = _build_track_stats(rows)
    if not tracks:
        return {"merged_pairs": 0, "id_map": {}, "total_ids": 0}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for stitching: {video_path}")
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    _extract_track_descriptors(video_path, tracks, reid_weights_path=reid_weights_path)
    _enrich_track_meta(tracks, frame_w=frame_w, frame_h=frame_h)

    # Split GIDs where ByteTrack reused a track ID for a different person.
    row_split_overrides = _split_recycled_gids(rows, tracks)

    gids = sorted(tracks.keys())
    candidates: List[Tuple[float, int, int]] = []
    pair_seen: set[Tuple[int, int]] = set()

    for ga in gids:
        ta = tracks[ga]
        if ta.mean_desc is None:
            continue
        for gb in gids:
            if gb == ga:
                continue
            tb = tracks[gb]
            if tb.mean_desc is None:
                continue
            pair = (min(ga, gb), max(ga, gb))
            if pair in pair_seen:
                continue
            pair_seen.add(pair)

            if ta.end_f <= tb.start_f:
                t_prev, t_next = ta, tb
                gid_prev, gid_next = ga, gb
            elif tb.end_f <= ta.start_f:
                t_prev, t_next = tb, ta
                gid_prev, gid_next = gb, ga
            else:
                overlap = max(0, min(ta.end_f, tb.end_f) - max(ta.start_f, tb.start_f) + 1)
                if overlap > 3:
                    continue
                if ta.end_f <= tb.end_f:
                    t_prev, t_next = ta, tb
                    gid_prev, gid_next = ga, gb
                else:
                    t_prev, t_next = tb, ta
                    gid_prev, gid_next = gb, ga

            gap = max(0, int(t_next.start_f - t_prev.end_f))
            if gap > max_gap_frames:
                continue

            sim_last_first = _cos(t_prev.last_desc, t_next.first_desc)
            sim_mean = _cos(t_prev.mean_desc, t_next.mean_desc)
            sim_topk = _topk_similarity(t_prev.sample_descs, t_next.sample_descs, k=6)
            if sim_topk < 0:
                continue
            spatial = _spatial_boundary_score(t_prev, t_next)
            h_ratio = min(t_prev.median_h_ratio, t_next.median_h_ratio) / max(t_prev.median_h_ratio, t_next.median_h_ratio, 1e-6)
            h_ratio = float(np.clip(h_ratio, 0.0, 1.0))
            aspect_ratio = min(t_prev.median_aspect, t_next.median_aspect) / max(t_prev.median_aspect, t_next.median_aspect, 1e-6)
            aspect_ratio = float(np.clip(aspect_ratio, 0.0, 1.0))
            shape_score = 0.65 * h_ratio + 0.35 * aspect_ratio
            zone = _zone_compatibility(t_prev.end_zone, t_next.start_zone, gap=gap, max_gap=max_gap_frames)

            p0 = np.array(t_prev.last_row.center, dtype=np.float32)
            p1 = np.array(t_next.first_row.center, dtype=np.float32)
            disp = p1 - p0
            dn = float(np.linalg.norm(disp))
            if dn > 1e-6:
                disp = disp / dn
            dir_align = float(np.dot(t_prev.dir_vec, disp)) if np.linalg.norm(t_prev.dir_vec) > 1e-6 else 0.0
            dir_align = float(np.clip(0.5 * (dir_align + 1.0), 0.0, 1.0))

            gap_norm = min(1.0, gap / max(1.0, float(max_gap_frames)))
            min_topk = 0.60 + 0.10 * gap_norm
            min_mean = 0.56 + 0.10 * gap_norm

            if sim_topk < min_topk:
                continue
            if sim_mean < min_mean:
                continue
            if shape_score < 0.58:
                continue
            if zone < (0.22 if gap >= 120 else 0.12):
                continue

            boundary_pair = (t_prev.end_zone != "center") and (t_next.start_zone != "center")
            if gap >= 180 and (not boundary_pair) and sim_topk < 0.78:
                continue

            # For very short gaps (≤30 frames) a legitimate same-person re-entry
            # at side-view should produce high combined-descriptor similarity.
            # Raising the topk floor here prevents color/attire confusion from
            # merging two different persons across a brief camera absence.
            if gap <= 30 and sim_topk < 0.77:
                continue

            score = (
                0.44 * sim_topk
                + 0.22 * sim_mean
                + 0.12 * max(0.0, sim_last_first)
                + 0.08 * shape_score
                + 0.06 * zone
                + 0.04 * spatial
                + 0.04 * dir_align
            )
            score -= 0.07 * gap_norm

            if score >= min_merge_score:
                candidates.append((float(score), int(gid_prev), int(gid_next)))

    # Greedy merge of connected components while forbidding temporal-overlap conflicts.
    candidates.sort(key=lambda x: x[0], reverse=True)
    comp_of: Dict[int, int] = {int(g): int(g) for g in gids}
    comp_members: Dict[int, set[int]] = {int(g): {int(g)} for g in gids}

    def _merge_components(a_root: int, b_root: int) -> int:
        ra = int(a_root)
        rb = int(b_root)
        if ra == rb:
            return ra
        keep = ra if len(comp_members[ra]) >= len(comp_members[rb]) else rb
        drop = rb if keep == ra else ra
        for gid in comp_members[drop]:
            comp_of[int(gid)] = int(keep)
        comp_members[keep].update(comp_members[drop])
        del comp_members[drop]
        return int(keep)

    for score, ga, gb in candidates:
        ra = int(comp_of[int(ga)])
        rb = int(comp_of[int(gb)])
        if ra == rb:
            continue
        if _components_overlap(comp_members[ra], comp_members[rb], tracks, max_overlap_frames=3):
            continue
        _merge_components(ra, rb)

    # Collapse each component to earliest-start gid.
    id_map: Dict[int, int] = {}
    for root, members in comp_members.items():
        best_gid = min(members, key=lambda g: (tracks[g].start_f, g))
        for gid in members:
            id_map[int(gid)] = int(best_gid)

    merged_pairs = sum(1 for g in gids if id_map[g] != g)
    if merged_pairs == 0 and not row_split_overrides:
        dedup_rows = _enforce_unique_positive_ids_per_frame(tracks_csv_path=tracks_csv_path)
        return {
            "merged_pairs": 0,
            "id_map": id_map,
            "total_ids": len(gids),
            "dedup_rows": int(dedup_rows),
        }

    # Rewrite CSV with stitched IDs (and any pre-split GID overrides).
    # row_split_overrides maps row index -> new_gid for post-gap rows that were
    # split off from a recycled ByteTrack ID; id_map then applies stitch merges
    # on top of those new GIDs.
    with tracks_csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        out_rows = []
        for csv_idx, raw in enumerate(reader):
            try:
                gid = int(raw["global_id"])
            except Exception:
                out_rows.append(raw)
                continue
            if gid > 0:
                split_gid = row_split_overrides.get(csv_idx, gid)
                raw["global_id"] = str(id_map.get(split_gid, split_gid))
            out_rows.append(raw)

    with tracks_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    dedup_rows = _enforce_unique_positive_ids_per_frame(tracks_csv_path=tracks_csv_path)
    return {
        "merged_pairs": merged_pairs,
        "id_map": id_map,
        "total_ids": len(gids),
        "dedup_rows": int(dedup_rows),
    }


def align_ids_to_reference_video(
    *,
    reference_video_path: Path,
    reference_tracks_csv_path: Path,
    target_video_path: Path,
    target_tracks_csv_path: Path,
    reid_weights_path: str | None = None,
    min_ref_rows: int = 45,
    min_target_rows: int = 18,
    min_score: float = 0.69,
    min_margin: float = 0.04,
    max_overlap_frames_per_ref: int = 6,
    max_source_gids_per_ref: int = 3,
    segment_max_gap: int = 24,
    enable_source_fallback: bool = True,
) -> dict:
    if (not reference_video_path.exists()) or (not reference_tracks_csv_path.exists()):
        return {"applied": False, "reason": "reference_missing", "mapped_ids": 0}
    if (not target_video_path.exists()) or (not target_tracks_csv_path.exists()):
        return {"applied": False, "reason": "target_missing", "mapped_ids": 0}

    _f_ref, ref_rows = _load_rows(reference_tracks_csv_path)
    _f_tgt, tgt_rows = _load_rows(target_tracks_csv_path)
    if not ref_rows or not tgt_rows:
        return {"applied": False, "reason": "empty_rows", "mapped_ids": 0}

    ref_tracks = _build_track_stats(ref_rows)
    tgt_tracks, tgt_row_to_seg, tgt_seg_to_src = _build_track_segments(
        tgt_rows,
        max_gap_frames=max(1, int(segment_max_gap)),
    )
    if not ref_tracks or not tgt_tracks:
        return {"applied": False, "reason": "no_positive_ids", "mapped_ids": 0}

    _extract_track_descriptors(reference_video_path, ref_tracks, reid_weights_path=reid_weights_path)
    _extract_track_descriptors(target_video_path, tgt_tracks, reid_weights_path=reid_weights_path)

    def _shape(st: TrackStats) -> Tuple[float, float]:
        hs: List[float] = []
        ars: List[float] = []
        for r in st.rows:
            h = max(1.0, float(r.y2 - r.y1))
            w = max(1.0, float(r.x2 - r.x1))
            hs.append(h)
            ars.append(w / h)
        if not hs:
            return 1.0, 0.5
        return float(np.median(hs)), float(np.median(ars))

    def _desc(st: TrackStats) -> Optional[np.ndarray]:
        if st.mean_desc is not None:
            return st.mean_desc
        if st.last_desc is not None and st.first_desc is not None:
            return _l2(0.55 * st.last_desc + 0.45 * st.first_desc)
        if st.last_desc is not None:
            return st.last_desc
        if st.first_desc is not None:
            return st.first_desc
        return None

    def _topk_sim(a: TrackStats, b: TrackStats, k: int = 5) -> float:
        aa = [x for x in (a.sample_descs or []) if x is not None]
        bb = [x for x in (b.sample_descs or []) if x is not None]
        if not aa or not bb:
            return -1.0
        sims: List[float] = []
        for da in aa:
            # best match of this sample in b
            sims.append(max(_cos(da, db) for db in bb))
        if not sims:
            return -1.0
        sims.sort(reverse=True)
        top = sims[: max(1, min(int(k), len(sims)))]
        return float(np.mean(top))

    ref_ids = [gid for gid, st in ref_tracks.items() if len(st.rows) >= int(min_ref_rows) and _desc(st) is not None]
    tgt_ids = [gid for gid, st in tgt_tracks.items() if len(st.rows) >= int(min_target_rows) and _desc(st) is not None]
    if not ref_ids or not tgt_ids:
        return {"applied": False, "reason": "insufficient_profiles", "mapped_ids": 0}

    # Candidate scores: higher is better.
    cand_by_tgt: Dict[int, List[Tuple[int, float, float, float, float]]] = defaultdict(list)
    for tgid in tgt_ids:
        st_t = tgt_tracks[int(tgid)]
        dt = _desc(st_t)
        if dt is None:
            continue
        ht, art = _shape(st_t)
        for rgid in ref_ids:
            st_r = ref_tracks[int(rgid)]
            dr = _desc(st_r)
            if dr is None:
                continue
            sim_main = _cos(dt, dr)
            sim_topk = _topk_sim(st_t, st_r, k=5)
            hr, arr = _shape(st_r)
            h_sim = float(np.exp(-abs(np.log((ht + 1e-6) / (hr + 1e-6)))))
            ar_sim = float(np.exp(-abs(np.log((art + 1e-6) / (arr + 1e-6)))))
            shape_sim = 0.60 * h_sim + 0.40 * ar_sim

            # Deep/main similarity is primary signal; top-k and shape are support cues.
            s_top = sim_topk if sim_topk >= 0.0 else sim_main
            score = 0.66 * sim_main + 0.24 * s_top + 0.10 * shape_sim
            cand_by_tgt[int(tgid)].append((int(rgid), float(score), float(sim_main), float(s_top), float(shape_sim)))
        cand_by_tgt[int(tgid)].sort(key=lambda x: x[1], reverse=True)

    if not cand_by_tgt:
        return {"applied": False, "reason": "no_candidates", "mapped_ids": 0}

    def _overlap_frames(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
        return max(0, min(int(a_end), int(b_end)) - max(int(a_start), int(b_start)) + 1)

    # Multi-fragment assignment: allow multiple target IDs -> same reference ID
    # only when their time spans do not materially overlap.
    t_sorted = sorted(tgt_ids, key=lambda g: len(tgt_tracks[int(g)].rows), reverse=True)
    mapped_t_to_r: Dict[int, int] = {}
    ref_claims: Dict[int, List[int]] = defaultdict(list)
    ref_source_claims: Dict[int, set[int]] = defaultdict(set)
    pair_score: Dict[Tuple[int, int], float] = {}
    rejected_ambiguous = 0
    rejected_overlap = 0
    rejected_source_cap = 0

    for tgid in t_sorted:
        cands = cand_by_tgt.get(int(tgid), [])
        if not cands:
            continue
        best_score = float(cands[0][1])
        second_score = float(cands[1][1]) if len(cands) > 1 else -1.0
        margin = float(best_score - second_score)
        # Require clearer margin for moderate scores; relax at very strong scores.
        need_margin = float(min_margin) if best_score < 0.83 else float(min_margin) * 0.5
        if best_score < float(min_score) or (margin < need_margin and best_score < 0.88):
            rejected_ambiguous += 1
            continue

        st_t = tgt_tracks[int(tgid)]
        src_gid = int(tgt_seg_to_src.get(int(tgid), -1))
        chosen = None
        for rgid, score, _sm, _st, _ss in cands:
            if float(score) < float(min_score):
                break
            conflict = False
            for other_t in ref_claims.get(int(rgid), []):
                st_o = tgt_tracks[int(other_t)]
                ov = _overlap_frames(st_t.start_f, st_t.end_f, st_o.start_f, st_o.end_f)
                if ov > int(max_overlap_frames_per_ref):
                    conflict = True
                    break
            if conflict:
                rejected_overlap += 1
                continue
            if src_gid > 0:
                src_claims = ref_source_claims.get(int(rgid), set())
                if (
                    int(src_gid) not in src_claims
                    and len(src_claims) >= int(max_source_gids_per_ref)
                ):
                    rejected_source_cap += 1
                    continue
            chosen = (int(rgid), float(score))
            break

        if chosen is None:
            continue
        rgid, sc = chosen
        mapped_t_to_r[int(tgid)] = int(rgid)
        ref_claims[int(rgid)].append(int(tgid))
        if src_gid > 0:
            ref_source_claims[int(rgid)].add(int(src_gid))
        pair_score[(int(tgid), int(rgid))] = float(sc)

    if not mapped_t_to_r:
        return {"applied": False, "reason": "no_mapping_selected", "mapped_ids": 0}

    # Build final remap for ALL target IDs.
    tgt_all = sorted(int(g) for g in tgt_tracks.keys() if int(g) > 0)
    max_ref_gid = max(int(g) for g in ref_tracks.keys() if int(g) > 0) if ref_tracks else 0
    used_new: set[int] = set(int(v) for v in mapped_t_to_r.values())
    next_gid = int(max_ref_gid + 1)
    final_gid_map: Dict[int, int] = {}

    # First assign mapped IDs to reference labels.
    for tgid in tgt_all:
        if int(tgid) in mapped_t_to_r:
            final_gid_map[int(tgid)] = int(mapped_t_to_r[int(tgid)])

    # Build source-gid dominant reference mapping from confident matches.
    src_ref_vote: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for tgid, rgid in mapped_t_to_r.items():
        src = int(tgt_seg_to_src.get(int(tgid), -1))
        if src <= 0:
            continue
        w = float(len(tgt_tracks[int(tgid)].rows))
        src_ref_vote[int(src)][int(rgid)] += max(1.0, w)

    src_primary_ref: Dict[int, int] = {}
    for src, votes in src_ref_vote.items():
        if not votes:
            continue
        best_r = max(votes.items(), key=lambda kv: kv[1])[0]
        src_primary_ref[int(src)] = int(best_r)

    # Then assign deterministic new IDs (for unmatched target IDs).
    unmatched = [g for g in tgt_all if g not in final_gid_map]
    unmatched.sort(key=lambda g: (int(tgt_tracks[int(g)].start_f), int(g)))
    for tgid in unmatched:
        src = int(tgt_seg_to_src.get(int(tgid), -1))
        # Fallback: preserve source-track consistency by inheriting the dominant
        # canonical reference ID already established for this source gid.
        if bool(enable_source_fallback) and src > 0 and src in src_primary_ref:
            rgid = int(src_primary_ref[int(src)])
            st_t = tgt_tracks[int(tgid)]
            conflict = False
            for other_t in ref_claims.get(int(rgid), []):
                st_o = tgt_tracks[int(other_t)]
                ov = _overlap_frames(st_t.start_f, st_t.end_f, st_o.start_f, st_o.end_f)
                if ov > int(max_overlap_frames_per_ref):
                    conflict = True
                    break
            if not conflict:
                final_gid_map[int(tgid)] = int(rgid)
                ref_claims[int(rgid)].append(int(tgid))
                continue

        while int(next_gid) in used_new:
            next_gid += 1
        final_gid_map[int(tgid)] = int(next_gid)
        used_new.add(int(next_gid))
        next_gid += 1

    # Rewrite target CSV.
    with target_tracks_csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        out_rows = []
        changed_rows = 0
        for row_idx, raw in enumerate(reader):
            try:
                gid = int(raw.get("global_id", 0) or 0)
            except Exception:
                out_rows.append(raw)
                continue
            if gid > 0:
                seg_id = int(tgt_row_to_seg.get(int(row_idx), -1))
                if seg_id > 0:
                    ng = int(final_gid_map.get(int(seg_id), int(gid)))
                else:
                    ng = int(gid)
                if ng != gid:
                    changed_rows += 1
                raw["global_id"] = str(int(ng))
            else:
                raw["global_id"] = "0"
            out_rows.append(raw)

    with target_tracks_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    dedup_rows = _enforce_unique_positive_ids_per_frame(tracks_csv_path=target_tracks_csv_path)
    mapped_rows = []
    for t, r in sorted(mapped_t_to_r.items(), key=lambda kv: (kv[1], kv[0])):
        mapped_rows.append(
            {
                "target_seg_id": int(t),
                "target_src_gid": int(tgt_seg_to_src.get(int(t), -1)),
                "reference_gid": int(r),
                "score": float(pair_score.get((int(t), int(r)), 0.0)),
            }
        )
    return {
        "applied": True,
        "mapped_ids": int(len(mapped_t_to_r)),
        "changed_rows": int(changed_rows),
        "dedup_rows": int(dedup_rows),
        "max_ref_gid": int(max_ref_gid),
        "rejected_ambiguous": int(rejected_ambiguous),
        "rejected_overlap": int(rejected_overlap),
        "rejected_source_cap": int(rejected_source_cap),
        "mapping": mapped_rows,
    }


def relabel_to_reference_profiles_with_memory(
    *,
    reference_video_path: Path,
    reference_tracks_csv_path: Path,
    target_video_path: Path,
    target_tracks_csv_path: Path,
    canonical_ids: Optional[set[int]] = None,
    reid_weights_path: str | None = None,
    overlap_iou_thresh: float = 0.12,
    temporal_weight: float = 0.22,
    temporal_max_age: int = 26,
    lock_hold_frames: int = 26,
    lock_bonus: float = 0.10,
    min_assign_score: float = 0.40,
    base_reassign_margin: float = 0.05,
    overlap_reassign_margin: float = 0.01,
    osnet_sparse_stride: int = 4,
    min_seed_area_ratio: float = 0.0036,
    proto_seed_max_iou: float = 0.09,
    min_seed_samples: int = 8,
    proto_seed_keep_top_k: int = 28,
) -> dict:
    if (not reference_video_path.exists()) or (not reference_tracks_csv_path.exists()):
        return {"applied": False, "reason": "reference_missing", "changed_rows": 0}
    if (not target_video_path.exists()) or (not target_tracks_csv_path.exists()):
        return {"applied": False, "reason": "target_missing", "changed_rows": 0}

    canonical = set(int(x) for x in (canonical_ids or set()) if int(x) > 0)

    _rf, ref_rows = _load_rows(reference_tracks_csv_path)
    fieldnames, tgt_rows = _load_rows(target_tracks_csv_path)
    if not ref_rows or not tgt_rows:
        return {"applied": False, "reason": "empty_rows", "changed_rows": 0}

    ref_tracks_all = _build_track_stats(ref_rows)
    if canonical:
        ref_tracks = {gid: st for gid, st in ref_tracks_all.items() if int(gid) in canonical}
    else:
        ref_tracks = dict(ref_tracks_all)
    if not ref_tracks:
        return {"applied": False, "reason": "no_reference_ids", "changed_rows": 0}

    _extract_track_descriptors(reference_video_path, ref_tracks, reid_weights_path=reid_weights_path)

    ref_proto: Dict[int, np.ndarray] = {}
    for gid, st in ref_tracks.items():
        d = st.mean_desc
        if d is None:
            d = st.last_desc
        if d is None:
            d = st.first_desc
        if d is not None:
            ref_proto[int(gid)] = _l2(d)
    if not ref_proto:
        return {"applied": False, "reason": "no_reference_profiles", "changed_rows": 0}

    id_by_idx: Dict[int, int] = {int(r.idx): int(r.gid) for r in tgt_rows}
    rows_by_frame: Dict[int, List[TrackRow]] = defaultdict(list)
    for r in tgt_rows:
        gid = int(r.gid)
        if gid <= 0:
            continue
        rows_by_frame[int(r.frame_idx)].append(r)
    if not rows_by_frame:
        return {"applied": False, "reason": "no_target_rows_for_ids", "changed_rows": 0}

    # Detect overlap-heavy frames where relabeling may need extra flexibility.
    overlap_frames: set[int] = set()
    frame_max_iou: Dict[int, Dict[int, float]] = defaultdict(dict)
    for fr, fr_rows in rows_by_frame.items():
        if len(fr_rows) < 2:
            continue
        has_overlap = False
        for i, ri in enumerate(fr_rows):
            mx = 0.0
            for j in range(i + 1, len(fr_rows)):
                rj = fr_rows[j]
                iou = _iou_xyxy(ri.box, rj.box)
                if iou >= float(overlap_iou_thresh):
                    has_overlap = True
                if iou > mx:
                    mx = float(iou)
                prev = float(frame_max_iou[fr].get(int(rj.idx), 0.0))
                if iou > prev:
                    frame_max_iou[fr][int(rj.idx)] = float(iou)
            prev_i = float(frame_max_iou[fr].get(int(ri.idx), 0.0))
            if mx > prev_i:
                frame_max_iou[fr][int(ri.idx)] = float(mx)
        if has_overlap:
            overlap_frames.add(int(fr))
    overlap_set = set(int(x) for x in overlap_frames)

    extractor: Optional[ReidExtractor] = None
    extractor_error: str = ""
    os_dim = 512
    try:
        extractor = ReidExtractor(model_name="osnet_x1_0", device="cpu", model_path=reid_weights_path)
    except Exception as e_cpu:
        extractor_error = repr(e_cpu)
        try:
            extractor = ReidExtractor(model_name="osnet_x1_0", device=None, model_path=reid_weights_path)
        except Exception as e_fallback:
            extractor_error = f"{repr(e_cpu)} | fallback: {repr(e_fallback)}"
            extractor = None

    cap = cv2.VideoCapture(str(target_video_path))
    if not cap.isOpened():
        return {"applied": False, "reason": "target_video_open_failed", "changed_rows": 0}
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if vw <= 0 or vh <= 0:
        cap.release()
        return {"applied": False, "reason": "invalid_target_video_shape", "changed_rows": 0}
    area_den = float(max(1, vw * vh))

    desc_by_row: Dict[int, np.ndarray] = {}
    seed_samples: Dict[int, List[Tuple[float, np.ndarray]]] = defaultdict(list)
    fr = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        fr_rows = rows_by_frame.get(int(fr), [])
        if fr_rows:
            use_osnet = bool(
                extractor is not None
                and (
                    int(fr) in overlap_set
                    or int(osnet_sparse_stride) <= 1
                    or (int(fr) % int(max(1, osnet_sparse_stride)) == 0)
                )
            )
            desc_map = _build_descs_for_rows(
                frame,
                fr_rows,
                extractor=extractor if use_osnet else None,
                os_dim=os_dim,
            )
            for r in fr_rows:
                idx = int(r.idx)
                d = desc_map.get(idx, None)
                if d is None:
                    continue
                desc_by_row[idx] = d.astype(np.float32)

                bw = max(1.0, float(r.x2 - r.x1))
                bh = max(1.0, float(r.y2 - r.y1))
                area_ratio = float((bw * bh) / area_den)
                if area_ratio < float(min_seed_area_ratio):
                    continue
                max_ov = float(frame_max_iou.get(int(fr), {}).get(idx, 0.0))
                if max_ov > float(proto_seed_max_iou):
                    continue
                gid = int(id_by_idx.get(idx, int(r.gid)))
                sc = float(area_ratio - 0.20 * max_ov)
                seed_samples[int(gid)].append((sc, d.astype(np.float32)))
        fr += 1
    cap.release()

    if not desc_by_row:
        return {"applied": False, "reason": "no_target_descriptors", "changed_rows": 0}

    def _build_proto_from_samples(
        samples: List[Tuple[float, np.ndarray]],
        *,
        min_needed: int,
    ) -> Optional[np.ndarray]:
        if not samples:
            return None
        samples_sorted = sorted(samples, key=lambda x: float(x[0]), reverse=True)
        top_desc = [d for _score, d in samples_sorted[: max(1, int(proto_seed_keep_top_k))]]
        if len(top_desc) < int(min_needed):
            return None
        return _l2(np.mean(np.stack(top_desc, axis=0), axis=0))

    missing_ref_ids: List[int] = []
    seeded_ids: List[int] = []
    seeded_from_source: Dict[int, int] = {}
    if canonical:
        missing_ref_ids = sorted(int(g) for g in canonical if int(g) not in ref_proto)

    min_needed_direct = max(3, int(min_seed_samples) // 3)
    for gid in missing_ref_ids:
        direct = _build_proto_from_samples(seed_samples.get(int(gid), []), min_needed=min_needed_direct)
        if direct is None:
            continue
        ref_proto[int(gid)] = direct
        seeded_ids.append(int(gid))
        seeded_from_source[int(gid)] = int(gid)

    unresolved_missing = sorted(int(g) for g in (canonical or set()) if int(g) not in ref_proto)
    if unresolved_missing:
        min_needed_fallback = max(2, int(min_seed_samples) // 4)
        descs_by_source: Dict[int, List[np.ndarray]] = defaultdict(list)
        frames_by_source: Dict[int, List[int]] = defaultdict(list)
        for r in tgt_rows:
            idx = int(r.idx)
            gid = int(id_by_idx.get(idx, int(r.gid)))
            if gid <= 0:
                continue
            d = desc_by_row.get(idx, None)
            if d is None:
                continue
            descs_by_source[int(gid)].append(d.astype(np.float32))
            frames_by_source[int(gid)].append(int(r.frame_idx))

        source_candidates: List[Tuple[float, int, np.ndarray, float, int, int]] = []
        for src_gid, src_descs in descs_by_source.items():
            if int(src_gid) in ref_proto:
                continue
            if len(src_descs) < int(min_needed_fallback):
                continue

            seed_proto = _build_proto_from_samples(
                seed_samples.get(int(src_gid), []),
                min_needed=min_needed_fallback,
            )
            if seed_proto is not None:
                proto = seed_proto
            else:
                proto = _l2(np.mean(np.stack(src_descs, axis=0), axis=0))

            if not np.isfinite(proto).all():
                continue
            sims = [float(_cos(proto, rp)) for rp in ref_proto.values()]
            max_sim = max(sims) if sims else 0.0
            novelty = max(0.0, 1.0 - float(max_sim))
            frs = frames_by_source.get(int(src_gid), [])
            span = (max(frs) - min(frs) + 1) if frs else 1
            support = int(len(src_descs))
            score = float(np.log1p(float(support)) + 0.0015 * float(span) + 3.2 * novelty)
            source_candidates.append((score, int(src_gid), proto, float(max_sim), int(support), int(span)))

        source_candidates.sort(key=lambda x: x[0], reverse=True)
        used_source_gids: set[int] = set()
        for miss_gid in list(unresolved_missing):
            chosen: Optional[Tuple[float, int, np.ndarray, float, int, int]] = None
            for cand in source_candidates:
                _score, src_gid, _proto, max_sim, support, span = cand
                if int(src_gid) in used_source_gids:
                    continue
                if support < int(min_needed_fallback):
                    continue
                if span < 3:
                    continue
                # Prefer candidates not too similar to already-known canonical identities.
                if max_sim <= 0.82:
                    chosen = cand
                    break
                if chosen is None:
                    chosen = cand
            if chosen is None:
                continue
            _score, src_gid, proto, _max_sim, _support, _span = chosen
            ref_proto[int(miss_gid)] = proto
            seeded_ids.append(int(miss_gid))
            seeded_from_source[int(miss_gid)] = int(src_gid)
            used_source_gids.add(int(src_gid))

        unresolved_missing = sorted(int(g) for g in (canonical or set()) if int(g) not in ref_proto)

    candidate_ids = sorted(int(g) for g in ref_proto.keys() if int(g) > 0 and ((not canonical) or int(g) in canonical))
    if len(candidate_ids) < 2:
        return {
            "applied": False,
            "reason": "insufficient_candidate_profiles",
            "candidate_ids": candidate_ids,
            "changed_rows": 0,
        }
    gid_to_col = {int(g): i for i, g in enumerate(candidate_ids)}

    # Temporal memory of each canonical identity to preserve continuity in overlap.
    mem_last: Dict[int, Tuple[int, Tuple[float, float, float, float], np.ndarray]] = {}
    mem_prev: Dict[int, Tuple[int, Tuple[float, float, float, float]]] = {}
    gid_lock_until: Dict[int, int] = {}
    mem_desc_proto: Dict[int, np.ndarray] = {int(g): _l2(d.copy()) for g, d in ref_proto.items()}

    def _temporal_score(
        gid: int,
        row: TrackRow,
        d: np.ndarray,
        frame_idx: int,
    ) -> float:
        rec = mem_last.get(int(gid), None)
        if rec is None:
            return 0.0
        last_f, last_box, last_desc = rec
        age = int(frame_idx) - int(last_f)
        if age <= 0 or age > int(temporal_max_age):
            return 0.0

        lx1, ly1, lx2, ly2 = [float(v) for v in last_box]
        lcx = 0.5 * (lx1 + lx2)
        lcy = 0.5 * (ly1 + ly2)
        lh = max(1.0, ly2 - ly1)

        pred_cx = lcx
        pred_cy = lcy
        prv = mem_prev.get(int(gid), None)
        if prv is not None:
            pf, pbox = prv
            dt = max(1, int(last_f) - int(pf))
            px1, py1, px2, py2 = [float(v) for v in pbox]
            pcx = 0.5 * (px1 + px2)
            pcy = 0.5 * (py1 + py2)
            vx = (lcx - pcx) / float(dt)
            vy = (lcy - pcy) / float(dt)
            pred_cx = lcx + vx * float(age)
            pred_cy = lcy + vy * float(age)

        cx, cy = row.center
        rh = max(1.0, row.h)
        dist_norm = float(np.hypot(cx - pred_cx, cy - pred_cy) / max(lh, rh, 1.0))
        spatial = float(np.exp(-dist_norm))
        app = max(0.0, float(_cos(d, last_desc)))
        age_pen = 1.0 - 0.35 * min(1.0, float(age) / max(1.0, float(temporal_max_age)))
        return float((0.72 * spatial + 0.28 * app) * age_pen)

    changed_rows = 0
    all_frames = sorted(int(k) for k in rows_by_frame.keys())
    for fr in all_frames:
        fr_rows = [r for r in rows_by_frame.get(int(fr), []) if int(r.idx) in desc_by_row]
        if not fr_rows:
            continue

        n = len(fr_rows)
        m = len(candidate_ids)
        score = np.full((n, m), -1.0, dtype=np.float32)
        row_old_col: List[int] = []
        row_old_score: List[float] = []
        lock_mask = np.zeros((n,), dtype=np.int32)
        in_overlap = bool(int(fr) in overlap_set)

        for i, r in enumerate(fr_rows):
            idx = int(r.idx)
            d = desc_by_row[int(idx)]
            old_gid = int(id_by_idx.get(int(idx), int(r.gid)))
            old_col = int(gid_to_col.get(int(old_gid), -1))
            row_old_col.append(old_col)
            for j, gid in enumerate(candidate_ids):
                sim_ref = float(_cos(d, ref_proto[int(gid)]))
                sim_mem = float(_cos(d, mem_desc_proto.get(int(gid), None)))
                temporal = _temporal_score(int(gid), r, d, int(fr))
                inertia = 0.08 if int(gid) == int(old_gid) else 0.0
                lock = 0.0
                if int(fr) <= int(gid_lock_until.get(int(gid), -1)) and temporal > 0.25:
                    lock = float(lock_bonus)
                score[i, j] = float(
                    0.62 * sim_ref
                    + 0.14 * sim_mem
                    + float(temporal_weight) * temporal
                    + inertia
                    + lock
                )
            if old_col >= 0:
                row_old_score.append(float(score[i, old_col]))
            else:
                row_old_score.append(-1.0)

            # Lock stable rows to old gid unless strong evidence says otherwise.
            if old_col >= 0:
                s_old = float(score[i, old_col])
                s_best = float(np.max(score[i, :])) if m > 0 else -1.0
                req = float(overlap_reassign_margin if in_overlap else base_reassign_margin)
                movable = bool(s_old < float(min_assign_score) or s_best >= s_old + req)
                if not movable:
                    score[i, :] = -1e6
                    score[i, old_col] = float(max(s_old, float(min_assign_score)) + 0.25)
                    lock_mask[i] = 1

        if linear_sum_assignment is not None:
            rr, cc = linear_sum_assignment(-score)
            assigned_pairs = [(int(r), int(c)) for r, c in zip(rr, cc)]
        else:
            assigned_pairs = []
            used_rows: set[int] = set()
            used_cols: set[int] = set()
            candidates = []
            for i in range(n):
                for j in range(m):
                    candidates.append((float(score[i, j]), int(i), int(j)))
            candidates.sort(reverse=True, key=lambda x: x[0])
            for sc, i, j in candidates:
                if i in used_rows or j in used_cols:
                    continue
                used_rows.add(i)
                used_cols.add(j)
                assigned_pairs.append((int(i), int(j)))
                if len(used_rows) >= n:
                    break

        assigned_col_by_row: Dict[int, int] = {int(i): int(j) for i, j in assigned_pairs}
        used_cols: set[int] = set(int(j) for _i, j in assigned_pairs)
        final_gid_by_row: Dict[int, int] = {}
        for i, r in enumerate(fr_rows):
            old_col = int(row_old_col[i])
            old_gid = int(id_by_idx.get(int(r.idx), int(r.gid)))
            j = int(assigned_col_by_row.get(int(i), -1))
            if j < 0 or j >= m:
                if old_col >= 0 and old_col not in used_cols:
                    sc_old = float(score[i, old_col])
                    if sc_old >= float(min_assign_score) - 0.02:
                        final_gid = int(candidate_ids[old_col])
                        used_cols.add(int(old_col))
                    else:
                        final_gid = 0
                else:
                    final_gid = 0
            else:
                cand_gid = int(candidate_ids[int(j)])
                sc = float(score[i, j])
                if sc < float(min_assign_score) and old_col >= 0:
                    final_gid = int(old_gid)
                else:
                    final_gid = int(cand_gid)
            final_gid_by_row[int(i)] = int(final_gid)

        # Update identities and temporal memory in row order.
        for i, r in enumerate(fr_rows):
            idx = int(r.idx)
            old_gid = int(id_by_idx.get(int(idx), int(r.gid)))
            new_gid = int(final_gid_by_row.get(int(i), int(old_gid)))
            if new_gid != old_gid:
                changed_rows += 1
                id_by_idx[int(idx)] = int(new_gid)

            d = desc_by_row.get(int(idx), None)
            if d is None or new_gid not in gid_to_col:
                continue

            prev_proto = mem_desc_proto.get(int(new_gid), None)
            if prev_proto is None:
                mem_desc_proto[int(new_gid)] = _l2(d)
            else:
                mem_desc_proto[int(new_gid)] = _l2(0.93 * prev_proto + 0.07 * d)

            rec = mem_last.get(int(new_gid), None)
            if rec is not None and int(rec[0]) < int(fr):
                mem_prev[int(new_gid)] = (int(rec[0]), tuple(rec[1]))
                last_desc = rec[2]
                merged_desc = _l2(0.84 * last_desc + 0.16 * d)
            else:
                merged_desc = _l2(d)
            mem_last[int(new_gid)] = (
                int(fr),
                (float(r.x1), float(r.y1), float(r.x2), float(r.y2)),
                merged_desc,
            )

            j = int(gid_to_col.get(int(new_gid), -1))
            if j >= 0:
                sc = float(score[i, j])
                row_scores = score[i, :]
                alt = [float(v) for k, v in enumerate(row_scores.tolist()) if int(k) != int(j)]
                second = max(alt) if alt else -1.0
                margin = float(sc - second)
                if sc >= max(float(min_assign_score) + 0.08, 0.58) and margin >= 0.02:
                    gid_lock_until[int(new_gid)] = max(
                        int(gid_lock_until.get(int(new_gid), -1)),
                        int(fr) + int(lock_hold_frames),
                    )

    if changed_rows <= 0:
        return {
            "applied": False,
            "reason": "no_changes",
            "candidate_ids": [int(x) for x in candidate_ids],
            "used_osnet": bool(extractor is not None),
            "osnet_error": str(extractor_error) if extractor is None and extractor_error else "",
            "missing_ref_ids": [int(x) for x in missing_ref_ids],
            "seeded_ids": [int(x) for x in seeded_ids],
            "seeded_from_source": {str(k): int(v) for k, v in seeded_from_source.items()},
            "unresolved_missing_ids": [int(x) for x in unresolved_missing],
            "changed_rows": 0,
        }

    import sys as _sys
    csv.field_size_limit(min(_sys.maxsize, 2 ** 31 - 1))
    with target_tracks_csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        out_rows = []
        for idx, raw in enumerate(reader):
            try:
                _gid_raw = str(raw.get("global_id", "0") or "0").replace("\x00", "").strip()
                _fallback_gid = int(round(float(_gid_raw))) if _gid_raw else 0
            except Exception:
                _fallback_gid = 0
            gid = int(id_by_idx.get(int(idx), _fallback_gid))
            raw["global_id"] = str(gid)
            out_rows.append(raw)

    with target_tracks_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    dedup_rows = _enforce_unique_positive_ids_per_frame(tracks_csv_path=target_tracks_csv_path)
    return {
        "applied": True,
        "changed_rows": int(changed_rows),
        "dedup_rows": int(dedup_rows),
        "candidate_ids": [int(x) for x in candidate_ids],
        "used_osnet": bool(extractor is not None),
        "osnet_error": str(extractor_error) if extractor is None and extractor_error else "",
        "missing_ref_ids": [int(x) for x in missing_ref_ids],
        "seeded_ids": [int(x) for x in seeded_ids],
        "seeded_from_source": {str(k): int(v) for k, v in seeded_from_source.items()},
        "unresolved_missing_ids": [int(x) for x in unresolved_missing],
    }


def relabel_to_seed_profiles_with_memory(
    *,
    video_path: Path,
    tracks_csv_path: Path,
    seed_profiles: Dict[int, List[Tuple[int, int]]],
    reid_weights_path: str | None = None,
    temporal_weight: float = 0.30,
    temporal_max_age: int = 42,
    lock_hold_frames: int = 52,
    lock_bonus: float = 0.12,
    min_assign_score: float = 0.48,
    strict_target_only: bool = False,
) -> dict:
    """
    Relabel rows to fixed target IDs using manually seeded profile anchors.

    `seed_profiles` format:
      {
        target_gid: [(frame_idx, source_gid_in_that_frame), ...],
        ...
      }
    """
    if not video_path.exists() or not tracks_csv_path.exists():
        return {"applied": False, "reason": "missing_input", "changed_rows": 0}
    if not seed_profiles:
        return {"applied": False, "reason": "empty_seed_profiles", "changed_rows": 0}

    fieldnames, rows = _load_rows(tracks_csv_path)
    if not rows:
        return {"applied": False, "reason": "empty_rows", "changed_rows": 0}

    target_ids = sorted(int(g) for g in seed_profiles.keys() if int(g) > 0)
    if len(target_ids) < 2:
        return {"applied": False, "reason": "insufficient_target_ids", "changed_rows": 0}

    rows_by_frame: Dict[int, List[TrackRow]] = defaultdict(list)
    for r in rows:
        if int(r.gid) > 0:
            rows_by_frame[int(r.frame_idx)].append(r)
    if not rows_by_frame:
        return {"applied": False, "reason": "no_positive_rows", "changed_rows": 0}

    seed_lookup: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for tgt, specs in seed_profiles.items():
        t = int(tgt)
        if t <= 0:
            continue
        for fr, src_gid in specs:
            seed_lookup[int(fr)].append((t, int(src_gid)))

    extractor: Optional[ReidExtractor] = None
    extractor_error: str = ""
    os_dim = 512
    try:
        extractor = ReidExtractor(model_name="osnet_x1_0", device="cpu", model_path=reid_weights_path)
    except Exception as e_cpu:
        extractor_error = repr(e_cpu)
        try:
            extractor = ReidExtractor(model_name="osnet_x1_0", device=None, model_path=reid_weights_path)
        except Exception as e_fallback:
            extractor_error = f"{repr(e_cpu)} | fallback: {repr(e_fallback)}"
            extractor = None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"applied": False, "reason": "video_open_failed", "changed_rows": 0}
    frame_w = float(max(1.0, cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1.0))
    frame_h = float(max(1.0, cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1.0))

    desc_by_idx: Dict[int, np.ndarray] = {}
    seed_descs: Dict[int, List[np.ndarray]] = defaultdict(list)
    seed_centers: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
    seed_locked_idx_to_target: Dict[int, int] = {}
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        fr_rows = rows_by_frame.get(int(frame_idx), [])
        if not fr_rows:
            frame_idx += 1
            continue

        desc_map = _build_descs_for_rows(frame, fr_rows, extractor=extractor, os_dim=os_dim)
        for r in fr_rows:
            d = desc_map.get(int(r.idx), None)
            if d is not None:
                desc_by_idx[int(r.idx)] = d.astype(np.float32)

        specs = seed_lookup.get(int(frame_idx), [])
        if specs:
            by_src_gid: Dict[int, List[TrackRow]] = defaultdict(list)
            for rr in fr_rows:
                by_src_gid[int(rr.gid)].append(rr)
            used_idx: set[int] = set()
            for tgt_gid, src_gid in specs:
                cands = [rr for rr in by_src_gid.get(int(src_gid), []) if int(rr.idx) not in used_idx]
                if not cands:
                    # Fallback A: use current target prototype similarity if any seed already exists.
                    cand_pool = [rr for rr in fr_rows if int(rr.idx) not in used_idx and int(rr.idx) in desc_map]
                    pick = None
                    cur_seed = seed_descs.get(int(tgt_gid), [])
                    if cand_pool and cur_seed:
                        proto_cur = _l2(np.mean(np.stack(cur_seed, axis=0), axis=0))
                        scored = []
                        for rr in cand_pool:
                            d_rr = desc_map.get(int(rr.idx), None)
                            if d_rr is None:
                                continue
                            sim = float(_cos(d_rr, proto_cur))
                            area = max(0.0, float(rr.x2 - rr.x1) * float(rr.y2 - rr.y1))
                            scored.append((sim, area, rr))
                        if scored:
                            scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
                            if float(scored[0][0]) >= 0.40:
                                pick = scored[0][2]
                    # Fallback B: if still missing, choose largest unmatched row in frame.
                    if pick is None and cand_pool:
                        pick = max(cand_pool, key=lambda rr: max(0.0, rr.x2 - rr.x1) * max(0.0, rr.y2 - rr.y1))
                    if pick is None:
                        continue
                else:
                    pick = max(cands, key=lambda rr: max(0.0, rr.x2 - rr.x1) * max(0.0, rr.y2 - rr.y1))
                d = desc_map.get(int(pick.idx), None)
                if d is not None:
                    seed_descs[int(tgt_gid)].append(d.astype(np.float32))
                    cx, cy = pick.center
                    seed_centers[int(tgt_gid)].append((float(cx) / frame_w, float(cy) / frame_h))
                    # Keep sparse seed anchors as hard identity evidence.
                    seed_locked_idx_to_target[int(pick.idx)] = int(tgt_gid)
                    used_idx.add(int(pick.idx))
        frame_idx += 1
    cap.release()

    if not desc_by_idx:
        return {"applied": False, "reason": "no_descriptors", "changed_rows": 0}

    proto: Dict[int, np.ndarray] = {}
    for tgt in target_ids:
        ds = seed_descs.get(int(tgt), [])
        if ds:
            proto[int(tgt)] = _l2(np.mean(np.stack(ds, axis=0), axis=0))

    # Fallback: if a target lacks seed descriptor, try rows currently using that gid.
    missing = [int(g) for g in target_ids if int(g) not in proto]
    if missing:
        by_gid_rows: Dict[int, List[TrackRow]] = defaultdict(list)
        for r in rows:
            if int(r.gid) > 0:
                by_gid_rows[int(r.gid)].append(r)
        for gid in missing:
            cands = []
            for r in by_gid_rows.get(int(gid), []):
                d = desc_by_idx.get(int(r.idx), None)
                if d is None:
                    continue
                area = max(0.0, float(r.x2 - r.x1)) * max(0.0, float(r.y2 - r.y1))
                cands.append((float(area), d))
            if len(cands) >= 3:
                cands.sort(key=lambda x: x[0], reverse=True)
                top = [d for _a, d in cands[:24]]
                proto[int(gid)] = _l2(np.mean(np.stack(top, axis=0), axis=0))

    unresolved = [int(g) for g in target_ids if int(g) not in proto]
    if unresolved and len(proto) < 2:
        return {
            "applied": False,
            "reason": "missing_seed_profiles",
            "missing_targets": unresolved,
            "seed_counts": {str(k): int(len(v)) for k, v in seed_descs.items()},
            "used_osnet": bool(extractor is not None),
            "osnet_error": str(extractor_error) if extractor is None and extractor_error else "",
            "changed_rows": 0,
        }

    active_target_ids = sorted(int(g) for g in proto.keys())
    if not active_target_ids:
        return {
            "applied": False,
            "reason": "no_active_seed_profiles",
            "missing_targets": unresolved,
            "seed_counts": {str(k): int(len(v)) for k, v in seed_descs.items()},
            "used_osnet": bool(extractor is not None),
            "osnet_error": str(extractor_error) if extractor is None and extractor_error else "",
            "changed_rows": 0,
        }

    seed_center_mean: Dict[int, Tuple[float, float]] = {}
    for gid, pts in seed_centers.items():
        if not pts:
            continue
        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]
        seed_center_mean[int(gid)] = (float(np.mean(xs)), float(np.mean(ys)))

    id_by_idx: Dict[int, int] = {int(r.idx): int(r.gid) for r in rows}
    gid_to_col = {int(g): i for i, g in enumerate(active_target_ids)}
    proto_dyn: Dict[int, np.ndarray] = {int(g): _l2(v.copy()) for g, v in proto.items()}
    target_set = set(int(g) for g in active_target_ids)

    mem_last: Dict[int, Tuple[int, Tuple[float, float, float, float], np.ndarray]] = {}
    mem_prev: Dict[int, Tuple[int, Tuple[float, float, float, float]]] = {}
    gid_lock_until: Dict[int, int] = {}

    def _temporal_score(
        gid: int,
        row: TrackRow,
        d: np.ndarray,
        fr: int,
    ) -> float:
        rec = mem_last.get(int(gid), None)
        if rec is None:
            return 0.0
        last_f, last_box, last_desc = rec
        age = int(fr) - int(last_f)
        if age <= 0 or age > int(temporal_max_age):
            return 0.0

        lx1, ly1, lx2, ly2 = [float(v) for v in last_box]
        lcx = 0.5 * (lx1 + lx2)
        lcy = 0.5 * (ly1 + ly2)
        lh = max(1.0, ly2 - ly1)

        pred_cx = lcx
        pred_cy = lcy
        prv = mem_prev.get(int(gid), None)
        if prv is not None:
            pf, pbox = prv
            dt = max(1, int(last_f) - int(pf))
            px1, py1, px2, py2 = [float(v) for v in pbox]
            pcx = 0.5 * (px1 + px2)
            pcy = 0.5 * (py1 + py2)
            vx = (lcx - pcx) / float(dt)
            vy = (lcy - pcy) / float(dt)
            pred_cx = lcx + vx * float(age)
            pred_cy = lcy + vy * float(age)

        cx, cy = row.center
        rh = max(1.0, row.h)
        dist_norm = float(np.hypot(cx - pred_cx, cy - pred_cy) / max(lh, rh, 1.0))
        spatial = float(np.exp(-dist_norm))
        app = max(0.0, float(_cos(d, last_desc)))
        age_pen = 1.0 - 0.35 * min(1.0, float(age) / max(1.0, float(temporal_max_age)))
        return float((0.72 * spatial + 0.28 * app) * age_pen)

    changed_rows = 0
    all_frames = sorted(int(k) for k in rows_by_frame.keys())
    for fr in all_frames:
        fr_rows = [r for r in rows_by_frame.get(int(fr), []) if int(r.idx) in desc_by_idx]
        if not fr_rows:
            continue

        n = len(fr_rows)
        m = len(active_target_ids)
        score = np.full((n, m), -1.0, dtype=np.float32)
        row_old_col: List[int] = []

        for i, r in enumerate(fr_rows):
            idx = int(r.idx)
            d = desc_by_idx[int(idx)]
            old_gid = int(id_by_idx.get(int(idx), int(r.gid)))
            old_col = int(gid_to_col.get(int(old_gid), -1))
            row_old_col.append(old_col)
            cx, cy = r.center
            cxx = float(cx) / frame_w
            cyy = float(cy) / frame_h
            for j, gid in enumerate(active_target_ids):
                sim_proto = float(_cos(d, proto_dyn[int(gid)]))
                temporal = _temporal_score(int(gid), r, d, int(fr))
                inertia = 0.06 if int(gid) == int(old_gid) else 0.0
                lock = 0.0
                if int(fr) <= int(gid_lock_until.get(int(gid), -1)) and temporal > 0.22:
                    lock = float(lock_bonus)
                spatial_pref = 0.0
                sxy = seed_center_mean.get(int(gid), None)
                if sxy is not None:
                    sx, sy = sxy
                    dist = float(np.hypot(cxx - sx, cyy - sy))
                    spatial_pref = float(np.exp(-dist / 0.45))
                score[i, j] = float(
                    0.66 * sim_proto
                    + float(temporal_weight) * temporal
                    + 0.06 * spatial_pref
                    + inertia
                    + lock
                )

        if linear_sum_assignment is not None:
            rr, cc = linear_sum_assignment(-score)
            assigned_pairs = [(int(r), int(c)) for r, c in zip(rr, cc)]
        else:
            assigned_pairs = []
            used_rows: set[int] = set()
            used_cols: set[int] = set()
            cands = []
            for i in range(n):
                for j in range(m):
                    cands.append((float(score[i, j]), int(i), int(j)))
            cands.sort(reverse=True, key=lambda x: x[0])
            for sc, i, j in cands:
                if i in used_rows or j in used_cols:
                    continue
                used_rows.add(i)
                used_cols.add(j)
                assigned_pairs.append((int(i), int(j)))
                if len(used_rows) >= n:
                    break

        assigned_col_by_row: Dict[int, int] = {int(i): int(j) for i, j in assigned_pairs}
        for i, r in enumerate(fr_rows):
            idx = int(r.idx)
            old_gid = int(id_by_idx.get(int(idx), int(r.gid)))
            old_col = int(row_old_col[i])
            locked_gid = int(seed_locked_idx_to_target.get(int(idx), 0))
            if locked_gid > 0 and locked_gid in gid_to_col:
                j_locked = int(gid_to_col[int(locked_gid)])
                new_gid = int(locked_gid)
                j = int(j_locked)
                sc = float(score[i, j_locked])
            else:
                j = int(assigned_col_by_row.get(int(i), old_col if old_col >= 0 else 0))
                j = max(0, min(int(m - 1), int(j)))
                new_gid = int(active_target_ids[int(j)])
                sc = float(score[i, j])
                if sc < float(min_assign_score) and old_col >= 0:
                    new_gid = int(active_target_ids[int(old_col)])
                    j = int(old_col)

            if new_gid != old_gid:
                changed_rows += 1
            id_by_idx[int(idx)] = int(new_gid)

            d = desc_by_idx.get(int(idx), None)
            if d is None:
                continue

            j_new = int(gid_to_col.get(int(new_gid), -1))
            if j_new >= 0:
                sc_new = float(score[i, j_new])
                row_scores = score[i, :]
                alt = [float(v) for k, v in enumerate(row_scores.tolist()) if int(k) != int(j_new)]
                second = max(alt) if alt else -1.0
                margin = float(sc_new - second)
                if sc_new >= max(float(min_assign_score) + 0.05, 0.56) and margin >= 0.02:
                    proto_dyn[int(new_gid)] = _l2(0.96 * proto_dyn[int(new_gid)] + 0.04 * d)

            rec = mem_last.get(int(new_gid), None)
            if rec is not None and int(rec[0]) < int(fr):
                mem_prev[int(new_gid)] = (int(rec[0]), tuple(rec[1]))
                last_desc = rec[2]
                merged_desc = _l2(0.84 * last_desc + 0.16 * d)
            else:
                merged_desc = _l2(d)
            mem_last[int(new_gid)] = (
                int(fr),
                (float(r.x1), float(r.y1), float(r.x2), float(r.y2)),
                merged_desc,
            )

            row_scores = score[i, :]
            alt = [float(v) for k, v in enumerate(row_scores.tolist()) if int(k) != int(j)]
            second = max(alt) if alt else -1.0
            margin = float(sc - second)
            if sc >= max(float(min_assign_score) + 0.07, 0.56) and margin >= 0.015:
                gid_lock_until[int(new_gid)] = max(
                    int(gid_lock_until.get(int(new_gid), -1)),
                    int(fr) + int(lock_hold_frames),
                )

    if changed_rows <= 0:
        return {
            "applied": False,
            "reason": "no_changes",
            "target_ids": [int(x) for x in target_ids],
            "seed_counts": {str(k): int(len(v)) for k, v in seed_descs.items()},
            "used_osnet": bool(extractor is not None),
            "osnet_error": str(extractor_error) if extractor is None and extractor_error else "",
            "changed_rows": 0,
        }

    non_target_zeroed = 0
    if strict_target_only and not unresolved:
        for idx, gid in list(id_by_idx.items()):
            g = int(gid)
            if g > 0 and g not in target_set:
                # Conservative identity policy:
                # do not keep unknown positive IDs when canonical target IDs are fully available.
                id_by_idx[int(idx)] = 0
                non_target_zeroed += 1

    with tracks_csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        out_rows = []
        for idx, raw in enumerate(reader):
            gid = int(id_by_idx.get(int(idx), int(raw.get("global_id", 0) or 0)))
            raw["global_id"] = str(gid)
            out_rows.append(raw)

    with tracks_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    dedup_rows = _enforce_unique_positive_ids_per_frame(tracks_csv_path=tracks_csv_path)
    present_targets = sorted(
        int(g) for g in target_set
        if any(int(v) == int(g) for v in id_by_idx.values())
    )
    missing_targets = sorted(int(g) for g in target_set if int(g) not in set(present_targets))
    return {
        "applied": True,
        "changed_rows": int(changed_rows),
        "dedup_rows": int(dedup_rows),
        "target_ids": [int(x) for x in target_ids],
        "active_target_ids": [int(x) for x in active_target_ids],
        "present_target_ids": [int(x) for x in present_targets],
        "missing_target_ids": [int(x) for x in missing_targets],
        "unresolved_seed_targets": [int(x) for x in unresolved],
        "seed_counts": {str(k): int(len(v)) for k, v in seed_descs.items()},
        "non_target_zeroed": int(non_target_zeroed),
        "used_osnet": bool(extractor is not None),
        "osnet_error": str(extractor_error) if extractor is None and extractor_error else "",
    }


def suppress_tiny_ids_keep_labels(
    *,
    tracks_csv_path: Path,
    min_rows_keep: int = 8,
    min_span_keep: int = 24,
    keep_ids: Optional[set[int]] = None,
) -> dict:
    with tracks_csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        raw_rows = list(reader)

    if not raw_rows:
        return {"applied": False, "reason": "empty_csv", "removed_ids": 0, "changed_rows": 0}

    keep: set[int] = set(int(x) for x in (keep_ids or set()))
    by_gid_rows: Dict[int, List[int]] = defaultdict(list)
    by_gid_frames: Dict[int, List[int]] = defaultdict(list)

    for i, raw in enumerate(raw_rows):
        try:
            gid = int(raw.get("global_id", 0) or 0)
            fr = int(raw.get("frame_idx", 0) or 0)
        except Exception:
            continue
        if gid <= 0:
            continue
        by_gid_rows[int(gid)].append(i)
        by_gid_frames[int(gid)].append(fr)

    remove_ids: set[int] = set()
    for gid, idxs in by_gid_rows.items():
        if gid in keep:
            continue
        frames = by_gid_frames.get(int(gid), [])
        if not frames:
            continue
        span = int(max(frames) - min(frames) + 1)
        # Always remove very short/noisy IDs, even if they are spread out.
        # Also remove short IDs that only exist in a tiny span.
        if len(idxs) < int(min_rows_keep) or (
            len(idxs) < int(2 * min_rows_keep) and span < int(min_span_keep)
        ):
            remove_ids.add(int(gid))

    if not remove_ids:
        return {"applied": False, "reason": "no_tiny_ids", "removed_ids": 0, "changed_rows": 0}

    changed_rows = 0
    for raw in raw_rows:
        try:
            gid = int(raw.get("global_id", 0) or 0)
        except Exception:
            continue
        if gid in remove_ids:
            raw["global_id"] = "0"
            changed_rows += 1

    with tracks_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(raw_rows)

    dedup_rows = _enforce_unique_positive_ids_per_frame(tracks_csv_path=tracks_csv_path)
    return {
        "applied": True,
        "removed_ids": int(len(remove_ids)),
        "changed_rows": int(changed_rows),
        "dedup_rows": int(dedup_rows),
        "removed_gid_list": sorted(int(g) for g in remove_ids),
    }


def suppress_non_person_ghost_boxes(
    *,
    video_path: Path,
    tracks_csv_path: Path,
    min_aspect_ratio: float = 0.18,
    max_aspect_ratio: float = 1.08,
    min_area_ratio: float = 0.0032,
    min_width_ratio: float = 0.020,
    min_height_ratio: float = 0.060,
    right_zone_min_x: float = 0.74,
    right_zone_min_y: float = 0.32,
    right_zone_max_aspect: float = 0.35,
    right_zone_max_width_ratio: float = 0.095,
    remove_gid_min_rows: int = 10,
    remove_gid_min_suspicious_ratio: float = 0.62,
    remove_gid_max_center_std: float = 0.060,
) -> dict:
    """
    Suppress non-human / ghost boxes that slip through detector+tracker.
    Targets invalid human shapes and right-side toilet/shelf ghost strips.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"applied": False, "reason": "video_open_failed", "changed_rows": 0}
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if w <= 0 or h <= 0:
        return {"applied": False, "reason": "invalid_video_shape", "changed_rows": 0}

    with tracks_csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        raw_rows = list(reader)
    if not raw_rows:
        return {"applied": False, "reason": "empty_csv", "changed_rows": 0}

    suspicious_row_idx: set[int] = set()
    by_gid_rows: Dict[int, List[int]] = defaultdict(list)
    by_gid_suspicious: Dict[int, int] = defaultdict(int)
    by_gid_cx: Dict[int, List[float]] = defaultdict(list)
    by_gid_cy: Dict[int, List[float]] = defaultdict(list)
    by_gid_right_hits: Dict[int, int] = defaultdict(int)
    row_reason_counts: Dict[str, int] = defaultdict(int)

    fw = float(max(1, w))
    fh = float(max(1, h))
    for idx, raw in enumerate(raw_rows):
        try:
            gid = int(raw.get("global_id", 0) or 0)
            x1 = float(raw.get("x1", 0.0) or 0.0)
            y1 = float(raw.get("y1", 0.0) or 0.0)
            x2 = float(raw.get("x2", 0.0) or 0.0)
            y2 = float(raw.get("y2", 0.0) or 0.0)
        except Exception:
            continue
        if gid <= 0:
            continue

        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        aspect = float(bw / bh)
        area_ratio = float((bw * bh) / max(1.0, fw * fh))
        bw_ratio = float(bw / fw)
        bh_ratio = float(bh / fh)
        cx = float((x1 + x2) * 0.5 / fw)
        cy = float((y1 + y2) * 0.5 / fh)
        right_zone = bool(cx >= float(right_zone_min_x) and cy >= float(right_zone_min_y))

        by_gid_rows[int(gid)].append(int(idx))
        by_gid_cx[int(gid)].append(float(cx))
        by_gid_cy[int(gid)].append(float(cy))
        if right_zone:
            by_gid_right_hits[int(gid)] += 1

        suspicious = False
        if aspect < float(min_aspect_ratio) or aspect > float(max_aspect_ratio):
            suspicious = True
            row_reason_counts["invalid_aspect"] += 1
        if area_ratio < float(min_area_ratio) or bw_ratio < float(min_width_ratio) or bh_ratio < float(min_height_ratio):
            suspicious = True
            row_reason_counts["small_partial_box"] += 1
        if right_zone and aspect <= float(right_zone_max_aspect) and bw_ratio <= float(right_zone_max_width_ratio):
            suspicious = True
            row_reason_counts["right_toilet_shelf_ghost"] += 1

        if suspicious:
            suspicious_row_idx.add(int(idx))
            by_gid_suspicious[int(gid)] += 1

    remove_gids: set[int] = set()
    for gid, idxs in by_gid_rows.items():
        n = int(len(idxs))
        if n < int(remove_gid_min_rows):
            continue
        susp = int(by_gid_suspicious.get(int(gid), 0))
        susp_ratio = float(susp / float(max(1, n)))
        right_ratio = float(by_gid_right_hits.get(int(gid), 0) / float(max(1, n)))
        xs = np.array(by_gid_cx.get(int(gid), []), dtype=np.float32)
        ys = np.array(by_gid_cy.get(int(gid), []), dtype=np.float32)
        center_std = float(np.hypot(xs.std(), ys.std())) if xs.size > 0 and ys.size > 0 else 1.0
        if (
            susp_ratio >= float(remove_gid_min_suspicious_ratio)
            and right_ratio >= 0.45
            and center_std <= float(remove_gid_max_center_std)
        ):
            remove_gids.add(int(gid))

    if not suspicious_row_idx and not remove_gids:
        return {"applied": False, "reason": "no_non_person_ghosts", "changed_rows": 0}

    changed_rows = 0
    removed_by_gid = 0
    removed_by_row = 0
    for idx, raw in enumerate(raw_rows):
        try:
            gid = int(raw.get("global_id", 0) or 0)
        except Exception:
            continue
        if gid <= 0:
            continue
        if int(gid) in remove_gids:
            raw["global_id"] = "0"
            changed_rows += 1
            removed_by_gid += 1
            continue
        if int(idx) in suspicious_row_idx:
            raw["global_id"] = "0"
            changed_rows += 1
            removed_by_row += 1

    with tracks_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(raw_rows)

    dedup_rows = _enforce_unique_positive_ids_per_frame(tracks_csv_path=tracks_csv_path)
    return {
        "applied": True,
        "changed_rows": int(changed_rows),
        "removed_rows_by_gid": int(removed_by_gid),
        "removed_rows_by_row_rule": int(removed_by_row),
        "removed_gids": sorted(int(g) for g in remove_gids),
        "row_reason_counts": {str(k): int(v) for k, v in sorted(row_reason_counts.items())},
        "dedup_rows": int(dedup_rows),
    }


def suppress_border_ghost_runs(
    *,
    video_path: Path,
    tracks_csv_path: Path,
    border_margin: float = 0.010,
    max_area_ratio: float = 0.020,
    max_width_ratio: float = 0.16,
    max_height_ratio: float = 0.38,
    max_run_len: int = 10,
) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"applied": False, "reason": "video_open_failed", "changed_rows": 0}
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if w <= 0 or h <= 0:
        return {"applied": False, "reason": "invalid_video_shape", "changed_rows": 0}

    with tracks_csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        raw_rows = list(reader)

    if not raw_rows:
        return {"applied": False, "reason": "empty_csv", "changed_rows": 0}

    mx = max(2, int(float(border_margin) * float(w)))
    my = max(2, int(float(border_margin) * float(h)))

    by_gid: Dict[int, List[Tuple[int, int]]] = defaultdict(list)  # gid -> [(frame, row_index)]
    cand_row: Dict[int, bool] = {}
    hard_strip_rows: set[int] = set()

    for i, raw in enumerate(raw_rows):
        try:
            gid = int(raw.get("global_id", 0) or 0)
            fr = int(raw.get("frame_idx", 0) or 0)
            x1 = float(raw.get("x1", 0.0) or 0.0)
            y1 = float(raw.get("y1", 0.0) or 0.0)
            x2 = float(raw.get("x2", 0.0) or 0.0)
            y2 = float(raw.get("y2", 0.0) or 0.0)
        except Exception:
            continue
        if gid <= 0:
            continue
        by_gid[int(gid)].append((int(fr), int(i)))

        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        wr = float(bw / float(w))
        hr = float(bh / float(h))
        ar = float((bw * bh) / float(max(1, w * h)))
        cx = float((x1 + x2) * 0.5 / float(w))
        cy = float((y1 + y2) * 0.5 / float(h))
        on_border = (
            x1 <= mx or y1 <= my or x2 >= (w - mx) or y2 >= (h - my)
        )
        near_edge_center = (
            cx <= 0.030 or cx >= 0.970 or cy <= 0.020 or cy >= 0.980
        )
        # Persistent outside-camera ghosts are usually very narrow strips hugging
        # a frame edge; remove these rows immediately (not only by short-run logic).
        is_hard_strip = bool(
            on_border
            and near_edge_center
            and ar <= float(max_area_ratio) * 1.35
            and wr <= min(0.075, float(max_width_ratio) * 0.55)
            and (bw / max(1.0, bh)) <= 0.42
        )
        if is_hard_strip:
            hard_strip_rows.add(int(i))
        cand = bool(
            on_border
            and ar <= float(max_area_ratio)
            and (wr <= float(max_width_ratio) or hr <= float(max_height_ratio))
        )
        cand_row[int(i)] = cand

    to_zero: set[int] = set()
    removed_runs = 0
    for gid, fr_idx in by_gid.items():
        fr_idx.sort(key=lambda t: t[0])
        run: List[int] = []
        prev_f: Optional[int] = None
        for fr, idx in fr_idx:
            if not bool(cand_row.get(int(idx), False)):
                if run and len(run) <= int(max_run_len):
                    to_zero.update(int(x) for x in run)
                    removed_runs += 1
                run = []
                prev_f = int(fr)
                continue
            if run and prev_f is not None and int(fr) - int(prev_f) > 2:
                if len(run) <= int(max_run_len):
                    to_zero.update(int(x) for x in run)
                    removed_runs += 1
                run = []
            run.append(int(idx))
            prev_f = int(fr)
        if run and len(run) <= int(max_run_len):
            to_zero.update(int(x) for x in run)
            removed_runs += 1

    to_zero.update(int(x) for x in hard_strip_rows)

    if not to_zero:
        return {"applied": False, "reason": "no_border_ghost_runs", "changed_rows": 0}

    changed_rows = 0
    for idx in sorted(to_zero):
        try:
            if int(raw_rows[idx].get("global_id", 0) or 0) > 0:
                raw_rows[idx]["global_id"] = "0"
                changed_rows += 1
        except Exception:
            continue

    with tracks_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(raw_rows)

    dedup_rows = _enforce_unique_positive_ids_per_frame(tracks_csv_path=tracks_csv_path)
    return {
        "applied": True,
        "changed_rows": int(changed_rows),
        "removed_runs": int(removed_runs),
        "hard_strip_rows": int(len(hard_strip_rows)),
        "dedup_rows": int(dedup_rows),
    }


def suppress_static_edge_ghost_ids(
    *,
    video_path: Path,
    tracks_csv_path: Path,
    min_rows: int = 120,
    max_center_std_norm: float = 0.010,
    min_border_hit_ratio: float = 0.85,
    max_width_ratio: float = 0.10,
    max_height_ratio: float = 0.45,
) -> dict:
    """
    Remove persistent edge-anchored static ghosts.
    This targets long-lived false boxes stuck near frame borders with near-zero motion.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"applied": False, "reason": "video_open_failed", "changed_rows": 0}
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if w <= 0 or h <= 0:
        return {"applied": False, "reason": "invalid_video_shape", "changed_rows": 0}

    with tracks_csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        raw_rows = list(reader)
    if not raw_rows:
        return {"applied": False, "reason": "empty_csv", "changed_rows": 0}

    by_gid_idx: Dict[int, List[int]] = defaultdict(list)
    cx_by_gid: Dict[int, List[float]] = defaultdict(list)
    cy_by_gid: Dict[int, List[float]] = defaultdict(list)
    bw_by_gid: Dict[int, List[float]] = defaultdict(list)
    bh_by_gid: Dict[int, List[float]] = defaultdict(list)
    border_hits: Dict[int, int] = defaultdict(int)

    border_mx = max(2, int(0.012 * float(w)))
    border_my = max(2, int(0.012 * float(h)))

    for i, raw in enumerate(raw_rows):
        try:
            gid = int(raw.get("global_id", 0) or 0)
            x1 = float(raw.get("x1", 0.0) or 0.0)
            y1 = float(raw.get("y1", 0.0) or 0.0)
            x2 = float(raw.get("x2", 0.0) or 0.0)
            y2 = float(raw.get("y2", 0.0) or 0.0)
        except Exception:
            continue
        if gid <= 0:
            continue
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        by_gid_idx[int(gid)].append(int(i))
        cx_by_gid[int(gid)].append(float(cx / float(w)))
        cy_by_gid[int(gid)].append(float(cy / float(h)))
        bw_by_gid[int(gid)].append(float(bw / float(w)))
        bh_by_gid[int(gid)].append(float(bh / float(h)))
        if x1 <= border_mx or y1 <= border_my or x2 >= (w - border_mx) or y2 >= (h - border_my):
            border_hits[int(gid)] += 1

    remove_ids: set[int] = set()
    for gid, idxs in by_gid_idx.items():
        n = len(idxs)
        if n < int(min_rows):
            continue
        xs = np.array(cx_by_gid.get(int(gid), []), dtype=np.float32)
        ys = np.array(cy_by_gid.get(int(gid), []), dtype=np.float32)
        wr = np.array(bw_by_gid.get(int(gid), []), dtype=np.float32)
        hr = np.array(bh_by_gid.get(int(gid), []), dtype=np.float32)
        if xs.size == 0 or ys.size == 0:
            continue
        std_norm = float(np.hypot(xs.std(), ys.std()))
        border_ratio = float(border_hits.get(int(gid), 0) / float(max(1, n)))
        w_med = float(np.median(wr)) if wr.size > 0 else 1.0
        h_med = float(np.median(hr)) if hr.size > 0 else 1.0

        static_like = std_norm <= float(max_center_std_norm)
        edge_like = border_ratio >= float(min_border_hit_ratio)
        strip_like = (w_med <= float(max_width_ratio)) or (h_med <= float(max_height_ratio))
        if static_like and edge_like and strip_like:
            remove_ids.add(int(gid))

    if not remove_ids:
        return {"applied": False, "reason": "no_static_edge_ghosts", "changed_rows": 0}

    changed_rows = 0
    for raw in raw_rows:
        try:
            gid = int(raw.get("global_id", 0) or 0)
        except Exception:
            continue
        if gid in remove_ids:
            raw["global_id"] = "0"
            changed_rows += 1

    with tracks_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(raw_rows)

    dedup_rows = _enforce_unique_positive_ids_per_frame(tracks_csv_path=tracks_csv_path)
    return {
        "applied": True,
        "changed_rows": int(changed_rows),
        "removed_ids": sorted(int(g) for g in remove_ids),
        "dedup_rows": int(dedup_rows),
    }


def split_ids_on_abrupt_jumps(
    *,
    tracks_csv_path: Path,
    video_path: Optional[Path] = None,
    max_gap_frames: int = 2,
    jump_dist_norm: float = 2.20,
    min_seg_rows: int = 18,
    shape_jump_h_ratio: float = 0.58,
    shape_jump_aspect_ratio: float = 0.55,
    appearance_jump_max_cos: float = 0.60,
    long_gap_split_frames: int = 0,
    long_gap_max_cos: float = 0.52,
) -> dict:
    """
    Split a single ID into new IDs when abrupt short-gap jumps are detected.

    Default behavior keeps the original spatial jump splitter. When ``video_path``
    is provided, we also require a lightweight appearance discontinuity check for
    shape-driven jumps; this helps split mixed-person ID reuse (e.g. adult->boy)
    without over-splitting pose changes of the same person.
    """
    fieldnames, rows = _load_rows(tracks_csv_path)
    if not rows:
        return {"applied": False, "reason": "empty_csv", "split_ids": 0, "changed_rows": 0}

    id_by_idx: Dict[int, int] = {int(r.idx): int(r.gid) for r in rows}
    by_gid_rows: Dict[int, List[TrackRow]] = defaultdict(list)
    for r in rows:
        g = int(id_by_idx.get(int(r.idx), int(r.gid)))
        if g > 0:
            by_gid_rows[int(g)].append(r)

    max_gid = max([int(g) for g in by_gid_rows.keys()] + [0])
    changed_rows = 0
    split_ids = 0
    new_ids_created = 0
    shape_split_events = 0
    appearance_split_events = 0
    long_gap_split_events = 0

    cap: Optional[cv2.VideoCapture] = None
    appearance_enabled = False
    if video_path is not None:
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            appearance_enabled = True
        else:
            cap = None

    desc_cache: Dict[int, Optional[np.ndarray]] = {}

    def _row_desc(row: TrackRow) -> Optional[np.ndarray]:
        if not appearance_enabled or cap is None:
            return None
        ridx = int(row.idx)
        if ridx in desc_cache:
            return desc_cache[ridx]
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(row.frame_idx))
        ok, frame = cap.read()
        if (not ok) or frame is None:
            desc_cache[ridx] = None
            return None
        d = _build_desc(frame, row, extractor=None)
        desc_cache[ridx] = d
        return d

    try:
        for gid, rs in by_gid_rows.items():
            rs = sorted(rs, key=lambda x: x.frame_idx)
            if len(rs) < max(2 * int(min_seg_rows), 24):
                continue

            segments: List[List[TrackRow]] = [[]]
            prev = rs[0]
            segments[-1].append(prev)
            for cur in rs[1:]:
                dt = int(cur.frame_idx) - int(prev.frame_idx)
                split_here = False
                if dt > int(max_gap_frames):
                    # Larger gaps can still be the same person re-entering.
                    # Optional mode: split only when long-gap + strong appearance
                    # discontinuity indicate likely ID reuse.
                    split_here = False
                    if (
                        int(long_gap_split_frames) > 0
                        and dt >= int(long_gap_split_frames)
                        and appearance_enabled
                    ):
                        d_prev = _row_desc(prev)
                        d_cur = _row_desc(cur)
                        if d_prev is not None and d_cur is not None:
                            gap_cos = float(_cos(d_prev, d_cur))
                            if gap_cos <= float(long_gap_max_cos):
                                split_here = True
                                long_gap_split_events += 1
                else:
                    pcx, pcy = prev.center
                    ccx, ccy = cur.center
                    ph = max(1.0, prev.h)
                    ch = max(1.0, cur.h)
                    pw = max(1.0, float(prev.x2 - prev.x1))
                    cw = max(1.0, float(cur.x2 - cur.x1))
                    prev_aspect = float(pw / ph)
                    cur_aspect = float(cw / ch)
                    dn = float(np.hypot(ccx - pcx, ccy - pcy) / max(ph, ch, 1.0))
                    h_ratio = float(min(ph, ch) / max(ph, ch, 1.0))
                    a_ratio = float(min(prev_aspect, cur_aspect) / max(prev_aspect, cur_aspect, 1e-6))

                    spatial_jump = bool(dn >= float(jump_dist_norm))
                    shape_jump = bool(
                        dt <= 1
                        and dn >= 0.35
                        and h_ratio <= float(shape_jump_h_ratio)
                        and a_ratio <= float(shape_jump_aspect_ratio)
                    )
                    if spatial_jump or shape_jump:
                        split_here = bool(spatial_jump)
                        # For shape-only jumps, require appearance disagreement.
                        # For spatial jumps, keep split for very large jumps even
                        # when appearance is unavailable.
                        if appearance_enabled:
                            d_prev = _row_desc(prev)
                            d_cur = _row_desc(cur)
                            if d_prev is not None and d_cur is not None:
                                jump_cos = float(_cos(d_prev, d_cur))
                                if spatial_jump:
                                    split_here = bool(
                                        jump_cos <= float(appearance_jump_max_cos)
                                        or dn >= float(jump_dist_norm + 0.65)
                                    )
                                else:
                                    split_here = bool(jump_cos <= float(appearance_jump_max_cos))
                                    if split_here:
                                        appearance_split_events += 1
                            else:
                                split_here = bool(spatial_jump and dn >= float(jump_dist_norm + 0.25))
                        else:
                            # Keep original behavior when appearance checks are disabled.
                            split_here = bool(spatial_jump)
                        if split_here and (not spatial_jump) and shape_jump:
                            shape_split_events += 1
                if split_here:
                    segments.append([cur])
                else:
                    segments[-1].append(cur)
                prev = cur

            valid = [seg for seg in segments if len(seg) >= int(min_seg_rows)]
            if len(valid) <= 1:
                continue

            split_ids += 1
            # Keep first valid segment on original gid; move later valid segments to new IDs.
            for seg in valid[1:]:
                max_gid += 1
                new_gid = int(max_gid)
                new_ids_created += 1
                for r in seg:
                    idx = int(r.idx)
                    if int(id_by_idx.get(idx, gid)) != int(new_gid):
                        id_by_idx[idx] = int(new_gid)
                        changed_rows += 1
    finally:
        if cap is not None:
            cap.release()

    if changed_rows <= 0:
        return {
            "applied": False,
            "reason": "no_abrupt_jumps",
            "split_ids": 0,
            "changed_rows": 0,
            "appearance_enabled": bool(appearance_enabled),
            "shape_split_events": int(shape_split_events),
            "appearance_split_events": int(appearance_split_events),
            "long_gap_split_events": int(long_gap_split_events),
        }

    with tracks_csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        out_rows = []
        for idx, raw in enumerate(reader):
            g = int(id_by_idx.get(int(idx), int(raw.get("global_id", 0) or 0)))
            raw["global_id"] = str(g)
            out_rows.append(raw)

    with tracks_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    dedup_rows = _enforce_unique_positive_ids_per_frame(tracks_csv_path=tracks_csv_path)
    return {
        "applied": True,
        "split_ids": int(split_ids),
        "new_ids_created": int(new_ids_created),
        "changed_rows": int(changed_rows),
        "dedup_rows": int(dedup_rows),
        "appearance_enabled": bool(appearance_enabled),
        "shape_split_events": int(shape_split_events),
        "appearance_split_events": int(appearance_split_events),
        "long_gap_split_events": int(long_gap_split_events),
    }


def suppress_stationary_tracks(
    tracks_csv_path: Path,
    *,
    cx_range_thresh: float = 60.0,
    cy_range_thresh: float = 60.0,
    percentile_lo: float = 5.0,
    percentile_hi: float = 95.0,
    min_rows: int = 30,
) -> dict:
    """
    Suppress positive-ID tracks that are essentially stationary (objects, not people).

    For each positive global_id, computes the p5–p95 range of the bounding-box
    centre in both x and y.  If both ranges are below the respective thresholds
    the track is suppressed (global_id set to 0).

    Designed for overhead-view uploaded clips where the detector can lock onto
    static product packaging instead of moving people.  Side-view clips trained
    with the retail-domain model should not trigger this guard because even a
    cashier standing still will have >60 px of postural movement over time.

    Returns a dict with keys: changed_ids (list), changed_rows (int).
    """
    import statistics as _stats

    with tracks_csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        raw_rows = list(reader)

    if not raw_rows:
        return {"changed_ids": [], "changed_rows": 0}

    # Collect centre coords per positive gid
    gid_cx: dict[int, list[float]] = {}
    gid_cy: dict[int, list[float]] = {}
    for row in raw_rows:
        gid = int(row.get("global_id", 0) or 0)
        if gid <= 0:
            continue
        try:
            x1, x2 = float(row["x1"]), float(row["x2"])
            y1, y2 = float(row["y1"]), float(row["y2"])
        except (KeyError, ValueError):
            continue
        gid_cx.setdefault(gid, []).append((x1 + x2) / 2.0)
        gid_cy.setdefault(gid, []).append((y1 + y2) / 2.0)

    def _pct_range(vals: list[float]) -> float:
        if len(vals) < 2:
            return 0.0
        sorted_v = sorted(vals)
        n = len(sorted_v)
        lo_i = max(0, int(n * percentile_lo / 100))
        hi_i = min(n - 1, int(n * percentile_hi / 100))
        return sorted_v[hi_i] - sorted_v[lo_i]

    suppressed: set[int] = set()
    for gid, cxs in gid_cx.items():
        if len(cxs) < min_rows:
            continue
        if _pct_range(cxs) < cx_range_thresh and _pct_range(gid_cy[gid]) < cy_range_thresh:
            suppressed.add(gid)

    if not suppressed:
        return {"changed_ids": [], "changed_rows": 0}

    changed = 0
    for row in raw_rows:
        gid = int(row.get("global_id", 0) or 0)
        if gid in suppressed:
            row["global_id"] = "0"
            changed += 1

    with tracks_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(raw_rows)

    return {"changed_ids": sorted(suppressed), "changed_rows": changed}


def render_tracks_video(
    *,
    video_path: Path,
    tracks_csv_path: Path,
    out_video_path: Path,
) -> dict:
    def _analyze_render_frame_quality(frame_bgr: np.ndarray) -> Dict[str, float]:
        """
        Detect visibly corrupted / washed-out frames in the source video.
        These glitches show up as very bright white/pink bands and low-detail blocks.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return {
                "is_bad": 1.0,
                "mean_gray": 255.0,
                "std_gray": 0.0,
                "white_ratio": 1.0,
                "bright_ratio": 1.0,
                "top_white_ratio": 1.0,
                "bottom_white_ratio": 1.0,
                "lap_var": 0.0,
            }

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        mean_gray = float(gray.mean())
        std_gray = float(gray.std())
        white_ratio = float((gray >= 245).mean())
        bright_ratio = float((gray >= 235).mean())
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        hh = int(gray.shape[0])
        top = gray[: max(1, hh // 2), :]
        bot = gray[max(1, hh // 2) :, :]
        top_white_ratio = float((top >= 245).mean()) if top.size > 0 else 0.0
        bottom_white_ratio = float((bot >= 245).mean()) if bot.size > 0 else 0.0

        # Base washed-out rules + extra partial-band rules for CCTV corruption bursts.
        is_bad = (
            white_ratio >= 0.16
            or (mean_gray >= 205.0 and std_gray <= 45.0)
            or (mean_gray >= 225.0)
            or (std_gray <= 8.0)
            or (max(top_white_ratio, bottom_white_ratio) >= 0.55 and min(top_white_ratio, bottom_white_ratio) >= 0.12)
            or (bright_ratio >= 0.62 and lap_var <= 28.0)
        )

        return {
            "is_bad": 1.0 if is_bad else 0.0,
            "mean_gray": mean_gray,
            "std_gray": std_gray,
            "white_ratio": white_ratio,
            "bright_ratio": bright_ratio,
            "top_white_ratio": top_white_ratio,
            "bottom_white_ratio": bottom_white_ratio,
            "lap_var": lap_var,
        }

    rows_by_frame: Dict[int, List[TrackRow]] = defaultdict(list)
    _, rows = _load_rows(tracks_csv_path)
    for r in rows:
        rows_by_frame[r.frame_idx].append(r)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for rendering: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(fps) if fps and fps > 1e-6 else 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    out_video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open writer: {out_video_path}")

    bad_frame_count = 0
    replaced_bad_frames = 0
    bad_without_history = 0
    bad_runs: List[Tuple[int, int]] = []
    current_bad_start: Optional[int] = None
    last_clean_annotated: Optional[np.ndarray] = None

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        quality = _analyze_render_frame_quality(frame)
        frame_is_bad = bool(quality["is_bad"] >= 0.5)
        if frame_is_bad:
            bad_frame_count += 1
            if current_bad_start is None:
                current_bad_start = int(frame_idx)
        elif current_bad_start is not None:
            bad_runs.append((int(current_bad_start), int(frame_idx - 1)))
            current_bad_start = None

        # Recovery strategy:
        # If current frame is corrupted and we have a previous clean rendered frame,
        # reuse that clean frame to remove visible corruption without changing timeline length.
        if frame_is_bad and last_clean_annotated is not None:
            writer.write(last_clean_annotated.copy())
            replaced_bad_frames += 1
            frame_idx += 1
            continue
        if frame_is_bad and last_clean_annotated is None:
            # No prior clean frame yet (very early corruption): keep current frame as fallback.
            bad_without_history += 1

        rs = rows_by_frame.get(frame_idx, [])
        for r in rs:
            x1, y1, x2, y2 = int(r.x1), int(r.y1), int(r.x2), int(r.y2)
            if int(r.gid) <= 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (35, 35, 35), 2, cv2.LINE_AA)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (145, 145, 145), 1, cv2.LINE_AA)
                continue
            gid = int(r.gid)
            # Stable per-ID bright color for inner box, with dark outline for contrast.
            hue = int((gid * 37) % 180)
            sat = 230
            val = 255
            color = cv2.cvtColor(np.uint8([[[hue, sat, val]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
            c = (int(color[0]), int(color[1]), int(color[2]))

            # Double-stroke box: dark thick outer + colored inner.
            cv2.rectangle(frame, (x1, y1), (x2, y2), (12, 12, 12), 4, cv2.LINE_AA)
            cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2, cv2.LINE_AA)

            label = f"ID {gid}"
            fs = 0.72
            th = 2
            (tw, th_txt), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
            pad_x = 7
            pad_y = 5
            lx1 = max(0, x1)
            ly2 = max(th_txt + baseline + 2 * pad_y + 2, y1)
            ly1 = max(0, ly2 - (th_txt + baseline + 2 * pad_y + 2))
            lx2 = min(w - 1, lx1 + tw + 2 * pad_x)

            # Dark label plate + colored border for strong readability.
            cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (12, 12, 12), -1, cv2.LINE_AA)
            cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), c, 2, cv2.LINE_AA)
            tx = lx1 + pad_x
            ty = ly2 - baseline - pad_y

            # White text with dark shadow.
            cv2.putText(frame, label, (tx + 1, ty + 1), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th + 2, cv2.LINE_AA)
            cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th, cv2.LINE_AA)

        writer.write(frame)
        if not frame_is_bad:
            last_clean_annotated = frame.copy()
        frame_idx += 1

    if current_bad_start is not None:
        bad_runs.append((int(current_bad_start), int(frame_idx - 1)))

    cap.release()
    writer.release()
    return {
        "total_frames": int(frame_idx),
        "bad_frames_detected": int(bad_frame_count),
        "bad_frames_replaced": int(replaced_bad_frames),
        "bad_frames_without_history": int(bad_without_history),
        "bad_runs": bad_runs,
    }


def compact_global_ids(
    *,
    tracks_csv_path: Path,
    min_rows_keep: int = 25,
    min_span_keep: int = 26,
) -> dict:
    with tracks_csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        raw_rows = list(reader)

    if not raw_rows:
        return {"total_ids": 0, "max_id": 0}

    # Suppress tiny fragmented IDs before compaction.
    by_gid_rows: Dict[int, List[int]] = defaultdict(list)
    by_gid_frames: Dict[int, List[int]] = defaultdict(list)
    for i, raw in enumerate(raw_rows):
        try:
            gid = int(raw["global_id"])
            fr = int(raw["frame_idx"])
        except Exception:
            continue
        if gid <= 0:
            continue
        by_gid_rows[gid].append(i)
        by_gid_frames[gid].append(fr)

    for gid, idxs in by_gid_rows.items():
        if gid <= 0:
            continue
        frames = by_gid_frames.get(gid, [])
        if not frames:
            continue
        span = int(max(frames) - min(frames) + 1)
        if len(idxs) < int(min_rows_keep) and span < int(min_span_keep):
            for idx in idxs:
                raw_rows[idx]["global_id"] = "0"

    first_seen: Dict[int, int] = {}
    for i, raw in enumerate(raw_rows):
        try:
            gid = int(raw["global_id"])
        except Exception:
            continue
        if gid <= 0:
            continue
        if gid not in first_seen:
            first_seen[gid] = i

    order = sorted(first_seen.keys(), key=lambda g: first_seen[g])
    remap = {gid: (i + 1) for i, gid in enumerate(order)}

    for raw in raw_rows:
        try:
            gid = int(raw["global_id"])
            if gid > 0:
                raw["global_id"] = str(remap.get(gid, gid))
            else:
                raw["global_id"] = "0"
        except Exception:
            continue

    with tracks_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(raw_rows)

    dedup_rows = _enforce_unique_positive_ids_per_frame(tracks_csv_path=tracks_csv_path)
    return {
        "total_ids": len(remap),
        "max_id": max(remap.values()) if remap else 0,
        "dedup_rows": int(dedup_rows),
    }


def canonicalize_first_appearance(
    *,
    tracks_csv_path: Path,
    stable_min_rows: int = 30,
    stable_min_span: int = 32,
    drop_unstable: bool = False,
) -> dict:
    """Enforce hard canonical ID rules:

    - Stable IDs are renumbered 1..N in strict first-appearance order.
      "Stable" here means: rows >= ``stable_min_rows`` AND frame-span >=
      ``stable_min_span``.
    - Unstable (tiny / fragmented) IDs get numbers N+1, N+2, ... (in their
      own first-appearance order) unless ``drop_unstable`` is True, in which
      case they are zeroed out.
    - IDs <= 0 are left untouched.

    This guarantees:
      * "first real person gets ID 1"
      * "second real person gets ID 2", etc.
      * no skipped IDs in the stable 1..N range
      * stable IDs stay unique per-frame (uniqueness enforce runs after).
    """
    with tracks_csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        raw_rows = list(reader)

    if not raw_rows:
        return {"applied": False, "reason": "empty_csv", "total_ids": 0, "max_id": 0}

    # Pass 1: aggregate row counts, frame spans, and first-appearance frame per gid.
    gid_rows: Dict[int, int] = defaultdict(int)
    gid_first_frame: Dict[int, int] = {}
    gid_frame_lo: Dict[int, int] = {}
    gid_frame_hi: Dict[int, int] = {}
    for raw in raw_rows:
        try:
            gid = int(raw.get("global_id", 0) or 0)
            fr = int(raw.get("frame_idx", 0) or 0)
        except Exception:
            continue
        if gid <= 0:
            continue
        gid_rows[gid] += 1
        if gid not in gid_first_frame:
            gid_first_frame[gid] = fr
        else:
            gid_first_frame[gid] = min(gid_first_frame[gid], fr)
        if gid not in gid_frame_lo:
            gid_frame_lo[gid] = fr
            gid_frame_hi[gid] = fr
        else:
            gid_frame_lo[gid] = min(gid_frame_lo[gid], fr)
            gid_frame_hi[gid] = max(gid_frame_hi[gid], fr)

    if not gid_rows:
        return {"applied": False, "reason": "no_positive_ids", "total_ids": 0, "max_id": 0}

    stable_gids: List[int] = []
    unstable_gids: List[int] = []
    for gid, rows_n in gid_rows.items():
        span = int(gid_frame_hi[gid] - gid_frame_lo[gid] + 1)
        if rows_n >= int(stable_min_rows) and span >= int(stable_min_span):
            stable_gids.append(gid)
        else:
            unstable_gids.append(gid)

    stable_sorted = sorted(stable_gids, key=lambda g: (int(gid_first_frame[g]), int(g)))
    unstable_sorted = sorted(unstable_gids, key=lambda g: (int(gid_first_frame[g]), int(g)))

    remap: Dict[int, int] = {}
    next_id = 1
    for gid in stable_sorted:
        remap[int(gid)] = int(next_id)
        next_id += 1
    stable_max = next_id - 1
    for gid in unstable_sorted:
        if drop_unstable:
            remap[int(gid)] = 0
        else:
            remap[int(gid)] = int(next_id)
            next_id += 1

    for raw in raw_rows:
        try:
            gid = int(raw.get("global_id", 0) or 0)
        except Exception:
            continue
        if gid <= 0:
            continue
        raw["global_id"] = str(remap.get(gid, gid))

    with tracks_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(raw_rows)

    dedup_rows = _enforce_unique_positive_ids_per_frame(tracks_csv_path=tracks_csv_path)

    return {
        "applied": True,
        "stable_count": int(len(stable_sorted)),
        "unstable_count": int(len(unstable_sorted)),
        "stable_max_id": int(stable_max),
        "max_id": int(next_id - 1),
        "first_appearance_order": [int(remap[int(g)]) for g in stable_sorted],
        "dedup_rows": int(dedup_rows),
        "drop_unstable": bool(drop_unstable),
    }


def identity_metrics(
    *,
    tracks_csv_path: Path,
    canonical_ids: Sequence[int] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
    stable_min_rows: int = 30,
    stable_min_span: int = 32,
) -> dict:
    """Quick, dependency-free identity KPI summary for a tracks CSV.

    Returns counts needed for before/after comparison:
      * total_rows, rows_with_positive_gid
      * unique_gids, unique_gids_in_canonical
      * stable_count, stable_canonical_count
      * same_frame_duplicate_rows (>0 means uniqueness violated)
      * rows_on_canonical, rows_off_canonical
      * canonical_coverage (rows_on_canonical / rows_with_positive_gid)
      * per_canonical_rows (each of the 11 canonical IDs)
      * per_gid_rows / per_gid_first / per_gid_last / per_gid_span
    """
    canonical_set = set(int(c) for c in canonical_ids)
    with Path(tracks_csv_path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    total_rows = len(rows)
    rows_with_pos = 0
    per_gid_rows: Dict[int, int] = defaultdict(int)
    per_gid_lo: Dict[int, int] = {}
    per_gid_hi: Dict[int, int] = {}
    per_gid_first: Dict[int, int] = {}
    per_frame_gid_count: Dict[Tuple[int, int], int] = defaultdict(int)

    for raw in rows:
        try:
            gid = int(raw.get("global_id", 0) or 0)
            fr = int(raw.get("frame_idx", 0) or 0)
        except Exception:
            continue
        if gid > 0:
            rows_with_pos += 1
            per_gid_rows[gid] += 1
            if gid not in per_gid_lo:
                per_gid_lo[gid] = fr
                per_gid_hi[gid] = fr
                per_gid_first[gid] = fr
            else:
                per_gid_lo[gid] = min(per_gid_lo[gid], fr)
                per_gid_hi[gid] = max(per_gid_hi[gid], fr)
                per_gid_first[gid] = min(per_gid_first[gid], fr)
            per_frame_gid_count[(fr, gid)] += 1

    same_frame_dup_rows = sum(c - 1 for c in per_frame_gid_count.values() if c > 1)
    unique_gids = sorted(per_gid_rows.keys())
    unique_gids_in_canonical = [g for g in unique_gids if g in canonical_set]
    stable = [
        g for g in unique_gids
        if per_gid_rows[g] >= int(stable_min_rows)
        and (per_gid_hi[g] - per_gid_lo[g] + 1) >= int(stable_min_span)
    ]
    stable_canonical = [g for g in stable if g in canonical_set]
    rows_on_canonical = sum(per_gid_rows[g] for g in unique_gids_in_canonical)
    rows_off_canonical = rows_with_pos - rows_on_canonical
    per_canonical_rows = {int(c): int(per_gid_rows.get(c, 0)) for c in sorted(canonical_set)}
    per_gid_detail = {
        int(g): {
            "rows": int(per_gid_rows[g]),
            "first": int(per_gid_first[g]),
            "last": int(per_gid_hi[g]),
            "span": int(per_gid_hi[g] - per_gid_lo[g] + 1),
        }
        for g in unique_gids
    }
    canonical_coverage = (
        float(rows_on_canonical) / float(rows_with_pos) if rows_with_pos else 0.0
    )
    return {
        "total_rows": int(total_rows),
        "rows_with_positive_gid": int(rows_with_pos),
        "unique_gids": [int(g) for g in unique_gids],
        "unique_gids_in_canonical": [int(g) for g in unique_gids_in_canonical],
        "stable_count": int(len(stable)),
        "stable_canonical_count": int(len(stable_canonical)),
        "same_frame_duplicate_rows": int(same_frame_dup_rows),
        "rows_on_canonical": int(rows_on_canonical),
        "rows_off_canonical": int(rows_off_canonical),
        "canonical_coverage": float(canonical_coverage),
        "per_canonical_rows": per_canonical_rows,
        "per_gid_detail": per_gid_detail,
    }


def converge_to_canonical_set(
    *,
    tracks_csv_path: Path,
    canonical_ids: Sequence[int] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
    stable_min_rows: int = 30,
    stable_min_span: int = 32,
    prune_fragments_below_rows: int = 0,
    prune_fragments_below_span: int = 0,
) -> dict:
    """Final convergence pass toward the canonical 11-person set.

    Called AFTER canonicalize_first_appearance and the CAM1 profile anchor.
    What it does:
      1. Compute ``identity_metrics(before)``.
      2. Optionally PRUNE noisy fragment gids (gid not in canonical_ids AND
         rows < ``prune_fragments_below_rows`` AND span <
         ``prune_fragments_below_span``). Zeroing pushes those rows to the
         unclassified pool — safer than letting them reuse a canonical ID.
         Set both thresholds to 0 (default) to disable pruning.
      3. Re-enforce same-frame uniqueness.
      4. Compute ``identity_metrics(after)`` and return the diff.

    The user's hard rule "wrong reuse is worse than fragmentation" means this
    function NEVER remaps fragments onto canonical IDs. It only zeroes the
    very smallest fragments (the ones that are statistical noise).
    """
    before = identity_metrics(
        tracks_csv_path=tracks_csv_path,
        canonical_ids=canonical_ids,
        stable_min_rows=stable_min_rows,
        stable_min_span=stable_min_span,
    )

    pruned_gids: List[int] = []
    if prune_fragments_below_rows > 0 or prune_fragments_below_span > 0:
        canonical_set = set(int(c) for c in canonical_ids)
        for gid, detail in before["per_gid_detail"].items():
            gid_i = int(gid)
            if gid_i <= 0 or gid_i in canonical_set:
                continue
            if (
                detail["rows"] < int(prune_fragments_below_rows)
                and detail["span"] < int(prune_fragments_below_span)
            ):
                pruned_gids.append(gid_i)
        if pruned_gids:
            pruned_set = set(pruned_gids)
            with Path(tracks_csv_path).open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []
                all_rows = list(reader)
            for raw in all_rows:
                try:
                    gid = int(raw.get("global_id", 0) or 0)
                except Exception:
                    continue
                if gid in pruned_set:
                    raw["global_id"] = "0"
            with Path(tracks_csv_path).open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_rows)

    dedup_rows = _enforce_unique_positive_ids_per_frame(tracks_csv_path=tracks_csv_path)

    after = identity_metrics(
        tracks_csv_path=tracks_csv_path,
        canonical_ids=canonical_ids,
        stable_min_rows=stable_min_rows,
        stable_min_span=stable_min_span,
    )

    delta = {
        "stable_count": int(after["stable_count"] - before["stable_count"]),
        "stable_canonical_count": int(
            after["stable_canonical_count"] - before["stable_canonical_count"]
        ),
        "rows_on_canonical": int(after["rows_on_canonical"] - before["rows_on_canonical"]),
        "rows_off_canonical": int(
            after["rows_off_canonical"] - before["rows_off_canonical"]
        ),
        "same_frame_duplicate_rows": int(
            after["same_frame_duplicate_rows"] - before["same_frame_duplicate_rows"]
        ),
    }
    return {
        "applied": True,
        "pruned_gids": [int(g) for g in pruned_gids],
        "dedup_rows": int(dedup_rows),
        "before": before,
        "after": after,
        "delta": delta,
        "converged_to_11": bool(
            after["stable_canonical_count"] == len(set(int(c) for c in canonical_ids))
            and after["same_frame_duplicate_rows"] == 0
        ),
    }


def enforce_same_frame_uniqueness(*, tracks_csv_path: Path) -> dict:
    """Hard same-frame uniqueness (public wrapper).

    Guarantees that every frame contains at most one row per positive gid.
    On conflict, the largest/most-stable box keeps the gid and the others are
    zeroed out (pushed to the unclassified pool) rather than reassigned. The
    internal helper ``_enforce_unique_positive_ids_per_frame`` is the single
    source of truth for conflict resolution; this function exists so the
    pipeline can call it in a clearly-named way at each stage (after tracker
    output, after overlap smoothing, after every relabel pass, after final
    cleanup).

    Returns a dict with ``{"applied": True, "dedup_rows": int}``.
    """
    n = _enforce_unique_positive_ids_per_frame(tracks_csv_path=Path(tracks_csv_path))
    return {"applied": True, "dedup_rows": int(n)}


def merge_fragment_to_canonical_by_appearance(
    *,
    video_path: Path,
    tracks_csv_path: Path,
    canonical_ids: Sequence[int] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
    reid_weights_path: Optional[str] = None,
    min_cos: float = 0.86,
    min_margin: float = 0.08,
    samples_per_gid: int = 14,
    stable_min_rows: int = 30,
    stable_min_span: int = 32,
    device: str = "cpu",
) -> dict:
    """Conservatively merge non-canonical stable fragments into canonical IDs.

    This is a CV-confirming post-pass intended to recover good fragments that
    the tracker created as fallback IDs (12+) but which, on appearance, clearly
    belong to a canonical person (1..N). Safety rules (in order):

    1. Only consider non-canonical gids that are STABLE (rows >= stable_min_rows
       and span >= stable_min_span). Noise stays untouched.
    2. A fragment may only merge into a canonical gid with which it shares
       ZERO frames. This preserves same-frame uniqueness by construction.
    3. Merge requires strong, unambiguous OSNet agreement:
       - top-1 cosine to the canonical's centroid >= ``min_cos``, AND
       - top-1 - top-2 margin >= ``min_margin``.
       Both conditions must hold. Otherwise the fragment is left alone
       (fragmentation is safer than wrong reuse).

    This function is a no-op if the reid extractor cannot be loaded.
    """
    tracks_csv_path = Path(tracks_csv_path)
    video_path = Path(video_path)
    try:
        import numpy as _np  # noqa: F401
        import cv2 as _cv2  # noqa: F401
    except Exception:
        return {"applied": False, "reason": "cv-imports-missing"}

    try:
        from src.reid.extractor import ReidExtractor
    except Exception as exc:  # pragma: no cover - import guard only
        return {"applied": False, "reason": f"extractor-import-failed:{type(exc).__name__}"}

    metrics = identity_metrics(
        tracks_csv_path=tracks_csv_path,
        canonical_ids=canonical_ids,
        stable_min_rows=stable_min_rows,
        stable_min_span=stable_min_span,
    )
    canonical_set = set(int(c) for c in canonical_ids)
    per_canonical_rows = {
        int(k): int(v) for k, v in (metrics.get("per_canonical_rows", {}) or {}).items()
    }
    missing_canonical_targets = {
        int(c) for c in canonical_set
        if int(per_canonical_rows.get(int(c), 0)) < int(stable_min_rows)
    }
    per_gid = metrics.get("per_gid_detail", {}) or {}
    fragments: List[int] = []
    targets: List[int] = []
    for gid, detail in per_gid.items():
        g = int(gid)
        if g <= 0:
            continue
        if int(detail.get("rows", 0)) < int(stable_min_rows):
            continue
        if int(detail.get("span", 0)) < int(stable_min_span):
            continue
        if g in canonical_set:
            targets.append(g)
        else:
            fragments.append(g)
    if not fragments or not targets:
        return {
            "applied": False,
            "reason": "no-eligible-fragments-or-targets",
            "merges": 0,
            "fragments_considered": len(fragments),
            "targets_considered": len(targets),
            "decisions": [],
            "dedup_rows": 0,
        }

    fieldnames, rows = _load_rows(tracks_csv_path)
    if not rows:
        return {"applied": False, "reason": "empty-csv"}

    frames_by_gid: Dict[int, set] = {}
    rows_by_gid: Dict[int, List[TrackRow]] = {}
    for r in rows:
        g = int(r.gid)
        if g <= 0:
            continue
        frames_by_gid.setdefault(g, set()).add(int(r.frame_idx))
        rows_by_gid.setdefault(g, []).append(r)

    # For each (fragment, target) pair: compatible only if they have no shared
    # frames. Otherwise merging would immediately create same-frame duplicates.
    compat: Dict[int, List[int]] = {}
    for f in fragments:
        f_frames = frames_by_gid.get(f, set())
        compat[f] = [
            t for t in targets
            if not (f_frames & frames_by_gid.get(t, set()))
        ]
    if not any(compat.values()):
        return {"applied": False, "reason": "no-disjoint-frame-candidates"}

    # Sample rows from each gid for feature extraction, spaced across time.
    import cv2 as _cv2
    def _pick_rows(rr: List[TrackRow], k: int) -> List[TrackRow]:
        if not rr:
            return []
        rr_sorted = sorted(rr, key=lambda r: int(r.frame_idx))
        n = len(rr_sorted)
        if n <= k:
            return rr_sorted
        step = max(1, n // k)
        return rr_sorted[::step][:k]

    relevant_gids = set(fragments) | set(targets)
    sample_rows_by_gid: Dict[int, List[TrackRow]] = {
        g: _pick_rows(rows_by_gid.get(g, []), int(samples_per_gid))
        for g in relevant_gids
    }
    wanted_frames: Dict[int, List[Tuple[int, int]]] = {}
    for g, rr in sample_rows_by_gid.items():
        for r in rr:
            wanted_frames.setdefault(int(r.frame_idx), []).append((g, int(r.idx)))

    cap = _cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"applied": False, "reason": "video-open-failed"}
    frame_w = float(cap.get(_cv2.CAP_PROP_FRAME_WIDTH) or 0.0)
    frame_h = float(cap.get(_cv2.CAP_PROP_FRAME_HEIGHT) or 0.0)
    frame_w = frame_w if frame_w > 1.0 else 1920.0
    frame_h = frame_h if frame_h > 1.0 else 1080.0

    try:
        extractor = ReidExtractor(
            model_name="osnet_x1_0",
            device=device,
            model_path=reid_weights_path,
        )
    except Exception as exc:
        cap.release()
        return {"applied": False, "reason": f"reid-init-failed:{type(exc).__name__}"}

    # Import color-feature helpers from the anchor module (shared logic).
    try:
        from src.reid.cam1_reference_anchor import (
            _hue_sat_histogram as _color_hue_hist,
            _dominant_color_descriptor as _color_dom,
            _upper_lower_regions as _color_split,
            _combined_color_similarity as _color_combined_sim,
        )
        _color_available = True
    except Exception:
        _color_available = False

    feats_by_gid: Dict[int, List[_np.ndarray]] = {g: [] for g in relevant_gids}
    shape_by_gid: Dict[int, List[_np.ndarray]] = {g: [] for g in relevant_gids}
    upper_hue_by_gid: Dict[int, List[_np.ndarray]] = {g: [] for g in relevant_gids}
    lower_hue_by_gid: Dict[int, List[_np.ndarray]] = {g: [] for g in relevant_gids}
    upper_dom_by_gid: Dict[int, List[_np.ndarray]] = {g: [] for g in relevant_gids}
    lower_dom_by_gid: Dict[int, List[_np.ndarray]] = {g: [] for g in relevant_gids}
    row_by_idx = {int(r.idx): r for r in rows}
    for g, rr in sample_rows_by_gid.items():
        for r in rr:
            bw = max(1.0, float(r.x2) - float(r.x1))
            bh = max(1.0, float(r.y2) - float(r.y1))
            area = bw * bh
            vec = _np.array(
                [
                    bw / bh,
                    bh / frame_h,
                    _np.sqrt(area) / _np.sqrt(frame_w * frame_h),
                    (bw + bh) / (frame_w + frame_h),
                ],
                dtype=_np.float32,
            )
            nn = float(_np.linalg.norm(vec))
            if nn > 1e-6:
                shape_by_gid.setdefault(g, []).append(vec / nn)
    frame_idx = 0
    last_requested = max(wanted_frames) if wanted_frames else -1
    while frame_idx <= last_requested:
        ok = cap.grab()
        if not ok:
            break
        if frame_idx in wanted_frames:
            ok2, frame = cap.retrieve()
            if ok2 and frame is not None:
                for (g, row_idx) in wanted_frames[frame_idx]:
                    r = row_by_idx.get(int(row_idx))
                    if r is None:
                        continue
                    x1 = max(0, int(r.x1))
                    y1 = max(0, int(r.y1))
                    x2 = max(x1 + 2, int(r.x2))
                    y2 = max(y1 + 2, int(r.y2))
                    crop = frame[y1:y2, x1:x2]
                    if crop is None or crop.size == 0:
                        continue
                    # OSNet embedding.
                    try:
                        ff = extractor([crop[:, :, ::-1]]).astype(_np.float32)
                    except Exception:
                        ff = None
                    if ff is not None:
                        if ff.ndim == 2 and ff.shape[0] == 1:
                            ff = ff[0]
                        nn = float(_np.linalg.norm(ff))
                        if nn > 1e-6:
                            feats_by_gid.setdefault(g, []).append(ff / nn)
                    # Color features (BGR crop).
                    if _color_available:
                        upper_bgr, lower_bgr = _color_split(crop)
                        uh = _color_hue_hist(upper_bgr)
                        lh = _color_hue_hist(lower_bgr)
                        ud = _color_dom(upper_bgr)
                        ld = _color_dom(lower_bgr)
                        if uh is not None:
                            upper_hue_by_gid.setdefault(g, []).append(uh)
                        if lh is not None:
                            lower_hue_by_gid.setdefault(g, []).append(lh)
                        if ud is not None:
                            upper_dom_by_gid.setdefault(g, []).append(ud)
                        if ld is not None:
                            lower_dom_by_gid.setdefault(g, []).append(ld)
        frame_idx += 1
    cap.release()

    centroids: Dict[int, _np.ndarray] = {}
    for g, flist in feats_by_gid.items():
        if not flist:
            continue
        m = _np.mean(_np.stack(flist, axis=0), axis=0)
        n = float(_np.linalg.norm(m))
        if n <= 1e-6:
            continue
        centroids[g] = (m / n).astype(_np.float32)

    def _unit_mean(vs: List[_np.ndarray]) -> Optional[_np.ndarray]:
        if not vs:
            return None
        m = _np.mean(_np.stack(vs, axis=0), axis=0)
        n = float(_np.linalg.norm(m))
        if n <= 1e-6:
            return None
        return (m / n).astype(_np.float32)

    color_upper_hue: Dict[int, _np.ndarray] = {}
    color_lower_hue: Dict[int, _np.ndarray] = {}
    color_upper_dom: Dict[int, _np.ndarray] = {}
    color_lower_dom: Dict[int, _np.ndarray] = {}
    shape_centroid: Dict[int, _np.ndarray] = {}
    for g in relevant_gids:
        shp = _unit_mean(shape_by_gid.get(g, []))
        if shp is not None:
            shape_centroid[g] = shp
    if _color_available:
        for g in relevant_gids:
            uh = _unit_mean(upper_hue_by_gid.get(g, []))
            lh = _unit_mean(lower_hue_by_gid.get(g, []))
            ud = _unit_mean(upper_dom_by_gid.get(g, []))
            ld = _unit_mean(lower_dom_by_gid.get(g, []))
            if uh is not None:
                color_upper_hue[g] = uh
            if lh is not None:
                color_lower_hue[g] = lh
            if ud is not None:
                color_upper_dom[g] = ud
            if ld is not None:
                color_lower_dom[g] = ld

    def _color_sim_pair(a: int, b: int) -> Tuple[float, int]:
        if not _color_available:
            return 0.0, 0
        return _color_combined_sim(
            color_upper_hue.get(a), color_lower_hue.get(a),
            color_upper_dom.get(a), color_lower_dom.get(a),
            color_upper_hue.get(b), color_lower_hue.get(b),
            color_upper_dom.get(b), color_lower_dom.get(b),
        )

    def _shape_sim_pair(a: int, b: int) -> float:
        sa = shape_centroid.get(a)
        sb = shape_centroid.get(b)
        if sa is None or sb is None:
            return 0.0
        return float(_np.clip(_np.dot(sa, sb), 0.0, 1.0))

    merge_map: Dict[int, int] = {}
    scores_log: Dict[int, Dict[int, float]] = {}
    color_log: Dict[int, Dict[int, float]] = {}
    shape_log: Dict[int, Dict[int, float]] = {}
    decisions: List[Dict[str, object]] = []
    claimed_target_best: Dict[int, Tuple[int, float]] = {}
    for f in fragments:
        if f not in centroids:
            decisions.append({
                "fragment_gid": int(f),
                "target_gid": None,
                "cos": 0.0,
                "margin": 0.0,
                "decision": "skip-no-centroid",
            })
            continue
        cf = centroids[f]
        cands = compat.get(f, [])
        sims: List[Tuple[int, float, float, int, float, float]] = []  # (t, osnet, color, n_valid_color, shape, combined)
        for t in cands:
            if t not in centroids:
                continue
            sim = float(_np.dot(cf, centroids[t]))
            csim, nv = _color_sim_pair(f, t)
            shp = _shape_sim_pair(f, t)
            if nv >= 2:
                comb = float(0.66 * sim + 0.24 * csim + 0.10 * shp)
            else:
                comb = float(0.82 * sim + 0.18 * shp)
            sims.append((t, sim, csim, nv, shp, comb))
        scores_log[f] = {t: s for (t, s, _, _, _, _) in sims}
        color_log[f] = {t: c for (t, _, c, _, _, _) in sims}
        shape_log[f] = {t: sh for (t, _, _, _, sh, _) in sims}
        if not sims:
            decisions.append({
                "fragment_gid": int(f),
                "target_gid": None,
                "cos": 0.0,
                "margin": 0.0,
                "decision": "skip-no-candidate",
            })
            continue
        sims.sort(key=lambda tup: tup[5], reverse=True)
        top_gid, top_sim, top_color, top_nv, top_shape, top_comb = sims[0]
        if len(sims) > 1:
            _, second_sim, second_color, second_nv, second_shape, second_comb = sims[1]
        else:
            second_sim, second_color, second_nv, second_shape, second_comb = -1.0, 0.0, 0, 0.0, -1.0
        margin = float(top_sim - second_sim) if second_sim > -1.0 else float(top_sim)
        color_margin = float(top_color - second_color)
        shape_margin = float(top_shape - second_shape)
        comb_margin = float(top_comb - second_comb) if second_comb > -1.0 else float(top_comb)
        secondary_support = float(0.70 * top_color + 0.30 * top_shape)
        secondary_margin = float(0.70 * color_margin + 0.30 * shape_margin)

        target_missing = bool(int(top_gid) in missing_canonical_targets)
        # Slightly relaxed recovery gates for missing canonicals only.
        req_cos = max(0.0, float(min_cos) - (0.02 if target_missing else 0.00)) + (0.00 if target_missing else 0.03)
        req_margin = max(0.0, float(min_margin) - (0.03 if target_missing else 0.00)) + (0.00 if target_missing else 0.04)
        req_comb_margin = 0.03 if target_missing else 0.07
        req_secondary = 0.56 if target_missing else 0.64

        # Prevent over-collapse into already-surviving canonicals:
        # if a missing canonical has nearly-equal evidence, keep the fragment
        # unresolved rather than forcing it into a survivor.
        best_missing: Optional[Tuple[int, float, float, float]] = None
        for t, osn, col, nv, shp, comb in sims:
            if int(t) not in missing_canonical_targets:
                continue
            sec = 0.70 * float(col) + 0.30 * float(shp)
            best_missing = (int(t), float(comb), float(osn), float(sec))
            break
        if (not target_missing) and best_missing is not None:
            miss_gid, miss_comb, miss_sim, miss_sec = best_missing
            if (float(top_comb) - miss_comb) <= 0.035 and miss_sec >= (secondary_support - 0.01):
                decisions.append({
                    "fragment_gid": int(f),
                    "target_gid": int(top_gid),
                    "alt_missing_target": int(miss_gid),
                    "cos": float(top_sim),
                    "margin": margin,
                    "combined": float(top_comb),
                    "combined_margin": float(comb_margin),
                    "decision": "reject-prevent-overcollapse-to-survivor",
                })
                continue

        # Color veto on suspicious picks.
        color_vetoed = False
        if (
            top_nv >= 2
            and top_color < 0.46
            and secondary_margin < -0.06
            and top_sim < (req_cos + 0.05)
        ):
            color_vetoed = True

        # Standard strict gate.
        passed_strict = bool(
            top_sim >= req_cos
            and margin >= req_margin
            and comb_margin >= req_comb_margin
            and secondary_support >= req_secondary
        )
        # Color rescue: accept if OSNet is in range and color clearly agrees.
        # Tightened 2026-04: was margin >= 0.0 (wrong-reuse risk between
        # similar-clothed persons, e.g. two black shirts). We now require a
        # real 0.035 OSNet margin AND a strong color-color margin (top
        # color must clearly lead second-best color by 0.06). The caller
        # side `allow_multi` gate (lines ~6449) still prevents multiple
        # fragments mapping to the same target without overwhelming
        # evidence.
        rescued = False
        if not passed_strict and not color_vetoed:
            if (
                target_missing
                and top_nv >= 3
                and top_color >= 0.63
                and secondary_support >= 0.64
                and secondary_margin >= 0.08
                and top_sim >= (req_cos - 0.03)
                and margin >= 0.035
                and color_margin >= 0.06
                and comb_margin >= 0.02
            ):
                rescued = True

        if color_vetoed:
            decisions.append({
                "fragment_gid": int(f),
                "target_gid": int(top_gid),
                "cos": float(top_sim),
                "margin": margin,
                "color_top1": float(top_color),
                "shape_top1": float(top_shape),
                "combined": float(top_comb),
                "color_margin": float(color_margin),
                "secondary_margin": float(secondary_margin),
                "decision": "color-veto",
            })
            continue

        if passed_strict or rescued:
            # Multiple fragments can map to the same target only with very
            # strong evidence; otherwise preserve purity over coverage.
            prior_claim = claimed_target_best.get(int(top_gid))
            if prior_claim is not None:
                prior_f, prior_comb = prior_claim
                allow_multi = bool(
                    top_sim >= 0.93
                    and margin >= (req_margin + 0.06)
                    and comb_margin >= (req_comb_margin + 0.04)
                    and secondary_support >= max(req_secondary, 0.70)
                )
                if not allow_multi and float(top_comb) <= float(prior_comb):
                    decisions.append({
                        "fragment_gid": int(f),
                        "target_gid": int(top_gid),
                        "cos": float(top_sim),
                        "margin": margin,
                        "combined": float(top_comb),
                        "decision": "skip-target-claimed-by-stronger",
                    })
                    continue
                if not allow_multi and float(top_comb) > float(prior_comb):
                    merge_map.pop(int(prior_f), None)
                    decisions.append({
                        "fragment_gid": int(prior_f),
                        "target_gid": int(top_gid),
                        "superseded_by_fragment": int(f),
                        "decision": "superseded-by-stronger-fragment",
                    })
            merge_map[int(f)] = int(top_gid)
            claimed_target_best[int(top_gid)] = (int(f), float(top_comb))
            decisions.append({
                "fragment_gid": int(f),
                "target_gid": int(top_gid),
                "cos": float(top_sim),
                "margin": margin,
                "color_top1": float(top_color),
                "shape_top1": float(top_shape),
                "combined": float(top_comb),
                "combined_margin": float(comb_margin),
                "color_margin": float(color_margin),
                "secondary_margin": float(secondary_margin),
                "decision": "color-rescue" if (rescued and not passed_strict) else "merge",
            })
        else:
            decisions.append({
                "fragment_gid": int(f),
                "target_gid": int(top_gid),
                "cos": float(top_sim),
                "margin": margin,
                "color_top1": float(top_color),
                "shape_top1": float(top_shape),
                "combined": float(top_comb),
                "combined_margin": float(comb_margin),
                "color_margin": float(color_margin),
                "secondary_margin": float(secondary_margin),
                "decision": "reject-below-threshold",
            })

    if not merge_map:
        return {
            "applied": False,
            "reason": "no-confident-merges",
            "fragments": [int(x) for x in fragments],
            "scores": scores_log,
            "color_scores": color_log,
            "shape_scores": shape_log,
            "missing_canonical_targets": sorted(int(x) for x in missing_canonical_targets),
            "merges": 0,
            "fragments_considered": len(fragments),
            "targets_considered": len(targets),
            "decisions": decisions,
            "dedup_rows": 0,
        }

    # Apply the merge.
    with tracks_csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        all_rows = list(reader)
    for raw in all_rows:
        try:
            g = int(raw.get("global_id", 0) or 0)
        except Exception:
            continue
        if g in merge_map:
            raw["global_id"] = str(int(merge_map[g]))
    with tracks_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    dedup = _enforce_unique_positive_ids_per_frame(tracks_csv_path=tracks_csv_path)
    return {
        "applied": True,
        "merged": {int(k): int(v) for k, v in merge_map.items()},
        "merges": len(merge_map),
        "fragments_considered": len(fragments),
        "targets_considered": len(targets),
        "decisions": decisions,
        "scores": scores_log,
        "color_scores": color_log,
        "shape_scores": shape_log,
        "missing_canonical_targets": sorted(int(x) for x in missing_canonical_targets),
        "dedup_rows": int(dedup),
    }
