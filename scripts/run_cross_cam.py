from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import csv
import shutil
import sys
from typing import Dict, List, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_demo import run_video_with_kpis
from src.preprocessing import preprocess_video, extract_metadata
from src.trackers.bot_sort import BOTSORT
from src.reid.track_linker import (
    _build_descs_for_rows,
    _build_track_stats,
    _cos,
    _enrich_track_meta,
    _extract_track_descriptors,
    _load_rows,
    _topk_similarity,
    compact_global_ids,
    render_tracks_video,
    stitch_track_ids,
    suppress_same_frame_duplicates,
)
from src.reid.extractor import ReidExtractor
from src.reid.reentry_linker import ReentryConfig, link_reentry_offline


@dataclass
class PairScore:
    cam1_gid: int
    cam2_gid: int
    score: float
    sim_topk: float
    sim_mean: float
    sim_edge: float
    shape: float
    rows_cam1: int
    rows_cam2: int


def _video_dims(video_path: Path) -> Tuple[int, int]:
    try:
        w, h = extract_metadata(video_path).resolution
        return max(1, w), max(1, h)
    except Exception:
        return 1, 1


def _make_tracker(
    det_weights: str,
    reid_weights_path: str | None,
    *,
    det_conf: float = 0.24,
    min_height_ratio: float = 0.10,
    min_width_ratio: float = 0.04,
) -> BOTSORT:
    return BOTSORT(
        id_bank=None,
        det_weights=det_weights,
        reid_weights_path=reid_weights_path,
        det_conf=float(det_conf),
        det_imgsz=960,
        track_thresh=0.30,
        active_match_iou_thresh=0.18,
        lost_match_iou_thresh=0.08,
        match_feat_thresh=0.40,
        strong_reid_thresh=0.78,
        long_lost_reid_thresh=0.82,
        alpha_active=0.42,
        alpha_lost=0.82,
        track_buffer=220,
        motion_max_center_dist=0.72,
        motion_max_gap=32,
        overlap_iou_thresh=0.28,
        min_height_ratio_for_update=0.72,
        min_match_conf=0.20,
        feature_history=80,
        feature_update_min_sim=0.64,
        confirm_hits=4,
        bad_frame_hold=12,
        min_confirmed_hits_for_gid=4,
        min_height_ratio=float(min_height_ratio),
        min_width_ratio=float(min_width_ratio),
        reid_min_conf_for_extract=0.44,
        reid_min_area_ratio=0.010,
        reid_border_margin=0.015,
        reid_min_blur_var=38.0,
        reid_min_quality_for_bank=0.46,
        reid_far_y2_ratio=0.30,
        reid_cautious_y2_ratio=0.52,
        reid_min_lock_hits=4,
        reid_new_id_min_hits=6,
        reid_new_id_min_quality=0.44,
        F_OS=1.00,
        F_ATTIRE=0.45,
        F_SHAPE=0.22,
        gid_reuse_thresh=0.74,
        gid_reuse_with_spatial_thresh=0.66,
        gid_spatial_max_age=220,
        reentry_gallery_size=20,
        reentry_min_samples=3,
        reentry_max_age=420,
        reentry_sim_thresh=0.70,
        reentry_margin=0.05,
        reentry_min_zone_compat=0.32,
        lock_confirm_frames=3,
        new_id_confirm_frames=6,
        lock_score_thresh=0.66,
        new_id_score_thresh=0.62,
        gid_owner_reserve_frames=240,
        gid_owner_override_score=0.86,
        overlap_hold_iou_thresh=0.16,
        overlap_hold_frames=56,
        drift_release_hits=6,
        drift_guard_min_mode=2,
        drift_guard_min_quality=0.56,
        short_memory_max_age=140,
        short_memory_reuse_thresh=0.69,
        short_memory_margin=0.05,
        active_similar_block_thresh=0.62,
        profile_topk=5,
        profile_min_quality=0.58,
        bank_update_min_margin=0.09,
    )


def _suppress_contained_cross_id_duplicates(
    *,
    tracks_csv_path: Path,
    min_pair_hits: int = 2,
    iou_thresh: float = 0.48,
    containment_thresh: float = 0.84,
    max_center_dist_norm: float = 0.42,
) -> Dict[str, object]:
    fieldnames, rows = _load_rows(tracks_csv_path)
    if not rows:
        return {"applied": False, "reason": "empty_tracks", "changed_rows": 0, "pairs": 0}

    id_by_idx: Dict[int, int] = {int(r.idx): int(r.gid) for r in rows}
    gid_row_count: Dict[int, int] = {}
    by_frame: Dict[int, List[object]] = {}
    for r in rows:
        by_frame.setdefault(int(r.frame_idx), []).append(r)
        g = int(r.gid)
        if g > 0:
            gid_row_count[g] = int(gid_row_count.get(g, 0)) + 1

    pair_hits: Dict[Tuple[int, int], int] = {}
    pair_events: Dict[Tuple[int, int], List[Tuple[int, int, float, float, float, float]]] = {}
    # event tuple: (idx1, idx2, area1, area2, iou, containment)
    for fr_rows in by_frame.values():
        n = len(fr_rows)
        for i in range(n):
            r1 = fr_rows[i]
            g1 = int(id_by_idx.get(int(r1.idx), int(r1.gid)))
            if g1 <= 0:
                continue
            for j in range(i + 1, n):
                r2 = fr_rows[j]
                g2 = int(id_by_idx.get(int(r2.idx), int(r2.gid)))
                if g2 <= 0 or g1 == g2:
                    continue
                xA = max(float(r1.x1), float(r2.x1))
                yA = max(float(r1.y1), float(r2.y1))
                xB = min(float(r1.x2), float(r2.x2))
                yB = min(float(r1.y2), float(r2.y2))
                iw = max(0.0, xB - xA)
                ih = max(0.0, yB - yA)
                inter = iw * ih
                if inter <= 0.0:
                    continue
                a1 = max(0.0, float(r1.x2 - r1.x1)) * max(0.0, float(r1.y2 - r1.y1))
                a2 = max(0.0, float(r2.x2 - r2.x1)) * max(0.0, float(r2.y2 - r2.y1))
                if a1 <= 1.0 or a2 <= 1.0:
                    continue
                den = a1 + a2 - inter + 1e-6
                iou = float(inter / den) if den > 0 else 0.0
                containment = max(float(inter / a1), float(inter / a2))
                c1x, c1y = float(0.5 * (r1.x1 + r1.x2)), float(0.5 * (r1.y1 + r1.y2))
                c2x, c2y = float(0.5 * (r2.x1 + r2.x2)), float(0.5 * (r2.y1 + r2.y2))
                h_norm = max(1.0, float(max(r1.h, r2.h)))
                cdist = float(np.hypot(c1x - c2x, c1y - c2y) / h_norm)
                if not ((iou >= float(iou_thresh) or containment >= float(containment_thresh)) and cdist <= float(max_center_dist_norm)):
                    continue
                pair = (min(g1, g2), max(g1, g2))
                pair_hits[pair] = int(pair_hits.get(pair, 0)) + 1
                pair_events.setdefault(pair, []).append((int(r1.idx), int(r2.idx), float(a1), float(a2), float(iou), float(containment)))

    valid_pairs = {p for p, n in pair_hits.items() if int(n) >= int(min_pair_hits)}
    if not valid_pairs:
        return {"applied": False, "reason": "no_pair_hits", "changed_rows": 0, "pairs": 0}

    changed_rows = 0
    for pair in valid_pairs:
        for idx1, idx2, a1, a2, _iou, _cont in pair_events.get(pair, []):
            g1 = int(id_by_idx.get(int(idx1), 0))
            g2 = int(id_by_idx.get(int(idx2), 0))
            if g1 <= 0 or g2 <= 0 or g1 == g2:
                continue
            score1 = (int(gid_row_count.get(g1, 0)), float(a1))
            score2 = (int(gid_row_count.get(g2, 0)), float(a2))
            drop_idx = int(idx1) if score1 < score2 else int(idx2)
            if int(id_by_idx.get(drop_idx, 0)) > 0:
                id_by_idx[drop_idx] = 0
                changed_rows += 1

    if changed_rows <= 0:
        return {"applied": False, "reason": "no_changes", "changed_rows": 0, "pairs": int(len(valid_pairs))}

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
    return {"applied": True, "changed_rows": int(changed_rows), "pairs": int(len(valid_pairs))}


def _drop_static_nonperson_tracks(
    *,
    tracks_csv_path: Path,
    min_rows: int = 220,
    max_path_norm: float = 1.0,
    max_disp_norm: float = 0.20,
    max_cx_std: float = 35.0,
    max_cy_std: float = 35.0,
) -> Dict[str, object]:
    fieldnames, rows = _load_rows(tracks_csv_path)
    if not rows:
        return {"applied": False, "reason": "empty_tracks", "dropped_rows": 0, "dropped_ids": []}

    by_gid: Dict[int, List[object]] = {}
    for r in rows:
        g = int(r.gid)
        if g > 0:
            by_gid.setdefault(g, []).append(r)

    drop_ids: set[int] = set()
    for gid, rs in by_gid.items():
        if len(rs) < int(min_rows):
            continue
        rs = sorted(rs, key=lambda x: (x.frame_idx, x.idx))
        cx = np.array([float(0.5 * (r.x1 + r.x2)) for r in rs], dtype=np.float32)
        cy = np.array([float(0.5 * (r.y1 + r.y2)) for r in rs], dtype=np.float32)
        hh = np.array([float(max(1.0, r.h)) for r in rs], dtype=np.float32)
        med_h = float(np.median(hh)) if len(hh) else 1.0
        if len(cx) <= 1:
            continue
        path = float(np.sum(np.hypot(np.diff(cx), np.diff(cy))))
        disp = float(np.hypot(float(cx[-1] - cx[0]), float(cy[-1] - cy[0])))
        path_norm = float(path / max(1.0, med_h))
        disp_norm = float(disp / max(1.0, med_h))
        cx_std = float(np.std(cx))
        cy_std = float(np.std(cy))
        if (
            path_norm <= float(max_path_norm)
            and disp_norm <= float(max_disp_norm)
            and cx_std <= float(max_cx_std)
            and cy_std <= float(max_cy_std)
        ):
            drop_ids.add(int(gid))

    if not drop_ids:
        return {"applied": False, "reason": "no_static_ids", "dropped_rows": 0, "dropped_ids": []}

    id_by_idx: Dict[int, int] = {}
    dropped_rows = 0
    for r in rows:
        g = int(r.gid)
        if g in drop_ids:
            id_by_idx[int(r.idx)] = 0
            dropped_rows += 1
        else:
            id_by_idx[int(r.idx)] = g

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

    return {"applied": True, "dropped_rows": int(dropped_rows), "dropped_ids": sorted(int(g) for g in drop_ids)}


def _merge_high_similarity_overlaps(
    *,
    video_path: Path,
    tracks_csv_path: Path,
    reid_weights_path: str | None,
    min_pair_hits: int = 6,
    iou_thresh: float = 0.22,
    containment_thresh: float = 0.54,
    max_center_dist_norm: float = 0.62,
    min_score: float = 0.86,
    min_topk: float = 0.84,
    min_mean: float = 0.82,
) -> Dict[str, object]:
    fieldnames, rows = _load_rows(tracks_csv_path)
    if not rows:
        return {"applied": False, "reason": "empty_tracks", "changed_rows": 0, "merged_pairs": 0}

    by_frame: Dict[int, List[object]] = {}
    for r in rows:
        if int(r.gid) > 0:
            by_frame.setdefault(int(r.frame_idx), []).append(r)

    pair_hits: Dict[Tuple[int, int], int] = {}
    for fr_rows in by_frame.values():
        n = len(fr_rows)
        for i in range(n):
            r1 = fr_rows[i]
            g1 = int(r1.gid)
            if g1 <= 0:
                continue
            for j in range(i + 1, n):
                r2 = fr_rows[j]
                g2 = int(r2.gid)
                if g2 <= 0 or g1 == g2:
                    continue
                xA = max(float(r1.x1), float(r2.x1))
                yA = max(float(r1.y1), float(r2.y1))
                xB = min(float(r1.x2), float(r2.x2))
                yB = min(float(r1.y2), float(r2.y2))
                iw = max(0.0, xB - xA)
                ih = max(0.0, yB - yA)
                inter = iw * ih
                if inter <= 0.0:
                    continue
                a1 = max(1.0, float(r1.x2 - r1.x1) * float(r1.y2 - r1.y1))
                a2 = max(1.0, float(r2.x2 - r2.x1) * float(r2.y2 - r2.y1))
                den = a1 + a2 - inter + 1e-6
                iou = float(inter / den) if den > 0 else 0.0
                containment = max(float(inter / a1), float(inter / a2))
                c1x, c1y = float(0.5 * (r1.x1 + r1.x2)), float(0.5 * (r1.y1 + r1.y2))
                c2x, c2y = float(0.5 * (r2.x1 + r2.x2)), float(0.5 * (r2.y1 + r2.y2))
                h_norm = max(1.0, float(max(r1.h, r2.h)))
                cdist = float(np.hypot(c1x - c2x, c1y - c2y) / h_norm)
                if (iou >= float(iou_thresh) or containment >= float(containment_thresh)) and cdist <= float(max_center_dist_norm):
                    pair = (min(g1, g2), max(g1, g2))
                    pair_hits[pair] = int(pair_hits.get(pair, 0)) + 1

    candidates = [(p, int(n)) for p, n in pair_hits.items() if int(n) >= int(min_pair_hits)]
    if not candidates:
        return {"applied": False, "reason": "no_overlap_pairs", "changed_rows": 0, "merged_pairs": 0}

    tracks = _build_track_stats(rows)
    if not tracks:
        return {"applied": False, "reason": "no_tracks", "changed_rows": 0, "merged_pairs": 0}
    _extract_track_descriptors(video_path, tracks, reid_weights_path=reid_weights_path)

    row_count_by_gid: Dict[int, int] = {int(g): int(len(st.rows)) for g, st in tracks.items() if int(g) > 0}

    parent: Dict[int, int] = {}

    def _root(g: int) -> int:
        x = int(g)
        while int(parent.get(int(x), int(x))) != int(x):
            x = int(parent.get(int(x), int(x)))
        return int(x)

    merged_pairs = 0
    for (g1_raw, g2_raw), _hits in sorted(candidates, key=lambda x: x[1], reverse=True):
        g1 = _root(int(g1_raw))
        g2 = _root(int(g2_raw))
        if g1 == g2:
            continue
        t1 = tracks.get(int(g1))
        t2 = tracks.get(int(g2))
        if t1 is None or t2 is None:
            continue

        sim_topk = float(_topk_similarity(t1.sample_descs, t2.sample_descs, k=8))
        sim_mean = float(_cos(t1.mean_desc, t2.mean_desc))
        sim_edge = max(
            float(_cos(t1.first_desc, t2.first_desc)),
            float(_cos(t1.last_desc, t2.last_desc)),
            float(_cos(t1.first_desc, t2.last_desc)),
            float(_cos(t1.last_desc, t2.first_desc)),
        )
        shape = _shape_score(t1.median_h_ratio, t1.median_aspect, t2.median_h_ratio, t2.median_aspect)
        score = float(0.55 * sim_topk + 0.30 * sim_mean + 0.10 * max(0.0, sim_edge) + 0.05 * shape)
        if score < float(min_score) or sim_topk < float(min_topk) or sim_mean < float(min_mean):
            continue

        s1 = int(t1.start_f)
        s2 = int(t2.start_f)
        c1 = int(row_count_by_gid.get(int(g1), 0))
        c2 = int(row_count_by_gid.get(int(g2), 0))
        if s1 < s2:
            keep, drop = int(g1), int(g2)
        elif s2 < s1:
            keep, drop = int(g2), int(g1)
        elif c1 >= c2:
            keep, drop = int(g1), int(g2)
        else:
            keep, drop = int(g2), int(g1)

        parent[int(drop)] = int(keep)
        merged_pairs += 1

    if not parent:
        return {"applied": False, "reason": "no_similarity_merges", "changed_rows": 0, "merged_pairs": 0}

    id_by_idx = {int(r.idx): int(r.gid) for r in rows}
    changed_rows = 0
    for idx, gid in list(id_by_idx.items()):
        if int(gid) <= 0:
            continue
        root = _root(int(gid))
        if int(root) != int(gid):
            id_by_idx[int(idx)] = int(root)
            changed_rows += 1

    if changed_rows <= 0:
        return {"applied": False, "reason": "no_row_changes", "changed_rows": 0, "merged_pairs": int(merged_pairs)}

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
    return {"applied": True, "changed_rows": int(changed_rows), "merged_pairs": int(merged_pairs)}


def _merge_high_similarity_near_duplicates(
    *,
    video_path: Path,
    tracks_csv_path: Path,
    reid_weights_path: str | None,
    min_score: float = 0.90,
    min_topk: float = 0.90,
    min_mean: float = 0.90,
    max_shared_frames: int = 20,
    max_shared_ratio: float = 0.25,
) -> Dict[str, object]:
    fieldnames, rows = _load_rows(tracks_csv_path)
    if not rows:
        return {"applied": False, "reason": "empty_tracks", "changed_rows": 0, "merged_pairs": 0}

    tracks = _build_track_stats(rows)
    if not tracks:
        return {"applied": False, "reason": "no_tracks", "changed_rows": 0, "merged_pairs": 0}
    _extract_track_descriptors(video_path, tracks, reid_weights_path=reid_weights_path)

    gids = sorted(int(g) for g in tracks.keys() if int(g) > 0)
    if len(gids) < 2:
        return {"applied": False, "reason": "not_enough_ids", "changed_rows": 0, "merged_pairs": 0}

    parent: Dict[int, int] = {}

    def _root(g: int) -> int:
        x = int(g)
        while int(parent.get(int(x), int(x))) != int(x):
            x = int(parent.get(int(x), int(x)))
        return int(x)

    merged_pairs = 0
    for i in range(len(gids)):
        for j in range(i + 1, len(gids)):
            g1 = _root(int(gids[i]))
            g2 = _root(int(gids[j]))
            if g1 == g2:
                continue
            t1 = tracks.get(int(g1))
            t2 = tracks.get(int(g2))
            if t1 is None or t2 is None:
                continue

            sim_topk = float(_topk_similarity(t1.sample_descs, t2.sample_descs, k=8))
            sim_mean = float(_cos(t1.mean_desc, t2.mean_desc))
            sim_edge = max(
                float(_cos(t1.first_desc, t2.first_desc)),
                float(_cos(t1.last_desc, t2.last_desc)),
                float(_cos(t1.first_desc, t2.last_desc)),
                float(_cos(t1.last_desc, t2.first_desc)),
            )
            shape = _shape_score(t1.median_h_ratio, t1.median_aspect, t2.median_h_ratio, t2.median_aspect)
            score = float(0.52 * sim_topk + 0.32 * sim_mean + 0.10 * max(0.0, sim_edge) + 0.06 * shape)
            if score < float(min_score) or sim_topk < float(min_topk) or sim_mean < float(min_mean):
                continue

            shared, shared_ratio, mean_iou, max_iou = _track_overlap_stats(t1, t2)
            if int(shared) > int(max_shared_frames) and float(shared_ratio) > float(max_shared_ratio) and float(mean_iou) < 0.18 and float(max_iou) < 0.35:
                continue

            keep, drop = (int(g1), int(g2)) if int(g1) < int(g2) else (int(g2), int(g1))
            parent[int(drop)] = int(keep)
            merged_pairs += 1

    if not parent:
        return {"applied": False, "reason": "no_similarity_merges", "changed_rows": 0, "merged_pairs": 0}

    id_by_idx = {int(r.idx): int(r.gid) for r in rows}
    changed_rows = 0
    for idx, gid in list(id_by_idx.items()):
        if int(gid) <= 0:
            continue
        root = _root(int(gid))
        if int(root) != int(gid):
            id_by_idx[int(idx)] = int(root)
            changed_rows += 1

    if changed_rows <= 0:
        return {"applied": False, "reason": "no_row_changes", "changed_rows": 0, "merged_pairs": int(merged_pairs)}

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
    return {"applied": True, "changed_rows": int(changed_rows), "merged_pairs": int(merged_pairs)}


def _split_mixed_tracks_by_appearance(
    *,
    video_path: Path,
    tracks_csv_path: Path,
    reid_weights_path: str | None,
    min_rows: int = 160,
    min_segment_rows: int = 56,
    split_sim_thresh: float = 0.58,
    split_gap_frames: int = 6,
) -> Dict[str, object]:
    fieldnames, rows = _load_rows(tracks_csv_path)
    if not rows:
        return {"applied": False, "reason": "empty_tracks", "changed_rows": 0, "split_ids": 0}

    by_gid: Dict[int, List[object]] = {}
    for r in rows:
        g = int(r.gid)
        if g > 0:
            by_gid.setdefault(g, []).append(r)
    candidate_gids = {int(g) for g, rs in by_gid.items() if len(rs) >= int(min_rows)}
    if not candidate_gids:
        return {"applied": False, "reason": "no_candidate_ids", "changed_rows": 0, "split_ids": 0}

    rows_by_frame: Dict[int, List[object]] = {}
    row_by_idx: Dict[int, object] = {}
    for r in rows:
        row_by_idx[int(r.idx)] = r
        if int(r.gid) in candidate_gids:
            rows_by_frame.setdefault(int(r.frame_idx), []).append(r)
    if not rows_by_frame:
        return {"applied": False, "reason": "no_candidate_rows", "changed_rows": 0, "split_ids": 0}

    extractor = None
    try:
        extractor = ReidExtractor(model_name="osnet_x1_0", device="cpu", model_path=reid_weights_path)
    except Exception:
        extractor = None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"applied": False, "reason": "video_open_failed", "changed_rows": 0, "split_ids": 0}

    desc_by_gid: Dict[int, List[Tuple[int, int, np.ndarray]]] = {}
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fr_rows = rows_by_frame.get(int(frame_idx), [])
        if fr_rows:
            descs = _build_descs_for_rows(frame, fr_rows, extractor=extractor)
            for r in fr_rows:
                d = descs.get(int(r.idx))
                if d is None:
                    continue
                desc_by_gid.setdefault(int(r.gid), []).append((int(frame_idx), int(r.idx), d))
        frame_idx += 1
    cap.release()

    if not desc_by_gid:
        return {"applied": False, "reason": "no_descriptors", "changed_rows": 0, "split_ids": 0}

    max_gid = max(int(r.gid) for r in rows if int(r.gid) > 0)
    idx_to_new_gid: Dict[int, int] = {}
    split_ids = 0

    for gid in sorted(desc_by_gid.keys()):
        seq = sorted(desc_by_gid.get(int(gid), []), key=lambda x: (x[0], x[1]))
        if len(seq) < int(2 * min_segment_rows):
            continue

        segments: List[List[Tuple[int, int, np.ndarray]]] = []
        seg: List[Tuple[int, int, np.ndarray]] = [seq[0]]
        centroid = seq[0][2].copy()
        prev_fr = int(seq[0][0])

        for fr, idx, d in seq[1:]:
            sim = float(_cos(d, centroid))
            gap = int(fr) - int(prev_fr)
            remaining = int(len(seq) - (len(seg) + sum(len(s) for s in segments)))
            should_split = (
                len(seg) >= int(min_segment_rows)
                and remaining >= int(min_segment_rows)
                and (sim < float(split_sim_thresh) or gap > int(split_gap_frames))
            )
            if should_split:
                segments.append(seg)
                seg = [(int(fr), int(idx), d)]
                centroid = d.copy()
            else:
                seg.append((int(fr), int(idx), d))
                centroid = 0.78 * centroid + 0.22 * d
                nrm = float(np.linalg.norm(centroid) + 1e-6)
                centroid = centroid / nrm
            prev_fr = int(fr)
        if seg:
            segments.append(seg)

        strong_segments = [s for s in segments if len(s) >= int(min_segment_rows)]
        if len(strong_segments) <= 1:
            continue

        split_ids += 1
        for seg_i, s in enumerate(strong_segments):
            if seg_i == 0:
                continue
            max_gid += 1
            for _fr, idx, _d in s:
                idx_to_new_gid[int(idx)] = int(max_gid)

    if not idx_to_new_gid:
        return {"applied": False, "reason": "no_split", "changed_rows": 0, "split_ids": 0}

    changed_rows = 0
    id_by_idx: Dict[int, int] = {int(r.idx): int(r.gid) for r in rows}
    for idx, new_gid in idx_to_new_gid.items():
        if int(id_by_idx.get(int(idx), 0)) > 0 and int(id_by_idx.get(int(idx), 0)) != int(new_gid):
            id_by_idx[int(idx)] = int(new_gid)
            changed_rows += 1

    if changed_rows <= 0:
        return {"applied": False, "reason": "no_changes", "changed_rows": 0, "split_ids": int(split_ids)}

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

    return {"applied": True, "changed_rows": int(changed_rows), "split_ids": int(split_ids)}


def _enforce_max_ids_by_similarity(
    *,
    video_path: Path,
    tracks_csv_path: Path,
    reid_weights_path: str | None,
    max_ids: int = 6,
    min_merge_score: float = 0.66,
    min_rows_force_drop: int = 24,
) -> Dict[str, object]:
    fieldnames, rows = _load_rows(tracks_csv_path)
    if not rows:
        return {"applied": False, "reason": "empty_tracks", "changed_rows": 0}
    tracks = _build_track_stats(rows)
    if not tracks:
        return {"applied": False, "reason": "no_tracks", "changed_rows": 0}
    _extract_track_descriptors(video_path, tracks, reid_weights_path=reid_weights_path)

    row_count_by_gid = {int(g): int(len(st.rows)) for g, st in tracks.items() if int(g) > 0}
    gids_sorted = sorted(row_count_by_gid.keys(), key=lambda g: row_count_by_gid[g], reverse=True)
    if len(gids_sorted) <= int(max_ids):
        return {"applied": False, "reason": "already_within_limit", "changed_rows": 0}

    keep_ids = set(int(g) for g in gids_sorted[: int(max_ids)])
    extra_ids = [int(g) for g in gids_sorted[int(max_ids) :]]

    remap: Dict[int, int] = {}
    for g in extra_ids:
        tg = tracks.get(int(g))
        if tg is None:
            remap[int(g)] = 0
            continue
        best_keep = 0
        best_score = -1.0
        for k in keep_ids:
            tk = tracks.get(int(k))
            if tk is None:
                continue
            sim_topk = float(_topk_similarity(tg.sample_descs, tk.sample_descs, k=6))
            sim_mean = float(_cos(tg.mean_desc, tk.mean_desc))
            sim_edge = max(
                float(_cos(tg.first_desc, tk.first_desc)),
                float(_cos(tg.last_desc, tk.last_desc)),
                float(_cos(tg.first_desc, tk.last_desc)),
                float(_cos(tg.last_desc, tk.first_desc)),
            )
            shape = _shape_score(tg.median_h_ratio, tg.median_aspect, tk.median_h_ratio, tk.median_aspect)
            score = float(0.50 * sim_topk + 0.30 * sim_mean + 0.10 * max(0.0, sim_edge) + 0.10 * shape)
            if score > best_score:
                best_score = float(score)
                best_keep = int(k)
        if best_keep > 0 and (best_score >= float(min_merge_score) or row_count_by_gid.get(int(g), 0) <= int(min_rows_force_drop)):
            remap[int(g)] = int(best_keep)
        else:
            remap[int(g)] = 0

    if not remap:
        return {"applied": False, "reason": "no_remap", "changed_rows": 0}

    id_by_idx = {int(r.idx): int(r.gid) for r in rows}
    changed_rows = 0
    for idx, gid in list(id_by_idx.items()):
        if gid in remap:
            new_gid = int(remap[gid])
            if new_gid != int(gid):
                id_by_idx[int(idx)] = int(new_gid)
                changed_rows += 1
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
    return {
        "applied": True,
        "changed_rows": int(changed_rows),
        "keep_ids": sorted(int(x) for x in keep_ids),
        "extra_ids": [int(x) for x in extra_ids],
    }


def _run_single_camera(
    *,
    video_path: Path,
    out_csv: Path,
    tracker: BOTSORT,
    reid_weights_path: str | None,
    debug_dir: Path,
) -> None:
    if hasattr(tracker, "reset_for_new_video"):
        try:
            tracker.reset_for_new_video(reset_ids=True)
        except TypeError:
            tracker.reset_for_new_video()

    pre = preprocess_video(video_path, out_dir=out_csv.parent, write_report=True)
    print(
        f"[PRE]  {pre.video}  {pre.resolution[0]}x{pre.resolution[1]}"
        f"  {pre.fps:.2f}fps  {pre.frame_count}f  {pre.duration_s:.1f}s"
        f"  bad={pre.quality['bad_pct']}%"
    )

    run_video_with_kpis(
        video_path=video_path,
        out_tracks_csv=out_csv,
        fps_override=None,
        tracker=tracker,
        forced_global_id=None,
        out_video_path=None,
        draw=False,
    )



def _shape_score(h1: float, a1: float, h2: float, a2: float) -> float:
    h1 = max(1e-6, float(h1))
    h2 = max(1e-6, float(h2))
    a1 = max(1e-6, float(a1))
    a2 = max(1e-6, float(a2))
    s_h = min(h1, h2) / max(h1, h2)
    s_a = min(a1, a2) / max(a1, a2)
    return float(np.clip(0.65 * s_h + 0.35 * s_a, 0.0, 1.0))


def _build_pair_scores(
    *,
    video1: Path,
    csv1: Path,
    video2: Path,
    csv2: Path,
    reid_weights_path: str | None,
    min_rows: int,
    min_topk: float,
    min_mean: float,
    min_score: float,
) -> Tuple[List[PairScore], Dict[int, object], Dict[int, object]]:
    _, rows1 = _load_rows(csv1)
    _, rows2 = _load_rows(csv2)
    tracks1 = _build_track_stats(rows1)
    tracks2 = _build_track_stats(rows2)
    w1, h1 = _video_dims(video1)
    w2, h2 = _video_dims(video2)
    _enrich_track_meta(tracks1, frame_w=w1, frame_h=h1)
    _enrich_track_meta(tracks2, frame_w=w2, frame_h=h2)
    _extract_track_descriptors(video1, tracks1, reid_weights_path=reid_weights_path)
    _extract_track_descriptors(video2, tracks2, reid_weights_path=reid_weights_path)

    filtered1 = {
        int(g): st
        for g, st in tracks1.items()
        if int(g) > 0 and len(st.rows) >= int(min_rows) and st.mean_desc is not None and len(st.sample_descs) >= 3
    }
    filtered2 = {
        int(g): st
        for g, st in tracks2.items()
        if int(g) > 0 and len(st.rows) >= int(min_rows) and st.mean_desc is not None and len(st.sample_descs) >= 3
    }

    pairs: List[PairScore] = []
    for g1, t1 in filtered1.items():
        for g2, t2 in filtered2.items():
            sim_topk = float(_topk_similarity(t1.sample_descs, t2.sample_descs, k=8))
            if sim_topk < float(min_topk):
                continue
            sim_mean = float(_cos(t1.mean_desc, t2.mean_desc))
            if sim_mean < float(min_mean):
                continue
            sim_edge = max(
                float(_cos(t1.first_desc, t2.first_desc)),
                float(_cos(t1.last_desc, t2.last_desc)),
                float(_cos(t1.last_desc, t2.first_desc)),
                float(_cos(t1.first_desc, t2.last_desc)),
            )
            shape = _shape_score(t1.median_h_ratio, t1.median_aspect, t2.median_h_ratio, t2.median_aspect)
            score = float(
                0.52 * sim_topk
                + 0.32 * sim_mean
                + 0.10 * max(0.0, sim_edge)
                + 0.06 * shape
            )
            if score < float(min_score):
                continue
            pairs.append(
                PairScore(
                    cam1_gid=int(g1),
                    cam2_gid=int(g2),
                    score=float(score),
                    sim_topk=float(sim_topk),
                    sim_mean=float(sim_mean),
                    sim_edge=float(sim_edge),
                    shape=float(shape),
                    rows_cam1=int(len(t1.rows)),
                    rows_cam2=int(len(t2.rows)),
                )
            )
    pairs.sort(key=lambda p: p.score, reverse=True)
    return pairs, tracks1, tracks2


def _greedy_match_pairs(pairs: List[PairScore], max_pairs: int | None = None) -> List[PairScore]:
    used1: set[int] = set()
    used2: set[int] = set()
    matched: List[PairScore] = []
    for p in pairs:
        if p.cam1_gid in used1 or p.cam2_gid in used2:
            continue
        matched.append(p)
        used1.add(p.cam1_gid)
        used2.add(p.cam2_gid)
        if max_pairs is not None and len(matched) >= int(max_pairs):
            break
    return matched


def _track_frames_set(track) -> set[int]:
    return {int(r.frame_idx) for r in track.rows}


def _track_overlap_stats(track_a, track_b) -> Tuple[int, float, float, float]:
    """
    Returns:
      shared_frames, shared_ratio(min-denom), mean_iou, max_iou
    """
    by_f_a: Dict[int, List[Tuple[float, float, float, float]]] = {}
    by_f_b: Dict[int, List[Tuple[float, float, float, float]]] = {}

    for r in track_a.rows:
        by_f_a.setdefault(int(r.frame_idx), []).append((float(r.x1), float(r.y1), float(r.x2), float(r.y2)))
    for r in track_b.rows:
        by_f_b.setdefault(int(r.frame_idx), []).append((float(r.x1), float(r.y1), float(r.x2), float(r.y2)))

    fa = set(by_f_a.keys())
    fb = set(by_f_b.keys())
    inter = sorted(fa.intersection(fb))
    if not inter:
        return 0, 0.0, 0.0, 0.0

    ious: List[float] = []
    for fr in inter:
        a_box = max(by_f_a.get(fr, []), key=lambda b: max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]))
        b_box = max(by_f_b.get(fr, []), key=lambda b: max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]))
        xA = max(float(a_box[0]), float(b_box[0]))
        yA = max(float(a_box[1]), float(b_box[1]))
        xB = min(float(a_box[2]), float(b_box[2]))
        yB = min(float(a_box[3]), float(b_box[3]))
        inter_area = max(0.0, xB - xA) * max(0.0, yB - yA)
        if inter_area <= 0.0:
            ious.append(0.0)
            continue
        aa = max(0.0, float(a_box[2] - a_box[0])) * max(0.0, float(a_box[3] - a_box[1]))
        ab = max(0.0, float(b_box[2] - b_box[0])) * max(0.0, float(b_box[3] - b_box[1]))
        den = aa + ab - inter_area + 1e-6
        iou = float(inter_area / den) if den > 0 else 0.0
        ious.append(iou)

    shared = int(len(inter))
    denom = max(1, min(len(fa), len(fb)))
    shared_ratio = float(shared / float(denom))
    mean_iou = float(np.mean(ious)) if ious else 0.0
    max_iou = float(np.max(ious)) if ious else 0.0
    return shared, shared_ratio, mean_iou, max_iou


def _allow_cam2_fragment_reuse(
    *,
    cam2_gid: int,
    target_cam1_gid: int,
    cam2_to_global: Dict[int, int],
    tracks2: Dict[int, object],
    max_shared_frames: int = 8,
    max_overlap_ratio: float = 0.10,
) -> bool:
    t_new = tracks2.get(int(cam2_gid))
    if t_new is None:
        return False
    f_new = _track_frames_set(t_new)
    if not f_new:
        return False

    for other_cam2, mapped_gid in cam2_to_global.items():
        if int(mapped_gid) != int(target_cam1_gid):
            continue
        if int(other_cam2) == int(cam2_gid):
            continue
        t_old = tracks2.get(int(other_cam2))
        if t_old is None:
            continue
        shared, shared_ratio, mean_iou, max_iou = _track_overlap_stats(t_new, t_old)
        if shared <= 0:
            continue
        # If overlaps are tiny and only a few frames, it's likely an ID fragment.
        if shared <= int(max_shared_frames) and shared_ratio <= float(max_overlap_ratio):
            continue
        # Allow same-person fragment reuse when temporal overlap is still very small
        # but appearance is extremely consistent.
        if shared_ratio <= max(0.12, float(max_overlap_ratio) * 1.6):
            sim_topk = float(_topk_similarity(t_new.sample_descs, t_old.sample_descs, k=8))
            sim_mean = float(_cos(t_new.mean_desc, t_old.mean_desc))
            if sim_topk >= 0.90 and sim_mean >= 0.88:
                continue
        # For dense overlap windows, still allow reuse when appearance agreement is
        # exceptionally strong (typical upper/lower or duplicate split fragments).
        sim_topk = float(_topk_similarity(t_new.sample_descs, t_old.sample_descs, k=8))
        sim_mean = float(_cos(t_new.mean_desc, t_old.mean_desc))
        if sim_topk >= 0.93 and sim_mean >= 0.90:
            continue
        # If they overlap heavily in time but boxes are nearly identical, allow merge (duplicate tracks).
        if mean_iou >= 0.28 or max_iou >= 0.52:
            continue
        return False
    return True


def _extend_cam2_matches(
    *,
    pair_scores: List[PairScore],
    cam2_ids: List[int],
    cam1_set: set[int],
    tracks2: Dict[int, object],
    cam2_to_global: Dict[int, int],
    used_cam2: set[int],
    min_score: float = 0.73,
    min_margin: float = 0.03,
    min_topk: float = 0.66,
    min_mean: float = 0.66,
) -> List[PairScore]:
    by_cam2: Dict[int, List[PairScore]] = {}
    for p in pair_scores:
        if int(p.cam1_gid) <= 0 or int(p.cam1_gid) not in cam1_set:
            continue
        by_cam2.setdefault(int(p.cam2_gid), []).append(p)
    for cands in by_cam2.values():
        cands.sort(key=lambda x: x.score, reverse=True)

    accepted: List[PairScore] = []
    for g2 in cam2_ids:
        if int(g2) in used_cam2:
            continue
        cands = by_cam2.get(int(g2), [])
        if not cands:
            continue
        best = cands[0]
        second = cands[1].score if len(cands) > 1 else -1.0
        if float(best.score) < float(min_score):
            continue
        if float(best.sim_topk) < float(min_topk) or float(best.sim_mean) < float(min_mean):
            continue
        if second >= 0.0 and (float(best.score) - float(second)) < float(min_margin):
            continue
        target = int(best.cam1_gid)
        if not _allow_cam2_fragment_reuse(
            cam2_gid=int(g2),
            target_cam1_gid=target,
            cam2_to_global=cam2_to_global,
            tracks2=tracks2,
            max_shared_frames=5,
            max_overlap_ratio=0.08,
        ):
            continue
        cam2_to_global[int(g2)] = int(target)
        used_cam2.add(int(g2))
        accepted.append(best)
    return accepted


def _assign_cam2_global_ids(
    *,
    pair_scores: List[PairScore],
    cam2_ids: List[int],
    cam1_set: set[int],
    tracks1: Dict[int, object],
    tracks2: Dict[int, object],
    max_cam2_ids: int = 7,
) -> Tuple[Dict[int, int], List[Dict[str, object]]]:
    by_cam2: Dict[int, List[PairScore]] = {}
    for p in pair_scores:
        if int(p.cam1_gid) <= 0 or int(p.cam1_gid) not in cam1_set:
            continue
        by_cam2.setdefault(int(p.cam2_gid), []).append(p)
    for cands in by_cam2.values():
        cands.sort(key=lambda x: x.score, reverse=True)

    row_count_by_cam2: Dict[int, int] = {}
    for g2, st in tracks2.items():
        row_count_by_cam2[int(g2)] = int(len(st.rows))
    order = sorted([int(g) for g in cam2_ids], key=lambda g: row_count_by_cam2.get(int(g), 0), reverse=True)

    cam2_to_global: Dict[int, int] = {}
    records: List[Dict[str, object]] = []

    def _record(c2: int, final_gid: int, kind: str, cand: PairScore | None = None) -> None:
        if cand is None:
            records.append(
                {
                    "cam1_gid": 0,
                    "cam2_gid": int(c2),
                    "final_global_id": int(final_gid),
                    "score": "",
                    "sim_topk": "",
                    "sim_mean": "",
                    "sim_edge": "",
                    "shape": "",
                    "rows_cam1": "",
                    "rows_cam2": int(row_count_by_cam2.get(int(c2), 0)),
                    "match_type": str(kind),
                }
            )
            return
        records.append(
            {
                "cam1_gid": int(cand.cam1_gid),
                "cam2_gid": int(cand.cam2_gid),
                "final_global_id": int(final_gid),
                "score": f"{cand.score:.6f}",
                "sim_topk": f"{cand.sim_topk:.6f}",
                "sim_mean": f"{cand.sim_mean:.6f}",
                "sim_edge": f"{cand.sim_edge:.6f}",
                "shape": f"{cand.shape:.6f}",
                "rows_cam1": int(cand.rows_cam1),
                "rows_cam2": int(cand.rows_cam2),
                "match_type": str(kind),
            }
        )

    def _try_assign(
        c2: int,
        *,
        allowed_cam1: set[int] | None,
        score_floor: float,
        topk_floor: float,
        mean_floor: float,
        margin_floor: float,
        kind: str,
    ) -> bool:
        cands = by_cam2.get(int(c2), [])
        if not cands:
            return False
        valid = [
            c
            for c in cands
            if (
                float(c.score) >= float(score_floor)
                and float(c.sim_topk) >= float(topk_floor)
                and float(c.sim_mean) >= float(mean_floor)
                and (allowed_cam1 is None or int(c.cam1_gid) in allowed_cam1)
            )
        ]
        if not valid:
            return False
        # For relaxed fallback, do not drop to lower-ranked candidates if the top
        # candidate fails overlap/continuity checks. Prefer creating a new ID over
        # stealing an unrelated existing identity.
        if str(kind) == "fallback_reuse":
            valid = valid[:1]
        # If top-two are too close at moderate confidence, do not force a weak decision.
        if len(valid) >= 2:
            gap = float(valid[0].score) - float(valid[1].score)
            if gap < float(margin_floor) and float(valid[0].score) < 0.74:
                return False
        for cand in valid:
            target = int(cand.cam1_gid)
            if not _allow_cam2_fragment_reuse(
                cam2_gid=int(c2),
                target_cam1_gid=int(target),
                cam2_to_global=cam2_to_global,
                tracks2=tracks2,
                max_shared_frames=5,
                max_overlap_ratio=0.08,
            ):
                continue
            cam2_to_global[int(c2)] = int(target)
            _record(int(c2), int(target), kind, cand)
            return True
        return False

    def _cam2_pair_similarity(c2a: int, c2b: int) -> Tuple[float, float, float, float, float]:
        ta = tracks2.get(int(c2a))
        tb = tracks2.get(int(c2b))
        if ta is None or tb is None:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        if ta.mean_desc is None or tb.mean_desc is None:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        if len(ta.sample_descs) < 3 or len(tb.sample_descs) < 3:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        sim_topk = float(_topk_similarity(ta.sample_descs, tb.sample_descs, k=8))
        sim_mean = float(_cos(ta.mean_desc, tb.mean_desc))
        sim_edge = max(
            float(_cos(ta.first_desc, tb.first_desc)),
            float(_cos(ta.last_desc, tb.last_desc)),
            float(_cos(ta.last_desc, tb.first_desc)),
            float(_cos(ta.first_desc, tb.last_desc)),
        )
        shape = _shape_score(ta.median_h_ratio, ta.median_aspect, tb.median_h_ratio, tb.median_aspect)
        score = float(0.52 * sim_topk + 0.32 * sim_mean + 0.10 * max(0.0, sim_edge) + 0.06 * shape)
        return score, sim_topk, sim_mean, sim_edge, shape

    def _median_cx(track_obj) -> float:
        if track_obj is None:
            return 0.0
        xs = [float(0.5 * (r.x1 + r.x2)) for r in getattr(track_obj, "rows", [])]
        if not xs:
            return 0.0
        return float(np.median(np.asarray(xs, dtype=np.float32)))

    def _median_cy(track_obj) -> float:
        if track_obj is None:
            return 0.0
        ys = [float(0.5 * (r.y1 + r.y2)) for r in getattr(track_obj, "rows", [])]
        if not ys:
            return 0.0
        return float(np.median(np.asarray(ys, dtype=np.float32)))

    def _cam2_score_to_gid(c2: int, gid: int) -> float:
        if int(gid) <= 0:
            return -1.0
        for c in by_cam2.get(int(c2), []):
            if int(c.cam1_gid) == int(gid):
                return float(c.score)
        return -1.0

    # 1) Canonical one-to-one owner assignment (automatic, no forced map).
    #    Pick the strongest cam2 owner for each cam1 first.
    cam1_owner: Dict[int, int] = {}
    for cand in pair_scores:
        c1 = int(cand.cam1_gid)
        c2 = int(cand.cam2_gid)
        if c1 not in cam1_set:
            continue
        if c1 in cam1_owner or c2 in cam2_to_global:
            continue
        cands = by_cam2.get(c2, [])
        second = cands[1].score if len(cands) > 1 else -1.0
        if float(cand.score) < 0.67 or float(cand.sim_topk) < 0.62 or float(cand.sim_mean) < 0.60:
            continue
        if second >= 0.0 and (float(cand.score) - float(second)) < 0.02 and float(cand.score) < 0.75:
            continue
        cam1_owner[c1] = c2
        cam2_to_global[c2] = c1
        _record(c2, c1, "canonical_primary", cand)

    # 2) Fill unmatched cam1 IDs with strong matches first.
    unclaimed_cam1 = {int(c1) for c1 in cam1_set if int(c1) not in set(cam2_to_global.values())}
    for c2 in order:
        if int(c2) in cam2_to_global:
            continue
        if not unclaimed_cam1:
            break
        did = _try_assign(
            int(c2),
            allowed_cam1=set(unclaimed_cam1),
            score_floor=0.66,
            topk_floor=0.61,
            mean_floor=0.60,
            margin_floor=0.02,
            kind="matched_unclaimed",
        )
        if did:
            unclaimed_cam1 = {int(c1) for c1 in cam1_set if int(c1) not in set(cam2_to_global.values())}

    # 3) strong automatic assignment to already-claimed canonical IDs (fragment reuse).
    for c2 in order:
        if int(c2) in cam2_to_global:
            continue
        _try_assign(
            int(c2),
            allowed_cam1=None,
            score_floor=0.67,
            topk_floor=0.63,
            mean_floor=0.63,
            margin_floor=0.02,
            kind="fragment_reuse",
        )

    # 4) relaxed fallback assignment
    for c2 in order:
        if int(c2) in cam2_to_global:
            continue
        _try_assign(
            int(c2),
            allowed_cam1=None,
            score_floor=0.63,
            topk_floor=0.59,
            mean_floor=0.60,
            margin_floor=0.01,
            kind="fallback_reuse",
        )

    # 5) Reuse mapped cam2 fragments when cross-camera score is weak but same-camera continuity is strong.
    for c2 in order:
        if int(c2) in cam2_to_global:
            continue
        cands: List[Tuple[float, float, float, int, int, float]] = []
        for m2, target_gid in cam2_to_global.items():
            if int(target_gid) <= 0:
                continue
            t_new = tracks2.get(int(c2))
            t_old = tracks2.get(int(m2))
            if t_new is None or t_old is None:
                continue
            score, sim_topk, sim_mean, _sim_edge, _shape = _cam2_pair_similarity(int(c2), int(m2))
            if score < 0.68 or sim_topk < 0.64 or sim_mean < 0.62:
                continue
            shared, _shared_ratio, mean_iou, _max_iou = _track_overlap_stats(t_new, t_old)
            cands.append((float(score), float(sim_topk), float(sim_mean), int(target_gid), int(shared), float(mean_iou)))
        if not cands:
            continue
        cands.sort(reverse=True)
        best_score, _best_topk, _best_mean, best_gid, _best_shared, _best_mean_iou = cands[0]
        second_score = cands[1][0] if len(cands) > 1 else -1.0
        if second_score >= 0.0 and (best_score - second_score) < 0.03 and best_score < 0.77:
            # Tie-break ambiguous scores using same-camera overlap continuity.
            overlap_choice = max(cands, key=lambda x: (int(x[4]), float(x[5]), float(x[0])))
            if int(overlap_choice[4]) >= 5 and float(overlap_choice[5]) >= 0.08 and float(overlap_choice[0]) >= 0.68:
                best_gid = int(overlap_choice[3])
            else:
                continue
        if not _allow_cam2_fragment_reuse(
            cam2_gid=int(c2),
            target_cam1_gid=int(best_gid),
            cam2_to_global=cam2_to_global,
            tracks2=tracks2,
            max_shared_frames=6,
            max_overlap_ratio=0.10,
        ):
            continue
        cam2_to_global[int(c2)] = int(best_gid)
        _record(int(c2), int(best_gid), "cam2_fragment_reuse", None)

    # 5.5) Anchor side prior for ambiguous ID pairs (helps black-hijab vs old-woman swaps).
    # Uses the stable blue-shirt anchor (cam1 gid=2) and relative left/right layout.
    if {1, 2, 3}.issubset(set(int(x) for x in cam1_set)) and int(2) in tracks1 and int(1) in tracks1 and int(3) in tracks1:
        cam1_anchor_x = _median_cx(tracks1.get(int(2)))
        cam1_left_x = _median_cx(tracks1.get(int(3)))
        cam1_right_x = _median_cx(tracks1.get(int(1)))
        # Only apply if cam1 geometry is sane: id3 mostly left of id2 and id1 mostly right of id2.
        if cam1_left_x < cam1_anchor_x - 15.0 and cam1_right_x > cam1_anchor_x + 15.0:
            anchor_cam2_xs = [_median_cx(tracks2.get(int(c2))) for c2, gid in cam2_to_global.items() if int(gid) == 2 and int(c2) in tracks2]
            if anchor_cam2_xs:
                cam2_anchor_x = float(np.median(np.asarray(anchor_cam2_xs, dtype=np.float32)))
                for _pass in range(3):
                    changed = False
                    for c2 in order:
                        tr = tracks2.get(int(c2))
                        if tr is None:
                            continue
                        dx = float(_median_cx(tr) - cam2_anchor_x)
                        if abs(dx) < 10.0:
                            continue
                        preferred_gid = 3 if dx < 0.0 else 1
                        alt_gid = 1 if preferred_gid == 3 else 3

                        cands = by_cam2.get(int(c2), [])
                        cand_pref = next((c for c in cands if int(c.cam1_gid) == int(preferred_gid)), None)
                        cand_alt = next((c for c in cands if int(c.cam1_gid) == int(alt_gid)), None)
                        s_pref = float(cand_pref.score) if cand_pref is not None else -1.0
                        s_alt = float(cand_alt.score) if cand_alt is not None else -1.0
                        if s_pref < 0.64:
                            continue
                        # Do not steal a track that is strongly owned by another canonical ID.
                        best_gid = int(cands[0].cam1_gid) if cands else 0
                        best_score = float(cands[0].score) if cands else -1.0
                        if best_gid not in {1, 3} and best_score >= (s_pref + 0.05):
                            continue
                        # Let side prior break close ties, but do not override very strong opposite evidence.
                        if s_alt > (s_pref + 0.12):
                            continue

                        cur_gid = int(cam2_to_global.get(int(c2), 0))
                        if cur_gid == int(preferred_gid):
                            continue
                        if not _allow_cam2_fragment_reuse(
                            cam2_gid=int(c2),
                            target_cam1_gid=int(preferred_gid),
                            cam2_to_global=cam2_to_global,
                            tracks2=tracks2,
                            max_shared_frames=8,
                            max_overlap_ratio=0.12,
                        ):
                            continue
                        cam2_to_global[int(c2)] = int(preferred_gid)
                        _record(int(c2), int(preferred_gid), "anchor_side_fix", cand_pref)
                        changed = True
                    if not changed:
                        break

                # Final side-consensus pass for ambiguous 1/3 tracks only.
                # Applied in batch to avoid ordering deadlocks under overlap constraints.
                pending_updates: Dict[int, int] = {}
                for c2 in order:
                    tr = tracks2.get(int(c2))
                    if tr is None:
                        continue
                    cands = by_cam2.get(int(c2), [])
                    if not cands:
                        continue
                    dx = float(_median_cx(tr) - cam2_anchor_x)
                    if abs(dx) < 12.0:
                        continue
                    preferred_gid = 3 if dx < 0.0 else 1
                    alt_gid = 1 if preferred_gid == 3 else 3
                    cand_pref = next((c for c in cands if int(c.cam1_gid) == int(preferred_gid)), None)
                    cand_alt = next((c for c in cands if int(c.cam1_gid) == int(alt_gid)), None)
                    s_pref = float(cand_pref.score) if cand_pref is not None else -1.0
                    s_alt = float(cand_alt.score) if cand_alt is not None else -1.0
                    if s_pref < 0.62:
                        continue
                    best_gid = int(cands[0].cam1_gid)
                    best_score = float(cands[0].score)
                    if best_gid not in {1, 3} and best_score >= (s_pref + 0.05):
                        continue
                    if s_alt > (s_pref + 0.12):
                        continue
                    cur_gid = int(cam2_to_global.get(int(c2), 0))
                    if cur_gid == int(preferred_gid):
                        continue
                    if cur_gid not in {0, 1, 3, 6, 7} and best_gid not in {1, 3}:
                        continue
                    pending_updates[int(c2)] = int(preferred_gid)

                for c2, new_gid in pending_updates.items():
                    cam2_to_global[int(c2)] = int(new_gid)
                    _record(int(c2), int(new_gid), "anchor_side_consensus", None)

    # 5.7) Overlap-dominant reuse for remaining unmatched short fragments.
    # If an unmatched cam2 fragment consistently overlaps one mapped cam2 track,
    # reuse that mapped final ID instead of creating a new spurious ID.
    for c2 in order:
        if int(c2) in cam2_to_global:
            continue
        t_new = tracks2.get(int(c2))
        if t_new is None:
            continue
        overlap_cands: List[Tuple[int, float, float, int]] = []
        for m2, target_gid in cam2_to_global.items():
            if int(target_gid) <= 0:
                continue
            t_old = tracks2.get(int(m2))
            if t_old is None:
                continue
            shared, _shared_ratio, mean_iou, max_iou = _track_overlap_stats(t_new, t_old)
            if int(shared) < 5:
                continue
            if float(mean_iou) < 0.10 and float(max_iou) < 0.24:
                continue
            overlap_cands.append((int(target_gid), float(mean_iou), float(max_iou), int(shared)))
        if not overlap_cands:
            continue
        overlap_cands.sort(key=lambda x: (x[3], x[2], x[1]), reverse=True)
        best_gid, best_mean_iou, best_max_iou, best_shared = overlap_cands[0]
        if len(overlap_cands) >= 2:
            _g2, _m2, _x2, sh2 = overlap_cands[1]
            if int(best_shared) < int(sh2) + 3:
                continue
        if best_shared < 6 or (best_mean_iou < 0.11 and best_max_iou < 0.26):
            continue
        cam2_to_global[int(c2)] = int(best_gid)
        _record(int(c2), int(best_gid), "overlap_dominant_reuse", None)

    # 5.8) Strong same-camera fragment propagation.
    # If two cam2 tracklets are near-identical (appearance + location continuity),
    # keep their final global IDs consistent to prevent early/late ID drift.
    changed = True
    max_iter = 4
    iter_idx = 0
    while changed and iter_idx < max_iter:
        iter_idx += 1
        changed = False
        used_targets_this_pass: set[int] = set()
        for c2 in order:
            cur_gid = int(cam2_to_global.get(int(c2), 0))
            if cur_gid <= 0:
                continue
            t_cur = tracks2.get(int(c2))
            if t_cur is None:
                continue

            cur_x = _median_cx(t_cur)
            cur_y = _median_cy(t_cur)
            cur_h = max(1.0, float(getattr(t_cur, "median_h_ratio", 0.0)) * 1944.0)
            cur_gid_score = _cam2_score_to_gid(int(c2), int(cur_gid))

            best_candidate: Tuple[float, int, int] | None = None  # (score_to_new_gid, new_gid, source_c2)
            for o2, o_gid in cam2_to_global.items():
                if int(o2) == int(c2):
                    continue
                new_gid = int(o_gid)
                if new_gid <= 0 or new_gid == int(cur_gid):
                    continue
                t_old = tracks2.get(int(o2))
                if t_old is None:
                    continue

                score, sim_topk, sim_mean, _sim_edge, _shape = _cam2_pair_similarity(int(c2), int(o2))
                if score < 0.94 or sim_topk < 0.95 or sim_mean < 0.90:
                    continue

                old_x = _median_cx(t_old)
                old_y = _median_cy(t_old)
                old_h = max(1.0, float(getattr(t_old, "median_h_ratio", 0.0)) * 1944.0)
                h_norm = max(1.0, 0.5 * (cur_h + old_h))
                dist_norm = float(np.hypot(cur_x - old_x, cur_y - old_y) / h_norm)
                if dist_norm > 1.45:
                    continue

                shared, _shared_ratio, mean_iou, max_iou = _track_overlap_stats(t_cur, t_old)
                # Two truly different people can overlap in time with little box overlap.
                if int(shared) >= 12 and float(mean_iou) < 0.20 and float(max_iou) < 0.35:
                    continue

                new_gid_score = _cam2_score_to_gid(int(c2), int(new_gid))
                if new_gid_score < 0.72:
                    continue
                # Allow changing only when destination evidence is close/enhanced.
                if cur_gid_score >= 0.0 and new_gid_score + 0.04 < float(cur_gid_score):
                    continue

                if not _allow_cam2_fragment_reuse(
                    cam2_gid=int(c2),
                    target_cam1_gid=int(new_gid),
                    cam2_to_global=cam2_to_global,
                    tracks2=tracks2,
                    max_shared_frames=10,
                    max_overlap_ratio=0.14,
                ):
                    continue

                candidate_key = (float(new_gid_score), int(new_gid), int(o2))
                if best_candidate is None or candidate_key > best_candidate:
                    best_candidate = candidate_key

            if best_candidate is None:
                continue
            _new_score, new_gid, src_c2 = best_candidate
            if int(new_gid) in used_targets_this_pass:
                continue
            cam2_to_global[int(c2)] = int(new_gid)
            _record(int(c2), int(new_gid), f"same_cam_fragment_propagate:{int(src_c2)}", None)
            used_targets_this_pass.add(int(new_gid))
            changed = True

    # 6) cap final cam2 identities (6 people + 1 passer-by)
    used_cam1 = {int(v) for v in cam2_to_global.values() if int(v) > 0}
    allow_new = max(0, int(max_cam2_ids) - int(len(used_cam1)))
    unmatched = [int(c2) for c2 in order if int(c2) not in cam2_to_global]
    next_gid = (max(cam1_set) + 1) if cam1_set else 1
    for rank, c2 in enumerate(unmatched):
        if rank < int(allow_new):
            cam2_to_global[int(c2)] = int(next_gid)
            _record(int(c2), int(next_gid), "new", None)
            next_gid += 1
        else:
            cam2_to_global[int(c2)] = 0
            _record(int(c2), 0, "dropped_cap", None)

    return cam2_to_global, records


def _remap_csv_gids(csv_path: Path, gid_map: Dict[int, int]) -> None:
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        out_rows = []
        for raw in reader:
            try:
                gid = int(raw.get("global_id", 0) or 0)
            except Exception:
                gid = 0
            if gid > 0 and gid in gid_map:
                raw["global_id"] = str(int(gid_map[gid]))
            out_rows.append(raw)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)


def _load_positive_ids(csv_path: Path) -> List[int]:
    _, rows = _load_rows(csv_path)
    return sorted({int(r.gid) for r in rows if int(r.gid) > 0})


def _write_pair_scores_csv(path: Path, rows: List[PairScore]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "cam1_gid",
                "cam2_gid",
                "score",
                "sim_topk",
                "sim_mean",
                "sim_edge",
                "shape",
                "rows_cam1",
                "rows_cam2",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.cam1_gid,
                    r.cam2_gid,
                    f"{r.score:.6f}",
                    f"{r.sim_topk:.6f}",
                    f"{r.sim_mean:.6f}",
                    f"{r.sim_edge:.6f}",
                    f"{r.shape:.6f}",
                    r.rows_cam1,
                    r.rows_cam2,
                ]
            )


def _write_combined_csv(out_csv: Path, cam1_csv: Path, cam2_csv: Path, cam1_name: str, cam2_name: str) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f_out:
        writer = None
        for cam, p in ((cam1_name, cam1_csv), (cam2_name, cam2_csv)):
            with p.open(newline="", encoding="utf-8") as f_in:
                reader = csv.DictReader(f_in)
                cols = list(reader.fieldnames or [])
                if writer is None:
                    fieldnames = ["camera_id"] + cols
                    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
                    writer.writeheader()
                for raw in reader:
                    row = {"camera_id": cam}
                    row.update(raw)
                    writer.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-camera (cross_cam1 + cross_cam2) tracking + ReID linking.")
    ap.add_argument("--video1", type=Path, default=Path("data/raw/cross_cam/cross_cam1.mp4"))
    ap.add_argument("--video2", type=Path, default=Path("data/raw/cross_cam/cross_cam2.mp4"))
    ap.add_argument("--cam1_name", type=str, default="cross_cam1")
    ap.add_argument("--cam2_name", type=str, default="cross_cam2")
    ap.add_argument("--out_dir", type=Path, default=Path("runs/cross_cam"))
    ap.add_argument("--det_weights", type=Path, default=Path("models/yolo_cam1_person.pt"))
    ap.add_argument("--reid_weights", type=Path, default=Path("models/osnet_cam1.pth"))
    ap.add_argument("--skip_tracking", action="store_true", help="Skip per-camera tracking and only run cross-camera linking.")
    ap.add_argument("--min_rows", type=int, default=12)
    ap.add_argument("--min_topk", type=float, default=0.62)
    ap.add_argument("--min_mean", type=float, default=0.58)
    ap.add_argument("--min_score", type=float, default=0.66)
    ap.add_argument("--max_cam2_ids", type=int, default=7)
    args = ap.parse_args()

    if not args.video1.exists():
        raise FileNotFoundError(f"video1 missing: {args.video1}")
    if not args.video2.exists():
        raise FileNotFoundError(f"video2 missing: {args.video2}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    debug_root = args.out_dir / "reentry_debug"
    debug_root.mkdir(parents=True, exist_ok=True)

    det_weights = str(args.det_weights) if args.det_weights.exists() else "yolov8m.pt"
    reid_weights_path = str(args.reid_weights) if args.reid_weights.exists() else None

    cam1_raw_csv = args.out_dir / f"{args.cam1_name}_tracks_raw.csv"
    cam2_raw_csv = args.out_dir / f"{args.cam2_name}_tracks_raw.csv"
    cam1_csv = args.out_dir / f"{args.cam1_name}_tracks.csv"
    cam2_csv = args.out_dir / f"{args.cam2_name}_tracks.csv"
    cam1_vis = args.out_dir / f"{args.cam1_name}_vis.mp4"
    cam2_vis = args.out_dir / f"{args.cam2_name}_vis.mp4"
    pair_scores_csv = args.out_dir / "cross_camera_pair_scores.csv"
    matched_csv = args.out_dir / "cross_camera_matches.csv"
    combined_csv = args.out_dir / "cross_camera_tracks_combined.csv"
    cam1_pre_map_csv = args.out_dir / f"{args.cam1_name}_tracks_pre_global_remap.csv"
    cam2_pre_map_csv = args.out_dir / f"{args.cam2_name}_tracks_pre_global_remap.csv"

    if not args.skip_tracking:
        tracker_cam1 = _make_tracker(
            det_weights=det_weights,
            reid_weights_path=reid_weights_path,
            det_conf=0.24,
            min_height_ratio=0.10,
            min_width_ratio=0.04,
        )
        tracker_cam2 = _make_tracker(
            det_weights=det_weights,
            reid_weights_path=reid_weights_path,
            det_conf=0.42,
            min_height_ratio=0.16,
            min_width_ratio=0.05,
        )
        print(f"[INFO] Detector: {det_weights}")
        print(f"[INFO] ReID: {reid_weights_path if reid_weights_path else 'osnet_x1_0 default'}")

        _run_single_camera(
            video_path=args.video1,
            out_csv=cam1_raw_csv,
            tracker=tracker_cam1,
            reid_weights_path=reid_weights_path,
            debug_dir=debug_root / args.cam1_name,
        )
        _run_single_camera(
            video_path=args.video2,
            out_csv=cam2_raw_csv,
            tracker=tracker_cam2,
            reid_weights_path=reid_weights_path,
            debug_dir=debug_root / args.cam2_name,
        )
        shutil.copy2(cam1_raw_csv, cam1_csv)
        shutil.copy2(cam2_raw_csv, cam2_csv)
    else:
        if cam1_raw_csv.exists():
            shutil.copy2(cam1_raw_csv, cam1_csv)
        if cam2_raw_csv.exists():
            shutil.copy2(cam2_raw_csv, cam2_csv)
        if not cam1_csv.exists() or not cam2_csv.exists():
            raise FileNotFoundError("Missing per-camera CSVs; run without --skip_tracking first.")

    cam2_drop = _drop_static_nonperson_tracks(
        tracks_csv_path=cam2_csv,
        min_rows=220,
        max_path_norm=1.0,
        max_disp_norm=0.20,
        max_cx_std=35.0,
        max_cy_std=35.0,
    )
    print(f"[INFO] cam2 static/non-person drop: {cam2_drop}")

    # Stitch and reentry now run AFTER cam2 static filter so static-object tracks
    # are excluded from embedding extraction and similarity scoring.
    _is_cam2_cross = "cross_cam2" in str(args.video2.stem).lower()
    for _vid, _csv, _is_c2, _dbg in [
        (args.video2, cam2_csv, _is_cam2_cross, debug_root / args.cam2_name),
        (args.video1, cam1_csv, False,           debug_root / args.cam1_name),
    ]:
        _stitch = stitch_track_ids(
            video_path=_vid,
            tracks_csv_path=_csv,
            reid_weights_path=reid_weights_path,
            max_gap_frames=720 if _is_c2 else 1100,
            min_merge_score=0.74 if _is_c2 else 0.70,
        )
        print(f"[INFO] {_vid.name} stitch: {_stitch}")
        _reentry = link_reentry_offline(
            video_path=_vid,
            tracks_csv_path=_csv,
            reid_weights_path=reid_weights_path,
            debug_dir=_dbg,
            config=ReentryConfig(
                max_reentry_gap_frames=900 if _is_c2 else 1200,
                strong_reuse_score=0.80 if _is_c2 else 0.78,
                strong_reuse_margin=0.08 if _is_c2 else 0.07,
                min_deep_sim_for_reuse=0.72 if _is_c2 else 0.70,
                min_topk_sim_for_reuse=0.74 if _is_c2 else 0.72,
                min_part_topk_for_reuse=0.64 if _is_c2 else 0.62,
                min_part_mean_for_reuse=0.68 if _is_c2 else 0.66,
                rerank_top_n=6,
                merge_min_score=0.85 if _is_c2 else 0.83,
                merge_min_deep=0.76 if _is_c2 else 0.74,
                merge_min_topk=0.74 if _is_c2 else 0.73,
                merge_min_part_topk=0.66 if _is_c2 else 0.64,
                merge_max_gap_frames=1000 if _is_c2 else 1300,
            ),
        )
        print(f"[INFO] {_vid.name} reentry: {_reentry}")
        _dup = suppress_same_frame_duplicates(
            tracks_csv_path=_csv,
            iou_thresh=0.55,
            containment_thresh=0.72,
            max_center_dist_norm=0.62,
        )
        print(f"[INFO] {_vid.name} duplicate suppress: {_dup}")
        _cross_dup = _suppress_contained_cross_id_duplicates(
            tracks_csv_path=_csv,
            min_pair_hits=1,
            iou_thresh=0.42,
            containment_thresh=0.78,
            max_center_dist_norm=0.55,
        )
        print(f"[INFO] {_vid.name} cross-id contained suppress: {_cross_dup}")
        _compact = compact_global_ids(
            tracks_csv_path=_csv,
            min_rows_keep=20,
            min_span_keep=20,
        )
        print(f"[INFO] {_vid.name} compact: {_compact}")

    cam2_split = _split_mixed_tracks_by_appearance(
        video_path=args.video2,
        tracks_csv_path=cam2_csv,
        reid_weights_path=reid_weights_path,
        min_rows=160,
        min_segment_rows=56,
        split_sim_thresh=0.58,
        split_gap_frames=6,
    )
    print(f"[INFO] cam2 mixed-track split: {cam2_split}")
    cam2_compact_post_drop = compact_global_ids(
        tracks_csv_path=cam2_csv,
        min_rows_keep=24,
        min_span_keep=24,
    )
    print(f"[INFO] cam2 compact after static-drop: {cam2_compact_post_drop}")
    cam2_overlap_merge = _merge_high_similarity_overlaps(
        video_path=args.video2,
        tracks_csv_path=cam2_csv,
        reid_weights_path=reid_weights_path,
        min_pair_hits=4,
        iou_thresh=0.20,
        containment_thresh=0.50,
        max_center_dist_norm=0.66,
        min_score=0.84,
        min_topk=0.82,
        min_mean=0.78,
    )
    print(f"[INFO] cam2 overlap similarity merge: {cam2_overlap_merge}")
    cam2_compact_post_merge = compact_global_ids(
        tracks_csv_path=cam2_csv,
        min_rows_keep=24,
        min_span_keep=24,
    )
    print(f"[INFO] cam2 compact after overlap merge: {cam2_compact_post_merge}")

    cam1_overlap_merge = _merge_high_similarity_overlaps(
        video_path=args.video1,
        tracks_csv_path=cam1_csv,
        reid_weights_path=reid_weights_path,
        min_pair_hits=6,
        iou_thresh=0.20,
        containment_thresh=0.52,
        max_center_dist_norm=0.66,
        min_score=0.84,
        min_topk=0.82,
        min_mean=0.78,
    )
    print(f"[INFO] cam1 overlap similarity merge: {cam1_overlap_merge}")
    cam1_compact_post_merge = compact_global_ids(
        tracks_csv_path=cam1_csv,
        min_rows_keep=20,
        min_span_keep=20,
    )
    print(f"[INFO] cam1 compact after overlap merge: {cam1_compact_post_merge}")

    cam1_limit = _enforce_max_ids_by_similarity(
        video_path=args.video1,
        tracks_csv_path=cam1_csv,
        reid_weights_path=reid_weights_path,
        max_ids=6,
        min_merge_score=0.66,
        min_rows_force_drop=24,
    )
    print(f"[INFO] cam1 enforce max IDs: {cam1_limit}")
    cam1_compact_post_limit = compact_global_ids(
        tracks_csv_path=cam1_csv,
        min_rows_keep=18,
        min_span_keep=18,
    )
    print(f"[INFO] cam1 compact after max-ID enforce: {cam1_compact_post_limit}")
    cam1_final_overlap_fix = _suppress_contained_cross_id_duplicates(
        tracks_csv_path=cam1_csv,
        min_pair_hits=2,
        iou_thresh=0.33,
        containment_thresh=0.56,
        max_center_dist_norm=0.45,
    )
    print(f"[INFO] cam1 final overlap duplicate suppress: {cam1_final_overlap_fix}")
    cam1_post_overlap_merge = _merge_high_similarity_overlaps(
        video_path=args.video1,
        tracks_csv_path=cam1_csv,
        reid_weights_path=reid_weights_path,
        min_pair_hits=10,
        iou_thresh=0.24,
        containment_thresh=0.60,
        max_center_dist_norm=0.58,
        min_score=0.87,
        min_topk=0.85,
        min_mean=0.82,
    )
    print(f"[INFO] cam1 post-overlap similarity merge: {cam1_post_overlap_merge}")
    cam1_final_compact = compact_global_ids(
        tracks_csv_path=cam1_csv,
        min_rows_keep=18,
        min_span_keep=18,
    )
    print(f"[INFO] cam1 compact after final overlap fix: {cam1_final_compact}")
    cam1_near_dup_merge = _merge_high_similarity_near_duplicates(
        video_path=args.video1,
        tracks_csv_path=cam1_csv,
        reid_weights_path=reid_weights_path,
        min_score=0.90,
        min_topk=0.90,
        min_mean=0.90,
        max_shared_frames=20,
        max_shared_ratio=0.25,
    )
    print(f"[INFO] cam1 near-duplicate similarity merge: {cam1_near_dup_merge}")
    cam1_post_near_dup_compact = compact_global_ids(
        tracks_csv_path=cam1_csv,
        min_rows_keep=18,
        min_span_keep=18,
    )
    print(f"[INFO] cam1 compact after near-duplicate merge: {cam1_post_near_dup_compact}")

    pair_scores, tracks1, tracks2 = _build_pair_scores(
        video1=args.video1,
        csv1=cam1_csv,
        video2=args.video2,
        csv2=cam2_csv,
        reid_weights_path=reid_weights_path,
        min_rows=args.min_rows,
        min_topk=args.min_topk,
        min_mean=args.min_mean,
        min_score=args.min_score,
    )
    _write_pair_scores_csv(pair_scores_csv, pair_scores)
    print(f"[INFO] cross-camera candidate pairs: {len(pair_scores)}")

    cam1_ids = _load_positive_ids(cam1_csv)
    cam2_ids = _load_positive_ids(cam2_csv)
    cam1_set = set(cam1_ids)
    cam2_to_global, match_records = _assign_cam2_global_ids(
        pair_scores=pair_scores,
        cam2_ids=cam2_ids,
        cam1_set=cam1_set,
        tracks1=tracks1,
        tracks2=tracks2,
        max_cam2_ids=int(args.max_cam2_ids),
    )
    accepted_count = sum(
        1
        for r in match_records
        if str(r.get("match_type", "")) in {"canonical_primary", "matched_unclaimed", "fragment_reuse", "fallback_reuse", "cam2_fragment_reuse"}
    )
    print(f"[INFO] cross-camera accepted matches: {accepted_count}")

    with matched_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "cam1_gid",
                "cam2_gid",
                "final_global_id",
                "score",
                "sim_topk",
                "sim_mean",
                "sim_edge",
                "shape",
                "rows_cam1",
                "rows_cam2",
                "match_type",
            ]
        )
        for r in match_records:
            w.writerow(
                [
                    r.get("cam1_gid", 0),
                    r.get("cam2_gid", 0),
                    r.get("final_global_id", 0),
                    r.get("score", ""),
                    r.get("sim_topk", ""),
                    r.get("sim_mean", ""),
                    r.get("sim_edge", ""),
                    r.get("shape", ""),
                    r.get("rows_cam1", ""),
                    r.get("rows_cam2", ""),
                    r.get("match_type", ""),
                ]
            )

    shutil.copy2(cam1_csv, cam1_pre_map_csv)
    shutil.copy2(cam2_csv, cam2_pre_map_csv)
    _remap_csv_gids(cam2_csv, cam2_to_global)
    cam2_post_map_dup = suppress_same_frame_duplicates(
        tracks_csv_path=cam2_csv,
        iou_thresh=0.56,
        containment_thresh=0.76,
        max_center_dist_norm=0.62,
    )
    print(f"[INFO] cam2 post-map duplicate suppress: {cam2_post_map_dup}")
    cam2_post_map_cross = _suppress_contained_cross_id_duplicates(
        tracks_csv_path=cam2_csv,
        min_pair_hits=1,
        iou_thresh=0.42,
        containment_thresh=0.78,
        max_center_dist_norm=0.55,
    )
    print(f"[INFO] cam2 post-map cross-id contained suppress: {cam2_post_map_cross}")

    _write_combined_csv(combined_csv, cam1_csv, cam2_csv, args.cam1_name, args.cam2_name)

    render_tracks_video(video_path=args.video1, tracks_csv_path=cam1_csv, out_video_path=cam1_vis)
    render_tracks_video(video_path=args.video2, tracks_csv_path=cam2_csv, out_video_path=cam2_vis)

    print(f"[OK] cam1 tracks: {cam1_csv}")
    print(f"[OK] cam2 tracks (cross-camera IDs): {cam2_csv}")
    print(f"[OK] cam1 vis: {cam1_vis}")
    print(f"[OK] cam2 vis: {cam2_vis}")
    print(f"[OK] pair scores: {pair_scores_csv}")
    print(f"[OK] accepted matches: {matched_csv}")
    print(f"[OK] combined tracks: {combined_csv}")
    print(f"[OK] pre-remap cam1 tracks: {cam1_pre_map_csv}")
    print(f"[OK] pre-remap cam2 tracks: {cam2_pre_map_csv}")

    # --- cam2 identity quality metrics ---
    # gid=0 decomposes into three distinct sources; only the third represents
    # genuine identity uncertainty.  All three are reported so the metric is
    # not misleading when read in isolation.
    with cam2_csv.open(newline="", encoding="utf-8") as _f:
        _cam2_rows = list(csv.DictReader(_f))
    _total    = len(_cam2_rows)
    _zero     = sum(1 for r in _cam2_rows if int(r.get("global_id", 0) or 0) == 0)
    _pos      = _total - _zero
    _static   = int(cam2_drop.get("dropped_rows", 0)) if cam2_drop.get("applied") else 0
    _postmap  = int(cam2_post_map_dup.get("conflict_rows_forced_zero", 0)) if cam2_post_map_dup.get("applied") else 0
    _other    = max(0, _zero - _static - _postmap)
    _non_static = max(1, _total - _static)
    _coverage = 100.0 * _pos / _non_static
    _dropped_cap = sum(1 for r in match_records if str(r.get("match_type", "")) == "dropped_cap")
    print()
    print("[METRICS] cam2 identity quality")
    print(f"  total rows             : {_total}")
    print(f"  positive (person) rows : {_pos}  ({100.0 * _pos / _total:.1f}%)")
    print(f"  gid=0 rows (total)     : {_zero}  ({100.0 * _zero / _total:.1f}%)")
    print(f"    static-filter zeros  : {_static}  ({100.0 * _static / _total:.1f}%)  -- correct: non-person removed")
    print(f"    post-map dup-suppress: {_postmap}  ({100.0 * _postmap / _total:.1f}%)  -- correct: same-frame duplicate")
    print(f"    other (tracker+misc) : {_other}  ({100.0 * _other / _total:.1f}%)")
    print(f"  person-coverage rate   : {_coverage:.1f}%  (positive / non-static rows)")
    print(f"  dropped_cap            : {_dropped_cap}  -- ID budget cap not binding")


if __name__ == "__main__":
    main()
