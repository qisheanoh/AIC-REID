# src/pipeline/run_stream.py
from __future__ import annotations

from pathlib import Path
import sys
import time
from typing import Dict, Any, Optional, Tuple, List
import warnings

import yaml
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]  # .../video-reid
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.trackers.bot_sort import BOTSORT
from src.trackers.id_bank import GlobalIDBank
from src.storage.db import get_connection, init_schema, upsert_zone, insert_track, delete_clip_data
from src.analytics.kpi_engine import refresh_events_from_tracks
from src.privacy.redact import blur_boxes


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_video_source(cfg_value: str) -> str:
    """If it looks like a URL/stream, keep it. Otherwise resolve relative to repo ROOT."""
    if "://" in cfg_value:
        return cfg_value
    return str((ROOT / cfg_value).resolve())


def resolve_repo_path(cfg_value: str) -> Path:
    """
    Resolve config path relative to repo root.
    Backward-compatible fallback:
    - "configs/zones_cam1.yaml" -> "configs/zones/zones_cam1.yaml" if needed
    """
    raw = Path(cfg_value)
    if raw.is_absolute() and raw.exists():
        return raw

    primary = (ROOT / raw).resolve()
    if primary.exists():
        return primary

    if raw.parts[:1] == ("configs",) and len(raw.parts) >= 2:
        fallback = (ROOT / "configs" / "zones" / raw.name).resolve()
        if fallback.exists():
            warnings.warn(
                f"Config path '{cfg_value}' does not exist. Using fallback '{fallback}'.",
                stacklevel=2,
            )
            return fallback

    return primary


def infer_clip_id(video_source: str, camera_id: str) -> str:
    """clip_id used for DB grouping."""
    if "://" in video_source:
        return "stream"
    try:
        p = Path(video_source)
        return p.stem if p.stem else str(camera_id)
    except Exception:
        return str(camera_id)


def load_zones(zones_cfg_path: Path):
    """
    Loads zones YAML:
      camera_id: cam1
      zones:
        - zone_id: entrance
          name: Entrance
          polygon: [[x,y], [x,y], ...]
    """
    with open(zones_cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    camera_id = cfg["camera_id"]
    zones = []
    for z in cfg.get("zones", []):
        poly = [(float(x), float(y)) for x, y in z["polygon"]]
        if len(poly) < 3:
            raise ValueError(f"Zone {z.get('zone_id')} polygon must have >= 3 points")
        zones.append((
            camera_id,
            str(z["zone_id"]),
            str(z.get("name", z["zone_id"])),
            poly,
            str(z.get("zone_type", "other")),
        ))
    return camera_id, zones


def point_in_polygon(x: float, y: float, polygon) -> bool:
    """Ray casting point-in-polygon. polygon: list[(x,y)] with len>=3"""
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_at_y = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
            if x < x_at_y:
                inside = not inside
    return inside


def pick_membership_point(x1: float, y1: float, x2: float, y2: float, mode: str = "ankle") -> Tuple[float, float]:
    """
    mode:
      - center: (cx, cy)
      - foot:   (cx, y2)
      - ankle:  (cx, y2 - 2.0)
    """
    cx = 0.5 * (x1 + x2)
    m = (mode or "ankle").lower()
    if m == "center":
        cy = 0.5 * (y1 + y2)
    elif m == "foot":
        cy = y2
    else:
        cy = y2 - 2.0
    return cx, cy


def assign_zone(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    zone_polys: Dict[str, list],
    point_mode: str = "ankle",
) -> Optional[str]:
    px, py = pick_membership_point(x1, y1, x2, y2, mode=point_mode)
    for zid, poly in zone_polys.items():
        if point_in_polygon(px, py, poly):
            return zid
    return None


def _zone_style(zone_id: str, zone_type: str) -> Dict[str, Any]:
    zid = str(zone_id).lower()
    zt = str(zone_type).lower()

    def has(*tokens: str) -> bool:
        return any(t in zt or t in zid for t in tokens)

    # BGR colors for OpenCV
    if has("entrance"):
        return {"edge": (40, 190, 70), "fill": (65, 160, 70)}
    if has("walkway", "aisle", "decision"):
        return {"edge": (35, 220, 240), "fill": (30, 150, 175)}
    if has("hot", "promo", "engage", "brows"):
        return {"edge": (35, 120, 255), "fill": (30, 90, 185)}
    if has("cashier", "checkout", "counter"):
        return {"edge": (205, 95, 200), "fill": (150, 70, 145)}
    return {"edge": (245, 120, 45), "fill": (160, 95, 35)}


def _rect_intersects(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def _draw_zone_overlays(
    frame: np.ndarray,
    zone_defs: List[Dict[str, Any]],
    occupied_boxes: List[Tuple[float, float, float, float]],
) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()

    for z in zone_defs:
        poly = z["poly"]
        if len(poly) < 3:
            continue
        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        style = _zone_style(z["zone_id"], z["zone_type"])
        cv2.fillPoly(overlay, [pts], style["fill"])

    cv2.addWeighted(overlay, 0.22, frame, 0.78, 0.0, frame)

    occ = [(int(x1), int(y1), int(x2), int(y2)) for (x1, y1, x2, y2) in occupied_boxes]
    for z in zone_defs:
        poly = z["poly"]
        if len(poly) < 3:
            continue
        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        style = _zone_style(z["zone_id"], z["zone_type"])
        cv2.polylines(frame, [pts], True, style["edge"], 4, cv2.LINE_AA)

        xs = [int(p[0]) for p in poly]
        ys = [int(p[1]) for p in poly]
        cx = int(sum(xs) / max(1, len(xs)))
        min_y = min(ys)
        max_y = max(ys)

        label = str(z["name"] or z["zone_id"])
        fs = 0.62
        th = 2
        (tw, tht), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        pad_x, pad_y = 8, 6
        bw = tw + pad_x * 2
        bh = tht + pad_y * 2

        candidates = []
        candidates.append((int(np.clip(cx - bw // 2, 4, w - bw - 4)), max(4, min_y - bh - 10)))  # above polygon
        candidates.append((int(np.clip(cx - bw // 2, 4, w - bw - 4)), min(h - bh - 4, max_y + 10)))  # below polygon
        candidates.append((max(4, min(xs) - bw - 10), int(np.clip(min_y - bh // 2, 4, h - bh - 4))))  # left side

        bx, by = candidates[0]
        for cand in candidates:
            cbx, cby = cand
            rect = (cbx, cby, cbx + bw, cby + bh)
            if not any(_rect_intersects(rect, o) for o in occ):
                bx, by = cbx, cby
                break

        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), style["edge"], -1, cv2.LINE_AA)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (18, 24, 32), 2, cv2.LINE_AA)
        tx = bx + pad_x
        ty = by + bh - pad_y - 2
        cv2.putText(frame, label, (tx + 1, ty + 1), cv2.FONT_HERSHEY_SIMPLEX, fs, (18, 24, 32), th + 1, cv2.LINE_AA)
        cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th, cv2.LINE_AA)


def open_video(video_source: str) -> cv2.VideoCapture:
    """Try CAP_FFMPEG first, then fall back to default backend."""
    cap = cv2.VideoCapture(video_source, apiPreference=cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(video_source)
    return cap


def main(config_path: str = "configs/cameras/retail_cam1.yaml"):
    cfg = load_config(Path(config_path))

    cam_id = cfg["camera_id"]
    video_source = resolve_video_source(cfg["video_source"])
    fps_override = cfg.get("fps_override")

    clip_id = infer_clip_id(video_source, cam_id)

    # ---- DB ----
    db_path = (ROOT / cfg["db"]["path"]).resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = get_connection(db_path)
    init_schema(conn)

    if bool(cfg.get("overwrite_clip", True)):
        delete_clip_data(conn, camera_id=cam_id, clip_id=clip_id)
        conn.commit()

    # ---- zones ----
    zones_cfg_path = resolve_repo_path(cfg["zones_config"])
    camera_id_z, zones = load_zones(zones_cfg_path)
    if camera_id_z != cam_id:
        raise ValueError(f"zones camera_id={camera_id_z} != cfg camera_id={cam_id}")

    zone_polys: Dict[str, list] = {}
    zone_defs: List[Dict[str, Any]] = []
    for camera_id, zone_id, name, poly, zone_type in zones:
        upsert_zone(conn, camera_id, zone_id, name, poly, zone_type=zone_type)
        zone_polys[zone_id] = poly
        zone_defs.append({
            "zone_id": zone_id,
            "name": name,
            "poly": poly,
            "zone_type": zone_type,
        })
    conn.commit()

    # zone membership point mode (default keeps your previous behavior)
    zone_point_mode = cfg.get("zones", {}).get("point", "ankle")

    # ---- tracker ----
    det_cfg = cfg["detector"]
    trk_cfg = cfg["tracker"]
    reid_cfg = cfg["reid"]
    id_bank_cfg = cfg.get("id_bank", {})
    privacy_cfg = cfg.get("privacy", {})

    device = "cuda" if torch.cuda.is_available() else "cpu"

    match_iou_thresh = float(trk_cfg.get("match_iou_thresh", 0.22))
    active_match_iou_thresh = float(trk_cfg.get("active_match_iou_thresh", match_iou_thresh))
    lost_match_iou_thresh = float(trk_cfg.get("lost_match_iou_thresh", 0.10))

    id_bank = GlobalIDBank(
        hard_thresh=float(id_bank_cfg.get("hard_thresh", 0.82)),
        soft_thresh=float(id_bank_cfg.get("soft_thresh", 0.75)),
        margin=float(id_bank_cfg.get("margin", 0.06)),
        ema=float(id_bank_cfg.get("ema", 0.92)),
        min_update_sim=float(id_bank_cfg.get("min_update_sim", 0.80)),
        enroll_reuse_thresh=id_bank_cfg.get("enroll_reuse_thresh", None),
        enroll_protect_states=int(id_bank_cfg.get("enroll_protect_states", 0)),
        verbose=bool(id_bank_cfg.get("verbose", False)),
    )

    min_match_conf_cfg = trk_cfg.get("min_match_conf", 0.30)
    min_match_conf = None if min_match_conf_cfg is None else float(min_match_conf_cfg)

    tracker = BOTSORT(
        det_weights=det_cfg["weights"],
        device=device,
        reid=bool(reid_cfg["enabled"]),
        reid_model_name=reid_cfg["model_name"],
        reid_weights_path=reid_cfg["weights_path"],
        det_imgsz=int(det_cfg["imgsz"]),
        det_conf=float(det_cfg["conf"]),
        det_iou=float(det_cfg["iou"]),
        det_classes=tuple(det_cfg.get("classes", [0])),
        track_thresh=float(trk_cfg["track_thresh"]),
        active_match_iou_thresh=active_match_iou_thresh,
        lost_match_iou_thresh=lost_match_iou_thresh,
        match_feat_thresh=float(trk_cfg.get("match_feat_thresh", 0.46)),
        track_buffer=int(trk_cfg["track_buffer"]),
        id_bank=id_bank,
        strong_reid_thresh=float(trk_cfg.get("strong_reid_thresh", 0.80)),
        long_lost_reid_thresh=float(trk_cfg.get("long_lost_reid_thresh", 0.86)),
        alpha_active=float(trk_cfg.get("alpha_active", 0.35)),
        alpha_lost=float(trk_cfg.get("alpha_lost", 0.90)),
        motion_max_center_dist=float(trk_cfg.get("motion_max_center_dist", 0.70)),
        motion_max_gap=int(trk_cfg.get("motion_max_gap", 24)),
        overlap_iou_thresh=float(trk_cfg.get("overlap_iou_thresh", 0.30)),
        min_height_ratio_for_update=float(trk_cfg.get("min_height_ratio_for_update", 0.72)),
        min_match_conf=min_match_conf,
        feature_history=int(trk_cfg.get("feature_history", 40)),
        feature_update_min_sim=float(trk_cfg.get("feature_update_min_sim", 0.66)),
        confirm_hits=int(trk_cfg.get("confirm_hits", 4)),
        bad_frame_hold=int(trk_cfg.get("bad_frame_hold", 12)),
        min_confirmed_hits_for_gid=int(trk_cfg.get("min_confirmed_hits_for_gid", 4)),
        min_height_ratio=float(trk_cfg.get("min_height_ratio", 0.10)),
        min_width_ratio=float(trk_cfg.get("min_width_ratio", 0.05)),
        F_OS=float(reid_cfg.get("f_os", 1.00)),
        F_ATTIRE=float(reid_cfg.get("f_attire", 0.55)),
        F_SHAPE=float(reid_cfg.get("f_shape", 0.22)),
    )

    # ---- video ----
    cap = open_video(video_source)
    if not cap.isOpened():
        print(f"[ERROR] cannot open source: {video_source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if (not fps) or np.isnan(fps) or fps <= 0:
        fps = float(fps_override) if fps_override is not None else 25.0
    elif fps_override is not None:
        fps = float(fps_override)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    print(f"[INFO] {video_source}: {w}x{h} @ {fps:.2f}fps | device={device} | clip_id={clip_id} | zone_point={zone_point_mode}")

    store_frames = bool(privacy_cfg.get("store_frames", False))
    redact = bool(privacy_cfg.get("redact", False))

    out_writer = None
    if store_frames:
        out_path = ROOT / "runs" / "stream" / f"{cam_id}_{clip_id}_stream.mp4"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        print(f"[INFO] saving stream to {out_path}")

    frame_idx = 0
    n_frames = 0
    t0 = time.time()
    commit_every = int(cfg.get("commit_every", 200))
    max_frames = int(cfg.get("max_frames", 0))

    while True:
        if max_frames > 0 and frame_idx >= max_frames:
            break

        ok, frame = cap.read()
        if not ok:
            break

        ts = frame_idx / fps

        # ✅ pass true seconds into tracker
        outputs = tracker.update(frame, frame_id=frame_idx, ts_sec=ts, return_raw_tracks=False)
        _draw_zone_overlays(frame, zone_defs=zone_defs, occupied_boxes=[(x1, y1, x2, y2) for (x1, y1, x2, y2, _) in outputs])

        boxes_for_blur = []
        for x1, y1, x2, y2, gid in outputs:
            zid = assign_zone(x1, y1, x2, y2, zone_polys, point_mode=zone_point_mode)

            insert_track(
                conn,
                ts=float(ts),
                frame_idx=int(frame_idx),
                camera_id=str(cam_id),
                clip_id=str(clip_id),          # ✅ REQUIRED BY SCHEMA
                global_id=int(gid),
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                zone_id=zid,
                conf=None,
            )

            boxes_for_blur.append((x1, y1, x2, y2))

            # draw for visualisation (not stored)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID {int(gid)}",
                (int(x1), max(0, int(y1) - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        if redact and boxes_for_blur:
            frame = blur_boxes(frame, boxes_for_blur)

        if out_writer is not None:
            out_writer.write(frame)

        n_frames += 1
        frame_idx += 1

        if commit_every > 0 and (frame_idx % commit_every == 0):
            conn.commit()

    cap.release()
    if out_writer is not None:
        out_writer.release()

    conn.commit()

    elapsed = time.time() - t0
    fps_measured = n_frames / elapsed if elapsed > 0 else 0.0
    print(f"[INFO] processed {n_frames} frames in {elapsed:.2f}s ({fps_measured:.2f} fps)")

    # build events for KPI (for this camera+clip)
    print("[INFO] refreshing events from tracks for KPI ...")
    refresh_events_from_tracks(
        conn,
        camera_id=cam_id,
        clip_id=clip_id,
        min_hold_s=float(cfg.get("kpi_min_hold_s", 0.0)),
        overwrite=True,
    )
    conn.commit()
    print("[INFO] done.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/cameras/retail_cam1.yaml")
    args = parser.parse_args()
    main(args.config)
