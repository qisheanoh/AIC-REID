#!/usr/bin/env python3
# scripts/ingest_kpi_to_db.py
from __future__ import annotations

import argparse
import csv
import glob
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import sys
import re
import yaml

ROOT = Path(__file__).resolve().parents[1]  # .../video-reid
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.storage.db import (
    get_connection,
    init_schema,
    insert_events_bulk,
    insert_tracks_bulk,
)


def _slugify(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "cam"


def load_config(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def clip_id_from_csv(csv_path: Path) -> str:
    # runs/kpi_batch/person_01_1_2_crop_tracks.csv -> person_01_1_2_crop
    stem = csv_path.stem
    if stem.endswith("_tracks"):
        stem = stem[: -len("_tracks")]
    return stem


def delete_existing_clip(conn, camera_id: str, clip_id: str) -> None:
    """Avoid duplicates if you re-run ingestion (per camera_id+clip_id)."""
    cur = conn.cursor()
    cur.execute("DELETE FROM events WHERE camera_id=? AND clip_id=?;", (camera_id, clip_id))
    cur.execute("DELETE FROM tracks WHERE camera_id=? AND clip_id=?;", (camera_id, clip_id))


def delete_existing_camera(conn, camera_id: str) -> None:
    """Hard reset: delete ALL events+tracks for camera_id before ingesting any CSV."""
    cur = conn.cursor()
    cur.execute("DELETE FROM events WHERE camera_id=?;", (camera_id,))
    cur.execute("DELETE FROM tracks WHERE camera_id=?;", (camera_id,))


def summed_dwell(ts_sorted: List[float], gap_cap_s: float) -> float:
    """Sum consecutive gaps, but cap large gaps (occlusion / dropped detections)."""
    if len(ts_sorted) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(ts_sorted[:-1], ts_sorted[1:]):
        dt = float(b) - float(a)
        if dt <= 0:
            continue
        total += min(dt, float(gap_cap_s))
    return max(0.0, total)


def segment_visits(ts_list: List[float], session_gap_s: float) -> List[List[float]]:
    """Split timestamps into multiple visits if time gap > session_gap_s."""
    if not ts_list:
        return []
    ts_sorted = sorted(ts_list)
    segments: List[List[float]] = []
    curr: List[float] = [ts_sorted[0]]
    for t in ts_sorted[1:]:
        if (float(t) - float(curr[-1])) > float(session_gap_s):
            segments.append(curr)
            curr = [t]
        else:
            curr.append(t)
    segments.append(curr)
    return segments


def ingest_one_csv(
    conn,
    csv_path: Path,
    camera_id: str,
    *,
    overwrite_clip: bool,
    session_gap_s: float,
    gap_cap_s: float,
    min_dwell_s: float,
    require_zone: bool,
) -> None:
    csv_path = Path(csv_path)
    clip_id = clip_id_from_csv(csv_path)
    print(f"[INGEST] {csv_path.name} -> clip_id={clip_id}")

    if overwrite_clip:
        delete_existing_clip(conn, camera_id=camera_id, clip_id=clip_id)

    # (global_id, zone_id) -> timestamps for in-zone rows
    per_key_ts: Dict[Tuple[int, str], List[float]] = defaultdict(list)

    # schema: (ts, frame_idx, camera_id, clip_id, global_id, x1,y1,x2,y2, zone_id, conf)
    track_rows: List[
        Tuple[float, int, str, str, int, float, float, float, float, Optional[str], Optional[float]]
    ] = []

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                frame_idx = int(row["frame_idx"])
                ts = float(row["ts_sec"])
                gid = int(row["global_id"])
                x1 = float(row["x1"])
                y1 = float(row["y1"])
                x2 = float(row["x2"])
                y2 = float(row["y2"])
            except Exception:
                continue

            zid = (row.get("zone_id") or "").strip()

            # If require_zone, skip tracks that have no zone_id (keeps DB smaller/cleaner)
            if require_zone and not zid:
                continue

            track_rows.append((ts, frame_idx, camera_id, clip_id, gid, x1, y1, x2, y2, zid or None, None))

            if zid:
                per_key_ts[(gid, zid)].append(ts)

    insert_tracks_bulk(conn, track_rows)

    # Build events with segmentation + summed dwell
    events_payload: List[Tuple[str, str, int, str, float, float, float]] = []

    for (gid, zid), ts_list in per_key_ts.items():
        if not ts_list:
            continue

        segments = segment_visits(ts_list, session_gap_s=session_gap_s)

        for seg in segments:
            if len(seg) < 2:
                continue

            seg_sorted = sorted(seg)
            t_enter = float(seg_sorted[0])
            t_exit = float(seg_sorted[-1])

            dwell_sum = summed_dwell(seg_sorted, gap_cap_s=gap_cap_s)

            if dwell_sum < float(min_dwell_s):
                continue

            events_payload.append((camera_id, clip_id, gid, zid, t_enter, t_exit, float(dwell_sum)))

    insert_events_bulk(conn, events_payload)

    print(f"[INGEST]   tracks={len(track_rows)} events={len(events_payload)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)

    # Default behavior: overwrite per-clip (same as your current script)
    ap.add_argument("--overwrite", action="store_true", help="Delete existing rows for same camera_id+clip_id before insert")
    ap.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    ap.set_defaults(overwrite=True)

    # NEW: reset entire camera at start (strong reset)
    ap.add_argument("--reset-camera", action="store_true", help="Delete ALL events+tracks for camera_id before ingesting any CSVs")

    # NEW: optionally keep only zoned tracks
    ap.add_argument("--require-zone", action="store_true", help="Only store tracks rows that have non-empty zone_id")

    ap.add_argument("--session_gap_s", type=float, default=1.0)
    ap.add_argument("--gap_cap_s", type=float, default=0.20)
    ap.add_argument("--min_dwell_s", type=float, default=0.50)
    args = ap.parse_args()

    cfg = load_config(Path(args.config))

    # Resolve db path relative to repo root if needed
    db_rel = Path(cfg.get("db", {}).get("path", "data/retail.db"))
    db_path = db_rel if db_rel.is_absolute() else (ROOT / db_rel)

    camera_id = _slugify(cfg.get("camera_id", "cam1"))

    csv_glob = cfg.get("kpi_csv_glob", "runs/kpi_batch/person_*_tracks.csv")
    csv_paths = sorted(Path(p) for p in glob.glob(str(ROOT / csv_glob)))

    print("[INGEST] DB:", db_path)
    print("[INGEST] camera_id:", camera_id)
    print("[INGEST] csv_glob:", csv_glob)
    print("[INGEST] found:", len(csv_paths))
    for p in csv_paths:
        try:
            print("  -", p.relative_to(ROOT))
        except Exception:
            print("  -", p)

    if not csv_paths:
        print("[INGEST] No CSVs found. Run run_kpi_batch.py first.")
        return

    conn = get_connection(db_path)
    init_schema(conn)

    try:
        if args.reset_camera:
            print(f"[INGEST] Resetting ALL rows for camera_id={camera_id}")
            delete_existing_camera(conn, camera_id=camera_id)

        for p in csv_paths:
            ingest_one_csv(
                conn,
                p,
                camera_id=camera_id,
                overwrite_clip=bool(args.overwrite),
                session_gap_s=args.session_gap_s,
                gap_cap_s=args.gap_cap_s,
                min_dwell_s=args.min_dwell_s,
                require_zone=bool(args.require_zone),
            )
        conn.commit()  # ✅ one commit for whole ingestion batch
    finally:
        conn.close()

    print("[INGEST] Done.")


if __name__ == "__main__":
    main()
