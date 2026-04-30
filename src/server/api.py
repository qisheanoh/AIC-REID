from __future__ import annotations

from pathlib import Path
import os
import sys
import sqlite3
import logging
import subprocess
import threading
import time
import uuid
from dataclasses import asdict
from typing import Any, Dict, Optional, List, Tuple
from contextlib import asynccontextmanager

import shutil

from fastapi import FastAPI, HTTPException, Body, Query, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import csv
import re
import cv2
import yaml
import json

from src.server.insight import summarize_kpis, ab_insights
from src.analytics.kpi_engine import refresh_events_from_tracks, clear_events
from src.storage.db import get_connection, init_schema, upsert_zone, list_zones, delete_zone, insert_tracks_bulk
from src.preprocessing.pipeline import preprocess_video

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = Path(os.environ.get("RETAIL_DB_PATH", str(ROOT / "data" / "retail.db")))
RAW_ROOT = ROOT / "data" / "raw"
RUNS_ROOT = ROOT / "runs"
CONFIGS_DIR = Path(os.environ.get("RETAIL_CONFIGS_DIR", str(ROOT / "configs")))

# Ordered list of run output directories searched by _find_preprocess_report.
_PREPROCESS_REPORT_SEARCH_DIRS: List[Path] = [
    RUNS_ROOT / "kpi_batch",
    RUNS_ROOT / "cross_cam",
]
TEMPLATES_DIR = ROOT / "src" / "server" / "templates"
logger = logging.getLogger(__name__)

SYNC_ZONES_FROM_YAML = os.environ.get("SYNC_ZONES_FROM_YAML", "1") == "1"

# In-memory pipeline job registry.  Keyed by job_id.  Cleared on server restart.
_JOBS: Dict[str, Any] = {}
_JOBS_LOCK = threading.Lock()


def _mount_static_if_exists(app: FastAPI) -> Optional[Path]:
    candidates = [
        ROOT / "server" / "serve" / "static",
        ROOT / "src" / "server" / "serve" / "static",
        ROOT / "src" / "server" / "static",
        ROOT / "static",
    ]
    for d in candidates:
        if d.exists() and d.is_dir():
            app.mount("/static", StaticFiles(directory=str(d)), name="static")
            return d
    return None


templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _slugify(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "zone"


def _norm_camera_id(s: str) -> str:
    return _slugify(s)


def _clean_clip_id(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s2 = str(s).strip()
    return s2 if s2 else None


def _clean_group(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s2 = str(s).strip()
    return s2 if s2 else None


def _clip_id_to_path(clip_id: str) -> Path:
    clip_id = str(clip_id).strip()

    alias_map: Dict[str, Path] = {
        "cross_cam1": RAW_ROOT / "cross_cam" / "cross_cam1.mp4",
        "cross_cam2": RAW_ROOT / "cross_cam" / "cross_cam2.mp4",
    }
    if clip_id in alias_map:
        return alias_map[clip_id]

    if "_tracks_" in clip_id:
        clip_id = clip_id.split("_tracks_")[0]

    parts = [p for p in clip_id.split("_") if p]
    if len(parts) >= 2:
        # Resolve by trying all possible group/file splits.
        # This supports both:
        # - retail-shop_CAM1       -> raw/retail-shop/CAM1.mp4
        # - retail-shop_FULL_CAM1  -> raw/retail-shop/FULL_CAM1.mp4
        # - person_01_1_2_crop     -> raw/person_01/1_2_crop.mp4
        for i in range(1, len(parts)):
            group = "_".join(parts[:i])
            rest = "_".join(parts[i:])
            if not group or not rest:
                continue
            cand = RAW_ROOT / group / f"{rest}.mp4"
            if cand.exists():
                return cand

        # Fallback to previous naming convention when file is not present yet.
        if len(parts) == 2:
            return RAW_ROOT / parts[0] / f"{parts[1]}.mp4"
        group = "_".join(parts[:2])
        rest = "_".join(parts[2:])
        return RAW_ROOT / group / f"{rest}.mp4"

    raise HTTPException(400, f"Invalid clip_id format: {clip_id}")


def _poly_to_yaml_safe(poly: Any) -> List[List[float]]:
    out: List[List[float]] = []
    if not poly:
        return out
    for p in poly:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            out.append([float(p[0]), float(p[1])])
        elif isinstance(p, dict) and "x" in p and "y" in p:
            out.append([float(p["x"]), float(p["y"])])
    return out


def _yaml_poly_to_pts(poly: Any) -> Optional[List[Tuple[float, float]]]:
    if not isinstance(poly, list) or len(poly) < 3:
        return None
    pts: List[Tuple[float, float]] = []
    for p in poly:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            pts.append((float(p[0]), float(p[1])))
        elif isinstance(p, dict) and "x" in p and "y" in p:
            pts.append((float(p["x"]), float(p["y"])))
        else:
            return None
    return pts


ZONE_ALIAS_MAP: Dict[str, str] = {
    "check_zone": "decision_zone",
    "decision_zone": "decision_zone",
    "counter_zone": "cashier_zone",
    "cashier_zone": "cashier_zone",
}


ZONE_ENGAGEMENT_THRESHOLDS: Dict[str, float] = {
    "entrance_zone": 2.0,
    "walkway_zone": 1.5,
    "decision_zone": 999.0,
    "hot_zone": 3.0,
    "promo_zone": 3.0,
    "browsing_zone": 2.0,
    "cashier_zone": 2.0,
}

DASHBOARD_CAMERA_MODES: Tuple[str, ...] = ("cam1", "cam1_hot", "cross_cam")
# Canonical raw sources used by Zone Editor reference-frame loading.
CAMERA_RAW_SOURCE_CLIPS: Dict[str, str] = {
    "cam1": "retail-shop_CAM1",
    "cam1_hot": "retail-shop_CAM1",
    "full_cam1": "retail-shop_FULL_CAM1",  # internal/back-compat only
    "cross_cam1": "cross_cam1",
    "cross_cam2": "cross_cam2",
}
ZONE_EDITOR_BASE_SOURCES: Dict[str, List[str]] = {
    "cam1": [CAMERA_RAW_SOURCE_CLIPS["cam1"]],
    "cam1_hot": [CAMERA_RAW_SOURCE_CLIPS["cam1_hot"]],
}
FRESHNESS_EVENT_ID_RATIO_THRESHOLDS: Dict[str, float] = {
    "cam1": 0.50,
    "cam1_hot": 0.50,
    "cross_cam": 0.25,
}
RECOMPUTE_MIN_HOLD_S: float = 0.50

# Camera bucket used when auto-ingesting uploaded-clip tracks into the DB.
_UPLOADED_CAMERA_ID = "uploaded"
# Canonical clips already ingested under their own camera_ids — skip auto-ingest for these.
_CANONICAL_CLIP_IDS: frozenset = frozenset({"retail-shop_CAM1", "retail-shop_FULL_CAM1"})


def _normalize_zone_alias(zone_id: str) -> str:
    z = _slugify(zone_id)
    return ZONE_ALIAS_MAP.get(z, z)


def _normalize_zone_type(zone_id: str, zone_type: Optional[str]) -> str:
    zt = _slugify(zone_type or "other")
    if zt and zt != "other":
        return zt

    zid = _normalize_zone_alias(zone_id)
    if "entrance" in zid:
        return "entrance"
    if "walkway" in zid or "aisle" in zid:
        return "walkway"
    if "hot" in zid:
        return "hot"
    if "promo" in zid:
        return "promo"
    if "browse" in zid:
        return "browsing"
    if "cashier" in zid or "checkout" in zid or "counter" in zid:
        return "checkout"
    return "other"


def _engagement_threshold(zone_id: str) -> float:
    z = _normalize_zone_alias(zone_id)
    if z in ZONE_ENGAGEMENT_THRESHOLDS:
        return ZONE_ENGAGEMENT_THRESHOLDS[z]

    if "walkway" in z:
        return 999.0
    if "decision" in z:
        return 999.0
    if "entrance" in z:
        return 2.0
    if "hot" in z or "promo" in z:
        return 3.0
    if "browse" in z:
        return 2.0
    if "cashier" in z or "checkout" in z:
        return 2.0
    return 2.5


def _zone_kind(zone_id: str) -> str:
    z = _normalize_zone_alias(zone_id)
    if "walkway" in z or "decision" in z:
        return "flow"
    if "entrance" in z:
        return "entrance"
    if "hot" in z or "promo" in z or "browse" in z:
        return "engagement"
    if "cashier" in z or "checkout" in z:
        return "checkout"
    return "other"


def _point_in_poly(x: float, y: float, poly: List[Tuple[float, float]]) -> bool:
    inside = False
    n = len(poly)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        inter = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
        if inter:
            inside = not inside
        j = i
    return inside


def _bbox_point(x1: float, y1: float, x2: float, y2: float, mode: str) -> Tuple[float, float]:
    cx = 0.5 * (x1 + x2)
    m = str(mode).lower()
    if m == "center":
        cy = 0.5 * (y1 + y2)
    elif m == "foot":
        cy = y2
    else:
        cy = y2 - 2.0
    return cx, cy


def _resolve_recompute_scope(camera_id: str, clip_id: Optional[str], available_clips: List[str]) -> Tuple[str, str, str, str]:
    cam = _norm_camera_id(camera_id)
    clip = _clean_clip_id(clip_id)

    mapping = {
        "cam1": ("cam1", "cam1", "ankle"),
        "cam1_hot": ("cam1_hot", "cam1_hot", "center"),
        # cross_cam1/2 use their own zone definitions, but analytics tracks/events
        # are stored under cross_cam dashboard clips.
        "cross_cam1": ("cross_cam", "cross_cam1", "center"),
        "cross_cam2": ("cross_cam", "cross_cam2", "center"),
        "cross_cam": ("cross_cam", "cross_cam", "center"),
        "uploaded": ("uploaded", "uploaded", "center"),
    }
    if cam not in mapping:
        raise HTTPException(400, f"Unsupported recompute camera_id: {cam}")

    target_cam, zone_cam, point_mode = mapping[cam]

    if not clip:
        raise HTTPException(400, "clip_id required for recompute. Select a clip in Zone Editor.")

    if clip in available_clips:
        return target_cam, zone_cam, point_mode, clip

    # Keep CAM1 and CAM1_HOT locked to the same canonical raw-source clip ID.
    # Zone Editor should not drift to derived clip IDs for these two modes.
    if target_cam in ("cam1", "cam1_hot"):
        canonical = CAMERA_RAW_SOURCE_CLIPS.get("cam1", "retail-shop_CAM1")
        if clip == canonical:
            if canonical in available_clips:
                return target_cam, zone_cam, point_mode, canonical
            raise HTTPException(
                400,
                f"Canonical CAM1 clip_id={canonical} is missing for camera_id={target_cam}. "
                f"Available clips: {available_clips}",
            )

    # Deterministic aliasing for datasets where zone-editor raw clip ids and
    # ingested analytics clip ids differ but represent the same source.
    clip_aliases: Dict[str, List[str]] = {
        "cross_cam1": ["dashboard_cross_cam1", "cross_cam_cross_cam1"],
        "cross_cam2": ["dashboard_cross_cam2", "cross_cam_cross_cam2"],
    }
    clip_norm = _slugify(clip)
    avail_norm_map = {_slugify(c): c for c in available_clips}
    for alias_norm in clip_aliases.get(clip_norm, []):
        if alias_norm in avail_norm_map:
            return target_cam, zone_cam, point_mode, avail_norm_map[alias_norm]

    raise HTTPException(
        400,
        f"Selected clip_id={clip} is not available for camera_id={target_cam}. Available clips: {available_clips}",
    )


def _all_camera_ids(conn: sqlite3.Connection) -> List[str]:
    rows = conn.execute("""
        SELECT camera_id FROM (
          SELECT DISTINCT camera_id FROM tracks
          UNION
          SELECT DISTINCT camera_id FROM events
        )
        ORDER BY camera_id;
    """).fetchall()
    return [str(r["camera_id"]) for r in rows if r["camera_id"]]


def _clip_has_reference_frame(clip_id: str) -> bool:
    try:
        p = _clip_id_to_path(clip_id)
    except HTTPException:
        return False
    return p.exists() and p.is_file()


def _find_preprocess_report(clip_id: str) -> Optional[Path]:
    """Return the path to {stem}_preprocess_report.json for a clip, or None.

    preprocess_video() names the JSON after the *video file* stem (e.g. "CAM1"),
    not the clip_id (e.g. "retail-shop_CAM1").  We therefore resolve the clip_id
    through _clip_id_to_path() to get the matching video stem before searching.
    """
    stem = str(clip_id).strip()
    if "_tracks_" in stem:
        stem = stem.split("_tracks_")[0]

    # Resolve to the video file stem so we match what preprocess_video() wrote.
    try:
        video_stem = _clip_id_to_path(stem).stem
    except HTTPException:
        video_stem = stem  # fallback: clip_id has no resolvable path

    for d in _PREPROCESS_REPORT_SEARCH_DIRS:
        p = d / f"{video_stem}_preprocess_report.json"
        if p.is_file():
            return p
    return None


def _out_dir_for_clip(clip_id: str) -> Path:
    """Map a clip_id to the runs output directory where its preprocess report belongs."""
    stem = str(clip_id).strip()
    if "_tracks_" in stem:
        stem = stem.split("_tracks_")[0]
    if stem.startswith("cross_cam"):
        return RUNS_ROOT / "cross_cam"
    return RUNS_ROOT / "kpi_batch"


def _tracks_csv_for_clip(clip_id: str) -> Path:
    """Return the expected path of the final processed tracks CSV for a clip."""
    return _out_dir_for_clip(clip_id) / f"{clip_id}_tracks.csv"


def _script_and_args_for_clip(clip_id: str) -> Tuple[Path, List[str]]:
    """
    Return (script_path, extra_cli_args) for the given clip_id.

    - cross_cam*        → run_cross_cam.py  (no extra args; defaults cover both cameras)
    - retail-shop_*     → run_batch.py --match {video_stem}
    """
    stem = str(clip_id).strip()
    if "_tracks_" in stem:
        stem = stem.split("_tracks_")[0]
    if stem.startswith("cross_cam"):
        return ROOT / "scripts" / "run_cross_cam.py", []
    try:
        match_token = _clip_id_to_path(stem).stem   # e.g. "FULL_CAM1" or "CAM1"
    except HTTPException as exc:
        raise HTTPException(400, f"Cannot determine script for clip_id '{clip_id}': {exc.detail}") from exc
    return ROOT / "scripts" / "run_batch.py", ["--match", match_token]


def _ingest_tracks_for_uploaded_clip(clip_id: str) -> None:
    """Ingest a processed tracks CSV into the DB under camera_id='uploaded'.

    Skipped for canonical clips that are already ingested under their own
    camera_ids (cam1, cam1_hot, etc.) and for cross_cam clips.
    Errors are logged but never raised — ingestion failure must not change job status.
    """
    if not clip_id or clip_id in _CANONICAL_CLIP_IDS or clip_id.startswith("cross_cam"):
        return
    csv_path = _tracks_csv_for_clip(clip_id)
    if not csv_path.exists():
        logger.warning("Auto-ingest skipped: CSV not found at %s", csv_path)
        return
    try:
        conn = get_connection(DB_PATH)
        init_schema(conn)
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM tracks WHERE camera_id=? AND clip_id=?;",
            (_UPLOADED_CAMERA_ID, clip_id),
        )
        cur.execute(
            "DELETE FROM events WHERE camera_id=? AND clip_id=?;",
            (_UPLOADED_CAMERA_ID, clip_id),
        )
        track_rows: List[Tuple] = []
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    gid = int(row["global_id"])
                    if gid <= 0:
                        continue
                    zid = (row.get("zone_id") or "").strip() or None
                    track_rows.append((
                        float(row["ts_sec"]),
                        int(row["frame_idx"]),
                        _UPLOADED_CAMERA_ID,
                        clip_id,
                        gid,
                        float(row["x1"]),
                        float(row["y1"]),
                        float(row["x2"]),
                        float(row["y2"]),
                        zid,
                        None,
                    ))
                except Exception:
                    continue
        insert_tracks_bulk(conn, track_rows)
        conn.commit()
        conn.close()
        logger.info(
            "Auto-ingested %d track rows for clip_id=%s under camera_id=%s",
            len(track_rows), clip_id, _UPLOADED_CAMERA_ID,
        )
    except Exception as exc:
        logger.error("Auto-ingest failed for clip_id=%s: %s", clip_id, exc)


def _run_job(job_id: str, cmd: List[str], log_path: Path, clip_id: str = "") -> None:
    """Background thread: run pipeline subprocess, write log, update _JOBS."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(log_path, "w", encoding="utf-8", buffering=1) as lf:
            proc = subprocess.Popen(
                cmd,
                stdout=lf,
                stderr=subprocess.STDOUT,
                cwd=str(ROOT),
            )
            proc.wait()
        rc = proc.returncode
        with _JOBS_LOCK:
            job = _JOBS[job_id]
            job["returncode"] = rc
            job["finished"] = time.time()
            job["elapsed_s"] = round(job["finished"] - job["started"], 1)
            job["status"] = "done" if rc == 0 else "error"
        if rc == 0 and clip_id:
            _ingest_tracks_for_uploaded_clip(clip_id)
    except Exception as exc:
        with _JOBS_LOCK:
            job = _JOBS[job_id]
            job["status"] = "error"
            job["finished"] = time.time()
            job["elapsed_s"] = round(job["finished"] - job["started"], 1)
            job["error"] = str(exc)


def _zone_editor_source_map() -> Dict[str, List[str]]:
    src: Dict[str, List[str]] = {k: list(v) for k, v in ZONE_EDITOR_BASE_SOURCES.items()}
    if _clip_has_reference_frame(CAMERA_RAW_SOURCE_CLIPS["cross_cam1"]):
        src["cross_cam1"] = [CAMERA_RAW_SOURCE_CLIPS["cross_cam1"]]
    if _clip_has_reference_frame(CAMERA_RAW_SOURCE_CLIPS["cross_cam2"]):
        src["cross_cam2"] = [CAMERA_RAW_SOURCE_CLIPS["cross_cam2"]]
    return src


def _zone_editor_mode_info(conn: sqlite3.Connection, camera_id: str) -> Dict[str, Any]:
    cam = _norm_camera_id(camera_id)
    source_map = _zone_editor_source_map()

    if cam == "cross_cam":
        return {
            "camera_id": cam,
            "editable": False,
            "reason": "cross_cam is analytics-only and has no single reference frame.",
            "default_clip_id": None,
            "clips": [],
        }

    if cam == _UPLOADED_CAMERA_ID:
        rows = conn.execute(
            "SELECT DISTINCT clip_id FROM tracks WHERE camera_id=? ORDER BY clip_id;",
            (_UPLOADED_CAMERA_ID,),
        ).fetchall()
        all_clips = [str(r["clip_id"]) for r in rows if r["clip_id"]]
        clips = [cid for cid in all_clips if _clip_has_reference_frame(cid)]
        if clips:
            return {
                "camera_id": cam,
                "editable": True,
                "reason": None,
                "default_clip_id": clips[0],
                "clips": clips,
            }
        return {
            "camera_id": cam,
            "editable": False,
            "reason": "No uploaded clips with an accessible video file.",
            "default_clip_id": None,
            "clips": [],
        }

    if cam not in source_map:
        return {
            "camera_id": cam,
            "editable": False,
            "reason": "This mode is not configured as a raw zone-editable source.",
            "default_clip_id": None,
            "clips": [],
        }

    # Zone Editor must stay on canonical raw reference clips, not derived analytics clip ids.
    clips = [cid for cid in source_map[cam] if _clip_has_reference_frame(cid)]
    if clips:
        return {
            "camera_id": cam,
            "editable": True,
            "reason": None,
            "default_clip_id": clips[0],
            "clips": clips,
        }

    return {
        "camera_id": cam,
        "editable": False,
        "reason": "Configured raw source clip is missing on disk.",
        "default_clip_id": None,
        "clips": [],
    }


def _dashboard_clips_for_camera(conn: sqlite3.Connection, camera_id: str) -> List[str]:
    cam = _norm_camera_id(camera_id)
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT DISTINCT clip_id
        FROM tracks
        WHERE camera_id = ?
        ORDER BY clip_id;
        """,
        (cam,),
    ).fetchall()
    all_clips = [str(r["clip_id"]) for r in rows if r["clip_id"]]

    # Uploaded camera should show all available uploaded clip_ids in DB.
    if cam == _UPLOADED_CAMERA_ID:
        return all_clips

    # Keep dashboard/zone/recompute mapping coherent for primary single-camera modes.
    if cam in ("cam1", "cam1_hot"):
        canonical = CAMERA_RAW_SOURCE_CLIPS.get("cam1", "retail-shop_CAM1")
        return [canonical] if canonical in all_clips else []

    return all_clips


def sync_zones_from_yaml_files(conn: sqlite3.Connection, configs_dir: Path) -> Dict[str, Any]:
    imported = 0
    skipped = 0
    deleted = 0
    details: List[str] = []

    if not configs_dir.exists():
        return {
            "imported": 0,
            "skipped": 0,
            "deleted": 0,
            "details": [f"configs_dir not found: {configs_dir}"],
        }

    # Collect canonical zone_ids per camera across all YAML files before writing,
    # so we can delete stale zones (e.g. hot_zone/promo_zone mis-filed under cam1).
    valid_zones_per_camera: Dict[str, List[str]] = {}

    yamls = sorted(set(configs_dir.rglob("*.yaml")) | set(configs_dir.rglob("*.yml")))
    for yp in yamls:
        try:
            doc = yaml.safe_load(yp.read_text(encoding="utf-8")) or {}
            cam = _norm_camera_id(doc.get("camera_id", ""))

            zones_in = doc.get("zones", [])
            if not cam or not isinstance(zones_in, list) or not zones_in:
                skipped += 1
                details.append(f"{yp.name}: missing camera_id or zones[]")
                continue

            for z in zones_in:
                if not isinstance(z, dict):
                    skipped += 1
                    details.append(f"{yp.name}: zone item not dict")
                    continue

                raw_zone_id = str(z.get("zone_id", z.get("name", "zone")))
                zid = _normalize_zone_alias(raw_zone_id)
                name = str(z.get("name", zid)).strip() or zid
                zone_type = _normalize_zone_type(zid, str(z.get("zone_type", "other")).strip() or "other")

                pts = _yaml_poly_to_pts(z.get("polygon", []))
                if not pts:
                    skipped += 1
                    details.append(f"{yp.name}:{zid} invalid polygon")
                    continue

                # preserve_polygon=True: never overwrite coordinates drawn in the zone
                # editor; YAML is only authoritative for zone membership and type/name.
                upsert_zone(conn, camera_id=cam, zone_id=zid, name=name, polygon=pts,
                            zone_type=zone_type, preserve_polygon=True)
                valid_zones_per_camera.setdefault(cam, []).append(zid)
                imported += 1

        except Exception as e:
            skipped += 1
            details.append(f"{yp.name}: load failed: {e}")

    # For every camera whose zones are fully described by YAML, remove any zone_id
    # that is no longer listed.  This is what evicts hot_zone/promo_zone from cam1
    # after the mis-labelled YAML is corrected.
    for cam, valid_ids in valid_zones_per_camera.items():
        placeholders = ",".join("?" * len(valid_ids))
        stale_rows = conn.execute(
            f"SELECT zone_id FROM zones WHERE camera_id=? AND zone_id NOT IN ({placeholders});",
            [cam] + valid_ids,
        ).fetchall()
        if stale_rows:
            stale = [r["zone_id"] for r in stale_rows]
            conn.execute(
                f"DELETE FROM zones WHERE camera_id=? AND zone_id NOT IN ({placeholders});",
                [cam] + valid_ids,
            )
            deleted += len(stale)
            details.append(f"Removed stale zones from {cam}: {stale}")

    return {"imported": imported, "skipped": skipped, "deleted": deleted, "details": details}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if SYNC_ZONES_FROM_YAML:
        try:
            if DB_PATH.exists():
                conn = get_connection(DB_PATH)
                init_schema(conn)
                result = sync_zones_from_yaml_files(conn, configs_dir=CONFIGS_DIR)
                conn.commit()
                conn.close()
                print(f"[startup] YAML sync: imported={result['imported']} skipped={result['skipped']}")
            else:
                print(f"[startup] DB not found at {DB_PATH}, skipping YAML sync.")
        except Exception as e:
            print(f"[startup] YAML sync failed: {e}")
    yield


app = FastAPI(
    title="Retail Analytics API",
    description="Backend for zone-level KPIs and per-person analytics.",
    version="0.4.3",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = _mount_static_if_exists(app)


def get_conn() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"DB file not found at {DB_PATH}. Run ingestion/pipeline first.",
        )
    conn = get_connection(DB_PATH)
    init_schema(conn)
    return conn


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/ui")


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "db_exists": DB_PATH.exists(),
        "db_path": str(DB_PATH),
        "static_dir": str(STATIC_DIR) if STATIC_DIR else None,
        "configs_dir": str(CONFIGS_DIR),
        "templates_dir": str(TEMPLATES_DIR),
        "sync_zones_from_yaml": SYNC_ZONES_FROM_YAML,
    }


def _freshness_for_camera(conn: sqlite3.Connection, camera_id: str) -> Dict[str, Any]:
    cam = _norm_camera_id(camera_id)
    cur = conn.cursor()

    tr = cur.execute(
        """
        SELECT
          COUNT(*) AS track_rows,
          COUNT(DISTINCT global_id) AS track_ids,
          COUNT(DISTINCT clip_id) AS clip_count,
          MIN(ts) AS min_ts,
          MAX(ts) AS max_ts,
          SUM(CASE WHEN zone_id IS NOT NULL AND TRIM(zone_id) <> '' THEN 1 ELSE 0 END) AS zoned_rows
        FROM tracks
        WHERE camera_id = ?;
        """,
        (cam,),
    ).fetchone()

    ev = cur.execute(
        """
        SELECT
          COUNT(*) AS event_rows,
          COUNT(DISTINCT global_id) AS event_ids,
          MIN(t_enter) AS min_t_enter,
          MAX(t_exit) AS max_t_exit
        FROM events
        WHERE camera_id = ?;
        """,
        (cam,),
    ).fetchone()

    clips_rows = cur.execute(
        """
        SELECT DISTINCT clip_id
        FROM tracks
        WHERE camera_id = ?
        ORDER BY clip_id;
        """,
        (cam,),
    ).fetchall()
    clips = [str(r["clip_id"]) for r in clips_rows if r["clip_id"]]

    zones_count = int(
        cur.execute("SELECT COUNT(*) AS n FROM zones WHERE camera_id = ?;", (cam,)).fetchone()["n"] or 0
    )

    track_rows = int(tr["track_rows"] or 0)
    track_ids = int(tr["track_ids"] or 0)
    zoned_rows = int(tr["zoned_rows"] or 0)
    event_rows = int(ev["event_rows"] or 0)
    event_ids = int(ev["event_ids"] or 0)
    zoned_ratio = (float(zoned_rows) / float(track_rows)) if track_rows > 0 else 0.0
    event_id_ratio = (float(event_ids) / float(track_ids)) if track_ids > 0 else 0.0
    min_event_id_ratio = float(FRESHNESS_EVENT_ID_RATIO_THRESHOLDS.get(cam, 0.25))

    level = "ok"
    messages: List[str] = []
    actions: List[str] = []

    def escalate(new_level: str) -> None:
        nonlocal level
        order = {"ok": 0, "warn": 1, "error": 2}
        if order.get(new_level, 0) > order.get(level, 0):
            level = new_level

    if track_rows <= 0:
        escalate("error")
        messages.append("No tracks found.")
        actions.append("Run tracking pipeline and ingest tracks for this camera.")

    if track_rows > 0 and zoned_rows <= 0:
        escalate("warn")
        messages.append("Tracks exist but zone_id is empty for all rows.")
        actions.append("Run add_zones.py on source tracks CSV before ingestion.")

    if zoned_rows > 0 and event_rows <= 0:
        escalate("warn")
        messages.append("Zoned tracks exist but no events were generated.")
        actions.append("Run ingest_kpi.py to rebuild events from zoned tracks.")

    if track_ids > 0 and event_ids > 0 and event_id_ratio < min_event_id_ratio:
        escalate("warn")
        messages.append(
            f"Event person coverage is low ({event_ids}/{track_ids}, {event_id_ratio * 100:.1f}% < {min_event_id_ratio * 100:.0f}% threshold)."
        )
        actions.append("Re-check zone coverage and re-run add_zones.py + ingest_kpi.py for this camera mode.")

    known_good_clips = {
        CAMERA_RAW_SOURCE_CLIPS.get("cam1", "retail-shop_CAM1"),
        CAMERA_RAW_SOURCE_CLIPS.get("full_cam1", "retail-shop_FULL_CAM1"),
    }
    if cam in ("cam1", "cam1_hot") and clips:
        unknown_clips = [c for c in clips if c not in known_good_clips]
        if unknown_clips:
            escalate("warn")
            messages.append(f"Unexpected clip(s) found: {unknown_clips}. Expected one of {sorted(known_good_clips)}.")
            actions.append("Re-check ingestion pipeline; clip_id may not match the configured raw source.")

    return {
        "camera_id": cam,
        "status": level,
        "messages": messages,
        "actions": sorted(set(actions)),
        "stats": {
            "track_rows": track_rows,
            "track_ids": track_ids,
            "zoned_rows": zoned_rows,
            "zoned_ratio": zoned_ratio,
            "event_rows": event_rows,
            "event_ids": event_ids,
            "event_id_ratio": event_id_ratio,
            "event_id_ratio_threshold": min_event_id_ratio,
            "clip_count": int(tr["clip_count"] or 0),
            "clips": clips,
            "zone_defs_count": zones_count,
            "track_ts_min": tr["min_ts"],
            "track_ts_max": tr["max_ts"],
            "event_t_min": ev["min_t_enter"],
            "event_t_max": ev["max_t_exit"],
        },
    }


@app.get("/health/data-freshness")
def health_data_freshness(camera_id: Optional[str] = None) -> Dict[str, Any]:
    conn = get_conn()
    try:
        targets = [_norm_camera_id(camera_id)] if camera_id else list(DASHBOARD_CAMERA_MODES)
        checks = [_freshness_for_camera(conn, cam) for cam in targets]
    finally:
        conn.close()

    overall = "ok"
    order = {"ok": 0, "warn": 1, "error": 2}
    for c in checks:
        if order.get(c["status"], 0) > order.get(overall, 0):
            overall = c["status"]

    return {
        "status": overall,
        "checked_cameras": targets,
        "checks": checks,
    }


@app.get("/ui")
def ui_page(request: Request):
    return templates.TemplateResponse(request, "dashboard.html", {"request": request})


@app.get("/ui/upload")
def ui_upload(request: Request):
    return templates.TemplateResponse(request, "upload.html", {"request": request})


@app.get("/ui/zones")
def ui_zones(request: Request):
    return templates.TemplateResponse(request, "zones.html", {"request": request})


@app.get("/meta/cameras")
def meta_cameras(for_zone_editor: bool = Query(False)) -> Dict[str, Any]:
    conn = get_conn()
    try:
        if not for_zone_editor:
            cams = list(DASHBOARD_CAMERA_MODES)
            # Keep uploaded mode discoverable at all times so users can
            # immediately navigate to custom-upload results after pipeline runs.
            if _UPLOADED_CAMERA_ID not in cams:
                cams.append(_UPLOADED_CAMERA_ID)
            return {"cameras": cams}

        source_map = _zone_editor_source_map()
        ordered = ["cam1", "cam1_hot"] + [c for c in ("cross_cam1", "cross_cam2") if c in source_map]
        modes = [_zone_editor_mode_info(conn, cam) for cam in ordered]
        uploaded_info = _zone_editor_mode_info(conn, _UPLOADED_CAMERA_ID)
        if uploaded_info.get("clips"):
            modes.append(uploaded_info)
        return {
            "cameras": [m["camera_id"] for m in modes],
            "camera_modes": modes,
        }
    finally:
        conn.close()


@app.get("/meta/groups")
def meta_groups(camera_id: str = Query(...), for_zone_editor: bool = Query(False)) -> Dict[str, Any]:
    camera_id = _norm_camera_id(camera_id)
    conn = get_conn()
    try:
        if for_zone_editor:
            info = _zone_editor_mode_info(conn, camera_id)
            if not info["editable"]:
                return {
                    "camera_id": camera_id,
                    "groups": [],
                    "editable": False,
                    "reason": info["reason"],
                }
            groups = []
            seen = set()
            for c in info["clips"]:
                try:
                    g = _clip_id_to_path(c).parent.name
                except HTTPException:
                    continue
                if g not in seen:
                    groups.append(g)
                    seen.add(g)
            groups.sort()
            return {
                "camera_id": camera_id,
                "groups": groups,
                "editable": True,
                "reason": None,
            }

        rows = conn.execute(
            "SELECT DISTINCT clip_id FROM tracks WHERE camera_id=? ORDER BY clip_id;",
            (camera_id,),
        ).fetchall()
        groups = sorted({"_".join(r["clip_id"].split("_")[:2]) for r in rows if r["clip_id"]})
        return {"camera_id": camera_id, "groups": groups}
    finally:
        conn.close()


@app.get("/meta/preprocess-report")
def meta_preprocess_report(clip_id: str = Query(...)) -> Dict[str, Any]:
    """
    Return the preprocessing quality report for a clip.

    Reads {stem}_preprocess_report.json written by preprocess_video() before
    each tracking run.  Returns found=false with a null report when the file
    does not exist yet (pipeline has not been run for this clip).
    """
    clip_id = str(clip_id).strip()
    report_path = _find_preprocess_report(clip_id)
    if report_path is None:
        return {"clip_id": clip_id, "found": False, "path": None, "report": None}
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(500, f"Failed to read preprocess report: {exc}") from exc
    return {
        "clip_id": clip_id,
        "found": True,
        "path": str(report_path.relative_to(ROOT)),
        "report": data,
    }


@app.post("/pipeline/preprocess")
def pipeline_preprocess(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """
    Run preprocess_video() for a clip and return the quality report.

    Calls Stage 1 preprocessing synchronously: reads video metadata, scores
    every frame for quality, aggregates results, and writes
    {stem}_preprocess_report.json to the appropriate runs directory.

    The response shape is identical to GET /meta/preprocess-report so the
    preprocessing banner JS can handle both without changes.
    """
    clip_id = _clean_clip_id(payload.get("clip_id", ""))
    if not clip_id:
        raise HTTPException(400, "clip_id is required")

    try:
        video_path = _clip_id_to_path(clip_id)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, f"Cannot resolve clip_id '{clip_id}': {exc}") from exc

    if not video_path.exists():
        raise HTTPException(
            404,
            f"Video file not found on disk: {video_path.relative_to(ROOT)}. "
            "Place the video in the expected path and retry.",
        )

    out_dir = _out_dir_for_clip(clip_id)

    try:
        report = preprocess_video(video_path, out_dir=out_dir, write_report=True)
    except Exception as exc:
        raise HTTPException(500, f"Preprocessing failed for '{clip_id}': {exc}") from exc

    report_dict = asdict(report)
    report_dict.pop("report_path", None)   # not part of the public response

    rel_path: Optional[str] = None
    if report.report_path:
        try:
            rel_path = str(Path(report.report_path).relative_to(ROOT))
        except ValueError:
            rel_path = report.report_path

    return {
        "clip_id": clip_id,
        "found": True,
        "path": rel_path,
        "report": report_dict,
    }


@app.post("/pipeline/run")
def pipeline_run(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """
    Start the analytics pipeline for a clip in a background thread.

    Maps clip_id to the correct script:
    - retail-shop_*  → scripts/run_batch.py --match {stem}
    - cross_cam*     → scripts/run_cross_cam.py  (default args, runs both cameras)

    Returns a job_id immediately.  Poll GET /pipeline/status/{job_id} to track
    progress.  Returns 409 if a job for this clip_id is already running.
    """
    clip_id = _clean_clip_id(payload.get("clip_id", ""))
    if not clip_id:
        raise HTTPException(400, "clip_id is required")

    try:
        video_path = _clip_id_to_path(clip_id)
    except HTTPException:
        raise

    if not video_path.exists():
        raise HTTPException(
            404,
            f"Video file not found: {video_path.relative_to(ROOT)}. "
            "Place the video at the expected path and retry.",
        )

    with _JOBS_LOCK:
        for j in _JOBS.values():
            if j["clip_id"] == clip_id and j["status"] == "running":
                raise HTTPException(
                    409,
                    f"A pipeline job for '{clip_id}' is already running "
                    f"(job_id={j['job_id']}). "
                    "Wait for it to finish or check its status.",
                )

    script, extra_args = _script_and_args_for_clip(clip_id)
    job_id = uuid.uuid4().hex[:8]
    cmd = [sys.executable, str(script)] + extra_args
    log_path = _out_dir_for_clip(clip_id) / f"{clip_id}_pipeline_{job_id}.log"

    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "job_id": job_id,
            "clip_id": clip_id,
            "script": script.name,
            "cmd": " ".join(str(x) for x in cmd),
            "status": "running",
            "started": time.time(),
            "finished": None,
            "elapsed_s": None,
            "returncode": None,
            "log_path": str(log_path),
            "error": None,
        }

    threading.Thread(target=_run_job, args=(job_id, cmd, log_path, clip_id), daemon=True).start()

    return {
        "job_id": job_id,
        "clip_id": clip_id,
        "script": script.name,
        "status": "running",
        "log_path": str(log_path.relative_to(ROOT)),
        "cmd": " ".join(str(x) for x in cmd),
    }


@app.get("/pipeline/status/{job_id}")
def pipeline_status(job_id: str) -> Dict[str, Any]:
    """
    Return the current status of a pipeline job.

    Reads the last 20 lines of the job's log file so the caller can show
    progress without a separate log endpoint.
    """
    with _JOBS_LOCK:
        job = dict(_JOBS.get(job_id) or {})

    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")

    log_tail = ""
    log_path_str = job.get("log_path", "")
    if log_path_str:
        try:
            lines = Path(log_path_str).read_text(encoding="utf-8", errors="replace").splitlines()
            log_tail = "\n".join(lines[-20:])
        except Exception:
            pass

    # For still-running jobs, compute elapsed live so the UI shows a real number.
    elapsed = job["elapsed_s"]
    if elapsed is None and job.get("started"):
        elapsed = round(time.time() - job["started"], 1)

    return {
        "job_id":    job["job_id"],
        "clip_id":   job["clip_id"],
        "script":    job["script"],
        "status":    job["status"],
        "started":   job["started"],
        "elapsed_s": elapsed,
        "returncode": job["returncode"],
        "error":     job.get("error"),
        "log_tail":  log_tail,
    }


@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    group: str = Form("retail-shop"),
) -> Dict[str, Any]:
    """
    Save an uploaded MP4 video to data/raw/{group}/{filename}.mp4 and return
    a usable clip_id for the existing pipeline endpoints.

    - 400 for non-.mp4 files or unsafe names
    - 409 if the file already exists at the target path
    """
    # --- validate extension ---
    original_name = Path(file.filename or "").name
    if not original_name.lower().endswith(".mp4"):
        raise HTTPException(400, "Only .mp4 files are accepted.")

    # --- sanitise group: only word chars, hyphens, dots ---
    safe_group = re.sub(r"[^\w\-.]", "_", group.strip()).strip("_. ")
    if not safe_group:
        raise HTTPException(400, "group must not be empty after sanitisation.")

    # --- sanitise filename: strip to basename, no path components ---
    safe_stem = re.sub(r"[^\w\-.]", "_", Path(original_name).stem.strip()).strip("_. ")
    if not safe_stem:
        raise HTTPException(400, "Filename stem is invalid after sanitisation.")
    safe_filename = f"{safe_stem}.mp4"

    # --- resolve target path and verify it stays inside RAW_ROOT ---
    dest_dir = RAW_ROOT / safe_group
    dest_path = (dest_dir / safe_filename).resolve()
    try:
        dest_path.relative_to(RAW_ROOT.resolve())
    except ValueError:
        raise HTTPException(400, "Resolved path escapes the raw data directory.")

    if dest_path.exists():
        raise HTTPException(
            409,
            f"File already exists: data/raw/{safe_group}/{safe_filename}. "
            "Delete it first or choose a different name.",
        )

    dest_dir.mkdir(parents=True, exist_ok=True)

    try:
        with dest_path.open("wb") as fh:
            shutil.copyfileobj(file.file, fh)
    except Exception as exc:
        # Clean up partial write before surfacing the error.
        dest_path.unlink(missing_ok=True)
        raise HTTPException(500, f"Failed to save file: {exc}") from exc
    finally:
        await file.close()

    size_mb = round(dest_path.stat().st_size / 1_048_576, 2)
    clip_id = f"{safe_group}_{safe_stem}"
    rel_path = str(dest_path.relative_to(ROOT))

    return {
        "clip_id": clip_id,
        "path": rel_path,
        "size_mb": size_mb,
    }


@app.get("/kpi/clips")
def kpi_clips(camera_id: str = "cam1", for_zone_editor: bool = Query(False)) -> Dict[str, Any]:
    camera_id = _norm_camera_id(camera_id)
    conn = get_conn()
    try:
        if for_zone_editor:
            info = _zone_editor_mode_info(conn, camera_id)
            return {
                "camera_id": camera_id,
                "clips": info["clips"],
                "editable": info["editable"],
                "reason": info["reason"],
                "default_clip_id": info["default_clip_id"],
            }

        return {"camera_id": camera_id, "clips": _dashboard_clips_for_camera(conn, camera_id)}
    finally:
        conn.close()


def _ev_where(camera_id: str, clip_id: Optional[str], group: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    where = "WHERE camera_id = :camera_id"
    params: Dict[str, Any] = {"camera_id": camera_id}

    clip_id = _clean_clip_id(clip_id)
    group = _clean_group(group)

    if clip_id:
        where += " AND clip_id = :clip_id"
        params["clip_id"] = clip_id
    elif group:
        where += " AND clip_id LIKE :clip_like"
        params["clip_like"] = f"{group}%"

    return where, params


def _cam1_bounce_rate_from_progression(
    conn: sqlite3.Connection,
    clip_id: Optional[str],
    group: Optional[str],
) -> float:
    where, params = _ev_where("cam1", clip_id, group)
    cur = conn.cursor()

    cur.execute(
        f"""
        SELECT DISTINCT global_id
        FROM events
        {where}
          AND zone_id = 'entrance_zone'
        ORDER BY global_id;
        """,
        params,
    )
    entrance_ids = {int(r["global_id"]) for r in cur.fetchall()}

    cur.execute(
        f"""
        SELECT DISTINCT global_id
        FROM events
        {where}
          AND zone_id = 'walkway_zone'
        ORDER BY global_id;
        """,
        params,
    )
    walkway_ids = {int(r["global_id"]) for r in cur.fetchall()}

    if not entrance_ids:
        return 0.0

    bounced_ids = entrance_ids - walkway_ids
    return len(bounced_ids) / len(entrance_ids)


@app.get("/kpi/zone-summary")
def zone_summary(
    camera_id: str = "cam1",
    clip_id: Optional[str] = None,
    group: Optional[str] = None,
) -> Dict[str, Any]:
    camera_id = _norm_camera_id(camera_id)
    conn = get_conn()
    zone_defs: List[Dict[str, Any]] = []

    try:
        cur = conn.cursor()
        where, params = _ev_where(camera_id, clip_id, group)
        zone_defs = list_zones(conn, camera_id=camera_id)

        cur.execute(
            f"""
            SELECT zone_id,
                   COUNT(*) AS visits,
                   SUM(dwell_s) AS dwell_s,
                   AVG(dwell_s) AS avg_dwell_s
            FROM events
            {where}
            GROUP BY zone_id
            ORDER BY zone_id;
            """,
            params,
        )
        zone_rows = cur.fetchall()

        cur.execute(
            f"""
            SELECT zone_id,
                   dwell_s
            FROM events
            {where}
            ORDER BY zone_id, t_enter;
            """,
            params,
        )
        event_rows = cur.fetchall()

        cur.execute(
            f"""
            SELECT COUNT(DISTINCT global_id) AS n_persons
            FROM tracks
            {where};
            """,
            params,
        )
        n_persons = int(cur.fetchone()["n_persons"] or 0)

        if camera_id == "cam1":
            bounce_rate = _cam1_bounce_rate_from_progression(conn, clip_id=clip_id, group=group)
        else:
            bounce_rate = 0.0

    finally:
        conn.close()

    zone_totals: Dict[str, Dict[str, float]] = {}
    for z in zone_defs:
        zid = _normalize_zone_alias(str(z.get("zone_id") or ""))
        if not zid:
            continue
        zone_totals[zid] = {
            "visits": 0,
            "dwell_s": 0.0,
            "avg_dwell_s": 0.0,
        }

    # strict=True when zone_defs exist: only populate zones that belong to this camera.
    # strict=False (no zone_defs yet) falls back to showing all event zones so a
    # camera with no drawn zones still reports data.
    strict = bool(zone_totals)
    for r in zone_rows:
        zone_id = _normalize_zone_alias(str(r["zone_id"]))
        if strict and zone_id not in zone_totals:
            continue
        zone_totals[zone_id] = {
            "visits": int(r["visits"] or 0),
            "dwell_s": float(r["dwell_s"] or 0.0),
            "avg_dwell_s": float(r["avg_dwell_s"] or 0.0),
        }

    event_stats: Dict[str, Dict[str, float]] = {}
    for r in event_rows:
        zone_id = _normalize_zone_alias(str(r["zone_id"]))
        dwell_s = float(r["dwell_s"] or 0.0)
        threshold_s = _engagement_threshold(zone_id)

        if zone_id not in event_stats:
            event_stats[zone_id] = {
                "qualified_visits": 0,
                "transit_visits": 0,
                "qualified_dwell_s": 0.0,
                "transit_dwell_s": 0.0,
            }

        if dwell_s >= threshold_s:
            event_stats[zone_id]["qualified_visits"] += 1
            event_stats[zone_id]["qualified_dwell_s"] += dwell_s
        else:
            event_stats[zone_id]["transit_visits"] += 1
            event_stats[zone_id]["transit_dwell_s"] += dwell_s

    per_zone: Dict[str, Any] = {}
    total_visits = 0
    total_qualified_visits = 0

    for zone_id, totals in zone_totals.items():
        visits = int(totals["visits"])
        dwell_s = float(totals["dwell_s"])
        avg_dwell_s = float(totals["avg_dwell_s"])
        threshold_s = _engagement_threshold(zone_id)

        stats = event_stats.get(zone_id, {
            "qualified_visits": 0,
            "transit_visits": 0,
            "qualified_dwell_s": 0.0,
            "transit_dwell_s": 0.0,
        })

        qualified_visits = int(stats["qualified_visits"])
        transit_visits = int(stats["transit_visits"])
        qualified_dwell_s = float(stats["qualified_dwell_s"])
        transit_dwell_s = float(stats["transit_dwell_s"])
        qualification_rate = (qualified_visits / visits) if visits > 0 else 0.0

        total_visits += visits
        total_qualified_visits += qualified_visits

        per_zone[zone_id] = {
            "zone_id": zone_id,
            "zone_kind": _zone_kind(zone_id),
            "qualification_threshold_s": threshold_s,
            "visits": visits,
            "dwell_s": dwell_s,
            "avg_dwell_s": avg_dwell_s,
            "qualified_visits": qualified_visits,
            "transit_visits": transit_visits,
            "qualified_dwell_s": qualified_dwell_s,
            "transit_dwell_s": transit_dwell_s,
            "qualification_rate": qualification_rate,
        }

    if camera_id != "cam1":
        if total_visits > 0:
            bounce_rate = max(0.0, 1.0 - (total_qualified_visits / total_visits))
        else:
            bounce_rate = 0.0

    return {
        "per_zone": per_zone,
        "bounce_rate": bounce_rate,
        "n_persons": n_persons,
    }


def _zone_stat(per_zone: Dict[str, Any], zone_id: str, key: str, default: float = 0.0) -> float:
    z = per_zone.get(_normalize_zone_alias(zone_id), {}) or {}
    try:
        return float(z.get(key, default) or default)
    except Exception:
        return default


def _zone_visits(per_zone: Dict[str, Any], zone_id: str) -> int:
    z = per_zone.get(_normalize_zone_alias(zone_id), {}) or {}
    try:
        return int(z.get("visits", 0) or 0)
    except Exception:
        return 0


def _zone_qualified_visits(per_zone: Dict[str, Any], zone_id: str) -> int:
    z = per_zone.get(_normalize_zone_alias(zone_id), {}) or {}
    try:
        return int(z.get("qualified_visits", 0) or 0)
    except Exception:
        return 0


def _sum_zone_metric(
    per_zone: Dict[str, Any],
    *,
    metric: str,
    zone_kind: Optional[str] = None,
    zone_tokens: Optional[Tuple[str, ...]] = None,
) -> float:
    total = 0.0
    tokens = tuple((zone_tokens or ()))
    for zone_id, stats in per_zone.items():
        zid = str(zone_id).lower()
        kind = str(stats.get("zone_kind", "")).lower()
        if zone_kind and kind != zone_kind:
            continue
        if tokens and not any(tok in zid for tok in tokens):
            continue
        total += float(stats.get(metric, 0.0) or 0.0)
    return total


@app.get("/kpi/camera-summary")
def camera_summary(
    camera_id: str = "cam1",
    clip_id: Optional[str] = None,
    group: Optional[str] = None,
) -> Dict[str, Any]:
    camera_id = _norm_camera_id(camera_id)
    zs = zone_summary(camera_id=camera_id, clip_id=clip_id, group=group)
    per_zone = zs.get("per_zone", {}) or {}
    n_persons = int(zs.get("n_persons", 0) or 0)
    bounce_rate = float(zs.get("bounce_rate", 0.0) or 0.0)

    cards: List[Dict[str, str]] = []

    if camera_id == "cam1":
        entrance_dwell = _zone_stat(per_zone, "entrance_zone", "dwell_s", 0.0)
        walkway_visits = _zone_visits(per_zone, "walkway_zone")

        cards = [
            {"label": "Persons", "value": str(n_persons), "hint": "unique IDs"},
            {"label": "Bounce rate", "value": f"{bounce_rate * 100:.1f}%", "hint": "entrance without walkway"},
            {"label": "Entrance dwell", "value": f"{entrance_dwell:.2f}s", "hint": "total dwell"},
            {"label": "Walkway visits", "value": str(walkway_visits), "hint": "traffic flow"},
        ]

    elif camera_id == "full_cam1":
        entrance_visits = _sum_zone_metric(
            per_zone,
            metric="visits",
            zone_tokens=("entrance",),
        )
        downstream_qualified = _sum_zone_metric(
            per_zone,
            metric="qualified_visits",
            zone_tokens=("walkway", "hot", "promo", "browse", "cashier", "checkout"),
        )
        progression_rate = 0.0 if entrance_visits <= 0 else min(1.0, downstream_qualified / entrance_visits)

        engagement_qualified = _sum_zone_metric(
            per_zone,
            metric="qualified_visits",
            zone_kind="engagement",
        )
        checkout_qualified = _sum_zone_metric(
            per_zone,
            metric="qualified_visits",
            zone_kind="checkout",
        )

        total_visits = sum(int(v.get("visits", 0) or 0) for v in per_zone.values())
        total_dwell = sum(float(v.get("dwell_s", 0.0) or 0.0) for v in per_zone.values())
        avg_dwell = (total_dwell / total_visits) if total_visits > 0 else 0.0

        cards = [
            {"label": "Persons", "value": str(n_persons), "hint": "full-clip unique IDs"},
            {"label": "Progression rate", "value": f"{progression_rate * 100:.1f}%", "hint": "entrance to downstream zones"},
            {"label": "Qualified engagement", "value": str(int(engagement_qualified)), "hint": "hot/promo/browsing intent"},
            {"label": "Checkout touches", "value": str(int(checkout_qualified)), "hint": f"avg dwell {avg_dwell:.2f}s"},
        ]

    elif camera_id == "cross_cam":
        handoff_visits = _sum_zone_metric(
            per_zone,
            metric="qualified_visits",
            zone_tokens=("handoff", "transfer", "cross", "bridge"),
        )
        if handoff_visits <= 0:
            handoff_visits = _sum_zone_metric(
                per_zone,
                metric="visits",
                zone_tokens=("handoff", "transfer", "cross", "bridge"),
            )

        active_zones = [z for z, s in per_zone.items() if int(s.get("visits", 0) or 0) > 0]
        top_zone = "-"
        if per_zone:
            top_zone = max(
                per_zone.items(),
                key=lambda kv: float(kv[1].get("dwell_s", 0.0) or 0.0)
            )[0]

        total_qualified = sum(int(v.get("qualified_visits", 0) or 0) for v in per_zone.values())
        total_qualified_dwell = sum(float(v.get("qualified_dwell_s", 0.0) or 0.0) for v in per_zone.values())
        avg_qualified_dwell = (total_qualified_dwell / total_qualified) if total_qualified > 0 else 0.0

        cards = [
            {"label": "Persons", "value": str(n_persons), "hint": "cross-view unique IDs"},
            {"label": "Cross-camera handoffs", "value": str(int(handoff_visits)), "hint": "handoff/transfer zones"},
            {"label": "Zone coverage", "value": str(len(active_zones)), "hint": "active semantic zones"},
            {"label": "Avg qualified dwell", "value": f"{avg_qualified_dwell:.2f}s", "hint": f"top zone {top_zone}"},
        ]

    elif camera_id == "cam1_hot":
        hot_qualified = _zone_qualified_visits(per_zone, "hot_zone")
        promo_qualified = _zone_qualified_visits(per_zone, "promo_zone")

        total_qualified = sum(int(v.get("qualified_visits", 0) or 0) for v in per_zone.values())
        total_qualified_dwell = sum(float(v.get("qualified_dwell_s", 0.0) or 0.0) for v in per_zone.values())
        avg_qualified_dwell = (total_qualified_dwell / total_qualified) if total_qualified > 0 else 0.0

        top_zone = "-"
        if per_zone:
            top_zone = max(
                per_zone.items(),
                key=lambda kv: float(kv[1].get("qualified_dwell_s", 0.0) or 0.0)
            )[0]

        cards = [
            {"label": "Qualified hot-zone visits", "value": str(hot_qualified), "hint": "threshold-based interest"},
            {"label": "Qualified promo visits", "value": str(promo_qualified), "hint": "threshold-based interest"},
            {"label": "Avg qualified dwell", "value": f"{avg_qualified_dwell:.2f}s", "hint": "qualified visits"},
            {"label": "Top engagement", "value": top_zone, "hint": "highest qualified dwell"},
        ]

    elif camera_id == "cam2":
        browsing_qualified = _zone_qualified_visits(per_zone, "browsing_zone")
        cashier_qualified = _zone_qualified_visits(per_zone, "cashier_zone")
        cashier_dwell = _zone_stat(per_zone, "cashier_zone", "dwell_s", 0.0)

        service_time = (
            _zone_stat(per_zone, "cashier_zone", "dwell_s", 0.0) +
            _zone_stat(per_zone, "promo_zone", "dwell_s", 0.0)
        )

        cards = [
            {"label": "Qualified browsing visits", "value": str(browsing_qualified), "hint": "threshold-based interest"},
            {"label": "Qualified cashier visits", "value": str(cashier_qualified), "hint": "threshold-based service"},
            {"label": "Cashier dwell", "value": f"{cashier_dwell:.2f}s", "hint": "service time"},
            {"label": "Service time", "value": f"{service_time:.2f}s", "hint": "promo + cashier"},
        ]

    else:
        total_visits = sum(int(v.get("visits", 0) or 0) for v in per_zone.values())
        total_dwell = sum(float(v.get("dwell_s", 0.0) or 0.0) for v in per_zone.values())
        avg_dwell = (total_dwell / total_visits) if total_visits > 0 else 0.0

        cards = [
            {"label": "Persons", "value": str(n_persons), "hint": "unique IDs"},
            {"label": "Bounce rate", "value": f"{bounce_rate * 100:.1f}%", "hint": "session-level"},
            {"label": "Visits", "value": str(total_visits), "hint": "all zones"},
            {"label": "Avg dwell", "value": f"{avg_dwell:.2f}s", "hint": "all zones"},
        ]

    return {"camera_id": camera_id, "cards": cards}


@app.get("/kpi/insights")
def kpi_insights(
    camera_id: str = "cam1",
    clip_id: Optional[str] = None,
    group: Optional[str] = None,
) -> Dict[str, Any]:
    zs = zone_summary(camera_id=camera_id, clip_id=clip_id, group=group)
    return summarize_kpis(
        camera_id=_norm_camera_id(camera_id),
        per_zone=zs.get("per_zone", {}),
        n_persons=int(zs.get("n_persons", 0)),
    )


@app.get("/kpi/person-summary")
def person_summary(
    camera_id: str = "cam1",
    clip_id: Optional[str] = None,
    group: Optional[str] = None,
    t_min: Optional[float] = None,
    t_max: Optional[float] = None,
) -> Dict[str, Any]:
    camera_id = _norm_camera_id(camera_id)
    clip_id = _clean_clip_id(clip_id)
    group = _clean_group(group)

    conn = get_conn()
    try:
        cur = conn.cursor()
        params: Dict[str, Any] = {
            "camera_id": camera_id,
            "clip_id": clip_id,
            "t_min": t_min,
            "t_max": t_max,
        }

        ev_where = "WHERE camera_id = :camera_id"
        if clip_id:
            ev_where += " AND clip_id = :clip_id"
        elif group:
            ev_where += " AND clip_id LIKE :clip_like"
            params["clip_like"] = f"{group}%"

        if t_min is not None:
            ev_where += " AND t_enter >= :t_min"
        if t_max is not None:
            ev_where += " AND t_exit <= :t_max"

        cur.execute(
            f"""
            SELECT global_id,
                   COUNT(*) AS visits,
                   SUM(dwell_s) AS total_dwell
            FROM events
            {ev_where}
            GROUP BY global_id
            ORDER BY global_id;
            """,
            params,
        )
        ev_rows = cur.fetchall()

        event_gids = [int(r["global_id"]) for r in ev_rows]
        frames_map: Dict[int, int] = {}
        track_fallback_rows: list = []

        if event_gids:
            placeholders = ",".join("?" for _ in event_gids)
            base = """
            SELECT global_id, COUNT(*) AS total_frames
            FROM tracks
            WHERE camera_id = ?
            """
            args: List[Any] = [camera_id]

            if clip_id:
                base += " AND clip_id = ?"
                args.append(clip_id)
            elif group:
                base += " AND clip_id LIKE ?"
                args.append(f"{group}%")

            base += f" AND global_id IN ({placeholders})"
            args.extend(event_gids)

            if t_min is not None:
                base += " AND ts >= ?"
                args.append(t_min)
            if t_max is not None:
                base += " AND ts <= ?"
                args.append(t_max)

            base += " GROUP BY global_id ORDER BY global_id;"
            cur.execute(base, args)
            tr_rows = cur.fetchall()
            frames_map = {int(r["global_id"]): int(r["total_frames"] or 0) for r in tr_rows}
        else:
            # No zone events — fall back to a track-level summary so callers
            # still get meaningful per-person rows (frame count, time window)
            # even when no zones have been configured for this camera.
            fb_base = (
                "SELECT global_id, COUNT(*) AS total_frames,"
                " MIN(ts) AS t_first, MAX(ts) AS t_last"
                " FROM tracks"
                " WHERE camera_id = ? AND global_id > 0"
            )
            fb_args: List[Any] = [camera_id]

            if clip_id:
                fb_base += " AND clip_id = ?"
                fb_args.append(clip_id)
            elif group:
                fb_base += " AND clip_id LIKE ?"
                fb_args.append(f"{group}%")

            if t_min is not None:
                fb_base += " AND ts >= ?"
                fb_args.append(t_min)
            if t_max is not None:
                fb_base += " AND ts <= ?"
                fb_args.append(t_max)

            fb_base += " GROUP BY global_id ORDER BY global_id;"
            cur.execute(fb_base, fb_args)
            track_fallback_rows = cur.fetchall()
    finally:
        conn.close()

    out: Dict[str, Any] = {}
    if ev_rows:
        for r in ev_rows:
            gid = int(r["global_id"])
            out[str(gid)] = {
                "visits": int(r["visits"] or 0),
                "total_dwell": float(r["total_dwell"] or 0.0),
                "total_frames": frames_map.get(gid, 0),
            }
    else:
        for r in track_fallback_rows:
            gid = int(r["global_id"])
            t_first = float(r["t_first"] or 0.0)
            t_last = float(r["t_last"] or 0.0)
            out[str(gid)] = {
                "visits": 0,
                "total_dwell": 0.0,
                "total_frames": int(r["total_frames"] or 0),
                "t_first": t_first,
                "t_last": t_last,
                "duration_s": round(max(0.0, t_last - t_first), 2),
            }
    return out


@app.get("/zones")
def api_list_zones(camera_id: str = Query(...)) -> List[Dict[str, Any]]:
    camera_id = _norm_camera_id(camera_id)
    conn = get_conn()
    try:
        zones = list_zones(conn, camera_id=camera_id)
        for z in zones:
            z["zone_id"] = _normalize_zone_alias(z["zone_id"])
            if z.get("name"):
                name_slug = _slugify(z["name"])
                if name_slug in ("check_zone", "decision_zone"):
                    z["name"] = "decision_zone"
                elif name_slug in ("counter_zone", "cashier_zone"):
                    z["name"] = "cashier_zone"
        return zones
    finally:
        conn.close()


@app.post("/zones")
def api_upsert_zone(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    camera_id = _norm_camera_id(payload.get("camera_id", ""))
    name = str(payload.get("name", "")).strip()
    zone_type_raw = str(payload.get("zone_type", "other")).strip()
    polygon = payload.get("polygon", [])

    if not camera_id:
        raise HTTPException(400, "camera_id required")
    if not name:
        raise HTTPException(400, "name required")
    if not isinstance(polygon, list) or len(polygon) < 3:
        raise HTTPException(400, "polygon must have >= 3 points")

    raw_zone_id = payload.get("zone_id")
    zone_id = _normalize_zone_alias(str(raw_zone_id)) if raw_zone_id else _normalize_zone_alias(name)
    zone_type = _normalize_zone_type(zone_id, zone_type_raw)

    pts = _yaml_poly_to_pts(polygon)
    if not pts:
        raise HTTPException(400, "Invalid polygon format.")

    conn = get_conn()
    try:
        upsert_zone(conn, camera_id=camera_id, zone_id=zone_id, name=name, polygon=pts, zone_type=zone_type)
        conn.commit()
    finally:
        conn.close()

    return {"ok": True, "camera_id": camera_id, "zone_id": zone_id}


@app.delete("/zones/{zone_id}")
def api_delete_zone(zone_id: str, camera_id: str = Query(...)) -> Dict[str, Any]:
    camera_id = _norm_camera_id(camera_id)
    zone_id = _normalize_zone_alias(zone_id)

    aliases = {zone_id}
    if zone_id == "decision_zone":
        aliases.add("check_zone")
    if zone_id == "cashier_zone":
        aliases.add("counter_zone")

    conn = get_conn()
    try:
        cur = conn.cursor()

        placeholders = ",".join("?" for _ in aliases)
        cur.execute(
            f"""
            DELETE FROM zones
            WHERE camera_id = ?
              AND zone_id IN ({placeholders})
            """,
            [camera_id, *aliases]
        )

        if cur.rowcount == 0:
            raise HTTPException(
                status_code=404,
                detail=f"Zone not found. Check camera_id and zone_id. (camera_id={camera_id}, zone_id={zone_id})"
            )

        conn.commit()
    finally:
        conn.close()

    return {"ok": True, "deleted": zone_id, "aliases_deleted": sorted(aliases)}


@app.get("/zones/export-yaml")
def api_export_yaml(camera_id: str = Query(...)) -> Response:
    camera_id = _norm_camera_id(camera_id)
    conn = get_conn()
    try:
        zones = list_zones(conn, camera_id=camera_id)
    finally:
        conn.close()

    doc = {
        "camera_id": camera_id,
        "zones": [
            {
                "zone_id": _normalize_zone_alias(z.get("zone_id")),
                "name": _normalize_zone_alias(z.get("name", z.get("zone_id"))),
                "zone_type": z.get("zone_type", "other"),
                "polygon": _poly_to_yaml_safe(
                    json.loads(z.get("polygon")) if isinstance(z.get("polygon"), str) else z.get("polygon", [])
                ),
            }
            for z in zones
        ],
    }
    y = yaml.safe_dump(doc, sort_keys=False)
    return Response(content=y, media_type="text/yaml")


@app.post("/zones/import-yaml")
def api_import_yaml(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    camera_id = _norm_camera_id(payload.get("camera_id", ""))
    yaml_text = str(payload.get("yaml_text", "")).strip()
    replace_existing = bool(payload.get("replace_existing", True))

    if not camera_id:
        raise HTTPException(400, "camera_id required")
    if not yaml_text:
        raise HTTPException(400, "yaml_text required")

    try:
        doc = yaml.safe_load(yaml_text) or {}
    except Exception as e:
        raise HTTPException(400, f"Bad YAML: {e}")

    zones_in = doc.get("zones", [])
    if not isinstance(zones_in, list) or len(zones_in) == 0:
        raise HTTPException(400, "No zones found under key: zones")

    conn = get_conn()
    imported = 0
    skipped = 0
    skipped_details: List[str] = []
    deleted_existing = 0

    try:
        if replace_existing:
            cur = conn.cursor()
            deleted_existing = int(
                cur.execute(
                    "SELECT COUNT(*) AS n FROM zones WHERE camera_id = ?;",
                    (camera_id,),
                ).fetchone()["n"] or 0
            )
            cur.execute("DELETE FROM zones WHERE camera_id = ?;", (camera_id,))

        for z in zones_in:
            if not isinstance(z, dict):
                skipped += 1
                skipped_details.append("Item is not a valid dictionary")
                continue

            zid = _normalize_zone_alias(str(z.get("zone_id", "zone")))
            zname = str(z.get("name", zid)).strip() or zid
            zone_type = _normalize_zone_type(zid, str(z.get("zone_type", "other")).strip() or "other")

            pts = _yaml_poly_to_pts(z.get("polygon", []))
            if not pts:
                skipped += 1
                skipped_details.append(f"{zid} (invalid/empty polygon)")
                continue

            upsert_zone(conn, camera_id=camera_id, zone_id=zid, name=zname, polygon=pts, zone_type=zone_type)
            imported += 1

        conn.commit()
    finally:
        conn.close()

    logger.info(
        "zones.import_yaml camera_id=%s replace_existing=%s deleted_existing=%d imported=%d skipped=%d",
        camera_id,
        replace_existing,
        deleted_existing,
        imported,
        skipped,
    )

    return {
        "ok": True,
        "camera_id": camera_id,
        "replace_existing": replace_existing,
        "deleted_existing_zones": deleted_existing,
        "imported": imported,
        "skipped": skipped,
        "skipped_details": skipped_details,
    }


@app.post("/zones/recompute")
def api_recompute_zones(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    camera_id = _norm_camera_id(payload.get("camera_id", ""))
    requested_clip_id = _clean_clip_id(payload.get("clip_id"))
    clip_id = requested_clip_id
    if not camera_id:
        raise HTTPException(400, "camera_id required")
    if not clip_id:
        raise HTTPException(400, "clip_id required")

    conn = get_conn()
    try:
        cur = conn.cursor()
        mode_map = {
            "cam1": ("cam1", "cam1", "ankle"),
            "cam1_hot": ("cam1_hot", "cam1_hot", "center"),
            "cross_cam1": ("cross_cam", "cross_cam1", "center"),
            "cross_cam2": ("cross_cam", "cross_cam2", "center"),
            "cross_cam": ("cross_cam", "cross_cam", "center"),
            "uploaded": ("uploaded", "uploaded", "center"),
        }
        if camera_id not in mode_map:
            raise HTTPException(400, f"Unsupported recompute camera_id: {camera_id}")
        target_cam_hint = mode_map[camera_id][0]

        clip_rows = cur.execute(
            """
            SELECT DISTINCT clip_id
            FROM tracks
            WHERE camera_id = ?
            ORDER BY clip_id;
            """,
            (target_cam_hint,),
        ).fetchall()
        all_available_clips = [str(r["clip_id"]) for r in clip_rows if r["clip_id"]]
        if target_cam_hint in ("cam1", "cam1_hot"):
            canonical = CAMERA_RAW_SOURCE_CLIPS.get("cam1", "retail-shop_CAM1")
            # Prefer the canonical clip; if not present, pass full list so derivation
            # in _resolve_recompute_scope can match derived clip IDs (e.g. *_tracks_cam1_hot).
            available_clips = [canonical] if canonical in all_available_clips else all_available_clips
        else:
            available_clips = all_available_clips

        target_cam, zone_cam, point_mode, target_clip = _resolve_recompute_scope(
            camera_id, clip_id, available_clips
        )

        zones = list_zones(conn, camera_id=zone_cam)
        zone_debug: List[Dict[str, Any]] = []
        zone_polys: List[Tuple[str, List[Tuple[float, float]]]] = []
        for z in zones:
            zid = str(z.get("zone_id") or "")
            ztype = str(z.get("zone_type") or "other")
            poly_raw = z.get("polygon", [])
            poly: List[Tuple[float, float]] = []
            if isinstance(poly_raw, list):
                for p in poly_raw:
                    if isinstance(p, (list, tuple)) and len(p) >= 2:
                        poly.append((float(p[0]), float(p[1])))
            if zid and len(poly) >= 3:
                zone_polys.append((zid, poly))
                zone_debug.append({
                    "zone_id": zid,
                    "zone_type": ztype,
                    "points": len(poly),
                })
        if not zone_polys:
            raise HTTPException(400, f"No valid zone polygons found for camera_id={zone_cam}")

        where = "WHERE camera_id = ?"
        args: List[Any] = [target_cam]
        if target_clip:
            where += " AND clip_id = ?"
            args.append(target_clip)

        rows = cur.execute(
            f"""
            SELECT rowid AS _rid, x1, y1, x2, y2
            FROM tracks
            {where}
            ORDER BY rowid;
            """,
            args,
        ).fetchall()
        pre_zoned_rows = int(
            cur.execute(
                f"SELECT COUNT(*) AS n FROM tracks {where} AND zone_id IS NOT NULL AND TRIM(zone_id) <> '';",
                args,
            ).fetchone()["n"] or 0
        )

        updates: List[Tuple[Optional[str], int]] = []
        zoned_rows = 0
        for r in rows:
            x1 = float(r["x1"])
            y1 = float(r["y1"])
            x2 = float(r["x2"])
            y2 = float(r["y2"])
            px, py = _bbox_point(x1, y1, x2, y2, mode=point_mode)
            zid: Optional[str] = None
            for zname, poly in zone_polys:
                if _point_in_poly(px, py, poly):
                    zid = zname
                    break
            if zid:
                zoned_rows += 1
            updates.append((zid, int(r["_rid"])))

        if updates:
            cur.executemany("UPDATE tracks SET zone_id = ? WHERE rowid = ?;", updates)

        old_events = int(
            cur.execute(
                "SELECT COUNT(*) AS n FROM events WHERE camera_id = ? AND clip_id = ?;",
                (target_cam, target_clip),
            ).fetchone()["n"] or 0
        )
        clear_events(conn, camera_id=target_cam, clip_id=target_clip, t_min=None, t_max=None)

        refresh_events_from_tracks(
            conn,
            camera_id=target_cam,
            clip_id=target_clip,
            min_hold_s=RECOMPUTE_MIN_HOLD_S,
            overwrite=False,
        )
        conn.commit()

        ev_count = int(
            cur.execute(
                "SELECT COUNT(*) AS n FROM events WHERE camera_id = ? AND clip_id = ?;",
                (target_cam, target_clip),
            ).fetchone()["n"] or 0
        )
        logger.info(
            "zones.recompute source_camera_id=%s target_camera_id=%s zone_camera_id=%s clip_id=%s zones_loaded=%d zone_ids=%s rows_scanned=%d rows_zoned=%d events_deleted=%d events_rebuilt=%d",
            camera_id,
            target_cam,
            zone_cam,
            target_clip,
            len(zone_debug),
            [z["zone_id"] for z in zone_debug],
            len(rows),
            zoned_rows,
            old_events,
            ev_count,
        )

        return {
            "ok": True,
            "source_camera_id": camera_id,
            "requested_clip_id": requested_clip_id,
            "target_camera_id": target_cam,
            "target_clip_id": target_clip,
            "clip_resolved": bool(requested_clip_id and requested_clip_id != target_clip),
            "zone_camera_id": zone_cam,
            "point_mode": point_mode,
            "rows_scanned": len(rows),
            "rows_previously_zoned": pre_zoned_rows,
            "rows_zoned": zoned_rows,
            "events_deleted": old_events,
            "events_rebuilt": ev_count,
            "events_after_rebuild": ev_count,
            "zones_loaded_from_db": len(zone_debug),
            "zones_used": zone_debug,
            "zones_source": "db.zones",
        }
    finally:
        conn.close()


@app.get("/zones/frame")
def api_frame(clip_id: str = Query(...), frame_idx: int = Query(0)) -> Response:
    try:
        clip_id = _clean_clip_id(clip_id)
        if not clip_id:
            raise HTTPException(400, "clip_id required")

        video_path = _clip_id_to_path(clip_id)
        if not video_path.exists():
            raise HTTPException(404, f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise HTTPException(500, f"Cannot open video: {video_path}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
        ok, frame = cap.read()
        cap.release()

        if not ok or frame is None:
            raise HTTPException(500, f"Cannot read frame={frame_idx} from: {video_path}")

        ok2, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok2:
            raise HTTPException(500, "Cannot encode jpg")

        return Response(content=buf.tobytes(), media_type="image/jpeg")

    except HTTPException:
        raise
    except Exception as e:
        print("[zones/frame] ERROR:", repr(e))
        raise HTTPException(500, f"zones/frame crashed: {e}")


@app.get("/kpi/ab-insights")
def kpi_ab_insights(
    cam_a: str = "cam1",
    cam_b: str = "cam1_hot",
    clip_a: Optional[str] = None,
    clip_b: Optional[str] = None,
):
    if clip_a and clip_b:
        a = zone_summary(camera_id=cam_a, clip_id=clip_a, group=None)
        b = zone_summary(camera_id=cam_b, clip_id=clip_b, group=None)
        return ab_insights(
            cam_a=cam_a,
            cam_b=cam_b,
            per_zone_a=a.get("per_zone", {}),
            per_zone_b=b.get("per_zone", {}),
            n_persons_a=int(a.get("n_persons", 0)),
            n_persons_b=int(b.get("n_persons", 0)),
        )

    a = zone_summary(camera_id=cam_a, clip_id=None, group=None)
    b = zone_summary(camera_id=cam_b, clip_id=None, group=None)
    return ab_insights(
        cam_a=cam_a,
        cam_b=cam_b,
        per_zone_a=a.get("per_zone", {}),
        per_zone_b=b.get("per_zone", {}),
        n_persons_a=int(a.get("n_persons", 0)),
        n_persons_b=int(b.get("n_persons", 0)),
    )
