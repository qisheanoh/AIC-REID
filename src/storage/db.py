# src/storage/db.py
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, List, Tuple, Optional, Dict, Any
import json
import re


def get_connection(db_path: str | Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row

    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute("PRAGMA temp_store=MEMORY;")
    return conn


def _has_column(conn: sqlite3.Connection, table: str, col: str) -> bool:
    cur = conn.cursor()
    rows = cur.execute(f"PRAGMA table_info({table});").fetchall()
    return any(r["name"] == col for r in rows)


def _slugify(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "id"


def _norm_camera_id(s: str) -> str:
    return _slugify(s)


def _zones_pk_is_old(conn: sqlite3.Connection) -> bool:
    cur = conn.cursor()
    rows = cur.execute("PRAGMA table_info(zones);").fetchall()
    if not rows:
        return False
    pk_cols = [r["name"] for r in rows if int(r["pk"] or 0) > 0]
    return pk_cols == ["zone_id"]


def _migrate_zones_to_composite_pk(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    if not _zones_pk_is_old(conn):
        return

    has_zone_type = _has_column(conn, "zones", "zone_type")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS zones_v2 (
        camera_id TEXT NOT NULL,
        zone_id TEXT NOT NULL,
        name TEXT NOT NULL,
        polygon TEXT NOT NULL,
        zone_type TEXT DEFAULT 'other',
        PRIMARY KEY (camera_id, zone_id)
    );
    """)

    # Fetch data first
    if has_zone_type:
        rows = cur.execute("SELECT camera_id, zone_id, name, polygon, COALESCE(zone_type, 'other') as zone_type FROM zones").fetchall()
    else:
        rows = cur.execute("SELECT camera_id, zone_id, name, polygon, 'other' as zone_type FROM zones").fetchall()

    # Process in Python and Insert
    for r in rows:
        # Normalize camera_id using the Python function
        cid = _norm_camera_id(r['camera_id'])
        
        cur.execute("""
            INSERT OR REPLACE INTO zones_v2(camera_id, zone_id, name, polygon, zone_type)
            VALUES (?, ?, ?, ?, ?)
        """, (cid, r['zone_id'], r['name'], r['polygon'], r['zone_type']))

    cur.execute("DROP TABLE zones;")
    cur.execute("ALTER TABLE zones_v2 RENAME TO zones;")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_zones_camera ON zones(camera_id);")
    
def _normalize_existing_camera_ids(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute("""
    UPDATE zones
    SET camera_id = lower(trim(replace(camera_id,' ','_')))
    WHERE camera_id != lower(trim(replace(camera_id,' ','_')));
    """)


def init_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS tracks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts REAL NOT NULL,
        frame_idx INTEGER NOT NULL,
        camera_id TEXT NOT NULL,
        clip_id TEXT NOT NULL,
        global_id INTEGER NOT NULL,
        x1 REAL NOT NULL,
        y1 REAL NOT NULL,
        x2 REAL NOT NULL,
        y2 REAL NOT NULL,
        zone_id TEXT,
        conf REAL
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS zones (
        camera_id TEXT NOT NULL,
        zone_id TEXT NOT NULL,
        name TEXT NOT NULL,
        polygon TEXT NOT NULL,
        zone_type TEXT DEFAULT 'other',
        PRIMARY KEY (camera_id, zone_id)
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        camera_id TEXT NOT NULL,
        clip_id TEXT NOT NULL,
        global_id INTEGER NOT NULL,
        zone_id TEXT NOT NULL,
        t_enter REAL NOT NULL,
        t_exit REAL NOT NULL,
        dwell_s REAL NOT NULL
    );
    """)

    # migrations / normalization (safe)
    try:
        _migrate_zones_to_composite_pk(conn)
    except Exception:
        pass
    try:
        _normalize_existing_camera_ids(conn)
    except Exception:
        pass

    # indexes
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tracks_cam_clip_ts ON tracks(camera_id, clip_id, ts);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tracks_cam_clip_gid_ts ON tracks(camera_id, clip_id, global_id, ts);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_events_cam_clip_enter ON events(camera_id, clip_id, t_enter);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_events_cam_clip_gid_enter ON events(camera_id, clip_id, global_id, t_enter);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_events_cam_clip_zone ON events(camera_id, clip_id, zone_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_zones_camera ON zones(camera_id);")

    conn.commit()


# ---------------------------
# Zones CRUD
# ---------------------------

def upsert_zone(
    conn: sqlite3.Connection,
    camera_id: str,
    zone_id: str,
    name: str,
    polygon: Iterable[Tuple[float, float]],
    zone_type: str = "other",
    preserve_polygon: bool = False,
) -> None:
    camera_id = _norm_camera_id(camera_id)
    zone_id = _slugify(zone_id)
    name = str(name).strip() or zone_id
    zone_type = str(zone_type).strip() or "other"

    poly_json = json.dumps([[float(x), float(y)] for (x, y) in polygon])

    cur = conn.cursor()
    if preserve_polygon:
        # Used by YAML sync: create the zone if missing (with YAML polygon as default),
        # but do NOT overwrite a polygon that the zone editor has already set.
        cur.execute("""
        INSERT INTO zones(camera_id, zone_id, name, polygon, zone_type)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(camera_id, zone_id) DO UPDATE SET
          name=excluded.name,
          zone_type=excluded.zone_type;
        """, (camera_id, zone_id, name, poly_json, zone_type))
    else:
        cur.execute("""
        INSERT INTO zones(camera_id, zone_id, name, polygon, zone_type)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(camera_id, zone_id) DO UPDATE SET
          name=excluded.name,
          polygon=excluded.polygon,
          zone_type=excluded.zone_type;
        """, (camera_id, zone_id, name, poly_json, zone_type))


def list_zones(conn: sqlite3.Connection, camera_id: str) -> List[Dict[str, Any]]:
    camera_id = _norm_camera_id(camera_id)

    cur = conn.cursor()
    rows = cur.execute(
        "SELECT * FROM zones WHERE camera_id=? ORDER BY name ASC;",
        (camera_id,),
    ).fetchall()

    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append({
            "zone_id": r["zone_id"],
            "camera_id": r["camera_id"],
            "name": r["name"],
            "polygon": json.loads(r["polygon"]),
            "zone_type": r["zone_type"] if "zone_type" in r.keys() else "other",
        })
    return out


def delete_zone(conn: sqlite3.Connection, camera_id: str, zone_id: str) -> None:
    camera_id = _norm_camera_id(camera_id)
    zone_id = _slugify(zone_id)

    cur = conn.cursor()
    cur.execute("DELETE FROM zones WHERE camera_id=? AND zone_id=?;", (camera_id, zone_id))


# ---------------------------
# Tracks / Events insert
# ---------------------------

def insert_track(
    conn: sqlite3.Connection,
    *,
    ts: float,
    frame_idx: int,
    camera_id: str,
    clip_id: str,
    global_id: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    zone_id: Optional[str],
    conf: Optional[float],
) -> None:
    camera_id = _norm_camera_id(camera_id)
    clip_id = str(clip_id).strip() or "clip"
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO tracks(ts, frame_idx, camera_id, clip_id, global_id,
                       x1, y1, x2, y2, zone_id, conf)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, (float(ts), int(frame_idx), camera_id, clip_id, int(global_id),
          float(x1), float(y1), float(x2), float(y2),
          zone_id, conf))


def insert_tracks_bulk(
    conn: sqlite3.Connection,
    rows: List[Tuple[float, int, str, str, int, float, float, float, float, Optional[str], Optional[float]]],
) -> None:
    if not rows:
        return
    cur = conn.cursor()
    cur.executemany("""
    INSERT INTO tracks(ts, frame_idx, camera_id, clip_id, global_id,
                       x1, y1, x2, y2, zone_id, conf)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, rows)


def insert_events_bulk(
    conn: sqlite3.Connection,
    events: List[Tuple[str, str, int, str, float, float, float]],
) -> None:
    if not events:
        return
    cur = conn.cursor()
    cur.executemany("""
    INSERT INTO events(camera_id, clip_id, global_id, zone_id, t_enter, t_exit, dwell_s)
    VALUES (?, ?, ?, ?, ?, ?, ?);
    """, events)


def delete_clip_data(conn: sqlite3.Connection, camera_id: str, clip_id: str) -> None:
    camera_id = _norm_camera_id(camera_id)
    clip_id = str(clip_id).strip()
    cur = conn.cursor()
    cur.execute("DELETE FROM events WHERE camera_id=? AND clip_id=?;", (camera_id, clip_id))
    cur.execute("DELETE FROM tracks WHERE camera_id=? AND clip_id=?;", (camera_id, clip_id))


# ---------------------------
# Fetch helpers (used by KPI engine)
# ---------------------------

def fetch_tracks(
    conn: sqlite3.Connection,
    camera_id: str,
    clip_id: Optional[str] = None,
    t_min: Optional[float] = None,
    t_max: Optional[float] = None,
) -> List[sqlite3.Row]:
    camera_id = _norm_camera_id(camera_id)
    params: List[Any] = [camera_id]
    where = ["camera_id = ?"]

    if clip_id is not None:
        where.append("clip_id = ?")
        params.append(str(clip_id))

    if t_min is not None:
        where.append("ts >= ?")
        params.append(float(t_min))
    if t_max is not None:
        where.append("ts <= ?")
        params.append(float(t_max))

    q = "SELECT * FROM tracks WHERE " + " AND ".join(where) + " ORDER BY ts ASC;"
    return conn.execute(q, params).fetchall()


def fetch_events(
    conn: sqlite3.Connection,
    camera_id: str,
    clip_id: Optional[str] = None,
    t_min: Optional[float] = None,
    t_max: Optional[float] = None,
) -> List[sqlite3.Row]:
    camera_id = _norm_camera_id(camera_id)
    params: List[Any] = [camera_id]
    where = ["camera_id = ?"]

    if clip_id is not None:
        where.append("clip_id = ?")
        params.append(str(clip_id))

    if t_min is not None:
        where.append("t_enter >= ?")
        params.append(float(t_min))
    if t_max is not None:
        where.append("t_exit <= ?")
        params.append(float(t_max))

    q = "SELECT * FROM events WHERE " + " AND ".join(where) + " ORDER BY t_enter ASC;"
    return conn.execute(q, params).fetchall()
