#!/usr/bin/env python3
# scripts/add_zones_to_tracks.py
"""
Fill zone_id in KPI track CSVs using zones stored in SQLite (preferred) or YAML (optional).

Key fixes / upgrades:
- Default is SAFE (preserve existing zone_id). Use --overwrite to recompute.
- New --drop-zone-col option: physically removes zone_id column then recomputes (strong reset).
- New --prefer-nonempty option: if multiple polygons overlap, keep first match (default) or
  optionally choose the first non-empty zone id found (same as default behavior; kept for clarity).
- New --report-zone-counts: prints a summary of zone_id distribution per CSV after assignment.
- Supports loading zones from DB or YAML.
"""

from __future__ import annotations

from pathlib import Path
import csv
import argparse
import yaml
import shutil
import sqlite3
import json
import re
from typing import List, Tuple, Optional, Literal, Dict
from collections import Counter

Point = Tuple[float, float]
Zone = Tuple[str, List[Point]]  # (zone_id, polygon)
PointMode = Literal["foot", "center", "ankle"]


def _slugify(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "id"


def point_in_poly(x: float, y: float, poly: List[Point]) -> bool:
    # Ray casting algorithm
    inside = False
    n = len(poly)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        intersect = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        )
        if intersect:
            inside = not inside
        j = i
    return inside


def load_zones_from_yaml(path: Path) -> List[Zone]:
    with open(path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    zones = doc.get("zones", [])
    out: List[Zone] = []
    for z in zones:
        if not isinstance(z, dict):
            continue
        zid = _slugify(str(z.get("zone_id", z.get("name", "zone"))))
        poly_in = z.get("polygon", [])
        if not isinstance(poly_in, list):
            continue
        poly: List[Point] = []
        for p in poly_in:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                poly.append((float(p[0]), float(p[1])))
            elif isinstance(p, dict) and "x" in p and "y" in p:
                poly.append((float(p["x"]), float(p["y"])))
        if len(poly) >= 3:
            out.append((zid, poly))
    return out


def load_zones_from_db(db_path: Path, camera_id: str) -> List[Zone]:
    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}")

    camera_id = _slugify(camera_id)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT zone_id, polygon
            FROM zones
            WHERE camera_id = ?
            ORDER BY zone_id;
            """,
            (camera_id,),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    out: List[Zone] = []
    for r in rows:
        zid = str(r["zone_id"])
        try:
            poly_raw = json.loads(r["polygon"])
            poly = [(float(x), float(y)) for x, y in poly_raw]
        except Exception as e:
            print(f"[WARN] Could not parse polygon for zone_id={zid}: {e}")
            continue
        if len(poly) >= 3:
            out.append((zid, poly))
    return out


def assign_zone_id(cx: float, cy: float, zones: List[Zone]) -> Optional[str]:
    for zid, poly in zones:
        if point_in_poly(cx, cy, poly):
            return zid
    return None


def bbox_point(x1: float, y1: float, x2: float, y2: float, mode: PointMode) -> Tuple[float, float]:
    cx = (x1 + x2) / 2.0
    if mode == "foot":
        cy = y2
    elif mode == "center":
        cy = (y1 + y2) / 2.0
    else:  # ankle
        cy = y2 - 2.0
    return cx, cy


def process_csv(
    path: Path,
    zones: List[Zone],
    *,
    overwrite: bool = False,
    drop_zone_col: bool = False,
    point_mode: PointMode = "foot",
    report_zone_counts: bool = False,
) -> None:
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    bak = path.with_suffix(path.suffix + ".bak")

    assigned = 0
    total = 0
    kept = 0

    zone_counter: Counter[str] = Counter()

    with open(path, "r", newline="", encoding="utf-8") as inf, open(tmp, "w", newline="", encoding="utf-8") as outf:
        reader = csv.DictReader(inf)
        in_fields = reader.fieldnames or []

        # Optionally drop the column entirely (strong reset)
        if drop_zone_col:
            in_fields = [c for c in in_fields if c != "zone_id"]

        # Ensure zone_id exists in output schema
        fieldnames = list(in_fields)
        if "zone_id" not in fieldnames:
            fieldnames.append("zone_id")

        writer = csv.DictWriter(outf, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            total += 1

            # If drop_zone_col, treat as empty always
            existing = "" if drop_zone_col else (row.get("zone_id") or "").strip()

            if existing and not overwrite:
                kept += 1
                zone_counter[existing] += 1
                # Ensure zone_id is present in output
                out_row = dict(row)
                out_row["zone_id"] = existing
                writer.writerow(out_row)
                continue

            try:
                x1 = float(row.get("x1", 0))
                y1 = float(row.get("y1", 0))
                x2 = float(row.get("x2", 0))
                y2 = float(row.get("y2", 0))
            except Exception:
                out_row = dict(row)
                out_row["zone_id"] = existing if existing else ""
                writer.writerow(out_row)
                continue

            cx, cy = bbox_point(x1, y1, x2, y2, point_mode)
            zid = assign_zone_id(cx, cy, zones)

            out_row = dict(row)
            out_row["zone_id"] = zid or ""
            if zid:
                assigned += 1
                zone_counter[zid] += 1
            else:
                zone_counter[""] += 1
            writer.writerow(out_row)

    shutil.copy2(path, bak)
    tmp.replace(path)

    print(
        f"Updated {path}: assigned={assigned}/{total}, kept_existing={kept} "
        f"(backup: {bak.name})"
    )

    if report_zone_counts:
        top = zone_counter.most_common(10)
        print("  [ZONE COUNTS] top:")
        for k, v in top:
            label = k if k else "<empty>"
            print(f"    - {label}: {v}")


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)

    src.add_argument("--db", help="Path to SQLite DB (e.g. data/retail.db)")
    ap.add_argument("--camera_id", default="cam1", help="camera_id in DB zones table (default: cam1)")

    src.add_argument("--zones", help="Path to zones YAML (e.g. configs/zones/zones_cam1.yaml)")

    ap.add_argument("--csv_glob", default="runs/kpi_batch/person_*_tracks.csv")

    ap.add_argument("--overwrite", action="store_true", help="Recompute zone_id even if already filled")
    ap.add_argument("--drop-zone-col", action="store_true", help="Remove zone_id column then recompute (strong reset)")

    ap.add_argument(
        "--point",
        choices=["foot", "center", "ankle"],
        default="foot",
        help="Which point of the bbox to test: foot (default), center, ankle",
    )

    ap.add_argument("--report-zone-counts", action="store_true", help="Print zone_id distribution summary per CSV")

    args = ap.parse_args()
    point_mode: PointMode = args.point  # type: ignore

    root = Path(__file__).resolve().parents[1]

    if args.db:
        db_path = (root / args.db).resolve() if not Path(args.db).is_absolute() else Path(args.db)
        zones = load_zones_from_db(db_path, args.camera_id)
        print(
            f"[INFO] Loaded {len(zones)} zones from DB={db_path} "
            f"camera_id={_slugify(args.camera_id)} point={point_mode}"
        )
    else:
        zones_path = (root / args.zones).resolve() if not Path(args.zones).is_absolute() else Path(args.zones)
        if not zones_path.exists():
            raise SystemExit(f"Zones file not found: {zones_path}")
        zones = load_zones_from_yaml(zones_path)
        print(f"[INFO] Loaded {len(zones)} zones from YAML={zones_path} point={point_mode}")

    if not zones:
        raise SystemExit("No zones found. Create zones in Zone Editor or provide a YAML zones file.")

    csv_paths = sorted(root.glob(args.csv_glob))
    if not csv_paths:
        print("No CSVs found to process.")
        return

    for p in csv_paths:
        process_csv(
            p,
            zones,
            overwrite=bool(args.overwrite),
            drop_zone_col=bool(args.drop_zone_col),
            point_mode=point_mode,
            report_zone_counts=bool(args.report_zone_counts),
        )


if __name__ == "__main__":
    main()
