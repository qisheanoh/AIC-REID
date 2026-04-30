# src/zones/engine.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Literal

import yaml

Point = Tuple[float, float]
PointMode = Literal["foot", "ankle", "center"]


@dataclass
class Zone:
    zone_id: str
    name: str
    polygon: List[Point]
    zone_type: str = "other"


@dataclass
class ZoneConfig:
    camera_id: str
    zones: List[Zone]

    @classmethod
    def from_yaml(cls, path: Path) -> "ZoneConfig":
        """
        Expected YAML format:

        camera_id: cam1
        zones:
          - zone_id: entrance
            name: Entrance
            zone_type: entrance
            polygon:
              - [x, y]
              - [x, y]
              - [x, y]
        """
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        camera_id = str(data.get("camera_id", "cam")).strip()
        zones_raw = data.get("zones", []) or []

        zones: List[Zone] = []
        for z in zones_raw:
            zid = str(z.get("zone_id", "")).strip()
            if not zid:
                continue
            name = str(z.get("name", zid)).strip() or zid
            ztype = str(z.get("zone_type", "other")).strip() or "other"
            poly_raw = z.get("polygon", []) or []
            poly = [(float(x), float(y)) for x, y in poly_raw]
            if len(poly) < 3:
                raise ValueError(f"Zone {zid} polygon must have >= 3 points")
            zones.append(Zone(zone_id=zid, name=name, polygon=poly, zone_type=ztype))

        return cls(camera_id=camera_id, zones=zones)


def point_in_poly(x: float, y: float, poly: List[Point]) -> bool:
    """Ray-casting point-in-polygon."""
    inside = False
    n = len(poly)
    if n < 3:
        return False
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_at_y = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
            if x < x_at_y:
                inside = not inside
    return inside


def point_from_box(box: Tuple[float, float, float, float], mode: PointMode = "ankle") -> Point:
    """
    Convert a bbox to a membership point.

    - foot:   (center_x, y2)
    - ankle:  (center_x, y2 - 2.0)  # consistent with run_stream.py
    - center: (center_x, center_y)
    """
    x1, y1, x2, y2 = box
    cx = 0.5 * (x1 + x2)
    m = (mode or "ankle").lower()

    if m == "center":
        cy = 0.5 * (y1 + y2)
    elif m == "foot":
        cy = y2
    else:
        cy = y2 - 2.0
    return float(cx), float(cy)


@dataclass
class Visit:
    zone_id: str
    t_enter: float
    t_exit: float


class ZoneEngine:
    """
    Online zone state engine:
      - assigns a stable zone per global_id
      - uses hysteresis_s to avoid flicker switches
      - records per-zone visit intervals
    """

    def __init__(
        self,
        cfg: ZoneConfig,
        fps: float,
        hysteresis_s: float = 1.0,
        point_mode: PointMode = "ankle",
    ):
        self.cfg = cfg
        self.fps = float(fps)
        self.hysteresis_s = float(hysteresis_s)
        self.point_mode: PointMode = point_mode

        # stable zone state
        self._curr_zone: Dict[int, Optional[str]] = {}        # gid -> stable zone_id or None
        self._zone_enter_ts: Dict[int, float] = {}            # gid -> ts when stable zone started

        # candidate zone state (for hysteresis)
        self._cand_zone: Dict[int, Optional[str]] = {}        # gid -> candidate zone
        self._cand_since_ts: Dict[int, float] = {}            # gid -> candidate start ts

        self._visits: Dict[int, List[Visit]] = {}             # gid -> list of closed visits
        self._seen_ids: set[int] = set()
        self._last_seen_ts: float = 0.0
        self._closed: bool = False

    def _zone_for_point(self, x: float, y: float) -> Optional[str]:
        for z in self.cfg.zones:
            if point_in_poly(x, y, z.polygon):
                return z.zone_id
        return None

    def update_for_box(
        self,
        global_id: int,
        frame_idx: int,
        ts_sec: float,
        box: Tuple[float, float, float, float],
    ) -> Optional[str]:
        """
        Update zone state for one bbox.
        Returns CURRENT STABLE zone_id (or None).
        """
        gid = int(global_id)
        ts = float(ts_sec)

        self._seen_ids.add(gid)
        self._last_seen_ts = max(self._last_seen_ts, ts)
        self._closed = False

        px, py = point_from_box(box, mode=self.point_mode)
        z_now = self._zone_for_point(px, py)

        # first time
        if gid not in self._curr_zone:
            self._curr_zone[gid] = z_now
            self._zone_enter_ts[gid] = ts
            self._cand_zone[gid] = z_now
            self._cand_since_ts[gid] = ts
            return z_now

        z_prev = self._curr_zone.get(gid)

        # no change observed -> reset candidate to stable
        if z_now == z_prev:
            self._cand_zone[gid] = z_prev
            self._cand_since_ts[gid] = ts
            return z_prev

        # candidate logic
        cand = self._cand_zone.get(gid)
        cand_since = float(self._cand_since_ts.get(gid, ts))

        # new candidate started
        if z_now != cand:
            self._cand_zone[gid] = z_now
            self._cand_since_ts[gid] = ts
            return z_prev

        # same candidate continues; must persist long enough
        if (ts - cand_since) < self.hysteresis_s:
            return z_prev

        # commit switch
        t_enter_prev = float(self._zone_enter_ts.get(gid, ts))
        if z_prev is not None:
            self._visits.setdefault(gid, []).append(
                Visit(zone_id=z_prev, t_enter=t_enter_prev, t_exit=ts)
            )

        self._curr_zone[gid] = z_now
        self._zone_enter_ts[gid] = ts
        self._cand_zone[gid] = z_now
        self._cand_since_ts[gid] = ts

        return z_now

    def close_all_visits(self, final_ts: float) -> None:
        """
        Close any currently-open stable visits once (idempotent).
        """
        if self._closed:
            return

        final_ts = float(final_ts)
        for gid, z_prev in list(self._curr_zone.items()):
            if z_prev is None:
                continue
            t_enter_prev = float(self._zone_enter_ts.get(gid, final_ts))
            if final_ts > t_enter_prev:
                self._visits.setdefault(gid, []).append(
                    Visit(zone_id=z_prev, t_enter=t_enter_prev, t_exit=final_ts)
                )

        self._closed = True

    def compute_kpis(
        self,
        final_ts: Optional[float] = None,
        bounce_tau_s: float = 3.0,
        min_session_dwell_s: float = 1.0,
        debug: bool = False,
    ) -> Dict:
        """
        Computes a simple online KPI summary:
          - per_zone: total dwell, visits, avg dwell
          - bounce_rate: persons with total_dwell < bounce_tau_s
          - n_persons: number of valid persons
        """
        if final_ts is None:
            final_ts = self._last_seen_ts
        self.close_all_visits(final_ts)

        per_zone: Dict[str, Dict[str, float]] = {}
        per_person_total_dwell: Dict[int, float] = {gid: 0.0 for gid in self._seen_ids}

        for gid, visits in self._visits.items():
            total = 0.0
            for v in visits:
                dwell = max(0.0, float(v.t_exit) - float(v.t_enter))
                total += dwell

                if v.zone_id not in per_zone:
                    per_zone[v.zone_id] = {"dwell_s": 0.0, "visits": 0}
                per_zone[v.zone_id]["dwell_s"] += dwell
                per_zone[v.zone_id]["visits"] += 1

            per_person_total_dwell[gid] = total

        for zid, stats in per_zone.items():
            visits = float(stats["visits"])
            stats["avg_dwell_s"] = (stats["dwell_s"] / visits) if visits > 0 else 0.0

        valid_persons = {
            gid: dwell
            for gid, dwell in per_person_total_dwell.items()
            if dwell >= float(min_session_dwell_s)
        }
        if not valid_persons:
            valid_persons = per_person_total_dwell

        if debug:
            print("[DEBUG] per_person_total_dwell:", per_person_total_dwell)
            print("[DEBUG] valid_persons:", valid_persons)

        n_persons = len(valid_persons)
        n_bounce = sum(1 for dwell in valid_persons.values() if dwell < float(bounce_tau_s))
        bounce_rate = (n_bounce / n_persons) if n_persons > 0 else 0.0

        return {"per_zone": per_zone, "bounce_rate": bounce_rate, "n_persons": n_persons}
