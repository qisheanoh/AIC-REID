# src/kpi/engine.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
import sqlite3
from dataclasses import dataclass
from collections import defaultdict

from src.storage.db import fetch_tracks, fetch_events, insert_events_bulk


@dataclass
class Event:
    camera_id: str
    clip_id: str
    global_id: int
    zone_id: str
    t_enter: float
    t_exit: float

    @property
    def dwell_s(self) -> float:
        return max(0.0, self.t_exit - self.t_enter)


_NO_CAND = object()  # sentinel: no active candidate zone (distinct from candidate=None zone)


def _build_events_from_tracks(
    rows: List[sqlite3.Row],
    camera_id: str,
    clip_id: str,
    *,
    min_hold_s: float = 0.0,
) -> List[Event]:
    events: List[Event] = []

    by_person: Dict[int, List[sqlite3.Row]] = defaultdict(list)
    for r in rows:
        by_person[int(r["global_id"])].append(r)

    for gid, trs in by_person.items():
        trs = sorted(trs, key=lambda r: float(r["ts"]))

        cur_zone: Optional[str] = None
        cur_start: Optional[float] = None

        # Use sentinel to distinguish "no pending candidate" from "candidate is None zone".
        # Previously both were represented as cand_zone=None, causing the hysteresis timer
        # to reset every frame when a person was outside all zones (z=None), so zone-exit
        # events were never committed via hysteresis.
        cand_zone: Any = _NO_CAND
        cand_start_t: Optional[float] = None

        def close_event(zone: str, t_exit: float):
            nonlocal cur_start
            if cur_start is None:
                return
            if t_exit > cur_start:
                events.append(
                    Event(
                        camera_id=camera_id,
                        clip_id=clip_id,
                        global_id=gid,
                        zone_id=zone,
                        t_enter=cur_start,
                        t_exit=t_exit,
                    )
                )

        for r in trs:
            z = r["zone_id"]  # may be None/""
            z = (str(z).strip() if z is not None else None)
            if z == "":
                z = None

            t = float(r["ts"])

            if min_hold_s <= 0.0:
                if cur_zone is None and z is not None:
                    cur_zone = z
                    cur_start = t
                elif cur_zone is not None and z == cur_zone:
                    pass
                elif cur_zone is not None and z != cur_zone:
                    close_event(cur_zone, t)
                    cur_zone = None
                    cur_start = None
                    if z is not None:
                        cur_zone = z
                        cur_start = t
                continue

            # hysteresis mode
            # Fix A: only cancel a pending candidate when re-observing a real (non-None)
            # stable zone.  Previously "if z == cur_zone" fired when both were None,
            # which killed a valid zone candidate every frame a person was between zones.
            if z == cur_zone and cur_zone is not None:
                # Re-confirmed in stable zone; cancel any pending candidate switch.
                cand_zone = _NO_CAND
                cand_start_t = None
                continue

            if cand_zone is _NO_CAND:
                # No active candidate yet; start tracking this new observation.
                cand_zone = z
                cand_start_t = t
                continue

            if z != cand_zone:
                # Fix B: candidate changed before the new observation, but if the
                # departing candidate already held >= min_hold_s, commit it now so
                # a person who leaves mid-candidate doesn't lose their event.
                if cand_start_t is not None and (t - cand_start_t) >= min_hold_s:
                    if cur_zone is not None:
                        close_event(cur_zone, cand_start_t)
                    cur_zone = cand_zone  # may be None
                    cur_start = cand_start_t if cur_zone is not None else None
                cand_zone = z
                cand_start_t = t
                continue

            # Same candidate continues; check if hold period has elapsed.
            assert cand_start_t is not None
            if (t - cand_start_t) >= min_hold_s:
                if cur_zone is not None:
                    close_event(cur_zone, cand_start_t)
                cur_zone = cand_zone  # may be None (person left all zones)
                cur_start = cand_start_t if cur_zone is not None else None
                cand_zone = _NO_CAND
                cand_start_t = None

        if cur_zone is not None and cur_start is not None and trs:
            end_t = float(trs[-1]["ts"])
            close_event(cur_zone, end_t)

    return events


def clear_events(
    conn: sqlite3.Connection,
    camera_id: str,
    clip_id: Optional[str] = None,
    t_min: Optional[float] = None,
    t_max: Optional[float] = None,
) -> None:
    camera_id = str(camera_id)

    where = ["camera_id = ?"]
    params: List[object] = [camera_id]

    if clip_id is not None:
        where.append("clip_id = ?")
        params.append(str(clip_id))

    # overlap delete on [t_min, t_max]
    if t_min is not None and t_max is not None:
        where.append("NOT (t_exit < ? OR t_enter > ?)")
        params.extend([float(t_min), float(t_max)])
    elif t_min is not None:
        where.append("t_exit >= ?")
        params.append(float(t_min))
    elif t_max is not None:
        where.append("t_enter <= ?")
        params.append(float(t_max))

    conn.execute("DELETE FROM events WHERE " + " AND ".join(where), params)


def refresh_events_from_tracks(
    conn: sqlite3.Connection,
    camera_id: str,
    clip_id: Optional[str] = None,
    t_min: Optional[float] = None,
    t_max: Optional[float] = None,
    *,
    min_hold_s: float = 0.0,
    overwrite: bool = True,
) -> None:
    tracks = fetch_tracks(conn, camera_id=camera_id, clip_id=clip_id, t_min=t_min, t_max=t_max)

    # If clip_id is None (all clips), build per-clip events separately
    if clip_id is None:
        by_clip: Dict[str, List[sqlite3.Row]] = defaultdict(list)
        for r in tracks:
            by_clip[str(r["clip_id"])].append(r)

        if overwrite:
            clear_events(conn, camera_id=camera_id, clip_id=None, t_min=t_min, t_max=t_max)

        payload: List[tuple] = []
        for cid, rows in by_clip.items():
            evs = _build_events_from_tracks(rows, camera_id, cid, min_hold_s=min_hold_s)
            payload.extend(
                (e.camera_id, e.clip_id, e.global_id, e.zone_id, e.t_enter, e.t_exit, e.dwell_s)
                for e in evs
                if e.dwell_s > 0.0
            )
        if payload:
            insert_events_bulk(conn, payload)
        return

    # Single clip mode
    evs = _build_events_from_tracks(tracks, camera_id, str(clip_id), min_hold_s=min_hold_s)

    payload = [
        (e.camera_id, e.clip_id, e.global_id, e.zone_id, e.t_enter, e.t_exit, e.dwell_s)
        for e in evs
        if e.dwell_s > 0.0
    ]
    if not payload:
        return

    if overwrite:
        clear_events(conn, camera_id=camera_id, clip_id=clip_id, t_min=t_min, t_max=t_max)

    insert_events_bulk(conn, payload)


def compute_zone_kpis(
    conn: sqlite3.Connection,
    camera_id: str,
    clip_id: Optional[str] = None,
    t_min: Optional[float] = None,
    t_max: Optional[float] = None,
    dwell_threshold_bounce: float = 3.0,
) -> Dict:
    evs = fetch_events(conn, camera_id=camera_id, clip_id=clip_id, t_min=t_min, t_max=t_max)

    per_zone = defaultdict(lambda: {"dwell_s": 0.0, "visits": 0})
    per_person_best_dwell = defaultdict(float)

    for e in evs:
        zid = e["zone_id"]
        gid = int(e["global_id"])
        dwell_s = float(e["dwell_s"])

        per_zone[zid]["dwell_s"] += dwell_s
        per_zone[zid]["visits"] += 1
        if dwell_s > per_person_best_dwell[gid]:
            per_person_best_dwell[gid] = dwell_s

    n_persons = len(per_person_best_dwell)
    n_bounce = sum(1 for best in per_person_best_dwell.values() if best < float(dwell_threshold_bounce))
    bounce_rate = float(n_bounce) / float(n_persons) if n_persons > 0 else 0.0

    out_per_zone = {}
    for zid, stats in per_zone.items():
        visits = int(stats["visits"])
        dwell_s = float(stats["dwell_s"])
        out_per_zone[zid] = {
            "dwell_s": dwell_s,
            "visits": visits,
            "avg_dwell_s": (dwell_s / visits) if visits > 0 else 0.0,
        }

    return {"per_zone": out_per_zone, "bounce_rate": bounce_rate, "n_persons": n_persons}
