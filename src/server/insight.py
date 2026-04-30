from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Insight:
    title: str
    severity: str
    message: str


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if v == v else default
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def summarize_kpis(
    *,
    camera_id: str,
    per_zone: Dict[str, Dict[str, Any]],
    n_persons: int,
) -> Dict[str, Any]:
    insights: List[Insight] = []

    zone_ids = sorted(per_zone.keys())
    if not zone_ids:
        if camera_id == "uploaded":
            person_word = "person" if n_persons == 1 else "persons"
            insights.append(
                Insight(
                    title="Track analytics ready",
                    severity="info",
                    message=(
                        f"{n_persons} {person_word} identified from the uploaded video. "
                        "Per-person appearance times are available in the table below. "
                        "To enable zone-level analytics (dwell time, visit counts, engagement), "
                        "draw zones for this camera in the Zone Editor."
                    ),
                )
            )
        else:
            insights.append(
                Insight(
                    title="No zone events found",
                    severity="warn",
                    message=f"No events exist for camera_id={camera_id}. Run add_zones_to_tracks + ingest.",
                )
            )
        return {
            "camera_id": camera_id,
            "zones": [],
            "n_persons": n_persons,
            "insights": [i.__dict__ for i in insights],
        }

    zone_summaries: List[Dict[str, Any]] = []
    total_visits = 0
    total_qualified_visits = 0
    total_dwell = 0.0

    for zid in zone_ids:
        stats = per_zone.get(zid, {})
        visits = _safe_int(stats.get("visits", 0))
        qualified_visits = _safe_int(stats.get("qualified_visits", 0))
        transit_visits = _safe_int(stats.get("transit_visits", 0))
        dwell_s = _safe_float(stats.get("dwell_s", 0.0))
        qualified_dwell_s = _safe_float(stats.get("qualified_dwell_s", 0.0))
        avg_dwell_s = _safe_float(stats.get("avg_dwell_s", 0.0))
        qualification_rate = _safe_float(stats.get("qualification_rate", 0.0))
        threshold_s = _safe_float(stats.get("qualification_threshold_s", 0.0))
        zone_kind = str(stats.get("zone_kind", "other"))

        total_visits += visits
        total_qualified_visits += qualified_visits
        total_dwell += dwell_s

        zone_summaries.append(
            {
                "zone_id": zid,
                "zone_kind": zone_kind,
                "qualification_threshold_s": threshold_s,
                "visits": visits,
                "qualified_visits": qualified_visits,
                "transit_visits": transit_visits,
                "dwell_s": dwell_s,
                "qualified_dwell_s": qualified_dwell_s,
                "avg_dwell_s": avg_dwell_s,
                "qualification_rate": qualification_rate,
            }
        )

    insights.append(
        Insight(
            title="Dataset summary",
            severity="info",
            message=(
                f"{n_persons} persons, {total_visits} zone visits, "
                f"{total_qualified_visits} qualified visits, "
                f"total dwell {total_dwell:.2f}s across {len(zone_ids)} zones."
            ),
        )
    )

    top_by_dwell = max(zone_summaries, key=lambda z: z["dwell_s"])
    insights.append(
        Insight(
            title="Top zone by total dwell (all zones)",
            severity="success",
            message=(
                f"{top_by_dwell['zone_id']}: {top_by_dwell['dwell_s']:.2f}s total dwell "
                f"over {top_by_dwell['visits']} visits (avg {top_by_dwell['avg_dwell_s']:.2f}s)."
            ),
        )
    )

    engagement_zones = [z for z in zone_summaries if z["zone_kind"] == "engagement"]
    if engagement_zones:
        top_engaged = max(engagement_zones, key=lambda z: z["qualified_dwell_s"])
        insights.append(
            Insight(
                title="Top engagement zone",
                severity="success",
                message=(
                    f"{top_engaged['zone_id']}: {top_engaged['qualified_visits']} qualified visits, "
                    f"{top_engaged['qualified_dwell_s']:.2f}s qualified dwell, "
                    f"qualification rate {top_engaged['qualification_rate'] * 100:.1f}%."
                ),
            )
        )

    flow_zone_ids: List[str] = []

    for z in zone_summaries:
        zid = z["zone_id"]
        avg = float(z["avg_dwell_s"])
        visits = int(z["visits"])
        qualified_visits = int(z["qualified_visits"])
        transit_visits = int(z["transit_visits"])
        rate = float(z["qualification_rate"])
        kind = z["zone_kind"]

        if kind == "entrance":
            if avg >= 8.0:
                insights.append(
                    Insight(
                        title="Entrance dwell is high",
                        severity="info",
                        message=f"{zid}: avg dwell {avg:.2f}s suggests pausing or congestion near entrance.",
                    )
                )
            elif avg <= 3.0 and visits > 5:
                insights.append(
                    Insight(
                        title="Entrance flow is smooth",
                        severity="success",
                        message=f"{zid}: avg dwell {avg:.2f}s suggests smooth movement through entrance.",
                    )
                )

        elif kind == "flow":
            flow_zone_ids.append(zid)

        elif kind == "engagement":
            if qualified_visits == 0 and visits > 0:
                insights.append(
                    Insight(
                        title="Mostly pass-through traffic",
                        severity="warn",
                        message=(
                            f"{zid}: {visits} visits but 0 qualified visits "
                            f"(threshold-based), suggesting customers mostly passed through."
                        ),
                    )
                )
            elif rate >= 0.6 and qualified_visits >= 3:
                insights.append(
                    Insight(
                        title="Engagement looks strong",
                        severity="success",
                        message=(
                            f"{zid}: {qualified_visits}/{visits} visits were qualified "
                            f"({rate * 100:.1f}%), suggesting meaningful shopper attention."
                        ),
                    )
                )
            elif transit_visits > qualified_visits and visits > 3:
                insights.append(
                    Insight(
                        title="Engagement is weaker than traffic",
                        severity="info",
                        message=(
                            f"{zid}: more transit visits than qualified visits, "
                            f"so raw entries may overstate true interest."
                        ),
                    )
                )

        elif kind == "checkout":
            if visits > 0 and avg >= 3.0:
                insights.append(
                    Insight(
                        title="Checkout or service activity detected",
                        severity="info",
                        message=f"{zid}: avg dwell {avg:.2f}s suggests queueing or service interaction.",
                    )
                )

    if flow_zone_ids:
        insights.append(
            Insight(
                title="Flow zones interpreted as traffic",
                severity="info",
                message=(
                    f"{', '.join(flow_zone_ids)} are treated as traffic-flow zones, "
                    f"not product-interest zones."
                ),
            )
        )

    return {
        "camera_id": camera_id,
        "zones": zone_summaries,
        "n_persons": n_persons,
        "insights": [i.__dict__ for i in insights],
    }


def _pct_change(new: float, old: float) -> Optional[float]:
    if old == 0:
        return None
    return (new - old) / old * 100.0


def _fmt_pct(p: Optional[float]) -> str:
    if p is None:
        return "N/A"
    return f"{p:+.1f}%"


def ab_insights(
    cam_a: str,
    cam_b: str,
    per_zone_a: Dict[str, Dict[str, float]],
    per_zone_b: Dict[str, Dict[str, float]],
    n_persons_a: int,
    n_persons_b: int,
) -> Dict[str, Any]:
    def totals(per_zone: Dict[str, Dict[str, float]]) -> Tuple[int, float]:
        visits = 0
        dwell = 0.0
        for _, s in per_zone.items():
            visits += int(s.get("visits", 0) or 0)
            dwell += float(s.get("dwell_s", 0.0) or 0.0)
        return visits, dwell

    visits_a, dwell_a = totals(per_zone_a)
    visits_b, dwell_b = totals(per_zone_b)
    avg_a = (dwell_a / visits_a) if visits_a > 0 else 0.0
    avg_b = (dwell_b / visits_b) if visits_b > 0 else 0.0

    zones = sorted(set(per_zone_a.keys()) | set(per_zone_b.keys()))
    per_zone_delta: List[Dict[str, Any]] = []

    for zid in zones:
        a = per_zone_a.get(zid, {})
        b = per_zone_b.get(zid, {})

        va = float(a.get("visits", 0) or 0)
        vb = float(b.get("visits", 0) or 0)
        da = float(a.get("dwell_s", 0.0) or 0.0)
        db = float(b.get("dwell_s", 0.0) or 0.0)

        aa = float(a.get("avg_dwell_s", (da / va) if va > 0 else 0.0) or 0.0)
        ab = float(b.get("avg_dwell_s", (db / vb) if vb > 0 else 0.0) or 0.0)

        per_zone_delta.append(
            {
                "zone_id": zid,
                "A": {"visits": int(va), "dwell_s": da, "avg_dwell_s": aa},
                "B": {"visits": int(vb), "dwell_s": db, "avg_dwell_s": ab},
                "delta": {
                    "visits_pct": _pct_change(vb, va),
                    "dwell_pct": _pct_change(db, da),
                    "avg_dwell_pct": _pct_change(ab, aa),
                },
            }
        )

    insights: List[Dict[str, str]] = []

    insights.append(
        {
            "title": "Overall change (B vs A)",
            "severity": "info",
            "message": (
                f"Persons: {n_persons_a} → {n_persons_b}. "
                f"Visits: {visits_a} → {visits_b} ({_fmt_pct(_pct_change(visits_b, visits_a))}). "
                f"Total dwell: {dwell_a:.2f}s → {dwell_b:.2f}s ({_fmt_pct(_pct_change(dwell_b, dwell_a))}). "
                f"Avg dwell/visit: {avg_a:.2f}s → {avg_b:.2f}s ({_fmt_pct(_pct_change(avg_b, avg_a))})."
            ),
        }
    )

    recs: List[str] = []
    recs.append(
        f"{cam_a} and {cam_b} should be interpreted as different operational modes, not a strict causal A/B test."
    )

    for r in per_zone_delta:
        zid = r["zone_id"]
        if "entrance" in zid and r["B"]["visits"] == 0 and r["A"]["visits"] > 0:
            recs.append("Entrance behaviour appears only in A.")
        if ("hot" in zid or "promo" in zid) and r["A"]["visits"] == 0 and r["B"]["visits"] > 0:
            recs.append("Hot/promo engagement appears mainly in B.")

    insights.append(
        {
            "title": "Interpretation & recommendations",
            "severity": "info",
            "message": " ".join(recs),
        }
    )

    return {
        "cam_a": cam_a,
        "cam_b": cam_b,
        "summary": {
            "A": {"n_persons": n_persons_a, "visits": visits_a, "dwell_s": dwell_a, "avg_dwell_s": avg_a},
            "B": {"n_persons": n_persons_b, "visits": visits_b, "dwell_s": dwell_b, "avg_dwell_s": avg_b},
        },
        "per_zone": per_zone_delta,
        "insights": insights,
    }
