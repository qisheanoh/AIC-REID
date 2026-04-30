"""
Diagnose ReID / reentry linker ambiguity for CAM1 GT persons 7, 10, and 11.

Uses the pre-recovery trace (cam1_missing_persons_90_150_trace.csv) as the
primary source because it was built from the accepted-baseline run's reentry
debug files before they were overwritten.

Output: runs/diagnostics/reid_ambiguity_root_cause_gt7_gt10_gt11.csv
"""

import csv
import os
from collections import defaultdict
import math

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRACE_PATH  = os.path.join(PROJECT, "runs/diagnostics/cam1_missing_persons_90_150_trace.csv")
CAM1_PATH   = os.path.join(PROJECT, "runs/kpi_batch/retail-shop_CAM1_tracks.csv")
OUT_DIR     = os.path.join(PROJECT, "runs/diagnostics")
OUT_PATH    = os.path.join(OUT_DIR, "reid_ambiguity_root_cause_gt7_gt10_gt11.csv")

# CAM1 ReentryConfig thresholds (from run_batch.py, is_full_cam1=False)
CFG = {
    "strong_reuse_score":         0.72,
    "strong_reuse_margin":        0.03,
    "cross_person_ambiguity_margin": 0.045,
    "min_deep_sim_for_reuse":     0.73,
    "min_topk_sim_for_reuse":     0.68,
    "min_part_topk_for_reuse":    0.58,
    "min_part_mean_for_reuse":    0.64,
    "strong_deep_relax_deep":     0.76,
    "strong_deep_relax_topk":     0.78,
    "same_candidate_safe_score":  -1.0,   # fallback to strong_reuse_score
    "same_candidate_min_deep":    -1.0,   # fallback to min_deep_sim_for_reuse
}

FPS = 12.0  # CAM1 approximate frame rate


def _safe_float(v, default=None):
    try:
        if v is None or str(v).strip() == "":
            return default
        return float(v)
    except (ValueError, TypeError):
        return default


def _safe_int(v, default=0):
    try:
        return int(float(v)) if v not in (None, "") else default
    except (ValueError, TypeError):
        return default


def load_cam1_csv(path):
    """Return dict: gid -> list of (frame_idx, ts_sec, cx, cy) sorted by frame."""
    by_gid = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            gid = _safe_int(r.get("global_id", 0))
            if gid <= 0:
                continue
            x1, y1, x2, y2 = (
                _safe_float(r["x1"], 0), _safe_float(r["y1"], 0),
                _safe_float(r["x2"], 0), _safe_float(r["y2"], 0),
            )
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            by_gid[gid].append((_safe_int(r["frame_idx"]), _safe_float(r["ts_sec"], 0.0), cx, cy))
    for gid in by_gid:
        by_gid[gid].sort()
    return by_gid


def last_pos_before(cam1_by_gid, gid, before_frame):
    """Last (cx, cy, ts_sec) for the given gid strictly before before_frame."""
    rows = cam1_by_gid.get(_safe_int(gid), [])
    result = None
    for frame_idx, ts, cx, cy in rows:
        if frame_idx < before_frame:
            result = (cx, cy, ts)
        else:
            break
    return result


def classify_rejection(row, tracklet_rows):
    """
    Classify WHY this tracklet is not assigned an existing identity.
    Returns (classification, failing_gates, recommendation).
    """
    raw = row.get("raw_decision_reason", "")
    primary_reason = raw.split("|")[0].strip()

    best_sim   = _safe_float(row.get("best_reid_sim"))
    best_deep  = _safe_float(row.get("best_reid_deep_sim"))
    margin     = _safe_float(row.get("margin"))
    second_sim = _safe_float(row.get("second_best_sim"))
    absorbed   = row.get("absorbed_into", "").strip()
    final_gid  = row.get("final_global_id", "0")

    if primary_reason == "accepted_reentry_prev_tracklet":
        return "ACCEPTED", [], "correctly assigned — investigate why track is still 0 elsewhere"

    failing = []
    classification = "unknown"
    recommendation = ""

    if primary_reason.startswith("rejected_cross_person_ambiguous"):
        classification = "CROSS_PERSON_AMBIGUOUS"
        if margin is not None:
            failing.append(
                f"margin={margin:.4f} < threshold={CFG['cross_person_ambiguity_margin']:.3f}"
                f"  (gap={CFG['cross_person_ambiguity_margin']-margin:.4f})"
            )
        if best_sim is not None and best_sim < CFG["strong_reuse_score"]:
            failing.append(
                f"best_score={best_sim:.4f} < strong_reuse_score={CFG['strong_reuse_score']:.2f}"
            )
        recommendation = (
            "Top-1 and top-2 candidates point at DIFFERENT existing persons with margin < 0.045. "
            "Cannot safely choose. "
            "Fix options: (A) improve OSNet feature separation between these two persons, "
            "(B) add spatial/temporal gating to break the ambiguity, "
            "(C) relax cross_person_ambiguity_margin for this specific scene."
        )
        if absorbed and "global_id=0" not in absorbed:
            recommendation += f" Currently absorbed into {absorbed} by group_merge — wrong merge."
        elif "global_id=0" in absorbed or not absorbed:
            recommendation += " Absorbed into zero (unassigned) — person simply invisible to the system."

    elif primary_reason.startswith("rejected_weak_score"):
        classification = "WEAK_SCORE"
        gates_checked = []
        if best_sim is not None:
            if best_sim < CFG["strong_reuse_score"]:
                failing.append(
                    f"best_score={best_sim:.4f} < strong_reuse_score={CFG['strong_reuse_score']:.2f}"
                    f"  (gap={CFG['strong_reuse_score']-best_sim:.4f})"
                )
            else:
                gates_checked.append(f"best_score={best_sim:.4f} ≥ {CFG['strong_reuse_score']:.2f} OK")
        if best_deep is not None:
            if best_deep < CFG["min_deep_sim_for_reuse"]:
                failing.append(
                    f"deep_sim={best_deep:.4f} < min_deep={CFG['min_deep_sim_for_reuse']:.2f}"
                    f"  (gap={CFG['min_deep_sim_for_reuse']-best_deep:.4f})"
                )
            else:
                gates_checked.append(f"deep_sim={best_deep:.4f} ≥ {CFG['min_deep_sim_for_reuse']:.2f} OK")
        recommendation = (
            "Best candidate fails one or more similarity thresholds. "
            "Failing: " + ("; ".join(failing) if failing else "unknown sub-gate — topk/part_topk likely") + ". "
            "Fix options: (A) lower strong_reuse_score or min_deep_sim thresholds for CAM1, "
            "(B) improve feature quality by increasing reid_min_quality_for_bank, "
            "(C) use temporal continuity to allow re-entry at slightly lower similarity."
        )

    elif primary_reason == "not_in_decisions":
        classification = "UNTRACKED"
        failing.append("No ByteTrack tracklet formed — YOLO detected but tracker never confirmed this person")
        recommendation = (
            "Person was detected by YOLO but ByteTrack did not form a confirmed track. "
            "Possible causes: (1) too few consecutive detections (< confirm_hits=4), "
            "(2) low YOLO confidence (< track_thresh=0.30), "
            "(3) heavy occlusion causing track breaks each frame. "
            "Fix: check YOLO confidence in this region; lower track_thresh or confirm_hits carefully."
        )

    elif primary_reason.startswith("accepted"):
        classification = "ACCEPTED"
        recommendation = "Correctly assigned — if final_global_id is still 0 this is a downstream issue"
    else:
        classification = f"OTHER:{primary_reason}"
        recommendation = "See raw_decision_reason for details"

    return classification, failing, recommendation


def gate_checks(best_sim, best_deep, margin, second_sim, best_match_id, second_match_id):
    """Return dict of per-gate pass/fail for primary_ok path."""
    checks = {}

    # Cross-person ambiguity
    cross_amb = (
        best_match_id is not None
        and second_match_id is not None
        and str(best_match_id) != str(second_match_id)
        and margin is not None
        and margin < CFG["cross_person_ambiguity_margin"]
    )
    checks["cross_person_ambiguous"] = "FIRE" if cross_amb else "clear"
    checks["cross_amb_margin_actual"] = f"{margin:.4f}" if margin is not None else "N/A"
    checks["cross_amb_margin_thresh"] = f"{CFG['cross_person_ambiguity_margin']:.3f}"

    # Primary gate components
    _fmt = lambda v: f"{v:.4f}" if v is not None else "N/A"
    checks["gate_best_score"] = "PASS" if (best_sim is not None and best_sim >= CFG["strong_reuse_score"]) else f"FAIL (need≥{CFG['strong_reuse_score']:.2f}, got {_fmt(best_sim)})"
    checks["gate_margin"]     = "PASS" if (margin is not None and margin >= CFG["strong_reuse_margin"]) else f"FAIL (need≥{CFG['strong_reuse_margin']:.2f}, got {_fmt(margin)})"
    checks["gate_deep"]       = "PASS" if (best_deep is not None and best_deep >= CFG["min_deep_sim_for_reuse"]) else f"FAIL (need≥{CFG['min_deep_sim_for_reuse']:.2f}, got {_fmt(best_deep)})"
    # topk and part_topk not in trace — mark as unknown
    checks["gate_topk"]       = f"unknown (need≥{CFG['min_topk_sim_for_reuse']:.2f}, not in trace)"
    checks["gate_part_topk"]  = f"unknown (need≥{CFG['min_part_topk_for_reuse']:.2f}, not in trace)"

    # Strong-deep-relax path (requires margin ≥ 0.04 + deep ≥ 0.76 + topk ≥ 0.78)
    strong_deep_ok = (
        best_deep is not None and best_deep >= CFG["strong_deep_relax_deep"]
        and margin is not None and margin >= 0.04
        and not cross_amb
    )
    checks["strong_deep_relax_path"] = "eligible" if strong_deep_ok else "blocked"

    return checks


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Load trace ────────────────────────────────────────────────────────────
    trace_rows = []
    with open(TRACE_PATH) as f:
        trace_rows = list(csv.DictReader(f))

    target_rows = [r for r in trace_rows if r.get("is_target_gt_7_10_11", "") == "YES"]
    print(f"GT7/10/11 rows in trace: {len(target_rows)}")

    # ── Load CAM1 CSV for spatial-jump computation ────────────────────────────
    cam1_by_gid = load_cam1_csv(CAM1_PATH)

    # ── Group by tracklet_id ─────────────────────────────────────────────────
    by_tracklet = defaultdict(list)
    for r in target_rows:
        by_tracklet[r["tracklet_id"]].append(r)

    report_rows = []

    for tid, trows in sorted(by_tracklet.items(), key=lambda kv: float(kv[1][0]["ts_sec"])):
        # Per-tracklet aggregates
        gts = sorted(set(r.get("resolved_gt_person", "") for r in trows if r.get("resolved_gt_person", "")))
        frames = sorted(int(float(r["frame_idx"])) for r in trows)
        ts_vals = sorted(float(r["ts_sec"]) for r in trows)
        src_gids = sorted(set(r.get("source_gid", "?") for r in trows))
        final_gids = sorted(set(r.get("final_global_id", "?") for r in trows))
        absorbed_set = sorted(set(r.get("absorbed_into", "") for r in trows if r.get("absorbed_into", "")))
        raw_reasons = sorted(set(r.get("raw_decision_reason", "?").split("|")[0] for r in trows))

        # Use first row with non-empty score as representative
        rep = trows[0]
        for r in trows:
            if r.get("best_reid_sim", "").strip():
                rep = r
                break

        best_sim   = _safe_float(rep.get("best_reid_sim"))
        best_deep  = _safe_float(rep.get("best_reid_deep_sim"))
        second_sim = _safe_float(rep.get("second_best_sim"))
        margin     = _safe_float(rep.get("margin"))
        best_mid   = rep.get("best_reid_match_id", "")
        second_mid = rep.get("second_best_match_id", "")
        best_match_gid = _safe_int(rep.get("best_reid_match_id", "0"))
        second_match_gid = _safe_int(rep.get("second_best_match_id", "0"))

        # Blocking reason from individual rows
        blocking_reasons = sorted(set(r.get("blocking_reason", "") for r in trows if r.get("blocking_reason", "")))

        # Spatial info (median cx/cy of tracklet)
        cxs = [_safe_float(r.get("cx")) for r in trows if r.get("cx", "").strip()]
        cys = [_safe_float(r.get("cy")) for r in trows if r.get("cy", "").strip()]
        med_cx = sorted(cxs)[len(cxs)//2] if cxs else None
        med_cy = sorted(cys)[len(cys)//2] if cys else None

        # ── Spatial jump to best-candidate's absorbed final GID ───────────────
        # "absorbed_into" tells us which final global_id this tracklet merged with.
        # We look up the last known position of that GID just before this tracklet.
        spatial_jump = "N/A"
        spatial_plausible = "N/A"
        time_gap_sec = "N/A"
        cand_last_ts = None
        for abs_str in absorbed_set:
            # Format: "global_id=X (unassigned)" or "global_id=X"
            if "global_id=0" in abs_str or "unassigned" in abs_str:
                continue
            try:
                abs_gid = int(abs_str.split("global_id=")[1].split()[0].split(")")[0])
            except (IndexError, ValueError):
                continue
            if abs_gid <= 0:
                continue
            last = last_pos_before(cam1_by_gid, abs_gid, frames[0])
            if last is not None:
                last_cx, last_cy, last_ts = last
                if med_cx is not None and med_cy is not None:
                    jump = math.sqrt((med_cx - last_cx)**2 + (med_cy - last_cy)**2)
                    spatial_jump = f"{jump:.1f}px"
                    # Plausibility: at 12fps, person moves ~10-30px/frame
                    # Gap in frames = frames[0] - last_frame
                    gap_frames = frames[0] - last[2] * FPS  # rough
                    spatial_plausible = "plausible" if jump < 400 else "IMPLAUSIBLE"
                cand_last_ts = last_ts
                break

        if cand_last_ts is not None:
            time_gap_sec = f"{ts_vals[0] - cand_last_ts:.2f}s"
        elif absorbed_set and all("unassigned" in a for a in absorbed_set):
            time_gap_sec = "absorbed=0 (no positive candidate)"

        # ── Classification ────────────────────────────────────────────────────
        primary_reason = raw_reasons[0] if raw_reasons else "unknown"
        classification, failing_gates, recommendation = classify_rejection(
            {"raw_decision_reason": primary_reason,
             "best_reid_sim": str(best_sim or ""),
             "best_reid_deep_sim": str(best_deep or ""),
             "margin": str(margin or ""),
             "absorbed_into": "; ".join(absorbed_set),
             "final_global_id": ",".join(str(g) for g in final_gids)},
            trows,
        )

        # ── Gate check ────────────────────────────────────────────────────────
        gates = gate_checks(best_sim, best_deep, margin, second_sim, best_mid, second_mid)

        # ── Should stay unassigned or become new ID? ──────────────────────────
        if classification == "CROSS_PERSON_AMBIGUOUS":
            if margin is not None and margin < 0.01:
                stay_unassigned = "YES — extreme ambiguity (margin<0.01), cannot distinguish from OSNet alone"
            elif margin is not None and margin < 0.045:
                stay_unassigned = (
                    f"CONDITIONAL — margin={margin:.3f} is below 0.045 threshold "
                    "but close enough that spatial/temporal constraints could disambiguate"
                )
            else:
                stay_unassigned = "NO — if spatial/temporal plausibility is confirmed, should become new ID"
        elif classification == "WEAK_SCORE":
            if best_sim is not None and best_sim >= 0.65 and best_deep is not None and best_deep >= 0.65:
                stay_unassigned = "NO — scores are close to threshold, likely same person; consider lowering min_deep"
            else:
                stay_unassigned = "YES — evidence too weak to assign; need more/better crops of this person"
        elif classification == "UNTRACKED":
            stay_unassigned = "N/A — never formed a tracklet; fix tracker gate first"
        elif classification == "ACCEPTED":
            stay_unassigned = "N/A — already accepted"
        else:
            stay_unassigned = "unknown"

        report_rows.append({
            "gt_persons":             "; ".join(gts),
            "tracklet_id":            tid,
            "source_gid":             "; ".join(src_gids),
            "final_global_id":        "; ".join(str(g) for g in final_gids),
            "start_frame":            frames[0],
            "end_frame":              frames[-1],
            "start_ts_sec":           round(ts_vals[0], 2),
            "end_ts_sec":             round(ts_vals[-1], 2),
            "row_count":              len(trows),
            "median_cx":              round(med_cx, 1) if med_cx is not None else "N/A",
            "median_cy":              round(med_cy, 1) if med_cy is not None else "N/A",
            "best_candidate_id":      best_mid,
            "best_score":             f"{best_sim:.4f}" if best_sim is not None else "N/A",
            "best_deep_sim":          f"{best_deep:.4f}" if best_deep is not None else "N/A",
            "second_candidate_id":    second_mid,
            "second_score":           f"{second_sim:.4f}" if second_sim is not None else "N/A",
            "margin":                 f"{margin:.4f}" if margin is not None else "N/A",
            "cross_person_ambiguous": gates["cross_person_ambiguous"],
            "gate_best_score":        gates["gate_best_score"],
            "gate_margin":            gates["gate_margin"],
            "gate_deep":              gates["gate_deep"],
            "gate_topk":              gates["gate_topk"],
            "gate_part_topk":         gates["gate_part_topk"],
            "strong_deep_relax_path": gates["strong_deep_relax_path"],
            "spatial_jump_to_candidate": spatial_jump,
            "candidate_spatial_plausible": spatial_plausible,
            "time_gap_to_candidate":  time_gap_sec,
            "absorbed_into":          "; ".join(absorbed_set) if absorbed_set else "none",
            "decision_reason":        primary_reason,
            "classification":         classification,
            "failing_gates":          " | ".join(failing_gates),
            "blocking_reason_detail": "; ".join(blocking_reasons),
            "stay_unassigned":        stay_unassigned,
            "recommendation":         recommendation,
        })

    # ── Write output ──────────────────────────────────────────────────────────
    fieldnames = [
        "gt_persons", "tracklet_id", "source_gid", "final_global_id",
        "start_frame", "end_frame", "start_ts_sec", "end_ts_sec", "row_count",
        "median_cx", "median_cy",
        "best_candidate_id", "best_score", "best_deep_sim",
        "second_candidate_id", "second_score", "margin",
        "cross_person_ambiguous",
        "gate_best_score", "gate_margin", "gate_deep", "gate_topk", "gate_part_topk",
        "strong_deep_relax_path",
        "spatial_jump_to_candidate", "candidate_spatial_plausible", "time_gap_to_candidate",
        "absorbed_into", "decision_reason", "classification",
        "failing_gates", "blocking_reason_detail",
        "stay_unassigned", "recommendation",
    ]

    with open(OUT_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(report_rows)

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  GT7/10/11 ReID ambiguity root-cause report  ({len(report_rows)} tracklets)")
    print(f"{'='*80}")

    for row in report_rows:
        print(f"\n  GT={row['gt_persons']:<6}  tid={row['tracklet_id']:<4}  "
              f"frames=[{row['start_frame']:4d}-{row['end_frame']:4d}]  "
              f"ts=[{row['start_ts_sec']:6.2f}-{row['end_ts_sec']:6.2f}s]  "
              f"rows={row['row_count']:3d}")
        print(f"    decision       : {row['decision_reason']}")
        print(f"    classification : {row['classification']}")
        print(f"    best_score     : {row['best_score']}  (need≥{CFG['strong_reuse_score']:.2f})  gate={row['gate_best_score']}")
        print(f"    best_deep      : {row['best_deep_sim']}  (need≥{CFG['min_deep_sim_for_reuse']:.2f})  gate={row['gate_deep']}")
        print(f"    margin         : {row['margin']}  (need≥{CFG['strong_reuse_margin']:.2f} primary, <{CFG['cross_person_ambiguity_margin']:.3f} ambiguous)")
        print(f"    cross_amb      : {row['cross_person_ambiguous']}")
        print(f"    best→second    : {row['best_candidate_id']} → {row['second_candidate_id']}")
        print(f"    spatial_jump   : {row['spatial_jump_to_candidate']}  plausible={row['candidate_spatial_plausible']}  time_gap={row['time_gap_to_candidate']}")
        print(f"    absorbed_into  : {row['absorbed_into']}")
        print(f"    stay_unassigned: {row['stay_unassigned']}")
        if row["failing_gates"]:
            print(f"    failing_gates  : {row['failing_gates']}")
        print(f"    recommendation : {row['recommendation'][:100]}...")

    print(f"\n  Report saved → {OUT_PATH}")
    print(f"{'='*80}\n")

    # ── Thresholds used ───────────────────────────────────────────────────────
    print("  CAM1 ReentryConfig thresholds applied:")
    for k, v in CFG.items():
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
