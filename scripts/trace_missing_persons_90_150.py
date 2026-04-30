"""
Trace CAM1 90-150s for GT persons 7, 10, and 11.
For every detection row in that window, join all available debug files
and determine why each detection stayed at global_id=0 or was absorbed
into the wrong existing ID.

Output: runs/diagnostics/cam1_missing_persons_90_150_trace.csv
"""

import csv
import os
import math
from collections import defaultdict

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── File paths ────────────────────────────────────────────────────────────────
TRACKS       = os.path.join(PROJECT, "runs/kpi_batch/retail-shop_CAM1_tracks.csv")
TROWS        = os.path.join(PROJECT, "runs/kpi_batch/reentry_debug/retail-shop_CAM1/tracklet_rows.csv")
DECISIONS    = os.path.join(PROJECT, "runs/kpi_batch/reentry_debug/retail-shop_CAM1/reentry_decisions.csv")
EXITED       = os.path.join(PROJECT, "runs/kpi_batch/reentry_debug/retail-shop_CAM1/exited_tracklets.csv")
ENTERED      = os.path.join(PROJECT, "runs/kpi_batch/reentry_debug/retail-shop_CAM1/entered_tracklets.csv")
CANDIDATES   = os.path.join(PROJECT, "runs/kpi_batch/reentry_debug/retail-shop_CAM1/reentry_candidates.csv")
AUDIT        = os.path.join(PROJECT, "experiments/audit/cam1_manual_audit_sheet.csv")
ARCHIVE_CAM1 = os.path.join(PROJECT, "archive/old_runs/kpi_batch_identity_safe/retail-shop_CAM1_tracks.csv")

OUT_DIR  = os.path.join(PROJECT, "runs/diagnostics")
OUT_PATH = os.path.join(OUT_DIR, "cam1_missing_persons_90_150_trace.csv")

TS_MIN, TS_MAX = 90.0, 150.0
TARGET_GT_IDS  = {7, 10, 11}        # GT persons of interest
ALL_GT_IDS     = set(range(1, 12))  # full GT set


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def bbox(row, x1k="x1", y1k="y1", x2k="x2", y2k="y2"):
    return (float(row[x1k]), float(row[y1k]), float(row[x2k]), float(row[y2k]))


def iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + ab - inter)


def cx_cy(row):
    b = bbox(row)
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def area_ratio(row, frame_w=2560, frame_h=1440):
    b = bbox(row)
    return (b[2] - b[0]) * (b[3] - b[1]) / (frame_w * frame_h)


def int_frame(r, key="frame_idx"):
    return int(float(r[key]))


# ── Canonical ID re-map (source_gid → final CSV global_id) ───────────────────
# Derived by cross-referencing tracklet_rows + tracks CSV
def build_source_to_final_map(trows, tracks):
    m = defaultdict(lambda: defaultdict(int))
    for t in trows:
        final = int(float(tracks[int(t["row_idx"])]["global_id"]))
        m[int(t["source_gid"])][final] += 1
    # majority vote per source_gid
    result = {}
    for src, cnt in m.items():
        result[src] = max(cnt, key=cnt.get)
    return result


# ── Best / 2nd-best candidate per tracklet ───────────────────────────────────
def build_candidate_index(cands):
    """Returns {tracklet_id: [sorted candidate dicts by score desc]}"""
    idx = defaultdict(list)
    for c in cands:
        score = c.get("score", "")
        if score == "":
            continue
        idx[c["new_tracklet_id"]].append(c)
    for tid in idx:
        idx[tid].sort(key=lambda x: float(x["score"]), reverse=True)
    return idx


# ── Match a bbox to the nearest audit row in the same frame ──────────────────
def match_audit(frame_idx, bbox_cur, audit_by_frame, iou_thresh=0.3):
    """
    Returns (gt_person_id, pred_global_id_in_audit, best_iou) or (None, None, 0).
    """
    candidates = audit_by_frame.get(frame_idx, [])
    best_iou_val = 0.0
    best_row = None
    for ar in candidates:
        try:
            ab = bbox(ar)
        except (ValueError, KeyError):
            continue
        v = iou(bbox_cur, ab)
        if v > best_iou_val:
            best_iou_val = v
            best_row = ar
    if best_row is None or best_iou_val < iou_thresh:
        return None, None, best_iou_val
    gt = best_row.get("gt_person_id", "").strip()
    pred_audit = best_row.get("pred_global_id", "").strip()
    gt = int(float(gt)) if gt else None
    pred_audit = int(float(pred_audit)) if pred_audit else None
    return gt, pred_audit, best_iou_val


# ── Match archive row (which has GT7/10/11 labels via pipeline ID) ─────────────
def build_archive_index(arc_rows, target_gids):
    """Index archive rows by frame for GT IDs of interest."""
    idx = defaultdict(list)
    for r in arc_rows:
        gid = int(float(r["global_id"]))
        if gid in target_gids:
            idx[int_frame(r)].append((gid, r))
    return idx


def blocking_reason(decision_reason, final_gid, source_gid):
    """
    Classify into a human-readable blocking reason.
    decision_reason is the pipe-separated string from reentry_decisions.
    final_gid is the canonical CSV global_id of this row.
    """
    if not decision_reason or decision_reason == "not_in_decisions":
        if source_gid == 0:
            return "source_gid=0: ByteTrack/detector never assigned a track ID"
        return "not_in_reentry_decisions (short tracklet or untracked)"

    parts = decision_reason.split("|")
    primary = parts[0] if parts else decision_reason

    if "rejected_weak_score_new_id" in primary:
        return "rejected_weak_score: ReID similarity too low to confirm new person vs existing"
    if "rejected_cross_person_ambiguous_new_id" in primary:
        return "rejected_ambiguous: best score matches multiple existing persons — cannot confirm new ID"
    if "new_id_no_candidate" in primary:
        return "new_id_no_candidate: first appearance, no prior to compare against"
    if "accepted_reentry" in primary:
        return f"accepted_reentry: relinked to previous tracklet"
    if "rejected" in primary:
        return f"rejected: {primary}"
    return f"other: {primary}"


def absorbed_into(final_gid):
    """Report which canonical ID this row was absorbed into."""
    if final_gid == 0:
        return "global_id=0 (unassigned)"
    return f"global_id={final_gid}"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    tracks    = load_csv(TRACKS)
    trows     = load_csv(TROWS)
    decisions = {r["tracklet_id"]: r for r in load_csv(DECISIONS)}
    exited    = {r["tracklet_id"]: r for r in load_csv(EXITED)}
    entered   = {r["tracklet_id"]: r for r in load_csv(ENTERED)}
    cands_raw = load_csv(CANDIDATES)
    audit_all = load_csv(AUDIT)
    arc_rows  = load_csv(ARCHIVE_CAM1)

    cand_idx = build_candidate_index(cands_raw)

    # Index tracklet_rows by row_idx
    trow_by_idx = {int(r["row_idx"]): r for r in trows}

    # Index audit by frame (using integer frame_idx key)
    audit_by_frame = defaultdict(list)
    for ar in audit_all:
        try:
            audit_by_frame[int_frame(ar)].append(ar)
        except (ValueError, KeyError):
            pass

    # Index archive by frame for target GT IDs
    arc_index = build_archive_index(arc_rows, TARGET_GT_IDS)

    # source_gid → canonical global_id majority map
    src_to_final = build_source_to_final_map(trows, tracks)

    # ── Filter tracks to 90-150s ─────────────────────────────────────────────
    rows_in_range = [
        (i, r) for i, r in enumerate(tracks)
        if TS_MIN <= float(r["ts_sec"]) <= TS_MAX
    ]
    print(f"Rows in 90-150s: {len(rows_in_range)}")

    # ── Build per-tracklet best/2nd-best candidate info ──────────────────────
    def get_top_candidates(tid_str):
        cs = cand_idx.get(tid_str, [])
        # de-duplicate by prev_final_gid — keep max score per prev
        seen = {}
        for c in cs:
            pfg = c["prev_final_gid"]
            sc  = float(c["score"])
            ds  = float(c["deep_sim"]) if c.get("deep_sim") else 0.0
            if pfg not in seen or sc > float(seen[pfg]["score"]):
                seen[pfg] = c
        ranked = sorted(seen.values(), key=lambda x: float(x["score"]), reverse=True)
        best   = ranked[0] if len(ranked) >= 1 else None
        second = ranked[1] if len(ranked) >= 2 else None
        margin = (float(best["score"]) - float(second["score"])) if (best and second) else None
        return best, second, margin

    # ── Produce output rows ───────────────────────────────────────────────────
    output_rows = []
    for i, tr in rows_in_range:
        frame_idx = int_frame(tr)
        ts_sec    = float(tr["ts_sec"])
        final_gid = int(float(tr["global_id"]))
        bb        = bbox(tr)
        cx, cy    = cx_cy(tr)
        ar_ratio  = area_ratio(tr)

        # From tracklet_rows
        trow = trow_by_idx.get(i, {})
        tid_str   = trow.get("tracklet_id", "0")
        src_gid   = int(trow.get("source_gid", "0")) if trow else 0
        asgn_fgid = trow.get("assigned_final_gid", "0") if trow else "0"

        # From decisions
        dec = decisions.get(tid_str, {})
        dec_reason = dec.get("decision_reason", "not_in_decisions")
        quality    = dec.get("quality", "") or exited.get(tid_str, {}).get("quality", "")
        local_tid  = dec.get("local_track_id", "") or exited.get(tid_str, {}).get("local_track_id", "")

        # Candidate scores
        best_c, sec_c, margin = get_top_candidates(tid_str)
        best_match_id  = best_c["prev_final_gid"]   if best_c  else ""
        best_sim       = best_c["score"]             if best_c  else ""
        best_deep      = best_c["deep_sim"]          if best_c  else ""
        sec_match_id   = sec_c["prev_final_gid"]     if sec_c   else ""
        sec_sim        = sec_c["score"]              if sec_c   else ""
        margin_str     = f"{margin:.4f}"             if margin is not None else ""

        # New ID allowed?
        new_id_allowed = "no" if any(
            kw in dec_reason for kw in ("rejected_weak_score", "rejected_cross_person")
        ) else ("yes" if "new_id_no_candidate" in dec_reason else "n/a")
        if src_gid == 0:
            new_id_allowed = "n/a (untracked)"

        # Blocking reason
        block = blocking_reason(dec_reason, final_gid, src_gid)

        # Absorbed into
        absorbed = absorbed_into(final_gid)

        # GT person match from audit
        gt_id, pred_audit, match_iou = match_audit(frame_idx, bb, audit_by_frame)

        # GT person match from archive (for GT7/10/11 specifically)
        arc_match_gid = None
        arc_best_iou  = 0.0
        for ag, ar in arc_index.get(frame_idx, []):
            v = iou(bb, bbox(ar))
            if v > arc_best_iou:
                arc_best_iou = v
                arc_match_gid = ag
        if arc_best_iou < 0.3:
            arc_match_gid = None

        # Resolve GT person: prefer audit match, fall back to archive match
        resolved_gt = gt_id if gt_id is not None else arc_match_gid
        is_target   = resolved_gt in TARGET_GT_IDS if resolved_gt else False

        output_rows.append({
            "frame_idx":               frame_idx,
            "ts_sec":                  f"{ts_sec:.4f}",
            "local_track_id":          local_tid,
            "tracklet_id":             tid_str,
            "source_gid":              src_gid,
            "assigned_final_gid_internal": asgn_fgid,
            "final_global_id":         final_gid,
            "x1":                      f"{bb[0]:.2f}",
            "y1":                      f"{bb[1]:.2f}",
            "x2":                      f"{bb[2]:.2f}",
            "y2":                      f"{bb[3]:.2f}",
            "cx":                      f"{cx:.1f}",
            "cy":                      f"{cy:.1f}",
            "bbox_area_ratio":         f"{ar_ratio:.5f}",
            "tracklet_quality":        quality,
            "stayed_global_id_0":      "yes" if final_gid == 0 else "no",
            "best_reid_match_id":      best_match_id,
            "best_reid_sim":           f"{float(best_sim):.4f}" if best_sim != "" else "",
            "best_reid_deep_sim":      f"{float(best_deep):.4f}" if best_deep != "" else "",
            "second_best_match_id":    sec_match_id,
            "second_best_sim":         f"{float(sec_sim):.4f}" if sec_sim != "" else "",
            "margin":                  margin_str,
            "new_global_id_allowed":   new_id_allowed,
            "blocking_reason":         block,
            "absorbed_into":           absorbed,
            "gt_person_id_from_audit": gt_id if gt_id else "",
            "pred_in_audit":           pred_audit if pred_audit else "",
            "audit_iou":               f"{match_iou:.3f}",
            "gt_from_archive_match":   arc_match_gid if arc_match_gid else "",
            "archive_iou":             f"{arc_best_iou:.3f}",
            "resolved_gt_person":      resolved_gt if resolved_gt else "",
            "is_target_gt_7_10_11":    "YES" if is_target else "no",
            "raw_decision_reason":     dec_reason,
        })

    # ── Write CSV ─────────────────────────────────────────────────────────────
    fieldnames = list(output_rows[0].keys()) if output_rows else []
    with open(OUT_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(output_rows)

    print(f"Written {len(output_rows)} rows → {OUT_PATH}")

    # ── Summary ───────────────────────────────────────────────────────────────
    from collections import Counter

    print(f"\n{'='*72}")
    print("  SUMMARY: CAM1 90-150s — missing persons trace")
    print(f"{'='*72}")

    total = len(output_rows)
    zero_rows   = [r for r in output_rows if r["stayed_global_id_0"] == "yes"]
    target_rows = [r for r in output_rows if r["is_target_gt_7_10_11"] == "YES"]

    print(f"  Total rows in 90-150s   : {total}")
    print(f"  Rows with global_id=0   : {len(zero_rows)}")
    print(f"  Rows matching GT7/10/11 : {len(target_rows)}")

    # Breakdown of target rows by GT person
    tgt_by_gt = Counter(r["resolved_gt_person"] for r in target_rows)
    print(f"\n  Target GT person row counts:")
    for gt, cnt in sorted(tgt_by_gt.items(), key=lambda x: int(x[0]) if x[0] else 0):
        absorbed_ids = Counter(r["absorbed_into"] for r in target_rows if r["resolved_gt_person"] == gt)
        print(f"    GT{gt}: {cnt} rows → absorbed into: {dict(absorbed_ids)}")

    # Blocking reasons for target rows
    print(f"\n  Blocking reasons for GT7/10/11 rows:")
    block_counts = Counter(r["blocking_reason"] for r in target_rows)
    for reason, cnt in block_counts.most_common():
        print(f"    {cnt:4d}  {reason}")

    # Blocking reasons for ALL zero-id rows
    print(f"\n  Blocking reasons for ALL global_id=0 rows in 90-150s:")
    block_zero = Counter(r["blocking_reason"] for r in zero_rows)
    for reason, cnt in block_zero.most_common():
        print(f"    {cnt:4d}  {reason}")

    # Show which GT persons were correctly vs incorrectly assigned
    print(f"\n  GT person final_global_id mapping (rows with audit match, iou>0.3):")
    matched_audit = [r for r in output_rows if float(r["audit_iou"]) >= 0.3 and r["gt_person_id_from_audit"]]
    gt_to_final = defaultdict(Counter)
    for r in matched_audit:
        gt_to_final[r["gt_person_id_from_audit"]][r["final_global_id"]] += 1
    for gt in sorted(gt_to_final.keys(), key=lambda x: int(float(x))):
        mapping = dict(gt_to_final[gt].most_common())
        flag = "  *** MISSING ***" if int(float(gt)) in TARGET_GT_IDS else ""
        print(f"    GT{gt}: {mapping}{flag}")

    # source_gid breakdown for GT7/10/11 rows
    print(f"\n  source_gid for GT7/10/11 rows:")
    src_counts = Counter(r["source_gid"] for r in target_rows)
    for src, cnt in sorted(src_counts.items(), key=lambda x: int(x[0])):
        print(f"    source_gid={src}: {cnt} rows")

    print(f"\n  Report saved → {OUT_PATH}\n")


if __name__ == "__main__":
    main()
