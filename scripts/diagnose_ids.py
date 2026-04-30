"""
Diagnostic script: analyse global_id distribution in the latest tracks CSVs.
Prints evidence for items 1-8 without modifying any tracking logic.

Usage:
    python scripts/diagnose_ids.py
"""

import csv
import os
from collections import defaultdict

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FILES = {
    "CAM1": os.path.join(PROJECT, "runs/kpi_batch/retail-shop_CAM1_tracks.csv"),
    "FULL_CAM1": os.path.join(PROJECT, "runs/kpi_batch/retail-shop_FULL_CAM1_tracks.csv"),
    "FULL_CAM1_pre_surgical": os.path.join(
        PROJECT, "runs/kpi_batch/retail-shop_FULL_CAM1_tracks.csv.pre-surgical-fix.csv"
    ),
    "Demo_Video": os.path.join(PROJECT, "runs/kpi_batch/retail-shop_Demo_Video_tracks.csv"),
    "Audit_Sheet": os.path.join(PROJECT, "experiments/audit/cam1_manual_audit_sheet.csv"),
}

GT_PERSON_COUNT = 11  # ground truth for CAM1 / FULL_CAM1
DEMO_VIDEO_DURATION_SEC = 30.0  # known: only first 30 s


# ── helpers ──────────────────────────────────────────────────────────────────

def load_tracks(path):
    """Return list of dicts from a tracks CSV."""
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def summarise_tracks(rows, label):
    """
    Compute per-global_id stats and print diagnostic table.
    Returns (id_counts, total_rows, zero_rows).
    """
    id_rows    = defaultdict(list)
    for r in rows:
        gid = int(float(r["global_id"]))
        ts  = float(r["ts_sec"])
        id_rows[gid].append(ts)

    total_rows = len(rows)
    zero_rows  = len(id_rows.get(0, []))

    print(f"\n{'='*70}")
    print(f"  {label}  |  total rows: {total_rows:,}")
    print(f"{'='*70}")
    print(f"  {'global_id':>10}  {'rows':>6}  {'pct%':>6}  {'first_ts':>9}  {'last_ts':>9}  {'duration':>9}")
    print(f"  {'-'*10}  {'-'*6}  {'-'*6}  {'-'*9}  {'-'*9}  {'-'*9}")

    sorted_ids = sorted(id_rows.keys())
    for gid in sorted_ids:
        ts_list  = id_rows[gid]
        count    = len(ts_list)
        pct      = 100.0 * count / total_rows
        first_ts = min(ts_list)
        last_ts  = max(ts_list)
        duration = last_ts - first_ts
        marker   = "  <-- ZERO (unassigned)" if gid == 0 else ""
        print(
            f"  {gid:>10}  {count:>6}  {pct:>5.1f}%  "
            f"{first_ts:>9.2f}  {last_ts:>9.2f}  {duration:>9.2f}{marker}"
        )

    return id_rows, total_rows, zero_rows


def check_id0_dominance(zero_rows, total_rows, label):
    pct = 100.0 * zero_rows / total_rows if total_rows else 0
    flag = "YES — global_id=0 dominates" if pct > 50 else "No"
    print(f"\n[5] global_id=0 dominance [{label}]: {zero_rows:,}/{total_rows:,} = {pct:.1f}%  →  {flag}")


# ── audit sheet ──────────────────────────────────────────────────────────────

def analyse_audit_sheet(path):
    print(f"\n{'='*70}")
    print("  AUDIT SHEET  (cam1_manual_audit_sheet.csv)")
    print(f"{'='*70}")

    gt_ids      = set()
    pred_ids    = set()
    row_count   = 0
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            row_count += 1
            if row.get("gt_person_id") not in ("", None):
                try:
                    gt_ids.add(int(float(row["gt_person_id"])))
                except ValueError:
                    pass
            if row.get("pred_global_id") not in ("", None):
                try:
                    pred_ids.add(int(float(row["pred_global_id"])))
                except ValueError:
                    pass

    print(f"  Audit rows          : {row_count}")
    print(f"  Unique gt_person_ids: {sorted(gt_ids)}")
    print(f"  Unique pred_ids     : {sorted(pred_ids)}")
    print(f"  GT persons covered  : {len(gt_ids)} / {GT_PERSON_COUNT}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*70)
    print("  DIAGNOSTIC  —  global_id distribution across latest CSVs")
    print("="*70)

    # ── CAM1 ─────────────────────────────────────────────────────────────────
    rows_cam1 = load_tracks(FILES["CAM1"])
    id_rows_cam1, total_cam1, zero_cam1 = summarise_tracks(rows_cam1, "CAM1")

    positive_ids_cam1 = [gid for gid in id_rows_cam1 if gid != 0]
    check_id0_dominance(zero_cam1, total_cam1, "CAM1")

    print(f"\n[6] CAM1 positive IDs: {sorted(positive_ids_cam1)}")
    print(f"    Count: {len(positive_ids_cam1)}  (expected: {GT_PERSON_COUNT})")
    if len(positive_ids_cam1) < GT_PERSON_COUNT:
        missing = GT_PERSON_COUNT - len(positive_ids_cam1)
        print(f"    WARNING: {missing} person(s) MISSING vs ground truth ({GT_PERSON_COUNT})")
    elif len(positive_ids_cam1) > GT_PERSON_COUNT:
        extra = len(positive_ids_cam1) - GT_PERSON_COUNT
        print(f"    WARNING: {extra} extra ID(s) vs ground truth ({GT_PERSON_COUNT})")
    else:
        print(f"    OK: ID count matches ground truth.")

    # ── FULL_CAM1 (current) ──────────────────────────────────────────────────
    rows_full = load_tracks(FILES["FULL_CAM1"])
    id_rows_full, total_full, zero_full = summarise_tracks(rows_full, "FULL_CAM1 (current)")

    positive_ids_full = [gid for gid in id_rows_full if gid != 0]
    check_id0_dominance(zero_full, total_full, "FULL_CAM1 (current)")

    # ── FULL_CAM1 (pre-surgical) — for comparison ─────────────────────────────
    pre_path = FILES["FULL_CAM1_pre_surgical"]
    if os.path.exists(pre_path):
        rows_pre = load_tracks(pre_path)
        id_rows_pre, total_pre, zero_pre = summarise_tracks(rows_pre, "FULL_CAM1 (pre-surgical backup)")
        check_id0_dominance(zero_pre, total_pre, "FULL_CAM1 pre-surgical")

        print(f"\n[7] FULL_CAM1  global_id=0 comparison:")
        print(f"    Pre-surgical : {zero_pre:,} zero-rows  / {total_pre:,} total")
        print(f"    Current      : {zero_full:,} zero-rows  / {total_full:,} total")
        delta = zero_full - zero_pre
        if delta > 0:
            print(f"    RESULT: current has {delta:+,} MORE zero-rows than pre-surgical  ← REGRESSION")
        elif delta < 0:
            print(f"    RESULT: current has {abs(delta):,} FEWER zero-rows than pre-surgical  ← expected (fixes applied)")
        else:
            print(f"    RESULT: identical zero-row count  (no change)")
    else:
        print(f"\n[7] Pre-surgical CSV not found: {pre_path}")

    # ── Demo_Video ────────────────────────────────────────────────────────────
    rows_demo = load_tracks(FILES["Demo_Video"])
    id_rows_demo, total_demo, zero_demo = summarise_tracks(rows_demo, "Demo_Video")

    positive_ids_demo = [gid for gid in id_rows_demo if gid != 0]
    check_id0_dominance(zero_demo, total_demo, "Demo_Video")

    ts_all_demo = [float(r["ts_sec"]) for r in rows_demo]
    demo_actual_duration = max(ts_all_demo) - min(ts_all_demo) if ts_all_demo else 0.0

    print(f"\n[8] Demo_Video 11-person evaluation eligibility:")
    print(f"    Actual duration in CSV : {demo_actual_duration:.2f}s")
    print(f"    Known clip duration    : {DEMO_VIDEO_DURATION_SEC:.0f}s")
    print(f"    Positive IDs present   : {sorted(positive_ids_demo)}  ({len(positive_ids_demo)} persons)")
    print(f"    Ground truth persons   : {GT_PERSON_COUNT}")
    if len(positive_ids_demo) < GT_PERSON_COUNT:
        print(f"    VERDICT: EXCLUDE from 11-person evaluation.")
        print(f"             Only {len(positive_ids_demo)} persons visible in {DEMO_VIDEO_DURATION_SEC:.0f}s — "
              f"not all {GT_PERSON_COUNT} GT persons have entered the scene yet.")
    else:
        print(f"    VERDICT: All {GT_PERSON_COUNT} persons appear — could be included (verify manually).")

    # ── Audit sheet ───────────────────────────────────────────────────────────
    analyse_audit_sheet(FILES["Audit_Sheet"])

    print(f"\n{'='*70}")
    print("  DIAGNOSTIC COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
