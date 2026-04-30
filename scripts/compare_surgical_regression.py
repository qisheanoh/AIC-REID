"""
Compare pre-surgical and current FULL_CAM1 CSVs row-by-row.
Identifies every row where global_id changed from positive → 0 and
classifies the cause.

Output: runs/diagnostics/fullcam1_zero_regression_278_rows.csv
"""

import csv
import math
import os

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PRE_PATH = os.path.join(
    PROJECT, "runs/kpi_batch/retail-shop_FULL_CAM1_tracks.csv.pre-surgical-fix.csv"
)
CUR_PATH = os.path.join(
    PROJECT, "runs/kpi_batch/retail-shop_FULL_CAM1_tracks.csv"
)
OUT_DIR  = os.path.join(PROJECT, "runs/diagnostics")
OUT_PATH = os.path.join(OUT_DIR, "fullcam1_zero_regression_278_rows.csv")

# ── Known surgical fix windows (frame_idx ranges, inclusive) ─────────────────
# Derived from the previous session's git audit trail / backup chain.
SURGICAL_RULES = [
    {
        "label": "surgical-fix: ID2 wrong-reentry-link",
        "pre_gid": 2,
        "frame_min": 1237,
        "frame_max": 1460,
        "detail": (
            "Re-entry linker merged ID2 with a new unrelated person after the "
            "original ID2 person departed. Frames 1237-1460 zeroed."
        ),
        "mechanism": "re-entry linker / track-stitching wrong match post-departure",
    },
    {
        "label": "surgical-fix: ID3 wrong-merge early frames",
        "pre_gid": 3,
        "frame_min": 84,
        "frame_max": 122,
        "detail": (
            "Tracker merged ID3 with a different person in early overlap frames "
            "84-122. Zeroed to restore per-person purity."
        ),
        "mechanism": "ByteTrack ID switch / wrong-merge at scene entry",
    },
    {
        "label": "surgical-fix: ID7 hijab-woman wrong-merge",
        "pre_gid": 7,
        "frame_min": 746,
        "frame_max": 760,
        "detail": (
            "ID7 was incorrectly merged with a hijab-wearing woman at cx~1317 "
            "(center-left position). Legitimate ID7 segments are at cx~1975-2078 "
            "and cx~1803-1932 (right-side area). Frames 746-760 zeroed."
        ),
        "mechanism": "re-entry linker wrong-merge across spatially distinct persons",
    },
]


def iou(a, b):
    """IoU between two boxes [x1,y1,x2,y2]."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def bbox(row):
    return (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]))


def cx_cy(row):
    b = bbox(row)
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def classify(pre_gid, frame_idx):
    """Return (rule_label, mechanism, detail) for a changed row, or unclassified."""
    for rule in SURGICAL_RULES:
        if rule["pre_gid"] == pre_gid and rule["frame_min"] <= frame_idx <= rule["frame_max"]:
            return rule["label"], rule["mechanism"], rule["detail"]
    return (
        f"UNCLASSIFIED (pre_gid={pre_gid} frame={frame_idx})",
        "unknown",
        "Row changed to zero but does not match any known surgical fix window.",
    )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Load CSVs ─────────────────────────────────────────────────────────────
    with open(PRE_PATH) as f:
        pre_rows = list(csv.DictReader(f))
    with open(CUR_PATH) as f:
        cur_rows = list(csv.DictReader(f))

    print(f"Pre-surgical rows : {len(pre_rows):,}")
    print(f"Current rows      : {len(cur_rows):,}")

    # Index current by exact key (frame_idx + bbox coords are unchanged by surgical edit)
    cur_index = {}
    for r in cur_rows:
        key = (r["frame_idx"], r["x1"], r["y1"], r["x2"], r["y2"])
        cur_index[key] = r

    # ── Find all changed rows ─────────────────────────────────────────────────
    changed   = []
    unmatched = []   # rows in pre with no exact bbox match in cur

    for r in pre_rows:
        pre_gid = int(float(r["global_id"]))
        if pre_gid == 0:
            continue   # only care about positive→zero transitions

        key = (r["frame_idx"], r["x1"], r["y1"], r["x2"], r["y2"])
        if key not in cur_index:
            unmatched.append(r)
            continue

        cur_r   = cur_index[key]
        cur_gid = int(float(cur_r["global_id"]))

        if cur_gid == 0:
            frame_idx = int(float(r["frame_idx"]))
            b         = bbox(r)
            cx, cy    = cx_cy(r)

            rule_label, mechanism, detail = classify(pre_gid, frame_idx)

            changed.append({
                "frame_idx":           frame_idx,
                "ts_sec":              float(r["ts_sec"]),
                "pre_global_id":       pre_gid,
                "cur_global_id":       0,
                "x1":                  b[0],
                "y1":                  b[1],
                "x2":                  b[2],
                "y2":                  b[3],
                "cx":                  round(cx, 1),
                "cy":                  round(cy, 1),
                "detection_in_current": "yes",   # exact match found
                "rule_label":          rule_label,
                "mechanism":           mechanism,
                "detail":              detail,
            })

    # ── Handle unmatched (bbox changed — shouldn't happen for surgical edits) ──
    for r in unmatched:
        pre_gid   = int(float(r["global_id"]))
        frame_idx = int(float(r["frame_idx"]))
        b         = bbox(r)
        cx, cy    = cx_cy(r)
        changed.append({
            "frame_idx":            frame_idx,
            "ts_sec":               float(r["ts_sec"]),
            "pre_global_id":        pre_gid,
            "cur_global_id":        "NO_MATCH",
            "x1":                   b[0],
            "y1":                   b[1],
            "x2":                   b[2],
            "y2":                   b[3],
            "cx":                   round(cx, 1),
            "cy":                   round(cy, 1),
            "detection_in_current": "no — bbox changed or row removed",
            "rule_label":           "NO_EXACT_MATCH",
            "mechanism":            "detection no longer present / bbox modified",
            "detail":               "Row existed in pre-surgical CSV but no matching bbox in current.",
        })

    # ── Sort by frame then by pre_global_id ──────────────────────────────────
    changed.sort(key=lambda r: (r["frame_idx"], r["pre_global_id"]))

    # ── Write CSV ─────────────────────────────────────────────────────────────
    fieldnames = [
        "frame_idx", "ts_sec",
        "pre_global_id", "cur_global_id",
        "x1", "y1", "x2", "y2", "cx", "cy",
        "detection_in_current",
        "rule_label", "mechanism", "detail",
    ]
    with open(OUT_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(changed)

    print(f"\nTotal changed rows written : {len(changed)}")
    print(f"Unmatched (bbox diff)      : {len(unmatched)}")

    # ── Summary by rule ───────────────────────────────────────────────────────
    from collections import Counter
    rule_counts = Counter(r["rule_label"] for r in changed)
    print(f"\n{'─'*72}")
    print("  Summary by rule:")
    print(f"{'─'*72}")
    for label, count in sorted(rule_counts.items(), key=lambda x: -x[1]):
        print(f"  {count:4d} rows  →  {label}")

    # ── Unclassified? ─────────────────────────────────────────────────────────
    unclassified = [r for r in changed if r["rule_label"].startswith("UNCLASSIFIED")]
    if unclassified:
        print(f"\n  *** {len(unclassified)} UNCLASSIFIED rows — need investigation ***")
        for r in unclassified[:10]:
            print(f"    frame={r['frame_idx']}  pre_gid={r['pre_global_id']}  "
                  f"cx={r['cx']}  cy={r['cy']}")
    else:
        print(f"\n  All {len(changed)} changed rows are fully classified.")

    # ── Per-group frame range verification ───────────────────────────────────
    print(f"\n{'─'*72}")
    print("  Per-group frame range:")
    print(f"{'─'*72}")
    from collections import defaultdict
    by_label = defaultdict(list)
    for r in changed:
        by_label[r["rule_label"]].append(r["frame_idx"])
    for label, frames in sorted(by_label.items()):
        print(f"  [{label}]")
        print(f"    rows={len(frames)}  frame_range=[{min(frames)}, {max(frames)}]")

    print(f"\n  Report saved → {OUT_PATH}\n")


if __name__ == "__main__":
    main()
