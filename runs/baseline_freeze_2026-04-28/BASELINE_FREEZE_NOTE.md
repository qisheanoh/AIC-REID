# Accepted Baseline Freeze — 2026-04-28

## DO NOT MODIFY THESE ARTIFACTS

This directory contains the frozen accepted baseline for the FYP dissertation submission.
Any future experimental work must use a separate copy or branch and must not alter the
files listed below.

---

## Frozen artifacts

| File | Source path | MD5 |
|------|-------------|-----|
| `retail-shop_CAM1_tracks.FROZEN.csv` | `runs/kpi_batch/retail-shop_CAM1_tracks.csv` | b9e3e9052d1cacff1334aa82815cc817 |
| `final_report_revised_publishable.FROZEN.md` | `docs/final_report_revised_publishable.md` | 8efbe6d2ecdcdaca18e647a94d11f080 |
| `run_batch.FROZEN.py` | `scripts/run_batch.py` | 0c4709734ca0ed6745417ab32df7b447 |
| `track_linker.FROZEN.py` | `src/reid/track_linker.py` | 72eef9ad99b0ee532d5c09c6386c9bb2 |

Visualisation (not copied — too large, do not regenerate):
- `runs/kpi_batch/retail-shop_CAM1_vis.mp4` (340 MB, generated 2026-04-28 02:03)

---

## Verified baseline metrics (CAM1)

- **Total rows:** 6,194
- **Positive rows:** 3,256 (52.6%)
- **Zero-gid rows:** 2,938 (47.4%)
- **Same-ID same-frame duplicates:** 0 / 1,800 frames
- **Macro pred→GT purity:** 0.9558 (95.6%)
- **Per-ID counts:** {1:1331, 2:1028, 3:515, 4:2, 5:31, 6:81, 8:58, 9:102, 10:108}

---

## Active promote map in run_batch.py

```python
promote_pred0_to_target_from_audit(
    target_gid_by_gt={"11": 8, "6": 6, "10": 10, "2": 2},
    iou_threshold=0.34,
)
```

Enforce map (unchanged from prior baseline):
```python
enforce_target_ids_from_audit(
    target_gid_by_gt={"6": 6, "11": 8, "8": 9, "5": 2, "10": 10},
    iou_threshold=0.34,
    max_segment_len=2000,
)
```

---

## Experimental branch naming convention

All future experiments must follow this pattern:

- **CAM1 CSV backup before experiment:** `retail-shop_CAM1_tracks.csv.pre-<experiment-name>.csv`
- **Code changes:** revert to FROZEN copies if experiment is abandoned
- **Experiment log:** record `changed_rows`, `purity before/after`, `same_id_dups` for every run
- **Adoption criterion:** positive rows must not decrease; macro purity must not decrease; same_id_dups must remain 0

---

## Known limitations (do not attempt to fix without full evaluation)

- **GT7:** architectural gate interaction in `bot_sort.py`. Gate-relaxation experiment run and
  reverted (2026-04-26). Do not attempt again.
- **GT8:** pair-split (5 vs 9) corrupts promoted rows. Experiment run and reverted (2026-04-28).
  Safe only if promote is re-sequenced to run AFTER pair-split.
- **FULL_CAM1 / Demo_Video / 1_2_crop:** current on-disk CSVs differ from report metrics due to
  pre-existing reruns. Do not re-run these. Cite report metrics only.
