# Experiment Log — reid-next-coverage-experiments

## Branch origin

- Starting from commit: `1fd96bd` (main — "Initial commit of BoT-SORT + ReID tracking pipeline")
- Frozen baseline directory: `runs/baseline_freeze_2026-04-28/`
- Frozen CAM1 CSV MD5: `b9e3e9052d1cacff1334aa82815cc817`

## Accepted baseline (do not modify)

| Metric | Value |
|--------|-------|
| Total rows | 6,194 |
| Positive rows | 3,256 (52.6%) |
| Zero-gid rows | 2,938 (47.4%) |
| Same-ID same-frame dups | 0 |
| Macro pred→GT purity | 0.9558 (95.6%) |
| Per-ID counts | {1:1331, 2:1028, 3:515, 4:2, 5:31, 6:81, 8:58, 9:102, 10:108} |

Active promote map:
```python
promote_pred0_to_target_from_audit(
    target_gid_by_gt={"11": 8, "6": 6, "10": 10, "2": 2},
    iou_threshold=0.34,
)
```

## Target

- Positive row coverage > 52.6%
- Macro pred→GT purity ≥ 0.90 (hard floor)
- Same-ID same-frame duplicates = 0 (hard floor)
- Positive rows must not decrease from 3,256

## Adoption criterion (per experiment)

Before adopting any experiment result:
1. `positive_rows >= 3256`
2. `macro_purity >= 0.90`
3. `same_id_dups == 0`
4. Per-ID counts for IDs {1, 2, 3, 5, 6, 8, 9, 10} must not regress materially

## Naming convention for CAM1 backups

`retail-shop_CAM1_tracks.csv.pre-<experiment-name>.csv`

## Known non-starters (do not attempt)

- **GT7:** Gate-relaxation experiment run and reverted 2026-04-26. Collapsed ID3 515→12. Do not attempt.
- **GT8:** pair-split (5 vs 9) corrupts promoted rows. Safe only if promote is re-sequenced AFTER pair-split.

## Experiment entries

| Date | Experiment | Changed rows | Purity before | Purity after | Dups | Decision |
|------|-----------|--------------|--------------|-------------|------|----------|
| — | (none yet) | — | 0.9558 | — | 0 | — |
