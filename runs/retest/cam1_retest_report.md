# CAM1 Re-test Report

- Generated: 2026-04-11T16:14:29
- Tracks CSV: `runs/kpi_batch/retail-shop_CAM1_tracks.csv`
- Audit CSV: `experiments/audit/cam1_manual_audit_sheet.csv`

## Global Metrics

- Positive IDs: 10 (max id=10)
- Unassigned rows: 3422
- Duplicate positive-ID frames: 0
- Re-entry attempts: 5
- Correct re-entry reuses: 5
- False re-entry merges: 0
- Missed re-entry merges: 0

## Per-Person Metrics

| gt_person_id | audited_frames | coverage | switches | dominant_gid | dominant_ratio |
|---|---:|---:|---:|---:|---:|
| 1.0 | 8 | 0.625 | 0 | 7 | 1.000 |
| 10.0 | 26 | 0.923 | 0 | 10 | 1.000 |
| 11.0 | 10 | 1.000 | 0 | 7 | 1.000 |
| 2.0 | 102 | 1.000 | 0 | 4 | 1.000 |
| 3.0 | 121 | 1.000 | 0 | 1 | 1.000 |
| 4.0 | 2 | 1.000 | 0 | 5 | 1.000 |
| 5.0 | 20 | 1.000 | 2 | 2 | 0.950 |
| 6.0 | 33 | 1.000 | 0 | 6 | 1.000 |
| 7.0 | 135 | 1.000 | 0 | 3 | 1.000 |
| 8.0 | 23 | 1.000 | 0 | 8 | 1.000 |
| 9.0 | 4 | 0.750 | 0 | 9 | 1.000 |

## Thresholds

- `max_positive_ids <= 12`
- `max_switches_per_person <= 2`
- `min_coverage_per_person >= 0.8`
- `duplicate_positive_id_frames == 0`

