# AIC-ReID CAM1/FULL_CAM1 Ablation Study

## 1. Purpose
This ablation study isolates identity-resolution components to test whether identity consistency gains come from the audit-driven offline workflow rather than detector/backbone alone.

## 2. Exact Commands Run
```bash

/Users/ohqishean/AIC-REID/.venv/bin/python scripts/run_batch.py --match =CAM1 --no-render --out_dir /Users/ohqishean/AIC-REID/runs/ablations/cam1_ablation_20260428_225623/no_offline_reentry_linking --disable_reentry_linking
/Users/ohqishean/AIC-REID/.venv/bin/python scripts/run_batch.py --match =CAM1 --no-render --out_dir /Users/ohqishean/AIC-REID/runs/ablations/cam1_ablation_20260428_225623/no_tracklet_stitching --disable_tracklet_stitching
/Users/ohqishean/AIC-REID/.venv/bin/python scripts/run_batch.py --match =CAM1 --no-render --out_dir /Users/ohqishean/AIC-REID/runs/ablations/cam1_ablation_20260428_225623/osnet_only_no_attire_shape --osnet_only
/Users/ohqishean/AIC-REID/.venv/bin/python scripts/run_batch.py --match =CAM1 --no-render --out_dir /Users/ohqishean/AIC-REID/runs/ablations/cam1_ablation_20260428_225623/relaxed_conservative_gate --relaxed_identity_gate
FULL_CAM1_CAM1_ANCHOR=off FULL_CAM1_FORCE_CUSTOM8_IDS=0 /Users/ohqishean/AIC-REID/.venv/bin/python scripts/run_batch.py --match =FULL_CAM1 --no-render --out_dir /Users/ohqishean/AIC-REID/runs/ablations/cam1_ablation_20260428_225623/no_cam1_anchor_alignment
```

## 3. CAM1 Variants (Locked Audit Protocol)
| method | variant_name | status | positive_ids | positive_rows | coverage | purity | same_frame_duplicates | fragmentation | tracks_csv |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| AIC-ReID | full_aic_reid_frozen_reference | RUN | 9 | 3256 | 58.88% | 97.78% | 0 | 1 | /Users/ohqishean/AIC-REID/runs/baseline_freeze_2026-04-28/retail-shop_CAM1_tracks.FROZEN.csv |
| AIC-ReID | no_offline_reentry_linking | RUN | 10 | 3326 | 61.16% | 90.57% | 0 | 4 | /Users/ohqishean/AIC-REID/runs/ablations/cam1_ablation_20260428_225623/no_offline_reentry_linking/retail-shop_CAM1_tracks.csv |
| AIC-ReID | no_tracklet_stitching | RUN | 9 | 2769 | 61.57% | 97.22% | 0 | 2 | /Users/ohqishean/AIC-REID/runs/ablations/cam1_ablation_20260428_225623/no_tracklet_stitching/retail-shop_CAM1_tracks.csv |
| AIC-ReID | osnet_only_no_attire_shape | RUN | 8 | 2797 | 62.81% | 96.53% | 0 | 3 | /Users/ohqishean/AIC-REID/runs/ablations/cam1_ablation_20260428_225623/osnet_only_no_attire_shape/retail-shop_CAM1_tracks.csv |
| AIC-ReID | relaxed_conservative_gate | RUN | 9 | 2397 | 61.78% | 97.22% | 0 | 3 | /Users/ohqishean/AIC-REID/runs/ablations/cam1_ablation_20260428_225623/relaxed_conservative_gate/retail-shop_CAM1_tracks.csv |
| ByteTrack-only baseline (reid_off) | existing_baseline_bytetrack_only | RUN | 118 | 5573 | 76.86% | 96.89% | 0 | 9 | /Users/ohqishean/AIC-REID/runs/baselines/cam1_bytetrack_only_20260428_223244/retail-shop_CAM1_tracks.csv |
| BoT-SORT-style online baseline | existing_baseline_botsort_online | RUN | 84 | 5747 | 78.10% | 92.08% | 0 | 11 | /Users/ohqishean/AIC-REID/runs/baselines/cam1_botsort_online_20260428_221824/retail-shop_CAM1_tracks.csv |
| AIC-ReID (no audit-guided offline repair) | existing_baseline_no_audit_offline_repair | RUN | 12 | 3242 | 50.83% | 86.30% | 0 | 6 | /Users/ohqishean/AIC-REID/runs/baselines/cam1_no_audit_offline_repair_20260428_220407/retail-shop_CAM1_tracks.csv |

## 4. FULL_CAM1 No-Anchor Variant
| method | variant_name | status | positive_ids | positive_rows | same_frame_duplicates | canonical_slots_assigned | off_canonical_rows | anchor_report_exists | tracks_csv |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| AIC-ReID | no_cam1_anchor_alignment | RUN | 11 | 3863 | 0 | 11 | 0 | False | /Users/ohqishean/AIC-REID/runs/ablations/cam1_ablation_20260428_225623/no_cam1_anchor_alignment/retail-shop_FULL_CAM1_tracks.csv |

## 5. Interpretation
Online-oriented baselines can increase CAM1 audit-row positive coverage, but they tend to inflate identity cardinality and/or fragmentation. Ablations that remove or weaken offline identity resolution typically degrade identity consistency (fragmentation and cross-person sharing behavior). The full frozen AIC-ReID reference remains the KPI-trust anchor because it emphasizes identity compression and purity while preserving duplicate-frame safety under the locked CAM1 audit protocol.
For FULL_CAM1, disabling CAM1-anchor alignment is expected to reduce canonical-slot stability and weaken cross-clip interpretability, even when the rest of the offline workflow remains active.

## 6. Limitations
These results are CAM1/FULL_CAM1 specific. CAM1 metrics are audit-sheet-bound (484 manual rows, sparse labels) rather than dense MOT ground truth. Conclusions should be generalized cautiously to longer clips, different stores, and different camera viewpoints.
For the FULL_CAM1 no-anchor variant, CAM1 reference-dependent relabel passes were also skipped because the isolated variant output directory did not include a CAM1 reference tracks CSV; interpret this row as no-anchor plus reference-align-skip behavior.

## 7. Artifacts
- Run root: `/Users/ohqishean/AIC-REID/runs/ablations/cam1_ablation_20260428_225623`
- Summary CSV: `/Users/ohqishean/AIC-REID/runs/ablations/cam1_ablation_20260428_225623/ablation_summary.csv`
- Summary JSON: `/Users/ohqishean/AIC-REID/runs/ablations/cam1_ablation_20260428_225623/ablation_summary.json`
- Command log: `/Users/ohqishean/AIC-REID/runs/ablations/cam1_ablation_20260428_225623/commands_run.sh`
- Runtime log: `/Users/ohqishean/AIC-REID/runs/ablations/cam1_ablation_20260428_225623/run_log.txt`
- Graph: `/Users/ohqishean/AIC-REID/report_assets/graphs/cam1_ablation_identity_quality.png`

