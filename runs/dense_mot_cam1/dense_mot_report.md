# Dense CAM1 MOT Evaluation Report

Generated at: 2026-04-29T01:22:09

## Section A: Dense MOT Metrics (valid only on dense_gt.csv)
| method | frames_evaluated | gt_boxes | pred_boxes | ignore_suppressed | ignore_iou_th | matched_boxes | FP | FN | IDSW | MOTA | IDP | IDR | IDF1 | HOTA | mostly_tracked | mostly_lost | det_precision | det_recall | mean_iou | notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Full AIC-ReID (frozen) | 300 | 1110 | 530 | 0 | 0.500 | 84 | 446 | 1026 | 3 | -0.328829 | NA | NA | NA | NA | 1 | 28 | 0.158491 | 0.075676 | 0.576971 | IDP/IDR/IDF1 not computed from a trusted library because motmetrics is unavailable.; HOTA not computed; requires TrackEval or an equivalent validated implementation. |
| ByteTrack-only baseline (reid_off) | 300 | 1110 | 971 | 0 | 0.500 | 87 | 884 | 1023 | 1 | -0.718919 | NA | NA | NA | NA | 1 | 28 | 0.089598 | 0.078378 | 0.583160 | IDP/IDR/IDF1 not computed from a trusted library because motmetrics is unavailable.; HOTA not computed; requires TrackEval or an equivalent validated implementation. |
| BoT-SORT-style online baseline | 300 | 1110 | 1001 | 0 | 0.500 | 89 | 912 | 1021 | 1 | -0.742342 | NA | NA | NA | NA | 1 | 28 | 0.088911 | 0.080180 | 0.581615 | IDP/IDR/IDF1 not computed from a trusted library because motmetrics is unavailable.; HOTA not computed; requires TrackEval or an equivalent validated implementation. |
| AIC-ReID (no audit-guided offline repair) | 300 | 1110 | 524 | 0 | 0.500 | 84 | 440 | 1026 | 0 | -0.320721 | NA | NA | NA | NA | 1 | 28 | 0.160305 | 0.075676 | 0.576971 | IDP/IDR/IDF1 not computed from a trusted library because motmetrics is unavailable.; HOTA not computed; requires TrackEval or an equivalent validated implementation. |

## Section B: Deployment-Oriented Sparse Audit Metrics (separate from dense MOT)
| method | pred_to_gt_purity | positive_coverage | same_frame_duplicate_positive_ids | reentry_accepted | reentry_rejected | fragmentation | notes |
|---|---:|---:|---:|---:|---:|---:|---|
| Full AIC-ReID (frozen) | 0.977791 | 0.588843 | 0 | None | None | 1 | reentry_stats_not_found |
| ByteTrack-only baseline (reid_off) | 0.968922 | 0.768595 | 0 | None | None | 9 | reentry_stats_not_found |
| BoT-SORT-style online baseline | 0.920839 | 0.780992 | 0 | None | None | 11 | reentry_stats_not_found |
| AIC-ReID (no audit-guided offline repair) | 0.863004 | 0.508264 | 0 | None | None | 6 | reentry_stats_not_found |

## Notes
- Standard MOT metrics are computed only from dense GT (`dense_gt.csv`).
- Sparse audit metrics are retained for deployment/KPI interpretation only.
- HOTA is reported as `null` unless a validated TrackEval-equivalent implementation is integrated.
- Dense metrics graph: `/Users/ohqishean/AIC-REID/report_assets/graphs/dense_mot_cam1_metrics_comparison.png`
