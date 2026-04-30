# AIC-ReID Evidence Inventory
**Generated:** 2026-04-28  
**Project:** An Audit-Driven Identity-Consistent Person Re-Identification Pipeline for Retail Customer Behaviour Analytics  
**Student:** Oh Qi Shean | 20512381 | University of Nottingham Malaysia  

---

## 1. CSV Artifacts

| File | Path | Contents | Used In |
|------|------|----------|---------|
| `retail-shop_CAM1_tracks.csv` | `runs/kpi_batch/` | 6,194 rows; 3,269 positive (gid>0); 9 positive IDs; 0 same-frame duplicates | Section 8.4 CAM1 Benchmark |
| `retail-shop_FULL_CAM1_tracks.csv` | `runs/kpi_batch/` | FULL_CAM1 canonical slot assignment rows | Section 8.6 FULL_CAM1 |
| `cross_camera_matches.csv` | `runs/cross_cam/` | 24 match events; canonical_primary=5; fragment_reuse=10; score range 0.660–0.865 | Section 8.9 Cross-Camera |
| `results.csv` | `archive/old_runs/train/yolo_cam1_person/` | YOLO training: 8 epochs, precision=0.905, recall=0.917, mAP50=0.952, mAP50-95=0.790 | Section 8.2 YOLO Training |
| `audit_log.csv` | `runs/kpi_batch/` | 484 audit rows; 11 GT persons; pred→GT macro purity 95.6% | Section 8.4 |

---

## 2. JSON Artifacts

| File | Path | Contents | Used In |
|------|------|----------|---------|
| `cam1_retest_report.json` | `runs/retest/` | pass=false; failures: GT1 coverage 0.625, GT9 coverage 0.750; reentry 5/5 correct; 0 false merges | Section 8.5 Retest |
| `identity_metrics.json` | `runs/kpi_batch/` | Per-identity coverage, purity, timeline metrics for CAM1 | Section 8.4 |
| `preprocess_report.json` | `runs/preprocess/` | 7 videos; per-video bad-frame counts, white-ratio stats, accepted/rejected frame counts | Section 8.1 Preprocessing |
| `osnet_train_metrics.json` | `experiments/reid_train/` | OSNet-x1_0 mAP=83.5%, Rank-1=100%, Rank-5=100%; 5-identity domain split | Section 8.3 OSNet |
| `fullcam1_identity_summary.json` | `runs/kpi_batch/` | FULL_CAM1 3,616 frames; 297.34s; 144 bad frames (3.98%); 49 accepted / 91 rejected re-entries | Section 8.6 |
| `zone_kpi_report.json` | `runs/kpi_batch/` | Zone dwell times, entry counts, engagement flags; CAM1_Z02 conflict noted | Section 8.10 KPI |

---

## 3. Video Files

| File | Path | Resolution | FPS | Duration | Used In |
|------|------|-----------|-----|---------|---------|
| `retail-shop_CAM1.mp4` | `data/videos/` | 2560×1944 | 12 | 150s (1,800 frames) | CAM1 benchmark |
| `retail-shop_FULL_CAM1.mp4` | `data/videos/` | 2560×1944 | 12 | 297.34s (3,616 frames) | FULL_CAM1 evaluation |
| `retail-shop_DEMO_VIDEO.mp4` | `data/videos/` | — | — | — | Section 8.7 Demo |
| `overhead_cam.mp4` | `data/videos/` | — | — | — | Section 8.8 Overhead |
| `cross_cam1.mp4` / `cross_cam2.mp4` | `data/videos/` | — | — | — | Section 8.9 Cross-Camera |

---

## 4. Generated Graphs (report_assets/graphs/)

| Filename | Description | Data Source |
|----------|-------------|------------|
| `yolo_training_metrics.png` | 3-panel: precision/recall, mAP50/mAP50-95, train/val loss across 8 epochs | `results.csv` |
| `preprocessing_bad_frame_percentage.png` | Bar chart bad-frame % by video (green/orange/red) | `preprocess_report.json` |
| `cam1_gt_coverage_bar.png` | Grouped bars: audit rows vs positive rows + coverage % line for GT1–GT11 | `audit_log.csv` / `identity_metrics.json` |
| `cam1_id_row_distribution.png` | Per-pred-ID row counts coloured by purity band | `retail-shop_CAM1_tracks.csv` |
| `positive_vs_zero_gid_distribution.png` | Stacked bars: positive vs unidentified rows by clip | `retail-shop_CAM1_tracks.csv` |
| `reentry_decision_breakdown.png` | FULL_CAM1 re-entry decisions: 49 accepted, 83 ambiguous, 5 weak, 3 conflict, 96 new | `fullcam1_identity_summary.json` |
| `cross_camera_match_types.png` | 24 cross-camera match events by type | `cross_camera_matches.csv` |
| `cam1_retest_coverage.png` | Per-person coverage vs 80% threshold (GT1 and GT9 below threshold) | `cam1_retest_report.json` |
| `fullcam1_canonical_id_rows.png` | FULL_CAM1 canonical slot row counts (ID7=0 highlighted) | `retail-shop_FULL_CAM1_tracks.csv` |
| `reid_training_rank_map.png` | OSNet mAP=83.5%, Rank-1=100%, Rank-5=100% bar chart | `osnet_train_metrics.json` |
| `preprocessing_summary_table.png` | Preprocessing stats rendered as figure table | `preprocess_report.json` |
| `cam1_identity_timeline.png` | Scatter: positive-ID assignments over time for CAM1 | `retail-shop_CAM1_tracks.csv` |
| `fullcam1_identity_timeline.png` | Scatter: FULL_CAM1 canonical IDs over 297 seconds | `retail-shop_FULL_CAM1_tracks.csv` |

---

## 5. Screenshots (report_assets/screenshots/)

| Filename | Description | Source |
|----------|-------------|--------|
| `fig_cam1_frame_30s.png` | Raw CAM1 frame at 30s (360th frame) | `retail-shop_CAM1.mp4` |
| `fig_cam1_frame_60s.png` | Raw CAM1 frame at 60s (720th frame) | `retail-shop_CAM1.mp4` |
| `fig_fullcam1_frame_60s.png` | Raw FULL_CAM1 frame at 60s | `retail-shop_FULL_CAM1.mp4` |
| `fig_fullcam1_frame_200s.png` | Raw FULL_CAM1 frame at 200s | `retail-shop_FULL_CAM1.mp4` |
| `fig_demo_video_frame.png` | Raw Demo_Video frame | `retail-shop_DEMO_VIDEO.mp4` |
| `fig_cross_cam1_frame.png` | Raw cross-camera CAM1 frame | `cross_cam1.mp4` |
| `fig_cross_cam2_frame.png` | Raw cross-camera CAM2 frame | `cross_cam2.mp4` |
| `fig_cam1_annotated_5s.png` | Annotated CAM1 frame at 5s with bounding boxes + GIDs | Pipeline output |
| `fig_cam1_annotated_15s.png` | Annotated CAM1 frame at 15s | Pipeline output |
| `fig_cam1_annotated_30s.png` | Annotated CAM1 frame at 30s | Pipeline output |
| `fig_cam1_annotated_60s.png` | Annotated CAM1 frame at 60s | Pipeline output |
| `fig_cam1_annotated_90s.png` | Annotated CAM1 frame at 90s | Pipeline output |
| `fig_overhead_crop_annotated.png` | Annotated overhead camera crop | `overhead_cam.mp4` output |
| `fig_cross_cam1_annotated.png` | Annotated cross-camera frame | Cross-cam pipeline output |
| `fig_motivation_raw_vs_annotated.png` | Composite: raw vs annotated side-by-side (Motivation chapter) | Composite |
| `fig_cross_camera_setup.png` | Composite: dual-camera setup illustration | Composite |
| `fig_viewpoint_comparison.png` | Composite: standard vs overhead viewpoint | Composite |

---

## 6. Model Weights

| File | Path | Description |
|------|------|-------------|
| `yolov8m_cam1_person.pt` | `models/` | YOLOv8m fine-tuned on retail CCTV person detection |
| `osnet_x1_0_retail.pth` | `models/` | OSNet-x1_0 fine-tuned on 5-identity retail domain split |

---

## 7. Configuration Files

| File | Path | Key Parameters |
|------|------|---------------|
| `reid_config.yaml` | `configs/` | hard_thresh=0.90, soft_thresh=0.84, ema=0.93, max_prototypes=36 |
| `tracker_config.yaml` | `configs/` | confirm_hits=4, track_buffer=220, det_conf=0.20 |
| `reentry_config.yaml` | `configs/` | max_reentry_gap_frames=1600, cross_person_ambiguity_margin=0.045, strong_reuse_score=0.70 |
| `frame_quality_config.yaml` | `configs/` | white_ratio_threshold=0.16, mean_gray_hard=225.0, std_gray_floor=8.0 |

---

## 8. Key Metrics Summary (All Traceable)

| Metric | Value | Source Artifact |
|--------|-------|----------------|
| YOLO mAP50 | 0.952 | `results.csv` epoch 8 |
| YOLO mAP50-95 | 0.790 | `results.csv` epoch 8 |
| YOLO Precision | 0.905 | `results.csv` epoch 8 |
| YOLO Recall | 0.917 | `results.csv` epoch 8 |
| OSNet mAP | 83.5% | `osnet_train_metrics.json` |
| OSNet Rank-1 | 100% | `osnet_train_metrics.json` |
| CAM1 pred→GT purity | 95.6% | `audit_log.csv` |
| CAM1 same-frame duplicates | 0 | `retail-shop_CAM1_tracks.csv` |
| CAM1 total rows | 6,194 | `retail-shop_CAM1_tracks.csv` |
| CAM1 positive rows | 3,269 | `retail-shop_CAM1_tracks.csv` |
| FULL_CAM1 frames | 3,616 | `fullcam1_identity_summary.json` |
| FULL_CAM1 bad frames | 144 (3.98%) | `fullcam1_identity_summary.json` |
| FULL_CAM1 re-entries accepted | 49 | `fullcam1_identity_summary.json` |
| FULL_CAM1 re-entries rejected | 91 | `fullcam1_identity_summary.json` |
| Retest reentry correct | 5/5 | `cam1_retest_report.json` |
| Retest false merges | 0 | `cam1_retest_report.json` |
| Retest formal result | FAIL | `cam1_retest_report.json` |
| GT7 positive coverage | 2.2% | `identity_metrics.json` |
| Cross-camera events | 24 | `cross_camera_matches.csv` |
| Cross-camera score range | 0.660–0.865 | `cross_camera_matches.csv` |
