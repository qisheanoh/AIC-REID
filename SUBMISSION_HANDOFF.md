# AIC-REID — Submission Handoff
**Date:** 2026-04-30  
**Student:** 20512381  
**Status:** Submission-ready. Final verification passed 2026-04-30.

---

## 1. What This Project Contains

An audit-driven, identity-consistent person re-identification pipeline for retail CCTV analytics.
The project covers:

- Custom YOLOv8 person detector (trained on CAM1 audit crops)
- Custom OSNet ReID model (trained on Market1501-style split from CAM1 audit data)
- ByteTrack + BoT-SORT-style online multi-object tracker with appearance fusion
- Offline identity resolution: tracklet stitching, re-entry linking, audit-guided ID enforcement
- CAM1 anchor alignment for FULL_CAM1 canonical identity mapping
- Cross-camera identity matching (CAM1 ↔ CAM2)
- Zone-based retail analytics and KPI export
- Dense MOT evaluation, ablation study, and retest evidence

All source code is in `src/`, all runnable scripts are in `scripts/`, configs in `configs/`, models in `models/`.

---

## 2. Final Report Files

All final report documents are at the **project root**:

| File | Description |
|------|-------------|
| `FYP_AIC-REID_20512381.docx` | Submitted dissertation (Word, with embedded figures) |
| `FYP_AIC-REID_20512381.pdf` | Submitted dissertation (PDF, primary submission copy) |
| `FYP_AIC-REID_20512381_AUDIT_APPLIED.pdf` | Audit-annotated version |
| `An Audit-Driven Identity-Consistent Person Re-Identification Pipeline for Retail Customer Behaviour Analytics.pdf` | Publish-level title variant |
| `dissertation_AIC_ReID_publish_level.pdf` | Publish-level full draft |

Supporting documentation:

| File | Description |
|------|-------------|
| `docs/final_report.md` | Master markdown source for the report |
| `docs/submission_checklist.md` | Pre-submission checklist |
| `docs/demo_rehearsal_checklist.md` | Viva/demo rehearsal checklist |
| `docs/demo_video_script.md` | Demo video narration script |
| `report_assets/graphs/` | All report-embedded graphs (PNG) |
| `report_assets/screenshots/` | Report-embedded screenshots (`fig_*` prefix only) |

---

## 3. Frozen Official CAM1 Baseline

The submitted, cited CAM1 baseline is frozen at:

```
runs/baseline_freeze_2026-04-28/retail-shop_CAM1_tracks.FROZEN.csv
```

**MD5:** `b9e3e9052d1cacff1334aa82815cc817`

Key metrics (all cited report figures refer to this file):

- Positive rows: **3,256**
- Positive IDs: **9** (IDs 1–6, 8–10)
- Pred→GT macro purity: **95.6%** (97.8% pred→GT)
- Positive-ID coverage: **52.6%** (3,256 / 6,194 total rows)
- Same-frame duplicate positive IDs: **0** across all 1,800 frames

Freeze notes: `runs/baseline_freeze_2026-04-28/BASELINE_FREEZE_NOTE.md`  
Frozen pipeline snapshot: `runs/baseline_freeze_2026-04-28/run_batch.FROZEN.py`  
Frozen linker snapshot: `runs/baseline_freeze_2026-04-28/track_linker.FROZEN.py`

---

## 4. Baseline Comparison Artifacts

### Method comparison baselines

| Method | Directory |
|--------|-----------|
| BoT-SORT-style online | `runs/baselines/cam1_botsort_online_20260428_221824/` |
| ByteTrack-only (reid=off) | `runs/baselines/cam1_bytetrack_only_20260428_223244/` |
| AIC-ReID no audit-guided offline repair | `runs/baselines/cam1_no_audit_offline_repair_20260428_220407/` |

Each baseline directory contains a `retail-shop_CAM1_tracks.csv` and a `score_locked_protocol.json` with locked scores.

### Dense MOT evaluation

```
runs/dense_mot_cam1/
  dense_mot_report.md          — narrative report
  dense_mot_summary.csv / .json
  per_method_metrics/          — locked JSON + TXT per method (4 files × 2 formats)
```

### Ablation study

```
runs/ablations/cam1_ablation_20260428_225623/
  ablation_report.md
  ablation_summary.csv / .json
  commands_run.sh
  no_offline_reentry_linking/
  no_tracklet_stitching/
  osnet_only_no_attire_shape/
  relaxed_conservative_gate/
  no_cam1_anchor_alignment/
```

### Retest results

```
runs/retest/
  cam1_retest_report.md
  cam1_retest_report.json
```

### Diagnostic evidence

```
runs/diagnostics/
  cam1_missing_persons_90_150_trace.csv
  fullcam1_zero_regression_278_rows.csv
  reid_ambiguity_root_cause_gt7_gt10_gt11.csv
  tracker_continuity_recovery_debug.csv
```

### Ground truth audit

```
experiments/audit/
  cam1_manual_audit_sheet.csv
  cam1_identity_map_template.csv
experiments/dense_mot_cam1/dense_gt.csv
```

---

## 5. Final Demo and Per-Person Rendered Videos

### KPI batch outputs

```
runs/kpi_batch/
  retail-shop_CAM1_vis.mp4           — CAM1 150 s benchmark
  retail-shop_FULL_CAM1_vis.mp4      — FULL_CAM1 ~20 min sequence
  retail-shop_Demo_Video_vis.mp4     — demo video (30 s)
  retail-shop_2_2_crop_vis.mp4       — overhead crop
```

### Cross-camera

```
runs/cross_cam/
  cross_cam1_vis.mp4
  cross_cam2_vis.mp4
```

### Per-person identity persistence

```
runs/Per_Person/
  person_01_1_2_crop_vis.mp4
  person_01_1_3_crop_vis.mp4
  person_02_2_1_crop_vis.mp4
  person_02_2_2_crop_vis.mp4
  person_02_2_3_crop_vis.mp4
```

### Demo slides

```
outputs/fyp_demo_slides/output.pptx
```

---

## 6. What Was Archived During Pre-Submission Cleanup

A pre-submission cleanup was performed on 2026-04-30. Archived material is kept intact at:

```
archive/pre_submission_cleanup_20260430/
  MOVE_LOG.txt    — 61 items moved, 0 errors
  DELETE_LOG.txt  — 24 items deleted, ~395 MB freed
```

Largest archived items:
- `runs/kpi_batch/_pre_rerender_backup/` (816 MB) — superseded by current renders
- `runs/kpi_batch/retail-shop_FULL_CAM1_vis.OLD_BEFORE_REGEN.mp4` (412 MB) — pre-regen render
- `runs/dense_mot_cam1/frames/` (478 MB) — extracted evaluation frames
- `runs/dense_mot_cam1/contact_sheets/` (53 MB) — evaluation contact sheet images
- Incremental CAM1 fix snapshots (`.pre-gt*-promo.csv`)
- Intermediate experiment outputs and old report drafts

Nothing in `PROTECTED_MANIFEST.txt` was moved or deleted. The manifest (`PROTECTED_MANIFEST.txt`) lists all 173 protected artifacts by path.

---

## 7. How to Rerun Key Results

### Prerequisites

```bash
bash scripts/setup_env.sh        # creates conda env aicenv
conda activate aicenv
```

Models required: `models/yolo_cam1_person.pt`, `models/osnet_cam1.pth`  
Raw videos required: `data/raw/retail-shop/` and `data/raw/cross_cam/`

### Reproduce the official CAM1 result

```bash
python scripts/run_batch.py \
    --group configs/cameras/kpi_retailshop_cam1.yaml
```

Output: `runs/kpi_batch/retail-shop_CAM1_tracks.csv` + `_vis.mp4`

> The current `runs/kpi_batch/retail-shop_CAM1_tracks.csv` reflects the **post-submission**
> gt8_reseq_v1 experiment (3,269 rows). To reproduce the **frozen submitted baseline**
> exactly, use `run_batch.FROZEN.py` from `runs/baseline_freeze_2026-04-28/`.

### Reproduce the FULL_CAM1 result

```bash
python scripts/run_batch.py \
    --group configs/cameras/kpi_retailshop_full_cam1_cam1.yaml
```

### Run cross-camera matching

```bash
python scripts/run_cross_cam.py \
    --config configs/cameras/kpi_cross_cam.yaml
```

### Rerun ablation study

```bash
python scripts/run_ablation_study.py
```

### Rerun dense MOT evaluation

```bash
python scripts/run_dense_mot_evaluation.py
python scripts/evaluate_dense_mot_metrics.py
```

### Online-only tracking (no offline resolution)

```bash
python scripts/run_online_tracking.py --video data/raw/retail-shop/CAM1.mp4
# Add --reid_off to approximate ByteTrack-only behavior
```

---

## 8. Viva: What Not to Confuse

### Official frozen baseline vs later exploratory results

| | Frozen submitted baseline | Post-submission exploratory (gt8_reseq_v1) |
|--|--|--|
| **File** | `runs/baseline_freeze_2026-04-28/retail-shop_CAM1_tracks.FROZEN.csv` | `runs/kpi_batch/retail-shop_CAM1_tracks.csv` |
| **Positive rows** | **3,256** | 3,269 |
| **ID9 rows** | 102 | 115 |
| **MD5** | `b9e3e9052d1cacff1334aa82815cc817` | different |
| **Cited in report?** | **Yes — all metrics refer to this file** | No — documented as post-submission experiment |

The 13-row difference comes from re-sequencing the GT8 gid=0 promotion to run after both pair-splits (gt8_reseq_v1). The experiment confirmed the fix but was not incorporated into the submitted baseline. The report explicitly acknowledges this at section 8.3 and in the frozen note.

### Other things to note for viva

- **GT7 (2.2% coverage):** architectural limitation of the active-owner zone protection gate — not an unresolved bug. Documented in section 8.10. Experiments confirmed it cannot be closed without causing regression in stable identities.
- **GT8 partial coverage in submitted baseline:** sequencing artifact (promotion ran before pair-splits). Root cause identified and resolved post-submission. Documented in section 8.3.
- **FULL_CAM1 "2,773 final positive rows":** this is the count after the custom 8-ID remap excludes the off-canonical GID (1,090 rows), from a pre-remap total of 3,863. The sub-label names in the breakdown ("old GID 12", "old GID 10") reflect GID numbering at the time the identity metrics JSON was recorded; the total of 1,090 excluded rows is unchanged.
- **Dense MOT MOTA values are negative:** expected — the evaluation window has sparse detections (530 pred boxes vs 1,110 GT boxes). IDF1/HOTA are null because motmetrics/TrackEval were unavailable on this machine; the audit-protocol purity/coverage metrics are the primary evaluation.
- **Coverage definition:** the report's "52.6% positive-ID coverage" is positive rows / total rows (3,256 / 6,194). The ablation summary's "coverage: 0.5888" is audit-row coverage (285 / 484 audit instances matched). These are two different metrics, both used consistently.
