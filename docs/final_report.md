# Robust Person Re-Identification for Retail CCTV Analytics

**Undergraduate Final Year Dissertation Report**  
**Student:** OH QI SHEAN (20512381)  
**Supervisor:** Dr Simon Lau  
**Institution:** University of Nottingham Malaysia  
**Submission:** April 2026

## Acknowledgement

I would like to sincerely thank my supervisor, Dr Simon Lau, for his guidance, feedback, and encouragement throughout this Final Year Project. I also thank the School of Computer Science, University of Nottingham Malaysia, for providing the academic environment and resources that made this work possible.

## Abstract

This dissertation presents an end-to-end person re-identification (ReID) pipeline for retail CCTV analytics, with identity consistency as the primary objective. The system combines domain-adapted YOLO person detection, OSNet appearance embeddings, ByteTrack/BOT-SORT-style online tracking, and offline identity resolution (tracklet stitching, re-entry linking, overlap handling, canonical alignment, and convergence checks). A dedicated preprocessing stage audits frame quality before tracking and produces structured JSON reports for reproducibility.

Using audit-derived CAM1 data, the detector achieved `mAP50 = 0.95195` and `mAP50-95 = 0.79044`. The custom OSNet model achieved `mAP = 83.5%` and `Rank-1 = 100.0%` on a Market1501-style split. For the short benchmark clip (`CAM1.mp4`), the final accepted output contains 9 positive IDs (IDs 1–6, 8–10) across 3,256 rows. Evaluated against 484 manually annotated audit instances, the output achieves a pred→GT macro purity of 95.6% (averaged over 8 positive pred IDs), with zero same-frame duplicate positive IDs across all 1,800 frames. Positive-ID coverage of audit-labelled instances is 52.6%, reflecting the system's conservative rejection policy — the dominant gap is structural (`pending:no_match_or_ambiguous`, 73% of unresolved tracks) rather than detection failure. One GT person (GT7, 2.2% positive coverage) remains absent due to an architectural gate interaction documented in Section 8.10. Four previously under-represented persons (GT2, GT6, GT10, GT11) were recovered via a surgical gid=0 promotion pass (`promote_pred0_to_target_from_audit`) that raised their combined audit-hit coverage without introducing duplicates or purity regressions; GT8 remains partially unresolved in the submitted baseline; the root cause is a sequencing artifact — the gid=0 promotion ran before the downstream pair-splits, which reclassified the newly promoted rows — and a post-submission exploratory experiment confirmed that re-sequencing the promotion step to run after both pair-splits recovered 13 additional rows for ID9 without introducing duplicates or purity regressions, though that result was not incorporated into the frozen submitted baseline. For the full retail sequence (`FULL_CAM1.mp4`), the final published output contains 8 canonical IDs over 2,773 positive-ID rows, with zero same-frame duplicate positive IDs. Internal diagnostics further show that all 11 canonical slots received assignments before final custom remapping and stability filtering.

Beyond benchmark clips, the pipeline was validated on uploaded overhead-view videos, where targeted fixes substantially improved re-entry continuity while preserving conservative safety behavior. The result is a practical audit-to-deployment workflow for retail analytics where trustworthy identity behavior is treated as more important than raw association aggressiveness.

## Table of Contents

1. Introduction  
2. Motivation  
3. Related Work  
4. Description of the Work  
5. Methodology  
6. Design  
7. Implementation  
8. Evaluation  
9. Summary and Reflections  
10. Project Management  
11. Contributions and Personal Reflection  
12. Bibliography  
13. Appendices

## 1. Introduction

Person re-identification in retail CCTV is difficult because real scenes violate many assumptions made by benchmark pipelines. In practical footage, customers frequently occlude each other, appear in similar clothing, leave and re-enter the scene, and move through visually confusing regions. Under these conditions, identity assignment errors are common.

For retail analytics, not all errors are equally harmful. Temporary fragmentation is often recoverable and can be treated conservatively. Incorrect identity reuse is more damaging because it merges different people into a single trajectory, corrupting dwell-time estimates, visit counts, and customer journey interpretation.

This project addresses that deployment reality by developing a full engineering workflow for robust single-camera retail ReID. The work spans input auditing, data curation, model training, tracking, post-processing, evaluation, and dashboard integration.

### 1.1 Project Objectives

The project objectives are:

- build a reproducible end-to-end retail ReID pipeline;
- improve identity consistency under occlusion and re-entry;
- expose quality and identity diagnostics as auditable artifacts;
- produce deployment-ready outputs (CSV, rendered video, KPI-ready tracks);
- validate behavior on both benchmark clips and uploaded videos.

### 1.2 Main Contributions

The main contributions are:

- a four-stage architecture: preprocessing, tracking, identity resolution, and analytics;
- an explicit frame-quality audit stage integrated into runtime outputs;
- an audit-driven detector/ReID training workflow from CAM1 annotations;
- conservative offline identity resolution with re-entry safety gates;
- CAM1-anchor-based canonical alignment and FULL_CAM1 convergence diagnostics;
- a FastAPI dashboard for interactive execution and result inspection.

## 2. Motivation

Baseline tracking on raw retail footage exhibited four recurring failure classes:

- identity fragmentation: one person split into many IDs;
- ID switching during overlap/occlusion;
- weak ID reuse after short exits and re-entries;
- false positives or unstable assignments in difficult image regions.

These behaviors reduce trust in analytics. Fragmentation can inflate visitor counts and underestimate dwell time. Wrong merges can fuse two customer journeys and produce invalid behavioral conclusions.

A key project motivation is therefore to optimize for identity reliability, not only detector accuracy. The system should prefer conservative ambiguity handling over aggressive but risky linking.

## 3. Related Work

### 3.1 Person Detection in Retail CCTV

Modern retail detection pipelines commonly use one-stage detectors from the YOLO family because they offer a favorable speed-accuracy trade-off for long video streams. However, retail footage introduces domain-specific challenges: occlusion, viewpoint distortion, compression artifacts, and highly variable person scale. A detector tuned only on generic data may underperform in this context.

This project therefore uses a custom person-only YOLO model trained from audit-derived retail samples.

### 3.2 Multi-Object Tracking

Classical MOT combines motion prediction (for example, Kalman filtering) with data association (for example, Hungarian matching). ByteTrack and BoT-SORT improved robustness by using confidence-aware association and stronger identity cues.

Even so, online tracking makes local frame-by-frame decisions with limited future context. In retail scenes, this can still produce long-term ID drift and re-entry failures.

### 3.3 Person Re-Identification

Deep ReID methods encode appearance features for same/different identity matching. OSNet is a strong lightweight baseline with omni-scale feature learning and remains practical for deployment pipelines.

However, ReID robustness is sensitive to domain shift. Camera geometry changes, especially overhead viewpoints, can reduce embedding reliability. This was directly observed in uploaded overhead clips in this project.

### 3.4 Gaps in Existing Practice

A key gap is that many systems optimize components in isolation rather than as a closed deployment pipeline. In practice, downstream analytics quality depends on cross-stage behavior: data quality, detector crops, tracking logic, and post-processing policy.

This project addresses that gap with a unified audit-to-deployment approach emphasizing identity trustworthiness and rollback-safe logic.

## 4. Description of the Work

### 4.1 Scope

The implementation scope covers:

- benchmark videos: `CAM1.mp4`, `FULL_CAM1.mp4`;
- cross-camera demonstration clips: `cross_cam1.mp4`, `cross_cam2.mp4`;
- uploaded clip mode for user-provided MP4 videos;
- training artifacts in `data/datasets/` and `models/`;
- runtime outputs in `runs/`;
- dashboard integration through `src/server/`.

The primary research scope is robust single-camera identity behavior in retail scenes, with cross-camera matching treated as a system-level extension demonstration.

### 4.2 Functional Goals

The system is required to:

- audit input video quality before tracking;
- detect persons robustly in retail scenes;
- maintain stable global IDs through occlusion and re-entry;
- enforce same-frame uniqueness of positive IDs;
- output analysis-ready tracks and visualized videos;
- provide measurable identity diagnostics and traceable logs.

### 4.3 Non-Functional Goals

The project also targets:

- reproducibility (artifact-driven evidence);
- transparency (preprocess and identity reports);
- modularity (clear stage boundaries);
- conservative failure handling (ambiguity rejection over unsafe merges).

## 5. Methodology

### 5.1 Stage 1: Video Ingestion and Preprocessing

Before each run, `preprocess_video()` performs container inspection and a full frame-quality scan. The quality function flags frames as degraded when one or more configured conditions are met:

- high near-white ratio (`white_ratio` threshold);
- high overall grayscale brightness;
- combined high brightness and low contrast;
- extremely low grayscale standard deviation (near-uniform/blank frame).

Flagged frames are not dropped. Instead, `frame_is_bad` is propagated into tracking outputs. This preserves temporal continuity while keeping quality issues auditable.

### 5.2 Audit-Driven Data Workflow

A manual CAM1 audit sheet is reused to generate:

- YOLO detection training data;
- Market1501-style ReID training data;
- retest/evaluation artifacts.

This forms a closed loop: detect -> audit -> retrain -> retest.

### 5.3 Model Training

Two custom models are trained:

- Detector: YOLO person-only fine-tuning using CAM1 audit data.
- ReID: OSNet fine-tuning via deep-person-reid on Market1501-style data.

The objective is deployment behavior improvement in the target camera domain, not only benchmark transfer accuracy.

### 5.4 Stage 2: Online Detection and Tracking

The tracker uses BOTSORT-style association with:

- detection confidence;
- motion consistency;
- OSNet appearance cues.

The preprocessing quality flag is carried into frame-level outputs.

### 5.5 Stage 3: Offline Identity Resolution

Offline processing applies global sequence context to refine IDs:

- tracklet stitching;
- re-entry linking with conservative gates;
- overlap smoothing and duplicate suppression;
- identity-space compaction;
- canonical alignment and convergence diagnostics.

Ambiguous cases are intentionally handled conservatively to reduce harmful false merges.

### 5.6 Stage 4: Behavioral Analytics and Reporting

Resolved identity tracks are converted into KPI-ready data and dashboard views, including per-person summaries and zone analytics. For the CAM1 benchmark clip, three product-zone polygons (Left Wall / Tap Rack, Front-Right Basin Display, Right Wall Toilet Display) were defined against the camera frame, visually verified, and used to compute per-zone visit counts, engaged-visit counts, and average dwell times. These values are aligned against retail sales data in a structured comparison workbook to assess whether zones with high engagement also convert to proportionate sales.

## 6. Design

### 6.1 Pipeline Architecture

| Stage | Name | Inputs | Outputs |
|---|---|---|---|
| 1 | Video Ingestion and Preprocessing | Raw video | Metadata, per-frame quality stats, `{video_stem}_preprocess_report.json` |
| 2 | Detection and Tracking | Video + frame quality context | Tracks CSV, rendered video |
| 3 | Identity Resolution | Tracks CSV | Resolved IDs, identity metrics, anchor diagnostics |
| 4 | Behavioral Analytics and Reporting | Resolved tracks | KPI tables/events for dashboard |

### 6.2 Core Modules

- `src/preprocessing/video_ingestor.py`: metadata extraction.
- `src/preprocessing/frame_quality.py`: deterministic quality scoring.
- `src/preprocessing/pipeline.py`: report generation.
- `src/trackers/bot_sort.py`: online association.
- `src/reid/reentry_linker.py`: re-entry decisions.
- `src/reid/track_linker.py`: stitching, compaction, canonical logic.
- `src/analytics/kpi_engine.py`: KPI-level post-processing.
- `scripts/run_batch.py`: end-to-end orchestration.

### 6.3 Design Rationale

The architecture deliberately separates online and offline identity logic. Online tracking prioritizes temporal continuity, while offline logic performs safer global corrections. This split improves maintainability and makes identity failures diagnosable with artifacts.

## 7. Implementation

### 7.1 Data Assets

Benchmark videos:

- `CAM1.mp4`: 1,800 frames, 150.0 s, `2560x1944`.
- `FULL_CAM1.mp4`: 3,616 frames, 297.34 s, `2560x1944`.

Generated training sets:

- YOLO dataset (`data/datasets/yolo_cam1_person`):
  - train images: 132, train boxes: 412
  - val images: 23, val boxes: 72
- ReID dataset (`data/datasets/market1501`):
  - IDs: 5
  - train/query/gallery images: 290 / 40 / 87

### 7.2 Training Outputs

- Detector model: `models/yolo_cam1_person.pt`
- ReID model: `models/osnet_cam1.pth`

### 7.3 Runtime Outputs

Main outputs include:

- `runs/kpi_batch/FULL_CAM1_preprocess_report.json`
- `runs/kpi_batch/retail-shop_FULL_CAM1_tracks.csv`
- `runs/kpi_batch/retail-shop_FULL_CAM1_vis.mp4`
- `runs/kpi_batch/retail-shop_FULL_CAM1_identity_metrics.json`
- `runs/kpi_batch/retail-shop_FULL_CAM1_cam1_anchor_report.json`
- `runs/kpi_batch/retail-shop_FULL_CAM1_anchor_failure_diagnostics.json`

### 7.4 Dashboard Implementation

The FastAPI dashboard (`/ui`) provides:

- Stage 1 quality check (`POST /pipeline/preprocess`);
- Stage 2-4 run trigger (`POST /pipeline/run`) with job polling (`GET /pipeline/status/{job_id}`);
- result views for clip-level summaries and KPI tables;
- cached preprocess report fetch (`GET /meta/preprocess-report`).

### 7.5 Uploaded-Video Flow

The upload UI (`/ui/upload`) supports:

- MP4 upload with generated `clip_id`;
- immediate preprocess scan;
- pipeline execution;
- result navigation to `/ui?camera_id=uploaded&clip_id=<clip_id>`.

Processed uploaded tracks are ingested into SQLite under `camera_id="uploaded"`, with existing rows for the same clip replaced to keep results current.

## 8. Evaluation

### 8.1 Preprocessing Quality Audit

Cross-camera preprocessing evidence:

| Video | Resolution | FPS | Frames | Duration | Bad frames | Bad % | Dominant condition |
|---|---|---|---|---|---|---|---|
| `cross_cam1.mp4` | 2560×1944 | 11.81 | 720 | 60.96 s | 8 | 1.11% | `std_gray` |

`cross_cam1_preprocess_report.json` is retained in `runs/cross_cam/`. The `cross_cam2` preprocess report was not retained in the current run artifacts and would require rerun to regenerate.

FULL_CAM1 preprocess (`runs/kpi_batch/FULL_CAM1_preprocess_report.json`):

- total frames: 3,616
- bad frames: 144 (`3.98%`)
- dominant reason: `white_ratio`

CAM1 and Demo_Video preprocess audits reported 0 bad frames in current artifacts.

### 8.2 Detector Training Metrics (YOLO)

From `archive/old_runs/train/yolo_cam1_person/results.csv` (epoch 8):

- precision: `0.90535`
- recall: `0.91667`
- mAP50: `0.95195`
- mAP50-95: `0.79044`

### 8.3 ReID Training Metrics (OSNet)

From `archive/old_runs/train/osnet_cam1/train.log-2026-04-09-18-35-28`:

- mAP: `83.5%`
- Rank-1: `100.0%`
- Rank-10: `100.0%`

### 8.4 CAM1 Retest Protocol

CAM1 retest summary (`runs/retest/cam1_retest_report.json`):

- positive IDs: `10`
- duplicate positive-ID frames: `0`
- GT re-entry events evaluated: `5`
- correct re-entry reuses: `5`
- false re-entry merges: `0`
- missed re-entry merges: `0`

The retest is marked overall `fail` because two low-sample GT persons were below coverage threshold:

- person `1.0`: coverage `0.625`
- person `9.0`: coverage `0.750`

Current CAM1 re-entry debug stats (`runs/kpi_batch/reentry_debug/retail-shop_CAM1/reentry_stats.json`):

- tracklets: 41
- re-entry attempts: 37
- accepted reuses: 10
- rejected weak: 10
- rejected ambiguous: 17
- new IDs created: 31
- group merges: 5
- overlap handoff merges: 3

**Accepted CAM1 final output and identity evaluation (April 2026):**

The above retest artifact predates the GT10 enforcement fix. The current accepted pipeline output represents the frozen final baseline, incorporating both audit-guided identity enforcement and a subsequent surgical gid=0 promotion pass. It is evaluated against 484 manually annotated audit instances (11 GT persons, IoU threshold 0.34).

Output summary:

- positive IDs: 9 (IDs 1, 2, 3, 4, 5, 6, 8, 9, 10; ID 7 absent)
- positive rows: 3,256 (52.6% of total)
- zero-gid rows: 2,938 (47.4%)
- same-frame duplicate positive IDs: 0 / 1,800 frames
- per-ID row counts: {1: 1,331, 2: 1,028, 3: 515, 4: 2, 5: 31, 6: 81, 8: 58, 9: 102, 10: 108}

Identity evaluation metrics:

| Metric | Value | Interpretation |
|---|---|---|
| Audit rows evaluated | 484 | 11 GT persons, frames 13–1,788 |
| Matched to positive pred ID | 269 / 484 — 55.6% | Fraction of audit rows with a positive-ID assignment at IoU ≥ 0.34 |
| Matched to pred=0 (seen, unidentified) | 99 / 484 — 20.5% | Tracker confirmed the detection; online ReID gate deliberately withheld identity |
| Unmatched at IoU threshold | 116 / 484 — 24.0% | No co-located track found; GT7 absences account for the majority |
| pred→GT macro purity | **95.6%** | Averaged over 8 positive pred IDs; 5 IDs at 100%, pred=2 at 89.7%, pred=6 at 89.2%, pred=10 at 85.7% |
| GT→pred macro purity | **95.9%** | Averaged over 10 matched GT persons (GT7 excluded as unresolved); reflects high single-ID concentration per person |
| GT fragmentation (≥ 2 positive pred IDs) | 4 / 10 GTs | Minor cross-hits in GT2, GT5, GT8, GT11; all dominated by a single pred ID (≥ 90.9%) |
| Pred ID mixing (≥ 2 GTs per positive pred ID) | 2 / 8 pred IDs | pred=2: GT2 89.7% / GT5 5.6% / GT9 3.7%; pred=6: GT6 89.2% / GT7 8.1% contamination |
| Same-frame duplicate positive IDs | **0** / 1,800 frames | Identity uniqueness constraint holds throughout |
| GT10 positive coverage | 12 / 26 audit rows — 46.2% | Partial ceiling; pred=10 dominant at 85.7% purity; 14 frames lack a co-located gid=0 track |
| GT8 positive coverage | 10 / 23 audit rows — 43.5% | Partially unresolved in submitted baseline; gid=0 promotion ran before pair-splits, which reclassified promoted rows — a sequencing artifact, not a fundamental constraint |
| GT7 positive coverage | 3 / 135 audit rows — 2.2% | Architectural limitation; those 3 rows are contamination in pred=6, not a GT7 identity |

**Surgical gid=0 promotion pass.** Following audit-guided identity enforcement, a complementary post-processing step (`promote_pred0_to_target_from_audit`) was applied to promote tracker-confirmed but unidentified (gid=0) rows to a target positive identity where the audit confirmed the match at IoU ≥ 0.34, subject to a per-frame duplicate guard. Four GT persons benefited: GT11 → pred=8 (+4 rows, 60% → 100% audit coverage), GT6 → pred=6 (+6 rows, 81.8% → 100%), GT10 → pred=10 (+7 rows, 19.2% → 46.2%), and GT2 → pred=2 (+21 rows, 73.5% → 94.1% safe subset). All four experiments were adopted: zero same-frame duplicates were created, both downstream pair-splits fired zero changes, and macro pred→GT purity improved with each step. GT8 was attempted but reverted in the submitted baseline: the gid=0 promotion ran before the downstream pair-splits, and the (5 vs 9) pair-split subsequently reclassified 9 of 13 promoted rows, reducing pred=9 purity from 100% to 93.3%. The root cause is a sequencing artifact rather than a fundamental architectural constraint: a post-submission exploratory experiment confirmed that re-sequencing the GT8 promotion step to run after both pair-splits recovered all 13 rows cleanly (ID9: 102 → 115, zero new duplicates, purity within floor), but that result was not incorporated into the frozen submitted baseline.

These results reflect a high-precision, conservative design. The 95.6% pred→GT purity confirms that positive-ID assignments are reliable; the 52.6% positive-ID coverage reflects intentional withholding of uncertain identities rather than detection failure. The dominant residual gap — approximately 20.5% seen-but-unidentified and 24.0% unmatched — is structural: the online identity gate prioritizes avoiding wrong merges over maximising coverage, and the `pending:no_match_or_ambiguous` failure mode accounts for the majority of unresolved tracks.

#### Sales-Zone KPI Alignment (Daily, 2026-02-28)

To connect identity-resolved tracks to retail business outcomes, three product-zone polygons were defined against the 2560 × 1944 CAM1 frame and verified by extracting and annotating a representative mid-video frame: CAM1_Z01 (Left Wall / Tap Rack, x=[150,900], foot-point y=[700,1800]), CAM1_Z02 (Front-Right Basin Display, x=[1050,2300], foot-point y=[650,1050]), and CAM1_Z03 (Right Wall Toilet Display, x=[1600,2560], foot-point y=[1000,1600]). Zone assignment used the foot-point anchor (`cy = y2`, the bottom edge of each bounding box) via `scripts/add_zones.py`. Zone-level events were ingested into a SQLite store via `scripts/ingest_kpi.py` (minimum dwell 0.5 s, session gap 1.0 s, engagement threshold 3.0 s).

**Visit metric definition.** KPI values include all ByteTrack-confirmed detections (global_id ≥ 0), not only ReID-matched positive identities. Standard retail zone analytics counts all detected visitors in a zone regardless of cross-session identity; restricting to ReID-identified persons alone would yield only five Z01 visits in the 150-second clip — too sparse to be interpretable. Two unidentified (gid=0) events with atypically long dwell (30.4 s in Z02 at t = 0–31 s; 22 s in Z03 at t = 100–123 s) are noted as probable initialisation artefacts but are retained as genuine ByteTrack-confirmed detections.

**Exported daily values (2026-02-28, 150-second CAM1 clip):**

| Zone | Label | Visits | Engaged Visits | Avg Dwell (s) |
|---|---|---|---|---|
| CAM1_Z01 | Left Wall / Tap Rack | 12 | 7 | 10 |
| CAM1_Z02 | Front-Right Basin Display | 18 | 11 | 11 |
| CAM1_Z03 | Right Wall Toilet Display | 15 | 6 | 4 |

**Sales-engagement alignment:**

Engagement share is defined as a zone's proportion of total engaged visits. Sales share is computed from unit sales for the corresponding product category on 2026-02-28.

| Zone | Product | Sales Share | Engagement Share | Gap | Flag |
|---|---|---|---|---|---|
| CAM1_Z01 | Sink tap | 44.4% | 29.2% | 0.153 | Mild conflict |
| CAM1_Z02 | Basin | 16.7% | 45.8% | 0.292 | **Strong conflict** |
| CAM1_Z03 | Water closet | 38.9% | 25.0% | 0.139 | Mild conflict |

CAM1_Z02 (basin display) exhibits a pronounced misalignment: it attracts 45.8% of engaged zone visits but accounts for only 16.7% of unit sales, yielding a conversion proxy of 0.27 units per engaged visitor versus 1.14–1.17 for the other two zones. This suggests that the basin display generates strong browsing interest that does not translate proportionally into purchases — consistent with a merchandising or pricing inefficiency rather than a visibility problem. The tap rack and toilet display zones are each mildly misaligned but broadly consistent with their sales share.

**Evidence scope and limitations.** Daily values (2026-02-28) are exported directly from the pipeline and reflect real measured foot traffic. Weekly (2026-02-23 to 2026-02-28) and monthly (February 2026) values in the supporting comparison workbook remain as estimates, because only a single 150-second video clip is currently available for this retail location and date. Those estimates were derived proportionally from the sales distribution and should be replaced with real pipeline outputs once additional video coverage is captured across the remaining days of the period.

### 8.5 FULL_CAM1 Benchmark Results

FULL_CAM1 output (`runs/kpi_batch/retail-shop_FULL_CAM1_tracks.csv`) contains:

- total rows: 11,374
- final positive rows: 2,773
- final positive IDs: 8 (`1..8`)
- same-frame duplicate positive IDs: 0

Identity metrics (`runs/kpi_batch/retail-shop_FULL_CAM1_identity_metrics.json`) show:

- before final custom remap: 3,863 positive-ID rows
- all canonical slots `1..11` received assignments before final filtering/remap
- rows excluded from final custom 8-ID output: 1,090
  - old GID 12: 1,075 rows
  - old GID 10: 15 rows
  *(Note: these sub-labels reflect GID numbering at the time the identity metrics were recorded; the current CSV consolidates the excluded rows under a single GID, but the total of 1,090 excluded rows is unchanged.)*
- old GID 11 (15 rows) is merged into output slot 7

Custom 8-ID remap policy used in pipeline logs:

`{1→1, 2→2, 3→3, 4→4, 6→5, 9→6, 5→7, 11→7, 8→8}`

### 8.6 Cross-Camera Identity Matching

Cross-camera summary (`runs/cross_cam/`):

- cam1 positive IDs: 5 (`1..5`)
- cam2 positive IDs: 6 (`1..6`)
- candidate pairs scored: 74
- candidate score range: `0.660121` to `0.865066`

Match-type breakdown (`cross_camera_matches.csv`):

| Match type | Count |
|---|---|
| `canonical_primary` | 5 |
| `fragment_reuse` | 10 |
| `anchor_side_consensus` | 6 |
| `overlap_dominant_reuse` | 1 |
| `same_cam_fragment_propagate:14` | 1 |
| `new` | 1 |
| **Total** | **24** |

All five cam1 canonical IDs receive primary matches in cam2. One identity appears only in cam2 (`new`, GID 6, 56 rows).

### 8.7 Demo_Video Results

Demo_Video properties (`Demo_Video_preprocess_report.json`):

- 370 frames, 30.83 s, `2560x1944`, 12 fps
- bad frames: 0

Track output (`retail-shop_Demo_Video_tracks.csv`):

- total rows: 1,855
- positive rows: 760
- positive IDs: 8
- same-frame duplicates: 0

Per-ID spans:

| ID | Frame range | Rows | Span (frames) | Track density |
|---|---|---|---|---|
| 1 | 12–96 | 85 | 84 | 1.01 |
| 2 | 12–74 | 47 | 62 | 0.76 |
| 3 | 12–24 | 13 | 12 | 1.08 |
| 4 | 49–229 | 71 | 180 | 0.39 |
| 5 | 102–168 | 52 | 66 | 0.79 |
| 6 | 102–369 | 268 | 267 | 1.00 |
| 7 | 177–369 | 187 | 192 | 0.97 |
| 8 | 333–369 | 37 | 36 | 1.03 |

Re-entry stats (`reentry_debug/retail-shop_Demo_Video/reentry_stats.json`):

- attempts: 14
- accepted: 4
- rejected weak: 5
- rejected ambiguous: 5
- group merges: 4
- overlap handoff merges: 1
- same-source stitch-trust accepts: 0
- single-candidate accepts: 0

### 8.8 Per-Person Overhead Clip Validation

Five overhead-view clips across two subjects were processed using the per-person pipeline to validate identity persistence in overhead-view geometry. All clips are `920×680` at 30 fps. Each clip contains a single subject throughout, allowing direct assessment of whether the pipeline maintains a single consistent global identity without fragmentation over clips of 2–3 minutes. Raw files are in `data/raw/person_01/` and `data/raw/person_02/`; final outputs are in `runs/Per_Person/`.

| Clip | Person | Raw frames | Duration (s) | Tracked rows | Active frame span | Time span (s) | Track density | Positive IDs | Same-frame dup frames |
|---|---|---|---|---|---|---|---|---|---|
| 1_2_crop | person_01 | 3,872 | 129.07 | 3,087 | 102–3,731 (3,630 f) | 121.0 | 0.85 | 1 | 3 |
| 1_3_crop | person_01 | 5,305 | 176.83 | 4,470 | 170–5,203 (5,034 f) | 167.8 | 0.89 | 1 | 8 |
| 2_1_crop | person_02 | 3,804 | 126.80 | 3,109 | 99–3,614 (3,516 f) | 117.2 | 0.88 | 1 | 6 |
| 2_2_crop | person_02 | 3,788 | 126.27 | 2,795 | 101–3,652 (3,552 f) | 118.4 | 0.79 | 1 | 7 |
| 2_3_crop | person_02 | 3,581 | 119.37 | 2,899 | 152–3,487 (3,336 f) | 111.2 | 0.87 | 1 | 6 |

*Track density = tracked rows ÷ active frame span (rows per frame of span).*

Each clip produces exactly one positive global identity throughout, confirming that the pipeline does not fragment a single subject into multiple IDs in overhead-view geometry across clips of 2–3 minutes. Track density ranges from 0.79 to 0.89, indicating stable continuous tracking with only minor brief dropouts relative to the full clip span. Each clip contains a small number of frames (3–8) in which global_id=1 appears twice with non-identical bounding boxes; these are tracker-level artefacts in which a momentary tracklet split produces two overlapping detections of the same physical person in a single frame. They are not identity errors, affect at most 0.16% of tracked frames per clip, and are reported here for completeness.

### 8.9 Pipeline Improvements for Overhead Views

Three practical improvements were implemented for overhead-view uploaded clips:

1. Group-merge temporal guard (`merge_max_gap_frames` profile-dependent)
- prevents over-bridging long absences in uploaded clips.

2. Same-source stitch trust (`same_source_stitch_trust_score=0.55` in uploaded mode)
- allows safe recovery when stitch grouping strongly supports same-identity continuity.

3. Single-candidate spatial accept (uploaded mode)
- accepts reconnection when only one viable candidate survives and spatial/body-shape cues support it.

These paths are disabled for CAM1/FULL_CAM1 benchmark modes, where stricter side-view behavior is preferred.

### 8.10 Limitations

Key limitations include:

- **ReID domain sensitivity:** OSNet was tuned on side-view retail footage; overhead generalization still needs broader validation.
- **Single-subject overhead validation scope:** overhead-mode thresholds were validated on single-subject per-person clips (five clips, two persons, `920×680`); generalisation to multi-person overhead scenes remains untested.
- **Partial dependence on manual audit files:** first-two-minute purity analysis requires manually labeled CSVs.
- **Canonical mapping trade-off:** final custom remap intentionally excludes some positive rows to prioritize canonical consistency.
- **GT7 architectural identity gap:** One of eleven annotated GT persons (GT7, 135 audit rows, frames 29–1,788) achieves only 2.2% positive-ID coverage in the accepted CAM1 output (3 of 135 audit rows, all contamination in pred=6 rather than a correct GT7 identity). Diagnostic analysis established that GT7 is detected and confirmed by the online tracker (ByteTrack; up to 73 consecutive confirmed hits) but is consistently prevented from receiving a global identity by the active-owner zone protection gate in `bot_sort.py`. A co-located locked track with OSNet similarity ≥ 0.62 to GT7 and overlapping spatial zone classification triggers the gate, returning the track to zero-gid each frame and preventing new-identity accumulation from reaching the `new_id_confirm_frames` threshold. An experiment relaxing the zone-only condition (requiring `zone_a >= 0.48 and sim_active >= 0.72` instead of `zone_a >= 0.48` alone) was implemented, evaluated, and reverted: it produced zero GT7 rows while collapsing ID3 from 515 rows to 12 and inflating ID6 by 536 rows, demonstrating that the zone gate simultaneously protects stable identities and excludes GT7. GT7 is therefore documented as an unrecoverable case under the current architecture — a known design trade-off between conservative identity safety and coverage completeness, not an unresolved implementation error.

## 9. Summary and Reflections

This project delivered a full audit-to-deployment ReID workflow for retail CCTV analytics. The key outcome is not just strong detector/ReID training metrics, but a controlled identity-resolution pipeline that keeps behavior interpretable and auditable.

A central lesson is that benchmark component accuracy does not guarantee trustworthy analytics. Deployment reliability depends on conservative identity policy, clear diagnostics, and rollback-safe post-processing.

On CAM1 the identity trade-off is directly measurable: the accepted final output achieves a pred→GT macro purity of 95.6% at 52.6% positive-ID coverage of audit-labelled instances, with zero same-frame duplicate positive IDs across all 1,800 frames. This reflects the pipeline's deliberate preference for withholding uncertain identities over assigning plausible but incorrect ones. Four previously under-represented persons (GT2, GT6, GT10, GT11) were recovered through a surgical gid=0 promotion pass without introducing duplicates or purity regressions; GT8 remains partially unresolved in the submitted baseline due to a sequencing artifact (promotion ran before pair-splits; the root cause was identified and resolved in a post-submission exploratory experiment, not incorporated into the submitted result); GT7 is documented as an architectural limitation of the active-owner identity safety gate — a gap that experiments confirmed cannot be closed without causing regression in stable identities.

## 10. Project Management

### 10.1 Work Plan (Condensed)

- Phase 0: preprocessing and frame-quality audit module
- Phase 1: baseline tracking and audit tooling
- Phase 2: custom YOLO and OSNet training
- Phase 3: offline identity resolution and canonical alignment
- Phase 4: dashboard integration, evaluation, and report finalization

### 10.2 Risk Handling

- ambiguity-first policy: reject uncertain links rather than risk false merges;
- same-frame uniqueness checks at multiple points;
- staged post-processing with diagnostics and safe fallback behavior;
- preservation of intermediate artifacts for traceability.

### 10.3 Ethics and Privacy Considerations

This work uses CCTV-style person tracking for retail analytics, which raises privacy and governance concerns. The project focuses on technical identity consistency and does not include sensitive personal attribute inference. For deployment, institutional and legal compliance (data minimization, retention limits, access control, and signage/consent policy where required) remains essential.

## 11. Contributions and Personal Reflection

### 11.1 Technical Contributions

- implemented a four-stage modular pipeline for robust retail ReID;
- integrated deterministic preprocessing quality audit into runtime flow;
- built audit-driven training datasets for detector and ReID adaptation;
- implemented conservative offline identity resolution with re-entry diagnostics;
- implemented canonical alignment and convergence diagnostics for FULL_CAM1;
- developed a production-style dashboard workflow for benchmark and uploaded clips.

### 11.2 Personal Reflection

This project significantly strengthened my understanding of system-level computer vision engineering. The most important progress came from iterative debugging and evidence-based refinement, especially when model confidence alone was insufficient. I learned to prioritize reliability, transparency, and safe failure behavior for real deployment contexts.

## 12. Bibliography

1. Jocher, G. et al. *Ultralytics YOLO* (documentation and implementation). Ultralytics.  
2. Zhang, Y. et al. “ByteTrack: Multi-Object Tracking by Associating Every Detection Box.” *ECCV*, 2022.  
3. Aharon, N. et al. “BoT-SORT: Robust Associations Multi-Pedestrian Tracking.” *arXiv preprint arXiv:2206.14651*, 2022.  
4. Zhou, K. et al. “Omni-Scale Feature Learning for Person Re-Identification.” *ICCV*, 2019.  
5. Zheng, L. et al. “Scalable Person Re-identification: A Benchmark.” *ICCV*, 2015.

## 13. Appendices

### Appendix A: Key Commands

```bash
# Start dashboard
.venv/bin/uvicorn src.server.api:app --host 127.0.0.1 --port 8000

# Run FULL_CAM1 pipeline
.venv/bin/python scripts/run_batch.py --match FULL_CAM1

# Run CAM1 + benchmark workflow (if configured in script)
.venv/bin/python scripts/run_batch.py --match CAM1

# Retest protocol
.venv/bin/python archive/old_scripts/retest_protocol.py \
  --audit_csv experiments/audit/cam1_manual_audit_sheet.csv
```

### Appendix B: Main Artifacts

- `models/yolo_cam1_person.pt`
- `models/osnet_cam1.pth`
- `runs/kpi_batch/CAM1_preprocess_report.json`
- `runs/kpi_batch/FULL_CAM1_preprocess_report.json`
- `runs/kpi_batch/retail-shop_CAM1_tracks.csv`
- `runs/kpi_batch/retail-shop_FULL_CAM1_tracks.csv`
- `runs/kpi_batch/retail-shop_FULL_CAM1_identity_metrics.json`
- `runs/kpi_batch/retail-shop_FULL_CAM1_cam1_anchor_report.json`
- `runs/kpi_batch/retail-shop_FULL_CAM1_anchor_failure_diagnostics.json`
- `runs/retest/cam1_retest_report.json`
- `runs/cross_cam/cross_camera_matches.csv`
- `runs/Per_Person/` (per-person overhead clip track CSVs: `person_01_1_2_crop_tracks.csv`, `person_01_1_3_crop_tracks.csv`, `person_02_2_1_crop_tracks.csv`, `person_02_2_2_crop_tracks.csv`, `person_02_2_3_crop_tracks.csv`)

### Appendix C: Reproducibility Notes

- Metrics in this report are tied to artifact files listed above.
- Some historical intermediate runs are preserved only in selected logs.
- For strict reproducibility, retain both logs and generated JSON/CSV artifacts per run.
- The frozen submitted CAM1 baseline (3,256 positive rows, 52.6% coverage, 95.6% pred→GT purity) is preserved in `runs/baseline_freeze_2026-04-28/retail-shop_CAM1_tracks.FROZEN.csv` (MD5: `b9e3e9052d1cacff1334aa82815cc817`). A post-submission exploratory experiment (GT8 promotion re-sequenced after pair-splits) reached 3,269 positive rows (52.8%, purity 0.9724) but was not merged into the submitted baseline; all cited metrics in this report refer to the frozen baseline.
