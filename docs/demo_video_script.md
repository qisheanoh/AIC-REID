# Demo Video Script (1 to 3 Minutes)

## Mandatory Format (from your guideline files)

- Duration: **1 to 3 minutes**
- Resolution: **1280x720**
- Codec/container: **H264 in MP4**
- First frame must show:
  - **Student Name + Student ID**
  - **Project Title**
- Focus on **project outcome**, not theory-heavy slides.

## Recommended Final Length

- **2 minutes 20 seconds** (safe inside the required window).

## Recording Assets in This Project

- Main tracked output: `runs/kpi_batch/retail-shop_FULL_CAM1_vis.mp4`
- Main track CSV: `runs/kpi_batch/retail-shop_FULL_CAM1_tracks.csv`
- Preprocessing report: `runs/kpi_batch/FULL_CAM1_preprocess_report.json`
- Supporting metrics:
  - `runs/kpi_batch/retail-shop_FULL_CAM1_identity_metrics.json`
  - `runs/retest/cam1_retest_report.json`
  - `archive/old_runs/train/yolo_cam1_person/results.csv`
- Dashboard (start before recording): `.venv/bin/uvicorn src.server.api:app --host 127.0.0.1 --port 8000`

## Storyboard and Voiceover

### 0:00 to 0:10 - Title Frame

On screen:

- Project title
- Your name and student ID
- "Final Year Project Demo"

Voiceover:

"I am OH QI SHEAN, 20512381, and this is my Final Year Project on robust person re-identification for retail CCTV analytics."

### 0:10 to 0:30 - Problem and Objective

On screen:

- 2 to 3 bullet points:
  - identity switches in crowded retail scenes
  - same person gets different IDs after re-entry
  - need stable analytics-ready identity tracking

Voiceover:

"The main challenge is keeping the same identity through occlusion and re-entry. My objective is to reduce identity switches and improve ID consistency in real retail footage."

### 0:30 to 1:00 - System Pipeline Overview

On screen:

- four-stage architecture slide:
  - Stage 1: Video Ingestion & Preprocessing — frame quality audit
  - Stage 2: Detection & Tracking — YOLO + BOTSORT + OSNet
  - Stage 3: Identity Resolution — stitch, re-entry, anchor alignment
  - Stage 4: Behavioural Analytics & Reporting — KPI dashboard

Voiceover:

"The system is organised as a four-stage pipeline. Stage 1 audits every input frame for quality before detection begins. Stage 2 runs YOLO detection and BOTSORT tracking with OSNet appearance features. Stage 3 applies offline identity resolution — stitching, re-entry linking, and CAM1 anchor alignment. Stage 4 produces the KPI analytics and reporting."

### 1:00 to 1:45 - Live Result Demonstration

On screen:

- show the web dashboard at `/ui`:
  - select camera `cam1`, select clip `retail-shop_CAM1`
  - click Check Quality — "Stage 1 · Preprocessing" banner shows: 1800 frames, 0 bad (0%) · green
  - KPI tables showing per-zone and per-person analytics
- switch to browser Tab 2 (`/meta/preprocess-report?clip_id=retail-shop_FULL_CAM1`) — show the FULL_CAM1 JSON: 3616 frames, 144 bad (3.98%) · washed out
- play `retail-shop_FULL_CAM1_vis.mp4`
- zoom/crop around difficult scenes (occlusion and re-entry)
- overlay callouts:
  - "ID reused after re-entry"
  - "same-frame duplicate IDs prevented"
  - "bad-frame handling enabled"

Voiceover:

"Here is the system in action. The CAM1 baseline clip is fully clean — zero bad frames. The longer FULL_CAM1 sequence had 144 washed-out frames flagged before tracking begins. The dashboard shows per-zone visit counts, dwell times, and qualification rates computed from the resolved tracks."

### 1:45 to 2:10 - Quantitative Highlights

On screen (simple table):

- YOLO: mAP50 `0.95195`, mAP50-95 `0.79044`
- OSNet: mAP `83.5%`, Rank-1 `100.0%`
- FULL_CAM1 identity diagnostics: canonical-slot mapping logged in artifact, duplicate positive IDs per frame: `0`

Voiceover:

"Quantitatively, the detector and ReID model both improved after domain adaptation. Archived training logs report YOLO and OSNet improvements, while FULL_CAM1 identity diagnostics confirm zero same-frame positive-ID duplication."

### 2:10 to 2:20 - Closing

On screen:

- "Thank you"
- optional QR/link to repository zip or user guide

Voiceover:

"In summary, this project delivers a complete four-stage, audit-driven ReID pipeline with interactive analytics — from frame quality audit through to behavioural KPI reporting. Thank you."

## Quick Production Checklist

- Keep narration clear and steady.
- Prefer one clean voice track over background music.
- Use large fonts for all metrics.
- Avoid fast cuts; evaluators need to see IDs clearly.
- Export exactly as `1280x720` H264 MP4.

## Suggested Filename

- `OH QI SHEAN_20512381_Demo Video.mp4`
