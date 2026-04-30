# One-Take Demo Video Guide (Slide-by-Slide)

Target output: `OH QI SHEAN_20512381_Demo Video.mp4`  
Target length: `2:20` (acceptable range is 1 to 3 minutes)

## Before You Record (2 Minutes Prep)

- Set export/canvas to `1280x720`, `H264`, `.mp4`.
- Prepare 6 slides in this exact order: Title, Problem, Method, Results Video, Metrics, Closing.
- Open these files/apps before recording:
  - Dashboard at `http://127.0.0.1:8000/ui` (start server: `.venv/bin/uvicorn src.server.api:app`)
  - `runs/kpi_batch/retail-shop_FULL_CAM1_vis.mp4`
  - `runs/kpi_batch/retail-shop_FULL_CAM1_identity_metrics.json`
  - `runs/retest/cam1_retest_report.json`
  - `runs/train/yolo_cam1_person/results.csv`
- Set system volume low and microphone gain stable.

## Slide Plan and Narration (One Take)

### Slide 1 (0:00 to 0:10) - Title

On screen text:
- `Robust Person Re-Identification for Retail CCTV Analytics`
- `OH QI SHEAN (20512381)`
- `Supervisor: Dr Simon Lau`
- `Final Year Project Demo`

Say:
"I am OH QI SHEAN, 20512381. This is my Final Year Project on robust person re-identification for retail CCTV analytics."

### Slide 2 (0:10 to 0:30) - Problem and Objective

On screen text:
- `Challenge: ID switches under occlusion and re-entry`
- `Impact: unstable analytics for customer behavior`
- `Goal: stable, reusable IDs in real retail footage`

Say:
"The key problem is identity switching when people overlap or leave and re-enter the scene. My objective is to keep identities stable so the tracking output is usable for retail analytics."

### Slide 3 (0:30 to 1:00) - Pipeline

On screen text:
- `Stage 1: Video Ingestion & Preprocessing — frame quality audit`
- `Stage 2: Detection & Tracking — YOLO + BOTSORT + OSNet`
- `Stage 3: Identity Resolution — stitch, re-entry, anchor`
- `Stage 4: Behavioural Analytics & Reporting — KPI dashboard`

Say:
"The system is a four-stage pipeline. Stage 1 audits every frame for quality before detection begins. Stage 2 runs YOLO detection and BOTSORT tracking with OSNet appearance features. Stage 3 applies offline identity resolution — stitching, re-entry linking, and CAM1 anchor alignment. Stage 4 produces KPI analytics served through the web dashboard."

### Slide 4 (1:00 to 1:45) - Live Demonstration

On screen action:
- Open the web dashboard at `/ui`.
- Select camera `cam1`, select clip `retail-shop_CAM1` from the clip selector (the only clip shown).
- Click "Check Quality" — show the "Stage 1 · Preprocessing" banner: 1800 frames, 0 bad (0%) · green.
- Show KPI tables: per-zone visits, dwell, qualification rate; per-person dwell summary.
- Switch to browser Tab 2 (`/meta/preprocess-report?clip_id=retail-shop_FULL_CAM1`) — show the JSON: 3616 frames, 144 bad (3.98%) · washed out.
- Switch to `retail-shop_FULL_CAM1_vis.mp4`.
- Point to one occlusion case and one re-entry case.
- Show at least one frame with multiple stable IDs.

Say:
"Here is the system in action. The CAM1 baseline is fully clean — zero bad frames confirmed. The longer FULL_CAM1 sequence had 144 washed-out frames flagged by Stage 1 before tracking begins. The dashboard shows per-zone and per-person analytics from the resolved tracks. In the tracked video, IDs stay stable through occlusion and re-entry, with same-frame uniqueness enforced throughout."

### Slide 5 (1:45 to 2:10) - Quantitative Evidence

On screen text:
- `YOLO: mAP50 0.95195, mAP50-95 0.79044`
- `OSNet: mAP 83.5%, Rank-1 100.0%`
- `FULL_CAM1 canonical coverage: 11/11`
- `Duplicate positive-ID frames: 0`

Say:
"Quantitatively, the detector and ReID model improved after domain adaptation. On full-sequence identity checks, canonical coverage is 11 out of 11 with zero same-frame duplicate positive IDs."

### Slide 6 (2:10 to 2:20) - Closing

On screen text:
- `End-to-end audit-driven ReID workflow completed`
- `Thank you`

Say:
"In summary, this project delivers a complete four-stage, audit-driven ReID pipeline — from frame quality audit through to behavioural KPI reporting — with an interactive web dashboard. Thank you."

## One-Take Timing Safety

- If you are too fast, pause 1 second before each transition.
- If you are too slow, shorten Slide 4 narration first.
- Keep final length between `2:10` and `2:30`.

## Final Export Check

- Resolution: `1280x720`
- Format: `MP4 (H264)`
- Duration: `1 to 3 minutes`
- First frame includes: student name, student ID, and project title
