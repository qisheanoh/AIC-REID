# AIC-REID

Audit-Driven Identity-Consistent Person Re-Identification pipeline for retail CCTV analytics.

This project combines:
- YOLO person detection
- ByteTrack/BOT-SORT-style online tracking
- ReID feature extraction (OSNet / `torchreid`)
- Offline tracklet stitching and re-entry linking
- Zone-aware KPI analytics
- FastAPI dashboard for running jobs and inspecting results

## Repository Layout

- `src/`: core pipeline, trackers, ReID, analytics, server
- `scripts/`: runnable entry points for batch runs, cross-camera runs, diagnostics, and evaluation
- `configs/`: camera, app, and zone YAML configurations
- `data/raw/`: input videos
- `models/`: detector/ReID weights (ignored in git)
- `runs/`: generated outputs (tracks, reports, diagnostics, visualizations)
- `experiments/`: audit sheets, dense-MOT protocol files, and evaluation assets
- `docs/`: project/report and demo documentation

## Clone and Run with Git LFS (Required)

Use this setup in a fresh clone:

```bash
git lfs install
git clone https://github.com/qisheanoh/AIC-REID.git
cd AIC-REID
git lfs pull
```

Then run:

```bash
uvicorn src.server.api:app --host 127.0.0.1 --port 8000
```

## Environment Setup

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Recommended) Use custom model weights if available:
- detector: `models/yolo_cam1_person.pt`
- reid: `models/osnet_cam1.pth`

If custom weights are missing, defaults are used where possible (for example `yolov8m.pt` from Ultralytics).

## Quick Start

### 1) Run Main Batch Pipeline (single-camera retail workflow)

```bash
python scripts/run_batch.py --match CAM1
```

Useful options:
- `--no-render` to skip output video rendering
- `--disable_reentry_linking` to disable offline re-entry linking
- `--disable_tracklet_stitching` to disable offline tracklet stitching
- `--cam1-recovery` to enable conservative CAM1 continuity recovery profile

### 2) Run Online Tracking Only

```bash
python scripts/run_online_tracking.py \
  --video data/raw/retail-shop/CAM1.mp4 \
  --out_csv runs/kpi_batch/retail-shop_CAM1_tracks.csv
```

### 3) Run Cross-Camera Linking

```bash
python scripts/run_cross_cam.py \
  --video1 data/raw/cross_cam/cross_cam1.mp4 \
  --video2 data/raw/cross_cam/cross_cam2.mp4 \
  --out_dir runs/cross_cam
```

### 4) Start FastAPI Dashboard

```bash
uvicorn src.server.api:app --host 127.0.0.1 --port 8000
```

Open:
- `http://127.0.0.1:8000/ui`

## Dense MOT Evaluation (CAM1)

Prepare dense subset assets:

```bash
python scripts/run_dense_mot_evaluation.py --prepare
```

Validate dense GT:

```bash
python scripts/validate_dense_mot_gt.py --gt_csv experiments/dense_mot_cam1/dense_gt.csv
```

Run evaluation:

```bash
python scripts/run_dense_mot_evaluation.py --evaluate
```

Outputs:
- `runs/dense_mot_cam1/dense_mot_summary.csv`
- `runs/dense_mot_cam1/dense_mot_summary.json`
- `runs/dense_mot_cam1/dense_mot_report.md`

## KPI/Event Ingestion

Ingest generated track CSVs into SQLite and refresh event analytics:

```bash
python scripts/ingest_kpi.py --config configs/cameras/kpi_retailshop_cam1.yaml
```

## Notes

- Large binary assets in this repository are tracked with Git LFS. Always run `git lfs pull` after cloning.
- Avoid downloading source as a ZIP for this project; ZIP archives do not fetch real Git LFS file contents.
- Model files under `models/` are still ignored and should be provided locally when needed.
- Several files in `runs/` and `experiments/` are committed as reproducibility artifacts from prior experiments.
