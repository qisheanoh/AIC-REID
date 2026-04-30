#!/usr/bin/env bash
set -e

# From repo root: ./scripts/run_demo.sh

# 1) Run stream pipeline (single video) to populate DB
echo "[DEMO] Running video pipeline..."
python -m src.pipeline.run_stream --config configs/cameras/retail_cam1.yaml

# 2) Start FastAPI server
echo "[DEMO] Starting FastAPI on http://localhost:8000/ui"
uvicorn src.server.api:app --reload
