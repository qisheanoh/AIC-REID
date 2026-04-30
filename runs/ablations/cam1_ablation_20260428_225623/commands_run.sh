#!/usr/bin/env bash
set -euo pipefail

/Users/ohqishean/AIC-REID/.venv/bin/python scripts/run_batch.py --match =CAM1 --no-render --out_dir /Users/ohqishean/AIC-REID/runs/ablations/cam1_ablation_20260428_225623/no_offline_reentry_linking --disable_reentry_linking
/Users/ohqishean/AIC-REID/.venv/bin/python scripts/run_batch.py --match =CAM1 --no-render --out_dir /Users/ohqishean/AIC-REID/runs/ablations/cam1_ablation_20260428_225623/no_tracklet_stitching --disable_tracklet_stitching
/Users/ohqishean/AIC-REID/.venv/bin/python scripts/run_batch.py --match =CAM1 --no-render --out_dir /Users/ohqishean/AIC-REID/runs/ablations/cam1_ablation_20260428_225623/osnet_only_no_attire_shape --osnet_only
/Users/ohqishean/AIC-REID/.venv/bin/python scripts/run_batch.py --match =CAM1 --no-render --out_dir /Users/ohqishean/AIC-REID/runs/ablations/cam1_ablation_20260428_225623/relaxed_conservative_gate --relaxed_identity_gate
FULL_CAM1_CAM1_ANCHOR=off FULL_CAM1_FORCE_CUSTOM8_IDS=0 /Users/ohqishean/AIC-REID/.venv/bin/python scripts/run_batch.py --match =FULL_CAM1 --no-render --out_dir /Users/ohqishean/AIC-REID/runs/ablations/cam1_ablation_20260428_225623/no_cam1_anchor_alignment
