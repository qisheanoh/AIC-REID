# FULL_CAM1 Render Refresh Report

Generated: 2026-04-29T13:52:15

## Actions Performed

- Backed up previous rendered MP4 before regeneration:
  - `/Users/ohqishean/AIC-REID/runs/kpi_batch/retail-shop_FULL_CAM1_vis.OLD_BEFORE_REGEN.mp4`
- Ran requested command:
  - `.venv/bin/python scripts/run_batch.py --match FULL_CAM1 --render --out_dir /Users/ohqishean/AIC-REID/runs/kpi_batch`
- Observed a lingering run_batch process continuously rewriting `retail-shop_FULL_CAM1_tracks.csv` without refreshing MP4 in lockstep; process was terminated.
- Executed final canonical render pass from the final FULL_CAM1 CSV:
  - `from src.reid.track_linker import render_tracks_video`
  - input video: `/Users/ohqishean/AIC-REID/data/raw/retail-shop/FULL_CAM1.mp4`
  - input CSV: `/Users/ohqishean/AIC-REID/runs/kpi_batch/retail-shop_FULL_CAM1_tracks.csv`
  - output MP4: `/Users/ohqishean/AIC-REID/runs/kpi_batch/retail-shop_FULL_CAM1_vis.mp4`

## Verification

- tracks CSV mtime: `2026-04-29T13:48:40`
- rendered MP4 mtime: `2026-04-29T13:51:24`
- rendered MP4 newer than CSV: **YES**
- FULL_CAM1 final/report-ready status: **CURRENT**

- OpenCV metadata (playability fallback):
  - width: 2560
  - height: 1944
  - fps: 12.161
  - frame_count: 3616
  - duration_sec: 297.34396842364936
  - probe: opencv_fallback

## Notes

- ffprobe is unavailable in this environment; OpenCV fallback was used.
- No frozen outputs were modified.
