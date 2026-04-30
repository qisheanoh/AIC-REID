# Demo Rehearsal Checklist

Target: `2:20` recorded video · `1280x720` H264 MP4

---

## 1. Pre-Demo Setup (Do This Before Opening Any Recording Tool)

### Terminal — start the server

```bash
cd "/Users/ohqishean/video-reid copy"
.venv/bin/uvicorn src.server.api:app --host 127.0.0.1 --port 8000 --log-level warning
```

Leave this terminal open and visible only if you need to show it. Keep it minimised during recording.

### Browser — open these exact URLs in separate tabs

| Tab | URL | Purpose |
|---|---|---|
| 1 | `http://127.0.0.1:8000/ui` | Main dashboard — keep on this tab during demo |
| 2 | `http://127.0.0.1:8000/meta/preprocess-report?clip_id=retail-shop_FULL_CAM1` | FULL_CAM1 bad-frame evidence (JSON view) |

**Warm up Tab 1 before recording:**
- Select camera `cam1`
- Select clip `retail-shop_CAM1`
- Let KPI tables load fully
- Click Check Quality once and wait for the green banner to appear
- Now you know it works — reset if needed for the recording

### Files — open before recording

| File | How to open | Used in slide |
|---|---|---|
| `runs/kpi_batch/retail-shop_FULL_CAM1_vis.mp4` | VLC or QuickTime | Slide 4 |
| `runs/train/yolo_cam1_person/results.png` | Preview | Slide 5 (optional) |

---

## 2. Exact UI Actions in Order (Slide 4 — Live Demonstration)

These steps must be done live or pre-recorded in sequence:

1. **Switch to browser Tab 1** (`/ui`)
2. Camera selector: choose **`cam1 - baseline entrance flow`**
3. Clip selector: choose **`retail-shop_CAM1`** ← the only clip in the selector; FULL_CAM1 is a separate artifact
4. **Wait** — KPI tables populate (12 unique IDs, entrance_zone and walkway_zone visits)
5. Click **Check Quality**
6. **Wait ~22 seconds** — banner appears: `Stage 1 · Preprocessing`
7. Banner reads: **`CAM1.mp4 · 2560×1944 · 12.00fps · 1800 frames · 150.0 s · Bad frames: 0 / 1800 (0%)`** — displayed in green
8. **Narrate:** "Stage 1 preprocessing confirms the CAM1 baseline clip is fully clean — zero bad frames."
9. **Switch to browser Tab 2** — show the FULL_CAM1 JSON report (`/meta/preprocess-report?clip_id=retail-shop_FULL_CAM1`)
10. **Narrate:** "The longer FULL_CAM1 sequence — 3,616 frames, nearly five minutes — had 144 frames, 3.98%, flagged as washed out."
11. **Switch to VLC/QuickTime** — play `retail-shop_FULL_CAM1_vis.mp4`
12. Pause at an occlusion frame. Pause at a re-entry case.
13. **Narrate** the tracked IDs and identity resolution result.

---

## 3. Exact Numbers to Have Ready

### Stage 1 — Preprocessing

| Clip | Frames | Bad | Bad % | Condition |
|---|---|---|---|---|
| `CAM1.mp4` (shown live) | 1800 | 0 | 0% | clean |
| `FULL_CAM1.mp4` (Tab 2 JSON) | 3616 | 144 | 3.98% | washed out (`white_ratio`) |
| `cross_cam1.mp4` (report only) | 720 | 8 | 1.11% | low texture (`std_gray`) |

### Stage 2 — Detector (YOLO)

- precision: `0.905` · recall: `0.917` · **mAP50: `0.952`** · mAP50-95: `0.790`

### Stage 2 — ReID (OSNet)

- **mAP: `83.5%`** · **Rank-1: `100.0%`** · Rank-10: `100.0%`

### Stage 3 — Identity Resolution (FULL_CAM1)

- stable IDs: `11` · canonical coverage: `11/11 (1.0)` · same-frame duplicate rows: `0`
- re-entry attempts: `130` · accepted: `81` · rejected ambiguous: `42`

### Stage 4 — Dashboard (CAM1, live)

- 12 unique IDs · entrance_zone: 1 visit · walkway_zone: 1 visit
- 2 persons in per-person table (IDs 2 and 5)

---

## 4. Common Failure Points and Quick Recovery

| Problem | Symptom | Fix |
|---|---|---|
| Server not started | Dashboard shows blank / connection refused | `cd` to project root, run uvicorn command from §1, reload page |
| Check Quality takes too long | Button stuck on "Scanning…" for >30s | The scan is CPU-bound (~22s for CAM1). Just wait. If truly stuck, reload page and retry. |
| KPI tables empty | All table rows say "No data" | Data freshness warning is normal. Tables populate from the DB. If empty, check server log for DB errors. |
| Check Quality shows error | Banner turns red | Likely the video file path changed. Confirm `data/raw/retail-shop/CAM1.mp4` exists. |
| Tab 2 shows `found: false` | FULL_CAM1 JSON not found | Run `POST /pipeline/preprocess` for `retail-shop_FULL_CAM1` once: `curl -s -X POST http://127.0.0.1:8000/pipeline/preprocess -H "Content-Type: application/json" -d '{"clip_id":"retail-shop_FULL_CAM1"}'` |
| Browser shows old JS | Stale cache, buttons missing | Hard reload: `Cmd+Shift+R` |
| vis.mp4 won't play | File not found in media player | Path: `runs/kpi_batch/retail-shop_FULL_CAM1_vis.mp4` (415 MB). Drag into VLC. |

---

## 5. Slide Timing Reference

| Slide | Time | Action |
|---|---|---|
| 1 — Title | 0:00–0:10 | Static text |
| 2 — Problem | 0:10–0:30 | Static bullets |
| 3 — Pipeline | 0:30–1:00 | Four-stage diagram |
| 4 — Live Demo | 1:00–1:45 | Browser → Check Quality → FULL_CAM1 JSON → vis.mp4 |
| 5 — Metrics | 1:45–2:10 | Static table with YOLO/OSNet/identity numbers |
| 6 — Closing | 2:10–2:20 | Static text |

**Slide 4 is the tightest.** It has 45 seconds for: two browser tabs + Check Quality wait (~22s) + vis.mp4 clip. Pre-warm the server and run Check Quality once before recording to cache the result. On the actual take the report is already on disk so the second run is still ~22s (it re-scans every time) — or switch camera/clip and switch back to trigger `loadPreprocessReport()` which reads from disk instantly (no rescan).

> **Fastest path for Slide 4:** select the clip → banner loads from disk in <1s (no button needed, if report already exists). Only click Check Quality if you want to show the scan happening live.
