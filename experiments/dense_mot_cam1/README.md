# Dense MOT CAM1 Protocol

## Purpose
This folder defines a **valid dense subset protocol** for CAM1 so standard MOT metrics (`MOTA`, `IDF1`, `IDSW`, etc.) are computed only where methodologically valid.

## Selected frame protocol
- Window-based subset for temporal continuity.
- 10 windows x 30 consecutive frames = 300 frames total.
- Frame list: `selected_frames.csv`
- Extracted frames: `frames/`

## Annotation status
- Template: `dense_gt_template.csv`
- Manual annotation target: `dense_gt.csv`
- Guide: `ANNOTATION_GUIDE.md`

## Why dense GT is required
Sparse audit logs do not provide dense, frame-complete tracking supervision. Therefore sparse audit cannot be used to compute standard MOT metrics reliably.

## Commands
Prepare subset and templates:
```bash
python scripts/run_dense_mot_evaluation.py --prepare
```

Validate dense GT:
```bash
python scripts/validate_dense_mot_gt.py --gt_csv experiments/dense_mot_cam1/dense_gt.csv
```

Run evaluation pipeline:
```bash
python scripts/run_dense_mot_evaluation.py --evaluate
```

## Evaluation outputs
- `runs/dense_mot_cam1/dense_mot_summary.csv`
- `runs/dense_mot_cam1/dense_mot_summary.json`
- `runs/dense_mot_cam1/dense_mot_report.md`

## Metric separation policy
Dense MOT metrics and sparse audit deployment metrics are kept in **separate sections**. They must not be mixed into one score.
