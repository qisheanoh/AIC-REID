# Dense CAM1 Annotation Guide (for valid MOT metrics)

## Why dense annotation is required
Standard MOT metrics such as `MOTA`, `IDF1`, and `HOTA` assume dense frame-level ground truth for every evaluated frame: every visible person must be annotated with a bounding box and consistent identity where known.

The existing CAM1 audit sheet is sparse and KPI-oriented. It is suitable for deployment-oriented auditing, but **not** valid as dense MOT ground truth. Therefore, standard MOT metrics must only be computed from `dense_gt.csv` after dense manual labelling.

## Dense subset protocol
- Video: `data/raw/retail-shop/CAM1.mp4`
- Resolution: `2560 x 1944`
- Dense subset: `300` frames
- Sampling mode: `10 windows x 30 consecutive frames`
- Frame list source: `selected_frames.csv`

## Required CSV schema
File: `dense_gt.csv` (copy from `dense_gt_template.csv`)

Columns:
- `frame_idx`
- `person_id`
- `x1`
- `y1`
- `x2`
- `y2`
- `visibility`
- `ignore`
- `notes`

## Annotation rules
1. Label every visible person in every selected frame.
2. Use consistent `person_id` across contiguous frames and across windows when identity is visually known.
3. If identity across distant windows is uncertain, assign a new `person_id` and put `uncertain_cross_window` in `notes`.
4. Use `ignore=1` for non-person/ambiguous regions (reflection, poster, severe truncation, etc.) that should not count.
5. Visibility values:
- `1.0`: fully or mostly visible
- `0.5`: partially occluded but still identifiable
- `0.25`: heavily occluded
6. Bounding boxes use pixel coordinates in original CAM1 frame space (`2560x1944`).

## Bounding box quality
- Tight around the person silhouette/body extent.
- Avoid excessive background.
- Keep temporal consistency (avoid jittering box position/size unnecessarily).
- Ensure `x1 < x2` and `y1 < y2`.

## Identity consistency guidance
- Inside each 30-frame window, keep IDs strictly consistent.
- Across windows, reuse IDs only when identity is reasonably certain.
- If unsure, create new ID and mark `uncertain_cross_window`.
- Do not leave `person_id` empty for `ignore=0` rows.

## Valid vs invalid row examples
Valid:
```csv
frame_idx,person_id,x1,y1,x2,y2,visibility,ignore,notes
720,5,1020,410,1204,930,1.0,0,
721,5,1018,412,1201,931,0.5,0,partial_occlusion
722,,430,280,520,470,0.25,1,reflection_or_ambiguous
```

Invalid:
```csv
frame_idx,person_id,x1,y1,x2,y2,visibility,ignore,notes
720,,1020,410,1204,930,1.0,0,missing_person_id_for_ignore0
721,5,1201,412,1018,931,0.5,0,x1_greater_than_x2
722,9,-10,280,520,470,1.2,0,out_of_bounds_and_bad_visibility
```

## Validation before evaluation
Run:
```bash
python scripts/validate_dense_mot_gt.py --gt_csv experiments/dense_mot_cam1/dense_gt.csv
```

The report is written to `experiments/dense_mot_cam1/gt_validation_report.json`.
