# CAM1 Dense MOT Manual Annotation Checklist

## Scope and Files
- Total selected frames: **300**
- Frame layout: **10 windows x 30 consecutive frames**
- Frame source list: `/Users/ohqishean/AIC-REID/experiments/dense_mot_cam1/selected_frames.csv`
- Frame images: `/Users/ohqishean/AIC-REID/experiments/dense_mot_cam1/frames/`
- Annotation target CSV: `/Users/ohqishean/AIC-REID/experiments/dense_mot_cam1/dense_gt.csv`

## Required Labeling Rules
- Label **every visible person** in every selected frame.
- Use `ignore=0` for valid person GT rows.
- For `ignore=0`, `person_id` must be **positive numeric** (`1,2,3,...`).
- Within each 30-frame window, keep `person_id` consistent for the same person.
- Across separate windows, reuse `person_id` only when visually confident.
- If cross-window identity is uncertain, assign a new `person_id` and write `uncertain_cross_window` in `notes`.

## Ignore Rules
- Use `ignore=1` for reflections, posters, non-persons, extreme partial bodies, or regions that should not count.
- For `ignore=1`, `person_id` may be blank.

## Coordinates and Visibility
- Coordinates must be in original CAM1 space: **2560 x 1944**.
- Format: `x1,y1,x2,y2` with `x1 < x2` and `y1 < y2`.
- Valid visibility values: `1.0`, `0.5`, `0.25`.

## Important Validator Convention for No-Person Frames
Current strict validator requires every selected frame to have at least one row in `dense_gt.csv`.
- If a selected frame has no visible person, add one sentinel ignore row:

```csv
frame_idx,person_id,x1,y1,x2,y2,visibility,ignore,notes
<frame_idx>,,0,0,1,1,0.25,1,no_visible_person
```

- Do not leave a selected frame with zero rows.
- This sentinel row is excluded from valid GT matching (`ignore=1`).

## Window-by-Window Workflow
1. Annotate one full 30-frame window.
2. Save `dense_gt.csv`.
3. Run progress update:
   - `.venv/bin/python scripts/dense_mot_annotation_progress.py`
4. Run strict validation:
   - `.venv/bin/python scripts/validate_dense_mot_gt.py --gt_csv experiments/dense_mot_cam1/dense_gt.csv --strict_complete`
5. Fix errors before moving on.

## Publication Safety Gate
- **Never compute MOT metrics until strict validation passes for the full CSV.**
