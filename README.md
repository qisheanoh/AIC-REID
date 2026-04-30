## CAM1 Upgrade Workflow

This repo now includes an end-to-end workflow:

1. Manual audit sheet
2. Custom YOLO training
3. Custom OSNet training
4. Re-test protocol with pass/fail report

### 1) Generate Manual Audit Sheet

```bash
.venv/bin/python scripts/generate_manual_audit_sheet.py \
  --tracks_csv runs/kpi_batch/retail-shop_CAM1_tracks.csv \
  --video_path data/raw/retail-shop/CAM1.mp4
```

Fill these files manually:

- `experiments/audit/cam1_manual_audit_sheet.csv`
- `experiments/audit/cam1_identity_map_template.csv`

Required columns for training/evaluation:

- `frame_idx`, `x1`, `y1`, `x2`, `y2`, `gt_person_id`

### 2) Train Custom YOLO (person-only)

Prepare YOLO dataset:

- `data/datasets/yolo_cam1_person/images/train`
- `data/datasets/yolo_cam1_person/images/val`
- `data/datasets/yolo_cam1_person/labels/train`
- `data/datasets/yolo_cam1_person/labels/val`

Run:

```bash
.venv/bin/python scripts/train_yolo_cam1.py \
  --data_yaml configs/training/yolo_cam1_person.yaml
```

Exports detector to:

- `models/yolo_cam1_person.pt`

### 3) Build ReID Dataset + Train Custom OSNet

Build Market1501-style dataset from audited rows:

```bash
.venv/bin/python scripts/build_reid_market1501_from_audit.py \
  --audit_csv experiments/audit/cam1_manual_audit_sheet.csv \
  --video_path data/raw/retail-shop/CAM1.mp4
```

Train OSNet:

```bash
.venv/bin/python scripts/train_osnet_cam1.py \
  --data_root data/datasets \
  --dataset_name market1501
```

Exports ReID weights to:

- `models/osnet_cam1.pth`

### 4) Re-test Protocol

Run batch + evaluate against manual audit sheet:

```bash
.venv/bin/python scripts/retest_protocol.py \
  --run_batch_first \
  --audit_csv experiments/audit/cam1_manual_audit_sheet.csv
```

Reports:

- `runs/retest/cam1_retest_report.md`
- `runs/retest/cam1_retest_report.json`

### One-command Runner

```bash
.venv/bin/python scripts/protocol_cam1_upgrade.py --step all
```
