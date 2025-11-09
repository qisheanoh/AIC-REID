import numpy as np
import torch
from ultralytics import YOLO
from src.trackers.byte_tracker import BYTETracker
from torchreid.utils import FeatureExtractor


class BOTSORT:
    def __init__(self, model_path="osnet_x1_0", device=None, reid=True, match_thresh=0.8):
        self.reid = reid
        self.match_thresh = match_thresh

        # --- tracker setup ---
        self.tracker = BYTETracker(
            track_thresh=0.5,
            match_thresh=0.45
        )

        # --- optional attributes for compatibility ---
        def _set(name, val):
            if hasattr(self.tracker, name):
                setattr(self.tracker, name, val)

        _set('track_low_thresh', 0.1)
        _set('new_track_thresh', 0.6)
        _set('track_buffer', 60)
        _set('buffer_size', 60)
        _set('max_time_lost', 60)

        # --- YOLO detector ---
        self.det_model = YOLO("yolov8s.pt")

        # --- ReID extractor ---
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.extractor = FeatureExtractor(
            model_name=model_path,
            model_path='/Users/ohqishean/.torchreid/osnet_x1_0_msmt17.pth',
            device=device
        )

        self.track_memory = {}

    def update(self, frame, frame_id=0):
        # Run YOLO detection
        results = self.det_model(frame, verbose=False)[0]
        dets = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        clss = results.boxes.cls.cpu().numpy().astype(int)

        # Keep only person class (0)
        keep = (clss == 0) & (confs > 0.3)
        dets, confs = dets[keep], confs[keep]

        feats = None
        dets_for_feats = []
        valid_confs = []

        # --- extract crops and keep alignment ---
        if self.reid and len(dets) > 0:
            crops = []
            for det, conf in zip(dets, confs):
                x1, y1, x2, y2 = map(int, det[:4])
                if x2 > x1 and y2 > y1:
                    crop = frame[y1:y2, x1:x2]
                    if crop.shape[0] > 10 and crop.shape[1] > 10:
                        crop_rgb = crop[:, :, ::-1]  # BGR -> RGB
                        crops.append(crop_rgb)
                        dets_for_feats.append(det)
                        valid_confs.append(conf)

            # --- extract and normalize features ---
            if len(crops) > 0:
                feats = self.extractor(crops)
                feats = torch.nn.functional.normalize(feats, dim=1)

        # Build aligned detection array
        if len(dets_for_feats) > 0:
            dets_full = np.concatenate(
                [np.array(dets_for_feats), np.array(valid_confs)[:, None]], axis=1
            )
        else:
            dets_full = np.empty((0, 5))
            feats = None

        # --- debug print (optional) ---
        # print(f"[DEBUG] Frame {frame_id}: dets={len(dets_full)}, feats={None if feats is None else feats.shape[0]}")

        # --- update tracker ---
        online_targets = self.tracker.update(dets_full, feats, frame_id)

        # --- collect results ---
        results_out = []
        for t in online_targets:
            if not t.is_activated:
                continue
            x1, y1, x2, y2 = t.tlbr
            results_out.append([x1, y1, x2, y2, t.track_id])

            # --- store ReID feature (optional EMA smoothing) ---
            if feats is not None and t.feature is not None:
                if t.track_id in self.track_memory:
                    prev_feat = self.track_memory[t.track_id]
                    self.track_memory[t.track_id] = 0.9 * prev_feat + 0.1 * t.feature
                else:
                    self.track_memory[t.track_id] = t.feature

        return results_out
