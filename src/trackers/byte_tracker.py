# src/trackers/byte_tracker.py

import numpy as np
from collections import deque


class STrack:
    _count = 0

    def __init__(self, tlbr, score, feature=None):
        self.track_id = STrack._count
        STrack._count += 1

        self.tlbr = tlbr  # [x1, y1, x2, y2]
        self.score = score
        self.feature = feature
        self.is_activated = True
        self.frame_id = 0
        self.age = 0
        self.time_since_update = 0

        self.features = deque(maxlen=30)
        if feature is not None:
            self.features.append(feature)

    def update_feature(self, new_feature):
        self.feature = new_feature
        self.features.append(new_feature)

    def predict(self):
        # Optionally include motion model prediction (e.g., Kalman filter)
        pass

    def activate(self, frame_id):
        self.frame_id = frame_id
        self.is_activated = True

    def mark_lost(self):
        self.is_activated = False


class BYTETracker:
    def __init__(self, track_thresh=0.5, match_thresh=0.8):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.tracks = []
        self.frame_id = 0

    def _iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    def _feature_cosine_similarity(self, f1, f2):
        if f1 is None or f2 is None:
            return 0.0
        f1 = f1 / np.linalg.norm(f1)
        f2 = f2 / np.linalg.norm(f2)
        return np.dot(f1, f2)

    def update(self, dets, feats=None, frame_id=0):
        self.frame_id = frame_id
        updated_tracks = []
        used_dets = set()

        for track in self.tracks:
            best_match = None
            best_sim = 0.0
            for i, det in enumerate(dets):
                if i in used_dets:
                    continue
                x1, y1, x2, y2, score = det
                sim = 0
                if feats is not None and track.feature is not None:
                    sim = self._feature_cosine_similarity(track.feature, feats[i])
                else:
                    sim = self._iou(track.tlbr, det[:4])

                if sim > best_sim and sim > self.match_thresh:
                    best_sim = sim
                    best_match = (i, det)

            if best_match:
                i, det = best_match
                used_dets.add(i)
                x1, y1, x2, y2, score = det
                track.tlbr = [x1, y1, x2, y2]
                if feats is not None:
                    track.update_feature(feats[i])
                updated_tracks.append(track)
            else:
                track.mark_lost()

        for i, det in enumerate(dets):
            if i in used_dets or det[4] < self.track_thresh:
                continue
            x1, y1, x2, y2, score = det
            feature = feats[i] if feats is not None else None
            new_track = STrack([x1, y1, x2, y2], score, feature)
            new_track.activate(frame_id)
            updated_tracks.append(new_track)

        self.tracks = [t for t in updated_tracks if t.is_activated]
        return self.tracks
