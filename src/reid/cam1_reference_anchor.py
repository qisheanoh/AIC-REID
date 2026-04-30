"""
CAM1 reference profile anchor for FULL_CAM1 canonical ID alignment.

Background
----------
`canonicalize_first_appearance` in track_linker.py renumbers FULL_CAM1 stable
gids 1..N strictly in order of first appearance. That satisfies the user's
hard rule "IDs start from 1 and fill without gaps in first-appearance order,"
but the resulting numbering does NOT automatically match the canonical ID
assignments used in the short CAM1 reference clip (where ID 1 = green-shirt
girl, ID 2 = red-and-blue-stripe-shirt man, ID 3 = beige woman sitting, etc.).

This module closes that gap. It:
  1. Extracts L2-normalized OSNet embeddings for each CAM1 canonical gid from
     the CAM1 reference video, using multiple frame samples per gid.
  2. Extracts the same per-gid OSNet profile for each stable FULL_CAM1 gid.
  3. Greedily matches FULL_CAM1 gids to CAM1 gids by cosine similarity,
     enforcing a strict accept threshold and top-1/top-2 margin.
  4. Returns a renumbering plan: each matched FULL_CAM1 gid is reassigned to
     its CAM1 canonical ID; unmatched gids are pushed to new IDs N+1, N+2, ...
     (never reused from CAM1, per the "wrong reuse is worse than
     fragmentation" rule).
  5. Optionally rewrites the FULL_CAM1 tracks CSV with the new numbering,
     backing up the original to ``*.pre-cam1-anchor.csv``.

Design principles (from user's hard rules)
------------------------------------------
- Strict thresholds (``min_cos``, ``min_margin``, plus a global fallback floor)
  are chosen to reject ambiguous matches rather than risk wrong reuse.
- Matching is bipartite-greedy: each CAM1 canonical gid can be the target of
  at most one FULL_CAM1 gid.
- If the mapping would reuse a CAM1 canonical ID for a second FULL_CAM1 gid,
  the second one is left unmatched and gets a new ID (N+1, ...).
- If the mapping as a whole removes coverage (i.e. not all CAM1 gids get
  matched), that's fine — it just means FULL_CAM1 has fewer people than
  CAM1, or the lookalike pass was too conservative to anchor some people.

This module is safe to call even if ``torchreid`` is unavailable: it raises
``ReferenceAnchorDependencyError`` early with a clean message.
"""

from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception as _e:  # pragma: no cover - depends on user env
    cv2 = None  # type: ignore
    _cv2_err = _e
else:
    _cv2_err = None

try:
    from .extractor import ReidExtractor, l2_normalize  # type: ignore
except Exception:
    try:
        from src.reid.extractor import ReidExtractor, l2_normalize  # type: ignore
    except Exception as _e:  # pragma: no cover
        ReidExtractor = None  # type: ignore
        l2_normalize = None  # type: ignore
        _reid_err = _e
    else:
        _reid_err = None
else:
    _reid_err = None


class ReferenceAnchorDependencyError(RuntimeError):
    """Raised when required deps (cv2, torchreid) are missing."""


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class _Sample:
    frame_idx: int
    x1: float
    y1: float
    x2: float
    y2: float
    overlap_n: float = 0.0
    crop_quality: Optional[float] = None
    blur_var: Optional[float] = None
    det_conf: Optional[float] = None
    occluded: float = 0.0


@dataclass
class ProfileSet:
    """Per-gid OSNet profile from one video.

    Optional color features (hue-saturation histograms, dominant-color vectors)
    are carried alongside the OSNet embeddings and used as secondary
    discriminators when OSNet alone is ambiguous.
    """
    source: str  # "cam1" or "full_cam1"
    per_gid_mean: Dict[int, np.ndarray] = field(default_factory=dict)
    per_gid_samples: Dict[int, List[np.ndarray]] = field(default_factory=dict)
    per_gid_rows: Dict[int, int] = field(default_factory=dict)
    per_gid_span: Dict[int, int] = field(default_factory=dict)

    # Color features (per gid, L2-normalized per region, mean over samples).
    per_gid_upper_hue_hist: Dict[int, np.ndarray] = field(default_factory=dict)
    per_gid_lower_hue_hist: Dict[int, np.ndarray] = field(default_factory=dict)
    per_gid_upper_dom: Dict[int, np.ndarray] = field(default_factory=dict)
    per_gid_lower_dom: Dict[int, np.ndarray] = field(default_factory=dict)
    # Lightweight body-shape descriptor from bbox geometry.
    per_gid_shape: Dict[int, np.ndarray] = field(default_factory=dict)
    # Prototype quality diagnostics.
    per_gid_proto_sample_count: Dict[int, int] = field(default_factory=dict)
    per_gid_proto_kept_count: Dict[int, int] = field(default_factory=dict)
    per_gid_proto_quality_mean: Dict[int, float] = field(default_factory=dict)
    # Mean cos(sample, prototype) over surviving samples — a direct signal of
    # prototype contamination. Low values (<0.78) mean the kept samples are
    # dispersed and the centroid is unreliable; this is a strong indicator
    # that CAM1 anchor will fail to separate this gid from similar people.
    per_gid_proto_coherence_mean: Dict[int, float] = field(default_factory=dict)


@dataclass
class MatchReport:
    """Result of aligning FULL_CAM1 stable gids to CAM1 canonical gids."""
    mapping: Dict[int, int] = field(default_factory=dict)  # old_gid -> new_canonical_id
    rejected: Dict[int, str] = field(default_factory=dict)  # old_gid -> reason
    scores: Dict[int, Dict[int, float]] = field(default_factory=dict)  # full_gid -> {cam1_gid: cos}
    fallback_assignments: Dict[int, int] = field(default_factory=dict)  # old_gid -> new_fallback_id (>= fallback_start)
    cam1_reference_gids: List[int] = field(default_factory=list)
    full_cam1_stable_gids: List[int] = field(default_factory=list)
    params: Dict[str, float] = field(default_factory=dict)
    color_scores: Dict[int, Dict[int, float]] = field(default_factory=dict)
    shape_scores: Dict[int, Dict[int, float]] = field(default_factory=dict)
    source_meta: Dict[int, Dict[str, float]] = field(default_factory=dict)
    # Conservative anchor controls may intentionally keep IDs unresolved
    # instead of forcing fallback reassignment.
    unresolved: Dict[int, str] = field(default_factory=dict)  # old_gid -> reason
    # Per-target decision metrics/flags to make remap behavior explainable.
    decision_details: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    def as_json(self) -> str:
        def _conv(v):
            if isinstance(v, dict):
                return {str(k): _conv(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [_conv(x) for x in v]
            if isinstance(v, (np.floating, np.integer)):
                return v.item()
            return v

        out = {
            "mapping": _conv(self.mapping),
            "rejected": _conv(self.rejected),
            "scores": _conv(self.scores),
            "color_scores": _conv(self.color_scores),
            "shape_scores": _conv(self.shape_scores),
            "fallback_assignments": _conv(self.fallback_assignments),
            "unresolved": _conv(self.unresolved),
            "decision_details": _conv(self.decision_details),
            "cam1_reference_gids": _conv(self.cam1_reference_gids),
            "full_cam1_stable_gids": _conv(self.full_cam1_stable_gids),
            "source_meta": _conv(self.source_meta),
            "params": _conv(self.params),
        }
        rr = getattr(self, "rescue_reasons", None)
        if rr:
            out["rescue_reasons"] = _conv(rr)
        return json.dumps(out, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Sample selection
# ---------------------------------------------------------------------------


def _load_tracks_csv(
    tracks_csv: Path,
) -> Tuple[List[str], Dict[int, List[_Sample]]]:
    def _first_float(raw: Mapping[str, str], keys: Sequence[str]) -> Optional[float]:
        for k in keys:
            if k not in raw:
                continue
            try:
                v = str(raw.get(k, "")).strip()
                if not v:
                    continue
                return float(v)
            except Exception:
                continue
        return None

    by_gid: Dict[int, List[_Sample]] = defaultdict(list)
    with Path(tracks_csv).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for raw in reader:
            try:
                gid = int(raw["global_id"])
                if gid <= 0:
                    continue
                overlap_n = _first_float(
                    raw,
                    (
                        "overlap_n",
                        "overlap_count",
                        "num_overlaps",
                        "n_overlap",
                        "overlap_neighbors",
                        "overlap",
                    ),
                )
                crop_quality = _first_float(
                    raw,
                    (
                        "crop_quality",
                        "reid_quality",
                        "sample_quality",
                        "quality",
                    ),
                )
                blur_var = _first_float(raw, ("blur_var", "lap_var", "laplacian_var"))
                det_conf = _first_float(raw, ("det_conf", "conf", "score"))
                occluded = _first_float(
                    raw,
                    (
                        "occluded",
                        "is_occluded",
                        "partial_occlusion",
                        "occlusion",
                    ),
                )
                by_gid[gid].append(
                    _Sample(
                        frame_idx=int(raw["frame_idx"]),
                        x1=float(raw["x1"]),
                        y1=float(raw["y1"]),
                        x2=float(raw["x2"]),
                        y2=float(raw["y2"]),
                        overlap_n=float(overlap_n or 0.0),
                        crop_quality=float(crop_quality) if crop_quality is not None else None,
                        blur_var=float(blur_var) if blur_var is not None else None,
                        det_conf=float(det_conf) if det_conf is not None else None,
                        occluded=float(occluded or 0.0),
                    )
                )
            except Exception:
                continue
    for gid in by_gid:
        by_gid[gid].sort(key=lambda s: s.frame_idx)
    return fieldnames, by_gid


def _select_stable_gids(
    by_gid: Mapping[int, Sequence[_Sample]],
    *,
    min_rows: int,
    min_span: int,
) -> List[int]:
    stable: List[int] = []
    for gid, samples in by_gid.items():
        if gid <= 0:
            continue
        if len(samples) < min_rows:
            continue
        span = samples[-1].frame_idx - samples[0].frame_idx + 1
        if span < min_span:
            continue
        stable.append(gid)
    stable.sort(key=lambda g: (by_gid[g][0].frame_idx, g))
    return stable


def _pick_spread_samples(samples: Sequence[_Sample], n: int) -> List[_Sample]:
    """Choose ``n`` samples spread evenly across the track's frame span."""
    if not samples:
        return []
    if n >= len(samples):
        return list(samples)
    # Filter out degenerate boxes first.
    clean = [s for s in samples if (s.x2 - s.x1) > 12 and (s.y2 - s.y1) > 24]
    if len(clean) < n:
        clean = list(samples)
    if n >= len(clean):
        return clean
    idx = np.linspace(0, len(clean) - 1, n).round().astype(int)
    out: List[_Sample] = []
    seen = set()
    for i in idx:
        i = int(i)
        if i in seen:
            continue
        seen.add(i)
        out.append(clean[i])
    return out


# ---------------------------------------------------------------------------
# OSNet profile extraction
# ---------------------------------------------------------------------------


def _ensure_deps() -> None:
    if cv2 is None:
        raise ReferenceAnchorDependencyError(
            f"OpenCV (cv2) import failed: {_cv2_err}. Install opencv-python."
        )
    if ReidExtractor is None:
        raise ReferenceAnchorDependencyError(
            f"torchreid import failed: {_reid_err}. Install torchreid."
        )


def _crop_rgb(frame_bgr: np.ndarray, s: _Sample) -> Optional[np.ndarray]:
    h, w = frame_bgr.shape[:2]
    x1 = max(0, min(w - 1, int(s.x1)))
    y1 = max(0, min(h - 1, int(s.y1)))
    x2 = max(0, min(w - 1, int(s.x2)))
    y2 = max(0, min(h - 1, int(s.y2)))
    if x2 <= x1 + 4 or y2 <= y1 + 8:
        return None
    crop_bgr = frame_bgr[y1:y2, x1:x2]
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    return cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)


def _crop_bgr(frame_bgr: np.ndarray, s: _Sample) -> Optional[np.ndarray]:
    """Return the BGR crop matching ``_crop_rgb`` without the colorspace swap."""
    h, w = frame_bgr.shape[:2]
    x1 = max(0, min(w - 1, int(s.x1)))
    y1 = max(0, min(h - 1, int(s.y1)))
    x2 = max(0, min(w - 1, int(s.x2)))
    y2 = max(0, min(h - 1, int(s.y2)))
    if x2 <= x1 + 4 or y2 <= y1 + 8:
        return None
    crop_bgr = frame_bgr[y1:y2, x1:x2]
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    return crop_bgr


def _hue_sat_histogram(region_bgr: Optional[np.ndarray], bins: int = 18) -> Optional[np.ndarray]:
    """18-bin hue histogram over *saturated* pixels (S>=40, 30<=V<=245).

    L2-normalized. Robust to lighting changes compared to raw RGB histograms.
    Returns None if fewer than 32 saturated pixels are found.
    """
    if region_bgr is None or region_bgr.size == 0 or cv2 is None:
        return None
    try:
        hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    except Exception:
        return None
    h_ch = hsv[..., 0].astype(np.float32)
    s_ch = hsv[..., 1].astype(np.float32)
    v_ch = hsv[..., 2].astype(np.float32)
    mask = (s_ch >= 40.0) & (v_ch >= 30.0) & (v_ch <= 245.0)
    n_valid = int(mask.sum())
    if n_valid < 32:
        return None
    h_vals = h_ch[mask]
    hist, _ = np.histogram(h_vals, bins=bins, range=(0.0, 180.0))
    hist = hist.astype(np.float32)
    n = float(np.linalg.norm(hist))
    if n <= 1e-6:
        return None
    return (hist / n).astype(np.float32)


def _dominant_color_descriptor(region_bgr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Low-dim dominant-color descriptor [H_mean,H_std,S_mean,S_std,V_mean,V_std].

    Computed over saturated pixels only. L2-normalized. Returns None if
    fewer than 32 saturated pixels are found.
    """
    if region_bgr is None or region_bgr.size == 0 or cv2 is None:
        return None
    try:
        hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    except Exception:
        return None
    h_ch = hsv[..., 0].astype(np.float32)
    s_ch = hsv[..., 1].astype(np.float32)
    v_ch = hsv[..., 2].astype(np.float32)
    mask = (s_ch >= 40.0) & (v_ch >= 30.0) & (v_ch <= 245.0)
    n_valid = int(mask.sum())
    if n_valid < 32:
        return None
    h_m = float(h_ch[mask].mean())
    s_m = float(s_ch[mask].mean())
    v_m = float(v_ch[mask].mean())
    h_s = float(h_ch[mask].std())
    s_s = float(s_ch[mask].std())
    v_s = float(v_ch[mask].std())
    vec = np.array([h_m, h_s, s_m, s_s, v_m, v_s], dtype=np.float32)
    n = float(np.linalg.norm(vec))
    if n <= 1e-6:
        return None
    return (vec / n).astype(np.float32)


def _upper_lower_regions(crop_bgr: Optional[np.ndarray]) -> Tuple[
    Optional[np.ndarray], Optional[np.ndarray]
]:
    """Partition a person crop into upper (torso) and lower (legs) regions.

    No pose dependency: we use fixed fractions of the bbox height because the
    anchor runs on crops of already-tracked persons. Upper = y in [0.10, 0.58]
    of bbox height; lower = y in [0.58, 0.98]. This is a robust fallback when
    pose keypoints are not available at anchor time.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return None, None
    h = int(crop_bgr.shape[0])
    if h < 24:
        return None, None
    y1u, y2u = int(0.10 * h), int(0.58 * h)
    y1l, y2l = int(0.58 * h), int(0.98 * h)
    y2u = max(y2u, y1u + 2)
    y2l = max(y2l, y1l + 2)
    upper = crop_bgr[y1u:y2u, :]
    lower = crop_bgr[y1l:y2l, :]
    if upper is None or upper.size == 0:
        upper = None
    if lower is None or lower.size == 0:
        lower = None
    return upper, lower


def _hue_hist_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    """Bhattacharyya-style similarity on two L2-normalized hue histograms."""
    if a is None or b is None:
        return 0.0
    # Cosine of normalized histograms (both unit-norm already).
    return float(np.clip(np.dot(a, b), 0.0, 1.0))


def _dom_color_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    """Cosine similarity between two dominant-color descriptors."""
    if a is None or b is None:
        return 0.0
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def _shape_descriptor_from_sample(
    sample: _Sample,
    frame_w: float,
    frame_h: float,
) -> Optional[np.ndarray]:
    frame_w = max(1.0, float(frame_w))
    frame_h = max(1.0, float(frame_h))
    bw = max(1.0, float(sample.x2) - float(sample.x1))
    bh = max(1.0, float(sample.y2) - float(sample.y1))
    area = bw * bh
    vec = np.array(
        [
            bw / bh,  # body aspect ratio
            bh / frame_h,  # relative height
            np.sqrt(area) / np.sqrt(frame_w * frame_h),  # relative scale
            (bw + bh) / (frame_w + frame_h),  # perimeter ratio
        ],
        dtype=np.float32,
    )
    n = float(np.linalg.norm(vec))
    if n <= 1e-6:
        return None
    return (vec / n).astype(np.float32)


def _blur_quality_score(crop_bgr: Optional[np.ndarray], fallback_blur_var: Optional[float]) -> float:
    """Convert blur signal into [0,1] quality.

    Uses Laplacian variance when available on the crop; falls back to metadata
    from tracks CSV when provided.
    """
    blur_var = None
    if crop_bgr is not None and crop_bgr.size > 0 and cv2 is not None:
        try:
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            blur_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        except Exception:
            blur_var = None
    if blur_var is None and fallback_blur_var is not None:
        blur_var = float(fallback_blur_var)
    if blur_var is None:
        return 0.50
    # Typical values in this pipeline range roughly 20..220.
    return float(np.clip((blur_var - 24.0) / 180.0, 0.0, 1.0))


def _sample_quality_score(
    *,
    sample: _Sample,
    crop_bgr: Optional[np.ndarray],
    frame_w: float,
    frame_h: float,
) -> float:
    """Estimate per-sample prototype quality in [0,1].

    Penalizes overlap-heavy, border-touching, tiny, blurred or low-confidence
    samples. This is deliberately conservative: low-quality samples are dropped
    from prototype construction to reduce cross-identity contamination.
    """
    frame_w = max(1.0, float(frame_w))
    frame_h = max(1.0, float(frame_h))
    bw = max(1.0, float(sample.x2) - float(sample.x1))
    bh = max(1.0, float(sample.y2) - float(sample.y1))
    area_ratio = float((bw * bh) / (frame_w * frame_h))
    size_score = float(np.clip((area_ratio - 0.0022) / 0.016, 0.0, 1.0))

    x1 = max(0.0, float(sample.x1))
    y1 = max(0.0, float(sample.y1))
    x2 = min(frame_w, float(sample.x2))
    y2 = min(frame_h, float(sample.y2))
    margin_px = min(x1, y1, max(0.0, frame_w - x2), max(0.0, frame_h - y2))
    border_score = float(np.clip(margin_px / (0.06 * min(frame_w, frame_h) + 1e-6), 0.0, 1.0))

    blur_score = _blur_quality_score(crop_bgr, sample.blur_var)
    conf_score = 0.58 if sample.det_conf is None else float(np.clip((float(sample.det_conf) - 0.24) / 0.56, 0.0, 1.0))
    crop_q_score = 0.58 if sample.crop_quality is None else float(np.clip(float(sample.crop_quality), 0.0, 1.0))

    overlap_pen = float(np.clip(0.24 * max(0.0, float(sample.overlap_n)), 0.0, 0.55))
    occ_pen = float(np.clip(0.28 * max(0.0, float(sample.occluded)), 0.0, 0.45))

    q = (
        0.30 * blur_score
        + 0.23 * size_score
        + 0.16 * border_score
        + 0.16 * conf_score
        + 0.15 * crop_q_score
        - overlap_pen
        - occ_pen
    )
    return float(np.clip(q, 0.0, 1.0))


def _select_clean_vectors(
    weighted_vectors: Sequence[Tuple[float, np.ndarray]],
    *,
    keep_ratio: float,
    min_keep: int,
    coherence_floor: float = 0.72,
    coherence_drop_below_median: float = 0.05,
) -> Tuple[List[np.ndarray], float, float]:
    """Pick clean vectors with quality-first + center-consistency + coherence pruning.

    Returns ``(selected_unit_vectors, quality_mean, coherence_mean)``. The third
    return value is the mean cosine of kept samples to the final prototype —
    higher means the surviving samples form a tighter cluster, which is the
    real signal of a non-contaminated prototype. ``coherence_floor`` and
    ``coherence_drop_below_median`` together implement a conservative
    second-pass pruning: samples whose cosine to the initial centroid falls
    under ``max(median - coherence_drop_below_median, coherence_floor)`` are
    dropped (down to ``min_keep``), and the prototype is rebuilt. This removes
    samples contaminated by near-identity crops or partial occlusions without
    widening any acceptance threshold.
    """
    if not weighted_vectors:
        return [], 0.0, 0.0
    items: List[Tuple[float, np.ndarray]] = [
        (float(q), np.asarray(v, dtype=np.float32)) for q, v in weighted_vectors if v is not None and np.asarray(v).size > 0
    ]
    if not items:
        return [], 0.0, 0.0
    items.sort(key=lambda x: x[0], reverse=True)
    k = int(round(len(items) * float(np.clip(keep_ratio, 0.25, 1.0))))
    k = max(int(min_keep), k)
    k = min(len(items), k)
    top = items[:k]
    arr = []
    for _, v in top:
        n = float(np.linalg.norm(v) + 1e-12)
        arr.append((v / n).astype(np.float32))
    proto = np.mean(np.stack(arr, axis=0), axis=0)
    pn = float(np.linalg.norm(proto) + 1e-12)
    proto = (proto / pn).astype(np.float32)
    rescored: List[Tuple[float, float, float, np.ndarray]] = []  # (score, q, sim, vu)
    for q, v in top:
        n = float(np.linalg.norm(v) + 1e-12)
        vu = (v / n).astype(np.float32)
        sim = float(np.clip(np.dot(vu, proto), -1.0, 1.0))
        score = 0.55 * float(q) + 0.45 * float(max(0.0, sim))
        rescored.append((score, float(q), sim, vu))
    rescored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    # Coherence pruning: drop samples whose cos-to-centroid is far from the
    # cluster. Conservatively bounded below by ``min_keep`` to preserve
    # coverage even when all samples disagree.
    sims = [s for _, _, s, _ in rescored]
    if sims:
        med = float(np.median(sims))
    else:
        med = 0.0
    cutoff = float(max(float(coherence_floor), med - float(coherence_drop_below_median)))
    kept = [(sc, q, s, vu) for (sc, q, s, vu) in rescored if s >= cutoff]
    if len(kept) < int(min_keep):
        # Never prune below min_keep; fall back to top rescored.
        kept = list(rescored[: max(int(min_keep), 1)])
    # Rebuild prototype from survivors (unit vectors) and report coherence.
    if kept:
        stack = np.stack([vu for _, _, _, vu in kept], axis=0)
        proto2 = np.mean(stack, axis=0)
        n2 = float(np.linalg.norm(proto2) + 1e-12)
        proto2 = (proto2 / n2).astype(np.float32)
        # Final coherence = mean cos(sample, proto2)
        coh_vals = [float(np.clip(np.dot(vu, proto2), -1.0, 1.0)) for _, _, _, vu in kept]
        coherence_mean = float(np.mean(coh_vals)) if coh_vals else 0.0
    else:
        coherence_mean = 0.0
    selected = [vu for _, _, _, vu in kept]
    q_mean = float(np.mean([q for _, q, _, _ in kept])) if kept else 0.0
    return selected, q_mean, coherence_mean


def _combined_color_similarity(
    a_upper_hue: Optional[np.ndarray],
    a_lower_hue: Optional[np.ndarray],
    a_upper_dom: Optional[np.ndarray],
    a_lower_dom: Optional[np.ndarray],
    b_upper_hue: Optional[np.ndarray],
    b_lower_hue: Optional[np.ndarray],
    b_upper_dom: Optional[np.ndarray],
    b_lower_dom: Optional[np.ndarray],
) -> Tuple[float, int]:
    """Combined color similarity across upper/lower hue hist + dom-color.

    Returns ``(score, n_valid_pairs)`` where ``n_valid_pairs`` counts how many
    of the 4 feature comparisons had data on both sides. Score is a weighted
    mean over valid pairs; weights favor upper-body (shirt) over lower-body
    (pants) and hue histograms slightly over dominant-color because
    histograms are more robust to specular/blur artifacts.
    """
    pieces: List[Tuple[float, float]] = []  # (score, weight)
    uh = _hue_hist_similarity(a_upper_hue, b_upper_hue)
    if a_upper_hue is not None and b_upper_hue is not None:
        pieces.append((uh, 0.35))
    lh = _hue_hist_similarity(a_lower_hue, b_lower_hue)
    if a_lower_hue is not None and b_lower_hue is not None:
        pieces.append((lh, 0.20))
    ud = _dom_color_similarity(a_upper_dom, b_upper_dom)
    if a_upper_dom is not None and b_upper_dom is not None:
        pieces.append((ud, 0.28))
    ld = _dom_color_similarity(a_lower_dom, b_lower_dom)
    if a_lower_dom is not None and b_lower_dom is not None:
        pieces.append((ld, 0.17))
    if not pieces:
        return 0.0, 0
    total_w = sum(w for _, w in pieces)
    if total_w <= 0:
        return 0.0, 0
    score = sum(s * w for s, w in pieces) / total_w
    # Penalize "shirt matches but pants disagree" and vice-versa.
    # This blocks over-collapse for similar tops but different lower body.
    if (a_upper_hue is not None and b_upper_hue is not None) and (a_lower_hue is not None and b_lower_hue is not None):
        hue_gap = abs(uh - lh)
        if hue_gap > 0.26:
            score -= 0.18 * (hue_gap - 0.26)
    if (a_upper_dom is not None and b_upper_dom is not None) and (a_lower_dom is not None and b_lower_dom is not None):
        dom_gap = abs(ud - ld)
        if dom_gap > 0.32:
            score -= 0.14 * (dom_gap - 0.32)
    return float(np.clip(score, 0.0, 1.0)), int(len(pieces))


def build_profiles(
    *,
    video_path: Path,
    tracks_csv: Path,
    source_label: str,
    gid_filter: Optional[Iterable[int]] = None,
    samples_per_gid: int = 18,
    stable_min_rows: int = 30,
    stable_min_span: int = 32,
    reid_weights_path: Optional[str] = None,
    device: str = "cpu",
    min_sample_quality: float = 0.42,
    clean_keep_ratio: float = 0.72,
    clean_min_keep: int = 6,
) -> ProfileSet:
    """Extract per-gid OSNet profile embeddings from one video + tracks CSV.

    Returns a ProfileSet where ``per_gid_mean[gid]`` is the L2-normalized
    centroid of up to ``samples_per_gid`` OSNet embeddings sampled evenly
    across the gid's frame span.

    If ``gid_filter`` is provided, only those gids are processed. Otherwise,
    all gids passing (``stable_min_rows``, ``stable_min_span``) are used.
    """
    _ensure_deps()

    video_path = Path(video_path)
    tracks_csv = Path(tracks_csv)

    _fieldnames, by_gid = _load_tracks_csv(tracks_csv)
    if gid_filter is not None:
        allowed = set(int(g) for g in gid_filter)
        by_gid = {g: s for g, s in by_gid.items() if g in allowed}

    stable = _select_stable_gids(
        by_gid, min_rows=stable_min_rows, min_span=stable_min_span
    )
    # Keep filter-specified gids even if they don't hit the stable gates.
    if gid_filter is not None:
        stable = list(by_gid.keys())
        stable.sort(key=lambda g: (by_gid[g][0].frame_idx, g))

    if not stable:
        return ProfileSet(source=source_label)

    # Pre-compute the list of (gid, sample) asks per frame.
    asks_by_frame: Dict[int, List[Tuple[int, _Sample]]] = defaultdict(list)
    meta_rows: Dict[int, int] = {}
    meta_span: Dict[int, int] = {}
    for gid in stable:
        samples = by_gid[gid]
        picks = _pick_spread_samples(samples, samples_per_gid)
        for s in picks:
            asks_by_frame[s.frame_idx].append((gid, s))
        meta_rows[gid] = len(samples)
        meta_span[gid] = samples[-1].frame_idx - samples[0].frame_idx + 1 if samples else 0

    try:
        extractor = ReidExtractor(
            model_name="osnet_x1_0", device=device, model_path=reid_weights_path
        )
    except Exception:
        # Try fallback device auto-select (e.g. GPU).
        extractor = ReidExtractor(
            model_name="osnet_x1_0", device=None, model_path=reid_weights_path
        )

    per_gid_samples: Dict[int, List[Tuple[float, np.ndarray]]] = defaultdict(list)

    # Accumulate per-gid lists of color descriptors in parallel with OSNet.
    per_gid_upper_hue: Dict[int, List[Tuple[float, np.ndarray]]] = defaultdict(list)
    per_gid_lower_hue: Dict[int, List[Tuple[float, np.ndarray]]] = defaultdict(list)
    per_gid_upper_dom: Dict[int, List[Tuple[float, np.ndarray]]] = defaultdict(list)
    per_gid_lower_dom: Dict[int, List[Tuple[float, np.ndarray]]] = defaultdict(list)
    per_gid_shape: Dict[int, List[Tuple[float, np.ndarray]]] = defaultdict(list)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for profile build: {video_path}")
    frame_w = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0)
    frame_h = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0)
    if frame_w <= 1.0:
        frame_w = 1920.0
    if frame_h <= 1.0:
        frame_h = 1080.0

    target_frames = set(asks_by_frame.keys())
    if not target_frames:
        cap.release()
        return ProfileSet(source=source_label)

    max_needed = max(target_frames)
    frame_idx = 0
    while frame_idx <= max_needed:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame_idx in target_frames:
            asks = asks_by_frame[frame_idx]
            crops: List[np.ndarray] = []
            gid_for_crop: List[int] = []
            q_for_crop: List[float] = []
            for gid, s in asks:
                bgr = _crop_bgr(frame, s)
                if bgr is None:
                    continue
                q = _sample_quality_score(
                    sample=s,
                    crop_bgr=bgr,
                    frame_w=frame_w,
                    frame_h=frame_h,
                )
                if q < float(min_sample_quality):
                    continue
                try:
                    c = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                except Exception:
                    continue
                crops.append(c)
                gid_for_crop.append(gid)
                q_for_crop.append(float(q))
                shp = _shape_descriptor_from_sample(s, frame_w=frame_w, frame_h=frame_h)
                if shp is not None:
                    per_gid_shape[gid].append((float(q), shp))
                # Color features: use BGR crop with the same bbox.
                upper_bgr, lower_bgr = _upper_lower_regions(bgr)
                uhist = _hue_sat_histogram(upper_bgr)
                lhist = _hue_sat_histogram(lower_bgr)
                udom = _dominant_color_descriptor(upper_bgr)
                ldom = _dominant_color_descriptor(lower_bgr)
                if uhist is not None:
                    per_gid_upper_hue[gid].append((float(q), uhist))
                if lhist is not None:
                    per_gid_lower_hue[gid].append((float(q), lhist))
                if udom is not None:
                    per_gid_upper_dom[gid].append((float(q), udom))
                if ldom is not None:
                    per_gid_lower_dom[gid].append((float(q), ldom))
            if crops:
                try:
                    feats = extractor(crops)
                except Exception:
                    feats = None
                if feats is not None and feats.size:
                    for gid, q, vec in zip(gid_for_crop, q_for_crop, feats):
                        per_gid_samples[gid].append((float(q), vec.astype(np.float32)))
        frame_idx += 1
    cap.release()

    profiles = ProfileSet(source=source_label)
    for gid in stable:
        vecs = per_gid_samples.get(gid, [])
        profiles.per_gid_rows[gid] = meta_rows.get(gid, 0)
        profiles.per_gid_span[gid] = meta_span.get(gid, 0)
        profiles.per_gid_proto_sample_count[gid] = int(len(vecs))
        selected_vecs, q_mean, coh_mean = _select_clean_vectors(
            vecs,
            keep_ratio=clean_keep_ratio,
            min_keep=clean_min_keep,
        )
        profiles.per_gid_proto_kept_count[gid] = int(len(selected_vecs))
        profiles.per_gid_proto_quality_mean[gid] = float(q_mean)
        profiles.per_gid_proto_coherence_mean[gid] = float(coh_mean)
        if not selected_vecs:
            continue
        profiles.per_gid_samples[gid] = selected_vecs
        mean = np.mean(np.stack(selected_vecs, axis=0), axis=0)
        if l2_normalize is not None:
            mean = l2_normalize(mean)
        else:
            n = float(np.linalg.norm(mean) + 1e-12)
            mean = (mean / n).astype(np.float32)
        profiles.per_gid_mean[gid] = mean.astype(np.float32)

    # Color-feature means (L2-normalized), one per region per gid.
    def _mean_unit(vs: List[Tuple[float, np.ndarray]], *, min_keep: int = 3) -> Optional[np.ndarray]:
        if not vs:
            return None
        picked, _q, _coh = _select_clean_vectors(
            vs,
            keep_ratio=clean_keep_ratio,
            min_keep=min_keep,
            # Color/shape features live in a different similarity regime
            # (histograms, dominant colors) than OSNet — loosen coherence so
            # we don't over-prune what is inherently multi-modal.
            coherence_floor=0.55,
            coherence_drop_below_median=0.10,
        )
        if not picked:
            return None
        m = np.mean(np.stack(picked, axis=0), axis=0)
        n = float(np.linalg.norm(m) + 1e-12)
        if n <= 1e-6:
            return None
        return (m / n).astype(np.float32)

    for gid in stable:
        uh = _mean_unit(per_gid_upper_hue.get(gid, []), min_keep=3)
        lh = _mean_unit(per_gid_lower_hue.get(gid, []), min_keep=3)
        ud = _mean_unit(per_gid_upper_dom.get(gid, []), min_keep=3)
        ld = _mean_unit(per_gid_lower_dom.get(gid, []), min_keep=3)
        if uh is not None:
            profiles.per_gid_upper_hue_hist[gid] = uh
        if lh is not None:
            profiles.per_gid_lower_hue_hist[gid] = lh
        if ud is not None:
            profiles.per_gid_upper_dom[gid] = ud
        if ld is not None:
            profiles.per_gid_lower_dom[gid] = ld
        shp = _mean_unit(per_gid_shape.get(gid, []), min_keep=max(3, clean_min_keep // 2))
        if shp is not None:
            profiles.per_gid_shape[gid] = shp

    return profiles


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    if a.size == 0 or b.size == 0:
        return 0.0
    na = float(np.linalg.norm(a) + 1e-12)
    nb = float(np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b) / (na * nb))


def _build_color_sim_matrix(
    target_profiles: ProfileSet,
    reference_profiles: ProfileSet,
) -> Tuple[Dict[int, Dict[int, float]], Dict[int, Dict[int, int]]]:
    """Compute a color-similarity matrix target gid -> ref gid -> score.

    Also returns the parallel count matrix of valid feature pairs (0..4) so the
    caller can decide whether to trust the score.
    """
    sim: Dict[int, Dict[int, float]] = {}
    nvp: Dict[int, Dict[int, int]] = {}
    tgts = list(target_profiles.per_gid_mean.keys())
    refs = list(reference_profiles.per_gid_mean.keys())
    for t in tgts:
        sim_t: Dict[int, float] = {}
        nvp_t: Dict[int, int] = {}
        tu_hue = target_profiles.per_gid_upper_hue_hist.get(t)
        tl_hue = target_profiles.per_gid_lower_hue_hist.get(t)
        tu_dom = target_profiles.per_gid_upper_dom.get(t)
        tl_dom = target_profiles.per_gid_lower_dom.get(t)
        for r in refs:
            ru_hue = reference_profiles.per_gid_upper_hue_hist.get(r)
            rl_hue = reference_profiles.per_gid_lower_hue_hist.get(r)
            ru_dom = reference_profiles.per_gid_upper_dom.get(r)
            rl_dom = reference_profiles.per_gid_lower_dom.get(r)
            score, nv = _combined_color_similarity(
                tu_hue, tl_hue, tu_dom, tl_dom,
                ru_hue, rl_hue, ru_dom, rl_dom,
            )
            sim_t[r] = score
            nvp_t[r] = nv
        sim[t] = sim_t
        nvp[t] = nvp_t
    return sim, nvp


def _build_shape_sim_matrix(
    target_profiles: ProfileSet,
    reference_profiles: ProfileSet,
) -> Dict[int, Dict[int, float]]:
    sim: Dict[int, Dict[int, float]] = {}
    tgts = list(target_profiles.per_gid_mean.keys())
    refs = list(reference_profiles.per_gid_mean.keys())
    for t in tgts:
        ts = target_profiles.per_gid_shape.get(t)
        row: Dict[int, float] = {}
        for r in refs:
            rs = reference_profiles.per_gid_shape.get(r)
            if ts is None or rs is None:
                row[r] = 0.0
            else:
                row[r] = float(np.clip(np.dot(ts, rs), 0.0, 1.0))
        sim[t] = row
    return sim


def _combined_pair_score(
    *,
    osnet: float,
    color: float,
    n_valid_color: int,
    shape: float,
    shape_informative: bool = True,
) -> float:
    """Shared combined score for anchor recovery decisions.

    ``shape_informative`` lets the caller suppress the shape cue when it is
    near-uniform across references (e.g. within ~0.05), since in that regime
    shape can flip the combined ordering without carrying real information.
    """
    # Shape is intentionally reject-only in anchor remapping:
    # saturated shape cues (0.97-1.00 across many wrong pairs) should not
    # rescue weak appearance matches. We only apply a tiny penalty when
    # shape is informative *and* very low.
    score = float(osnet)
    if int(n_valid_color) >= 2:
        score = float(0.82 * osnet + 0.18 * color)
    if bool(shape_informative) and shape > 0.0 and shape < 0.40:
        score -= 0.02 * float((0.40 - shape) / 0.40)
    return float(score)


def _source_quality_summary(profiles: ProfileSet, gid: int) -> Dict[str, float]:
    rows = float(profiles.per_gid_rows.get(int(gid), 0))
    span = float(profiles.per_gid_span.get(int(gid), 0))
    proto_q = float(profiles.per_gid_proto_quality_mean.get(int(gid), 0.0))
    proto_coh = float(profiles.per_gid_proto_coherence_mean.get(int(gid), 0.0))
    proto_kept = float(profiles.per_gid_proto_kept_count.get(int(gid), 0))
    proto_samples = float(profiles.per_gid_proto_sample_count.get(int(gid), 0))
    low_quality = (
        proto_q < 0.50
        or proto_coh < 0.82
        or proto_kept < 5.0
        or rows < 30.0
        or span < 32.0
    )
    return {
        "rows": rows,
        "span": span,
        "prototype_quality_mean": proto_q,
        "prototype_coherence_mean": proto_coh,
        "prototype_kept_count": proto_kept,
        "prototype_sample_count": proto_samples,
        "low_quality": 1.0 if low_quality else 0.0,
    }


def _recover_missing_reference_matches(
    *,
    target_profiles: ProfileSet,
    reference_profiles: ProfileSet,
    current_mapping: Mapping[int, int],
    min_cos: float,
    min_margin: float,
) -> Tuple[Dict[int, int], Dict[int, str]]:
    """Recover only still-missing canonical refs with mildly relaxed gates.

    Safety: this pass NEVER touches already-claimed references, so it cannot
    collapse fragments into the surviving canonical IDs. It only tries to map
    unmapped targets onto unmapped canonical refs.
    """
    mapped_tgts = set(int(t) for t in current_mapping.keys())
    mapped_refs = set(int(r) for r in current_mapping.values())
    all_tgts = sorted(int(t) for t in target_profiles.per_gid_mean.keys())
    all_refs = sorted(int(r) for r in reference_profiles.per_gid_mean.keys())
    candidate_tgts = [t for t in all_tgts if t not in mapped_tgts]
    missing_refs = [r for r in all_refs if r not in mapped_refs]
    if not candidate_tgts or not missing_refs:
        return {}, {}

    color_sim, color_nvp = _build_color_sim_matrix(target_profiles, reference_profiles)
    shape_sim = _build_shape_sim_matrix(target_profiles, reference_profiles)

    # Missing-canonical-only recovery gates.
    # Keep this noticeably looser than the primary matcher, but still guarded
    # by secondary evidence and one-to-one reference separation.
    rec_min_cos = max(0.0, float(min_cos) - 0.05)
    rec_min_margin = float(min_margin) - 0.05
    rec_min_combined_margin = max(0.010, 0.22 * float(min_margin))
    rec_min_secondary = 0.57
    rec_min_secondary_margin = -0.01

    # Per-target shape dispersion: if shape is near-uniform across references
    # (< 0.05 range), it is NOT discriminating this target — suppress it from
    # the combined score so it cannot flip OSNet's ordering. This addresses
    # the observed 0.97-1.00 shape saturation that made the combined score
    # select ref=6 for target gid 10 even though OSNet-unclaimed prefers
    # ref=8 by ~0.02.
    def _shape_informative(t_: int) -> bool:
        row = shape_sim.get(int(t_), {}) or {}
        vals = [float(v) for v in row.values() if v is not None]
        if len(vals) < 2:
            return False
        return (max(vals) - min(vals)) >= 0.05

    proposals: List[Tuple[int, int, float, float, float, float, float, int]] = []
    for t in candidate_tgts:
        tvec = target_profiles.per_gid_mean.get(int(t))
        if tvec is None:
            continue
        s_info = _shape_informative(int(t))
        scored: List[Tuple[int, float, float, float, float, int]] = []
        for r in missing_refs:
            rvec = reference_profiles.per_gid_mean.get(int(r))
            if rvec is None:
                continue
            osn = float(_cosine(tvec, rvec))
            col = float(color_sim.get(int(t), {}).get(int(r), 0.0))
            nv = int(color_nvp.get(int(t), {}).get(int(r), 0))
            shp = float(shape_sim.get(int(t), {}).get(int(r), 0.0))
            comb = _combined_pair_score(
                osnet=osn, color=col, n_valid_color=nv, shape=shp,
                shape_informative=s_info,
            )
            scored.append((int(r), osn, col, shp, comb, nv))
        if not scored:
            continue
        scored.sort(key=lambda x: x[4], reverse=True)
        best_r, best_osn, best_col, best_shp, best_comb, best_nv = scored[0]
        if len(scored) > 1:
            _r2, second_osn, second_col, second_shp, second_comb, _nv2 = scored[1]
        else:
            second_osn, second_col, second_shp, second_comb = -1.0, 0.0, 0.0, -1.0
        osn_margin = float(best_osn - second_osn) if second_osn >= 0.0 else float(best_osn)
        comb_margin = float(best_comb - second_comb) if second_comb >= 0.0 else float(best_comb)
        secondary_support = float(best_col)
        secondary_margin = float(best_col - second_col)
        q = _source_quality_summary(target_profiles, int(t))
        low_quality = bool(q.get("low_quality", 0.0) > 0.5)
        ambiguous = bool(
            second_osn >= 0.0
            and osn_margin < max(0.015, 0.45 * float(min_margin))
            and best_osn < (rec_min_cos + 0.08)
        )
        # Veto: decent OSNet but clear part/color disagreement to the top ref.
        if best_nv >= 2 and best_col < 0.46 and secondary_margin < -0.05 and best_osn < (rec_min_cos + 0.05):
            continue

        # Standard recovery path.
        std_ok = (
            best_osn >= rec_min_cos
            and (osn_margin >= rec_min_margin or second_osn < 0.0)
            and comb_margin >= rec_min_combined_margin
            and secondary_support >= rec_min_secondary
            and (secondary_margin >= rec_min_secondary_margin or second_comb < 0.0)
            and not ambiguous
            and (not low_quality or (best_osn >= rec_min_cos + 0.05 and osn_margin >= rec_min_margin + 0.02))
        )

        # Targeted ambiguity-breaking path: OSNet is in a plausible range and
        # among unmapped refs it clearly prefers best_r (osnet_margin >= 0.02)
        # AND color ALSO picks the same ref uniquely by >= 0.04. This is a
        # joint cue agreement — both independent signals pick the SAME
        # answer. We do NOT relax min_cos, and we do NOT accept if color is
        # absent. We do require osnet to clear rec_min_cos - 0.02 (a very
        # small extra margin vs. the standard recovery path).
        ambig_ok = (
            not std_ok
            and best_osn >= (rec_min_cos - 0.02)
            and osn_margin >= 0.02
            and best_nv >= 3
            and best_col >= 0.56
            and (best_col - second_col) >= 0.04
            and secondary_margin >= 0.03
            and comb_margin >= 0.0
            and not ambiguous
            and not low_quality
        )

        # OSNet-led path (for prototype-coherent fragments whose OSNet
        # score is moderate but clearly self-separated among unclaimed
        # refs, and where color does not actively disagree).
        #
        # Motivation: with coherence-based pruning in place, some source
        # fragments now have clean-but-moderate OSNet scores in the
        # 0.70-0.75 band. These cannot clear the standard (rec_min_cos)
        # or fallback thresholds, yet:
        #   - the fragment is prototype-coherent (pruning succeeded),
        #   - OSNet cleanly prefers one unclaimed ref over the others,
        #   - color either agrees with OSNet's pick, or is cue-neutral
        #     (|best_col - second_col| <= 0.03, i.e. color is effectively
        #     indifferent across missing refs — so it cannot veto).
        # We use a hard OSNet floor (0.70) and require a non-negative
        # combined margin to ensure we never flip the OSNet ordering on
        # pure color noise. We do NOT touch claimed refs.
        color_agrees_with_osnet = (best_nv >= 2 and (best_col - second_col) >= 0.0)
        color_neutral_on_unclaimed = abs(best_col - second_col) <= 0.03
        # Split into two variants. The color-neutral path is the primary
        # wrong-reuse risk for this recovery layer: when two references
        # have similar color (e.g. two people in black shirts), color
        # cannot disambiguate, so we must require OSNet to be STRICTLY
        # more self-separated.
        osnet_led_ok_color_agreeing = (
            not (std_ok or ambig_ok)
            and color_agrees_with_osnet
            and best_osn >= 0.70
            and osn_margin >= 0.022
            and comb_margin >= 0.008
            and best_nv >= 2
            and best_col >= 0.52
            and secondary_margin >= -0.015
            and not ambiguous
            and not low_quality
        )
        osnet_led_ok_color_neutral = (
            not (std_ok or ambig_ok)
            and color_neutral_on_unclaimed
            and not color_agrees_with_osnet
            and best_osn >= 0.72
            and osn_margin >= 0.035
            and comb_margin >= 0.012
            and best_nv >= 2
            and best_col >= 0.52
            and secondary_margin >= -0.010
            and not ambiguous
            and not low_quality
        )
        osnet_led_ok = osnet_led_ok_color_agreeing or osnet_led_ok_color_neutral

        if not (std_ok or ambig_ok or osnet_led_ok):
            continue
        proposals.append(
            (
                int(t),
                int(best_r),
                float(best_comb),
                float(best_osn),
                float(osn_margin),
                float(secondary_support),
                float(secondary_margin),
                int(best_nv),
            )
        )

    if not proposals:
        return {}, {}

    # Reference-side tie check: if two targets are near-equal for the same
    # missing canonical ref, require stronger secondary support before accept.
    by_ref_comb: Dict[int, List[float]] = {}
    for _t, _r, _comb, _osn, _om, _sec, _sec_m, _nv in proposals:
        by_ref_comb.setdefault(int(_r), []).append(float(_comb))
    ref_second_best: Dict[int, float] = {}
    for _r, _vals in by_ref_comb.items():
        _vals = sorted(_vals, reverse=True)
        ref_second_best[int(_r)] = float(_vals[1]) if len(_vals) > 1 else -1.0

    proposals.sort(key=lambda x: (x[2], x[4], x[5]), reverse=True)
    recovered: Dict[int, int] = {}
    notes: Dict[int, str] = {}
    used_t: set[int] = set()
    used_r: set[int] = set()
    for t, r, comb, osn, om, sec, sec_m, nv in proposals:
        if t in used_t or r in used_r:
            continue
        r2 = float(ref_second_best.get(int(r), -1.0))
        ref_margin = float(comb - r2) if r2 >= 0.0 else float(comb)
        # Slightly looser ref-margin guard (was 0.01 / 0.72): when two
        # missing-canonical candidates land on the same ref with nearly
        # identical combined scores, we still need secondary support to
        # disambiguate, but the previous gates were too strict to let
        # ANY coherent-but-moderate proposal through. 0.006 / 0.66 keeps
        # true ties rejected while letting a genuinely-clearer proposal
        # win. Note: the one-to-one `used_t` / `used_r` bookkeeping below
        # still prevents the same ref from being assigned twice.
        if ref_margin < 0.006 and sec < 0.66:
            continue
        recovered[int(t)] = int(r)
        notes[int(t)] = (
            f"missing-canonical-recovery ref={r} "
            f"osnet={osn:.3f} osnet_margin={om:.3f} "
            f"secondary={sec:.3f} secondary_margin={sec_m:.3f} "
            f"ref_margin={ref_margin:.3f} "
            f"n_color_pairs={nv}"
        )
        used_t.add(int(t))
        used_r.add(int(r))
    return recovered, notes


def match_profiles_strong(
    *,
    target_profiles: ProfileSet,
    reference_profiles: ProfileSet,
    min_mean_cos: float = 0.82,
    min_max_cos: float = 0.88,
    min_sample_vote_share: float = 0.55,
    min_margin: float = 0.04,
    require_bidirectional: bool = True,
    fallback_start: int = 12,
) -> MatchReport:
    """Strong per-sample consensus matcher with bidirectional voting.

    Rules (all must hold to accept target -> reference):
      1. mean(cos(t_samples, ref_mean)) >= ``min_mean_cos``
      2. max(cos(t_samples, ref_mean)) >= ``min_max_cos``
      3. Fraction of target samples whose top-1 reference is ``ref`` >=
         ``min_sample_vote_share``
      4. (mean_cos_top1 - mean_cos_top2) >= ``min_margin`` across *unclaimed*
         reference gids
      5. If ``require_bidirectional``: among reference ``ref``'s samples,
         the top-1 vote (counting only unclaimed targets) must be this target.

    Rejected non-canonical gids may get fallback canonical IDs starting at
    ``fallback_start``. Existing canonical gids are handled with a strict
    do-no-harm bias: when evidence is ambiguous or weak, keep self.
    """
    report = MatchReport()
    report.params = {
        "mode": "strong",
        "min_mean_cos": float(min_mean_cos),
        "min_max_cos": float(min_max_cos),
        "min_sample_vote_share": float(min_sample_vote_share),
        "min_margin": float(min_margin),
        "require_bidirectional": bool(require_bidirectional),
        "fallback_start": int(fallback_start),
        "shape_policy": "reject_only",
        "canonical_preservation_bias": 1.0,
    }

    ref_gids = sorted(reference_profiles.per_gid_mean.keys())
    tgt_gids = sorted(
        target_profiles.per_gid_mean.keys(),
        key=lambda g: (target_profiles.per_gid_span.get(g, 0) * -1, g),
    )
    report.cam1_reference_gids = list(ref_gids)
    report.full_cam1_stable_gids = list(tgt_gids)

    if not ref_gids or not tgt_gids:
        return report

    ref_mean = {r: reference_profiles.per_gid_mean[r] for r in ref_gids}

    mean_cos: Dict[int, Dict[int, float]] = {}
    max_cos: Dict[int, Dict[int, float]] = {}
    vote_share: Dict[int, Dict[int, float]] = {}
    for t in tgt_gids:
        tsamples = target_profiles.per_gid_samples.get(t, [])
        if not tsamples:
            continue
        mean_cos[t] = {}
        max_cos[t] = {}
        sample_best_ref: List[int] = []
        per_ref_cos: Dict[int, List[float]] = {r: [] for r in ref_gids}
        for s in tsamples:
            best_r = -1
            best_c = -1.0
            for r in ref_gids:
                c = _cosine(s, ref_mean[r])
                per_ref_cos[r].append(c)
                if c > best_c:
                    best_c = c
                    best_r = r
            sample_best_ref.append(best_r)
        for r in ref_gids:
            arr = per_ref_cos[r] or [0.0]
            mean_cos[t][r] = float(np.mean(arr))
            max_cos[t][r] = float(np.max(arr))
        vs: Dict[int, float] = {r: 0.0 for r in ref_gids}
        total = max(1, len(sample_best_ref))
        for r in sample_best_ref:
            if r >= 0:
                vs[r] = vs.get(r, 0.0) + (1.0 / total)
        vote_share[t] = vs

    report.scores = mean_cos
    color_sim, color_nvp = _build_color_sim_matrix(target_profiles, reference_profiles)
    shape_sim = _build_shape_sim_matrix(target_profiles, reference_profiles)
    report.color_scores = {int(t): {int(r): float(v) for r, v in row.items()} for t, row in color_sim.items()}
    report.shape_scores = {int(t): {int(r): float(v) for r, v in row.items()} for t, row in shape_sim.items()}

    tgt_mean = {t: target_profiles.per_gid_mean[t] for t in tgt_gids if t in target_profiles.per_gid_mean}
    reverse_votes: Dict[int, Dict[int, float]] = {r: {} for r in ref_gids}
    for r in ref_gids:
        rsamples = reference_profiles.per_gid_samples.get(r, [])
        if not rsamples:
            continue
        total = max(1, len(rsamples))
        for rs in rsamples:
            best_t = -1
            best_c = -1.0
            for t, tmean in tgt_mean.items():
                c = _cosine(rs, tmean)
                if c > best_c:
                    best_c = c
                    best_t = t
            if best_t >= 0:
                reverse_votes[r][best_t] = reverse_votes[r].get(best_t, 0.0) + (1.0 / total)

    shape_informative: Dict[int, bool] = {}
    for t in mean_cos.keys():
        vals = [float(v) for v in (shape_sim.get(t, {}) or {}).values() if v is not None]
        shape_informative[t] = bool(len(vals) >= 2 and (max(vals) - min(vals)) >= 0.05)

    combined: Dict[int, Dict[int, float]] = {}
    for t, row in mean_cos.items():
        csim = color_sim.get(t, {})
        cnvp = color_nvp.get(t, {})
        ssim = shape_sim.get(t, {})
        s_info = shape_informative.get(t, True)
        combined[t] = {}
        for r, c in row.items():
            combined[t][r] = _combined_pair_score(
                osnet=float(c),
                color=float(csim.get(r, 0.0)),
                n_valid_color=int(cnvp.get(r, 0)),
                shape=float(ssim.get(r, 0.0)),
                shape_informative=s_info,
            )

    ref_gid_set = set(int(r) for r in ref_gids)
    canonical_targets = set(int(t) for t in tgt_gids if int(t) in ref_gid_set)
    claimed: set[int] = set()
    ordering = sorted(
        [t for t in tgt_gids if t in mean_cos],
        key=lambda t: (
            0 if int(t) in canonical_targets else 1,
            -(mean_cos.get(t, {}).get(t, -1.0) if int(t) in canonical_targets else -1.0),
            -(max(combined.get(t, {}).values()) if combined.get(t) else -1.0),
            int(t),
        ),
    )

    for t in ordering:
        row_mean = mean_cos.get(t, {})
        if not row_mean:
            report.rejected[int(t)] = "no-ref-profiles"
            report.decision_details[int(t)] = {"decision": "rejected", "reason": "no-ref-profiles"}
            continue
        open_refs_osnet = [(r, c) for r, c in row_mean.items() if r not in claimed]
        if not open_refs_osnet:
            report.unresolved[int(t)] = "all-refs-claimed"
            report.decision_details[int(t)] = {
                "decision": "unresolved",
                "reason": "all-refs-claimed",
                "shape_downweighted": True,
                "shape_ignored": True,
            }
            continue

        open_refs_combined = [(r, combined.get(t, {}).get(r, c)) for r, c in open_refs_osnet]
        open_refs_combined.sort(key=lambda x: -x[1])
        best_ref = int(open_refs_combined[0][0])
        best_comb = float(open_refs_combined[0][1])
        second_ref = int(open_refs_combined[1][0]) if len(open_refs_combined) > 1 else -1
        second_comb = float(open_refs_combined[1][1]) if len(open_refs_combined) > 1 else -1.0

        best_mean = float(row_mean.get(best_ref, 0.0))
        second_mean = float(row_mean.get(second_ref, -1.0)) if second_ref >= 0 else -1.0
        osnet_margin = float(best_mean - second_mean)
        top1_top2_margin = float(osnet_margin)

        vs_best = float(vote_share.get(t, {}).get(best_ref, 0.0))
        max_c_best = float(max_cos.get(t, {}).get(best_ref, 0.0))
        color_best = float(color_sim.get(t, {}).get(best_ref, 0.0))
        color_second = float(color_sim.get(t, {}).get(second_ref, 0.0)) if second_ref >= 0 else 0.0
        color_margin = float(color_best - color_second)
        n_valid_color = int(color_nvp.get(t, {}).get(best_ref, 0))
        shape_best = float(shape_sim.get(t, {}).get(best_ref, 0.0))
        combined_margin = float(best_comb - second_comb) if second_comb >= 0.0 else float(best_comb)
        secondary_support = float(color_best)
        secondary_margin = float(color_margin)

        q = _source_quality_summary(target_profiles, int(t))
        low_quality = bool(q.get("low_quality", 0.0) > 0.5)
        ambiguous = bool(
            second_mean > 0.0
            and osnet_margin < max(0.018, 0.55 * float(min_margin))
            and best_mean < (float(min_mean_cos) + 0.09)
        )

        self_cosine: Optional[float] = float(row_mean.get(int(t), 0.0)) if int(t) in row_mean else None
        self_available = bool(int(t) in row_mean and int(t) not in claimed)
        other_open = sorted(
            [(int(r), float(v)) for r, v in row_mean.items() if int(r) not in claimed and int(r) != int(t)],
            key=lambda x: -x[1],
        )
        best_other_ref = int(other_open[0][0]) if other_open else None
        best_other_cos = float(other_open[0][1]) if other_open else None
        self_margin = (
            float(self_cosine - best_other_cos)
            if (self_cosine is not None and best_other_cos is not None)
            else None
        )
        self_separation_gap = (
            float(best_other_cos - self_cosine)
            if (self_cosine is not None and best_other_cos is not None)
            else None
        )
        osnet_prefers_self = (
            bool(self_margin >= 0.0)
            if self_margin is not None
            else None
        )
        color_on_self = float(color_sim.get(t, {}).get(int(t), 0.0)) if int(t) in row_mean else None
        color_agrees_with_self = (
            bool(color_on_self + 0.015 >= color_best)
            if color_on_self is not None
            else None
        )

        is_existing_canonical = bool(int(t) in canonical_targets)
        remap_away_from_self = bool(is_existing_canonical and self_available and best_ref != int(t))
        clear_other_win = bool(
            remap_away_from_self
            and best_mean >= float(min_mean_cos) + 0.03
            and max_c_best >= float(min_max_cos)
            and vs_best >= float(min_sample_vote_share)
            and osnet_margin >= max(float(min_margin) + 0.02, 0.05)
            and combined_margin >= max(0.03, 0.65 * float(min_margin))
            and (self_cosine is None or (best_mean - float(self_cosine)) >= 0.04)
            and secondary_support >= 0.56
            and not ambiguous
            and not low_quality
        )
        near_miss_self_rescue = bool(
            is_existing_canonical
            and self_available
            and self_cosine is not None
            and self_margin is not None
            and self_margin >= 0.003
            and osnet_prefers_self is True
            and color_agrees_with_self is True
            and float(q.get("prototype_quality_mean", 0.0)) >= 0.50
            and float(q.get("prototype_kept_count", 0.0)) >= 5.0
            and float(self_cosine) >= (float(min_mean_cos) - 0.11)
            and float(self_margin) < max(float(min_margin), 0.05)
        )

        canonical_preservation_block = False
        ambiguity_block = False
        low_quality_block = False
        rescue_path: Optional[str] = None
        if is_existing_canonical and self_available:
            if near_miss_self_rescue:
                claimed.add(int(t))
                report.mapping[int(t)] = int(t)
                rescue_path = "near-miss-self-rescue"
                if not hasattr(report, "rescue_reasons"):
                    report.rescue_reasons = {}  # type: ignore[attr-defined]
                report.rescue_reasons[int(t)] = (  # type: ignore[attr-defined]
                    f"near-miss-self-rescue self={self_cosine:.3f} "
                    f"self_margin={self_margin:.3f} color_self={color_on_self:.3f}"
                )
            elif best_ref == int(t):
                claimed.add(int(t))
                report.mapping[int(t)] = int(t)
            elif remap_away_from_self and not clear_other_win:
                canonical_preservation_block = True
                ambiguity_block = bool(ambiguous)
                low_quality_block = bool(low_quality)
                claimed.add(int(t))
                report.mapping[int(t)] = int(t)

        if int(t) in report.mapping:
            report.decision_details[int(t)] = {
                "decision": "mapped-self" if int(report.mapping[int(t)]) == int(t) else "mapped",
                "decision_reason": (
                    rescue_path
                    or ("canonical-preservation-bias" if canonical_preservation_block else "self-competitive")
                ),
                "best_ref": int(best_ref),
                "best_osnet": float(best_mean),
                "second_ref": int(second_ref) if second_ref >= 0 else None,
                "second_osnet": float(second_mean) if second_ref >= 0 else None,
                "top1_top2_margin": float(top1_top2_margin),
                "combined_margin": float(combined_margin),
                "self_cosine": float(self_cosine) if self_cosine is not None else None,
                "self_separation_gap": float(self_separation_gap) if self_separation_gap is not None else None,
                "self_preferred": osnet_prefers_self,
                "color_agrees_with_self": color_agrees_with_self,
                "canonical_preservation_blocked_remap": bool(canonical_preservation_block),
                "ambiguity_block_fired": bool(ambiguity_block),
                "low_prototype_quality_blocked_remap": bool(low_quality_block),
                "rescue_path_fired": rescue_path,
                "shape_downweighted": True,
                "shape_ignored": not bool(shape_informative.get(int(t), False)),
                "shape_top1": float(shape_best),
                "prototype_quality_mean": float(q.get("prototype_quality_mean", 0.0)),
                "prototype_coherence_mean": float(q.get("prototype_coherence_mean", 0.0)),
                "prototype_kept_count": int(q.get("prototype_kept_count", 0.0)),
                "row_count": int(q.get("rows", 0.0)),
                "frame_span": int(q.get("span", 0.0)),
            }
            continue

        shape_sanity_reject = bool(
            shape_informative.get(int(t), False)
            and shape_best < 0.32
            and best_mean < (float(min_mean_cos) + 0.05)
        )

        if (
            n_valid_color >= 2
            and color_best < 0.44
            and secondary_margin < -0.07
            and best_mean < (float(min_mean_cos) + 0.08)
        ):
            report.unresolved[int(t)] = (
                f"color-veto (osnet_top1={best_mean:.3f} color_top1={color_best:.3f} "
                f"color_margin={secondary_margin:.3f})"
            )
            report.decision_details[int(t)] = {
                "decision": "unresolved",
                "decision_reason": "color-veto",
                "best_ref": int(best_ref),
                "best_osnet": float(best_mean),
                "top1_top2_margin": float(top1_top2_margin),
                "self_cosine": float(self_cosine) if self_cosine is not None else None,
                "self_separation_gap": float(self_separation_gap) if self_separation_gap is not None else None,
                "self_preferred": osnet_prefers_self,
                "color_agrees_with_self": color_agrees_with_self,
                "shape_downweighted": True,
                "shape_ignored": not bool(shape_informative.get(int(t), False)),
                "shape_top1": float(shape_best),
            }
            continue

        passed_strict = (
            best_mean >= float(min_mean_cos)
            and max_c_best >= float(min_max_cos)
            and vs_best >= float(min_sample_vote_share)
            and (osnet_margin >= float(min_margin) or second_mean <= 0.0)
            and combined_margin >= max(0.022, 0.45 * float(min_margin))
            and secondary_support >= 0.50
            and not ambiguous
            and not shape_sanity_reject
            and (
                not low_quality
                or (
                    best_mean >= float(min_mean_cos) + 0.06
                    and osnet_margin >= max(float(min_margin) + 0.02, 0.05)
                    and secondary_support >= 0.58
                )
            )
        )

        if not passed_strict:
            reason = ""
            if ambiguous:
                reason = (
                    f"ambiguity-block (top1={best_mean:.3f} top2={second_mean:.3f} "
                    f"margin={osnet_margin:.3f})"
                )
                report.unresolved[int(t)] = reason
            elif low_quality and best_ref != int(t):
                reason = (
                    f"low-prototype-quality-block (top1={best_mean:.3f} "
                    f"quality={q.get('prototype_quality_mean', 0.0):.3f} "
                    f"coherence={q.get('prototype_coherence_mean', 0.0):.3f})"
                )
                report.unresolved[int(t)] = reason
            elif shape_sanity_reject:
                reason = (
                    f"shape-sanity-reject (shape_top1={shape_best:.3f} "
                    f"osnet_top1={best_mean:.3f})"
                )
                report.unresolved[int(t)] = reason
            elif best_mean < float(min_mean_cos):
                reason = (
                    f"below-min-mean-cos ({best_mean:.3f} < {min_mean_cos:.3f}) "
                    f"color_top1={color_best:.3f}"
                )
                report.rejected[int(t)] = reason
            elif max_c_best < float(min_max_cos):
                reason = (
                    f"below-min-max-cos (max={max_c_best:.3f} < {min_max_cos:.3f}) "
                    f"color_top1={color_best:.3f}"
                )
                report.rejected[int(t)] = reason
            elif vs_best < float(min_sample_vote_share):
                reason = (
                    f"below-vote-share (share={vs_best:.2f} < {min_sample_vote_share:.2f}) "
                    f"color_top1={color_best:.3f}"
                )
                report.rejected[int(t)] = reason
            elif osnet_margin < float(min_margin) and second_mean > 0.0:
                reason = (
                    f"below-min-margin (mean_top1={best_mean:.3f} top2={second_mean:.3f} "
                    f"margin={osnet_margin:.3f} < {min_margin:.3f}) "
                    f"color_top1={color_best:.3f}"
                )
                report.unresolved[int(t)] = reason
            else:
                reason = (
                    f"conservative-block (combined_margin={combined_margin:.3f} "
                    f"color_top1={color_best:.3f})"
                )
                report.unresolved[int(t)] = reason

            report.decision_details[int(t)] = {
                "decision": "unresolved" if int(t) in report.unresolved else "rejected",
                "decision_reason": reason,
                "best_ref": int(best_ref),
                "best_osnet": float(best_mean),
                "second_ref": int(second_ref) if second_ref >= 0 else None,
                "second_osnet": float(second_mean) if second_ref >= 0 else None,
                "top1_top2_margin": float(top1_top2_margin),
                "combined_margin": float(combined_margin),
                "self_cosine": float(self_cosine) if self_cosine is not None else None,
                "self_separation_gap": float(self_separation_gap) if self_separation_gap is not None else None,
                "self_preferred": osnet_prefers_self,
                "color_agrees_with_self": color_agrees_with_self,
                "canonical_preservation_blocked_remap": False,
                "ambiguity_block_fired": bool(ambiguous),
                "low_prototype_quality_blocked_remap": bool(low_quality and best_ref != int(t)),
                "rescue_path_fired": None,
                "shape_downweighted": True,
                "shape_ignored": not bool(shape_informative.get(int(t), False)),
                "shape_top1": float(shape_best),
                "prototype_quality_mean": float(q.get("prototype_quality_mean", 0.0)),
                "prototype_coherence_mean": float(q.get("prototype_coherence_mean", 0.0)),
                "prototype_kept_count": int(q.get("prototype_kept_count", 0.0)),
                "row_count": int(q.get("rows", 0.0)),
                "frame_span": int(q.get("span", 0.0)),
            }
            continue

        if require_bidirectional:
            rv = reverse_votes.get(best_ref, {})
            if rv:
                rv_open = {tg: s for tg, s in rv.items() if tg == t or tg not in set(report.mapping.keys())}
                if rv_open:
                    rv_top_tgt = max(rv_open, key=lambda k: rv_open[k])
                    if rv_top_tgt != t:
                        reason = (
                            f"bidirectional-fail (ref {best_ref} top target "
                            f"is {rv_top_tgt}, not {t})"
                        )
                        report.unresolved[int(t)] = reason
                        report.decision_details[int(t)] = {
                            "decision": "unresolved",
                            "decision_reason": reason,
                            "best_ref": int(best_ref),
                            "best_osnet": float(best_mean),
                            "top1_top2_margin": float(top1_top2_margin),
                            "self_cosine": float(self_cosine) if self_cosine is not None else None,
                            "self_separation_gap": float(self_separation_gap) if self_separation_gap is not None else None,
                            "self_preferred": osnet_prefers_self,
                            "color_agrees_with_self": color_agrees_with_self,
                            "shape_downweighted": True,
                            "shape_ignored": not bool(shape_informative.get(int(t), False)),
                        }
                        continue

        claimed.add(int(best_ref))
        report.mapping[int(t)] = int(best_ref)
        report.decision_details[int(t)] = {
            "decision": "mapped",
            "decision_reason": "strict-accept",
            "best_ref": int(best_ref),
            "best_osnet": float(best_mean),
            "second_ref": int(second_ref) if second_ref >= 0 else None,
            "second_osnet": float(second_mean) if second_ref >= 0 else None,
            "top1_top2_margin": float(top1_top2_margin),
            "combined_margin": float(combined_margin),
            "self_cosine": float(self_cosine) if self_cosine is not None else None,
            "self_separation_gap": float(self_separation_gap) if self_separation_gap is not None else None,
            "self_preferred": osnet_prefers_self,
            "color_agrees_with_self": color_agrees_with_self,
            "canonical_preservation_blocked_remap": False,
            "ambiguity_block_fired": False,
            "low_prototype_quality_blocked_remap": False,
            "rescue_path_fired": None,
            "shape_downweighted": True,
            "shape_ignored": not bool(shape_informative.get(int(t), False)),
            "shape_top1": float(shape_best),
            "prototype_quality_mean": float(q.get("prototype_quality_mean", 0.0)),
            "prototype_coherence_mean": float(q.get("prototype_coherence_mean", 0.0)),
            "prototype_kept_count": int(q.get("prototype_kept_count", 0.0)),
            "row_count": int(q.get("rows", 0.0)),
            "frame_span": int(q.get("span", 0.0)),
        }

    # Never push existing canonical IDs into fallback space by default.
    for t in [int(x) for x in list(report.rejected.keys())]:
        if int(t) in canonical_targets:
            report.unresolved.setdefault(int(t), str(report.rejected.get(int(t), "")))
            report.rejected.pop(int(t), None)

    rejected_tgts = [int(t) for t in tgt_gids if int(t) in report.rejected and int(t) not in report.unresolved]
    rejected_tgts.sort(key=lambda g: (tgt_gids.index(g), g))
    next_id = int(fallback_start)
    for t in rejected_tgts:
        report.fallback_assignments[int(t)] = int(next_id)
        next_id += 1
    return report


def match_profiles(
    *,
    target_profiles: ProfileSet,  # FULL_CAM1
    reference_profiles: ProfileSet,  # CAM1
    min_cos: float = 0.78,
    min_margin: float = 0.07,
    fallback_start: int = 12,
) -> MatchReport:
    """Greedy bipartite match of target (FULL_CAM1) gids to reference (CAM1) gids.

    - Each reference gid can be claimed by at most one target gid (the one
      with the highest cosine among candidates above ``min_cos``).
    - A target gid is accepted only if (top1_cos >= min_cos) AND
      (top1_cos - top2_cos >= min_margin) — where top2 is the next-best
      *unclaimed* reference gid.
    - Rejected non-canonical target gids may get fallback IDs starting at
      ``fallback_start``.
    - Existing canonical IDs are preserved unless competing evidence is
      clearly stronger and non-ambiguous.
    """
    report = MatchReport()
    report.params = {
        "min_cos": float(min_cos),
        "min_margin": float(min_margin),
        "fallback_start": int(fallback_start),
        "shape_policy": "reject_only",
        "canonical_preservation_bias": 1.0,
    }

    ref_gids = sorted(reference_profiles.per_gid_mean.keys())
    tgt_gids = sorted(
        target_profiles.per_gid_mean.keys(),
        key=lambda g: (target_profiles.per_gid_span.get(g, 0) * -1, g),
    )
    report.cam1_reference_gids = list(ref_gids)
    report.full_cam1_stable_gids = list(tgt_gids)

    cos_mat: Dict[int, Dict[int, float]] = {}
    for t in tgt_gids:
        tvec = target_profiles.per_gid_mean[t]
        cos_mat[t] = {}
        for r in ref_gids:
            rvec = reference_profiles.per_gid_mean[r]
            cos_mat[t][r] = _cosine(tvec, rvec)
    report.scores = cos_mat

    color_sim, color_nvp = _build_color_sim_matrix(target_profiles, reference_profiles)
    shape_sim = _build_shape_sim_matrix(target_profiles, reference_profiles)
    report.color_scores = {int(t): {int(r): float(v) for r, v in row.items()} for t, row in color_sim.items()}
    report.shape_scores = {int(t): {int(r): float(v) for r, v in row.items()} for t, row in shape_sim.items()}

    shape_informative: Dict[int, bool] = {}
    for t in cos_mat.keys():
        vals = [float(v) for v in (shape_sim.get(t, {}) or {}).values() if v is not None]
        shape_informative[t] = bool(len(vals) >= 2 and (max(vals) - min(vals)) >= 0.05)

    combined: Dict[int, Dict[int, float]] = {}
    for t, row in cos_mat.items():
        csim = color_sim.get(t, {})
        cnvp = color_nvp.get(t, {})
        ssim = shape_sim.get(t, {})
        s_info = shape_informative.get(t, True)
        combined[t] = {}
        for r, c in row.items():
            combined[t][r] = _combined_pair_score(
                osnet=float(c),
                color=float(csim.get(r, 0.0)),
                n_valid_color=int(cnvp.get(r, 0)),
                shape=float(ssim.get(r, 0.0)),
                shape_informative=s_info,
            )

    ref_gid_set = set(int(r) for r in ref_gids)
    canonical_targets = set(int(t) for t in tgt_gids if int(t) in ref_gid_set)
    claimed: set[int] = set()
    ordering = sorted(
        tgt_gids,
        key=lambda t: (
            0 if int(t) in canonical_targets else 1,
            -(cos_mat.get(t, {}).get(t, -1.0) if int(t) in canonical_targets else -1.0),
            -(max(combined.get(t, {}).values()) if combined.get(t) else -1.0),
            int(t),
        ),
    )

    for t in ordering:
        row = cos_mat.get(t, {})
        if not row:
            report.rejected[int(t)] = "no-ref-profiles"
            report.decision_details[int(t)] = {"decision": "rejected", "reason": "no-ref-profiles"}
            continue

        open_refs_osnet = [(r, c) for r, c in row.items() if r not in claimed]
        if not open_refs_osnet:
            report.unresolved[int(t)] = "all-refs-claimed"
            report.decision_details[int(t)] = {
                "decision": "unresolved",
                "reason": "all-refs-claimed",
                "shape_downweighted": True,
                "shape_ignored": True,
            }
            continue

        open_refs_combined = [(r, combined.get(t, {}).get(r, c)) for r, c in open_refs_osnet]
        open_refs_combined.sort(key=lambda x: -x[1])
        best_ref = int(open_refs_combined[0][0])
        best_comb = float(open_refs_combined[0][1])
        second_ref = int(open_refs_combined[1][0]) if len(open_refs_combined) > 1 else -1
        second_comb = float(open_refs_combined[1][1]) if len(open_refs_combined) > 1 else -1.0
        best_cos = float(row.get(best_ref, 0.0))
        second_cos = float(row.get(second_ref, -1.0)) if second_ref >= 0 else -1.0
        osnet_margin = float(best_cos - second_cos)
        top1_top2_margin = float(osnet_margin)
        color_best = float(color_sim.get(t, {}).get(best_ref, 0.0))
        color_second = float(color_sim.get(t, {}).get(second_ref, 0.0)) if second_ref >= 0 else 0.0
        color_margin = float(color_best - color_second)
        n_valid_color = int(color_nvp.get(t, {}).get(best_ref, 0))
        shape_best = float(shape_sim.get(t, {}).get(best_ref, 0.0))
        combined_margin = float(best_comb - second_comb) if second_comb >= 0.0 else float(best_comb)
        secondary_support = float(color_best)
        secondary_margin = float(color_margin)

        q = _source_quality_summary(target_profiles, int(t))
        low_quality = bool(q.get("low_quality", 0.0) > 0.5)
        ambiguous = bool(
            second_cos > 0.0
            and osnet_margin < max(0.018, 0.55 * float(min_margin))
            and best_cos < (float(min_cos) + 0.08)
        )

        self_cosine: Optional[float] = float(row.get(int(t), 0.0)) if int(t) in row else None
        self_available = bool(int(t) in row and int(t) not in claimed)
        other_open = sorted(
            [(int(r), float(v)) for r, v in row.items() if int(r) not in claimed and int(r) != int(t)],
            key=lambda x: -x[1],
        )
        best_other_cos = float(other_open[0][1]) if other_open else None
        self_margin = (
            float(self_cosine - best_other_cos)
            if (self_cosine is not None and best_other_cos is not None)
            else None
        )
        self_separation_gap = (
            float(best_other_cos - self_cosine)
            if (self_cosine is not None and best_other_cos is not None)
            else None
        )
        osnet_prefers_self = (
            bool(self_margin >= 0.0)
            if self_margin is not None
            else None
        )
        color_on_self = float(color_sim.get(t, {}).get(int(t), 0.0)) if int(t) in row else None
        color_agrees_with_self = (
            bool(color_on_self + 0.015 >= color_best)
            if color_on_self is not None
            else None
        )

        is_existing_canonical = bool(int(t) in canonical_targets)
        remap_away_from_self = bool(is_existing_canonical and self_available and best_ref != int(t))
        clear_other_win = bool(
            remap_away_from_self
            and best_cos >= float(min_cos) + 0.03
            and osnet_margin >= max(float(min_margin) + 0.02, 0.05)
            and combined_margin >= max(0.03, 0.65 * float(min_margin))
            and (self_cosine is None or (best_cos - float(self_cosine)) >= 0.04)
            and secondary_support >= 0.56
            and not ambiguous
            and not low_quality
        )
        near_miss_self_rescue = bool(
            is_existing_canonical
            and self_available
            and self_cosine is not None
            and self_margin is not None
            and self_margin >= 0.003
            and osnet_prefers_self is True
            and color_agrees_with_self is True
            and float(q.get("prototype_quality_mean", 0.0)) >= 0.50
            and float(q.get("prototype_kept_count", 0.0)) >= 5.0
            and float(self_cosine) >= (float(min_cos) - 0.08)
            and float(self_margin) < max(float(min_margin), 0.05)
        )

        canonical_preservation_block = False
        ambiguity_block = False
        low_quality_block = False
        rescue_path: Optional[str] = None
        if is_existing_canonical and self_available:
            if near_miss_self_rescue:
                claimed.add(int(t))
                report.mapping[int(t)] = int(t)
                rescue_path = "near-miss-self-rescue"
                if not hasattr(report, "rescue_reasons"):
                    report.rescue_reasons = {}  # type: ignore[attr-defined]
                report.rescue_reasons[int(t)] = (  # type: ignore[attr-defined]
                    f"near-miss-self-rescue self={self_cosine:.3f} "
                    f"self_margin={self_margin:.3f} color_self={color_on_self:.3f}"
                )
            elif best_ref == int(t):
                claimed.add(int(t))
                report.mapping[int(t)] = int(t)
            elif remap_away_from_self and not clear_other_win:
                canonical_preservation_block = True
                ambiguity_block = bool(ambiguous)
                low_quality_block = bool(low_quality)
                claimed.add(int(t))
                report.mapping[int(t)] = int(t)

        if int(t) in report.mapping:
            report.decision_details[int(t)] = {
                "decision": "mapped-self" if int(report.mapping[int(t)]) == int(t) else "mapped",
                "decision_reason": (
                    rescue_path
                    or ("canonical-preservation-bias" if canonical_preservation_block else "self-competitive")
                ),
                "best_ref": int(best_ref),
                "best_osnet": float(best_cos),
                "second_ref": int(second_ref) if second_ref >= 0 else None,
                "second_osnet": float(second_cos) if second_ref >= 0 else None,
                "top1_top2_margin": float(top1_top2_margin),
                "combined_margin": float(combined_margin),
                "self_cosine": float(self_cosine) if self_cosine is not None else None,
                "self_separation_gap": float(self_separation_gap) if self_separation_gap is not None else None,
                "self_preferred": osnet_prefers_self,
                "color_agrees_with_self": color_agrees_with_self,
                "canonical_preservation_blocked_remap": bool(canonical_preservation_block),
                "ambiguity_block_fired": bool(ambiguity_block),
                "low_prototype_quality_blocked_remap": bool(low_quality_block),
                "rescue_path_fired": rescue_path,
                "shape_downweighted": True,
                "shape_ignored": not bool(shape_informative.get(int(t), False)),
                "shape_top1": float(shape_best),
                "prototype_quality_mean": float(q.get("prototype_quality_mean", 0.0)),
                "prototype_coherence_mean": float(q.get("prototype_coherence_mean", 0.0)),
                "prototype_kept_count": int(q.get("prototype_kept_count", 0.0)),
                "row_count": int(q.get("rows", 0.0)),
                "frame_span": int(q.get("span", 0.0)),
            }
            continue

        shape_sanity_reject = bool(
            shape_informative.get(int(t), False)
            and shape_best < 0.32
            and best_cos < (float(min_cos) + 0.05)
        )

        if (
            n_valid_color >= 2
            and color_best < 0.44
            and secondary_margin < -0.07
            and best_cos < (float(min_cos) + 0.06)
        ):
            reason = (
                f"color-veto (osnet_top1={best_cos:.3f} color_top1={color_best:.3f} "
                f"color_margin={secondary_margin:.3f})"
            )
            report.unresolved[int(t)] = reason
            report.decision_details[int(t)] = {
                "decision": "unresolved",
                "decision_reason": reason,
                "best_ref": int(best_ref),
                "best_osnet": float(best_cos),
                "top1_top2_margin": float(top1_top2_margin),
                "self_cosine": float(self_cosine) if self_cosine is not None else None,
                "self_separation_gap": float(self_separation_gap) if self_separation_gap is not None else None,
                "self_preferred": osnet_prefers_self,
                "color_agrees_with_self": color_agrees_with_self,
                "shape_downweighted": True,
                "shape_ignored": not bool(shape_informative.get(int(t), False)),
                "shape_top1": float(shape_best),
            }
            continue

        passed_strict = (
            best_cos >= float(min_cos)
            and (osnet_margin >= float(min_margin) or second_cos <= 0.0)
            and combined_margin >= max(0.022, 0.45 * float(min_margin))
            and secondary_support >= 0.50
            and not ambiguous
            and not shape_sanity_reject
            and (
                not low_quality
                or (
                    best_cos >= float(min_cos) + 0.06
                    and osnet_margin >= max(float(min_margin) + 0.02, 0.05)
                    and secondary_support >= 0.58
                )
            )
        )

        if not passed_strict:
            reason = ""
            if ambiguous:
                reason = (
                    f"ambiguity-block (top1={best_cos:.3f} top2={second_cos:.3f} "
                    f"margin={osnet_margin:.3f})"
                )
                report.unresolved[int(t)] = reason
            elif low_quality and best_ref != int(t):
                reason = (
                    f"low-prototype-quality-block (top1={best_cos:.3f} "
                    f"quality={q.get('prototype_quality_mean', 0.0):.3f} "
                    f"coherence={q.get('prototype_coherence_mean', 0.0):.3f})"
                )
                report.unresolved[int(t)] = reason
            elif shape_sanity_reject:
                reason = (
                    f"shape-sanity-reject (shape_top1={shape_best:.3f} "
                    f"osnet_top1={best_cos:.3f})"
                )
                report.unresolved[int(t)] = reason
            elif best_cos < float(min_cos):
                reason = (
                    f"below-min-cos ({best_cos:.3f} < {min_cos:.3f}) "
                    f"color_top1={color_best:.3f}"
                )
                report.rejected[int(t)] = reason
            elif osnet_margin < float(min_margin) and second_cos > 0.0:
                reason = (
                    f"below-min-margin (top1={best_cos:.3f} top2={second_cos:.3f} "
                    f"margin={osnet_margin:.3f} < {min_margin:.3f}) "
                    f"color_top1={color_best:.3f}"
                )
                report.unresolved[int(t)] = reason
            else:
                reason = (
                    f"conservative-block (combined_margin={combined_margin:.3f} "
                    f"color_top1={color_best:.3f})"
                )
                report.unresolved[int(t)] = reason

            report.decision_details[int(t)] = {
                "decision": "unresolved" if int(t) in report.unresolved else "rejected",
                "decision_reason": reason,
                "best_ref": int(best_ref),
                "best_osnet": float(best_cos),
                "second_ref": int(second_ref) if second_ref >= 0 else None,
                "second_osnet": float(second_cos) if second_ref >= 0 else None,
                "top1_top2_margin": float(top1_top2_margin),
                "combined_margin": float(combined_margin),
                "self_cosine": float(self_cosine) if self_cosine is not None else None,
                "self_separation_gap": float(self_separation_gap) if self_separation_gap is not None else None,
                "self_preferred": osnet_prefers_self,
                "color_agrees_with_self": color_agrees_with_self,
                "canonical_preservation_blocked_remap": False,
                "ambiguity_block_fired": bool(ambiguous),
                "low_prototype_quality_blocked_remap": bool(low_quality and best_ref != int(t)),
                "rescue_path_fired": None,
                "shape_downweighted": True,
                "shape_ignored": not bool(shape_informative.get(int(t), False)),
                "shape_top1": float(shape_best),
                "prototype_quality_mean": float(q.get("prototype_quality_mean", 0.0)),
                "prototype_coherence_mean": float(q.get("prototype_coherence_mean", 0.0)),
                "prototype_kept_count": int(q.get("prototype_kept_count", 0.0)),
                "row_count": int(q.get("rows", 0.0)),
                "frame_span": int(q.get("span", 0.0)),
            }
            continue

        claimed.add(int(best_ref))
        report.mapping[int(t)] = int(best_ref)
        report.decision_details[int(t)] = {
            "decision": "mapped",
            "decision_reason": "strict-accept",
            "best_ref": int(best_ref),
            "best_osnet": float(best_cos),
            "second_ref": int(second_ref) if second_ref >= 0 else None,
            "second_osnet": float(second_cos) if second_ref >= 0 else None,
            "top1_top2_margin": float(top1_top2_margin),
            "combined_margin": float(combined_margin),
            "self_cosine": float(self_cosine) if self_cosine is not None else None,
            "self_separation_gap": float(self_separation_gap) if self_separation_gap is not None else None,
            "self_preferred": osnet_prefers_self,
            "color_agrees_with_self": color_agrees_with_self,
            "canonical_preservation_blocked_remap": False,
            "ambiguity_block_fired": False,
            "low_prototype_quality_blocked_remap": False,
            "rescue_path_fired": None,
            "shape_downweighted": True,
            "shape_ignored": not bool(shape_informative.get(int(t), False)),
            "shape_top1": float(shape_best),
            "prototype_quality_mean": float(q.get("prototype_quality_mean", 0.0)),
            "prototype_coherence_mean": float(q.get("prototype_coherence_mean", 0.0)),
            "prototype_kept_count": int(q.get("prototype_kept_count", 0.0)),
            "row_count": int(q.get("rows", 0.0)),
            "frame_span": int(q.get("span", 0.0)),
        }

    for t in [int(x) for x in list(report.rejected.keys())]:
        if int(t) in canonical_targets:
            report.unresolved.setdefault(int(t), str(report.rejected.get(int(t), "")))
            report.rejected.pop(int(t), None)

    rejected_tgts = sorted(
        (int(t) for t in tgt_gids if int(t) in report.rejected and int(t) not in report.unresolved),
        key=lambda g: (tgt_gids.index(g), g),
    )
    next_id = int(fallback_start)
    for t in rejected_tgts:
        report.fallback_assignments[int(t)] = int(next_id)
        next_id += 1
    return report


# ---------------------------------------------------------------------------
# Apply plan to tracks CSV
# ---------------------------------------------------------------------------


def apply_mapping_to_csv(
    *,
    tracks_csv: Path,
    report: MatchReport,
    also_write_backup: bool = True,
    only_touch_stable: bool = True,
) -> dict:
    """Rewrite the tracks CSV with new canonical IDs according to ``report``.

    - Old stable gids matched to CAM1 reference get CAM1's canonical ID.
    - Old stable gids rejected from matching get fallback IDs (>= fallback_start).
    - Old gid 0 (ghosts/unassigned) is preserved as 0.
    - Any old stable gid not listed in ``report`` is left alone (safety).

    Returns a summary dict with row counts per new id and a list of changes.
    """
    tracks_csv = Path(tracks_csv)

    fieldnames, by_gid = _load_tracks_csv(tracks_csv)
    # Build old->new mapping.
    new_id: Dict[int, int] = {}
    raw_map: Dict[int, int] = {int(k): int(v) for k, v in report.mapping.items()}
    raw_fb: Dict[int, int] = {int(k): int(v) for k, v in report.fallback_assignments.items()}
    # Safety guard:
    # If a canonical source ID is still unresolved/unmapped, do not allow another
    # source to be remapped onto that same canonical destination. Otherwise we
    # create same-frame ID collisions that get force-zeroed downstream.
    canonical_refs = set(int(x) for x in (report.cam1_reference_gids or []))
    stable_src = set(int(x) for x in (report.full_cam1_stable_gids or []))
    unresolved_reserved = set(
        int(t)
        for t in stable_src
        if int(t) in canonical_refs and int(t) not in raw_map and int(t) not in raw_fb
    )
    if unresolved_reserved:
        blocked_src: List[int] = []
        for src, dst in list(raw_map.items()):
            if int(src) != int(dst) and int(dst) in unresolved_reserved:
                raw_map.pop(int(src), None)
                blocked_src.append(int(src))
        for src in blocked_src:
            report.mapping.pop(int(src), None)
            raw_fb.pop(int(src), None)
            report.unresolved.setdefault(
                int(src),
                "blocked-by-unresolved-canonical-reservation(apply-guard)",
            )
            detail = report.decision_details.setdefault(int(src), {})
            detail["decision"] = "unresolved"
            detail["decision_reason"] = "blocked-by-unresolved-canonical-reservation"
            detail["canonical_preservation_blocked_remap"] = True
            detail["ambiguity_block_fired"] = True
            detail["shape_downweighted"] = True
            detail["shape_ignored"] = True
            detail["unresolved_canonical_reservation_blocked"] = True

    new_id.update(raw_map)
    new_id.update(raw_fb)
    # Preserve 0 explicitly.
    new_id[0] = 0
    if not new_id:
        return {"changed_rows": 0, "total_rows": 0}

    if also_write_backup:
        backup = tracks_csv.with_suffix(tracks_csv.suffix + ".pre-cam1-anchor.csv")
        if not backup.exists():
            shutil.copy2(tracks_csv, backup)

    # Rewrite by streaming through the original CSV.
    rows_out: List[Dict[str, str]] = []
    changed = 0
    total = 0
    with tracks_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or fieldnames)
        for raw in reader:
            total += 1
            try:
                old_gid = int(raw["global_id"])
            except Exception:
                rows_out.append(raw)
                continue
            if old_gid in new_id:
                mapped = new_id[old_gid]
                if mapped != old_gid:
                    raw["global_id"] = str(int(mapped))
                    changed += 1
            elif not only_touch_stable:
                # aggressive mode - also zero out unseen gids (not recommended).
                raw["global_id"] = "0"
                changed += 1
            rows_out.append(raw)

    with tracks_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_out:
            writer.writerow(r)

    # Compact summary.
    per_new_id: Dict[int, int] = defaultdict(int)
    for r in rows_out:
        try:
            g = int(r["global_id"])
        except Exception:
            continue
        per_new_id[g] += 1
    return {
        "changed_rows": changed,
        "total_rows": total,
        "rows_per_new_id": dict(sorted(per_new_id.items())),
        "mapping_applied": {int(k): int(v) for k, v in new_id.items() if int(v) != int(k)},
    }


# ---------------------------------------------------------------------------
# Top-level convenience
# ---------------------------------------------------------------------------


def align_full_cam1_to_cam1(
    *,
    cam1_video: Path,
    cam1_tracks_csv: Path,
    full_cam1_video: Path,
    full_cam1_tracks_csv: Path,
    cam1_canonical_gids: Sequence[int] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
    samples_per_gid: int = 18,
    min_cos: float = 0.78,
    min_margin: float = 0.07,
    fallback_start: int = 12,
    stable_min_rows: int = 30,
    stable_min_span: int = 32,
    reid_weights_path: Optional[str] = None,
    device: str = "cpu",
    apply: bool = True,
    report_path: Optional[Path] = None,
    matcher: str = "strong_with_fallback",
    strong_min_mean_cos: float = 0.82,
    strong_min_max_cos: float = 0.88,
    strong_min_sample_vote_share: float = 0.55,
    strong_min_margin: float = 0.04,
    strong_require_bidirectional: bool = True,
) -> MatchReport:
    """Full pipeline: build profiles for both videos, match, optionally rewrite CSV.

    ``matcher`` selects the matching strategy:
        - "strong"               : per-sample voting + bidirectional consensus
        - "mean_greedy"          : mean-cosine greedy bipartite (old behavior)
        - "strong_with_fallback" : strong matcher first; every target that
          strong rejects is then evaluated against the mean-greedy matcher on
          the REMAINING unclaimed references at its own thresholds.
          This is the default because it recovers borderline-ambiguous people
          without sacrificing the high-confidence accepts.
    """
    cam1_profiles = build_profiles(
        video_path=cam1_video,
        tracks_csv=cam1_tracks_csv,
        source_label="cam1",
        gid_filter=cam1_canonical_gids,
        samples_per_gid=samples_per_gid,
        stable_min_rows=1,
        stable_min_span=1,
        reid_weights_path=reid_weights_path,
        device=device,
    )
    # Source (full_cam1) profiles: raise the per-sample quality floor and
    # start with a wider keep ratio so the second-pass coherence pruning
    # inside `_select_clean_vectors` can drop contaminated samples without
    # dropping coverage. This specifically targets the "prototype ambiguity"
    # failure for canonicals 7, 9, 10, 11 — their source fragments have noisy
    # samples that flatten the OSNet top1/top2 margin.
    full_profiles = build_profiles(
        video_path=full_cam1_video,
        tracks_csv=full_cam1_tracks_csv,
        source_label="full_cam1",
        gid_filter=None,
        samples_per_gid=samples_per_gid,
        stable_min_rows=stable_min_rows,
        stable_min_span=stable_min_span,
        reid_weights_path=reid_weights_path,
        device=device,
        # NOTE: 0.50 over-pruned source prototypes (kept_count dropped to
        # 11-12) and caused canonical 6 to lose its mapping after anchor.
        # Back off slightly; coherence-floor pruning inside
        # `_select_clean_vectors` (cosine-to-centroid >= 0.72) still
        # protects against contaminated samples, and prototype coherence
        # in the latest run is already good (0.87-0.93).
        min_sample_quality=0.46,
        clean_keep_ratio=0.88,
        clean_min_keep=6,
    )

    if matcher == "mean_greedy":
        report = match_profiles(
            target_profiles=full_profiles,
            reference_profiles=cam1_profiles,
            min_cos=min_cos,
            min_margin=min_margin,
            fallback_start=fallback_start,
        )
    elif matcher == "strong":
        report = match_profiles_strong(
            target_profiles=full_profiles,
            reference_profiles=cam1_profiles,
            min_mean_cos=strong_min_mean_cos,
            min_max_cos=strong_min_max_cos,
            min_sample_vote_share=strong_min_sample_vote_share,
            min_margin=strong_min_margin,
            require_bidirectional=strong_require_bidirectional,
            fallback_start=fallback_start,
        )
    elif matcher == "strong_with_fallback":
        strong_report = match_profiles_strong(
            target_profiles=full_profiles,
            reference_profiles=cam1_profiles,
            min_mean_cos=strong_min_mean_cos,
            min_max_cos=strong_min_max_cos,
            min_sample_vote_share=strong_min_sample_vote_share,
            min_margin=strong_min_margin,
            require_bidirectional=strong_require_bidirectional,
            fallback_start=fallback_start,
        )
        # Build a residual ProfileSet of un-accepted targets and run
        # mean-greedy on the references that weren't claimed by strong.
        claimed_refs = set(strong_report.mapping.values())
        residual_tgt = ProfileSet(source=full_profiles.source)
        for t in full_profiles.per_gid_mean:
            if t in strong_report.mapping:
                continue
            residual_tgt.per_gid_mean[t] = full_profiles.per_gid_mean[t]
            residual_tgt.per_gid_samples[t] = full_profiles.per_gid_samples.get(t, [])
            residual_tgt.per_gid_rows[t] = full_profiles.per_gid_rows.get(t, 0)
            residual_tgt.per_gid_span[t] = full_profiles.per_gid_span.get(t, 0)
            if t in full_profiles.per_gid_upper_hue_hist:
                residual_tgt.per_gid_upper_hue_hist[t] = full_profiles.per_gid_upper_hue_hist[t]
            if t in full_profiles.per_gid_lower_hue_hist:
                residual_tgt.per_gid_lower_hue_hist[t] = full_profiles.per_gid_lower_hue_hist[t]
            if t in full_profiles.per_gid_upper_dom:
                residual_tgt.per_gid_upper_dom[t] = full_profiles.per_gid_upper_dom[t]
            if t in full_profiles.per_gid_lower_dom:
                residual_tgt.per_gid_lower_dom[t] = full_profiles.per_gid_lower_dom[t]
            if t in full_profiles.per_gid_shape:
                residual_tgt.per_gid_shape[t] = full_profiles.per_gid_shape[t]
        residual_ref = ProfileSet(source=cam1_profiles.source)
        for r, mean in cam1_profiles.per_gid_mean.items():
            if r in claimed_refs:
                continue
            residual_ref.per_gid_mean[r] = mean
            residual_ref.per_gid_samples[r] = cam1_profiles.per_gid_samples.get(r, [])
            residual_ref.per_gid_rows[r] = cam1_profiles.per_gid_rows.get(r, 0)
            residual_ref.per_gid_span[r] = cam1_profiles.per_gid_span.get(r, 0)
            if r in cam1_profiles.per_gid_upper_hue_hist:
                residual_ref.per_gid_upper_hue_hist[r] = cam1_profiles.per_gid_upper_hue_hist[r]
            if r in cam1_profiles.per_gid_lower_hue_hist:
                residual_ref.per_gid_lower_hue_hist[r] = cam1_profiles.per_gid_lower_hue_hist[r]
            if r in cam1_profiles.per_gid_upper_dom:
                residual_ref.per_gid_upper_dom[r] = cam1_profiles.per_gid_upper_dom[r]
            if r in cam1_profiles.per_gid_lower_dom:
                residual_ref.per_gid_lower_dom[r] = cam1_profiles.per_gid_lower_dom[r]
            if r in cam1_profiles.per_gid_shape:
                residual_ref.per_gid_shape[r] = cam1_profiles.per_gid_shape[r]
        fallback_report = match_profiles(
            target_profiles=residual_tgt,
            reference_profiles=residual_ref,
            min_cos=min_cos,
            min_margin=min_margin,
            fallback_start=fallback_start,
        )
        # Merge: strong mapping wins, fallback fills in.
        report = MatchReport()
        report.params = {
            "mode": "strong_with_fallback",
            "strong": dict(strong_report.params),
            "fallback": dict(fallback_report.params),
        }
        report.cam1_reference_gids = list(strong_report.cam1_reference_gids)
        report.full_cam1_stable_gids = list(strong_report.full_cam1_stable_gids)
        report.scores = dict(strong_report.scores)
        for t, row in fallback_report.scores.items():
            report.scores.setdefault(t, row)
        report.color_scores = dict(strong_report.color_scores or {})
        for t, row in (fallback_report.color_scores or {}).items():
            report.color_scores.setdefault(t, {}).update(row)
        report.shape_scores = dict(strong_report.shape_scores or {})
        for t, row in (fallback_report.shape_scores or {}).items():
            report.shape_scores.setdefault(t, {}).update(row)
        report.mapping.update(strong_report.mapping)
        report.mapping.update(fallback_report.mapping)
        report.unresolved.update(getattr(strong_report, "unresolved", {}) or {})
        report.unresolved.update(getattr(fallback_report, "unresolved", {}) or {})
        report.decision_details.update(getattr(strong_report, "decision_details", {}) or {})
        report.decision_details.update(getattr(fallback_report, "decision_details", {}) or {})
        report.fallback_assignments.update(getattr(strong_report, "fallback_assignments", {}) or {})
        report.fallback_assignments.update(getattr(fallback_report, "fallback_assignments", {}) or {})
        strong_rescue = getattr(strong_report, "rescue_reasons", {}) or {}
        fallback_rescue = getattr(fallback_report, "rescue_reasons", {}) or {}
        if strong_rescue or fallback_rescue:
            report.rescue_reasons = {}  # type: ignore[attr-defined]
            report.rescue_reasons.update(strong_rescue)  # type: ignore[attr-defined]
            report.rescue_reasons.update(fallback_rescue)  # type: ignore[attr-defined]
        # Missing-canonical recovery pass: loosen only the recovery side a bit,
        # and only for still-unclaimed canonical refs.
        recovered_map, recovered_notes = _recover_missing_reference_matches(
            target_profiles=full_profiles,
            reference_profiles=cam1_profiles,
            current_mapping=report.mapping,
            min_cos=min_cos,
            min_margin=min_margin,
        )
        if recovered_map:
            report.mapping.update({int(t): int(r) for t, r in recovered_map.items()})
            if not hasattr(report, "rescue_reasons"):
                report.rescue_reasons = {}  # type: ignore[attr-defined]
            for t, msg in recovered_notes.items():
                report.rescue_reasons[int(t)] = str(msg)  # type: ignore[attr-defined]
                report.decision_details.setdefault(int(t), {})
                report.decision_details[int(t)]["decision"] = "mapped"
                report.decision_details[int(t)]["decision_reason"] = "missing-canonical-recovery"
                report.decision_details[int(t)]["rescue_path_fired"] = "missing-canonical-recovery"
                report.decision_details[int(t)]["shape_downweighted"] = True
                report.unresolved.pop(int(t), None)
                report.rejected.pop(int(t), None)
                report.fallback_assignments.pop(int(t), None)
        # Rejected = rejected by BOTH passes.
        for t, reason in strong_report.rejected.items():
            if t in report.mapping:
                continue
            reason2 = fallback_report.rejected.get(t, "")
            report.rejected[t] = f"strong: {reason}" + (f" | fallback: {reason2}" if reason2 else "")
        # Reservation guard:
        # Keep unresolved canonical sources reserved so no other source can be
        # remapped onto their canonical ID in this pass.
        canonical_refs = set(int(r) for r in cam1_profiles.per_gid_mean.keys())
        stable_sources = set(int(x) for x in strong_report.full_cam1_stable_gids)
        unresolved_reserved = {
            int(t)
            for t in stable_sources
            if int(t) in canonical_refs and int(t) not in report.mapping
        }
        if unresolved_reserved:
            blocked_pairs: List[Tuple[int, int]] = []
            for src, dst in sorted(list(report.mapping.items())):
                src_i = int(src)
                dst_i = int(dst)
                if src_i == dst_i:
                    continue
                if src_i not in stable_sources:
                    continue
                if dst_i in unresolved_reserved:
                    blocked_pairs.append((src_i, dst_i))
            for src_i, dst_i in blocked_pairs:
                report.mapping.pop(int(src_i), None)
                report.fallback_assignments.pop(int(src_i), None)
                prev_reason = str(report.unresolved.get(int(src_i), "")).strip()
                block_reason = (
                    f"blocked-by-unresolved-canonical-reservation(src={int(src_i)},dst={int(dst_i)})"
                )
                if not prev_reason:
                    report.unresolved[int(src_i)] = block_reason
                elif block_reason not in prev_reason:
                    report.unresolved[int(src_i)] = f"{prev_reason} | {block_reason}"
                detail = report.decision_details.setdefault(int(src_i), {})
                detail["decision"] = "unresolved"
                detail["decision_reason"] = "blocked-by-unresolved-canonical-reservation"
                detail["canonical_preservation_blocked_remap"] = True
                detail["ambiguity_block_fired"] = True
                detail["shape_downweighted"] = True
                detail["shape_ignored"] = True
                detail["unresolved_canonical_reservation_blocked"] = True
        # Canonical safety net: preserve any still-unmapped canonical source on
        # its own ID when the canonical ref is still unclaimed.
        claimed_refs_final = set(int(r) for r in report.mapping.values())
        for t in [int(x) for x in strong_report.full_cam1_stable_gids]:
            if int(t) in report.mapping:
                continue
            if int(t) in canonical_refs and int(t) not in claimed_refs_final:
                report.mapping[int(t)] = int(t)
                claimed_refs_final.add(int(t))
                report.unresolved.pop(int(t), None)
                report.rejected.pop(int(t), None)
                report.fallback_assignments.pop(int(t), None)
                report.decision_details[int(t)] = {
                    "decision": "mapped-self",
                    "decision_reason": "canonical-preservation-final-safety-net",
                    "canonical_preservation_blocked_remap": True,
                    "ambiguity_block_fired": True,
                    "low_prototype_quality_blocked_remap": False,
                    "shape_downweighted": True,
                    "shape_ignored": True,
                }
                if not hasattr(report, "rescue_reasons"):
                    report.rescue_reasons = {}  # type: ignore[attr-defined]
                report.rescue_reasons[int(t)] = (  # type: ignore[attr-defined]
                    "canonical-preservation-final-safety-net"
                )

        # Assign fallback IDs only to still-unmapped, non-canonical, hard rejects.
        unmapped = [t for t in strong_report.full_cam1_stable_gids if t not in report.mapping]
        # Already-used fallback IDs from sub-reports:
        used_fallback = set(int(x) for x in report.fallback_assignments.values())
        next_id = int(fallback_start)
        for t in unmapped:
            if int(t) in canonical_refs:
                report.unresolved.setdefault(int(t), "canonical-preservation-unresolved")
                continue
            if int(t) in report.unresolved:
                continue
            if int(t) not in report.rejected:
                report.unresolved.setdefault(int(t), "conservative-unresolved")
                continue
            while next_id in used_fallback:
                next_id += 1
            report.fallback_assignments[int(t)] = int(next_id)
            report.decision_details.setdefault(int(t), {})
            report.decision_details[int(t)]["decision"] = "fallback"
            report.decision_details[int(t)]["decision_reason"] = "hard-reject-fallback"
            next_id += 1
    else:
        raise ValueError(f"unknown matcher: {matcher!r}")

    # Source-gid metadata used by downstream diagnostics.
    report.source_meta = {}
    for gid in sorted(full_profiles.per_gid_rows.keys()):
        report.source_meta[int(gid)] = {
            "rows": int(full_profiles.per_gid_rows.get(gid, 0)),
            "span": int(full_profiles.per_gid_span.get(gid, 0)),
            "prototype_sample_count": int(full_profiles.per_gid_proto_sample_count.get(gid, 0)),
            "prototype_kept_count": int(full_profiles.per_gid_proto_kept_count.get(gid, 0)),
            "prototype_quality_mean": float(full_profiles.per_gid_proto_quality_mean.get(gid, 0.0)),
            "prototype_coherence_mean": float(full_profiles.per_gid_proto_coherence_mean.get(gid, 0.0)),
        }

    if apply:
        summary = apply_mapping_to_csv(
            tracks_csv=full_cam1_tracks_csv,
            report=report,
            also_write_backup=True,
        )
        report.params["apply_summary"] = summary  # type: ignore[assignment]
    if report_path is not None:
        Path(report_path).write_text(report.as_json(), encoding="utf-8")
    return report


def build_anchor_failure_diagnostics(
    *,
    report: MatchReport,
    focus_canonical_ids: Sequence[int] = (7, 9, 10, 11),
    stable_min_rows: int = 30,
    after_anchor_rows: Optional[Mapping[int, int]] = None,
    after_convergence_rows: Optional[Mapping[int, int]] = None,
) -> Dict[str, object]:
    """Build an anchor bottleneck report for unresolved/weak canonicals."""

    def _int2(d: Optional[Mapping]) -> Dict[int, int]:
        if not d:
            return {}
        out: Dict[int, int] = {}
        for k, v in d.items():
            try:
                out[int(k)] = int(v)
            except Exception:
                continue
        return out

    def _float2d(d: Optional[Mapping]) -> Dict[int, Dict[int, float]]:
        out: Dict[int, Dict[int, float]] = {}
        if not d:
            return out
        for k, row in d.items():
            try:
                kk = int(k)
            except Exception:
                continue
            if not isinstance(row, Mapping):
                continue
            rr: Dict[int, float] = {}
            for rk, rv in row.items():
                try:
                    rr[int(rk)] = float(rv)
                except Exception:
                    continue
            out[kk] = rr
        return out

    mapping = _int2(report.mapping)
    fallback = _int2(report.fallback_assignments)
    rejected: Dict[int, str] = {int(k): str(v) for k, v in (report.rejected or {}).items()}
    unresolved: Dict[int, str] = {int(k): str(v) for k, v in (getattr(report, "unresolved", {}) or {}).items()}
    scores = _float2d(report.scores)
    color_scores = _float2d(getattr(report, "color_scores", {}) or {})
    shape_scores = _float2d(getattr(report, "shape_scores", {}) or {})
    decision_details: Dict[int, Dict[str, object]] = {}
    for k, v in (getattr(report, "decision_details", {}) or {}).items():
        try:
            kk = int(k)
        except Exception:
            continue
        if isinstance(v, Mapping):
            decision_details[kk] = dict(v)
    source_meta: Dict[int, Dict[str, float]] = {}
    for k, v in (getattr(report, "source_meta", {}) or {}).items():
        try:
            kk = int(k)
        except Exception:
            continue
        if not isinstance(v, Mapping):
            continue
        source_meta[kk] = {
            str(a): float(b)
            for a, b in v.items()
            if isinstance(b, (int, float, np.integer, np.floating))
        }

    after_anchor_rows_i = _int2(after_anchor_rows)
    after_convergence_rows_i = _int2(after_convergence_rows)
    mapped_refs = set(int(v) for v in mapping.values())
    candidate_sources = sorted(
        set(int(k) for k in scores.keys())
        | set(int(k) for k in rejected.keys())
        | set(int(k) for k in unresolved.keys())
        | set(int(k) for k in fallback.keys())
        | set(int(k) for k in mapping.keys())
    )

    def _classify(
        *,
        status: str,
        rows_anchor: int,
        rows_conv: int,
        source_rows: int,
        top1: Optional[float],
        margin: Optional[float],
        reject_reason: str,
        color_agree: Optional[float],
        shape_agree: Optional[float],
        close_candidate_count: int,
        coherence: Optional[float] = None,
        color_top1_unclaimed_ref: Optional[int] = None,
        osnet_top1_unclaimed_ref: Optional[int] = None,
        self_cosine: Optional[float] = None,
        self_separation_gap: Optional[float] = None,
        osnet_prefers_self: Optional[bool] = None,
        osnet_top1_unclaimed_cos: Optional[float] = None,
        osnet_top1_unclaimed_margin: Optional[float] = None,
    ) -> str:
        rr = (reject_reason or "").lower()
        if rows_anchor >= int(stable_min_rows) and rows_conv < int(stable_min_rows):
            return "merge-stage failure"
        if rows_conv > 0 and rows_conv < int(stable_min_rows):
            return "insufficient support"
        if source_rows > 0 and source_rows < int(stable_min_rows):
            return "insufficient support"
        # A low-coherence prototype is the most specific signal that the
        # source fragment's own samples are contaminated — the matcher cannot
        # separate anyone from this gid because the gid itself is blurry.
        if coherence is not None and coherence > 0.0 and coherence < 0.78:
            return "contaminated-prototype"
        if close_candidate_count >= 2:
            return "tracker fragmentation"

        # Low self-separation: OSNet prefers a DIFFERENT unclaimed ref
        # over this canonical's own self-ref by a real margin. The source
        # does not semantically "point at" this canonical, so forcing a
        # recovery here would risk wrong-binding.
        if (
            osnet_prefers_self is False
            and self_separation_gap is not None
            and self_separation_gap >= 0.02
            and (self_cosine or 0.0) < 0.70
        ):
            return "low-self-separation"

        # Color disagrees with OSNet on which unclaimed ref to pick — cue
        # conflict. Split by OSNet's own confidence:
        cue_conflict = (
            color_top1_unclaimed_ref is not None
            and osnet_top1_unclaimed_ref is not None
            and int(color_top1_unclaimed_ref) != int(osnet_top1_unclaimed_ref)
            and (color_agree or 0.0) >= 0.55
        )
        if cue_conflict:
            # OSNet is itself shaky → both cues weak, nothing to trust.
            if (osnet_top1_unclaimed_cos or 0.0) < 0.72 and (color_agree or 0.0) < 0.60:
                return "cue-disagreement-both-weak"
            # OSNet is clearly self-separated → trust OSNet, color is noise.
            if (
                (osnet_top1_unclaimed_cos or 0.0) >= 0.70
                and (osnet_top1_unclaimed_margin or 0.0) >= 0.02
            ):
                return "cue-disagreement-confident-osnet"
            return "cue-disagreement"

        # Insufficient evidence for any confident decision.
        if (
            (osnet_top1_unclaimed_cos or 0.0) < 0.68
            and (color_agree or 0.0) < 0.55
            and (coherence or 0.0) < 0.85
        ):
            return "insufficient-evidence"

        if "below-min-margin" in rr:
            return "prototype ambiguity"
        if margin is not None and top1 is not None and top1 >= 0.76 and margin < 0.03:
            return "prototype ambiguity"
        if "below-min-mean-cos" in rr:
            if (color_agree or 0.0) >= 0.62 and (shape_agree or 0.0) >= 0.90:
                return "prototype ambiguity"
            return "detector/crop quality"
        if top1 is not None and top1 < 0.72 and (color_agree or 0.0) < 0.58:
            return "detector/crop quality"
        if status == "missing":
            return "prototype ambiguity"
        if status == "resolved_weak":
            return "insufficient support"
        return "prototype ambiguity"

    diag_items: List[Dict[str, object]] = []
    for c in [int(x) for x in focus_canonical_ids]:
        rows_anchor = int(after_anchor_rows_i.get(c, 0))
        rows_conv = int(after_convergence_rows_i.get(c, 0))
        if rows_conv >= int(stable_min_rows):
            status = "resolved_stable"
        elif rows_conv > 0:
            status = "resolved_weak"
        else:
            status = "missing"

        scored_candidates: List[Tuple[float, int, int, int]] = []
        for s in candidate_sources:
            mapped_dst = mapping.get(int(s))
            if mapped_dst is not None and int(mapped_dst) != int(c):
                # Diagnostics should prefer fallback/unmapped source fragments
                # for unresolved canonicals, not already-claimed survivor IDs.
                continue
            sc = float(scores.get(int(s), {}).get(int(c), -1.0))
            if sc < 0.0:
                continue
            s_rows = int(source_meta.get(int(s), {}).get("rows", 0.0))
            s_span = int(source_meta.get(int(s), {}).get("span", 0.0))
            scored_candidates.append((sc, s_rows, s_span, int(s)))
        scored_candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)

        best_src: Optional[int] = scored_candidates[0][3] if scored_candidates else None
        best_score = float(scored_candidates[0][0]) if scored_candidates else None
        source_rows = int(scored_candidates[0][1]) if scored_candidates else 0
        source_span = int(scored_candidates[0][2]) if scored_candidates else 0
        close_candidate_count = 0
        if scored_candidates:
            b = float(scored_candidates[0][0])
            close_candidate_count = sum(
                1
                for sc, rr, _sp, _s in scored_candidates
                if rr >= int(stable_min_rows) and sc >= max(0.72, b - 0.02)
            )

        top1_ref = None
        top1_cos = None
        top2_ref = None
        top2_cos = None
        margin = None
        reject_reason = ""
        color_agree = None
        shape_agree = None
        mean_cosine = best_score
        fallback_gid = None
        mapped_to = None
        proto_count = 0
        proto_kept = 0
        proto_q = None
        proto_coh = None
        osnet_top1_unclaimed_ref = None
        osnet_top1_unclaimed_cos = None
        osnet_top2_unclaimed_ref = None
        osnet_top2_unclaimed_cos = None
        osnet_top1_unclaimed_margin = None
        color_top1_unclaimed_ref = None
        color_top1_unclaimed = None
        color_second_unclaimed = None
        color_top1_unclaimed_margin = None
        osnet_top1_ref_claimed_by: Optional[int] = None
        decision_meta: Dict[str, object] = {}

        if best_src is not None:
            row = dict(scores.get(int(best_src), {}))
            row_sorted = sorted(row.items(), key=lambda kv: kv[1], reverse=True)
            if row_sorted:
                top1_ref = int(row_sorted[0][0])
                top1_cos = float(row_sorted[0][1])
            if len(row_sorted) > 1:
                top2_ref = int(row_sorted[1][0])
                top2_cos = float(row_sorted[1][1])
            if top1_cos is not None:
                margin = float(top1_cos - (top2_cos if top2_cos is not None else 0.0))
            decision_meta = dict(decision_details.get(int(best_src), {}) or {})
            reject_reason = str(rejected.get(int(best_src), ""))
            if not reject_reason:
                reject_reason = str(unresolved.get(int(best_src), ""))
            color_val = color_scores.get(int(best_src), {}).get(int(c))
            shape_val = shape_scores.get(int(best_src), {}).get(int(c))
            color_agree = float(color_val) if color_val is not None else None
            shape_agree = float(shape_val) if shape_val is not None else None
            fallback_gid = int(fallback.get(int(best_src), 0)) if int(best_src) in fallback else None
            mapped_to = int(mapping.get(int(best_src), 0)) if int(best_src) in mapping else None
            meta = source_meta.get(int(best_src), {})
            proto_count = int(meta.get("prototype_sample_count", 0.0))
            proto_kept = int(meta.get("prototype_kept_count", 0.0))
            _pq = meta.get("prototype_quality_mean")
            proto_q = float(_pq) if _pq is not None else None
            _pc = meta.get("prototype_coherence_mean")
            proto_coh = float(_pc) if _pc is not None else None
            # Which ref does the OSNet top-1 point to, and was that ref
            # already claimed by a different source? If yes, the real
            # question is which UNCLAIMED ref this source prefers.
            if top1_ref is not None:
                for t_src, r_dst in mapping.items():
                    if int(r_dst) == int(top1_ref) and int(t_src) != int(best_src):
                        osnet_top1_ref_claimed_by = int(t_src)
                        break
            # Walk the unclaimed refs (not in `mapped_refs` already, and not
            # already used by any other source's proposed match) to get the
            # OSNet top-1/top-2 among them — this is the pool the recovery
            # path can actually reach.
            unclaimed_refs = {int(r) for r in row.keys() if int(r) not in mapped_refs}
            # `int(c)` is by definition unclaimed (this canonical is missing
            # or weak), so ensure it's considered.
            unclaimed_refs.add(int(c))
            row_unclaimed = sorted(
                [(int(rr_), float(vv)) for rr_, vv in row.items() if int(rr_) in unclaimed_refs],
                key=lambda x: -x[1],
            )
            if row_unclaimed:
                osnet_top1_unclaimed_ref = int(row_unclaimed[0][0])
                osnet_top1_unclaimed_cos = float(row_unclaimed[0][1])
            if len(row_unclaimed) > 1:
                osnet_top2_unclaimed_ref = int(row_unclaimed[1][0])
                osnet_top2_unclaimed_cos = float(row_unclaimed[1][1])
                osnet_top1_unclaimed_margin = float(osnet_top1_unclaimed_cos - osnet_top2_unclaimed_cos)
            elif osnet_top1_unclaimed_cos is not None:
                osnet_top1_unclaimed_margin = float(osnet_top1_unclaimed_cos)
            # Color's preferred unclaimed ref.
            crow = color_scores.get(int(best_src), {}) or {}
            crow_unclaimed = sorted(
                [(int(rr_), float(vv)) for rr_, vv in crow.items() if int(rr_) in unclaimed_refs],
                key=lambda x: -x[1],
            )
            if crow_unclaimed:
                color_top1_unclaimed_ref = int(crow_unclaimed[0][0])
                color_top1_unclaimed = float(crow_unclaimed[0][1])
            if len(crow_unclaimed) > 1:
                color_second_unclaimed = float(crow_unclaimed[1][1])
                if color_top1_unclaimed is not None:
                    color_top1_unclaimed_margin = float(color_top1_unclaimed - color_second_unclaimed)

        # Self-separation diagnostics: how does this source fragment
        # relate to its OWN canonical ref (`c`), and does OSNet actually
        # prefer that ref over the alternative unclaimed refs? This is
        # critical for distinguishing "OSNet could recover but thresholds
        # are too tight" from "OSNet semantically doesn't believe this
        # source is this canonical."
        self_cosine = None
        self_separation_gap = None
        osnet_prefers_self = None
        color_agrees_with = None
        if best_src is not None:
            row = scores.get(int(best_src), {}) or {}
            if int(c) in row:
                self_cosine = float(row[int(c)])
            # OSNet prefers self iff top1 among UNCLAIMED refs is this
            # canonical (unclaimed_refs set always contains `c`).
            if osnet_top1_unclaimed_ref is not None:
                osnet_prefers_self = bool(int(osnet_top1_unclaimed_ref) == int(c))
            # Positive gap => OSNet ranks some OTHER unclaimed ref above
            # self. Negative gap => OSNet already prefers self (recovery
            # path should be able to bind it if margins line up).
            if osnet_top1_unclaimed_cos is not None and self_cosine is not None:
                self_separation_gap = float(osnet_top1_unclaimed_cos - self_cosine)
            # Does color agree with self, with OSNet's top-1 unclaimed
            # pick, or with neither? Use a 0.02 tolerance around the
            # color-on-self vs color-top1-unclaimed comparison.
            if color_top1_unclaimed_ref is not None:
                crow_all = color_scores.get(int(best_src), {}) or {}
                color_on_self = float(crow_all.get(int(c), 0.0))
                color_on_osnet_top1 = (
                    float(crow_all.get(int(osnet_top1_unclaimed_ref), 0.0))
                    if osnet_top1_unclaimed_ref is not None
                    else 0.0
                )
                top_col = color_top1_unclaimed if color_top1_unclaimed is not None else 0.0
                if int(color_top1_unclaimed_ref) == int(c) or color_on_self + 0.02 >= top_col:
                    color_agrees_with = "self"
                elif (
                    osnet_top1_unclaimed_ref is not None
                    and (int(color_top1_unclaimed_ref) == int(osnet_top1_unclaimed_ref)
                         or color_on_osnet_top1 + 0.02 >= top_col)
                ):
                    color_agrees_with = "osnet-top1"
                else:
                    color_agrees_with = "neither"

        bottleneck = _classify(
            status=status,
            rows_anchor=rows_anchor,
            rows_conv=rows_conv,
            source_rows=source_rows,
            top1=top1_cos,
            margin=margin,
            reject_reason=reject_reason,
            color_agree=color_agree,
            shape_agree=shape_agree,
            close_candidate_count=close_candidate_count,
            coherence=proto_coh,
            color_top1_unclaimed_ref=color_top1_unclaimed_ref,
            osnet_top1_unclaimed_ref=osnet_top1_unclaimed_ref,
            self_cosine=self_cosine,
            self_separation_gap=self_separation_gap,
            osnet_prefers_self=osnet_prefers_self,
            osnet_top1_unclaimed_cos=osnet_top1_unclaimed_cos,
            osnet_top1_unclaimed_margin=osnet_top1_unclaimed_margin,
        )

        # Plain-English cue-health tags for at-a-glance triage.
        cue_health_tags: List[str] = []
        if proto_coh is not None:
            if proto_coh < 0.78:
                cue_health_tags.append("low-prototype-coherence")
            elif proto_coh >= 0.90:
                cue_health_tags.append("prototype-coherent")
        if top1_cos is not None and margin is not None and top1_cos >= 0.72 and margin < 0.02:
            cue_health_tags.append("osnet-top1-top2-tied")
        if osnet_top1_ref_claimed_by is not None:
            cue_health_tags.append(f"osnet-top1-ref-claimed-by-source-{osnet_top1_ref_claimed_by}")
        if (color_agree or 0.0) < 0.55:
            cue_health_tags.append("weak-color")
        if shape_agree is not None and shape_agree >= 0.97:
            cue_health_tags.append("saturated-shape")
        if (
            color_top1_unclaimed_ref is not None
            and osnet_top1_unclaimed_ref is not None
            and int(color_top1_unclaimed_ref) == int(c)
            and int(osnet_top1_unclaimed_ref) == int(c)
        ):
            cue_health_tags.append("both-cues-prefer-this-canonical")
        if (
            color_top1_unclaimed_ref is not None
            and osnet_top1_unclaimed_ref is not None
            and int(color_top1_unclaimed_ref) != int(osnet_top1_unclaimed_ref)
        ):
            cue_health_tags.append("cue-conflict-on-unclaimed-top1")
        if osnet_prefers_self is True:
            cue_health_tags.append("osnet-prefers-self")
        elif osnet_prefers_self is False:
            cue_health_tags.append("osnet-prefers-other-unclaimed")
        if self_separation_gap is not None and self_separation_gap >= 0.02 and (self_cosine or 0.0) < 0.70:
            cue_health_tags.append("low-self-separation")
        if color_agrees_with == "self":
            cue_health_tags.append("color-agrees-with-self")
        elif color_agrees_with == "osnet-top1":
            cue_health_tags.append("color-agrees-with-osnet-top1")
        elif color_agrees_with == "neither":
            cue_health_tags.append("color-agrees-with-neither")
        if bool(decision_meta.get("canonical_preservation_blocked_remap", False)):
            cue_health_tags.append("canonical-preservation-blocked-remap")
        if bool(decision_meta.get("ambiguity_block_fired", False)):
            cue_health_tags.append("ambiguity-block-fired")
        if bool(decision_meta.get("low_prototype_quality_blocked_remap", False)):
            cue_health_tags.append("low-prototype-quality-blocked-remap")
        if decision_meta.get("rescue_path_fired"):
            cue_health_tags.append("rescue-path-fired")
        if bool(decision_meta.get("shape_downweighted", False)):
            cue_health_tags.append("shape-downweighted")
        if bool(decision_meta.get("shape_ignored", False)):
            cue_health_tags.append("shape-ignored")

        decision_self_cos = decision_meta.get("self_cosine")
        decision_self_gap = decision_meta.get("self_separation_gap")
        decision_top_margin = decision_meta.get("top1_top2_margin")
        decision_self_pref = decision_meta.get("self_preferred")
        decision_color_self = decision_meta.get("color_agrees_with_self")
        try:
            decision_self_cos_f = float(decision_self_cos) if decision_self_cos is not None else None
        except Exception:
            decision_self_cos_f = None
        try:
            decision_self_gap_f = float(decision_self_gap) if decision_self_gap is not None else None
        except Exception:
            decision_self_gap_f = None
        try:
            decision_top_margin_f = float(decision_top_margin) if decision_top_margin is not None else None
        except Exception:
            decision_top_margin_f = None

        diag_items.append(
            {
                "canonical_id": int(c),
                "status": status,
                "rows_after_anchor": int(rows_anchor),
                "rows_after_convergence": int(rows_conv),
                "current_best_source_gid_candidate": int(best_src) if best_src is not None else None,
                "current_best_gid_after_anchor": int(fallback_gid) if fallback_gid is not None else (int(mapped_to) if mapped_to is not None else None),
                "top1_ref": int(top1_ref) if top1_ref is not None else None,
                "top1_cosine": float(top1_cos) if top1_cos is not None else None,
                "top2_ref": int(top2_ref) if top2_ref is not None else None,
                "top2_cosine": float(top2_cos) if top2_cos is not None else None,
                "margin": float(margin) if margin is not None else None,
                "top1_top2_margin": (
                    decision_top_margin_f if decision_top_margin_f is not None else (float(margin) if margin is not None else None)
                ),
                "mean_cosine": float(mean_cosine) if mean_cosine is not None else None,
                "color_agreement": float(color_agree) if color_agree is not None else None,
                "shape_agreement": float(shape_agree) if shape_agree is not None else None,
                "row_count": int(source_rows),
                "frame_span": int(source_span),
                "prototype_sample_count": int(proto_count),
                "prototype_kept_count": int(proto_kept),
                "prototype_quality_mean": float(proto_q) if proto_q is not None else None,
                "prototype_coherence_mean": float(proto_coh) if proto_coh is not None else None,
                "osnet_top1_unclaimed_ref": osnet_top1_unclaimed_ref,
                "osnet_top1_unclaimed_cos": osnet_top1_unclaimed_cos,
                "osnet_top2_unclaimed_ref": osnet_top2_unclaimed_ref,
                "osnet_top2_unclaimed_cos": osnet_top2_unclaimed_cos,
                "osnet_top1_unclaimed_margin": osnet_top1_unclaimed_margin,
                "osnet_top1_ref_claimed_by_source": osnet_top1_ref_claimed_by,
                "color_top1_unclaimed_ref": color_top1_unclaimed_ref,
                "color_top1_unclaimed": color_top1_unclaimed,
                "color_top1_unclaimed_margin": color_top1_unclaimed_margin,
                "self_cosine": (
                    decision_self_cos_f if decision_self_cos_f is not None else (float(self_cosine) if self_cosine is not None else None)
                ),
                "self_separation_gap": (
                    decision_self_gap_f if decision_self_gap_f is not None else (float(self_separation_gap) if self_separation_gap is not None else None)
                ),
                "osnet_prefers_self": (
                    decision_self_pref if decision_self_pref is not None else osnet_prefers_self
                ),
                "self_preferred_flag": (
                    decision_self_pref if decision_self_pref is not None else osnet_prefers_self
                ),
                "color_agrees_with": color_agrees_with,
                "color_agrees_with_self_flag": (
                    bool(decision_color_self)
                    if decision_color_self is not None
                    else (True if color_agrees_with == "self" else False if color_agrees_with is not None else None)
                ),
                "decision": decision_meta.get("decision"),
                "decision_reason": decision_meta.get("decision_reason"),
                "remap_blocked_by_canonical_preservation_bias": bool(
                    decision_meta.get("canonical_preservation_blocked_remap", False)
                ),
                "shape_ignored_or_downweighted": bool(
                    decision_meta.get("shape_downweighted", False) or decision_meta.get("shape_ignored", False)
                ),
                "shape_ignored": bool(decision_meta.get("shape_ignored", False)),
                "shape_downweighted": bool(decision_meta.get("shape_downweighted", False)),
                "rescue_path_fired": decision_meta.get("rescue_path_fired"),
                "ambiguity_block_fired": bool(decision_meta.get("ambiguity_block_fired", False)),
                "low_prototype_quality_blocked_remap": bool(
                    decision_meta.get("low_prototype_quality_blocked_remap", False)
                ),
                "reject_reason": reject_reason,
                "close_competitor_count": int(close_candidate_count),
                "likely_bottleneck_classification": bottleneck,
                "cue_health_tags": cue_health_tags,
            }
        )

    return {
        "focus_canonical_ids": [int(x) for x in focus_canonical_ids],
        "stable_min_rows": int(stable_min_rows),
        "mapped_canonical_ids": sorted(int(x) for x in mapped_refs),
        "canonicals": diag_items,
    }


__all__ = [
    "ReferenceAnchorDependencyError",
    "ProfileSet",
    "MatchReport",
    "build_profiles",
    "match_profiles",
    "match_profiles_strong",
    "build_anchor_failure_diagnostics",
    "apply_mapping_to_csv",
    "align_full_cam1_to_cam1",
]
