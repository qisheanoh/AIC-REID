from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Iterable
import numpy as np


@dataclass
class _IDState:
    gid: int
    feat: np.ndarray
    first_feat: np.ndarray
    prototypes: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=24))
    last_ts: Optional[float] = None
    n_obs: int = 1


@dataclass
class MatchCandidate:
    gid: int
    sim_fused: float
    sim_recent: float
    sim_first: float
    second_sim: float = -1.0
    margin: float = -1.0


class GlobalIDBank:
    """
    Global identity bank.
    Use this only after a local track is confirmed.
    """

    def __init__(
        self,
        hard_thresh: float = 0.82,
        soft_thresh: float = 0.75,
        margin: float = 0.06,
        ema: float = 0.92,
        min_update_sim: float = 0.80,
        enroll_reuse_thresh: float | None = None,
        enroll_protect_states: int = 0,
        max_prototypes: int = 24,
        prototype_weight: float = 0.35,
        prototype_min_delta: float = 0.03,
        prototype_topk: int = 4,
        observe_min_quality: float = 0.58,
        verbose: bool = False,
    ):
        self._states: List[_IDState] = []
        self.next_id: int = 1

        self.hard_thresh = float(hard_thresh)
        self.soft_thresh = float(soft_thresh)
        self.margin = float(margin)
        self.ema = float(ema)
        self.min_update_sim = float(min_update_sim)
        self.enroll_reuse_thresh = (
            float(enroll_reuse_thresh) if enroll_reuse_thresh is not None else None
        )
        self.enroll_protect_states = int(enroll_protect_states)
        self.max_prototypes = max(1, int(max_prototypes))
        self.prototype_weight = float(prototype_weight)
        self.prototype_min_delta = float(prototype_min_delta)
        self.prototype_topk = max(1, int(prototype_topk))
        self.observe_min_quality = float(observe_min_quality)
        self.verbose = bool(verbose)

    def reset(self) -> None:
        self._states = []
        self.next_id = 1

    def _normalize(self, f: np.ndarray) -> np.ndarray:
        f = f.astype(np.float32)
        return f / (np.linalg.norm(f) + 1e-12)

    def _cos(self, a: np.ndarray, b: np.ndarray) -> float:
        a = self._normalize(a)
        b = self._normalize(b)
        return float(np.dot(a, b))

    def _state_similarity(self, st: _IDState, f: np.ndarray) -> float:
        sim_main = self._cos(f, st.feat)
        if not st.prototypes:
            return sim_main

        proto_sims = sorted((self._cos(f, p) for p in st.prototypes), reverse=True)
        sim_proto_max = float(proto_sims[0])
        k = min(self.prototype_topk, len(proto_sims))
        sim_proto_topk = float(np.mean(proto_sims[:k]))
        blended = (1.0 - self.prototype_weight) * sim_main + self.prototype_weight * sim_proto_topk
        return float(max(blended, 0.96 * sim_proto_topk, 0.92 * sim_proto_max))

    def _state_scores(self, st: _IDState, f: np.ndarray) -> Tuple[float, float, float]:
        sim_recent = self._state_similarity(st, f)
        sim_first = self._cos(f, st.first_feat)
        sim_fused = 0.68 * sim_recent + 0.32 * sim_first
        return float(sim_recent), float(sim_first), float(sim_fused)

    def _state_by_gid(self, gid: int) -> Optional[_IDState]:
        for st in self._states:
            if int(st.gid) == int(gid):
                return st
        return None

    def similarity_to_gid(self, feat: Optional[np.ndarray], gid: int) -> float:
        if feat is None:
            return -1.0
        st = self._state_by_gid(int(gid))
        if st is None:
            return -1.0
        f = self._normalize(feat.astype(np.float32))
        _, _, sim_fused = self._state_scores(st, f)
        return sim_fused

    def observe(
        self,
        gid: int,
        feat: Optional[np.ndarray],
        ts_sec: Optional[float] = None,
        quality: Optional[float] = None,
    ) -> None:
        if feat is None:
            return
        if quality is not None and float(quality) < self.observe_min_quality:
            return
        st = self._state_by_gid(int(gid))
        if st is None:
            return
        f = self._normalize(feat.astype(np.float32))
        self._update_state(st, f, ts_sec=ts_sec, quality=quality)

    def best_candidate(
        self,
        feat: Optional[np.ndarray],
        forbidden_gids: Optional[Iterable[int]] = None,
    ) -> Optional[MatchCandidate]:
        if feat is None:
            return None
        f = self._normalize(feat.astype(np.float32))

        forbidden = set(int(x) for x in forbidden_gids) if forbidden_gids is not None else set()
        valid_states = [st for st in self._states if st.gid not in forbidden]
        if not valid_states:
            return None

        scored: List[Tuple[_IDState, float, float, float]] = []
        for st in valid_states:
            sr, sf, sx = self._state_scores(st, f)
            scored.append((st, sr, sf, sx))
        scored.sort(key=lambda x: x[3], reverse=True)
        st, sr, sf, sx = scored[0]
        second = float(scored[1][3]) if len(scored) >= 2 else -1.0
        margin = float(sx - second) if second >= 0.0 else float(sx)
        return MatchCandidate(
            gid=int(st.gid),
            sim_fused=float(sx),
            sim_recent=float(sr),
            sim_first=float(sf),
            second_sim=second,
            margin=margin,
        )

    def _best_two(
        self,
        f: np.ndarray,
        forbidden_gids: Optional[Iterable[int]] = None,
    ) -> Tuple[int, float, float]:
        forbidden = set(int(x) for x in forbidden_gids) if forbidden_gids is not None else set()

        valid_states = [st for st in self._states if st.gid not in forbidden]
        if not valid_states:
            return -1, -1.0, -1.0

        sims = np.array([self._state_scores(st, f)[2] for st in valid_states], dtype=np.float32)
        best_idx_local = int(np.argmax(sims))
        best_sim = float(sims[best_idx_local])
        second_sim = float(np.partition(sims, -2)[-2]) if len(sims) >= 2 else -1.0

        best_gid = valid_states[best_idx_local].gid
        best_idx_global = next(i for i, st in enumerate(self._states) if st.gid == best_gid)
        return best_idx_global, best_sim, second_sim

    def _new_id(self, f: np.ndarray, ts_sec: Optional[float] = None) -> int:
        gid = self.next_id
        self.next_id += 1
        st = _IDState(
            gid=gid,
            feat=f,
            first_feat=f.copy(),
            prototypes=deque(maxlen=self.max_prototypes),
            last_ts=ts_sec,
            n_obs=1,
        )
        st.prototypes.append(f)
        self._states.append(st)
        if self.verbose:
            print(f"[IDBANK] NEW ID {gid}")
        return gid

    def new_identity(self, feat: Optional[np.ndarray], ts_sec: Optional[float] = None) -> int:
        if feat is None:
            return self._new_id(self._normalize(np.zeros((1,), dtype=np.float32)), ts_sec=ts_sec)
        f = self._normalize(feat.astype(np.float32))
        return self._new_id(f, ts_sec=ts_sec)

    def _update_state(
        self,
        st: _IDState,
        f: np.ndarray,
        ts_sec: Optional[float],
        quality: Optional[float] = None,
    ) -> None:
        sim_main = self._cos(st.feat, f)
        if sim_main >= self.min_update_sim:
            st.feat = self._normalize(self.ema * st.feat + (1.0 - self.ema) * f)

        if quality is not None and float(quality) < self.observe_min_quality:
            st.last_ts = ts_sec
            st.n_obs += 1
            return

        add_proto = True
        if st.prototypes:
            best_proto_sim = max(self._cos(p, f) for p in st.prototypes)
            add_proto = best_proto_sim < (1.0 - self.prototype_min_delta)

        if add_proto:
            st.prototypes.append(f)
        st.last_ts = ts_sec
        st.n_obs += 1

    def assign(
        self,
        feat: Optional[np.ndarray],
        ts_sec: Optional[float] = None,
        forbidden_gids: Optional[Iterable[int]] = None,
    ) -> int:
        if feat is None:
            return self._new_id(self._normalize(np.zeros((1,), dtype=np.float32)), ts_sec=ts_sec)

        f = self._normalize(feat.astype(np.float32))

        if not self._states:
            return self._new_id(f, ts_sec=ts_sec)

        best_idx, best_sim, second_sim = self._best_two(f, forbidden_gids=forbidden_gids)
        if best_idx < 0:
            return self._new_id(f, ts_sec=ts_sec)

        best_gid = self._states[best_idx].gid
        best_recent, best_first, _ = self._state_scores(self._states[best_idx], f)

        reuse = False
        hard_thresh = self.hard_thresh
        if (
            self.enroll_reuse_thresh is not None
            and len(self._states) <= self.enroll_protect_states
        ):
            hard_thresh = max(hard_thresh, self.enroll_reuse_thresh)

        if best_sim >= hard_thresh:
            reuse = True
        elif best_sim >= self.soft_thresh and (second_sim < 0 or (best_sim - second_sim) >= self.margin):
            reuse = True
        elif best_sim >= (self.soft_thresh - 0.06) and best_first >= (self.soft_thresh - 0.02):
            # First-ID compatibility fallback: avoid unnecessary new IDs after overlap/partial occlusion.
            reuse = True

        if reuse:
            self._update_state(self._states[best_idx], f, ts_sec=ts_sec, quality=None)

            if self.verbose:
                t = f"{ts_sec:.2f}" if ts_sec is not None else "NA"
                print(f"[IDBANK] attach {best_gid} best={best_sim:.3f} second={second_sim:.3f} ts={t}")
            return best_gid

        if self.verbose:
            t = f"{ts_sec:.2f}" if ts_sec is not None else "NA"
            print(f"[IDBANK] NEW (best={best_sim:.3f} second={second_sim:.3f}) ts={t}")
        return self._new_id(f, ts_sec=ts_sec)
