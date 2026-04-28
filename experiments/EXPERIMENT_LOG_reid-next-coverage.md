# Experiment Log — reid-next-coverage-experiments

## Branch origin

- Starting from commit: `1fd96bd` (main — "Initial commit of BoT-SORT + ReID tracking pipeline")
- Frozen baseline directory: `runs/baseline_freeze_2026-04-28/`
- Frozen CAM1 CSV MD5: `b9e3e9052d1cacff1334aa82815cc817`

## Accepted baseline (do not modify)

| Metric | Value |
|--------|-------|
| Total rows | 6,194 |
| Positive rows | 3,256 (52.6%) |
| Zero-gid rows | 2,938 (47.4%) |
| Same-ID same-frame dups | 0 |
| Macro pred→GT purity | 0.9558 (95.6%) |
| Per-ID counts | {1:1331, 2:1028, 3:515, 4:2, 5:31, 6:81, 8:58, 9:102, 10:108} |

Active promote map:
```python
promote_pred0_to_target_from_audit(
    target_gid_by_gt={"11": 8, "6": 6, "10": 10, "2": 2},
    iou_threshold=0.34,
)
```

## Target

- Positive row coverage > 52.6%
- Macro pred→GT purity ≥ 0.90 (hard floor)
- Same-ID same-frame duplicates = 0 (hard floor)
- Positive rows must not decrease from 3,256

## Adoption criterion (per experiment)

Before adopting any experiment result:
1. `positive_rows >= 3256`
2. `macro_purity >= 0.90`
3. `same_id_dups == 0`
4. Per-ID counts for IDs {1, 2, 3, 5, 6, 8, 9, 10} must not regress materially

## Naming convention for CAM1 backups

`retail-shop_CAM1_tracks.csv.pre-<experiment-name>.csv`

## Known non-starters (do not attempt)

- **GT7:** Gate-relaxation experiment run and reverted 2026-04-26. Collapsed ID3 515→12. Do not attempt.
- **GT8:** pair-split (5 vs 9) corrupts promoted rows. Safe only if promote is re-sequenced AFTER pair-split.

## Experiment entries

| Date | Experiment | Changed rows | Purity before | Purity after | Dups | Decision |
|------|-----------|--------------|--------------|-------------|------|----------|
| 2026-04-28 | spatial_disambig_v1 | 0 | 0.9558 | 0.9558 (unchanged) | 0 | NULL RESULT — keep code, do not adopt |

### spatial_disambig_v1 — 2026-04-28 — NULL RESULT

**Tag:** `spatial_disambig_v1`  
**Files changed:** `src/reid/reentry_linker.py` (ReentryConfig + logic block + decisions_log), `scripts/run_batch.py` (enable flag)  
**Backup:** `retail-shop_CAM1_tracks.csv.pre-spatial-disambig.csv` (MD5 b9e3e9052d1cacff1334aa82815cc817)

**Implementation:** Gap-normalized spatial plausibility check inside the `cross_person_ambiguous` block. Top candidate labeled implausible if `jump > gap_frames * 180px/frame`. Second candidate labeled plausible if `jump <= gap_frames * 90px/frame`. Fires only when top is implausible, second is plausible, and second rerank score ≥ 0.68.

**Result:** `spatial_disambig_applied = 0` out of 16 cross_person_ambiguous cases. Positive rows: 3,256 → 3,256 (unchanged). Purity: unchanged. Dups: 0.

**Root cause of zero fires:** All 16 cross_person_ambiguous cases have large reentry gaps (min=4, median=204, max=834 frames). At 180px/frame normalization, the speed limit for even the shortest gap (4 frames) is 720px — and the largest observed jump_top for that case is only 100px, well within the limit. For the long-gap cases (gap=341, jump_top=1201px), the limit is 61,380px: no observed jump can exceed it. The gap-normalized threshold is structurally too permissive for this video's cross_person_ambiguous population.

**Design assumption that was wrong:** The design was based on the reid_ambiguity diagnostic which showed a 1617px jump at a 1-frame gap (Row 1). That tracklet was a special case (GT7/10 overlap region with an extremely short ByteTrack-generated gap). The general cross_person_ambiguous population has reentry gaps of 100–834 frames where spatial evidence is not discriminative regardless of jump distance.

**Code status:** Implementation is correct and feature-flagged (`spatial_disambig_enable=False` by default, `True` only for exact CAM1 in run_batch.py). Keep the code — it works and is harmless. Disable the flag until a redesign addresses the gap problem.

**Next direction:** The gap-normalized approach only works if cross_person_ambiguous cases have short gaps. They don't in this video. A redesign should use either:
1. Absolute distance normalized by frame dimensions (not time) — compare jump to frame diagonal (e.g., ~3040px for 2560×1944), flag as implausible if > 50% of frame width with a SHORT gap
2. Ratio check: require `jump_top / jump_sec >= 3.0` (top is at least 3× further than second) regardless of absolute distance — this is gap-independent
3. Abandon spatial lever entirely and prioritize open-set fresh-ID assignment or pair-split redesign instead
