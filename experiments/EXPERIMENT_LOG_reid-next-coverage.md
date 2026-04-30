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

## Experiment entries

| Date | Experiment | Changed rows | Purity before | Purity after | Dups | Decision |
|------|-----------|--------------|--------------|-------------|------|----------|
| 2026-04-28 | spatial_disambig_v1 | 0 | 0.9778 | 0.9778 (unchanged) | 0 | NULL RESULT — keep code, do not adopt |
| 2026-04-28 | gt8_reseq_v1 | +13 | 0.9778 | 0.9724 | 0 | ADOPTED — ID9: 102→115, purity drop geometric artifact |

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

---

### gt8_reseq_v1 — 2026-04-28 — ADOPTED

**Tag:** `gt8_reseq_v1`  
**Files changed:** `scripts/run_batch.py` (15-line block inserted after pair-split (2 vs 5), inside `if use_cam1_forced_targets:`)  
**Backup:** `retail-shop_CAM1_tracks.csv.pre-gt8-reseq.csv` (MD5 b9e3e9052d1cacff1334aa82815cc817)

**Implementation:** Moved GT8 promote from pass-1 (before both pair-splits) to pass-2 (after both pair-splits). A second `promote_pred0_to_target_from_audit(target_gid_by_gt={"8": 9}, iou_threshold=0.34)` call is inserted immediately after the pair-split (2 vs 5) `[INFO]` line. The GT11/GT6/GT10/GT2 promote stays in pass-1 (before pair-splits) — only GT8 moves.

**Root cause of prior failure:** The original GT8 experiment (reverted 2026-04-28) ran promote before pair-split (5 vs 9). The pair-split classifier moved 9 of 13 promoted GT8 rows from ID 9 to ID 5. Re-sequencing promote after both pair-splits means the classifier has already exited and cannot reclassify the new rows.

**Result:**
- Pair-split (5 vs 9): total_changed=0, samples_a=31, samples_b=102 (enforce rows only, not yet promoted)
- Pair-split (2 vs 5): total_changed=0 (samples_b=0, still inactive)
- GT8 promote: applied=True, changed_rows=13, skipped_dup=10, skipped_iou=0
- Positive rows: 3,256 → 3,269 (+13, 52.8%)
- Per-ID counts: {1:1331, 2:1028, 3:515, 4:2, 5:31, 6:81, 8:58, **9:115**, 10:108} (only ID9 changed)
- Same-frame dups: 0
- Macro pred→GT purity: 0.9778 → 0.9724 (−0.0054; above 0.90 floor)

**Purity drop root cause:** In frame 1416, the promoted pred=9 box (1604.76→1988.24, 486.31→1133.34) overlaps both GT8 audit box (IoU=0.413) and GT5 audit box (IoU=0.825). Greedy IoU matching takes GT5 first, so pred=9 is counted as a GT5 hit. This is the same class of geometric overlap artifact as the documented GT8→pred10 cross-hit at frame 1440 (two people physically overlapping). The promotion itself is correct (the gid=0 row has IoU ≥ 0.34 with the GT8 audit box); the audit-metric contamination is measurement noise, not a real identity error.

**skipped_dup=10 root cause:** 10 GT8 audit rows already had pred=9 present in the same frame (from the enforce step). The per-frame duplicate guard correctly blocked these.

**Adopted state:**
- New positive rows: 3,269 (52.8%)
- New ID9 count: 115 (was 102)
- Backup before experiment confirmed identical to frozen baseline (MD5 match)
- All other IDs unchanged

---

## Branch-close summary — 2026-04-28

**Branch:** `reid-next-coverage-experiments`  
**Status:** CLOSED — no further experiments planned  
**Final experimental-branch state:** positive rows = 3,269 (52.8%), macro pred→GT purity = 0.9724, same-frame duplicate positive IDs = 0

### Why the branch stops here

All provably safe gid=0 rows have been promoted. "Provably safe" is defined as: the audit confirms the GT identity at IoU ≥ 0.34, no same-frame duplicate is created, and the downstream pair-splits do not reclassify the promoted rows. These constraints are jointly satisfied only for GT11→pred=8, GT6→pred=6, GT10→pred=10, GT2→pred=2 (frames 120–360), and GT8→pred=9 (re-sequenced post-split). All five candidate sets are exhausted.

The remaining 2,925 zero-gid rows fall into three structural categories, none of which are addressable without changing the identity philosophy of the system:
1. **Architecturally blocked (GT7):** Confirmed by ByteTrack (up to 73 consecutive hits) but permanently rejected by the active-owner zone protection gate. Gate relaxation collapses ID3 515→12 (documented 2026-04-26, do not retry).
2. **Genuinely ambiguous (`pending:no_match_or_ambiguous`, 73% of zero rows):** The ReID bank cannot assign these tracks to any known identity. Assigning fresh IDs would change "recognized" coverage into "tracked" coverage — a different claim.
3. **Below IoU threshold:** No gid=0 track overlaps a GT audit box at IoU ≥ 0.34 in the relevant frame, so no audit-safe promotion is possible.

### Experiments on this branch

| Experiment | Result | Code status |
|-----------|--------|-------------|
| spatial_disambig_v1 | NULL — 0 fires (gap-normalized threshold too permissive for long-gap ambiguous cases) | Feature-flagged, disabled; keep for future redesign |
| gt8_reseq_v1 | ADOPTED — +13 rows to ID9; purity drop is geometric overlap artifact at frame 1416 | Active in `run_batch.py` pass-2 promote block |

All prior experiments (GT11/GT6/GT10/GT2 promotions, GT8 first attempt) were adopted or reverted on the main branch before this experimental branch was opened and are not re-listed here.

### Why open-set fresh-ID assignment was not attempted

Open-set fresh-ID assignment would assign new positive IDs to the `pending:no_match_or_ambiguous` population — tracks the ReID bank explicitly could not identify. This was rejected for three independent reasons:

1. **Identity-philosophy break:** The system's core claim is high-precision closed-set recognition: a positive ID means the system is confident the track belongs to a known person. Open-set assignment converts this to "any ByteTrack-confirmed track gets a positive ID," which is a different and weaker claim. The frozen dissertation report argues for the conservative design; invalidating that argument post-submission adds no value.
2. **Unquantified purity risk:** Ambiguous tracks are ambiguous because the ReID bank cannot distinguish them from existing known persons. Fresh IDs assigned to re-entering known persons would register as new pred IDs overlapping known GT audit boxes, either inflating coverage with incorrect attributions or producing unmeasurable coverage (fresh IDs not in the GT mapping). Either outcome requires reframing the audit metric.
3. **Implementation risk with no safe revert:** Unlike promotion (which is a bounded, auditable CSV mutation), open-set assignment interacts with pair-splits, group-merge, overlap-handoff, and local-consistency passes that were designed for a closed-set population. A cascade regression across stable IDs {1,2,3,5,6,8,9,10} would require a full revert and leave no experimental gain.

### Merge recommendation

**Do not merge into the frozen baseline.** The frozen baseline (`runs/baseline_freeze_2026-04-28/`) and the dissertation submission cite 52.6% / 95.6%. Those numbers are correct and must not change.

**Selective code carry-forward for future work only:**
- `src/reid/reentry_linker.py` — spatial_disambig_v1 block is feature-flagged (`spatial_disambig_enable=False` default). Safe to carry forward as dead code pending a redesign that addresses the long-gap problem.
- `scripts/run_batch.py` — gt8_reseq_v1 two-pass promote block is live and correct. This is the only code change that should be carried forward if the pipeline is re-run on new data.
- All other experimental code (diagnostics, trace scripts) is read-only and carries no risk.

**If a future rerun of CAM1 is required:** start from the frozen backup (`retail-shop_CAM1_tracks.csv.pre-gt8-reseq.csv`, MD5 b9e3e9052d1cacff1334aa82815cc817), apply only the two-pass promote in `run_batch.py`, and verify the acceptance gate (positive_rows ≥ 3,256, purity ≥ 0.90, dups = 0) before using the output.

---

## Future-work paragraph (for dissertation viva / written future-work section)

The primary coverage limitation — 47.2% zero-gid rows — is dominated by `pending:no_match_or_ambiguous` tracks: ByteTrack-confirmed tracklets that pass the consecutive-hit gate but cannot be matched to any identity in the ReID bank. Addressing this structurally requires either improving the ReID bank (fine-tuning OSNet on in-domain hard negatives to reduce embedding ambiguity) or moving to an open-set identity assignment strategy in which unmatched confirmed tracks are assigned provisional fresh IDs and later merged by appearance similarity across re-entries. The latter would require a post-hoc clustering step over track embeddings — rather than online bank lookup — and a revised audit metric that can score provisional IDs against the GT mapping. A second structural ceiling is GT7 (2.2% positive-ID coverage), which is blocked by the active-owner zone protection gate rather than by ReID quality; resolving it requires either weakening the gate's same-owner similarity threshold selectively for tracks with very high consecutive-hit counts, or redesigning the gate to distinguish co-located occlusion (legitimate re-entry blocked) from true same-person relock (correct gate fire). Both directions are tractable as standalone engineering changes but were excluded from the current work to preserve the high-precision, zero-duplicate property of the accepted output.
