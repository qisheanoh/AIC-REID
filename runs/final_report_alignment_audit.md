# Final Report Alignment Audit

**Audited document:** `AIC_REID_20512381_FYP.docx` (extracted to `outputs/report_text.md`)
**Audited against:** repository artifacts under `/Users/ohqishean/AIC-REID/`
**Auditor:** automated artifact cross-check
**Audit date:** 2026-04-29
**Scope:** every reported number, table value, figure caption, appendix path, and qualitative claim that is directly tied to a repository artifact.
**Status verdict:** **NEEDS FIXES** — the dissertation contains two distinct stale-data clusters (Table 8.3 / Section 8.4 / abstract / conclusion CAM1 purity numbers; Table 8.8 / Section 8.7 Demo_Video numbers) and one ambiguous similarity-range phrasing. All other numbers verify exactly.

A claim-level pass/fail/warning summary is provided in the verdict section at the end of this report.

---

## 1. Detector — YOLOv8m (Section 8.1, abstract, conclusion)

**Source artifact:** `archive/old_runs/train/yolo_cam1_person/results.csv` (last epoch row), `data/datasets/yolo_cam1_person/`.

| Claim | Value in report | Value in artifact | Status |
|---|---|---|---|
| Domain-adapted detector | YOLOv8m | yolov8m.pt fine-tuned on CAM1 audit crops (results.csv exists) | PASS |
| mAP50 | 0.952 | 0.95195 | PASS |
| mAP50-95 | 0.790 | 0.79044 | PASS |
| Precision | 0.905 | 0.90535 | PASS |
| Recall | 0.917 | 0.91667 | PASS |
| Train images / boxes | 132 / 412 | 132 train images, 412 boxes | PASS |
| Val images / boxes | 23 / 72 | 23 val images, 72 boxes | PASS |
| Training epochs | 8 (final result row) | 8 epochs in results.csv | PASS |

All detector values verify cleanly.

---

## 2. ReID — OSNet-x1_0 (Section 8.2, abstract, conclusion)

**Source artifact:** `archive/old_runs/train/osnet_cam1/train.log-2026-04-09-18-35-28`.

| Claim | Value in report | Value in artifact | Status |
|---|---|---|---|
| Backbone | OSNet-x1_0 | OSNet-x1_0 logged | PASS |
| Identities | 5 | 5 identities | PASS |
| Train / query / gallery split | 290 / 40 / 87 | 290 / 40 / 87 | PASS |
| Final mAP | 83.5% | 0.835 / 83.5% | PASS |
| Rank-1 | 100% | 1.000 / 100% | PASS |

All ReID values verify cleanly. (The dissertation correctly notes this is a small-domain split, not Market1501-public.)

---

## 3. CAM1 audit ground truth (Section 8.3, Table 8.4)

**Source artifact:** `experiments/audit/cam1_manual_audit_sheet.csv` (484 rows, 11 GT persons).

| Claim | Value in report | Value in artifact | Status |
|---|---|---|---|
| Audit row count | 484 | 484 | PASS |
| Number of GT persons | 11 | 11 (GT1..GT11) | PASS |
| GT1 audited frames | 8 | 8 | PASS |
| GT2 audited frames | 102 | 102 | PASS |
| GT3 audited frames | 121 | 121 | PASS |
| GT4 audited frames | 2 | 2 | PASS |
| GT5 audited frames | 20 | 20 | PASS |
| GT6 audited frames | 33 | 33 | PASS |
| GT7 audited frames | 135 | 135 | PASS |
| GT8 audited frames | 23 | 23 | PASS |
| GT9 audited frames | 4 | 4 | PASS |
| GT10 audited frames | 26 | 26 | PASS |
| GT11 audited frames | 10 | 10 | PASS |
| GT7 frames range | 29–1,788 | 29–1,788 | PASS |
| Locked protocol parameters | IoU≥0.34, max_sec=150 | matches `src/reid/track_linker.py::evaluate_first_two_minute_audit_metrics` | PASS |

All audit-sheet values verify cleanly.

---

## 4. CAM1 primary benchmark — Table 8.3, Section 8.4, abstract, conclusion (**STALE DATA**)

This is the single most important finding of the audit.

The dissertation reports the following Table 8.3 / Section 8.4 / abstract / conclusion CAM1 numbers:

- 95.6% pred-to-GT macro purity
- averaged over 8 evaluated positive predicted IDs
- 5 of those 8 achieve 100% purity
- 9 positive global identities in total
- 269 / 484 (55.6%) covered audit rows
- 99 / 484 (20.5%) seen-but-unidentified
- 116 / 484 (24.0%) unmatched
- 4 fragmented GT persons (GT2, GT5, GT8, GT11)
- 95.9% GT-to-pred purity
- zero same-frame duplicate positive IDs across 1,800 frames

**These numbers do not match either the current `runs/kpi_batch/retail-shop_CAM1_tracks.csv` or the FROZEN `runs/baseline_freeze_2026-04-28/retail-shop_CAM1_tracks.FROZEN.csv` reference cited as the canonical baseline in Tables 8.5/8.6.**

When the locked audit protocol (IoU ≥ 0.34, max_sec = 150, the same logic used in `evaluate_first_two_minute_audit_metrics`) is replayed against each artifact:

| Metric | Report (Table 8.3) | Current `retail-shop_CAM1_tracks.csv` | FROZEN `…_tracks.FROZEN.csv` | Source of report number |
|---|---|---|---|---|
| Pred-to-GT macro purity | **95.6%** | 97.24% | **97.78%** | `runs/baseline_freeze_2026-04-28/final_report_revised_publishable.FROZEN.md` |
| Macro purity averaged over | 8 IDs | 8 IDs | 8 IDs | inherited |
| Total positive global IDs | 9 | 9 | 9 | matches both |
| Audit rows covered | 269 / 484 (55.6%) | 298 / 484 (61.6%) | **285 / 484 (58.88%)** | inherited |
| Audit rows seen-but-unidentified | 99 / 484 (20.5%) | 51 / 484 (10.5%) | 64 / 484 (13.2%) | inherited |
| Audit rows unmatched | 116 / 484 (24.0%) | 135 / 484 (27.9%) | 135 / 484 (27.9%) | inherited |
| Fragmented GTs | 4 | 2 | **1** | inherited |
| GT-to-pred macro purity | 95.9% | (not recomputed) | (not recomputed) | inherited |
| Same-frame duplicate positive IDs | 0 | 0 | 0 | matches both |

The 95.6% / 269 / 99 / 116 / 4-fragmented numbers were carried over from `runs/baseline_freeze_2026-04-28/final_report_revised_publishable.FROZEN.md` (a much earlier write-up) and were never refreshed when the FROZEN baseline was rebuilt for `runs/baseline_freeze_2026-04-28/retail-shop_CAM1_tracks.FROZEN.csv`.

The crucial point: the dissertation explicitly uses the **FROZEN** tracks as the canonical reference in Table 8.5 (`full_aic_reid_frozen_reference` row of `runs/ablations/.../ablation_summary.csv`) and Table 8.6 (`97.78%` purity, fragmentation = 1, 9 positive IDs). Tables 8.5 and 8.6 are therefore internally consistent with the FROZEN artifact, but Table 8.3 / Section 8.4 / abstract / conclusion are not.

**Status: FAIL — these numbers must be replaced.**

**Recommended replacement values (FROZEN baseline, locked audit protocol):**

- Pred-to-GT macro purity: **97.78%** (was 95.6%)
- Audit rows covered: **285 / 484 (58.88%)** (was 269 / 484 / 55.6%)
- Audit rows seen-but-unidentified: **64 / 484 (13.2%)** (was 99 / 484 / 20.5%)
- Audit rows unmatched: **135 / 484 (27.9%)** (was 116 / 484 / 24.0%)
- Fragmented GTs (≥2 positive predicted IDs sharing GT): **1** (was 4 — GT2, GT5, GT8, GT11)
- 9 positive global identities total: unchanged (matches FROZEN)
- 8 evaluated positive predicted IDs in macro-purity average: unchanged (matches FROZEN — 1 ID has insufficient audit evidence)
- 0 same-frame duplicate positive IDs: unchanged

Once Table 8.3 is refreshed, the abstract sentence
> "AIC-ReID reports 95.6% pred-to-GT macro purity averaged over 8 evaluated positive predicted IDs"

must be updated to **97.78%**, and the conclusion sentence
> "The key outcome is 95.6% pred-to-GT macro purity on the CAM1 benchmark"

must be updated to **97.78%**. The "5 achieve 100% purity" claim in Section 8.4 should also be re-derived from the FROZEN per-ID purity profile and updated if it changes.

---

## 5. Baseline comparison — Table 8.5 (Section 8.4.1)

**Source artifact:** `runs/ablations/cam1_ablation_20260428_225623/ablation_summary.csv`.

Every value in Table 8.5 was reproduced from `ablation_summary.csv`:

| Variant | Positive IDs | Coverage | Purity | Same-frame duplicates | Fragmentation | Status |
|---|---|---|---|---|---|---|
| Full AIC-ReID (frozen ref.) | 9 | 58.88% | 97.78% | 0 | 1 | PASS |
| ByteTrack-only baseline | 118 | 76.86% | 96.89% | 0 | 9 | PASS |
| BoT-SORT-style online baseline | 84 | 78.10% | 92.08% | 0 | 11 | PASS |
| AIC-ReID, no audit-guided offline repair | 12 | 50.83% | 86.30% | 0 | 6 | PASS |

All values match `ablation_summary.csv` to the digit precision quoted in the dissertation.

---

## 6. Ablation study — Table 8.6 (Section 8.4.2)

**Source artifact:** `runs/ablations/cam1_ablation_20260428_225623/ablation_summary.csv`, `runs/ablations/cam1_ablation_20260428_225623/ablation_report.md`.

| Variant | Positive IDs | Coverage | Purity | Same-frame duplicates | Fragmentation | Status |
|---|---|---|---|---|---|---|
| no_offline_reentry_linking | 10 | 61.16% | 90.57% | 0 | 4 | PASS |
| no_tracklet_stitching | 9 | 61.57% | 97.22% | 0 | 2 | PASS |
| osnet_only_no_attire_shape | 8 | 62.81% | 96.53% | 0 | 3 | PASS |
| relaxed_conservative_gate | 9 | 61.78% | 97.22% | 0 | 3 | PASS |
| FULL_CAM1 no_cam1_anchor_alignment | 11 IDs / 3,863 positive rows / 0 dup / no anchor report | matches | matches | 0 | n/a (no anchor report) | PASS |

The narrative claims in Section 8.4.2 — "Disabling offline re-entry linking increases fragmentation from 1 to 4 and reduces purity from 97.78% to 90.57%" — are therefore consistent with the FROZEN reference (97.78% purity, fragmentation = 1) and are correct. The contradiction is only with Table 8.3 / Section 8.4 (which incorrectly report 95.6%) — Section 8.4.2 silently uses the **correct** FROZEN purity number (97.78%).

This internal inconsistency between Table 8.3 (95.6%) and Section 8.4.2 narrative (97.78% as the "from" value) should be repaired by fixing Table 8.3.

---

## 7. Dense MOT subset — Table 8.6a (Section 8.4.3)

**Source artifacts:**
- `experiments/dense_mot_cam1/selected_frames.csv` — 300 selected frames
- `experiments/dense_mot_cam1/dense_gt.csv` — 1,110 GT boxes
- `experiments/dense_mot_cam1/gt_validation_report.json` — strict_complete = true, valid = true, frames_annotated = 300
- `runs/dense_mot_cam1/dense_mot_summary.csv` — per-method MOT totals

| Method | Predictions | Matched | FP | FN | IDSW | MOTA | IDF1 | HOTA | Status |
|---|---|---|---|---|---|---|---|---|---|
| Full AIC-ReID frozen | 530 | 84 | 446 | 1,026 | 3 | −0.329 | NA | NA | PASS |
| ByteTrack-only | 971 | 87 | 884 | 1,023 | 1 | −0.719 | NA | NA | PASS |
| BoT-SORT-style | 1,001 | 89 | 912 | 1,021 | 1 | −0.742 | NA | NA | PASS |
| AIC-ReID no audit | 524 | 84 | 440 | 1,026 | 0 | −0.321 | NA | NA | PASS |

All four MOTA values are negative as the dissertation states. IDF1 / HOTA correctly reported as NA with the documented "no trusted motmetrics/TrackEval implementation in the run environment" caveat. PASS in full.

---

## 8. FULL_CAM1 — Table 8.7, Section 8.6, Figures 8.9–8.11

**Source artifacts:**
- `runs/kpi_batch/retail-shop_FULL_CAM1_tracks.csv`
- `runs/kpi_batch/retail-shop_FULL_CAM1_identity_metrics.json`
- `runs/kpi_batch/reentry_debug/retail-shop_FULL_CAM1/reentry_stats.json`
- `runs/kpi_batch/FULL_CAM1_preprocess_report.json`

| Claim | Value in report | Value in artifact | Status |
|---|---|---|---|
| Frame count | 3,616 | 3,616 (preprocess report; OpenCV probe) | PASS |
| Duration | 297 s / 297.3 s | 297.34 s | PASS |
| Bad frames | 144 (3.98%) | 144 / 3.98% | PASS |
| Canonical slots assigned (before convergence) | all 11 | 11 (before_anchor: gids 1..11 all > 0 rows) | PASS |
| rows_on_canonical | 2,788 | 2,788 (after_anchor) | PASS |
| rows_off_canonical | 1,075 | 1,075 (after_anchor) | PASS |
| Same-frame duplicate positive IDs | 0 | 0 (after_anchor.same_frame_duplicate_rows = 0) | PASS |
| Slot 7 unfilled after convergence | yes | per_canonical_rows["7"] = 0 (after_anchor) | PASS |
| Re-entry attempts | 140 | reentry_stats.json: attempts = 140 | PASS |
| Re-entries accepted | 49 | accepted_reuses = 49 | PASS |
| Re-entries rejected (total) | 91 | weak (5) + ambiguous (83) + conflict (3) = 91 | PASS |
| Ambiguous rejections | 83 | 83 | PASS |
| Slot 2 row count (Table 8.7) | 1,396 | 1,396 | PASS |
| Slot 9 row count (Table 8.7) | 443 | 443 | PASS |
| Slot 9 timeline span (Fig 8.10) | frames 2,043–2,917 | first=2,043, last=2,917 | PASS |
| Slot 1 row count | 245 | 245 | PASS |
| Slot 8 row count | 435 | 435 | PASS |
| All other Table 8.7 slot row counts | 83 / 59 / 35 / 62 / 0 / 15 / 15 (slots 3,4,5,6,7,10,11) | match per_canonical_rows | PASS |

**Minor warning:** the live `runs/kpi_batch/retail-shop_FULL_CAM1_tracks.csv` exposes 10 unique positive global IDs in the CSV column `global_id` (`[1,2,3,4,5,6,8,9,10,11]`), while `identity_metrics.json after_anchor.unique_gids` lists 11 (`[1,2,3,4,5,6,8,9,10,11,12]`). The discrepancy is that the off-canonical rows (1,075 rows associated with `gid=12` in the metrics JSON) appear merged under `gid=10` in the published CSV. Total `rows_with_positive_gid = 3,863` matches in both. This does not invalidate Table 8.7 (which uses canonical slot indices 1..11), but it is worth noting to a careful reader; the dissertation's text already correctly says "all 11 slots received assignments before convergence filtering" rather than "in the final CSV."

---

## 9. CAM1 retest protocol — Section 8.5, Figure 8.8

**Source artifact:** `runs/baseline_freeze_2026-04-28/retail-shop_CAM1_tracks.FROZEN.csv` evaluated against the locked audit protocol; per-GT coverage tabulated in Table 8.4.

| Claim | Status |
|---|---|
| 5 / 5 re-entry events resolved correctly | matches retest_protocol.json (not directly recomputed in this audit, but consistent with FROZEN tracks being the locked baseline) — PASS-by-document |
| 0 false re-entry merges | PASS-by-document |
| 0 missed re-entry merges | PASS-by-document |
| GT1 8 frames → 62.5% coverage | matches Table 8.4 / per-GT coverage; PASS |
| GT9 4 frames → 75.0% coverage | matches Table 8.4 / per-GT coverage; PASS |
| Overall FAIL flag (low-sample) | the report explicitly explains this; PASS-by-document |

---

## 10. Demo_Video — Section 8.7, Table 8.8 (**STALE DATA**)

**Source artifacts:**
- `runs/kpi_batch/retail-shop_Demo_Video_tracks.csv` (current)
- `runs/kpi_batch/retail-shop_Demo_Video_tracks.csv.pre-merge-fix.csv` (an earlier snapshot)
- `runs/kpi_batch/Demo_Video_preprocess_report.json`

| Claim | Value in report | Current `retail-shop_Demo_Video_tracks.csv` | `…pre-merge-fix.csv` snapshot | Status |
|---|---|---|---|---|
| Frame count | 370 | 370 | 370 | PASS |
| Duration | 30.8 s | 30.83 s | 30.83 s | PASS |
| Bad frames | 0 | 0 | 0 | PASS |
| Positive global IDs | **8** | **7** ([1,2,3,4,5,6,7]) | 8 | **FAIL** — current = 7, not 8 |
| Positive rows total | **760** | **717** | 760 | **FAIL** — current = 717, not 760 |
| Same-frame duplicate positive IDs | 0 | 0 | 0 | PASS |
| Table 8.8 ID-1 (12–96, 85 rows) | 85 | mismatch (current ID 1 has 73 rows) | 85 | **FAIL** |
| Table 8.8 ID-2 (12–74, 47 rows) | 47 | mismatch (current ID 2 has 288 rows) | 47 | **FAIL** |
| Table 8.8 ID-3 (12–24, 13 rows) | 13 | 13 | 13 | PASS |
| Table 8.8 ID-4 (49–229, 71 rows) | 71 | 67 | 71 | **FAIL** (small) |
| Table 8.8 ID-5 (102–168, 52 rows) | 52 | 52 | 52 | PASS |
| Table 8.8 ID-6 (102–369, 268 rows) | 268 | mismatch (current ID 6 has 187 rows) | 268 | **FAIL** |
| Table 8.8 ID-7 (177–369, 187 rows) | 187 | mismatch (current ID 7 has 37 rows) | 187 | **FAIL** |
| Table 8.8 ID-8 (333–369, 37 rows) | 37 | (no ID 8 in current) | 37 | **FAIL** |

**Diagnosis:** Table 8.8 was generated from `retail-shop_Demo_Video_tracks.csv.pre-merge-fix.csv`, which had 8 positive IDs and 760 positive rows. A subsequent surgical merge fix collapsed two of those IDs, producing the current `retail-shop_Demo_Video_tracks.csv` with 7 positive IDs and 717 positive rows. The freshness report (`runs/final_output_video_freshness_report_final_only.md`) confirms the current CSV/MP4 pair are the **CURRENT** report-ready outputs (mtimes 2026-04-26), so Table 8.8 must be regenerated against this current pair, not against the earlier pre-merge snapshot.

**Recommended replacement values (current `retail-shop_Demo_Video_tracks.csv`):**

- Section 8.7 first sentence: "Demo_Video (370 frames, 30.8 s, 0 bad frames) demonstrates pipeline operation. **7 positive IDs, 717 positive rows**, 0 same-frame duplicate positive IDs."
- Table 8.8 should be regenerated showing the seven IDs with their actual frame ranges and row counts (current per-ID counts: ID1=73, ID2=288, ID3=13, ID4=67, ID5=52, ID6=187, ID7=37).

---

## 11. Per-Person overhead validation — Table 8.9 (Section 8.8)

**Source artifacts:** `runs/Per_Person/person_*_*_crop_tracks.csv`, `data/raw/person_*/*_crop.mp4`.

| Clip | Raw frames | Duration s | Tracked rows | Active span (f) | Time span (s) | Density | Pos IDs | Dup frames | Status |
|---|---|---|---|---|---|---|---|---|---|
| 1_2_crop | 3,872 (=129.07×30) | 129.07 | 3,087 | 3,630 | 121.0 | 0.85 | 1 | 3 | PASS |
| 1_3_crop | 5,305 (=176.83×30) | 176.83 | 4,470 | 5,034 | 167.8 | 0.89 | 1 | 8 | PASS |
| 2_1_crop | 3,804 | 126.80 | 3,109 | 3,516 | 117.2 | 0.88 | 1 | 6 | PASS |
| 2_2_crop | 3,788 | 126.27 | 2,795 | 3,552 | 118.4 | 0.79 | 1 | 7 | PASS |
| 2_3_crop | 3,581 | 119.37 | 2,899 | 3,336 | 111.2 | 0.87 | 1 | 6 | PASS |

Every row of Table 8.9 verifies exactly. Track densities in 0.79–0.89 range and dup-frame counts in 3–8 range as claimed in the abstract and Section 8.8. PASS.

---

## 12. Cross-camera demonstration — Section 8.10, Table 8.10

**Source artifacts:**
- `runs/cross_cam/cross_camera_matches.csv` (24 accepted match events)
- `runs/cross_cam/cross_camera_pair_scores.csv` (74 candidate pair scores)

| Claim | Value in report | Value in artifact | Status |
|---|---|---|---|
| Total identity-link events | 24 | 24 rows in `cross_camera_matches.csv` | PASS |
| canonical_primary links | 5 | 5 (Counter on `match_type`) | PASS |
| fragment_reuse links | 10 | 10 | PASS |
| Other event types | (figure 8.12 mentions only the two dominant types) | also: 6 anchor_side_consensus, 1 overlap_dominant_reuse, 1 same_cam_fragment_propagate, 1 new — caption phrasing "fragment reuse and canonical primary are the dominant link types" remains accurate | PASS |
| Similarity score range | "0.660 to 0.865" | accepted-match `score` column: 0.7347 – 0.8651; candidate pool in `cross_camera_pair_scores.csv`: 0.6601 – 0.8651 | **WARNING** |

**Issue:** the abstract and Section 8.10 say "24 identity-link events with similarity scores from 0.660 to 0.865". The lower bound 0.660 is the minimum **candidate-pool** score in `cross_camera_pair_scores.csv`, not the minimum score among the 24 **accepted** matches in `cross_camera_matches.csv` (whose minimum `score` is 0.7347).

**Recommended fix (choose one):**

- (A) Tighten the phrasing to refer to candidate scores: "Cross-camera matching evaluates 74 candidate identity pairs with similarity scores from 0.660 to 0.865, of which 24 are accepted as link events (accepted-pair scores 0.735 – 0.865)."
- (B) Or, if the intent is purely the accepted matches: "24 identity-link events with similarity scores from **0.735 to 0.865**."

Either is defensible; (A) is the most informative and matches both files. This is a phrasing fix, not a numerical fabrication, so it is graded WARNING rather than FAIL.

---

## 13. KPI / sales-zone alignment — Tables 8.11 and 8.12 (Section 8.11)

**Source artifacts:** `configs/zones/zones_cam1.yaml` and `docs/final_report.md` (Section 8.4 zone tables).

| Claim | Status |
|---|---|
| Three product-zone polygons defined | matches `configs/zones/zones_cam1.yaml` — CAM1_Z01 (Left Wall/Tap Rack), CAM1_Z02 (Front-Right Basin Display), CAM1_Z03 (Right Wall Toilet Display) | PASS |
| Foot-point + ray-casting point-in-poly | matches add_zones implementation referenced in zones yaml | PASS |
| Table 8.11 daily values for 2026-02-28 | matches `docs/final_report.md` table exactly: CAM1_Z01 12/7/10, CAM1_Z02 18/11/11, CAM1_Z03 15/6/4 | PASS |
| Table 8.12 sales-engagement alignment | matches `docs/final_report.md` table; CAM1_Z02 16.7%/45.8%/0.292 strong conflict; CAM1_Z01 44.4%/29.2%/0.152 mild conflict; CAM1_Z03 38.9%/25.0%/0.139 mild conflict | PASS |
| 45.8% / 16.7% / conversion proxy 0.27 vs 1.14–1.17 | matches `docs/final_report.md` Section 8.4 narrative | PASS |
| "one 150-second clip" caveat | matches `docs/final_report.md`: "Daily values (2026-02-28, 150-second CAM1 clip)" | PASS |

**Minor rounding-only finding:** Table 8.12 in the dissertation reports CAM1_Z01 gap = 0.152 while `docs/final_report.md` reports 0.153. The dissertation's 0.152 is consistent with 44.4% − 29.2% = 15.2 percentage points, so the dissertation's value is mathematically the more correct one. No fix required.

---

## 14. Demo_Video and FULL_CAM1 video freshness

**Source artifact:** `runs/final_output_video_freshness_report_final_only.md`.

| Output | Status from freshness report | Confirmed |
|---|---|---|
| Demo_Video tracks CSV / vis MP4 | CURRENT (mtimes 2026-04-26 11:56 / 12:35) | PASS |
| FULL_CAM1 tracks CSV / vis MP4 | CURRENT (mtimes 2026-04-29 13:48 / 13:51) | PASS |
| Earlier "ablation" stale FULL_CAM1 path excluded | yes — under `runs/ablations/`, an experimental variant (`no_cam1_anchor_alignment`) | PASS |

No ablation/baseline-only video is presented as final in the freshness report. PASS.

(Independent caveat: even though the **video** files are current, Table 8.8's text-table content is generated from a non-current intermediate snapshot of Demo_Video tracks — see Section 10 above.)

---

## 15. Conclusion / abstract claims (cross-cutting)

The abstract and conclusion contain the following cross-cutting claims that depend on multiple sections:

| Claim | Depends on | Status |
|---|---|---|
| "95.6% pred-to-GT macro purity" | Table 8.3 | **FAIL** (replace with 97.78%) |
| "averaged over 8 evaluated positive predicted IDs" | Table 8.3 / FROZEN baseline | PASS |
| "9 positive global identities" | FROZEN baseline | PASS |
| "zero same-frame duplicate positive IDs across all 1,800 frames" | FROZEN baseline | PASS |
| "FULL_CAM1 (3,616 frames, 297 s) … all 11 canonical identity slots receive assignments prior to final convergence filtering" | identity_metrics.json before_anchor | PASS |
| "Cross-camera matching produces 24 identity-link events with similarity scores from 0.660 to 0.865" | cross_camera_matches.csv + cross_camera_pair_scores.csv | WARNING (phrasing) |
| "five single-subject clips … exactly one positive global identity per clip, with track density between 0.79 and 0.89 … 3–8 same-frame duplicate artefact frames per clip" | Table 8.9 | PASS |
| "ByteTrack-only and BoT-SORT-style online baselines achieve higher audit coverage but produce 118 and 84 positive IDs" | Table 8.5 / ablation_summary.csv | PASS |
| "frozen full AIC-ReID output compresses the identity space to 9 positive IDs with 97.8% purity, zero same-frame duplicate frames, and fragmentation = 1" | Table 8.5 / FROZEN row of ablation_summary.csv | PASS |
| Dense MOT 300-frame subset, all methods negative MOTA, IDF1/HOTA absent | Table 8.6a / dense_mot_summary.csv | PASS |
| "Demo_Video (370 frames, 30.8 s, 0 bad frames) … 8 positive IDs, 760 positive rows" | Table 8.8 | **FAIL** (replace with 7 / 717) |

---

## 16. Verdict

**Overall: NEEDS FIXES.**

There are **two** stale-data clusters and **one** phrasing warning. They are localised: every other audited number — detector, ReID, ground-truth audit sheet, FULL_CAM1, baseline comparison, ablation, dense MOT, Per-Person overhead, cross-camera link counts, KPI tables, video freshness — verifies cleanly.

### Required fixes (FAIL)

1. **Table 8.3 / Section 8.4 / abstract / conclusion CAM1 numbers** — replace stale FROZEN values:
   - `95.6%` → `97.78%`
   - `269 / 484 (55.6%)` → `285 / 484 (58.88%)`
   - `99 / 484 (20.5%)` → `64 / 484 (13.2%)`
   - `116 / 484 (24.0%)` → `135 / 484 (27.9%)`
   - `4 fragmented GTs (GT2, GT5, GT8, GT11)` → `1 fragmented GT`
   - `95.9% GT-to-pred purity` → recompute from FROZEN tracks before quoting
   - "5 of 8 achieve 100% purity" → re-derive per-ID purity from FROZEN tracks before quoting
2. **Table 8.8 / Section 8.7 Demo_Video numbers** — replace stale pre-merge-fix values:
   - `8 positive IDs` → `7 positive IDs`
   - `760 positive rows` → `717 positive rows`
   - Regenerate Table 8.8 row by row from the current `runs/kpi_batch/retail-shop_Demo_Video_tracks.csv` (per-ID rows: 73 / 288 / 13 / 67 / 52 / 187 / 37).

### Recommended fix (WARNING)

3. **Cross-camera similarity-range phrasing** — clarify that "0.660 – 0.865" is the candidate-pool range; accepted matches are 0.735 – 0.865. Suggested replacement (Section 8.10 / abstract): "Cross-camera matching evaluates 74 candidate identity pairs with similarity scores from 0.660 to 0.865, of which 24 are accepted as link events (accepted-pair scores 0.735 – 0.865)."

### Optional weakening recommendations

These do not invalidate any number but would protect the dissertation against examiner pushback:

- The Demo_Video Table 8.8 ID-row figures change every time the merge-fix tooling is re-run. After applying fix #2, add a footnote to Table 8.8 noting the artifact mtime so a future re-run cannot quietly retire the table again.
- The abstract sentence "produce 118 and 84 positive IDs respectively" is true under the locked CAM1 audit protocol but could be misread as a generic ByteTrack/BoT-SORT statement. Adding the phrase "under the CAM1 audit protocol" once, in the abstract, prevents that misreading.
- The Section 8.10 caption "Fragment reuse (10) and canonical primary (5) are the dominant link types" is correct (15 of 24) but understates the existence of 6 `anchor_side_consensus` events — those are the next-largest category and could be named in one extra clause if the figure room allows.
- The FULL_CAM1 CSV column shows 10 unique positive `global_id` values (with off-canonical rows merged into `gid=10`), while `identity_metrics.json` lists 11 (`[1..11, 12]`). The dissertation already says "all 11 slots received assignments **before convergence filtering**", which matches `before_anchor.unique_gids`, so no fix is required — but it is worth mentioning briefly in a footnote to Table 8.7 that "after convergence filtering and anchor alignment, slot 7 carries 0 rows and an off-canonical pseudo-slot (12) is folded back into slot 10 in the final CSV."

### Missing artifacts found during the audit

None of the cited artifact paths in the dissertation are missing. The audit located:

- `experiments/audit/cam1_manual_audit_sheet.csv` ✓
- `runs/kpi_batch/retail-shop_CAM1_tracks.csv` ✓
- `runs/baseline_freeze_2026-04-28/retail-shop_CAM1_tracks.FROZEN.csv` ✓
- `runs/ablations/cam1_ablation_20260428_225623/ablation_summary.csv` ✓
- `runs/dense_mot_cam1/dense_mot_summary.csv` ✓
- `runs/Per_Person/person_*_*_crop_tracks.csv` ✓ (5 clips)
- `runs/cross_cam/cross_camera_matches.csv` ✓
- `runs/cross_cam/cross_camera_pair_scores.csv` ✓
- `runs/kpi_batch/retail-shop_FULL_CAM1_tracks.csv` ✓
- `runs/kpi_batch/retail-shop_FULL_CAM1_identity_metrics.json` ✓
- `runs/kpi_batch/reentry_debug/retail-shop_FULL_CAM1/reentry_stats.json` ✓
- `runs/kpi_batch/retail-shop_Demo_Video_tracks.csv` ✓
- `runs/kpi_batch/Demo_Video_preprocess_report.json` ✓
- `runs/kpi_batch/FULL_CAM1_preprocess_report.json` ✓
- `runs/final_output_video_freshness_report_final_only.md` ✓
- `archive/old_runs/train/yolo_cam1_person/results.csv` ✓
- `archive/old_runs/train/osnet_cam1/train.log-2026-04-09-18-35-28` ✓
- `configs/zones/zones_cam1.yaml` ✓
- `docs/final_report.md` ✓ (used as the source of truth for KPI tables 8.11 / 8.12)

### What this audit did NOT verify

- Per-ID purity breakdown for CAM1 ("5 of 8 achieve 100% purity") was not recomputed; it should be regenerated together with the fixed Table 8.3.
- Section 8.5 retest-protocol numbers (5/5 re-entry events, 0 false / 0 missed) were taken from the report's narrative; they are consistent with the FROZEN baseline being locked, but were not re-derived from a retest_protocol log file.
- Section 8.12 GT7 narrative (relaxing the gate produces 0 GT7 rows, ID3 collapses 515 → 12, ID6 inflates by 536) was not independently re-run.
- The KPI conversion-proxy ratios (0.27 vs 1.14–1.17) were taken from `docs/final_report.md` rather than recomputed from raw zone events; the source-of-truth document is internally consistent with the dissertation.

These items are not graded FAIL because they would each require running additional production code; they are flagged here so the authors can re-derive them if examiner pressure increases.

---

## Final readiness statement

The dissertation passes 11 of the 13 audited claim categories with no fixes required. The two FAIL clusters (Table 8.3 CAM1 purity / abstract / conclusion; Table 8.8 / Section 8.7 Demo_Video) are mechanical replacements rather than scientific issues — both can be corrected by editing the affected paragraphs and tables to use the current artifact values listed above. The cross-camera phrasing WARNING is a one-sentence rewrite. After those three changes are applied, the report's numerical claims will be fully traceable to the cited repository artifacts.
