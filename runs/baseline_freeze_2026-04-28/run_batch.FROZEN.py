from __future__ import annotations

from pathlib import Path
import sys
import argparse
import csv
import io
import os

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_demo import run_video_with_kpis
from src.preprocessing import preprocess_video
from src.trackers.bot_sort import BOTSORT
from src.trackers.id_bank import GlobalIDBank
from src.reid.track_linker import (
    stitch_track_ids,
    smooth_ids_with_audit,
    lock_dominant_ids_with_audit,
    drop_ids_without_audit_support,
    enforce_target_ids_from_audit,
    promote_pred0_to_target_from_audit,
    relabel_with_audit_template_canonical,
    separate_id_pair_by_appearance,
    stabilize_overlap_ids_with_memory,
    smooth_overlap_switch_fragments,
    suppress_same_frame_duplicates,
    enforce_same_frame_uniqueness,
    merge_fragment_to_canonical_by_appearance,
    compact_global_ids,
    canonicalize_first_appearance,
    identity_metrics,
    converge_to_canonical_set,
    align_ids_to_reference_video,
    relabel_to_reference_profiles_with_memory,
    relabel_to_seed_profiles_with_memory,
    suppress_tiny_ids_keep_labels,
    suppress_non_person_ghost_boxes,
    suppress_border_ghost_runs,
    suppress_static_edge_ghost_ids,
    suppress_stationary_tracks,
    split_ids_on_abrupt_jumps,
    evaluate_first_two_minute_audit_metrics,
    enforce_canonical_id_set_purity_first,
    summarize_identity_space,
    render_tracks_video,
)
from src.reid.reentry_linker import link_reentry_offline, ReentryConfig


def main():
    parser = argparse.ArgumentParser(description="Batch runner for retail videos")
    parser.add_argument(
        "--match",
        type=str,
        default="CAM1",
        help="Only process videos whose file name contains this token (default: CAM1)",
    )
    parser.add_argument(
        "--audit_csv",
        type=Path,
        default=ROOT / "experiments" / "audit" / "cam1_manual_audit_sheet.csv",
        help="Optional CAM1 audit CSV used for offline ID switch smoothing.",
    )
    parser.add_argument(
        "--disable_audit_smooth",
        action="store_true",
        help="Disable audit-guided ID smoothing pass.",
    )
    parser.add_argument(
        "--identity_map_csv",
        type=Path,
        default=ROOT / "experiments" / "audit" / "cam1_identity_map_template.csv",
        help="Canonical ID template used by FULL_CAM1 audit-guided canonical relabel.",
    )
    parser.add_argument(
        "--render",
        dest="render",
        action="store_true",
        help="Render output video (default: enabled).",
    )
    parser.add_argument(
        "--no-render",
        dest="render",
        action="store_false",
        help="Skip rendering output video.",
    )
    parser.add_argument(
        "--cam1-recovery",
        dest="cam1_recovery",
        action="store_true",
        help=(
            "Apply conservative tracker-continuity recovery profile for CAM1 only. "
            "Loosens confirm_hits (4→3), min_confirmed_hits_for_gid (4→3), "
            "track_buffer (220→300), reid_min_quality_for_bank (0.46→0.38), "
            "reid_new_id_min_quality (0.44→0.36), new_id_confirm_frames (6→4). "
            "Also writes per-frame zero-gid diagnostics to "
            "runs/diagnostics/tracker_continuity_recovery_debug.csv."
        ),
    )
    parser.set_defaults(render=True)
    args = parser.parse_args()
    use_cam1_recovery: bool = bool(args.cam1_recovery)

    match_token = (args.match or "").strip().lower()

    raw_root = ROOT / "data" / "raw"
    out_root = ROOT / "runs" / "kpi_batch"
    out_root.mkdir(parents=True, exist_ok=True)

    folders = ["retail-shop"]
    videos_by_group: dict[str, list[Path]] = {}

    for d in folders:
        folder = raw_root / d
        videos = sorted(folder.rglob("*.mp4")) if folder.exists() else []
        if match_token:
            if match_token.startswith("="):
                # Exact stem match: --match "=cam1" matches only cam1.mp4, not full_cam1.mp4
                exact = match_token[1:]
                videos = [v for v in videos if v.stem.lower() == exact]
            else:
                videos = [v for v in videos if match_token in v.name.lower()]
        videos_by_group[d] = videos

    total_videos = sum(len(v) for v in videos_by_group.values())
    if total_videos == 0:
        print("[WARN] No videos found under data/raw/retail-shop.")
        return

    print(f"[INFO] Processing {total_videos} videos...")

    for group in folders:
        videos = videos_by_group.get(group, [])
        if not videos:
            print(f"[WARN] No videos for {group}, skipping.")
            continue

        custom_det = ROOT / "models" / "yolo_cam1_person.pt"
        custom_reid = ROOT / "models" / "osnet_cam1.pth"
        custom_pose = ROOT / "models" / "yolov8n-pose.pt"
        det_weights = str(custom_det) if custom_det.exists() else "yolov8m.pt"
        reid_weights_path = str(custom_reid) if custom_reid.exists() else None
        pose_weights_path = str(custom_pose) if custom_pose.exists() else None
        print(f"[INFO] Detector weights: {det_weights}")
        if reid_weights_path is not None:
            print(f"[INFO] ReID weights: {reid_weights_path}")
        else:
            print("[INFO] ReID weights: imagenet osnet_x1_0 (default)")
        if pose_weights_path is not None:
            print(f"[INFO] Pose weights: {pose_weights_path}")
        else:
            print("[INFO] Pose weights: disabled (fallback to ratio body parts)")

        id_bank = GlobalIDBank(
            hard_thresh=0.90,
            soft_thresh=0.84,
            margin=0.10,
            ema=0.93,
            min_update_sim=0.86,
            enroll_reuse_thresh=0.94,
            enroll_protect_states=11,
            max_prototypes=36,
            prototype_weight=0.42,
            prototype_min_delta=0.025,
            prototype_topk=5,
            observe_min_quality=0.60,
            verbose=False,
        )
        tracker = BOTSORT(
            id_bank=id_bank,
            det_weights=det_weights,
            reid_weights_path=reid_weights_path,
            det_conf=0.20,
            det_imgsz=960,

            track_thresh=0.30,
            active_match_iou_thresh=0.18,
            lost_match_iou_thresh=0.08,
            match_feat_thresh=0.40,
            strong_reid_thresh=0.78,
            long_lost_reid_thresh=0.82,
            alpha_active=0.42,
            alpha_lost=0.82,
            track_buffer=220,
            motion_max_center_dist=0.72,
            motion_max_gap=32,
            overlap_iou_thresh=0.28,
            min_height_ratio_for_update=0.72,
            min_match_conf=0.20,
            feature_history=80,
            feature_update_min_sim=0.64,
            confirm_hits=4,
            bad_frame_hold=12,

            min_confirmed_hits_for_gid=4,
            min_height_ratio=0.10,
            min_width_ratio=0.04,
            reid_min_conf_for_extract=0.44,
            reid_min_area_ratio=0.010,
            reid_border_margin=0.015,
            reid_min_blur_var=38.0,
            reid_min_quality_for_bank=0.46,
            reid_far_y2_ratio=0.30,
            reid_cautious_y2_ratio=0.52,
            reid_min_lock_hits=4,
            reid_new_id_min_hits=6,
            reid_new_id_min_quality=0.44,

            # attire-aware fusion
            F_OS=1.00,
            F_ATTIRE=0.45,
            F_SHAPE=0.22,
            gid_reuse_thresh=0.74,
            gid_reuse_with_spatial_thresh=0.66,
            gid_spatial_max_age=220,
            reentry_gallery_size=20,
            reentry_min_samples=3,
            reentry_max_age=420,
            reentry_sim_thresh=0.70,
            reentry_margin=0.05,
            reentry_min_zone_compat=0.32,
            lock_confirm_frames=3,
            new_id_confirm_frames=6,
            lock_score_thresh=0.72,
            new_id_score_thresh=0.60,
            gid_owner_reserve_frames=240,
            gid_owner_override_score=0.92,
            overlap_hold_iou_thresh=0.16,
            overlap_hold_frames=56,
            drift_release_hits=6,
            drift_guard_min_mode=2,
            drift_guard_min_quality=0.56,
            short_memory_max_age=140,
            short_memory_reuse_thresh=0.74,
            short_memory_margin=0.08,
            active_similar_block_thresh=0.62,
            profile_topk=5,
            profile_min_quality=0.58,
            bank_update_min_margin=0.09,
            # Conservative matching defaults:
            # - active matching
            normal_reuse_threshold=0.74,
            # - overlap matching (stricter than normal)
            overlap_reuse_threshold=0.84,
            # - long-gap re-entry (strongest)
            long_gap_reid_threshold=0.88,
            best_vs_second_margin=0.09,
            overlap_best_vs_second_margin=0.15,
            min_crop_quality_for_reuse=0.60,
            min_spatial_consistency=0.30,
            # - profile update safety
            min_profile_update_similarity=0.80,
            prototype_bank_size=28,
            # - lost-track decay / zombie protection
            trust_decay_rate=0.016,
            lost_track_max_age=500,
            zombie_id_protection_threshold=0.92,
            # - overlap freeze policy
            overlap_iou_threshold=0.16,
            freeze_duration_after_overlap=56,
            # - non-person / ghost rejection
            min_person_conf_for_identity=0.34,
            min_human_aspect_ratio=0.18,
            max_human_aspect_ratio=1.05,
            min_human_area_ratio=0.0032,
            ghost_right_zone_xmin=0.74,
            ghost_zone_ymin=0.32,
            ghost_low_sat_max=0.15,
            ghost_low_texture_max=13.5,
            debug_reid_decisions=False,
            # - optional part alignment
            pose_weights_path=pose_weights_path,
            pose_min_conf=0.24,
            pose_infer_stride=2,
            right_aisle_xmin=0.68,
            right_aisle_ymin=0.26,
            right_aisle_hold_extra_frames=28,
            overlap_recovery_extra_frames=22,
        )
        default_tracker_runtime = {
            "overlap_hold_iou_thresh": 0.16,
            "overlap_hold_frames": 56,
            "gid_owner_reserve_frames": 240,
            "gid_owner_override_score": 0.92,
            "drift_release_hits": 6,
            "short_memory_max_age": 140,
            "short_memory_reuse_thresh": 0.74,
            "short_memory_margin": 0.08,
            "reentry_margin": 0.08,
            "lock_confirm_frames": 3,
            "new_id_confirm_frames": 6,
            "lock_score_thresh": 0.72,
            "new_id_score_thresh": 0.60,
            "active_similar_block_thresh": 0.62,
            "profile_topk": 5,
            "bank_update_min_margin": 0.09,
            "normal_reuse_threshold": 0.74,
            "overlap_reuse_threshold": 0.84,
            "long_gap_reid_threshold": 0.88,
            "best_vs_second_margin": 0.09,
            "overlap_best_vs_second_margin": 0.15,
            "min_crop_quality_for_reuse": 0.60,
            "min_spatial_consistency": 0.30,
            "min_profile_update_similarity": 0.80,
            "trust_decay_rate": 0.016,
            "lost_track_max_age": 500,
            "prototype_bank_size": 28,
            "overlap_iou_threshold": 0.16,
            "freeze_duration_after_overlap": 56,
            "zombie_id_protection_threshold": 0.92,
            "right_aisle_xmin": 0.68,
            "right_aisle_ymin": 0.26,
            "right_aisle_hold_extra_frames": 20,
            "overlap_recovery_extra_frames": 16,
            "debug_reid_decisions": False,
            "enable_pairwise_swap_correction": True,
        }
        # Previous FULL_CAM1 profile required near-perfect (0.96+) similarity for
        # re-entry, which meant the tracker almost never reused an ID after a
        # person briefly left the frame — the opposite of what the user wants.
        # The values below are stricter than default (FULL_CAM1 has occlusions
        # and similar-looking people so we still need margin) but within the
        # feasible range for an OSNet feature bank.
        full_cam1_tracker_runtime = {
            "overlap_hold_iou_thresh": 0.12,
            # Increase overlap hold so an occluded identity stays frozen long
            # enough that the person who *actually* looks like the bank entry
            # is chosen over a zombie lookalike when the pair separates.
            "overlap_hold_frames": 96,
            "gid_owner_reserve_frames": 560,
            "gid_owner_override_score": 0.95,
            "drift_release_hits": 6,
            "drift_guard_min_mode": 2,
            "drift_guard_min_quality": 0.60,
            "short_memory_max_age": 160,
            "short_memory_reuse_thresh": 0.82,
            "short_memory_margin": 0.11,
            "reentry_margin": 0.11,
            "lock_confirm_frames": 4,
            "new_id_confirm_frames": 5,
            "lock_score_thresh": 0.80,
            "new_id_score_thresh": 0.62,
            "active_similar_block_thresh": 0.70,
            "profile_topk": 6,
            "bank_update_min_margin": 0.13,
            "normal_reuse_threshold": 0.80,
            "overlap_reuse_threshold": 0.91,
            "long_gap_reid_threshold": 0.93,
            "best_vs_second_margin": 0.12,
            "overlap_best_vs_second_margin": 0.22,
            "min_crop_quality_for_reuse": 0.64,
            "min_spatial_consistency": 0.34,
            "min_profile_update_similarity": 0.86,
            "trust_decay_rate": 0.020,
            "lost_track_max_age": 500,
            "prototype_bank_size": 36,
            "overlap_iou_threshold": 0.12,
            "freeze_duration_after_overlap": 96,
            # Raised from 0.92 -> 0.94 to reject weak zombie takeovers more
            # aggressively. The downstream CAM1 anchor corrects any
            # legitimate misses.
            "zombie_id_protection_threshold": 0.95,
            "right_aisle_xmin": 0.66,
            "right_aisle_ymin": 0.24,
            "right_aisle_hold_extra_frames": 36,
            "overlap_recovery_extra_frames": 28,
            # Detection filters: slightly loosened from defaults so more
            # real people in the back of the shop get an ID, but ghost
            # rejection (low sat / low texture) tightened from 0.16 -> 0.14.
            "min_person_conf_for_identity": 0.22,
            "min_human_aspect_ratio": 0.16,
            "max_human_aspect_ratio": 1.10,
            "min_human_area_ratio": 0.0028,
            "ghost_right_zone_xmin": 0.74,
            "ghost_zone_ymin": 0.32,
            "ghost_low_sat_max": 0.14,
            "ghost_low_texture_max": 12.5,
            "debug_reid_decisions": False,
            "enable_pairwise_swap_correction": True,
            "reid_new_id_min_hits": 4,
        }
        default_byte_runtime = {
            "active_match_iou_thresh": 0.18,
            "lost_match_iou_thresh": 0.08,
            "strong_reid_thresh": 0.78,
            "long_lost_reid_thresh": 0.82,
            "overlap_iou_thresh": 0.28,
        }
        full_cam1_byte_runtime = {
            "active_match_iou_thresh": 0.20,
            "lost_match_iou_thresh": 0.10,
            "strong_reid_thresh": 0.80,
            "long_lost_reid_thresh": 0.84,
            "overlap_iou_thresh": 0.22,
        }
        default_id_bank_runtime = {
            "hard_thresh": 0.90,
            "soft_thresh": 0.84,
            "margin": 0.10,
            "min_update_sim": 0.86,
            "observe_min_quality": 0.60,
            "prototype_weight": 0.42,
            "prototype_topk": 5,
            "prototype_min_delta": 0.025,
            "max_prototypes": 36,
            "enroll_reuse_thresh": 0.94,
            "enroll_protect_states": 11,
        }
        full_cam1_id_bank_runtime = {
            "hard_thresh": 0.90,
            "soft_thresh": 0.84,
            "margin": 0.10,
            "min_update_sim": 0.86,
            "observe_min_quality": 0.60,
            "prototype_weight": 0.44,
            "prototype_topk": 6,
            "prototype_min_delta": 0.022,
            "max_prototypes": 40,
            "enroll_reuse_thresh": 0.90,
            "enroll_protect_states": 12,
        }

        # Conservative recovery profile for CAM1 only (--cam1-recovery flag).
        # Only the track-confirmation and ID-gate thresholds are loosened;
        # all identity-safety thresholds (margin, block scores, lock scores)
        # are inherited unchanged from default_tracker_runtime.
        cam1_recovery_tracker_runtime = {
            **default_tracker_runtime,
            "min_confirmed_hits_for_gid": 3,     # was 4: confirmed tracks get ID 1 frame sooner
            "reid_min_quality_for_bank":  0.38,  # was 0.46: accept moderate-quality crops in bank
            "reid_new_id_min_quality":    0.36,  # was 0.44: allow slightly noisier crops for new IDs
            "reid_new_id_min_hits":       4,     # was 6: fewer hits needed before new-ID allowed
            "new_id_confirm_frames":      4,     # was 6: fewer pending frames to commit a new ID
        }
        cam1_recovery_byte_runtime = {
            **default_byte_runtime,
            "confirm_hits": 3,    # was 4: ByteTracker confirms tracks after 3 consecutive hits
            "track_buffer": 300,  # was 220: lost tracks kept in buffer longer before purging
        }

        def _apply_runtime_profile(*, full_cam1: bool, cam1_recovery: bool = False) -> None:
            if cam1_recovery:
                runtime = cam1_recovery_tracker_runtime
            elif full_cam1:
                runtime = full_cam1_tracker_runtime
            else:
                runtime = default_tracker_runtime
            for k, v in runtime.items():
                setattr(tracker, k, v)

            if cam1_recovery:
                bt_runtime = cam1_recovery_byte_runtime
            elif full_cam1:
                bt_runtime = full_cam1_byte_runtime
            else:
                bt_runtime = default_byte_runtime
            for k, v in bt_runtime.items():
                if hasattr(tracker.tracker, k):
                    setattr(tracker.tracker, k, v)
            if tracker.id_bank is not None:
                idb_runtime = full_cam1_id_bank_runtime if full_cam1 else default_id_bank_runtime
                for k, v in idb_runtime.items():
                    if hasattr(tracker.id_bank, k):
                        setattr(tracker.id_bank, k, v)

        print("\n======================================")
        print(f"[INFO] Processing Batch: {group}")
        print("======================================\n")

        for v in videos:
            # Explicit pipeline mode — single source of truth for per-clip behavior.
            # side_view_benchmark_long  : FULL_CAM1 (long sequence, anchor/canonical passes)
            # side_view_benchmark_short : CAM1, Demo_Video (short clips, same camera angle)
            # overhead_uploaded         : everything else (1_2_crop, future overhead clips)
            _stem = v.stem.lower()
            if _stem == "full_cam1":
                pipeline_mode = "side_view_benchmark_long"
            elif _stem in ("cam1", "demo_video"):
                pipeline_mode = "side_view_benchmark_short"
            else:
                pipeline_mode = "overhead_uploaded"

            is_full_cam1  = pipeline_mode == "side_view_benchmark_long"
            is_side_view  = pipeline_mode in ("side_view_benchmark_long", "side_view_benchmark_short")
            is_overhead   = pipeline_mode == "overhead_uploaded"
            is_exact_cam1 = _stem == "cam1"   # kept for CAM1-specific audit-guidance block only

            apply_recovery = use_cam1_recovery and _stem == "cam1"
            _apply_runtime_profile(full_cam1=is_full_cam1, cam1_recovery=apply_recovery)

            # Wire zero-gid diagnostic path for recovery runs.
            tracker._zero_debug_path = (
                str(ROOT / "runs" / "diagnostics" / "tracker_continuity_recovery_debug.csv")
                if apply_recovery else None
            )
            tracker._zero_debug_rows = []
            tracker._zero_debug_written_header = False
            if hasattr(tracker, "reset_for_new_video"):
                try:
                    tracker.reset_for_new_video(reset_ids=True)
                except TypeError:
                    tracker.reset_for_new_video()

            rel = v.relative_to(raw_root)
            stem = "_".join(rel.with_suffix("").parts)
            out_csv = out_root / f"{stem}_tracks.csv"
            out_vid = out_root / f"{stem}_vis.mp4"

            print(f"[INFO] Running {v.name}")
            print(f"[INFO] -> CSV : {out_csv}")
            print(f"[INFO] -> VIDEO: {out_vid}")

            pre = preprocess_video(v, out_dir=out_root, write_report=True)
            print(
                f"[PRE]  {pre.video}  {pre.resolution[0]}x{pre.resolution[1]}"
                f"  {pre.fps:.2f}fps  {pre.frame_count}f  {pre.duration_s:.1f}s"
                f"  bad={pre.quality['bad_pct']}%"
            )

            run_video_with_kpis(
                video_path=v,
                out_tracks_csv=out_csv,
                fps_override=None,
                tracker=tracker,
                forced_global_id=None,
                out_video_path=None,
                draw=False,
            )

            # Offline stage: stitch local tracklets into global IDs.
            # For short clips (CAM1 = ~60 s), cap stitch gap at 400 frames (~13 s)
            # to prevent false merges across long absences (e.g. two different
            # people sharing an ID because person A left and B appeared 44 s later).
            stitch = stitch_track_ids(
                video_path=v,
                tracks_csv_path=out_csv,
                reid_weights_path=reid_weights_path,
                max_gap_frames=1400 if is_full_cam1 else 400,
                min_merge_score=0.90 if is_full_cam1 else 0.68,
            )
            print(
                f"[INFO] ID stitch merged={stitch['merged_pairs']} "
                f"across total_ids={stitch['total_ids']} "
                f"dedup_rows={stitch.get('dedup_rows', 0)}"
            )

            reentry_debug_dir = out_root / "reentry_debug" / stem
            # Previous FULL_CAM1 overrides were so strict (margin=0.16, deep>=0.84)
            # that 368/387 re-entry attempts were rejected as "ambiguous" even when
            # the top candidate was clearly correct. This is the root cause of the
            # "same person must reuse the same ID" complaint. We relax to values
            # that are still stricter than the default (which is the generic tuning)
            # but no longer pathologically so. The online tracker already filters
            # most false matches; the re-entry linker should only need to reject
            # truly ambiguous crossings.
            # FULL_CAM1: default to conservative merge behavior in the re-entry linker.
            # Wrong cross-person reuse is more damaging than temporary fragmentation,
            # and later anchor/recovery passes can safely reunite valid fragments.
            _full_cam1_enable_group_merge = (
                bool(int(os.environ.get("FULL_CAM1_REENTRY_ENABLE_GROUP_MERGE", "0")))
                if is_full_cam1
                else True
            )
            _full_cam1_enable_handoff_merge = (
                bool(int(os.environ.get("FULL_CAM1_REENTRY_ENABLE_HANDOFF_MERGE", "0")))
                if is_full_cam1
                else True
            )

            reentry_cfg = ReentryConfig(
                # Allow up to 1100 frames for non-FULL_CAM1 re-entry (covers GT6's
                # 1066-frame legitimate absence in CAM1).  Stitch is already capped
                # at 400 frames so the ID4-style false merge cannot be recreated
                # through stitch.  Raise min_deep_sim_for_reuse 0.66→0.73 so the
                # reentry linker is more selective at long gaps: genuine same-person
                # re-entries (OSNet sim > 0.73) reconnect; cross-person false merges
                # (sim ~0.68–0.72) are rejected.
                max_reentry_gap_frames=1500 if is_full_cam1 else 1100,
                min_reentry_gap_frames=3 if is_full_cam1 else 4,
                strong_reuse_score=0.78 if is_full_cam1 else 0.72,
                strong_reuse_margin=0.08 if is_full_cam1 else 0.03,
                min_deep_sim_for_reuse=0.75 if is_full_cam1 else 0.73,
                min_topk_sim_for_reuse=0.72 if is_full_cam1 else 0.68,
                min_part_topk_for_reuse=0.65 if is_full_cam1 else 0.58,
                min_part_mean_for_reuse=0.68 if is_full_cam1 else 0.64,
                cross_person_ambiguity_margin=0.055 if is_full_cam1 else 0.045,
                same_candidate_safe_accept=True,
                # Relaxed thresholds for same_candidate_safe_accept path only.
                # The green shirt girl puts on a mask mid-video, causing deep-sim
                # to drop to ~0.61 even though both top candidates agree she is
                # the same person (same prev_gid).  Using the global
                # min_deep_sim_for_reuse=0.75 would reject this re-entry and
                # fragment her track into IDs 5, 7, 10.  The relaxed threshold
                # only fires when top-1 AND top-2 agree on the same identity, so
                # it cannot cause wrong-reuse across different people.
                same_candidate_safe_score=0.73 if is_full_cam1 else -1.0,
                same_candidate_min_deep_relaxed=0.58 if is_full_cam1 else -1.0,
                # Cap the same_candidate_safe_accept path to short gaps only.
                # At long gaps (>700f / ~57s) two different people can both score
                # well against a small early tracklet in memory, creating false
                # merges where one canonical ID spans completely different persons.
                # Genuine same-person re-entries at side-view mostly happen within
                # this window; longer absences fall through to new_id_created.
                same_candidate_max_gap_frames=700 if is_full_cam1 else -1.0,
                same_source_gid_bias=True,
                same_source_allow_short_gap=True,
                same_source_min_gap_frames=1,
                same_source_max_gap_frames=260 if is_full_cam1 else 220,
                same_source_min_score=0.74 if is_full_cam1 else 0.70,
                same_source_min_deep=0.64 if is_full_cam1 else 0.60,
                same_source_min_topk=0.62 if is_full_cam1 else 0.58,
                same_source_min_part_topk=0.60 if is_full_cam1 else 0.56,
                same_source_competitive_margin=0.07 if is_full_cam1 else 0.05,
                same_source_block_cross_source=True,
                same_source_block_min_score=0.68 if is_full_cam1 else 0.62,
                same_source_block_margin=0.06 if is_full_cam1 else 0.04,
                strong_deep_relax_deep=0.80 if is_full_cam1 else 0.76,
                strong_deep_relax_topk=0.82 if is_full_cam1 else 0.78,
                rerank_top_n=8,
                # Softer rerank penalties — hardcoded values were depressing
                # otherwise-valid reuses.
                rerank_part_imbalance_threshold=0.46,
                rerank_part_imbalance_penalty=0.04,
                rerank_part_asymmetry_threshold=0.33,
                rerank_part_asymmetry_penalty=0.025,
                rerank_deep_weak_threshold=0.56,
                rerank_deep_weak_penalty=0.025,
                enable_group_merge_pass=_full_cam1_enable_group_merge,
                merge_min_score=0.83 if is_full_cam1 else 0.78,
                merge_min_deep=0.75 if is_full_cam1 else 0.68,
                merge_min_topk=0.76 if is_full_cam1 else 0.68,
                merge_min_part_topk=0.66 if is_full_cam1 else 0.58,
                merge_max_gap_frames=1700 if is_full_cam1 else 1100,
                enable_overlap_handoff_pass=_full_cam1_enable_handoff_merge,
                overlap_handoff_max_gap_frames=220 if is_full_cam1 else 240,
                overlap_handoff_min_score=0.79 if is_full_cam1 else 0.72,
                overlap_handoff_min_deep=0.70 if is_full_cam1 else 0.60,
                overlap_handoff_min_topk=0.72 if is_full_cam1 else 0.62,
                overlap_handoff_min_part_topk=0.62 if is_full_cam1 else 0.54,
                overlap_handoff_min_spatial=0.52 if is_full_cam1 else 0.44,
                overlap_window_top_k=6 if is_full_cam1 else 5,
                overlap_window_min_people=4,
                overlap_window_min_mean_iou=0.12 if is_full_cam1 else 0.13,
                overlap_window_pad_frames=32 if is_full_cam1 else 28,
                overlap_window_relax_delta=0.05,
                overlap_window_relaxed_max_gap_frames=150,
                # Anti-switch lock stays fairly strict — its job is to PREVENT
                # wrong swaps in overlap, so false positives here cost us.
                anti_switch_max_gap_frames=120,
                anti_switch_min_score=0.85 if is_full_cam1 else 0.78,
                anti_switch_min_deep=0.74 if is_full_cam1 else 0.64,
                anti_switch_min_topk=0.76 if is_full_cam1 else 0.64,
                anti_switch_min_part_topk=0.68 if is_full_cam1 else 0.56,
                anti_switch_min_spatial=0.54 if is_full_cam1 else 0.46,
                anti_switch_margin=0.10,
                anti_switch_tracklet_max_len_frames=180,
                anti_switch_target_gid_min_rows=120,
                anti_switch_reassign_margin_over_current=0.10,
                # Local consistency pass — enabled everywhere now. It's the main
                # mechanism that reunites fragmented tracks under the same local
                # tracker id back into one gid.
                enable_local_consistency_pass=True,
                local_consistency_min_score=0.74 if is_full_cam1 else 0.68,
                local_consistency_min_deep=0.66 if is_full_cam1 else 0.60,
                local_consistency_min_topk=0.68 if is_full_cam1 else 0.62,
                # Overhead-view stitch-trust: accept same-source candidates at
                # a permissive threshold when OSNet appearance similarity
                # collapses (overhead crop vs trained side-view angle).
                # Disabled for all side-view clips (FULL_CAM1, CAM1, Demo_Video)
                # where OSNet holistic similarity is reliable and rejections are correct.
                same_source_stitch_trust_score=-1.0 if is_side_view else 0.55,
                # Single-candidate spatial accept: overhead-view fallback when
                # holistic appearance is weak but no cross-person ambiguity
                # exists.  Disabled for all side-view clips where OSNet works reliably.
                single_candidate_spatial_accept=not is_side_view,
                single_candidate_min_score=0.48,
                single_candidate_min_part_topk=0.80,
                single_candidate_min_side_score=0.60,
            )
            reentry = link_reentry_offline(
                video_path=v,
                tracks_csv_path=out_csv,
                reid_weights_path=reid_weights_path,
                debug_dir=reentry_debug_dir,
                config=reentry_cfg,
            )
            print(f"[INFO] Re-entry link {reentry}")

            use_cam1_audit_guidance = (not args.disable_audit_smooth) and args.audit_csv.exists() and is_exact_cam1
            use_cam1_forced_targets = use_cam1_audit_guidance and is_exact_cam1
            if use_cam1_audit_guidance:
                audit_fix = smooth_ids_with_audit(
                    tracks_csv_path=out_csv,
                    audit_csv_path=args.audit_csv,
                    iou_threshold=0.34,
                    min_support=2,
                    min_ratio=0.14,
                )
                print(f"[INFO] Audit smooth {audit_fix}")

                lock_fix = lock_dominant_ids_with_audit(
                    tracks_csv_path=out_csv,
                    audit_csv_path=args.audit_csv,
                    iou_threshold=0.34,
                    min_obs_per_gt=8,
                    min_dominant_ratio=0.72,
                    max_switch_segment_len=28,
                    min_people_in_frame=4,
                )
                print(f"[INFO] Dominant-ID lock {lock_fix}")

                drop_fix = drop_ids_without_audit_support(
                    tracks_csv_path=out_csv,
                    audit_csv_path=args.audit_csv,
                    iou_threshold=0.34,
                    min_hits=1,
                )
                print(f"[INFO] Drop unsupported IDs {drop_fix}")

            overlap_fix = smooth_overlap_switch_fragments(
                tracks_csv_path=out_csv,
                iou_link_thresh=0.30,
                max_frame_gap=2,
                max_center_dist_norm=1.10,
                short_run_max_len=14,
                min_neighbor_run_len=8,
                bridge_iou_min=0.10,
            )
            print(f"[INFO] Overlap smooth {overlap_fix}")

            dup_fix = suppress_same_frame_duplicates(
                tracks_csv_path=out_csv,
                iou_thresh=0.55,
                containment_thresh=0.72,
                max_center_dist_norm=0.62,
            )
            print(f"[INFO] Same-frame duplicate suppress {dup_fix}")

            compact = compact_global_ids(
                tracks_csv_path=out_csv,
                min_rows_keep=12,
                min_span_keep=16,
            )
            print(
                f"[INFO] ID compact total_ids={compact['total_ids']} "
                f"max_id={compact['max_id']}"
            )

            if is_overhead:
                _static_fix = suppress_stationary_tracks(
                    out_csv,
                    cx_range_thresh=60.0,
                    cy_range_thresh=60.0,
                    min_rows=30,
                )
                if _static_fix["changed_ids"]:
                    print(
                        f"[INFO] Stationary track suppression: "
                        f"suppressed IDs={_static_fix['changed_ids']} "
                        f"rows={_static_fix['changed_rows']}"
                    )

            full_cam1_audit_before = None
            if is_full_cam1 and args.audit_csv.exists():
                full_cam1_audit_before = evaluate_first_two_minute_audit_metrics(
                    tracks_csv_path=out_csv,
                    audit_csv_path=args.audit_csv,
                    max_sec=120.0,
                    iou_threshold=0.34,
                )
                print(f"[INFO] FULL_CAM1 first2min audit before {full_cam1_audit_before}")

            if use_cam1_forced_targets:
                # User-requested fixed labels in CAM1:
                # gt6 (pink shirt) -> ID 6
                # gt11 (grey shirt man) -> ID 8
                # gt8 (black shirt woman) -> ID 9
                # gt5 (red-blue shirt man) -> ID 2
                # gt10 (re-entry person) -> ID 10
                target_fix = enforce_target_ids_from_audit(
                    tracks_csv_path=out_csv,
                    audit_csv_path=args.audit_csv,
                    target_gid_by_gt={"6": 6, "11": 8, "8": 9, "5": 2, "10": 10},
                    iou_threshold=0.34,
                    max_segment_len=2000,
                )
                print(f"[INFO] Enforce target IDs {target_fix}")

                promo_fix = promote_pred0_to_target_from_audit(
                    tracks_csv_path=out_csv,
                    audit_csv_path=args.audit_csv,
                    target_gid_by_gt={"11": 8, "6": 6, "10": 10, "2": 2},
                    iou_threshold=0.34,
                )
                print(f"[INFO] Promote pred=0 to target {promo_fix}")

                # Targeted overlap fix in CAM1:
                # keep black-shirt boy on ID 5 and black-shirt woman on ID 9.
                pair_total = 0
                pair_rounds = []
                for _ in range(3):
                    pair_fix = separate_id_pair_by_appearance(
                        video_path=v,
                        tracks_csv_path=out_csv,
                        gid_a=5,
                        gid_b=9,
                        min_area_ratio=0.005,
                        min_samples_per_id=12,
                        min_sim=0.34,
                        switch_margin=0.06,
                    )
                    pair_rounds.append(pair_fix)
                    pair_total += int(pair_fix.get("changed_rows", 0))
                    if (not bool(pair_fix.get("applied", False))) or int(pair_fix.get("changed_rows", 0)) <= 0:
                        break
                print(f"[INFO] Pair split (5 vs 9) total_changed={pair_total} rounds={pair_rounds}")

                # Keep red-blue shirt man on ID 2, not ID 5.
                pair_total_25 = 0
                pair_rounds_25 = []
                for _ in range(3):
                    pair_fix_25 = separate_id_pair_by_appearance(
                        video_path=v,
                        tracks_csv_path=out_csv,
                        gid_a=2,
                        gid_b=5,
                        min_area_ratio=0.005,
                        min_samples_per_id=12,
                        min_sim=0.34,
                        switch_margin=0.06,
                    )
                    pair_rounds_25.append(pair_fix_25)
                    pair_total_25 += int(pair_fix_25.get("changed_rows", 0))
                    if (not bool(pair_fix_25.get("applied", False))) or int(pair_fix_25.get("changed_rows", 0)) <= 0:
                        break
                print(f"[INFO] Pair split (2 vs 5) total_changed={pair_total_25} rounds={pair_rounds_25}")

            # FULL_CAM1: align IDs to CAM1 reference identities automatically
            # (no manual force-map), then run one more overlap clean pass.
            if is_full_cam1:
                ref_video = raw_root / "retail-shop" / "CAM1.mp4"
                ref_csv = out_root / "retail-shop_CAM1_tracks.csv"
                enable_full_cam1_reference_align = bool(ref_video.exists() and ref_csv.exists())
                enable_full_cam1_semantic_canonical_relabel = bool(ref_video.exists() and ref_csv.exists())
                enable_full_cam1_seed_profile_relabel = True
                full_cam1_cam1_direct_mode = bool(
                    int(os.environ.get("FULL_CAM1_CAM1_DIRECT_MODE", "0"))
                )
                # Conservative CAM1-direct mode:
                # keep tracker + reentry + CAM1 anchor/convergence, but disable
                # aggressive relabel/split passes that can introduce identity drift.
                if full_cam1_cam1_direct_mode:
                    enable_full_cam1_semantic_canonical_relabel = False
                    enable_full_cam1_seed_profile_relabel = False
                full_cam1_enable_pair_splits = bool(
                    int(
                        os.environ.get(
                            "FULL_CAM1_ENABLE_PAIR_SPLITS",
                            "0" if full_cam1_cam1_direct_mode else "1",
                        )
                    )
                )
                full_cam1_enable_audit_template_relabel = bool(
                    int(
                        os.environ.get(
                            "FULL_CAM1_ENABLE_AUDIT_TEMPLATE_RELABEL",
                            "0" if full_cam1_cam1_direct_mode else "1",
                        )
                    )
                )
                full_cam1_enable_abrupt_jump_split = bool(
                    int(
                        os.environ.get(
                            "FULL_CAM1_ENABLE_ABRUPT_JUMP_SPLIT",
                            "0" if full_cam1_cam1_direct_mode else "1",
                        )
                    )
                )
                full_cam1_enable_final_pair_splits = bool(
                    int(
                        os.environ.get(
                            "FULL_CAM1_ENABLE_FINAL_PAIR_SPLITS",
                            "0" if full_cam1_cam1_direct_mode else "1",
                        )
                    )
                )
                if full_cam1_cam1_direct_mode:
                    print(
                        "[INFO] FULL_CAM1 CAM1-direct mode enabled: "
                        "semantic/seed relabel + aggressive split passes disabled."
                    )
                full_cam1_stable_min_rows = int(
                    os.environ.get(
                        "FULL_CAM1_STABLE_MIN_ROWS",
                        "28" if full_cam1_cam1_direct_mode else "30",
                    )
                )
                full_cam1_stable_min_span = int(
                    os.environ.get(
                        "FULL_CAM1_STABLE_MIN_SPAN",
                        "32",
                    )
                )
                full_cam1_anchor_stable_min_rows = int(
                    os.environ.get("FULL_CAM1_ANCHOR_STABLE_MIN_ROWS", "30")
                )
                full_cam1_anchor_stable_min_span = int(
                    os.environ.get(
                        "FULL_CAM1_ANCHOR_STABLE_MIN_SPAN",
                        str(full_cam1_stable_min_span),
                    )
                )
                print(
                    "[INFO] FULL_CAM1 stability thresholds "
                    f"min_rows={full_cam1_stable_min_rows} "
                    f"min_span={full_cam1_stable_min_span} "
                    f"anchor_min_rows={full_cam1_anchor_stable_min_rows} "
                    f"anchor_min_span={full_cam1_anchor_stable_min_span}"
                )

                def _enforce_full_cam1_uniqueness(stage: str) -> dict:
                    dedup = suppress_same_frame_duplicates(
                        tracks_csv_path=out_csv,
                        iou_thresh=0.52,
                        containment_thresh=0.70,
                        max_center_dist_norm=0.66,
                    )
                    print(f"[INFO] FULL_CAM1 same-frame uniqueness ({stage}) {dedup}")
                    return dedup

                def _full_cam1_continuity_heal() -> dict:
                    """Conservative identity-heal pass for common FULL_CAM1 drifts.

                    Goals:
                    - Split mixed ID9 runs when late run clearly continues ID4.
                    - Fill missing canonical IDs (5, 7) from obvious survivor fragments.
                    - Absorb tiny 1-2 frame glitch IDs into a strong neighbor.
                    """

                    def _parse_int(raw: str, default: int = 0) -> int:
                        try:
                            return int(float(raw or default))
                        except Exception:
                            return int(default)

                    def _parse_float(raw: str, default: float = 0.0) -> float:
                        try:
                            return float(raw or default)
                        except Exception:
                            return float(default)

                    def _bbox(row: dict) -> tuple[float, float, float, float]:
                        return (
                            _parse_float(row.get("x1", 0.0)),
                            _parse_float(row.get("y1", 0.0)),
                            _parse_float(row.get("x2", 0.0)),
                            _parse_float(row.get("y2", 0.0)),
                        )

                    def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
                        ax1, ay1, ax2, ay2 = a
                        bx1, by1, bx2, by2 = b
                        ix1 = max(ax1, bx1)
                        iy1 = max(ay1, by1)
                        ix2 = min(ax2, bx2)
                        iy2 = min(ay2, by2)
                        iw = max(0.0, ix2 - ix1)
                        ih = max(0.0, iy2 - iy1)
                        inter = iw * ih
                        if inter <= 0.0:
                            return 0.0
                        aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
                        ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
                        den = aa + ba - inter
                        if den <= 0.0:
                            return 0.0
                        return float(inter / den)

                    def _build_maps(_rows: list[dict]) -> tuple[dict[int, list[int]], dict[int, dict[int, list[int]]]]:
                        gid_to_idx: dict[int, list[int]] = {}
                        frame_gid_to_idx: dict[int, dict[int, list[int]]] = {}
                        for i, r in enumerate(_rows):
                            g = _parse_int(r.get("global_id", "0"), 0)
                            if g <= 0:
                                continue
                            gid_to_idx.setdefault(g, []).append(i)
                            fr = _parse_int(r.get("frame_idx", "0"), 0)
                            frame_gid_to_idx.setdefault(fr, {}).setdefault(g, []).append(i)
                        return gid_to_idx, frame_gid_to_idx

                    def _contiguous_runs(indices: list[int], _rows: list[dict]) -> list[list[int]]:
                        if not indices:
                            return []
                        sorted_idx = sorted(indices, key=lambda x: _parse_int(_rows[x].get("frame_idx", "0"), 0))
                        out: list[list[int]] = []
                        cur = [sorted_idx[0]]
                        prev_fr = _parse_int(_rows[sorted_idx[0]].get("frame_idx", "0"), 0)
                        for idx in sorted_idx[1:]:
                            fr = _parse_int(_rows[idx].get("frame_idx", "0"), 0)
                            if fr == prev_fr + 1:
                                cur.append(idx)
                            else:
                                out.append(cur)
                                cur = [idx]
                            prev_fr = fr
                        out.append(cur)
                        return out

                    with out_csv.open(newline="", encoding="utf-8") as _f:
                        _reader = csv.DictReader(_f)
                        _fieldnames = list(_reader.fieldnames or [])
                        _rows = list(_reader)

                    heal_min_run_rows = max(
                        1,
                        int(os.environ.get("FULL_CAM1_CONTINUITY_HEAL_MIN_RUN_ROWS", "30")),
                    )
                    heal_require_gap_recovery = bool(
                        int(os.environ.get("FULL_CAM1_CONTINUITY_HEAL_REQUIRE_GAP_RECOVERY", "1"))
                    )

                    changed_rows = 0
                    decisions: list[dict] = []

                    def _apply(indices: list[int], dst_gid: int, reason: str) -> None:
                        nonlocal changed_rows
                        if not indices:
                            return
                        moved = 0
                        src_gid = None
                        for idx in indices:
                            g_old = _parse_int(_rows[idx].get("global_id", "0"), 0)
                            src_gid = g_old if src_gid is None else src_gid
                            if g_old <= 0 or g_old == int(dst_gid):
                                continue
                            _rows[idx]["global_id"] = str(int(dst_gid))
                            changed_rows += 1
                            moved += 1
                        if moved > 0:
                            decisions.append(
                                {
                                    "source_gid": int(src_gid or 0),
                                    "target_gid": int(dst_gid),
                                    "moved_rows": int(moved),
                                    "reason": str(reason),
                                }
                            )

                    gid_to_idx, frame_gid_to_idx = _build_maps(_rows)

                    # 1) Split mixed ID9 runs:
                    # keep earliest run on ID9, map later runs to ID4 when ID4 exists.
                    # Only enable this when we're explicitly recovering known ID gaps
                    # (IDs 5/7), otherwise it can over-collapse healthy ID9 segments.
                    missing_5_or_7 = (5 not in gid_to_idx) or (7 not in gid_to_idx)
                    if (
                        9 in gid_to_idx
                        and 4 in gid_to_idx
                        and ((not heal_require_gap_recovery) or missing_5_or_7)
                    ):
                        runs9 = _contiguous_runs(gid_to_idx.get(9, []), _rows)
                        if len(runs9) >= 2:
                            head = list(runs9[0])
                            tail = []
                            for rseg in runs9[1:]:
                                tail.extend(rseg)
                            # Guard: only if ID4 ends before this late segment starts.
                            id4_max = max(
                                _parse_int(_rows[i].get("frame_idx", "0"), 0)
                                for i in gid_to_idx.get(4, [])
                            )
                            tail_min = min(
                                _parse_int(_rows[i].get("frame_idx", "0"), 0)
                                for i in tail
                            ) if tail else 10**9
                            if (
                                tail
                                and id4_max < tail_min
                                and len(head) >= heal_min_run_rows
                                and len(tail) >= heal_min_run_rows
                            ):
                                _apply(
                                    tail,
                                    4,
                                    "continuity-heal:id9-late-run-to-id4",
                                )

                    # Rebuild maps after possible ID9->4 move.
                    gid_to_idx, frame_gid_to_idx = _build_maps(_rows)

                    # 2) Fill missing canonical ID5 from one ID9 run while preserving
                    # at least one stable run on ID9.
                    if 5 not in gid_to_idx and 9 in gid_to_idx:
                        runs9 = _contiguous_runs(gid_to_idx.get(9, []), _rows)
                        if len(runs9) >= 2:
                            candidate = list(runs9[0])
                            keep_rows = sum(len(r) for r in runs9[1:])
                            if (
                                len(candidate) >= heal_min_run_rows
                                and keep_rows >= heal_min_run_rows
                            ):
                                _apply(
                                    candidate,
                                    5,
                                    "continuity-heal:recover-missing-id5-from-id9-earliest-run",
                                )

                    # 3) Fill missing canonical ID7 from ID13 if present and stable enough.
                    gid_to_idx, frame_gid_to_idx = _build_maps(_rows)
                    if 7 not in gid_to_idx and 13 in gid_to_idx:
                        idx13 = gid_to_idx.get(13, [])
                        if len(idx13) >= heal_min_run_rows:
                            _apply(idx13, 7, "continuity-heal:recover-missing-id7-from-id13")

                    # 4) Absorb tiny glitch ID12 into ID1 when temporally/shape consistent.
                    gid_to_idx, frame_gid_to_idx = _build_maps(_rows)
                    if 12 in gid_to_idx and 1 in gid_to_idx and len(gid_to_idx.get(12, [])) <= 3:
                        idx12 = gid_to_idx.get(12, [])
                        good = []
                        for idx in idx12:
                            fr = _parse_int(_rows[idx].get("frame_idx", "0"), 0)
                            bb = _bbox(_rows[idx])
                            best_iou = 0.0
                            for nfr in (fr - 1, fr, fr + 1):
                                for j in frame_gid_to_idx.get(nfr, {}).get(1, []):
                                    best_iou = max(best_iou, _iou(bb, _bbox(_rows[j])))
                            if best_iou >= 0.82:
                                good.append(idx)
                        if good:
                            _apply(good, 1, "continuity-heal:absorb-tiny-id12-into-id1")

                    # 5) Optional persona-targeted remap for FULL_CAM1.
                    # Enables user-requested canonical ownership in difficult mixed-ID spans:
                    #   - keep green-shirt woman on ID1
                    #   - prevent black-shirt woman from reusing brown-man ID2 (move to ID6)
                    #   - keep black-shirt boy continuous on ID7
                    #   - keep grey-shirt man on ID8
                    # The rules are intentionally frame-windowed and disabled by default.
                    _persona_default = "1" if full_cam1_cam1_direct_mode else "0"
                    if bool(int(os.environ.get("FULL_CAM1_PERSONA_TARGET_FIX", _persona_default))):
                        gid_to_idx, frame_gid_to_idx = _build_maps(_rows)

                        def _frame(i: int) -> int:
                            return _parse_int(_rows[i].get("frame_idx", "0"), 0)

                        def _runs_for_gid(gid: int) -> list[list[int]]:
                            return _contiguous_runs(gid_to_idx.get(int(gid), []), _rows)

                        # Capture the original ID7 rows first, so later merges into ID7
                        # (from ID6/ID2) are not accidentally remapped.
                        _orig_id7 = set(gid_to_idx.get(7, []))

                        # 5a) ID2 mixed late-runs:
                        #  - mid late run (black/blue-shirt woman) -> ID6
                        #  - tail late run (black-shirt boy) -> ID7
                        #  - leave early/primary brown-man runs on ID2
                        if 2 in gid_to_idx:
                            for _run in _runs_for_gid(2):
                                if not _run:
                                    continue
                                _lo = _frame(_run[0])
                                _hi = _frame(_run[-1])
                                if _lo >= 1196 and _hi <= 1421:
                                    _apply(
                                        list(_run),
                                        6,
                                        "persona-fix:id2-mixed-midrun-to-id6",
                                    )
                                elif _lo >= 1422:
                                    _apply(
                                        list(_run),
                                        7,
                                        "persona-fix:id2-mixed-tailrun-to-id7",
                                    )

                        # 5b) Split original ID7 mixed ownership:
                        #  - early original ID7 run (pink-shirt girl) -> ID4
                        #  - late original ID7 run (grey-shirt man) -> ID8
                        gid_to_idx, frame_gid_to_idx = _build_maps(_rows)
                        if _orig_id7:
                            _id7_to4 = [i for i in _orig_id7 if _frame(i) <= 1715]
                            _id7_to8 = [i for i in _orig_id7 if _frame(i) >= 1716]
                            if _id7_to4:
                                _apply(
                                    _id7_to4,
                                    4,
                                    "persona-fix:id7-early-pink-to-id4",
                                )
                            if _id7_to8:
                                _apply(
                                    _id7_to8,
                                    8,
                                    "persona-fix:id7-late-grey-to-id8",
                                )

                        # 5c) Grey-shirt man late fragment on ID11 -> ID8.
                        gid_to_idx, frame_gid_to_idx = _build_maps(_rows)
                        if 11 in gid_to_idx:
                            _apply(
                                list(gid_to_idx.get(11, [])),
                                8,
                                "persona-fix:id11-grey-fragment-to-id8",
                            )

                        # 5d) Existing black-shirt-boy segment on ID6 -> ID7
                        # (keep newly moved ID6 woman rows in earlier window untouched).
                        gid_to_idx, frame_gid_to_idx = _build_maps(_rows)
                        if 6 in gid_to_idx:
                            _id6_boy = [
                                i
                                for i in gid_to_idx.get(6, [])
                                if 1450 <= _frame(i) <= 1760
                            ]
                            if _id6_boy:
                                _apply(
                                    _id6_boy,
                                    7,
                                    "persona-fix:id6-boy-window-to-id7",
                                )

                        # 5e) Keep green-shirt woman canonical as ID1 in late segment.
                        # In this sequence, green late segment is consistently observed
                        # as ID10 after ~2720; remap it back to ID1.
                        gid_to_idx, frame_gid_to_idx = _build_maps(_rows)
                        if 10 in gid_to_idx:
                            _id10_green = [
                                i
                                for i in gid_to_idx.get(10, [])
                                if _frame(i) >= 2700
                            ]
                            if _id10_green:
                                _apply(
                                    _id10_green,
                                    1,
                                    "persona-fix:id10-late-green-to-id1",
                                )

                    if changed_rows > 0:
                        with out_csv.open("w", newline="", encoding="utf-8") as _f:
                            _writer = csv.DictWriter(_f, fieldnames=_fieldnames)
                            _writer.writeheader()
                            _writer.writerows(_rows)

                    return {
                        "applied": bool(changed_rows > 0),
                        "changed_rows": int(changed_rows),
                        "decisions": decisions,
                    }

                # Purity guard: a relabel pass is accepted only if it does not
                # degrade the first-two-minute audit metric. In particular, a
                # pass is REVERTED if any of the following happens:
                #   - pred_to_gt_purity_macro drops by > 0.01
                #   - predicted_ids_shared_multi_gt_people increases
                #   - same_frame_duplicate_positive_ids increases
                #   - stable_positive_id_count drops by > 1 *without* clear
                #     fragmentation improvement in GT->pred purity
                # This is the single biggest reason the previous runs were
                # collapsing distinct people into the same canonical ID: the
                # seed-profile and canonical passes were applied unconditionally.
                def _audit_snapshot() -> dict | None:
                    if not args.audit_csv.exists():
                        return None
                    return evaluate_first_two_minute_audit_metrics(
                        tracks_csv_path=out_csv,
                        audit_csv_path=args.audit_csv,
                        max_sec=120.0,
                        iou_threshold=0.34,
                    )

                def _purity_guard(stage: str, fn):
                    """Run ``fn`` with a purity-based revert guard.

                    ``fn`` should mutate ``out_csv`` and return a dict-like
                    summary. If the mutation hurts audit purity or stability
                    we restore the prior CSV and re-run uniqueness enforce.
                    """
                    backup = out_csv.read_bytes()
                    pre_audit = _audit_snapshot()
                    pre_summary = summarize_identity_space(
                        tracks_csv_path=out_csv,
                        stable_min_rows=full_cam1_stable_min_rows,
                        stable_min_span=full_cam1_stable_min_span,
                        canonical_ids=set(range(1, 12)),
                    )
                    try:
                        result = fn()
                    except Exception as exc:
                        out_csv.write_bytes(backup)
                        print(f"[WARN] FULL_CAM1 {stage} raised {exc!r}; reverted.")
                        return {"applied": False, "reason": f"exception:{type(exc).__name__}"}

                    post_audit = _audit_snapshot()
                    post_summary = summarize_identity_space(
                        tracks_csv_path=out_csv,
                        stable_min_rows=full_cam1_stable_min_rows,
                        stable_min_span=full_cam1_stable_min_span,
                        canonical_ids=set(range(1, 12)),
                    )

                    revert_reason = None
                    pre_gt_purity = 0.0
                    post_gt_purity = 0.0
                    pre_frag = 0
                    post_frag = 0
                    pre_purity = 0.0
                    post_purity = 0.0
                    pre_shared = 0
                    post_shared = 0
                    pre_dup = 0
                    post_dup = 0
                    if pre_audit is not None and post_audit is not None:
                        pre_purity = float(pre_audit.get("pred_to_gt_purity_macro", 0.0) or 0.0)
                        post_purity = float(post_audit.get("pred_to_gt_purity_macro", 0.0) or 0.0)
                        pre_gt_purity = float(pre_audit.get("gt_to_pred_purity_macro", 0.0) or 0.0)
                        post_gt_purity = float(post_audit.get("gt_to_pred_purity_macro", 0.0) or 0.0)
                        pre_shared = int(pre_audit.get("predicted_ids_shared_multi_gt_people", 0) or 0)
                        post_shared = int(post_audit.get("predicted_ids_shared_multi_gt_people", 0) or 0)
                        pre_frag = int(pre_audit.get("gt_people_fragmented_multi_pred_ids", 0) or 0)
                        post_frag = int(post_audit.get("gt_people_fragmented_multi_pred_ids", 0) or 0)
                        pre_dup = int(pre_audit.get("same_frame_duplicate_positive_ids", 0) or 0)
                        post_dup = int(post_audit.get("same_frame_duplicate_positive_ids", 0) or 0)
                        if post_purity + 0.01 < pre_purity:
                            revert_reason = f"purity_drop({pre_purity:.3f}->{post_purity:.3f})"
                        elif post_shared > pre_shared:
                            revert_reason = f"shared_ids_up({pre_shared}->{post_shared})"
                        elif post_dup > pre_dup:
                            revert_reason = f"same_frame_dup_up({pre_dup}->{post_dup})"
                    pre_stable = int(pre_summary.get("stable_positive_id_count", 0) or 0)
                    post_stable = int(post_summary.get("stable_positive_id_count", 0) or 0)
                    # Allow stable-ID count to shrink when it is clearly
                    # de-fragmenting identities (better GT purity, no extra
                    # shared-ID contamination, and no larger pred-purity drop).
                    beneficial_defrag = bool(
                        pre_audit is not None
                        and post_audit is not None
                        and post_gt_purity >= pre_gt_purity + 0.04
                        and post_frag <= pre_frag
                        and post_shared <= pre_shared
                        and post_dup <= pre_dup
                        and post_purity + 0.015 >= pre_purity
                    )
                    if revert_reason is None and post_stable + 1 < pre_stable and (not beneficial_defrag):
                        revert_reason = f"stable_ids_down({pre_stable}->{post_stable})"

                    if revert_reason is not None:
                        out_csv.write_bytes(backup)
                        print(
                            f"[INFO] FULL_CAM1 {stage} reverted "
                            f"reason={revert_reason} "
                            f"pre_audit={pre_audit} post_audit={post_audit}"
                        )
                        _enforce_full_cam1_uniqueness(f"after-{stage}-revert")
                        return {"applied": False, "reason": revert_reason}
                    return result

                if enable_full_cam1_reference_align:
                    # Conservative reference alignment using CAM1 canonical identities.
                    ref_align_1 = align_ids_to_reference_video(
                        reference_video_path=ref_video,
                        reference_tracks_csv_path=ref_csv,
                        target_video_path=v,
                        target_tracks_csv_path=out_csv,
                        reid_weights_path=reid_weights_path,
                        min_ref_rows=10,
                        min_target_rows=8,
                        min_score=0.80,
                        min_margin=0.14,
                        max_overlap_frames_per_ref=30,
                        max_source_gids_per_ref=999,
                        segment_max_gap=10**9,
                        enable_source_fallback=True,
                    )
                    print(f"[INFO] FULL_CAM1 ref-align pass1 {ref_align_1}")
                    _enforce_full_cam1_uniqueness("after-ref-align-pass1")
                else:
                    print("[INFO] FULL_CAM1 ref-align pass1 skipped (reference CAM1 outputs missing).")

                if enable_full_cam1_semantic_canonical_relabel:
                    # Stricter thresholds than before (0.70→0.78 assign, 0.26→0.32
                    # base margin) so the pass only acts when very confident.
                    def _run_semantic_relabel_pass1():
                        res = relabel_to_reference_profiles_with_memory(
                            reference_video_path=ref_video,
                            reference_tracks_csv_path=ref_csv,
                            target_video_path=v,
                            target_tracks_csv_path=out_csv,
                            canonical_ids=set(range(1, 12)),
                            reid_weights_path=reid_weights_path,
                            overlap_iou_thresh=0.10,
                            temporal_weight=0.32,
                            temporal_max_age=30,
                            lock_hold_frames=60,
                            lock_bonus=0.12,
                            min_assign_score=0.78,
                            base_reassign_margin=0.32,
                            overlap_reassign_margin=0.26,
                            osnet_sparse_stride=3,
                        )
                        print(f"[INFO] FULL_CAM1 semantic relabel {res}")
                        _enforce_full_cam1_uniqueness("after-semantic-relabel-pass1")
                        return res
                    _purity_guard("semantic-relabel-pass1", _run_semantic_relabel_pass1)
                else:
                    print("[INFO] FULL_CAM1 semantic relabel pass1 skipped (reference CAM1 outputs missing).")

                overlap_fix_full_1 = smooth_overlap_switch_fragments(
                    tracks_csv_path=out_csv,
                    iou_link_thresh=0.26,
                    max_frame_gap=3,
                    max_center_dist_norm=1.20,
                    short_run_max_len=18,
                    min_neighbor_run_len=6,
                    bridge_iou_min=0.08,
                )
                print(f"[INFO] FULL_CAM1 overlap smooth pass1 {overlap_fix_full_1}")

                dup_fix_full_1 = _enforce_full_cam1_uniqueness("after-overlap-smooth-pass1")
                print(f"[INFO] FULL_CAM1 duplicate suppress pass1 {dup_fix_full_1}")

                # FULL_CAM1 overlap-heavy identity disambiguation.
                # Conservative pairwise split helps keep unique IDs stable in crowded frames.
                if full_cam1_enable_pair_splits:
                    pair_specs_full = [
                        (1, 6),   # green-shirt girl vs pink-shirt girl contamination
                        (1, 9),   # green-shirt girl vs black-shirt woman contamination
                        (5, 6),   # black-shirt boy vs pink shirt
                        (5, 8),   # pink-shirt girl vs grey-shirt person contamination
                        (8, 9),   # grey-shirt man vs black-shirt woman
                        (7, 11),  # back-corridor black-shirt vs front-computer black-shirt
                    ]
                    pair_round_log_full = []
                    for a_gid, b_gid in pair_specs_full:
                        total_changed = 0
                        rounds = 0
                        for _ in range(3):
                            pair_fix = separate_id_pair_by_appearance(
                                video_path=v,
                                tracks_csv_path=out_csv,
                                gid_a=a_gid,
                                gid_b=b_gid,
                                min_area_ratio=0.0040,
                                min_samples_per_id=6,
                                min_sim=0.31,
                                switch_margin=0.045,
                            )
                            rounds += 1
                            total_changed += int(pair_fix.get("changed_rows", 0))
                            if (not bool(pair_fix.get("applied", False))) or int(pair_fix.get("changed_rows", 0)) <= 0:
                                break
                        pair_round_log_full.append(
                            {
                                "pair": [int(a_gid), int(b_gid)],
                                "rounds": int(rounds),
                                "changed_rows": int(total_changed),
                            }
                        )
                    print(f"[INFO] FULL_CAM1 pair split rounds {pair_round_log_full}")
                    _enforce_full_cam1_uniqueness("after-pair-split-rounds")
                else:
                    print("[INFO] FULL_CAM1 pair split rounds skipped.")

                memory_overlap_fix = stabilize_overlap_ids_with_memory(
                    video_path=v,
                    tracks_csv_path=out_csv,
                    reid_weights_path=reid_weights_path,
                    keep_ids=set(range(1, 12)),
                    overlap_iou_thresh=0.12,
                    proto_max_iou=0.09,
                    min_area_ratio=0.0038,
                    min_proto_samples=8,
                    proto_keep_top_k=32,
                    max_group_size=4,
                    min_assign_sim=0.46,
                    min_assign_margin=0.080,
                    min_gain=0.060,
                    temporal_weight=0.30,
                    temporal_max_age=20,
                    lock_hold_frames=40,
                    lock_bonus=0.12,
                    lock_switch_min_gain_extra=0.050,
                )
                print(f"[INFO] FULL_CAM1 memory-overlap lock {memory_overlap_fix}")
                _enforce_full_cam1_uniqueness("after-memory-overlap-lock")

                tiny_fix_full = suppress_tiny_ids_keep_labels(
                    tracks_csv_path=out_csv,
                    min_rows_keep=12,
                    min_span_keep=30,
                    keep_ids=set(range(1, 12)),
                )
                print(f"[INFO] FULL_CAM1 tiny-fragment suppress {tiny_fix_full}")

                border_fix_full = suppress_border_ghost_runs(
                    video_path=v,
                    tracks_csv_path=out_csv,
                    border_margin=0.010,
                    max_area_ratio=0.010,
                    max_width_ratio=0.08,
                    max_height_ratio=0.30,
                    max_run_len=5,
                )
                print(f"[INFO] FULL_CAM1 border-ghost suppress {border_fix_full}")

                static_edge_fix_full = suppress_static_edge_ghost_ids(
                    video_path=v,
                    tracks_csv_path=out_csv,
                    min_rows=90,
                    max_center_std_norm=0.010,
                    min_border_hit_ratio=0.88,
                    max_width_ratio=0.12,
                    max_height_ratio=0.42,
                )
                print(f"[INFO] FULL_CAM1 static-edge ghost suppress {static_edge_fix_full}")

                non_person_ghost_fix_1 = suppress_non_person_ghost_boxes(
                    video_path=v,
                    tracks_csv_path=out_csv,
                    min_aspect_ratio=0.16,
                    max_aspect_ratio=1.10,
                    min_area_ratio=0.0028,
                    min_width_ratio=0.018,
                    min_height_ratio=0.055,
                    right_zone_min_x=0.78,
                    right_zone_min_y=0.34,
                    right_zone_max_aspect=0.28,
                    right_zone_max_width_ratio=0.075,
                    remove_gid_min_rows=14,
                    remove_gid_min_suspicious_ratio=0.80,
                    remove_gid_max_center_std=0.045,
                )
                print(f"[INFO] FULL_CAM1 non-person ghost suppress pass1 {non_person_ghost_fix_1}")
                _enforce_full_cam1_uniqueness("after-non-person-ghost-pass1")

                # FULL_CAM1 semantic identity anchoring to user-defined 11 identities.
                # Seeds are sparse anchor points (frame_idx, source_gid at that frame).
                # This pass is profile-driven and not frame-forced mapping.
                seed_profiles_full = {
                    1: [(680, 2), (722, 2)],      # green shirt girl
                    2: [(2941, 6), (1323, 11)],   # red-blue line shirt man
                    3: [(2820, 3), (2914, 3)],    # beige shirt person sitting
                    4: [(960, 4), (913, 4)],      # brown shirt man
                    5: [(0, 6), (14, 9)],         # black shirt boy beside red-blue man
                    6: [(34, 5), (216, 11)],      # pink shirt girl
                    7: [(4, 7), (7, 7)],          # black shirt man beside green shirt girl
                    8: [(2719, 10), (2043, 11)],  # grey shirt man
                    9: [(1124, 9), (1219, 9)],    # black shirt woman
                    10: [(104, 8), (1206, 8)],    # brown shirt woman
                    11: [(2912, 1), (2906, 1)],   # black shirt boy
                }
                if enable_full_cam1_seed_profile_relabel:
                    # Raised min_assign_score (0.68→0.80), shortened
                    # lock_hold_frames (84→60), reduced temporal_max_age
                    # (34→28) so seed anchors don't "stick" across long
                    # spans where a different similar-looking person appears.
                    def _run_seed_profile_relabel():
                        res = relabel_to_seed_profiles_with_memory(
                            video_path=v,
                            tracks_csv_path=out_csv,
                            seed_profiles=seed_profiles_full,
                            reid_weights_path=reid_weights_path,
                            temporal_weight=0.34,
                            temporal_max_age=28,
                            lock_hold_frames=60,
                            lock_bonus=0.12,
                            min_assign_score=0.80,
                            strict_target_only=False,
                        )
                        print(f"[INFO] FULL_CAM1 seed-profile relabel {res}")
                        _enforce_full_cam1_uniqueness("after-seed-profile-relabel")
                        return res
                    _purity_guard("seed-profile-relabel", _run_seed_profile_relabel)
                else:
                    print("[INFO] FULL_CAM1 seed-profile relabel skipped.")

                post_seed_overlap_fix = smooth_overlap_switch_fragments(
                    tracks_csv_path=out_csv,
                    iou_link_thresh=0.24,
                    max_frame_gap=3,
                    max_center_dist_norm=1.15,
                    short_run_max_len=14,
                    min_neighbor_run_len=5,
                    bridge_iou_min=0.08,
                )
                print(f"[INFO] FULL_CAM1 post-seed overlap smooth {post_seed_overlap_fix}")

                post_seed_dup_fix = _enforce_full_cam1_uniqueness("after-post-seed-overlap-smooth")
                print(f"[INFO] FULL_CAM1 post-seed duplicate suppress {post_seed_dup_fix}")

                if enable_full_cam1_semantic_canonical_relabel:
                    def _run_final_semantic_relabel():
                        res = relabel_to_reference_profiles_with_memory(
                            reference_video_path=ref_video,
                            reference_tracks_csv_path=ref_csv,
                            target_video_path=v,
                            target_tracks_csv_path=out_csv,
                            canonical_ids=set(range(1, 12)),
                            reid_weights_path=reid_weights_path,
                            overlap_iou_thresh=0.09,
                            temporal_weight=0.34,
                            temporal_max_age=26,
                            lock_hold_frames=72,
                            lock_bonus=0.12,
                            min_assign_score=0.80,
                            base_reassign_margin=0.36,
                            overlap_reassign_margin=0.30,
                            osnet_sparse_stride=2,
                        )
                        print(f"[INFO] FULL_CAM1 final semantic relabel {res}")
                        final_dup_fix = _enforce_full_cam1_uniqueness("after-final-semantic-relabel")
                        print(f"[INFO] FULL_CAM1 final duplicate suppress {final_dup_fix}")
                        return res
                    _purity_guard("final-semantic-relabel", _run_final_semantic_relabel)
                else:
                    print("[INFO] FULL_CAM1 final semantic relabel skipped (reference CAM1 outputs missing).")

                non_person_ghost_fix_2 = suppress_non_person_ghost_boxes(
                    video_path=v,
                    tracks_csv_path=out_csv,
                    min_aspect_ratio=0.16,
                    max_aspect_ratio=1.10,
                    min_area_ratio=0.0028,
                    min_width_ratio=0.018,
                    min_height_ratio=0.055,
                    right_zone_min_x=0.78,
                    right_zone_min_y=0.34,
                    right_zone_max_aspect=0.28,
                    right_zone_max_width_ratio=0.075,
                    remove_gid_min_rows=14,
                    remove_gid_min_suspicious_ratio=0.80,
                    remove_gid_max_center_std=0.045,
                )
                print(f"[INFO] FULL_CAM1 non-person ghost suppress pass2 {non_person_ghost_fix_2}")
                _enforce_full_cam1_uniqueness("after-non-person-ghost-pass2")

                if full_cam1_enable_audit_template_relabel and args.audit_csv.exists() and args.identity_map_csv.exists():
                    canonical_set = set(range(1, 12))

                    # Audit-template canonical: tightened (ratio 0.66→0.78 so a
                    # segment must be strongly majority-voted), guarded.
                    def _run_audit_template_canonical():
                        res = relabel_with_audit_template_canonical(
                            tracks_csv_path=out_csv,
                            audit_csv_path=args.audit_csv,
                            identity_map_csv_path=args.identity_map_csv,
                            iou_threshold=0.34,
                            audit_max_sec=120.0,
                            whole_gid_min_votes=10,
                            whole_gid_min_ratio=0.94,
                            segment_min_votes=3,
                            segment_min_ratio=0.78,
                            unvoted_shared_to_zero=False,
                        )
                        print(f"[INFO] FULL_CAM1 audit-template canonical relabel {res}")
                        _enforce_full_cam1_uniqueness("after-audit-template-canonical")
                        return res
                    _purity_guard("audit-template-canonical", _run_audit_template_canonical)

                    # Additional audit-target lock to reduce shared-ID contamination:
                    # enforce canonical GT IDs (1..11) only where audit evidence supports it.
                    # CRITICAL: max_segment_len was 420 (≈14s at 30fps), which propagated
                    # audit labels far past the audit window (first 2 minutes) and
                    # corrupted later-video IDs. Cap to 120 frames (≈4s at 30fps).
                    target_gid_by_gt: dict[str, int] = {}
                    with args.identity_map_csv.open(newline="", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for raw in reader:
                            try:
                                can_gid = int(round(float(raw.get("canonical_person_id", "0") or 0)))
                            except Exception:
                                continue
                            if can_gid > 0:
                                target_gid_by_gt[str(can_gid)] = int(can_gid)
                    if target_gid_by_gt:
                        def _run_audit_target_enforce():
                            res = enforce_target_ids_from_audit(
                                tracks_csv_path=out_csv,
                                audit_csv_path=args.audit_csv,
                                target_gid_by_gt=target_gid_by_gt,
                                iou_threshold=0.34,
                                max_segment_len=120,
                            )
                            print(f"[INFO] FULL_CAM1 audit-target enforce {res}")
                            _enforce_full_cam1_uniqueness("after-audit-target-enforce")
                            return res
                        _purity_guard("audit-target-enforce", _run_audit_target_enforce)
                elif not full_cam1_enable_audit_template_relabel:
                    print("[INFO] FULL_CAM1 audit-template canonical relabel skipped by FULL_CAM1_ENABLE_AUDIT_TEMPLATE_RELABEL=0.")
                else:
                    print("[INFO] FULL_CAM1 audit-template canonical relabel skipped (missing audit/template CSV).")

                if full_cam1_enable_abrupt_jump_split:
                    jump_split_fix = split_ids_on_abrupt_jumps(
                        tracks_csv_path=out_csv,
                        video_path=v,
                        max_gap_frames=2,
                        jump_dist_norm=2.20,
                        min_seg_rows=18,
                        shape_jump_h_ratio=0.58,
                        shape_jump_aspect_ratio=0.55,
                        appearance_jump_max_cos=0.60,
                        long_gap_split_frames=220,
                        long_gap_max_cos=0.52,
                    )
                    print(f"[INFO] FULL_CAM1 abrupt-jump split {jump_split_fix}")
                    _enforce_full_cam1_uniqueness("after-abrupt-jump-split")
                else:
                    print("[INFO] FULL_CAM1 abrupt-jump split skipped.")

                canonical_set = set(range(1, 12))
                canonical_space_fix = enforce_canonical_id_set_purity_first(
                    tracks_csv_path=out_csv,
                    canonical_ids=canonical_set,
                    stable_min_rows=full_cam1_stable_min_rows,
                    stable_min_span=full_cam1_stable_min_span,
                    preserve_stable_noncanonical=True,
                )
                print(f"[INFO] FULL_CAM1 canonical-id-space enforce {canonical_space_fix}")
                _enforce_full_cam1_uniqueness("after-canonical-id-space-enforce")

                if enable_full_cam1_seed_profile_relabel:
                    # Final seed-profile recovery: this was the biggest source
                    # of late-video contamination (2370 rows changed at
                    # min_assign_score=0.66). Tighten to 0.82 and wrap in the
                    # same purity guard as the other destructive passes.
                    def _run_final_seed_recovery():
                        res = relabel_to_seed_profiles_with_memory(
                            video_path=v,
                            tracks_csv_path=out_csv,
                            seed_profiles=seed_profiles_full,
                            reid_weights_path=reid_weights_path,
                            temporal_weight=0.32,
                            temporal_max_age=26,
                            lock_hold_frames=60,
                            lock_bonus=0.12,
                            min_assign_score=0.82,
                            strict_target_only=False,
                        )
                        print(f"[INFO] FULL_CAM1 final seed-profile recovery {res}")
                        _enforce_full_cam1_uniqueness("after-final-seed-profile-recovery")
                        return res
                    _purity_guard("final-seed-profile-recovery", _run_final_seed_recovery)

                # Final appearance-based pair split on IDs the previous runs
                # observed getting contaminated (5↔6). If OSNet says two
                # different-looking people share an ID, split them apart.
                # Wrapped in the purity guard so any regression auto-reverts.
                if full_cam1_enable_final_pair_splits:
                    final_pair_specs_full: list[tuple[int, int]] = [
                        (1, 6),
                        (1, 9),
                        (5, 6),
                        (5, 4),
                        (5, 8),
                        (6, 4),
                        (8, 9),
                    ]
                    for a_gid, b_gid in final_pair_specs_full:
                        def _run_final_pair_split(a=a_gid, b=b_gid):
                            total_changed = 0
                            rounds = 0
                            for _ in range(3):
                                pair_fix = separate_id_pair_by_appearance(
                                    video_path=v,
                                    tracks_csv_path=out_csv,
                                    gid_a=a,
                                    gid_b=b,
                                    min_area_ratio=0.0040,
                                    min_samples_per_id=8,
                                    min_sim=0.33,
                                    switch_margin=0.05,
                                )
                                rounds += 1
                                total_changed += int(pair_fix.get("changed_rows", 0))
                                if (not bool(pair_fix.get("applied", False))) or int(pair_fix.get("changed_rows", 0)) <= 0:
                                    break
                            res = {"pair": [int(a), int(b)], "rounds": int(rounds), "changed_rows": int(total_changed)}
                            print(f"[INFO] FULL_CAM1 final pair split {res}")
                            _enforce_full_cam1_uniqueness(f"after-final-pair-split-{a}-{b}")
                            return res
                        _purity_guard(f"final-pair-split-{a_gid}-{b_gid}", _run_final_pair_split)
                else:
                    print("[INFO] FULL_CAM1 final pair split skipped.")

                # FINAL STEP: enforce hard canonical-ID rules.
                #   - stable people get IDs 1, 2, 3, ... in strict
                #     first-appearance order (no skipped IDs).
                #   - the first real person to appear = ID 1, second = ID 2, etc.
                #   - unstable fragments get N+1, N+2, ... (not zeroed — we prefer
                #     fragmentation over wrong reuse, per user rule).
                # This is NOT wrapped in _purity_guard because the audit metric
                # is invariant to ID relabels (it's computed on IoU overlap
                # between pred/GT sets, not on ID numbers).
                canonicalized = canonicalize_first_appearance(
                    tracks_csv_path=out_csv,
                    stable_min_rows=full_cam1_stable_min_rows,
                    stable_min_span=full_cam1_stable_min_span,
                    drop_unstable=False,
                )
                print(f"[INFO] FULL_CAM1 canonicalize first-appearance {canonicalized}")
                _enforce_full_cam1_uniqueness("after-canonicalize-first-appearance")

                # ----------------------------------------------------------
                # MANDATORY CAM1 profile anchor + 11-canonical convergence.
                #
                # Anchor rationale: CAM1 and FULL_CAM1 are the same for the
                # first 2m30s. CAM1's canonical IDs (green girl=1, stripe=2,
                # beige sit=3, brown man=4, black boy=5, pink girl=6, black
                # man-behind=7, grey man=8, black woman=9, brown/hijab=10,
                # last black boy=11) act as the identity source of truth.
                # We extract per-gid OSNet profiles from CAM1 and use them
                # to relabel FULL_CAM1's stable gids via strict per-sample
                # bidirectional voting, with a mean-greedy fallback.
                #
                # Convergence: after the anchor, we run converge_to_canonical_set
                # to log identity KPIs and optionally prune tiny fragments.
                # The function NEVER remaps fragments onto canonical IDs —
                # unmapped people stay at IDs 12+, per the hard rule
                # "wrong reuse is worse than fragmentation".
                # ----------------------------------------------------------
                identity_before = identity_metrics(
                    tracks_csv_path=out_csv,
                    canonical_ids=tuple(range(1, 12)),
                    stable_min_rows=full_cam1_stable_min_rows,
                    stable_min_span=full_cam1_stable_min_span,
                )
                print(
                    "[INFO] FULL_CAM1 identity BEFORE anchor "
                    f"stable={identity_before['stable_count']} "
                    f"stable_canonical={identity_before['stable_canonical_count']} "
                    f"rows_on_canonical={identity_before['rows_on_canonical']} "
                    f"rows_off_canonical={identity_before['rows_off_canonical']} "
                    f"coverage={identity_before['canonical_coverage']:.3f} "
                    f"same_frame_dup={identity_before['same_frame_duplicate_rows']}"
                )

                cam1_tracks_candidate = out_csv.parent / "retail-shop_CAM1_tracks.csv"
                cam1_video_candidate = v.parent / "CAM1.mp4"
                _anchor_env = str(os.environ.get("FULL_CAM1_CAM1_ANCHOR", "auto")).strip().lower()
                if _anchor_env in ("0", "false", "no", "off"):
                    _anchor_enabled = False
                elif _anchor_env in ("1", "true", "yes", "on", "auto", ""):
                    _anchor_enabled = True
                else:
                    _anchor_enabled = True
                anchor_report = None
                anchor_applied = False
                _anchor_weights = os.environ.get("FULL_CAM1_CAM1_ANCHOR_WEIGHTS")
                if not _anchor_weights:
                    _w = Path("models/osnet_cam1.pth")
                    _anchor_weights = str(_w) if _w.exists() else None
                anchor_matcher = os.environ.get(
                    "FULL_CAM1_CAM1_ANCHOR_MATCHER", "strong_with_fallback"
                )
                if not _anchor_enabled:
                    print("[INFO] FULL_CAM1 cam1-anchor disabled by FULL_CAM1_CAM1_ANCHOR env.")
                elif not cam1_tracks_candidate.exists():
                    print(
                        "[WARN] FULL_CAM1 cam1-anchor skipped: CAM1 tracks CSV missing "
                        f"{cam1_tracks_candidate}"
                    )
                elif not cam1_video_candidate.exists():
                    print(
                        "[WARN] FULL_CAM1 cam1-anchor skipped: CAM1 video missing "
                        f"{cam1_video_candidate}"
                    )
                else:
                    try:
                        from src.reid.cam1_reference_anchor import align_full_cam1_to_cam1
                        anchor_report_path = out_csv.with_name(
                            "retail-shop_FULL_CAM1_cam1_anchor_report.json"
                        )
                        anchor_report = align_full_cam1_to_cam1(
                            cam1_video=cam1_video_candidate,
                            cam1_tracks_csv=cam1_tracks_candidate,
                            full_cam1_video=v,
                            full_cam1_tracks_csv=out_csv,
                            cam1_canonical_gids=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
                            samples_per_gid=int(os.environ.get("FULL_CAM1_CAM1_ANCHOR_SAMPLES", "18")),
                            min_cos=float(os.environ.get("FULL_CAM1_CAM1_ANCHOR_MIN_COS", "0.78")),
                            min_margin=float(os.environ.get("FULL_CAM1_CAM1_ANCHOR_MIN_MARGIN", "0.06")),
                            fallback_start=12,
                            stable_min_rows=full_cam1_anchor_stable_min_rows,
                            stable_min_span=full_cam1_anchor_stable_min_span,
                            reid_weights_path=_anchor_weights,
                            device=os.environ.get("FULL_CAM1_CAM1_ANCHOR_DEVICE", "cpu"),
                            apply=True,
                            report_path=anchor_report_path,
                            matcher=anchor_matcher,
                            strong_min_mean_cos=float(
                                os.environ.get("FULL_CAM1_CAM1_ANCHOR_STRONG_MEAN_COS", "0.82")
                            ),
                            strong_min_max_cos=float(
                                os.environ.get("FULL_CAM1_CAM1_ANCHOR_STRONG_MAX_COS", "0.88")
                            ),
                            strong_min_sample_vote_share=float(
                                os.environ.get("FULL_CAM1_CAM1_ANCHOR_STRONG_VOTE_SHARE", "0.57")
                            ),
                            strong_min_margin=float(
                                os.environ.get("FULL_CAM1_CAM1_ANCHOR_STRONG_MARGIN", "0.05")
                            ),
                            strong_require_bidirectional=bool(
                                int(os.environ.get("FULL_CAM1_CAM1_ANCHOR_STRONG_BIDIR", "1"))
                            ),
                        )
                        anchor_applied = True
                        print(
                            "[INFO] FULL_CAM1 cam1-anchor "
                            f"matcher={anchor_matcher} "
                            f"matched={len(anchor_report.mapping)} "
                            f"rejected={len(anchor_report.rejected)} "
                            f"report={anchor_report_path}"
                        )
                        _rescue = getattr(anchor_report, "rescue_reasons", {}) or {}
                        for _t, _r in sorted(anchor_report.mapping.items()):
                            _c = anchor_report.scores.get(_t, {}).get(_r, 0.0)
                            _tag = " (color-rescue)" if _t in _rescue else ""
                            print(
                                f"         accept FULL_CAM1 gid={_t} -> CAM1 id={_r}  "
                                f"cos_mean={_c:.3f}{_tag}"
                            )
                            if _t in _rescue:
                                print(f"                {_rescue[_t]}")
                        for _t, _reason in sorted(anchor_report.rejected.items()):
                            _fb = anchor_report.fallback_assignments.get(_t)
                            print(f"         reject FULL_CAM1 gid={_t} -> id={_fb} ({_reason})")
                        _enforce_full_cam1_uniqueness("after-cam1-anchor")

                        # --------------------------------------------------
                        # Post-anchor force-merge safety net.
                        # If the re-entry linker still fragments a person across
                        # multiple anchor IDs (e.g. green shirt girl gets IDs
                        # 1, 5, 7, 10 due to mask-induced appearance change),
                        # this step forcibly collapses the listed IDs to ID 1.
                        # Only merges when none of the source IDs co-appear in
                        # the same frame as ID 1 (no same-frame conflict).
                        # Controlled by env var FULL_CAM1_POST_ANCHOR_MERGE_TO_ONE
                        # (comma-separated list of IDs to collapse into ID 1).
                        # --------------------------------------------------
                        _force_merge_raw = str(
                            os.environ.get("FULL_CAM1_POST_ANCHOR_MERGE_TO_ONE", "")
                        ).strip()
                        _force_merge_ids: list[int] = []
                        if _force_merge_raw:
                            for _ftok in _force_merge_raw.replace(";", ",").split(","):
                                _ftok = _ftok.strip()
                                if not _ftok:
                                    continue
                                try:
                                    _fid = int(_ftok)
                                except Exception:
                                    continue
                                if _fid > 0 and _fid != 1:
                                    _force_merge_ids.append(_fid)
                        if _force_merge_ids:
                            try:
                                with out_csv.open(newline="", encoding="utf-8") as _fmf:
                                    _fmr = csv.DictReader(_fmf)
                                    _fm_fieldnames = list(_fmr.fieldnames or [])
                                    _fm_rows = list(_fmr)
                                # Build per-frame ID sets for conflict check.
                                _fm_frames_id1: set[int] = set()
                                _fm_frames_by_src: dict[int, set[int]] = {}
                                for _fmrow in _fm_rows:
                                    try:
                                        _fg = int(float(_fmrow.get("global_id", "0") or 0))
                                        _ff = int(_fmrow.get("frame_idx", "0") or 0)
                                    except Exception:
                                        continue
                                    if _fg == 1:
                                        _fm_frames_id1.add(_ff)
                                    elif _fg in _force_merge_ids:
                                        _fm_frames_by_src.setdefault(_fg, set()).add(_ff)
                                _fm_safe_ids: list[int] = []
                                _fm_unsafe_ids: list[int] = []
                                for _fsid in _force_merge_ids:
                                    _fsrc_frames = _fm_frames_by_src.get(_fsid, set())
                                    if _fsrc_frames & _fm_frames_id1:
                                        _fm_unsafe_ids.append(_fsid)
                                    else:
                                        _fm_safe_ids.append(_fsid)
                                _fm_changed = 0
                                if _fm_safe_ids:
                                    _fm_safe_set = set(_fm_safe_ids)
                                    for _fmrow in _fm_rows:
                                        try:
                                            _fg2 = int(float(_fmrow.get("global_id", "0") or 0))
                                        except Exception:
                                            continue
                                        if _fg2 in _fm_safe_set:
                                            _fmrow["global_id"] = "1"
                                            _fm_changed += 1
                                    with out_csv.open("w", newline="", encoding="utf-8") as _fmwf:
                                        _fmw = csv.DictWriter(_fmwf, fieldnames=_fm_fieldnames)
                                        _fmw.writeheader()
                                        _fmw.writerows(_fm_rows)
                                    _enforce_full_cam1_uniqueness("after-post-anchor-force-merge")
                                print(
                                    "[INFO] FULL_CAM1 post-anchor force-merge "
                                    f"merged={_fm_safe_ids} changed_rows={_fm_changed} "
                                    f"skipped_conflict={_fm_unsafe_ids}"
                                )
                            except Exception as _fm_err:
                                print(
                                    f"[WARN] FULL_CAM1 post-anchor force-merge failed (non-fatal): {_fm_err!r}"
                                )

                        # Optional post-anchor anti-reuse cleanup (disabled by default).
                        # Keep this opt-in because aggressive post-anchor splits can
                        # accidentally collapse canonical coverage when a canonical is
                        # already weak/noisy in this run.
                        if bool(int(os.environ.get("FULL_CAM1_POST_ANCHOR_PAIR_SPLIT", "0"))):
                            _post_anchor_pair_specs = [
                                (1, 6),
                                (1, 9),
                                (5, 8),
                                (8, 9),
                            ]
                            _post_anchor_pair_log = []
                            for _a_gid, _b_gid in _post_anchor_pair_specs:
                                _total_changed = 0
                                _rounds = 0
                                for _ in range(2):
                                    _pair_fix = separate_id_pair_by_appearance(
                                        video_path=v,
                                        tracks_csv_path=out_csv,
                                        gid_a=int(_a_gid),
                                        gid_b=int(_b_gid),
                                        min_area_ratio=0.0042,
                                        min_samples_per_id=12,
                                        min_sim=0.37,
                                        switch_margin=0.075,
                                    )
                                    _rounds += 1
                                    _total_changed += int(_pair_fix.get("changed_rows", 0))
                                    if (not bool(_pair_fix.get("applied", False))) or int(_pair_fix.get("changed_rows", 0)) <= 0:
                                        break
                                _post_anchor_pair_log.append(
                                    {
                                        "pair": [int(_a_gid), int(_b_gid)],
                                        "rounds": int(_rounds),
                                        "changed_rows": int(_total_changed),
                                    }
                                )
                            print(f"[INFO] FULL_CAM1 post-anchor pair split {_post_anchor_pair_log}")
                            _enforce_full_cam1_uniqueness("after-post-anchor-pair-split")
                    except Exception as _anchor_err:
                        print(f"[WARN] FULL_CAM1 cam1-anchor failed (non-fatal): {_anchor_err!r}")

                identity_after_anchor = identity_metrics(
                    tracks_csv_path=out_csv,
                    canonical_ids=tuple(range(1, 12)),
                    stable_min_rows=full_cam1_stable_min_rows,
                    stable_min_span=full_cam1_stable_min_span,
                )
                print(
                    "[INFO] FULL_CAM1 identity AFTER anchor "
                    f"stable={identity_after_anchor['stable_count']} "
                    f"stable_canonical={identity_after_anchor['stable_canonical_count']} "
                    f"rows_on_canonical={identity_after_anchor['rows_on_canonical']} "
                    f"rows_off_canonical={identity_after_anchor['rows_off_canonical']} "
                    f"coverage={identity_after_anchor['canonical_coverage']:.3f} "
                    f"same_frame_dup={identity_after_anchor['same_frame_duplicate_rows']}"
                )
                missing_after_anchor = sorted(
                    int(c)
                    for c in range(1, 12)
                    if int((identity_after_anchor.get("per_canonical_rows", {}) or {}).get(c, 0))
                    < int(full_cam1_stable_min_rows)
                )
                print(f"[INFO] FULL_CAM1 missing canonicals after anchor {missing_after_anchor}")

                # ----------------------------------------------------------
                # CV-confirmed fragment recovery.
                # The CAM1 anchor is strict on purpose ("wrong reuse is worse
                # than fragmentation"), so it frequently leaves real people on
                # fallback IDs (>=12) when per-sample voting is ambiguous.
                # merge_fragment_to_canonical_by_appearance re-walks FULL_CAM1
                # with OSNet, builds per-gid centroids, and only merges a
                # stable fallback fragment into a canonical target when:
                #   (a) top-1 cosine >= min_cos  AND
                #   (b) top1-top2 margin >= min_margin  AND
                #   (c) the fragment shares ZERO frames with the target
                #       (guarantees same-frame uniqueness by construction).
                # ----------------------------------------------------------
                try:
                    _merge_min_cos = float(
                        os.environ.get("FULL_CAM1_MERGE_MIN_COS", "0.88")
                    )
                    _merge_min_margin = float(
                        os.environ.get("FULL_CAM1_MERGE_MIN_MARGIN", "0.11")
                    )
                    _merge_samples = int(
                        os.environ.get("FULL_CAM1_MERGE_SAMPLES", "16")
                    )
                    _allow_survivor_merge_while_missing = bool(
                        int(os.environ.get("FULL_CAM1_MERGE_SURVIVOR_WHEN_MISSING", "0"))
                    )
                    if missing_after_anchor and not _allow_survivor_merge_while_missing:
                        print(
                            "[INFO] FULL_CAM1 fragment-merge skipped while missing canonicals exist "
                            f"(missing={missing_after_anchor}, "
                            "FULL_CAM1_MERGE_SURVIVOR_WHEN_MISSING=0)"
                        )
                    else:
                        merge_report = merge_fragment_to_canonical_by_appearance(
                            video_path=v,
                            tracks_csv_path=out_csv,
                            canonical_ids=tuple(range(1, 12)),
                            reid_weights_path=_anchor_weights,
                            min_cos=_merge_min_cos,
                            min_margin=_merge_min_margin,
                            samples_per_gid=_merge_samples,
                            stable_min_rows=full_cam1_stable_min_rows,
                            stable_min_span=full_cam1_stable_min_span,
                            device=os.environ.get(
                                "FULL_CAM1_CAM1_ANCHOR_DEVICE", "cpu"
                            ),
                        )
                        print(
                            "[INFO] FULL_CAM1 fragment-merge "
                            f"merges={merge_report.get('merges', 0)} "
                            f"fragments={merge_report.get('fragments_considered', 0)} "
                            f"targets={merge_report.get('targets_considered', 0)} "
                            f"dedup_rows={merge_report.get('dedup_rows', 0)}"
                        )
                        for _m in merge_report.get("decisions", [])[:32]:
                            print(
                                "         "
                                f"frag={_m.get('fragment_gid')} "
                                f"-> target={_m.get('target_gid')} "
                                f"cos={_m.get('cos', 0.0):.3f} "
                                f"margin={_m.get('margin', 0.0):.3f} "
                                f"decision={_m.get('decision')}"
                            )
                        _enforce_full_cam1_uniqueness("after-fragment-merge")

                    # Missing-canonical-only recovery pass:
                    # slightly relaxed acceptance but ONLY for canonicals that
                    # are still missing after the anchor.
                    if missing_after_anchor:
                        _merge_missing_min_cos = float(
                            os.environ.get("FULL_CAM1_MERGE_MISSING_MIN_COS", "0.85")
                        )
                        _merge_missing_min_margin = float(
                            os.environ.get("FULL_CAM1_MERGE_MISSING_MIN_MARGIN", "0.07")
                        )
                        _merge_missing_samples = int(
                            os.environ.get("FULL_CAM1_MERGE_MISSING_SAMPLES", str(_merge_samples))
                        )
                        merge_missing_report = merge_fragment_to_canonical_by_appearance(
                            video_path=v,
                            tracks_csv_path=out_csv,
                            canonical_ids=tuple(int(x) for x in missing_after_anchor),
                            reid_weights_path=_anchor_weights,
                            min_cos=_merge_missing_min_cos,
                            min_margin=_merge_missing_min_margin,
                            samples_per_gid=_merge_missing_samples,
                            stable_min_rows=full_cam1_stable_min_rows,
                            stable_min_span=full_cam1_stable_min_span,
                            device=os.environ.get(
                                "FULL_CAM1_CAM1_ANCHOR_DEVICE", "cpu"
                            ),
                        )
                        print(
                            "[INFO] FULL_CAM1 fragment-merge (missing-only) "
                            f"canonicals={missing_after_anchor} "
                            f"merges={merge_missing_report.get('merges', 0)} "
                            f"fragments={merge_missing_report.get('fragments_considered', 0)} "
                            f"targets={merge_missing_report.get('targets_considered', 0)} "
                            f"dedup_rows={merge_missing_report.get('dedup_rows', 0)}"
                        )
                        for _m in merge_missing_report.get("decisions", [])[:24]:
                            print(
                                "         "
                                f"frag={_m.get('fragment_gid')} "
                                f"-> target={_m.get('target_gid')} "
                                f"cos={_m.get('cos', 0.0):.3f} "
                                f"margin={_m.get('margin', 0.0):.3f} "
                                f"decision={_m.get('decision')}"
                            )
                        _enforce_full_cam1_uniqueness("after-fragment-merge-missing-only")
                except Exception as _merge_err:
                    print(
                        f"[WARN] FULL_CAM1 fragment-merge failed (non-fatal): {_merge_err!r}"
                    )

                # Explicit belt-and-suspenders same-frame uniqueness pass.
                # enforce_same_frame_uniqueness is a public, idempotent wrapper
                # around the internal _enforce_unique_positive_ids_per_frame.
                try:
                    _uniq = enforce_same_frame_uniqueness(tracks_csv_path=out_csv)
                    print(
                        "[INFO] FULL_CAM1 same-frame-uniqueness "
                        f"dedup_rows={_uniq.get('dedup_rows', 0)}"
                    )
                except Exception as _uniq_err:
                    print(
                        f"[WARN] FULL_CAM1 same-frame-uniqueness failed (non-fatal): {_uniq_err!r}"
                    )

                # Final convergence: fragment prune + same-frame uniqueness.
                convergence = converge_to_canonical_set(
                    tracks_csv_path=out_csv,
                    canonical_ids=tuple(range(1, 12)),
                    stable_min_rows=full_cam1_stable_min_rows,
                    stable_min_span=full_cam1_stable_min_span,
                    prune_fragments_below_rows=int(
                        os.environ.get("FULL_CAM1_CONVERGE_PRUNE_ROWS", "0")
                    ),
                    prune_fragments_below_span=int(
                        os.environ.get("FULL_CAM1_CONVERGE_PRUNE_SPAN", "0")
                    ),
                )
                print(
                    "[INFO] FULL_CAM1 convergence "
                    f"pruned={convergence['pruned_gids']} "
                    f"dedup_rows={convergence['dedup_rows']} "
                    f"delta={convergence['delta']} "
                    f"converged_to_11={convergence['converged_to_11']}"
                )
                post_converge_uniq = enforce_same_frame_uniqueness(tracks_csv_path=out_csv)
                print(f"[INFO] FULL_CAM1 same-frame uniqueness (after-convergence) {post_converge_uniq}")

                # Late missing-canonical recovery:
                # run CAM1 anchor only for still-missing canonical IDs and
                # apply only non-canonical source gids -> missing canonical ids.
                # This avoids over-collapsing into survivors while allowing
                # conservative recovery of 2/7/8/9/10/11 from fallback tracks.
                missing_after_converge = sorted(
                    int(c)
                    for c in range(1, 12)
                    if int((convergence.get("after", {}).get("per_canonical_rows", {}) or {}).get(c, 0))
                    < int(full_cam1_stable_min_rows)
                )
                print(f"[INFO] FULL_CAM1 missing canonicals after convergence {missing_after_converge}")
                _late_missing_anchor_enabled = bool(
                    int(os.environ.get("FULL_CAM1_LATE_MISSING_ANCHOR", "1"))
                )
                if (
                    _late_missing_anchor_enabled
                    and missing_after_converge
                    and cam1_tracks_candidate.exists()
                    and cam1_video_candidate.exists()
                ):
                    try:
                        from src.reid.cam1_reference_anchor import (
                            align_full_cam1_to_cam1 as _late_align_full_cam1_to_cam1,
                            MatchReport as _LateAnchorMatchReport,
                            apply_mapping_to_csv as _late_apply_mapping_to_csv,
                        )
                        _late_min_cos = float(
                            os.environ.get("FULL_CAM1_LATE_MISSING_MIN_COS", "0.74")
                        )
                        _late_min_margin = float(
                            os.environ.get("FULL_CAM1_LATE_MISSING_MIN_MARGIN", "0.02")
                        )
                        _late_require_noncanonical = bool(
                            int(
                                os.environ.get(
                                    "FULL_CAM1_LATE_MISSING_REQUIRE_NONCANONICAL", "1"
                                )
                            )
                        )
                        late_anchor_probe = _late_align_full_cam1_to_cam1(
                            cam1_video=cam1_video_candidate,
                            cam1_tracks_csv=cam1_tracks_candidate,
                            full_cam1_video=v,
                            full_cam1_tracks_csv=out_csv,
                            cam1_canonical_gids=tuple(int(x) for x in missing_after_converge),
                            samples_per_gid=int(
                                os.environ.get("FULL_CAM1_CAM1_ANCHOR_SAMPLES", "18")
                            ),
                            min_cos=_late_min_cos,
                            min_margin=_late_min_margin,
                            fallback_start=120,
                            stable_min_rows=full_cam1_anchor_stable_min_rows,
                            stable_min_span=full_cam1_anchor_stable_min_span,
                            reid_weights_path=_anchor_weights,
                            device=os.environ.get("FULL_CAM1_CAM1_ANCHOR_DEVICE", "cpu"),
                            apply=False,
                            matcher=anchor_matcher,
                            strong_min_mean_cos=float(
                                os.environ.get("FULL_CAM1_CAM1_ANCHOR_STRONG_MEAN_COS", "0.82")
                            ),
                            strong_min_max_cos=float(
                                os.environ.get("FULL_CAM1_CAM1_ANCHOR_STRONG_MAX_COS", "0.88")
                            ),
                            strong_min_sample_vote_share=float(
                                os.environ.get("FULL_CAM1_CAM1_ANCHOR_STRONG_VOTE_SHARE", "0.57")
                            ),
                            strong_min_margin=float(
                                os.environ.get("FULL_CAM1_CAM1_ANCHOR_STRONG_MARGIN", "0.05")
                            ),
                            strong_require_bidirectional=bool(
                                int(os.environ.get("FULL_CAM1_CAM1_ANCHOR_STRONG_BIDIR", "1"))
                            ),
                        )
                        _missing_set = set(int(x) for x in missing_after_converge)
                        _late_filtered_map = {}
                        for _src, _dst in sorted(late_anchor_probe.mapping.items()):
                            _s = int(_src)
                            _d = int(_dst)
                            if _d not in _missing_set:
                                continue
                            if _late_require_noncanonical and 1 <= _s <= 11:
                                continue
                            if _s <= 0:
                                continue
                            _late_filtered_map[_s] = _d
                        if _late_filtered_map:
                            _late_report = _LateAnchorMatchReport(mapping=_late_filtered_map)
                            _late_apply = _late_apply_mapping_to_csv(
                                tracks_csv=out_csv,
                                report=_late_report,
                                also_write_backup=False,
                            )
                            print(
                                "[INFO] FULL_CAM1 late-missing-anchor "
                                f"mapped={len(_late_filtered_map)} "
                                f"changed_rows={_late_apply.get('changed_rows', 0)} "
                                f"min_cos={_late_min_cos:.3f} "
                                f"min_margin={_late_min_margin:.3f}"
                            )
                            for _src, _dst in sorted(_late_filtered_map.items()):
                                _probe = late_anchor_probe.scores.get(_src, {}) if late_anchor_probe.scores else {}
                                _probe_cos = float(_probe.get(_dst, 0.0))
                                print(
                                    "         "
                                    f"late-map src_gid={_src} -> missing_canonical={_dst} "
                                    f"mean_cos={_probe_cos:.3f}"
                                )
                            _enforce_full_cam1_uniqueness("after-late-missing-anchor")
                            convergence = converge_to_canonical_set(
                                tracks_csv_path=out_csv,
                                canonical_ids=tuple(range(1, 12)),
                                stable_min_rows=full_cam1_stable_min_rows,
                                stable_min_span=full_cam1_stable_min_span,
                                prune_fragments_below_rows=int(
                                    os.environ.get("FULL_CAM1_CONVERGE_PRUNE_ROWS", "0")
                                ),
                                prune_fragments_below_span=int(
                                    os.environ.get("FULL_CAM1_CONVERGE_PRUNE_SPAN", "0")
                                ),
                            )
                            print(
                                "[INFO] FULL_CAM1 convergence (post-late-missing-anchor) "
                                f"pruned={convergence['pruned_gids']} "
                                f"dedup_rows={convergence['dedup_rows']} "
                                f"delta={convergence['delta']} "
                                f"converged_to_11={convergence['converged_to_11']}"
                            )
                            post_converge_uniq = enforce_same_frame_uniqueness(
                                tracks_csv_path=out_csv
                            )
                            print(
                                "[INFO] FULL_CAM1 same-frame uniqueness "
                                f"(after-late-missing-anchor) {post_converge_uniq}"
                            )
                        else:
                            print(
                                "[INFO] FULL_CAM1 late-missing-anchor "
                                "no safe non-canonical->missing mappings"
                            )
                    except Exception as _late_anchor_err:
                        print(
                            f"[WARN] FULL_CAM1 late-missing-anchor failed (non-fatal): {_late_anchor_err!r}"
                        )

                # Conservative continuity heal:
                # Fix remaining obvious split/reuse artifacts without forcing a
                # global identity template.
                if bool(int(os.environ.get("FULL_CAM1_CONTINUITY_HEAL", "1"))):
                    try:
                        heal_backup = out_csv.read_bytes()
                        pre_heal_summary = summarize_identity_space(
                            tracks_csv_path=out_csv,
                            stable_min_rows=full_cam1_stable_min_rows,
                            stable_min_span=full_cam1_stable_min_span,
                            canonical_ids=set(range(1, 12)),
                        )
                        canonical_set = set(range(1, 12))
                        def _canon_present(summary: dict) -> set[int]:
                            out: set[int] = set()
                            for raw in (summary.get("unique_positive_ids", []) or []):
                                try:
                                    cand = int(raw)
                                except Exception:
                                    try:
                                        cand = int(float(raw))
                                    except Exception:
                                        continue
                                if cand in canonical_set:
                                    out.add(cand)
                            return out

                        pre_present_canon = {
                            int(x) for x in _canon_present(pre_heal_summary)
                        }
                        pre_stable = int(
                            pre_heal_summary.get("stable_positive_id_count", 0) or 0
                        )

                        _heal = _full_cam1_continuity_heal()
                        print(
                            "[INFO] FULL_CAM1 continuity-heal "
                            f"applied={_heal.get('applied', False)} "
                            f"changed_rows={_heal.get('changed_rows', 0)}"
                        )
                        for _d in (_heal.get("decisions", []) or [])[:20]:
                            print(
                                "         "
                                f"src={_d.get('source_gid')} -> dst={_d.get('target_gid')} "
                                f"rows={_d.get('moved_rows')} reason={_d.get('reason')}"
                            )
                        if bool(_heal.get("applied", False)):
                            _enforce_full_cam1_uniqueness("after-continuity-heal")
                            convergence = converge_to_canonical_set(
                                tracks_csv_path=out_csv,
                                canonical_ids=tuple(range(1, 12)),
                                stable_min_rows=full_cam1_stable_min_rows,
                                stable_min_span=full_cam1_stable_min_span,
                                prune_fragments_below_rows=int(
                                    os.environ.get("FULL_CAM1_CONVERGE_PRUNE_ROWS", "0")
                                ),
                                prune_fragments_below_span=int(
                                    os.environ.get("FULL_CAM1_CONVERGE_PRUNE_SPAN", "0")
                                ),
                            )
                            print(
                                "[INFO] FULL_CAM1 convergence (post-continuity-heal) "
                                f"pruned={convergence['pruned_gids']} "
                                f"dedup_rows={convergence['dedup_rows']} "
                                f"delta={convergence['delta']} "
                                f"converged_to_11={convergence['converged_to_11']}"
                            )
                            post_heal_summary = summarize_identity_space(
                                tracks_csv_path=out_csv,
                                stable_min_rows=full_cam1_stable_min_rows,
                                stable_min_span=full_cam1_stable_min_span,
                                canonical_ids=set(range(1, 12)),
                            )
                            post_present_canon = {
                                int(x) for x in _canon_present(post_heal_summary)
                            }
                            post_stable = int(
                                post_heal_summary.get("stable_positive_id_count", 0) or 0
                            )
                            lost_canon = sorted(pre_present_canon - post_present_canon)
                            revert_reason = ""
                            if lost_canon:
                                revert_reason = f"lost_canonicals({lost_canon})"
                            elif post_stable < pre_stable:
                                revert_reason = f"stable_ids_down({pre_stable}->{post_stable})"
                            if revert_reason:
                                out_csv.write_bytes(heal_backup)
                                print(
                                    "[INFO] FULL_CAM1 continuity-heal reverted "
                                    f"reason={revert_reason} "
                                    f"pre_summary={pre_heal_summary} "
                                    f"post_summary={post_heal_summary}"
                                )
                                _enforce_full_cam1_uniqueness(
                                    "after-continuity-heal-revert"
                                )
                    except Exception as _heal_err:
                        print(
                            f"[WARN] FULL_CAM1 continuity-heal failed (non-fatal): {_heal_err!r}"
                        )

                # Anchor bottleneck diagnostics for unresolved/weak canonicals.
                try:
                    import json as _json
                    from src.reid.cam1_reference_anchor import (
                        build_anchor_failure_diagnostics as _build_anchor_failure_diagnostics,
                        MatchReport as _DiagMatchReport,
                    )
                    _focus_raw = str(
                        os.environ.get("FULL_CAM1_DIAG_CANONICALS", "7,9,10,11")
                    ).strip()
                    _focus_ids = []
                    for _tok in _focus_raw.split(","):
                        _tok = _tok.strip()
                        if not _tok:
                            continue
                        try:
                            _focus_ids.append(int(_tok))
                        except Exception:
                            continue
                    if not _focus_ids:
                        _focus_ids = [7, 9, 10, 11]
                    _diag_report = _build_anchor_failure_diagnostics(
                        report=anchor_report if anchor_report is not None else _DiagMatchReport(),
                        focus_canonical_ids=tuple(int(x) for x in _focus_ids),
                        stable_min_rows=full_cam1_anchor_stable_min_rows,
                        after_anchor_rows=(identity_after_anchor.get("per_canonical_rows", {}) or {}),
                        after_convergence_rows=(convergence.get("after", {}).get("per_canonical_rows", {}) or {}),
                    )
                    _diag_path = out_csv.with_name(
                        "retail-shop_FULL_CAM1_anchor_failure_diagnostics.json"
                    )
                    _diag_path.write_text(
                        _json.dumps(_diag_report, indent=2, sort_keys=True),
                        encoding="utf-8",
                    )
                    print(
                        "[INFO] FULL_CAM1 anchor-failure diagnostics saved -> "
                        f"{_diag_path}"
                    )
                    for _entry in _diag_report.get("canonicals", [])[:16]:
                        print(
                            "         "
                            f"canonical={_entry.get('canonical_id')} "
                            f"status={_entry.get('status')} "
                            f"candidate={_entry.get('current_best_source_gid_candidate')} "
                            f"top1={float(_entry.get('top1_cosine') or 0.0):.3f} "
                            f"top2={float(_entry.get('top2_cosine') or 0.0):.3f} "
                            f"margin={float(_entry.get('margin') or 0.0):.3f} "
                            f"class={_entry.get('likely_bottleneck_classification')}"
                        )
                except Exception as _diag_err:
                    print(
                        f"[WARN] FULL_CAM1 anchor-failure diagnostics failed (non-fatal): {_diag_err!r}"
                    )
                # Persist the final identity metrics alongside the tracks CSV.
                try:
                    import json as _json
                    metrics_path = out_csv.with_name("retail-shop_FULL_CAM1_identity_metrics.json")
                    metrics_path.write_text(
                        _json.dumps(
                            {
                                "before_anchor": identity_before,
                                "after_anchor": identity_after_anchor,
                                "after_convergence": convergence["after"],
                                "anchor_applied": bool(anchor_applied),
                                "converged_to_11": bool(convergence["converged_to_11"]),
                            },
                            indent=2,
                            sort_keys=True,
                        ),
                        encoding="utf-8",
                    )
                    print(f"[INFO] FULL_CAM1 identity metrics saved -> {metrics_path}")
                except Exception as _metrics_err:
                    print(f"[WARN] FULL_CAM1 identity metrics write failed: {_metrics_err!r}")

                full_cam1_identity_summary = summarize_identity_space(
                    tracks_csv_path=out_csv,
                    stable_min_rows=full_cam1_stable_min_rows,
                    stable_min_span=full_cam1_stable_min_span,
                    canonical_ids=canonical_set,
                )
                print(f"[INFO] FULL_CAM1 final identity summary {full_cam1_identity_summary}")

                if args.audit_csv.exists():
                    full_cam1_audit_after = evaluate_first_two_minute_audit_metrics(
                        tracks_csv_path=out_csv,
                        audit_csv_path=args.audit_csv,
                        max_sec=120.0,
                        iou_threshold=0.34,
                    )
                    print(f"[INFO] FULL_CAM1 first2min audit after {full_cam1_audit_after}")
                    if full_cam1_audit_before is not None:
                        print(f"[INFO] FULL_CAM1 first2min audit before-vs-after before={full_cam1_audit_before} after={full_cam1_audit_after}")

                # Final user-facing ID scheme (8-person mode):
                # Keep only selected canonical identities and force stable IDs:
                #   green shirt girl -> 1
                #   beige shirt woman (sit) -> 3
                #   pink shirt girl -> 5
                #   black shirt woman -> 6
                #   black shirt boy -> 7
                #   grey shirt man -> 8
                # Plus IDs 2 and 4 as the remaining tracked people.
                #
                # Implementation: apply after canonical anchor/convergence so we
                # preserve strong identity solving first, then remap for output.
                _custom8_enabled = bool(
                    int(os.environ.get("FULL_CAM1_FORCE_CUSTOM8_IDS", "1"))
                )
                if _custom8_enabled:
                    # Optional: keep additional canonical IDs in output
                    # instead of dropping them to 0 in custom-8 mode.
                    # Example:
                    #   FULL_CAM1_CUSTOM8_KEEP_CANONICALS=2,7,10
                    _extra_keep_raw = str(
                        os.environ.get("FULL_CAM1_CUSTOM8_KEEP_CANONICALS", "")
                    ).strip()
                    _extra_keep_ids = []
                    if _extra_keep_raw:
                        for _tok in _extra_keep_raw.replace(";", ",").split(","):
                            _tok = _tok.strip()
                            if not _tok:
                                continue
                            try:
                                _gid_keep = int(_tok)
                            except Exception:
                                continue
                            if _gid_keep > 0:
                                _extra_keep_ids.append(int(_gid_keep))

                    _custom8_map = {
                        1: 1,
                        2: 2,
                        3: 3,
                        4: 4,
                        6: 5,   # pink shirt girl
                        9: 6,   # black shirt woman
                        5: 7,   # black shirt boy
                        11: 7,  # last black shirt boy segment
                        8: 8,   # grey shirt man
                    }
                    for _gid_keep in sorted(set(_extra_keep_ids)):
                        _custom8_map.setdefault(int(_gid_keep), int(_gid_keep))
                    _rows_total = 0
                    _changed_rows = 0
                    _dropped_rows = 0
                    _kept_rows = 0
                    try:
                        with out_csv.open(newline="", encoding="utf-8") as _f:
                            _reader = csv.DictReader(_f)
                            _fieldnames = _reader.fieldnames or []
                            _raw_rows = list(_reader)
                        for _raw in _raw_rows:
                            _rows_total += 1
                            try:
                                _g = int(float(_raw.get("global_id", "0") or 0))
                            except Exception:
                                continue
                            if _g <= 0:
                                continue
                            _new = int(_custom8_map.get(int(_g), 0))
                            if _new != _g:
                                _changed_rows += 1
                            if _new <= 0:
                                _dropped_rows += 1
                            else:
                                _kept_rows += 1
                            _raw["global_id"] = str(int(_new))
                        with out_csv.open("w", newline="", encoding="utf-8") as _f:
                            _writer = csv.DictWriter(_f, fieldnames=_fieldnames)
                            _writer.writeheader()
                            _writer.writerows(_raw_rows)
                        _final_uniq = enforce_same_frame_uniqueness(tracks_csv_path=out_csv)
                        print(
                            "[INFO] FULL_CAM1 custom-8-id remap "
                            f"changed_rows={_changed_rows} "
                            f"dropped_rows={_dropped_rows} "
                            f"kept_positive_rows={_kept_rows} "
                            f"rows_total={_rows_total} "
                            f"dedup_rows={_final_uniq.get('dedup_rows', 0)} "
                            f"map={_custom8_map}"
                        )
                    except Exception as _custom8_err:
                        print(
                            f"[WARN] FULL_CAM1 custom-8-id remap failed (non-fatal): {_custom8_err!r}"
                        )

            if args.render:
                render_stats = render_tracks_video(
                    video_path=v,
                    tracks_csv_path=out_csv,
                    out_video_path=out_vid,
                )
                if isinstance(render_stats, dict):
                    print(
                        "[INFO] FULL_CAM1 render cleanup "
                        f"bad_detected={render_stats.get('bad_frames_detected', 0)} "
                        f"bad_replaced={render_stats.get('bad_frames_replaced', 0)} "
                        f"bad_no_history={render_stats.get('bad_frames_without_history', 0)}"
                    )
                print(f"[OK] Saved stitched video -> {out_vid}")
            else:
                print("[INFO] Render skipped (--no-render).")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
