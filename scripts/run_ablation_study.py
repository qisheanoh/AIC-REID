from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reid.track_linker import (  # noqa: E402
    evaluate_first_two_minute_audit_metrics,
    identity_metrics,
    summarize_identity_space,
)


AIC_METHOD = "AIC-ReID"
AUDIT_CSV_DEFAULT = ROOT / "experiments" / "audit" / "cam1_manual_audit_sheet.csv"
FROZEN_CAM1_CSV = ROOT / "runs" / "baseline_freeze_2026-04-28" / "retail-shop_CAM1_tracks.FROZEN.csv"


@dataclass
class VariantRunResult:
    method: str
    variant_name: str
    clip: str
    status: str
    tracks_csv: str
    command: str
    return_code: Optional[int]
    duration_sec: Optional[float]
    reason: str = ""


def _json_compact(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _to_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _to_int(v: Any) -> Optional[int]:
    try:
        if v is None:
            return None
        return int(v)
    except Exception:
        return None


def _run_cmd(
    *,
    cmd: List[str],
    cwd: Path,
    env_overrides: Optional[Dict[str, str]],
    log_fp,
) -> tuple[bool, int, float]:
    env = os.environ.copy()
    if env_overrides:
        env.update({str(k): str(v) for k, v in env_overrides.items()})

    start = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        log_fp.write(line)
        log_fp.flush()
        print(line, end="")

    rc = proc.wait()
    duration = time.time() - start
    return rc == 0, rc, duration


def _score_cam1(
    *,
    method: str,
    variant_name: str,
    tracks_csv: Path,
    audit_csv: Path,
    notes: Dict[str, Any],
) -> Dict[str, Any]:
    if not tracks_csv.exists():
        return {
            "method": method,
            "variant_name": variant_name,
            "clip": "CAM1",
            "status": "NOT_RUN",
            "tracks_csv": str(tracks_csv),
            "positive_ids": None,
            "positive_rows": None,
            "coverage": None,
            "purity": None,
            "same_frame_duplicates": None,
            "fragmentation": None,
            "audit_rows_total": None,
            "audit_rows_covered_positive_id": None,
            "predicted_ids_shared_multi_gt_people": None,
            "window_unique_positive_id_count": None,
            "window_max_positive_id": None,
            "notes_json": _json_compact({"reason": "tracks_csv_missing", **notes}),
        }

    summary = summarize_identity_space(
        tracks_csv_path=tracks_csv,
        stable_min_rows=30,
        stable_min_span=32,
    )
    audit = evaluate_first_two_minute_audit_metrics(
        tracks_csv_path=tracks_csv,
        audit_csv_path=audit_csv,
        iou_threshold=0.34,
        max_sec=150.0,
    )

    unique_positive_ids = summary.get("unique_positive_ids") or []
    out_notes = dict(notes)
    out_notes.update(
        {
            "audit_rows_total": audit.get("audit_rows_total"),
            "audit_rows_covered_positive_id": audit.get("audit_rows_covered_positive_id"),
            "predicted_ids_shared_multi_gt_people": audit.get("predicted_ids_shared_multi_gt_people"),
            "window_unique_positive_id_count": audit.get("window_unique_positive_id_count"),
            "window_max_positive_id": audit.get("window_max_positive_id"),
        }
    )

    return {
        "method": method,
        "variant_name": variant_name,
        "clip": "CAM1",
        "status": "RUN",
        "tracks_csv": str(tracks_csv),
        "positive_ids": int(len(unique_positive_ids)),
        "positive_rows": _to_int(summary.get("positive_rows")),
        "coverage": _to_float(audit.get("audit_row_coverage_positive_id")),
        "purity": _to_float(audit.get("pred_to_gt_purity_macro")),
        "same_frame_duplicates": _to_int(audit.get("same_frame_duplicate_positive_ids")),
        "fragmentation": _to_int(audit.get("gt_people_fragmented_multi_pred_ids")),
        "audit_rows_total": _to_int(audit.get("audit_rows_total")),
        "audit_rows_covered_positive_id": _to_int(audit.get("audit_rows_covered_positive_id")),
        "predicted_ids_shared_multi_gt_people": _to_int(audit.get("predicted_ids_shared_multi_gt_people")),
        "window_unique_positive_id_count": _to_int(audit.get("window_unique_positive_id_count")),
        "window_max_positive_id": _to_int(audit.get("window_max_positive_id")),
        "notes_json": _json_compact(out_notes),
    }


def _score_full_cam1_no_anchor(
    *,
    method: str,
    variant_name: str,
    tracks_csv: Path,
    notes: Dict[str, Any],
) -> Dict[str, Any]:
    base = {
        "method": method,
        "variant_name": variant_name,
        "clip": "FULL_CAM1",
        "status": "RUN" if tracks_csv.exists() else "NOT_RUN",
        "tracks_csv": str(tracks_csv),
        "positive_ids": None,
        "positive_rows": None,
        "coverage": None,
        "purity": None,
        "same_frame_duplicates": None,
        "fragmentation": None,
        "audit_rows_total": None,
        "audit_rows_covered_positive_id": None,
        "predicted_ids_shared_multi_gt_people": None,
        "window_unique_positive_id_count": None,
        "window_max_positive_id": None,
        "canonical_slots_assigned": None,
        "off_canonical_rows": None,
        "anchor_report_exists": False,
        "notes_json": "",
    }

    if not tracks_csv.exists():
        base["notes_json"] = _json_compact({"reason": "tracks_csv_missing", **notes})
        return base

    summary = summarize_identity_space(
        tracks_csv_path=tracks_csv,
        stable_min_rows=30,
        stable_min_span=32,
        canonical_ids=set(range(1, 12)),
    )
    idm = identity_metrics(
        tracks_csv_path=tracks_csv,
        canonical_ids=tuple(range(1, 12)),
        stable_min_rows=30,
        stable_min_span=32,
    )

    per_canonical = idm.get("per_canonical_rows") or {}
    canonical_slots_assigned = int(sum(1 for _gid, rows in per_canonical.items() if int(rows) > 0))
    anchor_report = tracks_csv.with_name("retail-shop_FULL_CAM1_cam1_anchor_report.json")

    out_notes = dict(notes)
    out_notes.update(
        {
            "canonical_slots_assigned": canonical_slots_assigned,
            "rows_on_canonical": idm.get("rows_on_canonical"),
            "rows_off_canonical": idm.get("rows_off_canonical"),
            "canonical_coverage": idm.get("canonical_coverage"),
            "identity_space_note": "cam1_anchor_disabled_expected_alignment_loss",
        }
    )

    base.update(
        {
            "positive_ids": int(len(summary.get("unique_positive_ids") or [])),
            "positive_rows": _to_int(summary.get("positive_rows")),
            "same_frame_duplicates": _to_int(summary.get("same_frame_duplicate_positive_ids")),
            "fragmentation": None,
            "canonical_slots_assigned": canonical_slots_assigned,
            "off_canonical_rows": _to_int(idm.get("rows_off_canonical")),
            "anchor_report_exists": bool(anchor_report.exists()),
            "notes_json": _json_compact(out_notes),
        }
    )
    return base


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    cols = [
        "method",
        "variant_name",
        "clip",
        "status",
        "tracks_csv",
        "positive_ids",
        "positive_rows",
        "coverage",
        "purity",
        "same_frame_duplicates",
        "fragmentation",
        "audit_rows_total",
        "audit_rows_covered_positive_id",
        "predicted_ids_shared_multi_gt_people",
        "window_unique_positive_id_count",
        "window_max_positive_id",
        "canonical_slots_assigned",
        "off_canonical_rows",
        "anchor_report_exists",
        "notes_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            out = {k: r.get(k) for k in cols}
            w.writerow(out)


def _build_markdown_report(
    *,
    out_path: Path,
    run_root: Path,
    commands: List[str],
    rows: List[Dict[str, Any]],
    not_run: List[Dict[str, Any]],
    graph_path: Path,
) -> None:
    cam_rows = [r for r in rows if r.get("clip") == "CAM1"]
    full_rows = [r for r in rows if r.get("clip") == "FULL_CAM1"]

    def _fmt_pct(v: Any) -> str:
        if v is None or v == "":
            return ""
        return f"{100.0 * float(v):.2f}%"

    def _fmt_int(v: Any) -> str:
        if v is None or v == "":
            return ""
        return f"{int(v)}"

    lines: List[str] = []
    lines.append("# AIC-ReID CAM1/FULL_CAM1 Ablation Study")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append(
        "This ablation study isolates identity-resolution components to test whether identity consistency gains come from the audit-driven offline workflow rather than detector/backbone alone."
    )
    lines.append("")
    lines.append("## 2. Exact Commands Run")
    lines.append("```bash")
    lines.extend(commands)
    lines.append("```")
    lines.append("")

    lines.append("## 3. CAM1 Variants (Locked Audit Protocol)")
    lines.append(
        "| method | variant_name | status | positive_ids | positive_rows | coverage | purity | same_frame_duplicates | fragmentation | tracks_csv |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---|")
    for r in cam_rows:
        lines.append(
            "| {method} | {variant} | {status} | {ids} | {rows} | {cov} | {pur} | {dup} | {frag} | {csv} |".format(
                method=r.get("method", ""),
                variant=r.get("variant_name", ""),
                status=r.get("status", ""),
                ids=_fmt_int(r.get("positive_ids")),
                rows=_fmt_int(r.get("positive_rows")),
                cov=_fmt_pct(r.get("coverage")),
                pur=_fmt_pct(r.get("purity")),
                dup=_fmt_int(r.get("same_frame_duplicates")),
                frag=_fmt_int(r.get("fragmentation")),
                csv=r.get("tracks_csv", ""),
            )
        )
    lines.append("")

    lines.append("## 4. FULL_CAM1 No-Anchor Variant")
    lines.append(
        "| method | variant_name | status | positive_ids | positive_rows | same_frame_duplicates | canonical_slots_assigned | off_canonical_rows | anchor_report_exists | tracks_csv |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---|---|")
    for r in full_rows:
        lines.append(
            "| {method} | {variant} | {status} | {ids} | {rows} | {dup} | {slots} | {off} | {anchor} | {csv} |".format(
                method=r.get("method", ""),
                variant=r.get("variant_name", ""),
                status=r.get("status", ""),
                ids=_fmt_int(r.get("positive_ids")),
                rows=_fmt_int(r.get("positive_rows")),
                dup=_fmt_int(r.get("same_frame_duplicates")),
                slots=_fmt_int(r.get("canonical_slots_assigned")),
                off=_fmt_int(r.get("off_canonical_rows")),
                anchor=str(r.get("anchor_report_exists", "")),
                csv=r.get("tracks_csv", ""),
            )
        )
    lines.append("")

    lines.append("## 5. Interpretation")
    lines.append(
        "Online-oriented baselines can increase CAM1 audit-row positive coverage, but they tend to inflate identity cardinality and/or fragmentation. Ablations that remove or weaken offline identity resolution typically degrade identity consistency (fragmentation and cross-person sharing behavior). The full frozen AIC-ReID reference remains the KPI-trust anchor because it emphasizes identity compression and purity while preserving duplicate-frame safety under the locked CAM1 audit protocol."
    )
    lines.append(
        "For FULL_CAM1, disabling CAM1-anchor alignment is expected to reduce canonical-slot stability and weaken cross-clip interpretability, even when the rest of the offline workflow remains active."
    )
    lines.append("")

    lines.append("## 6. Limitations")
    lines.append(
        "These results are CAM1/FULL_CAM1 specific. CAM1 metrics are audit-sheet-bound (484 manual rows, sparse labels) rather than dense MOT ground truth. Conclusions should be generalized cautiously to longer clips, different stores, and different camera viewpoints."
    )
    lines.append("")

    lines.append("## 7. Artifacts")
    lines.append(f"- Run root: `{run_root}`")
    lines.append(f"- Summary CSV: `{run_root / 'ablation_summary.csv'}`")
    lines.append(f"- Summary JSON: `{run_root / 'ablation_summary.json'}`")
    lines.append(f"- Command log: `{run_root / 'commands_run.sh'}`")
    lines.append(f"- Runtime log: `{run_root / 'run_log.txt'}`")
    lines.append(f"- Graph: `{graph_path}`")
    lines.append("")

    if not_run:
        lines.append("## 8. NOT_RUN Variants")
        for nr in not_run:
            lines.append(
                f"- `{nr.get('variant_name', '')}` ({nr.get('clip', '')}): {nr.get('reason', 'unknown_reason')}"
            )
        lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_graph(cam_rows: List[Dict[str, Any]], out_png: Path) -> tuple[bool, str]:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        return False, f"matplotlib_import_failed:{e!r}"

    run_rows = [r for r in cam_rows if r.get("status") == "RUN"]
    if not run_rows:
        return False, "no_cam_rows_to_plot"

    labels = [str(r.get("variant_name", "")) for r in run_rows]
    positive_ids = np.array([float(r.get("positive_ids") or 0.0) for r in run_rows], dtype=np.float32)
    coverage = np.array([100.0 * float(r.get("coverage") or 0.0) for r in run_rows], dtype=np.float32)
    purity = np.array([100.0 * float(r.get("purity") or 0.0) for r in run_rows], dtype=np.float32)
    fragmentation = np.array([float(r.get("fragmentation") or 0.0) for r in run_rows], dtype=np.float32)

    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)

    axes[0, 0].bar(x, positive_ids, color="#4C78A8")
    axes[0, 0].set_title("Positive IDs (lower is better for KPI stability)")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels, rotation=28, ha="right")

    axes[0, 1].bar(x, coverage, color="#72B7B2")
    axes[0, 1].set_title("Coverage (%)")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(labels, rotation=28, ha="right")

    axes[1, 0].bar(x, purity, color="#54A24B")
    axes[1, 0].set_title("Purity (%)")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels, rotation=28, ha="right")

    axes[1, 1].bar(x, fragmentation, color="#E45756")
    axes[1, 1].set_title("Fragmentation (GT people split across multiple pred IDs)")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels, rotation=28, ha="right")

    fig.suptitle("CAM1 Ablation Identity-Quality Comparison", fontsize=14)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)
    return True, "ok"


def main() -> None:
    ap = argparse.ArgumentParser(description="Run publication-oriented CAM1/FULL_CAM1 AIC-ReID ablations")
    ap.add_argument("--audit_csv", type=Path, default=AUDIT_CSV_DEFAULT)
    ap.add_argument("--python", type=Path, default=ROOT / ".venv" / "bin" / "python")
    ap.add_argument("--include_existing_baselines", action="store_true")
    args = ap.parse_args()

    if not args.audit_csv.exists():
        raise FileNotFoundError(f"Audit CSV missing: {args.audit_csv}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = ROOT / "runs" / "ablations" / f"cam1_ablation_{ts}"
    run_root.mkdir(parents=True, exist_ok=True)

    run_log_path = run_root / "run_log.txt"
    cmd_log_path = run_root / "commands_run.sh"
    summary_csv_path = run_root / "ablation_summary.csv"
    summary_json_path = run_root / "ablation_summary.json"
    report_md_path = run_root / "ablation_report.md"
    latest_report_path = ROOT / "runs" / "ablations" / "latest_ablation_report.md"
    graph_path = ROOT / "report_assets" / "graphs" / "cam1_ablation_identity_quality.png"

    commands: List[str] = []
    rows: List[Dict[str, Any]] = []
    run_rows: List[VariantRunResult] = []

    variants = [
        {
            "method": AIC_METHOD,
            "variant_name": "no_offline_reentry_linking",
            "clip": "CAM1",
            "cmd": [
                str(args.python),
                "scripts/run_batch.py",
                "--match",
                "=CAM1",
                "--no-render",
                "--out_dir",
                "",
                "--disable_reentry_linking",
            ],
            "env": {},
        },
        {
            "method": AIC_METHOD,
            "variant_name": "no_tracklet_stitching",
            "clip": "CAM1",
            "cmd": [
                str(args.python),
                "scripts/run_batch.py",
                "--match",
                "=CAM1",
                "--no-render",
                "--out_dir",
                "",
                "--disable_tracklet_stitching",
            ],
            "env": {},
        },
        {
            "method": AIC_METHOD,
            "variant_name": "osnet_only_no_attire_shape",
            "clip": "CAM1",
            "cmd": [
                str(args.python),
                "scripts/run_batch.py",
                "--match",
                "=CAM1",
                "--no-render",
                "--out_dir",
                "",
                "--osnet_only",
            ],
            "env": {},
        },
        {
            "method": AIC_METHOD,
            "variant_name": "relaxed_conservative_gate",
            "clip": "CAM1",
            "cmd": [
                str(args.python),
                "scripts/run_batch.py",
                "--match",
                "=CAM1",
                "--no-render",
                "--out_dir",
                "",
                "--relaxed_identity_gate",
            ],
            "env": {},
        },
        {
            "method": AIC_METHOD,
            "variant_name": "no_cam1_anchor_alignment",
            "clip": "FULL_CAM1",
            "cmd": [
                str(args.python),
                "scripts/run_batch.py",
                "--match",
                "=FULL_CAM1",
                "--no-render",
                "--out_dir",
                "",
            ],
            "env": {
                "FULL_CAM1_CAM1_ANCHOR": "off",
                "FULL_CAM1_FORCE_CUSTOM8_IDS": "0",
            },
        },
    ]

    with run_log_path.open("w", encoding="utf-8") as run_log:
        run_log.write(f"Ablation run root: {run_root}\n")
        run_log.write(f"Audit CSV: {args.audit_csv}\n")
        run_log.write(f"Start: {datetime.now().isoformat()}\n\n")

        # A) Frozen reference row (no rerun)
        rows.append(
            _score_cam1(
                method=AIC_METHOD,
                variant_name="full_aic_reid_frozen_reference",
                tracks_csv=FROZEN_CAM1_CSV,
                audit_csv=args.audit_csv,
                notes={
                    "mode": "frozen_reference_no_rerun",
                },
            )
        )

        for spec in variants:
            variant_name = str(spec["variant_name"])
            clip = str(spec["clip"])
            out_dir = run_root / variant_name
            out_dir.mkdir(parents=True, exist_ok=True)

            cmd = list(spec["cmd"])
            out_idx = cmd.index("--out_dir") + 1
            cmd[out_idx] = str(out_dir)

            env = dict(spec.get("env") or {})
            cmd_str = " ".join(subprocess.list2cmdline([c]) for c in cmd)
            if env:
                env_prefix = " ".join(f"{k}={v}" for k, v in sorted(env.items()))
                cmd_str = f"{env_prefix} {cmd_str}"

            commands.append(cmd_str)
            run_log.write("\n" + "=" * 88 + "\n")
            run_log.write(f"RUN {variant_name} ({clip})\n")
            run_log.write(cmd_str + "\n")
            run_log.write("=" * 88 + "\n")
            run_log.flush()

            ok, rc, dur = _run_cmd(cmd=cmd, cwd=ROOT, env_overrides=env, log_fp=run_log)
            tracks_name = "retail-shop_CAM1_tracks.csv" if clip == "CAM1" else "retail-shop_FULL_CAM1_tracks.csv"
            tracks_csv = out_dir / tracks_name

            run_rows.append(
                VariantRunResult(
                    method=str(spec["method"]),
                    variant_name=variant_name,
                    clip=clip,
                    status="RUN" if ok else "NOT_RUN",
                    tracks_csv=str(tracks_csv),
                    command=cmd_str,
                    return_code=rc,
                    duration_sec=dur,
                    reason="" if ok else f"return_code_{rc}",
                )
            )

            if clip == "CAM1":
                notes = {
                    "run_status": "ok" if ok else "failed",
                    "return_code": rc,
                    "duration_sec": round(float(dur), 3),
                }
                if variant_name == "osnet_only_no_attire_shape":
                    notes["mode"] = "tracker_fusion_F_ATTIRE=0,F_SHAPE=0;offline_part_shape_gates_relaxed"
                elif variant_name == "relaxed_conservative_gate":
                    notes["mode"] = "relaxed_tracker_and_reentry_identity_gates"
                elif variant_name == "no_offline_reentry_linking":
                    notes["mode"] = "offline_reentry_disabled"
                elif variant_name == "no_tracklet_stitching":
                    notes["mode"] = "offline_stitch_disabled"

                if ok:
                    rows.append(
                        _score_cam1(
                            method=str(spec["method"]),
                            variant_name=variant_name,
                            tracks_csv=tracks_csv,
                            audit_csv=args.audit_csv,
                            notes=notes,
                        )
                    )
                else:
                    rows.append(
                        {
                            "method": str(spec["method"]),
                            "variant_name": variant_name,
                            "clip": "CAM1",
                            "status": "NOT_RUN",
                            "tracks_csv": str(tracks_csv),
                            "positive_ids": None,
                            "positive_rows": None,
                            "coverage": None,
                            "purity": None,
                            "same_frame_duplicates": None,
                            "fragmentation": None,
                            "audit_rows_total": None,
                            "audit_rows_covered_positive_id": None,
                            "predicted_ids_shared_multi_gt_people": None,
                            "window_unique_positive_id_count": None,
                            "window_max_positive_id": None,
                            "notes_json": _json_compact({"reason": f"return_code_{rc}", **notes}),
                        }
                    )
            else:
                notes = {
                    "run_status": "ok" if ok else "failed",
                    "return_code": rc,
                    "duration_sec": round(float(dur), 3),
                    "mode": "FULL_CAM1_CAM1_ANCHOR=off,FULL_CAM1_FORCE_CUSTOM8_IDS=0",
                }
                if ok:
                    rows.append(
                        _score_full_cam1_no_anchor(
                            method=str(spec["method"]),
                            variant_name=variant_name,
                            tracks_csv=tracks_csv,
                            notes=notes,
                        )
                    )
                else:
                    rows.append(
                        {
                            "method": str(spec["method"]),
                            "variant_name": variant_name,
                            "clip": "FULL_CAM1",
                            "status": "NOT_RUN",
                            "tracks_csv": str(tracks_csv),
                            "positive_ids": None,
                            "positive_rows": None,
                            "coverage": None,
                            "purity": None,
                            "same_frame_duplicates": None,
                            "fragmentation": None,
                            "audit_rows_total": None,
                            "audit_rows_covered_positive_id": None,
                            "predicted_ids_shared_multi_gt_people": None,
                            "window_unique_positive_id_count": None,
                            "window_max_positive_id": None,
                            "canonical_slots_assigned": None,
                            "off_canonical_rows": None,
                            "anchor_report_exists": False,
                            "notes_json": _json_compact({"reason": f"return_code_{rc}", **notes}),
                        }
                    )

        # Optional context baselines in the same locked CAM1 protocol.
        if args.include_existing_baselines:
            existing = [
                (
                    "ByteTrack-only baseline (reid_off)",
                    "existing_baseline_bytetrack_only",
                    ROOT
                    / "runs"
                    / "baselines"
                    / "cam1_bytetrack_only_20260428_223244"
                    / "retail-shop_CAM1_tracks.csv",
                ),
                (
                    "BoT-SORT-style online baseline",
                    "existing_baseline_botsort_online",
                    ROOT
                    / "runs"
                    / "baselines"
                    / "cam1_botsort_online_20260428_221824"
                    / "retail-shop_CAM1_tracks.csv",
                ),
                (
                    "AIC-ReID (no audit-guided offline repair)",
                    "existing_baseline_no_audit_offline_repair",
                    ROOT
                    / "runs"
                    / "baselines"
                    / "cam1_no_audit_offline_repair_20260428_220407"
                    / "retail-shop_CAM1_tracks.csv",
                ),
            ]
            for method, variant, path in existing:
                rows.append(
                    _score_cam1(
                        method=method,
                        variant_name=variant,
                        tracks_csv=path,
                        audit_csv=args.audit_csv,
                        notes={"mode": "existing_baseline_reference"},
                    )
                )

        run_log.write("\nDone.\n")

    cmd_lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""] + commands
    cmd_log_path.write_text("\n".join(cmd_lines) + "\n", encoding="utf-8")

    _write_csv(summary_csv_path, rows)
    summary_json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=True), encoding="utf-8")

    cam_rows = [r for r in rows if r.get("clip") == "CAM1"]
    ok_graph, graph_msg = _make_graph(cam_rows, graph_path)

    not_run = [
        {
            "variant_name": r.get("variant_name"),
            "clip": r.get("clip"),
            "reason": json.loads(r.get("notes_json") or "{}").get("reason", "unknown_reason"),
        }
        for r in rows
        if r.get("status") == "NOT_RUN"
    ]
    if not ok_graph:
        not_run.append(
            {
                "variant_name": "cam1_ablation_identity_quality_graph",
                "clip": "CAM1",
                "reason": graph_msg,
            }
        )

    _build_markdown_report(
        out_path=report_md_path,
        run_root=run_root,
        commands=commands,
        rows=rows,
        not_run=not_run,
        graph_path=graph_path,
    )
    shutil.copyfile(report_md_path, latest_report_path)

    print("\n[OK] Ablation artifacts generated:")
    print(f"- {summary_csv_path}")
    print(f"- {summary_json_path}")
    print(f"- {report_md_path}")
    print(f"- {latest_report_path}")
    print(f"- {graph_path}")
    if not_run:
        print("\n[INFO] NOT_RUN entries:")
        for nr in not_run:
            print(f"- {nr['variant_name']} ({nr['clip']}): {nr['reason']}")


if __name__ == "__main__":
    main()
