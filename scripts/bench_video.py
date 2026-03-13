#!/usr/bin/env python3
"""Video benchmark harness for CorridorKey MLX temporal experiments.

Supports V0 baseline (skip=1) and V2 backbone skip (skip=2,3,5).
When skip>1, compares per-frame fidelity against V0 reference PNGs.

Usage:
    uv run python scripts/bench_video.py                    # V0 baseline
    uv run python scripts/bench_video.py --skip 3           # V2 skip=3
    uv run python scripts/bench_video.py --sweep 2 3 5      # V2 sweep
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from corridorkey_mlx.inference.pipeline import DEFAULT_CHECKPOINT, load_model
from corridorkey_mlx.inference.video import FrameResult, VideoProcessor

DEFAULT_IMG_SIZE = 1024
DEFAULT_INPUT = Path("reference/video/Input.mp4")
DEFAULT_HINT = Path("reference/video/hint.mp4")
REFERENCE_DIR = Path("research/artifacts/video_v0_reference")
ARTIFACT_DIR = Path("research/artifacts")
WARMUP_FRAMES = 3
FIDELITY_THRESHOLD = 5e-3


def _load_reference_frame(ref_dir: Path, frame_idx: int, key: str) -> np.ndarray:
    """Load a V0 reference PNG as float32 [0,1]."""
    path = ref_dir / f"{key}_{frame_idx:04d}.png"
    if not path.exists():
        msg = f"Reference frame not found: {path}"
        raise FileNotFoundError(msg)
    img = np.asarray(Image.open(path), dtype=np.float32) / 255.0
    return img


def _compute_fidelity(
    results: list[FrameResult], ref_dir: Path
) -> dict[str, object]:
    """Compare per-frame outputs against V0 reference. Returns fidelity metrics."""
    alpha_errors: list[float] = []
    fg_errors: list[float] = []
    failed_frames: list[dict[str, object]] = []

    for r in results:
        # Alpha: (H, W) uint8 -> float32
        alpha_f32 = r.alpha.astype(np.float32) / 255.0
        ref_alpha = _load_reference_frame(ref_dir, r.frame_idx, "alpha")
        alpha_err = float(np.max(np.abs(alpha_f32 - ref_alpha)))
        alpha_errors.append(alpha_err)

        # FG: (H, W, 3) uint8 -> float32
        fg_f32 = r.fg.astype(np.float32) / 255.0
        ref_fg = _load_reference_frame(ref_dir, r.frame_idx, "fg")
        fg_err = float(np.max(np.abs(fg_f32 - ref_fg)))
        fg_errors.append(fg_err)

        max_err = max(alpha_err, fg_err)
        if max_err > FIDELITY_THRESHOLD:
            failed_frames.append({
                "frame": r.frame_idx,
                "alpha_err": round(alpha_err, 6),
                "fg_err": round(fg_err, 6),
                "is_keyframe": r.is_keyframe,
            })

    alpha_arr = np.array(alpha_errors)
    fg_arr = np.array(fg_errors)

    return {
        "alpha_max_abs": round(float(np.max(alpha_arr)), 6),
        "alpha_mean_abs": round(float(np.mean(alpha_arr)), 6),
        "alpha_p95_abs": round(float(np.percentile(alpha_arr, 95)), 6),
        "fg_max_abs": round(float(np.max(fg_arr)), 6),
        "fg_mean_abs": round(float(np.mean(fg_arr)), 6),
        "fg_p95_abs": round(float(np.percentile(fg_arr, 95)), 6),
        "threshold": FIDELITY_THRESHOLD,
        "passed": len(failed_frames) == 0,
        "failed_frame_count": len(failed_frames),
        "failed_frames": failed_frames[:10],  # cap for readability
    }


def _run_benchmark(
    model: object,
    args: argparse.Namespace,
    skip: int,
    save_reference: bool,
) -> dict[str, object]:
    """Run a single benchmark pass with given skip interval."""
    out_dir = REFERENCE_DIR if save_reference else None
    processor = VideoProcessor(
        model=model,  # type: ignore[arg-type]
        img_size=args.img_size,
        output_dir=out_dir,
        async_save=True,
        skip_interval=skip,
    )

    label = f"skip={skip}" if skip > 1 else "V0 baseline"
    print(f"\nBenchmark run ({label})...")
    results: list[FrameResult] = []
    mx.reset_peak_memory()
    t_start = time.perf_counter()

    for result in processor.process_video(args.input, args.hint):
        results.append(result)
        if result.frame_idx in {0, 10, 20, 30, 36}:
            kf = "K" if result.is_keyframe else " "
            print(
                f"  [{kf}] Frame {result.frame_idx:3d}: "
                f"infer={result.infer_time_ms:5.1f}ms  "
                f"decode={result.decode_time_ms:5.1f}ms  "
                f"peak_mem={result.peak_memory_mb:.0f}MB"
            )

    total_wall_s = time.perf_counter() - t_start
    num_frames = len(results)

    infer_times = np.array([r.infer_time_ms for r in results])
    decode_times = np.array([r.decode_time_ms for r in results])
    peak_mems = np.array([r.peak_memory_mb for r in results])

    keyframe_count = sum(1 for r in results if r.is_keyframe)
    keyframe_times = np.array([r.infer_time_ms for r in results if r.is_keyframe])
    non_keyframe_times = np.array([r.infer_time_ms for r in results if not r.is_keyframe])

    metrics: dict[str, object] = {
        "experiment": f"video-v2-skip-{skip}" if skip > 1 else "video-v0-baseline",
        "skip_interval": skip,
        "img_size": args.img_size,
        "num_frames": num_frames,
        "total_wall_clock_s": round(total_wall_s, 3),
        "effective_fps": round(num_frames / total_wall_s, 2),
        "backbone_hit_rate": round(keyframe_count / num_frames, 3),
        "inference": {
            "median_ms": round(float(np.median(infer_times)), 1),
            "mean_ms": round(float(np.mean(infer_times)), 1),
            "p95_ms": round(float(np.percentile(infer_times, 95)), 1),
            "min_ms": round(float(np.min(infer_times)), 1),
            "max_ms": round(float(np.max(infer_times)), 1),
            "frame0_ms": round(results[0].infer_time_ms, 1),
            "keyframe_median_ms": round(float(np.median(keyframe_times)), 1),
            "non_keyframe_median_ms": (
                round(float(np.median(non_keyframe_times)), 1)
                if len(non_keyframe_times) > 0
                else None
            ),
            "per_frame_ms": [round(r.infer_time_ms, 1) for r in results],
        },
        "decode": {
            "median_ms": round(float(np.median(decode_times)), 1),
            "mean_ms": round(float(np.mean(decode_times)), 1),
            "max_ms": round(float(np.max(decode_times)), 1),
        },
        "memory": {
            "peak_mb": round(float(np.max(peak_mems)), 0),
            "final_mb": round(results[-1].peak_memory_mb, 0),
            "monitored": {
                str(r.frame_idx): round(r.peak_memory_mb, 0)
                for r in results
                if r.frame_idx in {0, 10, 20, 30, 36}
            },
        },
        "thermal": {
            "first_5_median_ms": round(float(np.median(infer_times[:5])), 1),
            "last_5_median_ms": round(float(np.median(infer_times[-5:])), 1),
            "drift_pct": round(
                (float(np.median(infer_times[-5:])) - float(np.median(infer_times[:5])))
                / float(np.median(infer_times[:5]))
                * 100,
                1,
            ),
        },
    }

    # Fidelity comparison against V0 reference (skip>1 only)
    if skip > 1 and REFERENCE_DIR.exists():
        print("  Checking fidelity vs V0 reference...")
        fidelity = _compute_fidelity(results, REFERENCE_DIR)
        metrics["fidelity"] = fidelity

    # --- Report ---
    print(f"\n{'=' * 60}")
    print(f"Results: {label} ({num_frames} frames @ {args.img_size}x{args.img_size})")
    print(f"{'=' * 60}")
    print(f"Total wall-clock:      {total_wall_s:.2f}s")
    print(f"Effective FPS:         {metrics['effective_fps']}")
    inf = metrics["inference"]
    print(f"Inference median:      {inf['median_ms']}ms")  # type: ignore[index]
    print(f"Inference p95:         {inf['p95_ms']}ms")  # type: ignore[index]
    print(f"Keyframe median:       {inf['keyframe_median_ms']}ms")  # type: ignore[index]
    if inf["non_keyframe_median_ms"] is not None:  # type: ignore[index]
        print(f"Non-keyframe median:   {inf['non_keyframe_median_ms']}ms")  # type: ignore[index]
    print(f"Backbone hit rate:     {metrics['backbone_hit_rate']}")
    print(f"Peak memory:           {metrics['memory']['peak_mb']}MB")  # type: ignore[index]
    print(f"Thermal drift:         {metrics['thermal']['drift_pct']}%")  # type: ignore[index]

    if "fidelity" in metrics:
        fid = metrics["fidelity"]
        status = "PASS" if fid["passed"] else "FAIL"  # type: ignore[index]
        print(f"Fidelity:              {status}")
        print(f"  alpha max_abs:       {fid['alpha_max_abs']}")  # type: ignore[index]
        print(f"  fg max_abs:          {fid['fg_max_abs']}")  # type: ignore[index]
        if not fid["passed"]:  # type: ignore[index]
            print(f"  failed frames:       {fid['failed_frame_count']}")  # type: ignore[index]

    if save_reference:
        print(f"Reference outputs:     {REFERENCE_DIR}/")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="CorridorKey MLX video benchmark")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input video")
    parser.add_argument("--hint", type=Path, default=DEFAULT_HINT, help="Hint video")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--skip", type=int, default=1, help="Backbone skip interval")
    parser.add_argument(
        "--sweep", type=int, nargs="+", metavar="N",
        help="Sweep multiple skip intervals (e.g. --sweep 2 3 5)",
    )
    parser.add_argument(
        "--no-save-reference", action="store_true", help="Skip saving V0 reference outputs"
    )
    args = parser.parse_args()

    for p, name in [(args.input, "Input"), (args.hint, "Hint"), (args.checkpoint, "Checkpoint")]:
        if not p.exists():
            print(f"{name} not found: {p}")
            raise SystemExit(1)

    # --- Load model ---
    print(f"Loading model (img_size={args.img_size})...")
    t0 = time.perf_counter()
    model = load_model(args.checkpoint, img_size=args.img_size, slim=True)
    load_time_s = time.perf_counter() - t0
    print(f"  Loaded in {load_time_s:.2f}s")

    # --- Warmup: 3 single-frame calls to trigger mx.compile traces ---
    print(f"Warmup ({WARMUP_FRAMES} frames)...")
    warmup_processor = VideoProcessor(model=model, img_size=args.img_size)
    for i, _ in enumerate(warmup_processor.process_video(args.input, args.hint)):
        if i + 1 >= WARMUP_FRAMES:
            break

    # Determine skip intervals to run
    skip_intervals = args.sweep if args.sweep else [args.skip]

    all_metrics: list[dict[str, object]] = []
    for skip in skip_intervals:
        save_ref = (skip == 1) and not args.no_save_reference
        metrics = _run_benchmark(model, args, skip, save_reference=save_ref)
        metrics["load_time_s"] = round(load_time_s, 2)
        all_metrics.append(metrics)

        # Save individual artifact
        if skip == 1:
            artifact_path = ARTIFACT_DIR / "video_baseline.json"
        else:
            artifact_path = ARTIFACT_DIR / f"video_v2_skip{skip}.json"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps(metrics, indent=2) + "\n")
        print(f"Saved: {artifact_path}")

    # Summary table for sweeps
    if len(all_metrics) > 1:
        print(f"\n{'=' * 70}")
        print("SWEEP SUMMARY")
        print(f"{'=' * 70}")
        print(f"{'Skip':>6} {'FPS':>6} {'Median':>8} {'NonKF':>8} {'HitRate':>8} {'Fidelity':>10}")
        print(f"{'':>6} {'':>6} {'(ms)':>8} {'(ms)':>8} {'':>8} {'':>10}")
        print("-" * 70)
        for m in all_metrics:
            inf = m["inference"]
            fid_status = "N/A"
            if "fidelity" in m:
                fid_status = "PASS" if m["fidelity"]["passed"] else "FAIL"  # type: ignore[index]
            nkf = inf["non_keyframe_median_ms"]  # type: ignore[index]
            nkf_str = f"{nkf}" if nkf is not None else "N/A"
            print(
                f"{m['skip_interval']:>6} "
                f"{m['effective_fps']:>6} "
                f"{inf['median_ms']:>8} "  # type: ignore[index]
                f"{nkf_str:>8} "
                f"{m['backbone_hit_rate']:>8} "
                f"{fid_status:>10}"
            )


if __name__ == "__main__":
    main()
