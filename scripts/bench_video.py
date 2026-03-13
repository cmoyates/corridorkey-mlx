#!/usr/bin/env python3
"""Video benchmark harness for CorridorKey MLX temporal experiments.

Supports V0 baseline, V1 EMA blending, V2 async pipeline, and backbone skip.
Computes Tier 1 (max_abs) + Tier 2 (PSNR/SSIM/dtSSD) fidelity metrics.

Usage:
    uv run python scripts/bench_video.py                              # V0 baseline
    uv run python scripts/bench_video.py --ema-alpha 0.7              # V1 EMA
    uv run python scripts/bench_video.py --ema-sweep 0.6 0.7 0.8     # V1 sweep
    uv run python scripts/bench_video.py --async-decode               # V2 async
    uv run python scripts/bench_video.py --async-decode --ema-alpha 0.7  # V1+V2
    uv run python scripts/bench_video.py --skip 3                     # backbone skip
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

# Tier 2 thresholds (from benchmark_spec.md)
TIER2_ALPHA_PSNR_MIN = 35.0  # dB
TIER2_FG_PSNR_MIN = 33.0  # dB
TIER2_SSIM_MIN = 0.97
TIER2_DTSSD_MAX = 1.5


def _uniform_filter_2d(img: np.ndarray, size: int) -> np.ndarray:
    """Fast 2D uniform (box) filter via cumulative sums. No scipy needed."""
    pad = size // 2
    padded = np.pad(img, pad, mode="reflect")
    # Cumsum rows
    cs = np.cumsum(padded, axis=0)
    cs = cs[size:] - cs[:-size]
    # Cumsum cols
    cs = np.cumsum(cs, axis=1)
    cs = cs[:, size:] - cs[:, :-size]
    return cs / (size * size)


def _compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """PSNR between two float32 images in [0,1]. Returns dB (inf if identical)."""
    mse = float(np.mean((img1 - img2) ** 2))
    if mse < 1e-12:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


def _compute_ssim(img1: np.ndarray, img2: np.ndarray, win_size: int = 11) -> float:
    """Windowed SSIM (Wang et al. 2004) for 2D float32 images in [0,1].

    For multi-channel images, computes per-channel then averages.
    """
    c1 = 0.01**2
    c2 = 0.03**2

    if img1.ndim == 3:
        # Per-channel SSIM, averaged
        return float(
            np.mean(
                [
                    _compute_ssim(img1[:, :, ch], img2[:, :, ch], win_size)
                    for ch in range(img1.shape[2])
                ]
            )
        )

    mu1 = _uniform_filter_2d(img1, win_size)
    mu2 = _uniform_filter_2d(img2, win_size)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12 = mu1 * mu2

    sigma1_sq = _uniform_filter_2d(img1 * img1, win_size) - mu1_sq
    sigma2_sq = _uniform_filter_2d(img2 * img2, win_size) - mu2_sq
    sigma12 = _uniform_filter_2d(img1 * img2, win_size) - mu12

    num = (2.0 * mu12 + c1) * (2.0 * sigma12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = num / den
    return float(np.mean(ssim_map))


def _compute_dtssd(
    candidate_frames: list[np.ndarray],
    reference_frames: list[np.ndarray],
) -> float:
    """Temporal coherence metric (dtSSD).

    dtSSD = mean over consecutive pairs of:
        mean(||(c_t - c_{t-1}) - (r_t - r_{t-1})||)

    Lower = more temporally consistent with reference.
    Operates on alpha channel (2D float32 [0,1]).
    """
    if len(candidate_frames) < 2:
        return 0.0
    dtssd_values: list[float] = []
    for i in range(1, len(candidate_frames)):
        cand_diff = candidate_frames[i] - candidate_frames[i - 1]
        ref_diff = reference_frames[i] - reference_frames[i - 1]
        # Per-pixel absolute difference of temporal derivatives, spatially averaged
        frame_dtssd = float(np.mean(np.abs(cand_diff - ref_diff)) * 255.0)
        dtssd_values.append(frame_dtssd)
    return float(np.mean(dtssd_values))


def _load_reference_frame(ref_dir: Path, frame_idx: int, key: str) -> np.ndarray:
    """Load a V0 reference PNG as float32 [0,1]."""
    path = ref_dir / f"{key}_{frame_idx:04d}.png"
    if not path.exists():
        msg = f"Reference frame not found: {path}"
        raise FileNotFoundError(msg)
    img = np.asarray(Image.open(path), dtype=np.float32) / 255.0
    return img


def _compute_fidelity(results: list[FrameResult], ref_dir: Path) -> dict[str, object]:
    """Compare per-frame outputs against V0 reference.

    Returns Tier 1 (max_abs) and Tier 2 (PSNR/SSIM/dtSSD) metrics.
    """
    alpha_errors: list[float] = []
    fg_errors: list[float] = []
    alpha_psnrs: list[float] = []
    fg_psnrs: list[float] = []
    alpha_ssims: list[float] = []
    failed_frames: list[dict[str, object]] = []

    # Collect frames for dtSSD (alpha only — most sensitive channel)
    candidate_alpha_frames: list[np.ndarray] = []
    reference_alpha_frames: list[np.ndarray] = []

    for r in results:
        # Alpha: (H, W) uint8 -> float32
        alpha_f32 = r.alpha.astype(np.float32) / 255.0
        ref_alpha = _load_reference_frame(ref_dir, r.frame_idx, "alpha")

        # Tier 1: max abs error
        alpha_err = float(np.max(np.abs(alpha_f32 - ref_alpha)))
        alpha_errors.append(alpha_err)

        # Tier 2: PSNR, SSIM
        alpha_psnrs.append(_compute_psnr(alpha_f32, ref_alpha))
        alpha_ssims.append(_compute_ssim(alpha_f32, ref_alpha))

        # Collect for dtSSD
        candidate_alpha_frames.append(alpha_f32)
        reference_alpha_frames.append(ref_alpha)

        # FG: (H, W, 3) uint8 -> float32
        fg_f32 = r.fg.astype(np.float32) / 255.0
        ref_fg = _load_reference_frame(ref_dir, r.frame_idx, "fg")

        fg_err = float(np.max(np.abs(fg_f32 - ref_fg)))
        fg_errors.append(fg_err)
        fg_psnrs.append(_compute_psnr(fg_f32, ref_fg))

        max_err = max(alpha_err, fg_err)
        if max_err > FIDELITY_THRESHOLD:
            failed_frames.append(
                {
                    "frame": r.frame_idx,
                    "alpha_err": round(alpha_err, 6),
                    "fg_err": round(fg_err, 6),
                    "is_keyframe": r.is_keyframe,
                }
            )

    alpha_arr = np.array(alpha_errors)
    fg_arr = np.array(fg_errors)

    # Tier 2: dtSSD (temporal coherence on alpha)
    dtssd = _compute_dtssd(candidate_alpha_frames, reference_alpha_frames)

    # Filter out inf PSNR values for stats (identical frames → inf)
    finite_alpha_psnrs = [p for p in alpha_psnrs if np.isfinite(p)]
    finite_fg_psnrs = [p for p in fg_psnrs if np.isfinite(p)]
    alpha_psnr_min = float(min(alpha_psnrs)) if alpha_psnrs else 0.0
    fg_psnr_min = float(min(fg_psnrs)) if fg_psnrs else 0.0
    alpha_ssim_min = float(min(alpha_ssims)) if alpha_ssims else 0.0

    # Tier 2 pass/fail
    tier2_passed = (
        alpha_psnr_min >= TIER2_ALPHA_PSNR_MIN
        and fg_psnr_min >= TIER2_FG_PSNR_MIN
        and alpha_ssim_min >= TIER2_SSIM_MIN
        and dtssd <= TIER2_DTSSD_MAX
    )

    return {
        # Tier 1 (precision)
        "alpha_max_abs": round(float(np.max(alpha_arr)), 6),
        "alpha_mean_abs": round(float(np.mean(alpha_arr)), 6),
        "alpha_p95_abs": round(float(np.percentile(alpha_arr, 95)), 6),
        "fg_max_abs": round(float(np.max(fg_arr)), 6),
        "fg_mean_abs": round(float(np.mean(fg_arr)), 6),
        "fg_p95_abs": round(float(np.percentile(fg_arr, 95)), 6),
        "threshold": FIDELITY_THRESHOLD,
        "tier1_passed": len(failed_frames) == 0,
        "failed_frame_count": len(failed_frames),
        "failed_frames": failed_frames[:10],
        # Tier 2 (perceptual/temporal)
        "alpha_psnr_min_db": round(alpha_psnr_min, 2),
        "alpha_psnr_mean_db": round(
            float(np.mean(finite_alpha_psnrs)) if finite_alpha_psnrs else float("inf"), 2
        ),
        "fg_psnr_min_db": round(fg_psnr_min, 2),
        "fg_psnr_mean_db": round(
            float(np.mean(finite_fg_psnrs)) if finite_fg_psnrs else float("inf"), 2
        ),
        "alpha_ssim_min": round(alpha_ssim_min, 4),
        "alpha_ssim_mean": round(float(np.mean(alpha_ssims)), 4),
        "dtssd": round(dtssd, 4),
        "tier2_passed": tier2_passed,
        # Overall
        "passed": len(failed_frames) == 0 and tier2_passed,
    }


def _run_benchmark(
    model: object,
    args: argparse.Namespace,
    skip: int,
    save_reference: bool,
    ema_alpha: float | None = None,
    async_decode: bool = False,
    tile_skip_threshold: float | None = None,
) -> dict[str, object]:
    """Run a single benchmark pass with given skip interval and optional EMA."""
    # Set tile skip confidence on model (runtime toggle, no reload needed)
    model._refiner_skip_confidence = tile_skip_threshold  # type: ignore[attr-defined]

    out_dir = REFERENCE_DIR if save_reference else None
    processor = VideoProcessor(
        model=model,  # type: ignore[arg-type]
        img_size=args.img_size,
        output_dir=out_dir,
        async_save=True,
        skip_interval=skip,
        ema_alpha=ema_alpha,
        async_decode=async_decode,
    )

    parts = []
    if skip > 1:
        parts.append(f"skip={skip}")
    if ema_alpha is not None:
        parts.append(f"ema={ema_alpha}")
    if async_decode:
        parts.append("async")
    if tile_skip_threshold is not None:
        parts.append(f"tileskip={tile_skip_threshold}")
    label = ", ".join(parts) if parts else "V0 baseline"
    print(f"\nBenchmark run ({label})...")
    results: list[FrameResult] = []
    mx.reset_peak_memory()
    t_start = time.perf_counter()

    for result in processor.process_video(args.input, args.hint):
        results.append(result)
        if result.frame_idx in {0, 10, 20, 30, 36}:
            kf = "K" if result.is_keyframe else " "
            skip_str = ""
            if result.tiles_total > 0:
                skip_str = f"  tiles={result.tiles_skipped}/{result.tiles_total}"
            print(
                f"  [{kf}] Frame {result.frame_idx:3d}: "
                f"infer={result.infer_time_ms:5.1f}ms  "
                f"decode={result.decode_time_ms:5.1f}ms  "
                f"peak_mem={result.peak_memory_mb:.0f}MB{skip_str}"
            )

    total_wall_s = time.perf_counter() - t_start
    num_frames = len(results)

    infer_times = np.array([r.infer_time_ms for r in results])
    decode_times = np.array([r.decode_time_ms for r in results])
    peak_mems = np.array([r.peak_memory_mb for r in results])

    keyframe_count = sum(1 for r in results if r.is_keyframe)
    keyframe_times = np.array([r.infer_time_ms for r in results if r.is_keyframe])
    non_keyframe_times = np.array([r.infer_time_ms for r in results if not r.is_keyframe])

    # Tile skip stats
    total_tiles_skipped = sum(r.tiles_skipped for r in results)
    total_tiles = sum(r.tiles_total for r in results)
    tile_skip_rate = total_tiles_skipped / total_tiles if total_tiles > 0 else 0.0

    # Build experiment name
    exp_parts = []
    if skip > 1:
        exp_parts.append(f"skip-{skip}")
    if ema_alpha is not None:
        exp_parts.append(f"ema-{ema_alpha}")
    if async_decode:
        exp_parts.append("async")
    if tile_skip_threshold is not None:
        exp_parts.append(f"tileskip-{tile_skip_threshold}")
    experiment_name = f"video-{'_'.join(exp_parts)}" if exp_parts else "video-v0-baseline"

    metrics: dict[str, object] = {
        "experiment": experiment_name,
        "skip_interval": skip,
        "ema_alpha": ema_alpha,
        "async_decode": async_decode,
        "tile_skip_threshold": tile_skip_threshold,
        "tile_skip_rate": round(tile_skip_rate, 3),
        "tiles_skipped": total_tiles_skipped,
        "tiles_total": total_tiles,
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

    # Fidelity comparison against V0 reference (when reference exists)
    if REFERENCE_DIR.exists():
        print("  Computing fidelity vs V0 reference...")
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
    if total_tiles > 0:
        print(f"Tile skip rate:        {total_tiles_skipped}/{total_tiles} ({tile_skip_rate:.1%})")
    print(f"Thermal drift:         {metrics['thermal']['drift_pct']}%")  # type: ignore[index]

    if "fidelity" in metrics:
        fid = metrics["fidelity"]
        t1 = "PASS" if fid["tier1_passed"] else "FAIL"  # type: ignore[index]
        t2 = "PASS" if fid["tier2_passed"] else "FAIL"  # type: ignore[index]
        overall = "PASS" if fid["passed"] else "FAIL"  # type: ignore[index]
        print(f"Fidelity:              {overall}")
        print(f"  Tier 1 (precision):  {t1}")
        print(f"    alpha max_abs:     {fid['alpha_max_abs']}")  # type: ignore[index]
        print(f"    fg max_abs:        {fid['fg_max_abs']}")  # type: ignore[index]
        print(f"  Tier 2 (perceptual): {t2}")
        print(f"    alpha PSNR min:    {fid['alpha_psnr_min_db']}dB (>={TIER2_ALPHA_PSNR_MIN})")  # type: ignore[index]
        print(f"    fg PSNR min:       {fid['fg_psnr_min_db']}dB (>={TIER2_FG_PSNR_MIN})")  # type: ignore[index]
        print(f"    alpha SSIM min:    {fid['alpha_ssim_min']} (>={TIER2_SSIM_MIN})")  # type: ignore[index]
        print(f"    dtSSD:             {fid['dtssd']} (<={TIER2_DTSSD_MAX})")  # type: ignore[index]
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
        "--sweep",
        type=int,
        nargs="+",
        metavar="N",
        help="Sweep multiple skip intervals (e.g. --sweep 2 3 5)",
    )
    parser.add_argument(
        "--no-save-reference", action="store_true", help="Skip saving V0 reference outputs"
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=None,
        help="EMA blending coefficient (0.6-0.8 typical, None=disabled)",
    )
    parser.add_argument(
        "--ema-sweep",
        type=float,
        nargs="+",
        metavar="A",
        help="Sweep EMA alpha values (e.g. --ema-sweep 0.6 0.7 0.8)",
    )
    parser.add_argument(
        "--async-decode",
        action="store_true",
        help="V2: overlap frame decode (CPU) with GPU inference",
    )
    parser.add_argument(
        "--tile-skip-threshold",
        type=float,
        default=None,
        help="V3: skip refiner tiles w/ confident alpha (0.02 = skip <0.02 or >0.98)",
    )
    parser.add_argument(
        "--tile-skip-sweep",
        type=float,
        nargs="+",
        metavar="T",
        help="Sweep tile skip thresholds (e.g. --tile-skip-sweep 0.01 0.02 0.05)",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=1024,
        help="Refiner tile size in pixels (default 1024)",
    )
    parser.add_argument(
        "--frozen-gn",
        action="store_true",
        help="Precompute full-image GroupNorm stats for tiled refiner",
    )
    args = parser.parse_args()

    for p, name in [(args.input, "Input"), (args.hint, "Hint"), (args.checkpoint, "Checkpoint")]:
        if not p.exists():
            print(f"{name} not found: {p}")
            raise SystemExit(1)

    # --- Load model ---
    print(f"Loading model (img_size={args.img_size}, tile_size={args.tile_size})...")
    t0 = time.perf_counter()
    model = load_model(
        args.checkpoint,
        img_size=args.img_size,
        slim=True,
        refiner_tile_size=args.tile_size,
        refiner_frozen_gn=args.frozen_gn,
    )
    load_time_s = time.perf_counter() - t0
    print(f"  Loaded in {load_time_s:.2f}s")

    # --- Warmup: 3 single-frame calls to trigger mx.compile traces ---
    print(f"Warmup ({WARMUP_FRAMES} frames)...")
    warmup_processor = VideoProcessor(model=model, img_size=args.img_size)
    for i, _ in enumerate(warmup_processor.process_video(args.input, args.hint)):
        if i + 1 >= WARMUP_FRAMES:
            break

    # Determine skip intervals, EMA alphas, and tile skip thresholds to run
    skip_intervals = args.sweep if args.sweep else [args.skip]
    ema_alphas: list[float | None] = [*args.ema_sweep] if args.ema_sweep else [args.ema_alpha]
    tile_skip_thresholds: list[float | None] = (
        [*args.tile_skip_sweep] if args.tile_skip_sweep else [args.tile_skip_threshold]
    )

    all_metrics: list[dict[str, object]] = []
    for skip in skip_intervals:
        for ema_alpha in ema_alphas:
            for tile_skip_thresh in tile_skip_thresholds:
                is_vanilla = (
                    skip == 1
                    and ema_alpha is None
                    and not args.async_decode
                    and tile_skip_thresh is None
                )
                save_ref = is_vanilla and not args.no_save_reference
                metrics = _run_benchmark(
                    model,
                    args,
                    skip,
                    save_reference=save_ref,
                    ema_alpha=ema_alpha,
                    async_decode=args.async_decode,
                    tile_skip_threshold=tile_skip_thresh,
                )
                metrics["load_time_s"] = round(load_time_s, 2)
                all_metrics.append(metrics)

                # Save individual artifact
                artifact_name = metrics["experiment"]
                artifact_path = ARTIFACT_DIR / f"{artifact_name}.json"
                artifact_path.parent.mkdir(parents=True, exist_ok=True)
                artifact_path.write_text(json.dumps(metrics, indent=2) + "\n")
                print(f"Saved: {artifact_path}")

    # Summary table for sweeps
    if len(all_metrics) > 1:
        print(f"\n{'=' * 90}")
        print("SWEEP SUMMARY")
        print(f"{'=' * 90}")
        print(
            f"{'Skip':>6} {'TSkip':>7} {'FPS':>6} {'Median':>8} "
            f"{'αPSNR':>7} {'SSIM':>6} {'dtSSD':>6} {'TileRate':>9} {'Gate':>6}"
        )
        print(f"{'':>6} {'':>7} {'':>6} {'(ms)':>8} {'(dB)':>7} {'':>6} {'':>6} {'':>9} {'':>6}")
        print("-" * 90)
        for m in all_metrics:
            inf = m["inference"]
            tskip = m.get("tile_skip_threshold")
            tskip_str = f"{tskip}" if tskip is not None else "off"
            tile_rate = m.get("tile_skip_rate", 0)
            tile_rate_str = f"{tile_rate:.1%}" if tile_rate > 0 else "0%"
            if "fidelity" in m:
                fid = m["fidelity"]
                gate = "PASS" if fid["passed"] else "FAIL"  # type: ignore[index]
                psnr_str = f"{fid['alpha_psnr_min_db']}"  # type: ignore[index]
                ssim_str = f"{fid['alpha_ssim_min']}"  # type: ignore[index]
                dtssd_str = f"{fid['dtssd']}"  # type: ignore[index]
            else:
                gate, psnr_str, ssim_str, dtssd_str = "N/A", "N/A", "N/A", "N/A"
            print(
                f"{m['skip_interval']:>6} "
                f"{tskip_str:>7} "
                f"{m['effective_fps']:>6} "
                f"{inf['median_ms']:>8} "  # type: ignore[index]
                f"{psnr_str:>7} "
                f"{ssim_str:>6} "
                f"{dtssd_str:>6} "
                f"{tile_rate_str:>9} "
                f"{gate:>6}"
            )


if __name__ == "__main__":
    main()
