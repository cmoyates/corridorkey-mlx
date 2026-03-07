#!/usr/bin/env python3
"""Benchmark MLX inference on Apple Silicon.

Reports eager vs compiled latency, warmup cost, and steady-state performance
across multiple resolutions.

Usage:
    uv run python scripts/bench_mlx.py
    uv run python scripts/bench_mlx.py --checkpoint checkpoints/corridorkey_mlx.safetensors
    uv run python scripts/bench_mlx.py --resolutions 256 512 1024 --bench-runs 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from rich.console import Console
from rich.table import Table

# Ensure package is importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from corridorkey_mlx.model.corridorkey import GreenFormer
from corridorkey_mlx.utils.profiling import memory_snapshot, reset_peak, warmup_and_bench

console = Console()

DEFAULT_RESOLUTIONS = [256, 512, 1024]
DEFAULT_WARMUP_RUNS = 3
DEFAULT_BENCH_RUNS = 10


def make_dummy_input(img_size: int, batch_size: int = 1) -> mx.array:
    """Create random input tensor matching model expectations."""
    mx.random.seed(42)
    return mx.random.normal((batch_size, img_size, img_size, 4))


def load_model_weights(model: GreenFormer, checkpoint: Path | None) -> None:
    """Load real weights if checkpoint exists, otherwise use random init."""
    if checkpoint and checkpoint.exists():
        model.load_checkpoint(checkpoint)
    else:
        model.eval()
        # NOTE: mx.eval here is MLX array materialization, not Python eval()
        mx.eval(model.parameters())  # noqa: S307


def bench_resolution(
    img_size: int,
    checkpoint: Path | None,
    warmup_runs: int,
    bench_runs: int,
    batch_size: int,
) -> dict[str, object]:
    """Benchmark eager and compiled inference at a given resolution."""
    x = make_dummy_input(img_size, batch_size)
    mx.eval(x)  # noqa: S307 — materialize input

    results: dict[str, object] = {"resolution": img_size, "batch_size": batch_size}

    reset_peak()

    # --- Eager ---
    model_eager = GreenFormer(img_size=img_size)
    load_model_weights(model_eager, checkpoint)

    def run_eager() -> dict[str, mx.array]:
        return model_eager(x)

    try:
        eager_warmup, eager_steady, eager_times = warmup_and_bench(
            run_eager,
            warmup_runs=warmup_runs,
            bench_runs=bench_runs,
            label="eager",
        )
        results["eager_warmup_ms"] = round(eager_warmup.elapsed_ms, 1)
        results["eager_steady_ms"] = round(eager_steady.elapsed_ms, 1)
        results["eager_min_ms"] = round(min(eager_times), 1)
    except Exception as e:
        console.print(f"  [red]Eager failed at {img_size}: {e}[/red]")
        results["eager_warmup_ms"] = "FAIL"
        results["eager_steady_ms"] = "FAIL"
        results["eager_min_ms"] = "FAIL"

    # --- Compiled (fixed-shape) ---
    model_compiled = GreenFormer(img_size=img_size)
    load_model_weights(model_compiled, checkpoint)
    compiled_call = mx.compile(model_compiled.__call__)

    def run_compiled() -> dict[str, mx.array]:
        return compiled_call(x)

    try:
        comp_warmup, comp_steady, comp_times = warmup_and_bench(
            run_compiled,
            warmup_runs=warmup_runs,
            bench_runs=bench_runs,
            label="compiled",
        )
        results["compiled_warmup_ms"] = round(comp_warmup.elapsed_ms, 1)
        results["compiled_steady_ms"] = round(comp_steady.elapsed_ms, 1)
        results["compiled_min_ms"] = round(min(comp_times), 1)
    except Exception as e:
        console.print(f"  [red]Compiled failed at {img_size}: {e}[/red]")
        results["compiled_warmup_ms"] = "FAIL"
        results["compiled_steady_ms"] = "FAIL"
        results["compiled_min_ms"] = "FAIL"

    # --- Memory snapshot ---
    mem = memory_snapshot()
    results["peak_mb"] = round(mem.peak_mb, 1)
    results["active_mb"] = round(mem.active_mb, 1)

    # --- Parity check ---
    try:
        eager_out = run_eager()
        compiled_out = run_compiled()
        mx.eval(eager_out, compiled_out)  # noqa: S307 — materialize for comparison
        max_diff = max(
            float(mx.max(mx.abs(eager_out[k] - compiled_out[k])))
            for k in ("alpha_final", "fg_final")
        )
        results["parity_max_diff"] = f"{max_diff:.2e}"
    except Exception:
        results["parity_max_diff"] = "N/A"

    return results


def bench_engine_config(
    checkpoint: Path,
    img_size: int,
    fp16: bool,
    backbone_size: int | None,
    tile_size: int | None,
    tile_overlap: int,
    bench_runs: int,
) -> dict[str, object]:
    """Benchmark a specific engine configuration via process_frame()."""
    from corridorkey_mlx import CorridorKeyMLXEngine

    label = f"{img_size} fp16={fp16} bb={backbone_size or img_size} tile={tile_size or 'none'}"

    reset_peak()
    rng = np.random.default_rng(42)
    image = rng.integers(0, 256, (img_size, img_size, 3), dtype=np.uint8)
    mask = rng.integers(0, 256, (img_size, img_size), dtype=np.uint8)

    try:
        engine = CorridorKeyMLXEngine(
            checkpoint_path=checkpoint,
            img_size=img_size,
            compile=True,
            fp16=fp16,
            backbone_size=backbone_size,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
        )

        # Warmup
        engine.process_frame(image, mask)
        reset_peak()

        # Bench
        times: list[float] = []
        for _ in range(bench_runs):
            import time

            start = time.perf_counter()
            engine.process_frame(image, mask)
            times.append((time.perf_counter() - start) * 1000.0)

        mem = memory_snapshot()
        times.sort()
        median_ms = times[len(times) // 2]

        return {
            "config": label,
            "median_ms": round(median_ms, 1),
            "min_ms": round(min(times), 1),
            "peak_mb": round(mem.peak_mb, 1),
            "active_mb": round(mem.active_mb, 1),
        }
    except Exception as e:
        return {
            "config": label,
            "median_ms": "FAIL",
            "min_ms": "FAIL",
            "peak_mb": "FAIL",
            "active_mb": "FAIL",
            "error": str(e),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MLX CorridorKey inference")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/corridorkey_mlx.safetensors"),
        help="MLX safetensors checkpoint (uses random weights if missing)",
    )
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=DEFAULT_RESOLUTIONS,
        help="Resolutions to benchmark",
    )
    parser.add_argument("--warmup-runs", type=int, default=DEFAULT_WARMUP_RUNS)
    parser.add_argument("--bench-runs", type=int, default=DEFAULT_BENCH_RUNS)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="Run engine config matrix at 2048 (fp16/fp32 x backbone x tiled)",
    )
    args = parser.parse_args()

    ckpt = args.checkpoint if args.checkpoint.exists() else None
    weights_label = str(args.checkpoint) if ckpt else "random (no checkpoint)"
    console.print("[bold]CorridorKey MLX Benchmark[/bold]")
    console.print(f"  Weights: {weights_label}")

    if args.matrix:
        if not ckpt:
            console.print("[red]--matrix requires a real checkpoint[/red]")
            sys.exit(1)

        console.print("  Mode: engine configuration matrix")
        console.print()

        configs = [
            # Regression checks at smaller resolutions
            {"img_size": 512, "fp16": False, "backbone_size": None, "tile_size": None},
            {"img_size": 512, "fp16": True, "backbone_size": None, "tile_size": None},
            {"img_size": 1024, "fp16": False, "backbone_size": None, "tile_size": None},
            {"img_size": 1024, "fp16": True, "backbone_size": None, "tile_size": None},
            # 2048 matrix: {fp16, fp32} x {backbone: None, 1024} x {tiled, non-tiled}
            {"img_size": 2048, "fp16": False, "backbone_size": None, "tile_size": None},
            {"img_size": 2048, "fp16": True, "backbone_size": None, "tile_size": None},
            {"img_size": 2048, "fp16": False, "backbone_size": 1024, "tile_size": None},
            {"img_size": 2048, "fp16": True, "backbone_size": 1024, "tile_size": None},
            {"img_size": 2048, "fp16": False, "backbone_size": None, "tile_size": 512},
            {"img_size": 2048, "fp16": True, "backbone_size": None, "tile_size": 512},
        ]

        matrix_results = []
        for cfg in configs:
            console.print(f"  Benchmarking {cfg}...")
            result = bench_engine_config(
                checkpoint=ckpt,
                img_size=cfg["img_size"],
                fp16=cfg["fp16"],
                backbone_size=cfg["backbone_size"],
                tile_size=cfg["tile_size"],
                tile_overlap=96,
                bench_runs=args.bench_runs,
            )
            matrix_results.append(result)

        table = Table(title="Engine Configuration Matrix")
        table.add_column("Config", justify="left")
        table.add_column("Median ms", justify="right")
        table.add_column("Min ms", justify="right")
        table.add_column("Peak MB", justify="right")
        table.add_column("Active MB", justify="right")

        for r in matrix_results:
            table.add_row(
                str(r["config"]),
                str(r.get("median_ms", "")),
                str(r.get("min_ms", "")),
                str(r.get("peak_mb", "")),
                str(r.get("active_mb", "")),
            )

        console.print(table)
        return

    console.print(f"  Resolutions: {args.resolutions}")
    console.print(
        f"  Warmup: {args.warmup_runs}, Bench: {args.bench_runs}, Batch: {args.batch_size}"
    )
    console.print()

    all_results = []
    for res in args.resolutions:
        console.print(f"  Benchmarking {res}x{res}...")
        result = bench_resolution(res, ckpt, args.warmup_runs, args.bench_runs, args.batch_size)
        all_results.append(result)

    # Print results table
    table = Table(title="Benchmark Results")
    table.add_column("Resolution", justify="right")
    table.add_column("Eager Warmup", justify="right")
    table.add_column("Eager Steady", justify="right")
    table.add_column("Compiled Warmup", justify="right")
    table.add_column("Compiled Steady", justify="right")
    table.add_column("Speedup", justify="right")
    table.add_column("Peak MB", justify="right")
    table.add_column("Active MB", justify="right")
    table.add_column("Parity", justify="right")

    for r in all_results:
        speedup = ""
        eager_s = r.get("eager_steady_ms")
        comp_s = r.get("compiled_steady_ms")
        if isinstance(eager_s, (int, float)) and isinstance(comp_s, (int, float)) and comp_s > 0:
            speedup = f"{eager_s / comp_s:.2f}x"

        table.add_row(
            f"{r['resolution']}x{r['resolution']}",
            str(r.get("eager_warmup_ms", "")),
            str(r.get("eager_steady_ms", "")),
            str(r.get("compiled_warmup_ms", "")),
            str(r.get("compiled_steady_ms", "")),
            speedup,
            str(r.get("peak_mb", "")),
            str(r.get("active_mb", "")),
            str(r.get("parity_max_diff", "")),
        )

    console.print(table)


if __name__ == "__main__":
    main()
