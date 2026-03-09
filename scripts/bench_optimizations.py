#!/usr/bin/env python3
"""Benchmark matrix for wave 2 optimizations.

Runs an exhaustive matrix of optimization toggles and reports latency + peak
memory for each configuration. Helps isolate the impact of each optimization.

Usage:
    uv run python scripts/bench_optimizations.py
    uv run python scripts/bench_optimizations.py --resolution 512 --bench-runs 5
    uv run python scripts/bench_optimizations.py --checkpoint ckpt.safetensors

All mx.eval() calls are MLX array materialization, NOT Python eval().
"""

from __future__ import annotations

import argparse
import gc
import itertools
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
from rich.console import Console
from rich.table import Table

# Ensure package is importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from corridorkey_mlx.io.image import preprocess as preprocess_numpy
from corridorkey_mlx.io.preprocess_mlx import preprocess_mlx
from corridorkey_mlx.model.corridorkey import GreenFormer

console = Console()

# ---------------------------------------------------------------------------
# Optimization flags
# ---------------------------------------------------------------------------

TOGGLE_NAMES = ["slim", "stage_gc", "sdpa", "bf16", "fused_decode", "gpu_preprocess"]


@dataclass
class OptConfig:
    """Single optimization configuration to benchmark."""

    slim: bool = True
    stage_gc: bool = True
    sdpa: bool = True
    bf16: bool = True
    fused_decode: bool = True
    gpu_preprocess: bool = True

    @property
    def label(self) -> str:
        flags = []
        for name in TOGGLE_NAMES:
            val = getattr(self, name)
            flags.append(f"{name}={'on' if val else 'off'}")
        return " | ".join(flags)

    @property
    def short_label(self) -> str:
        parts = []
        for name in TOGGLE_NAMES:
            if getattr(self, name):
                parts.append(name)
        return "+".join(parts) if parts else "baseline"


@dataclass
class BenchResult:
    """Result from a single benchmark configuration."""

    config: OptConfig
    resolution: int
    warmup_ms: float = 0.0
    median_ms: float = 0.0
    min_ms: float = 0.0
    peak_memory_mb: float = 0.0
    all_times: list[float] = field(default_factory=list)
    error: str | None = None


# ---------------------------------------------------------------------------
# Benchmark logic
# ---------------------------------------------------------------------------


def build_model(
    config: OptConfig,
    img_size: int,
    checkpoint: Path | None,
) -> GreenFormer:
    """Build model with the given optimization config."""
    dtype = mx.bfloat16 if config.bf16 else mx.float32
    model = GreenFormer(
        img_size=img_size,
        dtype=dtype,
        fused_decode=config.fused_decode,
        slim=config.slim,
        use_sdpa=config.sdpa,
        stage_gc=config.stage_gc,
    )
    if checkpoint and checkpoint.exists():
        model.load_checkpoint(checkpoint)
    else:
        model.eval()
        mx.eval(model.parameters())  # noqa: S307
    return model


def make_input(img_size: int, gpu_preprocess: bool) -> mx.array:
    """Create input tensor, optionally via GPU preprocessing path."""
    import numpy as np

    np.random.seed(42)
    rgb_f32 = np.random.rand(img_size, img_size, 3).astype(np.float32)
    mask_f32 = np.random.rand(img_size, img_size, 1).astype(np.float32)

    if gpu_preprocess:
        return preprocess_mlx(rgb_f32, mask_f32)
    return preprocess_numpy(rgb_f32, mask_f32)


def time_inference(
    model: GreenFormer,
    x: mx.array,
    warmup_runs: int,
    bench_runs: int,
) -> tuple[float, float, float, list[float]]:
    """Run warmup + benchmark, return (warmup_ms, median_ms, min_ms, all_times)."""

    def run() -> None:
        out = model(x)
        mx.eval(out)  # noqa: S307

    # Warmup
    start = time.perf_counter()
    run()
    warmup_ms = (time.perf_counter() - start) * 1000.0

    for _ in range(warmup_runs - 1):
        run()

    # Benchmark
    times: list[float] = []
    for _ in range(bench_runs):
        start = time.perf_counter()
        run()
        times.append((time.perf_counter() - start) * 1000.0)

    times.sort()
    median_ms = times[len(times) // 2]
    min_ms = times[0]
    return warmup_ms, median_ms, min_ms, times


def measure_peak_memory(model: GreenFormer, x: mx.array) -> float:
    """Measure peak Metal memory for a single forward pass (MB)."""
    gc.collect()
    mx.clear_cache()

    # Use non-deprecated API if available, fall back to mx.metal.*
    reset_fn = getattr(mx, "reset_peak_memory", None) or getattr(
        mx.metal, "reset_peak_memory", None
    )
    get_fn = getattr(mx, "get_peak_memory", None) or getattr(mx.metal, "get_peak_memory", None)
    if not reset_fn or not get_fn:
        return 0.0

    reset_fn()

    out = model(x)
    mx.eval(out)  # noqa: S307

    peak_bytes = get_fn()

    del out
    gc.collect()
    mx.clear_cache()

    return peak_bytes / (1024 * 1024)


def bench_config(
    config: OptConfig,
    resolution: int,
    checkpoint: Path | None,
    warmup_runs: int,
    bench_runs: int,
) -> BenchResult:
    """Benchmark a single optimization configuration."""
    result = BenchResult(config=config, resolution=resolution)

    try:
        model = build_model(config, resolution, checkpoint)
        x = make_input(resolution, config.gpu_preprocess)
        mx.eval(x)  # noqa: S307

        warmup_ms, median_ms, min_ms, all_times = time_inference(model, x, warmup_runs, bench_runs)
        result.warmup_ms = warmup_ms
        result.median_ms = median_ms
        result.min_ms = min_ms
        result.all_times = all_times

        # Measure peak memory with a fresh model to avoid cached state
        model_fresh = build_model(config, resolution, checkpoint)
        result.peak_memory_mb = measure_peak_memory(model_fresh, x)

        # Cleanup
        del model, model_fresh, x
        gc.collect()
        mx.clear_cache()

    except Exception as e:
        result.error = str(e)

    return result


# ---------------------------------------------------------------------------
# Matrix generation
# ---------------------------------------------------------------------------


def generate_matrix(sweep: str) -> list[OptConfig]:
    """Generate benchmark configurations.

    Args:
        sweep: "full" for exhaustive 2^6 matrix, "ablation" for toggle-one-off,
               "key" for important combinations only.
    """
    all_on = OptConfig()

    if sweep == "ablation":
        # All-on baseline + toggle each flag off one at a time
        configs = [all_on]
        for name in TOGGLE_NAMES:
            off = OptConfig(**{**vars(all_on), name: False})
            configs.append(off)
        # Also add all-off baseline
        configs.append(
            OptConfig(
                slim=False,
                stage_gc=False,
                sdpa=False,
                bf16=False,
                fused_decode=False,
                gpu_preprocess=False,
            )
        )
        return configs

    if sweep == "full":
        # Exhaustive 2^6 = 64 configurations
        configs = []
        for combo in itertools.product([True, False], repeat=len(TOGGLE_NAMES)):
            kwargs = dict(zip(TOGGLE_NAMES, combo, strict=True))
            configs.append(OptConfig(**kwargs))
        return configs

    # "key" — important configurations
    return [
        # All off (wave 1 baseline)
        OptConfig(
            slim=False,
            stage_gc=False,
            sdpa=False,
            bf16=False,
            fused_decode=False,
            gpu_preprocess=False,
        ),
        # Wave 1 only (bf16 + fused_decode)
        OptConfig(
            slim=False,
            stage_gc=False,
            sdpa=False,
            bf16=True,
            fused_decode=True,
            gpu_preprocess=False,
        ),
        # Wave 2 only (slim + stage_gc + sdpa + gpu_preprocess)
        OptConfig(
            slim=True,
            stage_gc=True,
            sdpa=True,
            bf16=False,
            fused_decode=False,
            gpu_preprocess=True,
        ),
        # All on
        all_on,
        # SDPA only
        OptConfig(
            slim=False,
            stage_gc=False,
            sdpa=True,
            bf16=False,
            fused_decode=False,
            gpu_preprocess=False,
        ),
        # Stage GC only
        OptConfig(
            slim=False,
            stage_gc=True,
            sdpa=False,
            bf16=False,
            fused_decode=False,
            gpu_preprocess=False,
        ),
    ]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def print_results(results: list[BenchResult]) -> None:
    """Print results as a rich table."""
    table = Table(title="Optimization Benchmark Matrix")

    table.add_column("Config", max_width=50)
    table.add_column("Res", justify="right")
    table.add_column("Warmup (ms)", justify="right")
    table.add_column("Median (ms)", justify="right")
    table.add_column("Min (ms)", justify="right")
    table.add_column("Peak Mem (MB)", justify="right")
    table.add_column("Status", justify="center")

    # Sort by resolution then median time
    results.sort(key=lambda r: (r.resolution, r.median_ms))

    for r in results:
        if r.error:
            table.add_row(
                r.config.short_label,
                str(r.resolution),
                "-",
                "-",
                "-",
                "-",
                f"[red]FAIL: {r.error[:30]}[/red]",
            )
        else:
            table.add_row(
                r.config.short_label,
                str(r.resolution),
                f"{r.warmup_ms:.1f}",
                f"{r.median_ms:.1f}",
                f"{r.min_ms:.1f}",
                f"{r.peak_memory_mb:.0f}" if r.peak_memory_mb > 0 else "N/A",
                "[green]OK[/green]",
            )

    console.print(table)

    # Print speedup summary vs baseline
    baselines = {r.resolution: r for r in results if r.config.short_label == "baseline"}
    if baselines:
        console.print("\n[bold]Speedup vs baseline (all-off):[/bold]")
        for r in results:
            if r.error or r.config.short_label == "baseline":
                continue
            baseline = baselines.get(r.resolution)
            if baseline and baseline.median_ms > 0:
                speedup = baseline.median_ms / r.median_ms
                mem_ratio = (
                    f"{r.peak_memory_mb / baseline.peak_memory_mb:.2f}x"
                    if baseline.peak_memory_mb > 0 and r.peak_memory_mb > 0
                    else "N/A"
                )
                console.print(
                    f"  {r.config.short_label:40s} "
                    f"{speedup:.2f}x speed, {mem_ratio} memory "
                    f"({r.resolution})"
                )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark matrix for wave 2 optimizations")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/corridorkey_mlx.safetensors"),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs="+",
        default=[512],
        help="Resolution(s) to benchmark",
    )
    parser.add_argument(
        "--sweep",
        choices=["key", "ablation", "full"],
        default="ablation",
        help="key=6 configs, ablation=8 configs (toggle one off), full=64 configs",
    )
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--bench-runs", type=int, default=10)
    args = parser.parse_args()

    ckpt = args.checkpoint if args.checkpoint.exists() else None
    configs = generate_matrix(args.sweep)

    console.print("[bold]CorridorKey MLX — Optimization Benchmark Matrix[/bold]")
    console.print(f"  Weights: {args.checkpoint if ckpt else 'random (no checkpoint)'}")
    console.print(f"  Resolutions: {args.resolution}")
    console.print(f"  Sweep: {args.sweep} ({len(configs)} configs)")
    console.print(f"  Warmup: {args.warmup_runs}, Bench: {args.bench_runs}")
    console.print()

    all_results: list[BenchResult] = []
    total = len(configs) * len(args.resolution)
    idx = 0

    for res in args.resolution:
        for config in configs:
            idx += 1
            console.print(
                f"  [{idx}/{total}] {res}x{res} — {config.short_label}...",
                end=" ",
            )
            result = bench_config(config, res, ckpt, args.warmup_runs, args.bench_runs)
            if result.error:
                console.print("[red]FAIL[/red]")
            else:
                console.print(
                    f"[green]{result.median_ms:.1f}ms[/green] "
                    f"(peak: {result.peak_memory_mb:.0f}MB)"
                    if result.peak_memory_mb > 0
                    else f"[green]{result.median_ms:.1f}ms[/green]"
                )
            all_results.append(result)

    console.print()
    print_results(all_results)


if __name__ == "__main__":
    main()
