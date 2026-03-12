#!/usr/bin/env python3
"""A/B benchmark: normal compiled forward vs mx.export_function AOT.

Tests whether exporting the compiled graph to disk and reimporting it
eliminates Python graph-building overhead on each inference call.

Usage:
    uv run python scripts/bench_export_function.py
    uv run python scripts/bench_export_function.py --img-size 1024
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from corridorkey_mlx.model.corridorkey import GreenFormer

CHECKPOINT = Path("checkpoints/corridorkey_mlx.safetensors")
EXPORT_PATH = Path("research/artifacts/forward_exported.mlxfn")
WARMUP = 5
BENCH_RUNS = 20
IMG_SIZE = 512


def _materialize(*args: mx.array | dict | tuple) -> None:
    """Force MLX lazy arrays to materialize on GPU.

    Uses mx.eval — MLX array materialization (NOT Python eval).
    """
    mx.eval(*args)  # noqa: S307  -- mx.eval, not Python eval()


def build_model(img_size: int) -> GreenFormer:
    """Build and load the model with standard settings."""
    model = GreenFormer(img_size=img_size, slim=True)
    model.load_checkpoint(CHECKPOINT)
    return model


def make_input(img_size: int) -> mx.array:
    """Create deterministic dummy input."""
    mx.random.seed(42)
    x = mx.random.normal((1, img_size, img_size, 4))
    _materialize(x)
    return x


def bench_times(fn: object, x: mx.array, warmup: int, runs: int) -> list[float]:
    """Generic benchmark loop."""
    for _ in range(warmup):
        out = fn(x)
        _materialize(out)

    times: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        out = fn(x)
        _materialize(out)
        times.append((time.perf_counter() - start) * 1000.0)
    return sorted(times)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--runs", type=int, default=BENCH_RUNS)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    args = parser.parse_args()

    if not CHECKPOINT.exists():
        print(f"ERROR: checkpoint not found at {CHECKPOINT}")
        sys.exit(1)

    img_size = args.img_size
    print(f"Benchmarking export_function @ {img_size}x{img_size}")
    print(f"Warmup: {args.warmup}, Runs: {args.runs}\n")

    # Build model
    model = build_model(img_size)
    x = make_input(img_size)

    # Warmup model (triggers per-component compilation)
    print("Warming up model...")
    out = model(x)
    _materialize(out)

    # Export the forward function
    # Set _compiled flag to skip async_eval/stage_gc inside forward
    # (not allowed inside graph transformations like export)
    model._compiled = True

    # Wrap forward to return tuple (export_function requires array/tuple output)
    # slim=True returns: alpha_coarse, fg_coarse, alpha_final, fg_final
    output_keys = ["alpha_coarse", "fg_coarse", "alpha_final", "fg_final"]

    def forward_tuple(x: mx.array) -> tuple:
        d = model(x)
        return tuple(d[k] for k in output_keys)

    print(f"Exporting forward graph to {EXPORT_PATH}...")
    EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    mx.export_function(str(EXPORT_PATH), forward_tuple, x)
    print("Export complete.")

    # Import the exported function
    print("Importing exported function...")
    imported_fn = mx.import_function(str(EXPORT_PATH))
    print("Import complete.\n")

    # Verify: check exported fn output
    print("Verifying exported function...")
    exported_out = imported_fn(x)
    _materialize(exported_out)
    if isinstance(exported_out, tuple):
        print(f"  Returns {len(exported_out)} arrays")
        for i, arr in enumerate(exported_out):
            print(f"  [{i}] shape={arr.shape} dtype={arr.dtype}")

    # Reset _compiled so normal path uses stage_gc
    model._compiled = False

    # A: Normal compiled forward
    print("\nRunning normal compiled forward...")
    normal_times = bench_times(model, x, args.warmup, args.runs)
    normal_median = normal_times[len(normal_times) // 2]

    # B: Exported/imported forward
    print("Running exported forward...")
    export_times = bench_times(imported_fn, x, args.warmup, args.runs)
    export_median = export_times[len(export_times) // 2]

    diff_ms = normal_median - export_median
    diff_pct = (diff_ms / normal_median) * 100 if normal_median > 0 else 0

    p95_idx = max(0, int(len(normal_times) * 0.95) - 1)

    print(f"\n{'Strategy':<20} {'Median':>10} {'Min':>10} {'P95':>10}")
    print("-" * 52)
    print(f"{'normal':<20} {normal_median:>9.2f}ms {normal_times[0]:>9.2f}ms {normal_times[p95_idx]:>9.2f}ms")
    print(f"{'exported':<20} {export_median:>9.2f}ms {export_times[0]:>9.2f}ms {export_times[p95_idx]:>9.2f}ms")
    print(f"\nDelta: {diff_ms:+.2f}ms ({diff_pct:+.1f}%)")
    print(f"Winner: {'exported' if export_median < normal_median else 'normal'}")

    # Cleanup
    if EXPORT_PATH.exists():
        EXPORT_PATH.unlink()
        print(f"\nCleaned up {EXPORT_PATH}")


if __name__ == "__main__":
    main()
