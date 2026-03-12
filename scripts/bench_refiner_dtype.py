#!/usr/bin/env python3
"""A/B benchmark: refiner fp16 vs bf16 via load_model() pipeline.

Tests whether matching refiner dtype to decoder dtype (bf16) eliminates
redundant copy kernels identified in Metal GPU trace (7.3% g3_copyfloat16float16).

Usage:
    uv run python scripts/bench_refiner_dtype.py
    uv run python scripts/bench_refiner_dtype.py --img-size 1024 --runs 20
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from corridorkey_mlx.inference.pipeline import load_model

CHECKPOINT = Path("checkpoints/corridorkey_mlx.safetensors")
WARMUP = 5
BENCH_RUNS = 15


def _materialize(*args: mx.array | dict) -> None:
    """Force MLX lazy arrays to materialize on GPU.

    Uses mx.eval which is MLX's array materialization function,
    NOT Python's built-in eval().
    """
    mx.eval(*args)  # noqa: S307  -- mx.eval, not Python eval()


def bench(img_size: int, refiner_dtype: mx.Dtype, warmup: int, runs: int) -> list[float]:
    """Benchmark load_model pipeline with given refiner dtype."""
    model = load_model(
        checkpoint=CHECKPOINT,
        img_size=img_size,
        refiner_dtype=refiner_dtype,
    )

    mx.random.seed(42)
    x = mx.random.normal((1, img_size, img_size, 4))
    _materialize(x)

    # Warmup
    for _ in range(warmup):
        out = model(x)
        _materialize(out)

    # Benchmark
    times: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        out = model(x)
        _materialize(out)
        times.append((time.perf_counter() - start) * 1000.0)

    del model, x
    gc.collect()
    mx.clear_cache()

    return sorted(times)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--runs", type=int, default=BENCH_RUNS)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    args = parser.parse_args()

    if not CHECKPOINT.exists():
        print(f"ERROR: checkpoint not found at {CHECKPOINT}")
        sys.exit(1)

    print(f"Benchmarking refiner dtype @ {args.img_size}x{args.img_size}")
    print(f"Warmup: {args.warmup}, Runs: {args.runs}\n")

    # A: fp16 (current default in load_model)
    print("Running fp16 refiner...")
    fp16_times = bench(args.img_size, mx.float16, args.warmup, args.runs)
    fp16_median = fp16_times[len(fp16_times) // 2]

    # B: bf16 (matching decoder dtype)
    print("Running bf16 refiner...")
    bf16_times = bench(args.img_size, mx.bfloat16, args.warmup, args.runs)
    bf16_median = bf16_times[len(bf16_times) // 2]

    diff_ms = fp16_median - bf16_median
    diff_pct = (diff_ms / fp16_median) * 100 if fp16_median > 0 else 0

    p95_idx = max(0, int(len(fp16_times) * 0.95) - 1)

    print(f"\n{'Variant':<12} {'Median':>10} {'Min':>10} {'P95':>10}")
    print("-" * 44)
    print(f"{'fp16':<12} {fp16_median:>9.2f}ms {fp16_times[0]:>9.2f}ms {fp16_times[p95_idx]:>9.2f}ms")
    print(f"{'bf16':<12} {bf16_median:>9.2f}ms {bf16_times[0]:>9.2f}ms {bf16_times[p95_idx]:>9.2f}ms")
    print(f"\nDelta: {diff_ms:+.2f}ms ({diff_pct:+.1f}%)")
    print(f"Winner: {'bf16' if bf16_median < fp16_median else 'fp16'}")


if __name__ == "__main__":
    main()
