#!/usr/bin/env python3
"""A/B benchmark: per-component compile vs whole-forward compile.

Tests whether compile_forward=True eliminates inter-component CPU dispatch
gaps visible in the Metal GPU trace.

Usage:
    uv run python scripts/bench_compile_forward.py
    uv run python scripts/bench_compile_forward.py --img-size 1024
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
WARMUP = 5
BENCH_RUNS = 20


def _materialize(*args: mx.array | dict) -> None:
    """Force MLX lazy arrays to materialize on GPU.

    Uses mx.eval — MLX's array materialization function,
    NOT Python's built-in eval(). See MLX docs.
    """
    mx.eval(*args)  # noqa: S307  -- mx.eval, not Python eval()


def bench(img_size: int, compile_forward: bool, warmup: int, runs: int) -> list[float]:
    """Benchmark with given compile strategy."""
    model = GreenFormer(
        img_size=img_size,
        slim=True,
        compile_forward=compile_forward,
        # Disable per-component compile when testing whole-forward
        compile_backbone=not compile_forward,
        compile_decoders=not compile_forward,
        compile_refiner=not compile_forward,
    )
    model.load_checkpoint(CHECKPOINT)

    mx.random.seed(42)
    x = mx.random.normal((1, img_size, img_size, 4))
    _materialize(x)

    # Warmup (extra for compile_forward — first call triggers compilation)
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

    print(f"Benchmarking compile strategy @ {args.img_size}x{args.img_size}")
    print(f"Warmup: {args.warmup}, Runs: {args.runs}\n")

    # A: per-component compile (current default)
    print("Running per-component compile...")
    per_times = bench(args.img_size, compile_forward=False, warmup=args.warmup, runs=args.runs)
    per_median = per_times[len(per_times) // 2]

    # B: whole-forward compile
    print("Running whole-forward compile...")
    whole_times = bench(args.img_size, compile_forward=True, warmup=args.warmup, runs=args.runs)
    whole_median = whole_times[len(whole_times) // 2]

    diff_ms = per_median - whole_median
    diff_pct = (diff_ms / per_median) * 100 if per_median > 0 else 0

    p95_idx = max(0, int(len(per_times) * 0.95) - 1)

    print(f"\n{'Strategy':<20} {'Median':>10} {'Min':>10} {'P95':>10}")
    print("-" * 52)
    print(f"{'per-component':<20} {per_median:>9.2f}ms {per_times[0]:>9.2f}ms {per_times[p95_idx]:>9.2f}ms")
    print(f"{'whole-forward':<20} {whole_median:>9.2f}ms {whole_times[0]:>9.2f}ms {whole_times[p95_idx]:>9.2f}ms")
    print(f"\nDelta: {diff_ms:+.2f}ms ({diff_pct:+.1f}%)")
    print(f"Winner: {'whole-forward' if whole_median < per_median else 'per-component'}")


if __name__ == "__main__":
    main()
