#!/usr/bin/env python3
"""Micro-benchmark GroupNorm overhead in the refiner at 1024x1024.

Measures:
1. Isolated GN call time (compiled vs eager)
2. Full refiner with standard GN vs GN replaced by identity (to isolate GN cost)
3. Manual mean/var approach (no transpose) vs pytorch_compatible (transpose+layer_norm)

Usage:
    uv run python scripts/bench_groupnorm.py

Note: mx.eval() calls throughout are MLX array materialization (forces GPU compute),
NOT Python's eval(). This is standard MLX benchmarking practice.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from corridorkey_mlx.model.refiner import (
    CNNRefinerModule,
    REFINER_CHANNELS,
    REFINER_GROUPS,
)

WARMUP = 5
BENCH = 20
SPATIAL = 1024


def time_fn(fn, warmup=WARMUP, bench=BENCH, label=""):
    """Time a function, return median ms."""
    # warmup
    for _ in range(warmup):
        out = fn()
        # mx.eval = MLX array materialization, not Python eval
        mx.eval(out)  # noqa: S307

    times = []
    for _ in range(bench):
        start = time.perf_counter()
        out = fn()
        mx.eval(out)  # noqa: S307
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    times.sort()
    median = times[len(times) // 2]
    p95 = times[int(len(times) * 0.95)]
    print(f"  {label:40s}  median={median:7.2f}ms  p95={p95:7.2f}ms  min={times[0]:7.2f}ms")
    return median


def bench_isolated_gn():
    """Benchmark a single GroupNorm call at 1024x1024."""
    print(f"\n=== Isolated GroupNorm @ {SPATIAL}x{SPATIAL} ===")
    x = mx.random.normal((1, SPATIAL, SPATIAL, REFINER_CHANNELS)).astype(mx.bfloat16)
    mx.eval(x)  # noqa: S307

    # Standard pytorch_compatible GN
    gn_compat = nn.GroupNorm(REFINER_GROUPS, REFINER_CHANNELS, pytorch_compatible=True)
    mx.eval(gn_compat.parameters())  # noqa: S307

    # Eager
    time_fn(lambda: gn_compat(x), label="GN pytorch_compatible (eager)")

    # Compiled
    gn_compiled = mx.compile(gn_compat.__call__)
    time_fn(lambda: gn_compiled(x), label="GN pytorch_compatible (compiled)")

    # Manual mean/var (no transpose) -- for comparison
    gn_weight = gn_compat.weight
    gn_bias = gn_compat.bias

    def manual_gn(x_in):
        B, H, W, C = x_in.shape
        gs = C // REFINER_GROUPS
        x_g = x_in.reshape(B, -1, REFINER_GROUPS, gs)
        # fp32 accumulation for precision
        x_f32 = x_g.astype(mx.float32)
        mean = mx.mean(x_f32, axis=(1, 3), keepdims=True)
        var = mx.var(x_f32, axis=(1, 3), keepdims=True)
        x_norm = (x_g - mean.astype(x_g.dtype)) * mx.rsqrt(var.astype(x_g.dtype) + 1e-5)
        out = x_norm.reshape(B, H, W, C)
        return gn_weight * out + gn_bias

    time_fn(lambda: manual_gn(x), label="GN manual mean/var fp32 accum (eager)")

    manual_gn_compiled = mx.compile(manual_gn)
    time_fn(lambda: manual_gn_compiled(x), label="GN manual mean/var fp32 accum (compiled)")

    # GN + ReLU (fused timing)
    time_fn(lambda: nn.relu(gn_compat(x)), label="GN + ReLU (eager)")
    gn_relu_compiled = mx.compile(lambda x_in: nn.relu(gn_compat(x_in)))
    time_fn(lambda: gn_relu_compiled(x), label="GN + ReLU (compiled)")


def bench_refiner_with_without_gn():
    """Compare full refiner time vs refiner with GN replaced by identity."""
    print(f"\n=== Full Refiner @ {SPATIAL}x{SPATIAL} ===")

    rgb = mx.random.normal((1, SPATIAL, SPATIAL, 3)).astype(mx.bfloat16)
    coarse = mx.random.normal((1, SPATIAL, SPATIAL, 4)).astype(mx.bfloat16)
    mx.eval(rgb, coarse)  # noqa: S307

    # Standard refiner
    refiner = CNNRefinerModule()
    refiner.prepare_inference()
    mx.eval(refiner.parameters())  # noqa: S307

    time_fn(lambda: refiner(rgb, coarse), label="Refiner standard (eager)")

    refiner_compiled = mx.compile(refiner.__call__)
    time_fn(lambda: refiner_compiled(rgb, coarse), label="Refiner standard (compiled)")

    # Refiner with identity GN (to measure GN contribution)
    class IdentityGN(nn.Module):
        """Drop-in replacement that skips normalization."""

        def __init__(self, num_groups, num_channels, **kwargs):
            super().__init__()
            self.weight = mx.ones((num_channels,))
            self.bias = mx.zeros((num_channels,))

        def __call__(self, x):
            return x  # identity -- just to measure GN's time contribution

    refiner_no_gn = CNNRefinerModule()
    refiner_no_gn.prepare_inference()
    mx.eval(refiner_no_gn.parameters())  # noqa: S307
    # Replace all GN instances with identity
    refiner_no_gn.stem_gn = IdentityGN(REFINER_GROUPS, REFINER_CHANNELS)
    for blk_name in ("res1", "res2", "res3", "res4"):
        blk = getattr(refiner_no_gn, blk_name)
        blk.gn1 = IdentityGN(REFINER_GROUPS, REFINER_CHANNELS)
        blk.gn2 = IdentityGN(REFINER_GROUPS, REFINER_CHANNELS)
    mx.eval(refiner_no_gn.parameters())  # noqa: S307

    time_fn(lambda: refiner_no_gn(rgb, coarse), label="Refiner NO GN (eager)")

    refiner_no_gn_compiled = mx.compile(refiner_no_gn.__call__)
    time_fn(lambda: refiner_no_gn_compiled(rgb, coarse), label="Refiner NO GN (compiled)")


def bench_gn_count_verification():
    """Verify the actual GN instance count."""
    print("\n=== GN Instance Count ===")
    refiner = CNNRefinerModule()
    count = 0
    for name, module in refiner.named_modules():
        if isinstance(module, nn.GroupNorm):
            count += 1
            print(f"  {name}")
    print(f"  Total: {count} GroupNorm instances")


def main():
    print(f"MLX version: {mx.__version__}")
    print(f"Spatial: {SPATIAL}x{SPATIAL}, dtype: bf16")
    print(f"Warmup: {WARMUP}, Bench runs: {BENCH}")

    mx.random.seed(42)

    bench_gn_count_verification()
    bench_isolated_gn()
    bench_refiner_with_without_gn()


if __name__ == "__main__":
    main()
