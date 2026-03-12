#!/usr/bin/env python3
"""Test GroupNorm variant strategies for the refiner.

Variants tested:
1. Standard nn.GroupNorm (pytorch_compatible=True) -- baseline
2. GN with explicit contiguous copy after output
3. "Transposed-affine" GN -- apply affine before second transpose
4. Two-pass manual: fp32 stats + normalize (no transpose at all)

All mx.eval() calls are MLX array materialization (GPU sync), not Python eval().

Usage:
    uv run python scripts/bench_groupnorm_variants.py
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
    REFINER_SCALE,
)

WARMUP = 5
BENCH = 20
SPATIAL = 1024
EPS = 1e-5


def mlx_sync(*args):
    """MLX array materialization (GPU sync). Wrapper to avoid S307 noise."""
    mx.eval(*args)  # noqa: S307


def time_fn(fn, warmup=WARMUP, bench=BENCH, label=""):
    """Time a function, return median ms."""
    for _ in range(warmup):
        mlx_sync(fn())

    times = []
    for _ in range(bench):
        start = time.perf_counter()
        mlx_sync(fn())
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    times.sort()
    median = times[len(times) // 2]
    p95 = times[int(len(times) * 0.95)]
    print(f"  {label:50s}  median={median:7.2f}ms  p95={p95:7.2f}ms")
    return median


def check_parity(ref, test, name):
    """Check max abs diff between two arrays."""
    diff = float(mx.max(mx.abs(ref.astype(mx.float32) - test.astype(mx.float32))))
    status = "OK" if diff < 1e-3 else "FAIL"
    print(f"  {name:50s}  max_abs_diff={diff:.2e} [{status}]")
    return diff


class ContiguousCopyGN(nn.Module):
    """Standard GN + explicit copy to force contiguous output."""

    def __init__(self, num_groups, num_channels):
        super().__init__()
        self.inner = nn.GroupNorm(num_groups, num_channels, pytorch_compatible=True)

    @property
    def weight(self):
        return self.inner.weight

    @property
    def bias(self):
        return self.inner.bias

    def __call__(self, x):
        out = self.inner(x)
        # Force contiguous via broadcast add with scalar zero
        return out + 0.0


class TransposedAffineGN(nn.Module):
    """Apply affine BEFORE the second transpose, then transpose once."""

    def __init__(self, num_groups, num_channels):
        super().__init__()
        self.num_groups = num_groups
        self.group_size = num_channels // num_groups
        self.weight = mx.ones((num_channels,))
        self.bias = mx.zeros((num_channels,))

    def __call__(self, x):
        batch, *rest, dims = x.shape
        gs = self.group_size
        ng = self.num_groups

        # Transpose #1 -- same as standard
        x = x.reshape(batch, -1, ng, gs)
        x = x.transpose(0, 2, 1, 3).reshape(batch, ng, -1)

        # layer_norm (fused kernel, fast)
        x = mx.fast.layer_norm(x, eps=EPS, weight=None, bias=None)

        # Apply affine IN transposed layout: (B, G, HW*gs)
        hw = x.shape[2] // gs
        x = x.reshape(batch, ng, hw, gs)     # (B, G, HW, gs)
        w = self.weight.reshape(1, ng, 1, gs)  # (1, G, 1, gs)
        b = self.bias.reshape(1, ng, 1, gs)    # (1, G, 1, gs)
        x = w * x + b                          # broadcast: (1,G,1,gs) * (B,G,HW,gs)

        # Single transpose back
        x = x.transpose(0, 2, 1, 3).reshape(batch, *rest, dims)
        return x


class TwoPassFP32GN(nn.Module):
    """Two-pass: compute stats in fp32, normalize in input dtype. No transpose."""

    def __init__(self, num_groups, num_channels):
        super().__init__()
        self.num_groups = num_groups
        self.group_size = num_channels // num_groups
        self.weight = mx.ones((num_channels,))
        self.bias = mx.zeros((num_channels,))

    def __call__(self, x):
        batch, *rest, dims = x.shape
        gs = self.group_size
        ng = self.num_groups

        # Reshape to group layout -- NO transpose
        x_g = x.reshape(batch, -1, ng, gs)  # (B, HW, G, gs)

        # fp32 stats (reduce over spatial=1 and channel_within_group=3)
        x_f32 = x_g.astype(mx.float32)
        mean = mx.mean(x_f32, axis=(1, 3), keepdims=True)  # (B, 1, G, 1)
        var = mx.var(x_f32, axis=(1, 3), keepdims=True)     # (B, 1, G, 1)

        # Normalize in input dtype
        x_norm = (x_g - mean.astype(x_g.dtype)) * mx.rsqrt(var.astype(x_g.dtype) + EPS)

        # Affine in group layout: weight (C,) -> (1, 1, G, gs)
        w = self.weight.reshape(1, 1, ng, gs)
        b = self.bias.reshape(1, 1, ng, gs)
        out = w * x_norm + b

        return out.reshape(batch, *rest, dims)


def make_refiner_with_gn(gn_class):
    """Create refiner with custom GN class replacing all GroupNorm instances."""
    refiner = CNNRefinerModule()

    # Replace stem_gn
    old = refiner.stem_gn
    new_gn = gn_class(REFINER_GROUPS, REFINER_CHANNELS)
    if hasattr(new_gn, "inner"):
        new_gn.inner.weight = old.weight
        new_gn.inner.bias = old.bias
    else:
        new_gn.weight = old.weight
        new_gn.bias = old.bias
    refiner.stem_gn = new_gn

    # Replace block GNs
    for blk_name in ("res1", "res2", "res3", "res4"):
        blk = getattr(refiner, blk_name)
        for gn_name in ("gn1", "gn2"):
            old = getattr(blk, gn_name)
            new_gn = gn_class(REFINER_GROUPS, REFINER_CHANNELS)
            if hasattr(new_gn, "inner"):
                new_gn.inner.weight = old.weight
                new_gn.inner.bias = old.bias
            else:
                new_gn.weight = old.weight
                new_gn.bias = old.bias
            setattr(blk, gn_name, new_gn)

    refiner.prepare_inference()
    mlx_sync(refiner.parameters())
    return refiner


def main():
    print(f"MLX {mx.__version__}, {SPATIAL}x{SPATIAL}, bf16, warmup={WARMUP}, bench={BENCH}")
    mx.random.seed(42)

    rgb = mx.random.normal((1, SPATIAL, SPATIAL, 3)).astype(mx.bfloat16)
    coarse = mx.random.normal((1, SPATIAL, SPATIAL, 4)).astype(mx.bfloat16)
    mlx_sync(rgb, coarse)

    # --- Isolated GN parity check ---
    print("\n=== Parity Check (isolated GN @ 1024) ===")
    x_test = mx.random.normal((1, SPATIAL, SPATIAL, REFINER_CHANNELS)).astype(mx.bfloat16)
    mlx_sync(x_test)

    ref_gn = nn.GroupNorm(REFINER_GROUPS, REFINER_CHANNELS, pytorch_compatible=True)
    mlx_sync(ref_gn.parameters())
    ref_out = ref_gn(x_test)
    mlx_sync(ref_out)

    for name, cls in [
        ("ContiguousCopyGN", ContiguousCopyGN),
        ("TransposedAffineGN", TransposedAffineGN),
        ("TwoPassFP32GN", TwoPassFP32GN),
    ]:
        gn = cls(REFINER_GROUPS, REFINER_CHANNELS)
        if hasattr(gn, "inner"):
            gn.inner.weight = ref_gn.weight
            gn.inner.bias = ref_gn.bias
        else:
            gn.weight = ref_gn.weight
            gn.bias = ref_gn.bias
        mlx_sync(gn.parameters())
        test_out = gn(x_test)
        mlx_sync(test_out)
        check_parity(ref_out, test_out, name)

    # --- Isolated GN timing ---
    print("\n=== Isolated GN Timing (compiled) ===")
    variants = [
        ("nn.GroupNorm (baseline)", None),
        ("ContiguousCopyGN", ContiguousCopyGN),
        ("TransposedAffineGN", TransposedAffineGN),
        ("TwoPassFP32GN", TwoPassFP32GN),
    ]
    for name, cls in variants:
        if cls is None:
            gn = ref_gn
        else:
            gn = cls(REFINER_GROUPS, REFINER_CHANNELS)
            if hasattr(gn, "inner"):
                gn.inner.weight = ref_gn.weight
                gn.inner.bias = ref_gn.bias
            else:
                gn.weight = ref_gn.weight
                gn.bias = ref_gn.bias
        mlx_sync(gn.parameters())
        fn_compiled = mx.compile(gn.__call__)
        time_fn(lambda fn=fn_compiled: fn(x_test), label=name)

    # --- Full refiner timing ---
    print("\n=== Full Refiner Timing (compiled) ===")

    # Baseline
    ref_refiner = CNNRefinerModule()
    ref_refiner.prepare_inference()
    mlx_sync(ref_refiner.parameters())
    ref_compiled = mx.compile(ref_refiner.__call__)
    baseline_median = time_fn(
        lambda: ref_compiled(rgb, coarse), label="Baseline (nn.GroupNorm)"
    )

    # Get reference output for parity
    ref_refiner_out = ref_refiner(rgb, coarse)
    mlx_sync(ref_refiner_out)

    for name, cls in [
        ("ContiguousCopyGN", ContiguousCopyGN),
        ("TransposedAffineGN", TransposedAffineGN),
        ("TwoPassFP32GN", TwoPassFP32GN),
    ]:
        variant_refiner = make_refiner_with_gn(cls)
        variant_compiled = mx.compile(variant_refiner.__call__)
        variant_median = time_fn(
            lambda fn=variant_compiled: fn(rgb, coarse), label=f"{name}"
        )

        # Parity vs baseline refiner
        variant_out = variant_refiner(rgb, coarse)
        mlx_sync(variant_out)
        diff = float(
            mx.max(mx.abs(ref_refiner_out.astype(mx.float32) - variant_out.astype(mx.float32)))
        )
        delta_pct = (variant_median - baseline_median) / baseline_median * 100
        print(f"    -> delta={delta_pct:+.1f}%  parity_max_abs={diff:.2e}")


if __name__ == "__main__":
    main()
