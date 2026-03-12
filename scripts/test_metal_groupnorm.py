#!/usr/bin/env python3
"""Prototype: Custom Metal GroupNorm via simd_sum + atomics.

Tests whether mx.fast.metal_kernel() can implement GroupNorm without
the transpose overhead that nn.GroupNorm(pytorch_compatible=True) requires.

Two-kernel approach (no global barrier available):
  Kernel A: per-group sum + sum_sq via simd_sum -> atomic accumulate to stats buffer
  Kernel B: normalize + affine + optional relu, reading stats from kernel A

Key unknowns being tested:
  1. Can 262K atomic adds per group (at 1024^2) converge to correct stats?
  2. Is the atomic contention faster or slower than the transpose approach?
  3. Does fp32 accumulation via atomics preserve enough precision?

Usage:
    uv run python scripts/test_metal_groupnorm.py
    uv run python scripts/test_metal_groupnorm.py --spatial 512
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from corridorkey_mlx.model.refiner import REFINER_CHANNELS, REFINER_GROUPS

# Hardcoded for refiner: G=8, C=64, gs=8
NUM_GROUPS = REFINER_GROUPS  # 8
NUM_CHANNELS = REFINER_CHANNELS  # 64
GROUP_SIZE = NUM_CHANNELS // NUM_GROUPS  # 8


def _make_stats_kernel(num_groups: int, group_size: int, num_channels: int):
    """Build stats kernel with baked-in constants."""
    source = f"""
    uint elem = thread_position_in_grid.x;

    uint B = inp_shape[0];
    uint H = inp_shape[1];
    uint W = inp_shape[2];
    const uint C = {num_channels};
    const uint G = {num_groups};
    const uint gs = {group_size};
    uint HW = H * W;
    uint group_elements = HW * gs;
    uint total = B * G * group_elements;

    if (elem >= total) return;

    // Map so consecutive threads stay within SAME group
    uint within = elem % group_elements;
    uint bg = elem / group_elements;
    uint g = bg % G;
    uint b = bg / G;

    // Map within-group index to NHWC position
    uint spatial_idx = within / gs;
    uint channel_in_group = within % gs;
    uint c = g * gs + channel_in_group;
    uint inp_idx = b * HW * C + spatial_idx * C + c;

    // Read value in fp32
    float val = static_cast<float>(inp[inp_idx]);
    float val_sq = val * val;

    // SIMD reduction (32 threads, all within same group)
    float partial_sum = simd_sum(val);
    float partial_sq = simd_sum(val_sq);

    // Lane 0 atomically accumulates to stats buffer
    if (thread_index_in_simdgroup == 0) {{
        uint stats_idx = (b * G + g) * 2;
        atomic_fetch_add_explicit(&stats[stats_idx], partial_sum, memory_order_relaxed);
        atomic_fetch_add_explicit(&stats[stats_idx + 1], partial_sq, memory_order_relaxed);
    }}
    """
    return mx.fast.metal_kernel(
        name=f"groupnorm_stats_g{num_groups}_c{num_channels}",
        input_names=["inp"],
        output_names=["stats"],
        source=source,
        ensure_row_contiguous=False,
        atomic_outputs=True,
    )


def _make_normalize_kernel(num_groups: int, group_size: int, num_channels: int, relu: bool):
    """Build normalize kernel with baked-in constants. Output is SAME dtype as input."""
    relu_code = "result = result > 0.0f ? result : 0.0f;" if relu else ""
    source = f"""
    uint elem = thread_position_in_grid.x;

    uint B = inp_shape[0];
    uint H = inp_shape[1];
    uint W = inp_shape[2];
    const uint C = {num_channels};
    const uint G = {num_groups};
    const uint gs = {group_size};
    uint HW = H * W;
    uint total = B * HW * C;

    if (elem >= total) return;

    uint c = elem % C;
    uint b = elem / (HW * C);
    uint g = c / gs;

    // Read stats (sum, sum_sq) -- fp32
    uint stats_idx = (b * G + g) * 2;
    float s = stats[stats_idx];
    float sq = stats[stats_idx + 1];

    // Compute mean and variance
    float n = static_cast<float>(HW * gs);
    float mean = s / n;
    float var = sq / n - mean * mean;
    float inv_std = 1.0f / sqrt(var + 1e-5f);

    // Normalize in fp32
    float val = static_cast<float>(inp[elem]);
    float normed = (val - mean) * inv_std;

    // Affine (weight and bias are fp32)
    float w = weight[c];
    float bi = bias[c];
    float result = normed * w + bi;

    // Optional ReLU
    {relu_code}

    out[elem] = static_cast<T>(result);
    """
    suffix = "_relu" if relu else ""
    return mx.fast.metal_kernel(
        name=f"groupnorm_norm_g{num_groups}_c{num_channels}{suffix}",
        input_names=["inp", "stats", "weight", "bias"],
        output_names=["out"],
        source=source,
    )


# Build kernels
_stats_kernel = _make_stats_kernel(NUM_GROUPS, GROUP_SIZE, NUM_CHANNELS)
_norm_kernel = _make_normalize_kernel(NUM_GROUPS, GROUP_SIZE, NUM_CHANNELS, relu=False)
_norm_relu_kernel = _make_normalize_kernel(NUM_GROUPS, GROUP_SIZE, NUM_CHANNELS, relu=True)


def metal_groupnorm(
    x: mx.array,
    weight: mx.array,
    bias: mx.array,
    num_groups: int = NUM_GROUPS,
    eps: float = 1e-5,
    relu: bool = False,
) -> mx.array:
    """GroupNorm via two custom Metal kernels (no transpose)."""
    B, H, W, C = x.shape
    gs = C // num_groups
    total_elements = B * H * W * C
    total_group_elements = B * num_groups * H * W * gs

    # Ensure weight/bias are fp32 for the normalize kernel
    w32 = weight.astype(mx.float32) if weight.dtype != mx.float32 else weight
    b32 = bias.astype(mx.float32) if bias.dtype != mx.float32 else bias

    # Kernel A: compute stats (sum, sum_sq per group)
    stats_shape = (B, num_groups, 2)
    stats_outputs = _stats_kernel(
        inputs=[x],
        template=[("T", x.dtype)],
        output_shapes=[stats_shape],
        output_dtypes=[mx.float32],
        grid=(total_group_elements, 1, 1),
        threadgroup=(256, 1, 1),
        init_value=0,
    )
    stats = stats_outputs[0]

    # Kernel B: normalize + affine + optional relu
    # Output same dtype as input (bf16 in practice)
    kern = _norm_relu_kernel if relu else _norm_kernel
    norm_outputs = kern(
        inputs=[x, stats, w32, b32],
        template=[("T", x.dtype)],
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
        grid=(total_elements, 1, 1),
        threadgroup=(256, 1, 1),
    )
    return norm_outputs[0]


# ── Benchmark harness ───────────────────────────────────────────────────────

WARMUP = 5
BENCH = 20


def time_fn(fn, warmup=WARMUP, bench=BENCH, label=""):
    """Time a function, return median ms."""
    for _ in range(warmup):
        out = fn()
        # mx.eval = MLX array materialization (forces GPU compute), not Python eval
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
    print(f"  {label:45s}  median={median:7.2f}ms  p95={p95:7.2f}ms  min={times[0]:7.2f}ms")
    return median


def test_correctness(spatial: int, dtype=mx.bfloat16):
    """Verify Metal GN matches nn.GroupNorm output."""
    print(f"\n=== Correctness Test @ {spatial}x{spatial} ===")
    B, H, W, C = 1, spatial, spatial, NUM_CHANNELS
    G = NUM_GROUPS

    mx.random.seed(42)
    x = mx.random.normal((B, H, W, C)).astype(dtype)
    mx.eval(x)  # noqa: S307

    # ── Test 1: unit weight/bias ──
    weight = mx.ones((C,))
    bias = mx.zeros((C,))
    mx.eval(weight, bias)  # noqa: S307

    gn = nn.GroupNorm(G, C, pytorch_compatible=True)
    mx.eval(gn.parameters())  # noqa: S307
    ref = gn(x).astype(dtype)  # cast ref to input dtype for fair compare
    metal = metal_groupnorm(x, weight, bias, G)
    mx.eval(ref, metal)  # noqa: S307

    ref_np = np.array(ref.astype(mx.float32))
    metal_np = np.array(metal.astype(mx.float32))
    max_abs = float(np.max(np.abs(ref_np - metal_np)))
    print(f"  unit w/b:   max_abs={max_abs:.6e}")

    # ── Test 2: random weight/bias ──
    weight2 = mx.random.normal((C,))
    bias2 = mx.random.normal((C,)) * 0.1
    mx.eval(weight2, bias2)  # noqa: S307

    gn2 = nn.GroupNorm(G, C, pytorch_compatible=True)
    gn2.weight = weight2
    gn2.bias = bias2
    ref2 = gn2(x).astype(dtype)
    metal2 = metal_groupnorm(x, weight2, bias2, G)
    mx.eval(ref2, metal2)  # noqa: S307

    ref2_np = np.array(ref2.astype(mx.float32))
    metal2_np = np.array(metal2.astype(mx.float32))
    max_abs2 = float(np.max(np.abs(ref2_np - metal2_np)))
    mean_abs2 = float(np.mean(np.abs(ref2_np - metal2_np)))
    print(f"  rand w/b:   max_abs={max_abs2:.6e}  mean_abs={mean_abs2:.6e}")

    # ── Test 3: with ReLU ──
    ref_relu = nn.relu(gn(x)).astype(dtype)
    metal_relu = metal_groupnorm(x, weight, bias, G, relu=True)
    mx.eval(ref_relu, metal_relu)  # noqa: S307

    ref_relu_np = np.array(ref_relu.astype(mx.float32))
    metal_relu_np = np.array(metal_relu.astype(mx.float32))
    max_abs_relu = float(np.max(np.abs(ref_relu_np - metal_relu_np)))
    print(f"  relu:       max_abs={max_abs_relu:.6e}")

    # ── Stats accuracy check ──
    gs = C // G
    total_group_elements = B * G * spatial * spatial * gs
    stats_out = _stats_kernel(
        inputs=[x],
        template=[("T", x.dtype)],
        output_shapes=[(B, G, 2)],
        output_dtypes=[mx.float32],
        grid=(total_group_elements, 1, 1),
        threadgroup=(256, 1, 1),
        init_value=0,
    )[0]
    mx.eval(stats_out)  # noqa: S307

    x_np = np.array(x.astype(mx.float32)).reshape(B, spatial * spatial, G, gs)
    ref_sums = x_np.sum(axis=(1, 3))
    ref_sumsq = (x_np ** 2).sum(axis=(1, 3))
    stats_np = np.array(stats_out)
    sum_err = float(np.max(np.abs(stats_np[0, :, 0] - ref_sums[0])))
    sumsq_err = float(np.max(np.abs(stats_np[0, :, 1] - ref_sumsq[0])))
    print(f"  stats sum err:  {sum_err:.6e}")
    print(f"  stats sumsq err: {sumsq_err:.6e}")

    # ── Atomic determinism check ──
    stats_out2 = _stats_kernel(
        inputs=[x],
        template=[("T", x.dtype)],
        output_shapes=[(B, G, 2)],
        output_dtypes=[mx.float32],
        grid=(total_group_elements, 1, 1),
        threadgroup=(256, 1, 1),
        init_value=0,
    )[0]
    mx.eval(stats_out2)  # noqa: S307
    stats2_np = np.array(stats_out2)
    det_err = float(np.max(np.abs(stats_np - stats2_np)))
    print(f"  stats determinism: {det_err:.6e} (0 = deterministic)")

    TOLERANCE = 5e-3
    passed = max_abs < TOLERANCE and max_abs_relu < TOLERANCE and max_abs2 < TOLERANCE
    print(f"  PASS: {passed} (tolerance: {TOLERANCE})")
    return passed


def bench_comparison(spatial: int, dtype=mx.bfloat16):
    """Benchmark Metal GN vs nn.GroupNorm."""
    print(f"\n=== Benchmark @ {spatial}x{spatial} ===")
    B, H, W, C = 1, spatial, spatial, NUM_CHANNELS
    G = NUM_GROUPS

    mx.random.seed(42)
    x = mx.random.normal((B, H, W, C)).astype(dtype)
    weight = mx.ones((C,))
    bias = mx.zeros((C,))
    mx.eval(x, weight, bias)  # noqa: S307

    # nn.GroupNorm (eager)
    gn = nn.GroupNorm(G, C, pytorch_compatible=True)
    mx.eval(gn.parameters())  # noqa: S307
    time_fn(lambda: gn(x), label="nn.GroupNorm (eager)")

    # nn.GroupNorm (compiled)
    gn_compiled = mx.compile(gn.__call__)
    t_ref_compiled = time_fn(lambda: gn_compiled(x), label="nn.GroupNorm (compiled)")

    # nn.GroupNorm + ReLU (compiled)
    gn_relu_compiled = mx.compile(lambda xi: nn.relu(gn(xi)))
    t_ref_relu = time_fn(lambda: gn_relu_compiled(x), label="nn.GroupNorm + ReLU (compiled)")

    # Metal GN (custom kernels can't be mx.compiled)
    t_metal = time_fn(lambda: metal_groupnorm(x, weight, bias, G), label="Metal GN (2-kernel)")

    # Metal GN + ReLU
    t_metal_relu = time_fn(
        lambda: metal_groupnorm(x, weight, bias, G, relu=True),
        label="Metal GN + fused ReLU (2-kernel)",
    )

    print(f"\n  Summary:")
    print(f"    nn.GroupNorm compiled:          {t_ref_compiled:7.2f}ms")
    print(f"    nn.GroupNorm+ReLU compiled:     {t_ref_relu:7.2f}ms")
    print(f"    Metal GN (2-kernel):            {t_metal:7.2f}ms")
    print(f"    Metal GN+ReLU (2-kernel):       {t_metal_relu:7.2f}ms")
    if t_ref_compiled > 0:
        print(f"    Metal vs compiled GN:           {(t_metal / t_ref_compiled - 1) * 100:+.1f}%")
    if t_ref_relu > 0:
        print(f"    Metal+ReLU vs compiled GN+ReLU: {(t_metal_relu / t_ref_relu - 1) * 100:+.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Metal GroupNorm prototype")
    parser.add_argument("--spatial", type=int, default=1024, help="Spatial dimension")
    parser.add_argument("--bench-only", action="store_true", help="Skip correctness test")
    args = parser.parse_args()

    print(f"MLX version: {mx.__version__}")
    print(f"Spatial: {args.spatial}x{args.spatial}")
    print(f"Groups: {NUM_GROUPS}, Channels: {NUM_CHANNELS}, GroupSize: {GROUP_SIZE}")
    elems_per_group = args.spatial * args.spatial * GROUP_SIZE
    print(f"Elements per group: {elems_per_group:,}")
    print(f"Atomic adds per group (stats kernel): ~{elems_per_group // 32:,}")

    if not args.bench_only:
        # Test at small size first
        passed_small = test_correctness(64)
        if not passed_small:
            print("\nFAILED at 64x64 -- checking if perf is worth investigating anyway")
            bench_comparison(64)
            print("\nRunning at target size for perf comparison...")

        # Test at target size
        if args.spatial != 64:
            test_correctness(args.spatial)

    bench_comparison(args.spatial)


if __name__ == "__main__":
    main()
