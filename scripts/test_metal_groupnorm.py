#!/usr/bin/env python3
"""Prototype: Custom Metal GroupNorm via threadgroup shared memory.

Single-kernel approach — eliminates the NHWC↔NCHW transposes that
nn.GroupNorm(pytorch_compatible=True) requires.

Kernel design:
  - Grid: (B * G, 1, 1) — one threadgroup per (batch, group) pair
  - Two-pass within each threadgroup:
    Pass 1: accumulate sum/sumsq → simd_sum → shared memory → barrier → mean/var
    Pass 2: normalize + affine → write output
  - No atomics, fully deterministic

Usage:
    uv run python scripts/test_metal_groupnorm.py
    uv run python scripts/test_metal_groupnorm.py --spatial 2048
    uv run python scripts/test_metal_groupnorm.py --spatial 1024 --threads 512
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

NUM_GROUPS = REFINER_GROUPS  # 8
NUM_CHANNELS = REFINER_CHANNELS  # 64
GROUP_SIZE = NUM_CHANNELS // NUM_GROUPS  # 8


def _make_fused_groupnorm_kernel(
    num_groups: int,
    group_size: int,
    num_channels: int,
    relu: bool = False,
) -> object:
    """Build single-kernel GroupNorm with shared-memory tree reduction.

    One threadgroup per (batch, group). Two passes over group elements:
      Pass 1: parallel reduction for mean/variance via shared memory
      Pass 2: normalize + affine (+ optional ReLU), write output
    """
    relu_code = "result = result > 0.0f ? result : 0.0f;" if relu else ""

    # Max 1024 threads per threadgroup → max 32 SIMD groups
    max_simd_groups = 32

    source = f"""
    // One threadgroup per (batch, group) pair
    uint bg = threadgroup_position_in_grid.x;
    const uint G = {num_groups};
    const uint GS = {group_size};
    const uint C = {num_channels};
    uint b = bg / G;
    uint g = bg % G;

    // MLX provides thread_position_in_threadgroup (uint3), not scalar indices
    uint tid = thread_position_in_threadgroup.x;
    uint num_threads = threads_per_threadgroup.x;

    // Compute SIMD indices from thread position (Apple Silicon: 32-wide SIMD)
    const uint SIMD_WIDTH = 32;
    uint simd_lane = tid % SIMD_WIDTH;
    uint simd_gid = tid / SIMD_WIDTH;
    uint num_simd_groups = (num_threads + SIMD_WIDTH - 1) / SIMD_WIDTH;

    uint H = inp_shape[1];
    uint W = inp_shape[2];
    uint HW = H * W;
    uint group_elems = HW * GS;

    // Base offset for this batch in the NHWC tensor
    uint batch_offset = b * HW * C;
    // Channel offset for this group
    uint group_chan_offset = g * GS;

    // ── Pass 1: accumulate partial sum and sum_sq ──
    float local_sum = 0.0f;
    float local_sumsq = 0.0f;

    for (uint i = tid; i < group_elems; i += num_threads) {{
        uint spatial = i / GS;
        uint c_in_group = i % GS;
        uint idx = batch_offset + spatial * C + group_chan_offset + c_in_group;
        float val = static_cast<float>(inp[idx]);
        local_sum += val;
        local_sumsq += val * val;
    }}

    // SIMD-level reduction (32 threads)
    float simd_s = simd_sum(local_sum);
    float simd_sq = simd_sum(local_sumsq);

    // Write SIMD partial results to shared memory
    threadgroup float shared_sum[{max_simd_groups}];
    threadgroup float shared_sumsq[{max_simd_groups}];

    if (simd_lane == 0) {{
        shared_sum[simd_gid] = simd_s;
        shared_sumsq[simd_gid] = simd_sq;
    }}

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduction by first SIMD group → compute mean/inv_std
    // Then broadcast via shared memory
    if (simd_gid == 0) {{
        float s = (simd_lane < num_simd_groups) ? shared_sum[simd_lane] : 0.0f;
        float sq = (simd_lane < num_simd_groups) ? shared_sumsq[simd_lane] : 0.0f;
        s = simd_sum(s);
        sq = simd_sum(sq);

        if (simd_lane == 0) {{
            float n = static_cast<float>(group_elems);
            float mean = s / n;
            float var = sq / n - mean * mean;
            float inv_std = rsqrt(var + 1e-5f);
            shared_sum[0] = mean;
            shared_sumsq[0] = inv_std;
        }}
    }}

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float mean = shared_sum[0];
    float inv_std = shared_sumsq[0];

    // ── Pass 2: normalize + affine (+ optional ReLU) ──
    for (uint i = tid; i < group_elems; i += num_threads) {{
        uint spatial = i / GS;
        uint c_in_group = i % GS;
        uint c = group_chan_offset + c_in_group;
        uint idx = batch_offset + spatial * C + c;
        float val = static_cast<float>(inp[idx]);
        float normed = (val - mean) * inv_std;
        float result = normed * weight[c] + bias_arr[c];
        {relu_code}
        out[idx] = static_cast<T>(result);
    }}
    """
    suffix = "_relu" if relu else ""
    return mx.fast.metal_kernel(
        name=f"groupnorm_fused_g{num_groups}_c{num_channels}{suffix}",
        input_names=["inp", "weight", "bias_arr"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=False,
    )


# Build kernels
_fused_kernel = _make_fused_groupnorm_kernel(NUM_GROUPS, GROUP_SIZE, NUM_CHANNELS)
_fused_relu_kernel = _make_fused_groupnorm_kernel(
    NUM_GROUPS, GROUP_SIZE, NUM_CHANNELS, relu=True
)


def metal_groupnorm(
    x: mx.array,
    weight: mx.array,
    bias: mx.array,
    num_groups: int = NUM_GROUPS,
    relu: bool = False,
    threads_per_group: int = 256,
) -> mx.array:
    """GroupNorm via single fused Metal kernel (no transpose, no atomics)."""
    B, H, W, C = x.shape

    w32 = weight.astype(mx.float32) if weight.dtype != mx.float32 else weight
    b32 = bias.astype(mx.float32) if bias.dtype != mx.float32 else bias

    kern = _fused_relu_kernel if relu else _fused_kernel
    # MLX uses dispatchThreads semantics: grid = total threads, not threadgroups
    num_threadgroups = B * num_groups
    total_threads = num_threadgroups * threads_per_group
    outputs = kern(
        inputs=[x, w32, b32],
        template=[("T", x.dtype)],
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
        grid=(total_threads, 1, 1),
        threadgroup=(threads_per_group, 1, 1),
    )
    return outputs[0]


# ── V2: Multi-threadgroup approach ──────────────────────────────────────────
# Split each group across many threadgroups for better GPU utilization.
# Kernel A: partial reduction → 1 atomic per threadgroup (not per SIMD group)
# Kernel B: normalize + affine using precomputed stats


def _make_stats_kernel_v2(num_groups: int, group_size: int, num_channels: int) -> object:
    """Stats kernel: many threadgroups per group, shared-mem reduction + 1 atomic each."""
    max_simd_groups = 32
    source = f"""
    const uint G = {num_groups};
    const uint GS = {group_size};
    const uint C = {num_channels};

    // chunks_per_group is passed via chunks_shape[0]
    uint cpg = chunks_shape[0];

    // Decode threadgroup ID → (batch, group, chunk)
    uint tg_id = threadgroup_position_in_grid.x;
    uint chunk = tg_id % cpg;
    uint bg = tg_id / cpg;
    uint g = bg % G;
    uint b = bg / G;

    uint tid = thread_position_in_threadgroup.x;
    uint num_threads = threads_per_threadgroup.x;
    const uint SIMD_WIDTH = 32;
    uint simd_lane = tid % SIMD_WIDTH;
    uint simd_gid = tid / SIMD_WIDTH;
    uint num_simd_groups = (num_threads + SIMD_WIDTH - 1) / SIMD_WIDTH;

    uint H = inp_shape[1];
    uint W = inp_shape[2];
    uint HW = H * W;
    uint group_elems = HW * GS;

    // This chunk's range within the group
    uint chunk_size = (group_elems + cpg - 1) / cpg;
    uint chunk_start = chunk * chunk_size;
    uint chunk_end = min(chunk_start + chunk_size, group_elems);

    uint batch_offset = b * HW * C;
    uint group_chan_offset = g * GS;

    // ── Accumulate partial sums for this chunk ──
    float local_sum = 0.0f;
    float local_sumsq = 0.0f;

    for (uint i = chunk_start + tid; i < chunk_end; i += num_threads) {{
        uint spatial = i / GS;
        uint c_in_group = i % GS;
        uint idx = batch_offset + spatial * C + group_chan_offset + c_in_group;
        float val = static_cast<float>(inp[idx]);
        local_sum += val;
        local_sumsq += val * val;
    }}

    // SIMD reduction
    float simd_s = simd_sum(local_sum);
    float simd_sq = simd_sum(local_sumsq);

    threadgroup float shared_sum[{max_simd_groups}];
    threadgroup float shared_sumsq[{max_simd_groups}];

    if (simd_lane == 0) {{
        shared_sum[simd_gid] = simd_s;
        shared_sumsq[simd_gid] = simd_sq;
    }}

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduction within this threadgroup
    if (simd_gid == 0) {{
        float s = (simd_lane < num_simd_groups) ? shared_sum[simd_lane] : 0.0f;
        float sq = (simd_lane < num_simd_groups) ? shared_sumsq[simd_lane] : 0.0f;
        s = simd_sum(s);
        sq = simd_sum(sq);

        // ONE atomic per threadgroup (not per SIMD group)
        if (simd_lane == 0) {{
            uint stats_idx = (b * G + g) * 2;
            atomic_fetch_add_explicit(&stats[stats_idx], s, memory_order_relaxed);
            atomic_fetch_add_explicit(&stats[stats_idx + 1], sq, memory_order_relaxed);
        }}
    }}
    """
    return mx.fast.metal_kernel(
        name=f"groupnorm_stats_v2_g{num_groups}_c{num_channels}",
        input_names=["inp", "chunks"],
        output_names=["stats"],
        source=source,
        ensure_row_contiguous=False,
        atomic_outputs=True,
    )


def _make_norm_kernel_v2(
    num_groups: int, group_size: int, num_channels: int, relu: bool = False
) -> object:
    """Normalize kernel: reads precomputed stats, fully parallel over all elements."""
    relu_code = "result = result > 0.0f ? result : 0.0f;" if relu else ""
    source = f"""
    const uint G = {num_groups};
    const uint GS = {group_size};
    const uint C = {num_channels};

    // 1D dispatch over all elements
    uint elem = thread_position_in_grid.x;
    uint H = inp_shape[1];
    uint W = inp_shape[2];
    uint HW = H * W;
    uint total = inp_shape[0] * HW * C;
    if (elem >= total) return;

    uint c = elem % C;
    uint spatial = (elem / C) % HW;
    uint b = elem / (HW * C);
    uint g = c / GS;

    // Read precomputed stats
    uint stats_idx = (b * G + g) * 2;
    float s = stats[stats_idx];
    float sq = stats[stats_idx + 1];
    float n = static_cast<float>(HW * GS);
    float mean = s / n;
    float var = sq / n - mean * mean;
    float inv_std = rsqrt(var + 1e-5f);

    float val = static_cast<float>(inp[elem]);
    float normed = (val - mean) * inv_std;
    float result = normed * weight[c] + bias_arr[c];
    {relu_code}
    out[elem] = static_cast<T>(result);
    """
    suffix = "_relu" if relu else ""
    return mx.fast.metal_kernel(
        name=f"groupnorm_norm_v2_g{num_groups}_c{num_channels}{suffix}",
        input_names=["inp", "stats", "weight", "bias_arr"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=False,
    )


_stats_kernel_v2 = _make_stats_kernel_v2(NUM_GROUPS, GROUP_SIZE, NUM_CHANNELS)
_norm_kernel_v2 = _make_norm_kernel_v2(NUM_GROUPS, GROUP_SIZE, NUM_CHANNELS)
_norm_relu_kernel_v2 = _make_norm_kernel_v2(NUM_GROUPS, GROUP_SIZE, NUM_CHANNELS, relu=True)


def metal_groupnorm_v2(
    x: mx.array,
    weight: mx.array,
    bias: mx.array,
    num_groups: int = NUM_GROUPS,
    relu: bool = False,
    threads_per_tg: int = 1024,
    chunks_per_group: int = 32,
) -> mx.array:
    """GroupNorm via multi-threadgroup stats + parallel normalize."""
    B, H, W, C = x.shape

    w32 = weight.astype(mx.float32) if weight.dtype != mx.float32 else weight
    b32 = bias.astype(mx.float32) if bias.dtype != mx.float32 else bias

    # Dummy input to pass chunks_per_group to stats kernel via shape
    chunks_dummy = mx.zeros((chunks_per_group,))

    # Kernel A: stats reduction
    num_tg_stats = B * num_groups * chunks_per_group
    stats_shape = (B * num_groups * 2,)
    stats = _stats_kernel_v2(
        inputs=[x, chunks_dummy],
        template=[("T", x.dtype)],
        output_shapes=[stats_shape],
        output_dtypes=[mx.float32],
        grid=(num_tg_stats * threads_per_tg, 1, 1),
        threadgroup=(threads_per_tg, 1, 1),
        init_value=0,
    )[0]

    # Kernel B: normalize + affine
    total_elements = B * H * W * C
    # Round up to multiple of threadgroup size
    grid_size = ((total_elements + threads_per_tg - 1) // threads_per_tg) * threads_per_tg
    kern = _norm_relu_kernel_v2 if relu else _norm_kernel_v2
    out = kern(
        inputs=[x, stats, w32, b32],
        template=[("T", x.dtype)],
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
        grid=(grid_size, 1, 1),
        threadgroup=(threads_per_tg, 1, 1),
    )[0]

    return out


# ── Benchmark harness ───────────────────────────────────────────────────────

WARMUP = 5
BENCH = 20


def time_fn(fn, warmup=WARMUP, bench=BENCH, label=""):
    """Time a function, return median ms."""
    for _ in range(warmup):
        out = fn()
        # mx.eval is MLX array materialization (forces GPU), not Python eval
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
    print(f"  {label:50s}  median={median:7.2f}ms  p95={p95:7.2f}ms  min={times[0]:7.2f}ms")
    return median


def test_correctness(spatial: int, threads: int, dtype=mx.bfloat16):
    """Verify Metal GN matches nn.GroupNorm output."""
    print(f"\n=== Correctness Test @ {spatial}x{spatial}, {threads} threads ===")
    B, H, W, C = 1, spatial, spatial, NUM_CHANNELS
    G = NUM_GROUPS

    mx.random.seed(42)
    x = mx.random.normal((B, H, W, C)).astype(dtype)
    # mx.eval is MLX array materialization, not Python eval
    mx.eval(x)  # noqa: S307

    # ── Reference: nn.GroupNorm ──
    gn = nn.GroupNorm(G, C, pytorch_compatible=True)
    # mx.eval is MLX array materialization, not Python eval
    mx.eval(gn.parameters())  # noqa: S307
    ref = gn(x).astype(dtype)

    # ── Test 1: unit weight/bias ──
    weight = mx.ones((C,))
    bias = mx.zeros((C,))
    # mx.eval is MLX array materialization, not Python eval
    mx.eval(weight, bias)  # noqa: S307

    metal = metal_groupnorm(x, weight, bias, G, threads_per_group=threads)
    # mx.eval is MLX array materialization, not Python eval
    mx.eval(ref, metal)  # noqa: S307

    ref_np = np.array(ref.astype(mx.float32))
    metal_np = np.array(metal.astype(mx.float32))
    max_abs = float(np.max(np.abs(ref_np - metal_np)))
    mean_abs = float(np.mean(np.abs(ref_np - metal_np)))
    print(f"  unit w/b:   max_abs={max_abs:.6e}  mean_abs={mean_abs:.6e}")

    # ── Test 2: random weight/bias ──
    weight2 = mx.random.normal((C,))
    bias2 = mx.random.normal((C,)) * 0.1
    # mx.eval is MLX array materialization, not Python eval
    mx.eval(weight2, bias2)  # noqa: S307

    gn2 = nn.GroupNorm(G, C, pytorch_compatible=True)
    gn2.weight = weight2
    gn2.bias = bias2
    ref2 = gn2(x).astype(dtype)
    metal2 = metal_groupnorm(x, weight2, bias2, G, threads_per_group=threads)
    # mx.eval is MLX array materialization, not Python eval
    mx.eval(ref2, metal2)  # noqa: S307

    ref2_np = np.array(ref2.astype(mx.float32))
    metal2_np = np.array(metal2.astype(mx.float32))
    max_abs2 = float(np.max(np.abs(ref2_np - metal2_np)))
    mean_abs2 = float(np.mean(np.abs(ref2_np - metal2_np)))
    print(f"  rand w/b:   max_abs={max_abs2:.6e}  mean_abs={mean_abs2:.6e}")

    # ── Test 3: with ReLU ──
    ref_relu = nn.relu(gn(x)).astype(dtype)
    metal_relu = metal_groupnorm(x, weight, bias, G, relu=True, threads_per_group=threads)
    # mx.eval is MLX array materialization, not Python eval
    mx.eval(ref_relu, metal_relu)  # noqa: S307

    ref_relu_np = np.array(ref_relu.astype(mx.float32))
    metal_relu_np = np.array(metal_relu.astype(mx.float32))
    max_abs_relu = float(np.max(np.abs(ref_relu_np - metal_relu_np)))
    print(f"  relu:       max_abs={max_abs_relu:.6e}")

    # ── Determinism check ──
    metal_a = metal_groupnorm(x, weight, bias, G, threads_per_group=threads)
    metal_b = metal_groupnorm(x, weight, bias, G, threads_per_group=threads)
    # mx.eval is MLX array materialization, not Python eval
    mx.eval(metal_a, metal_b)  # noqa: S307
    det_err = float(mx.max(mx.abs(metal_a - metal_b)))
    print(f"  determinism: {det_err:.6e} (0 = deterministic)")

    # Relaxed tolerance: bf16 affine precision differs between fp32-throughout
    # (our kernel) vs layer_norm-in-bf16-then-affine (reference). The pipeline
    # fidelity gate (golden.npz at 5e-3) is the real correctness check.
    tolerance = 0.1
    passed = max_abs < tolerance and max_abs2 < tolerance and max_abs_relu < tolerance
    det_ok = det_err == 0.0
    print(f"  PASS: {passed} (tol={tolerance})  DETERMINISTIC: {det_ok}")
    return passed and det_ok


def bench_comparison(spatial: int, threads: int, dtype=mx.bfloat16):
    """Benchmark Metal GN vs nn.GroupNorm."""
    print(f"\n=== Benchmark @ {spatial}x{spatial}, {threads} threads/group ===")
    B, H, W, C = 1, spatial, spatial, NUM_CHANNELS
    G = NUM_GROUPS

    mx.random.seed(42)
    x = mx.random.normal((B, H, W, C)).astype(dtype)
    weight = mx.ones((C,))
    bias = mx.zeros((C,))
    # mx.eval is MLX array materialization, not Python eval
    mx.eval(x, weight, bias)  # noqa: S307

    # nn.GroupNorm baselines
    gn = nn.GroupNorm(G, C, pytorch_compatible=True)
    # mx.eval is MLX array materialization, not Python eval
    mx.eval(gn.parameters())  # noqa: S307
    time_fn(lambda: gn(x), label="nn.GroupNorm (eager)")

    gn_compiled = mx.compile(gn.__call__)
    t_ref = time_fn(lambda: gn_compiled(x), label="nn.GroupNorm (compiled)")

    gn_relu_compiled = mx.compile(lambda xi: nn.relu(gn(xi)))
    t_ref_relu = time_fn(lambda: gn_relu_compiled(x), label="nn.GroupNorm + ReLU (compiled)")

    # Metal GN (shared memory, single kernel)
    t_metal = time_fn(
        lambda: metal_groupnorm(x, weight, bias, G, threads_per_group=threads),
        label=f"Metal GN (shared-mem, {threads} threads)",
    )

    t_metal_relu = time_fn(
        lambda: metal_groupnorm(x, weight, bias, G, relu=True, threads_per_group=threads),
        label=f"Metal GN + fused ReLU ({threads} threads)",
    )

    # V2: multi-threadgroup
    for cpg in [16, 32, 64]:
        t_v2 = time_fn(
            lambda c=cpg: metal_groupnorm_v2(
                x, weight, bias, G, threads_per_tg=1024, chunks_per_group=c
            ),
            label=f"Metal GN v2 (multi-tg, {cpg} chunks)",
        )

    t_v2_relu = time_fn(
        lambda: metal_groupnorm_v2(
            x, weight, bias, G, relu=True, threads_per_tg=1024, chunks_per_group=32
        ),
        label="Metal GN v2 + fused ReLU (32 chunks)",
    )

    print(f"\n  Summary:")
    print(f"    nn.GroupNorm compiled:          {t_ref:7.2f}ms")
    print(f"    nn.GroupNorm+ReLU compiled:     {t_ref_relu:7.2f}ms")
    print(f"    Metal GN v1 (shared-mem):       {t_metal:7.2f}ms")
    print(f"    Metal GN v1+ReLU:               {t_metal_relu:7.2f}ms")
    if t_ref > 0:
        delta = (t_metal / t_ref - 1) * 100
        print(f"    v1 vs compiled GN:             {delta:+.1f}%")

    return t_ref, t_metal


def sweep_threads(spatial: int, dtype=mx.bfloat16):
    """Sweep thread counts to find optimal configuration."""
    print(f"\n=== Thread Sweep @ {spatial}x{spatial} ===")
    B, H, W, C = 1, spatial, spatial, NUM_CHANNELS
    G = NUM_GROUPS

    mx.random.seed(42)
    x = mx.random.normal((B, H, W, C)).astype(dtype)
    weight = mx.ones((C,))
    bias = mx.zeros((C,))
    # mx.eval is MLX array materialization, not Python eval
    mx.eval(x, weight, bias)  # noqa: S307

    # Baseline
    gn = nn.GroupNorm(G, C, pytorch_compatible=True)
    # mx.eval is MLX array materialization, not Python eval
    mx.eval(gn.parameters())  # noqa: S307
    gn_compiled = mx.compile(gn.__call__)
    t_ref = time_fn(lambda: gn_compiled(x), label="nn.GroupNorm (compiled)")

    for threads in [32, 64, 128, 256, 512, 1024]:
        t = time_fn(
            lambda t=threads: metal_groupnorm(x, weight, bias, G, threads_per_group=t),
            label=f"Metal GN ({threads} threads)",
        )
        delta = (t / t_ref - 1) * 100
        print(f"    → {delta:+.1f}% vs compiled")


def main():
    parser = argparse.ArgumentParser(description="Metal GroupNorm v2: shared memory")
    parser.add_argument("--spatial", type=int, default=1024, help="Spatial dimension")
    parser.add_argument("--threads", type=int, default=256, help="Threads per threadgroup")
    parser.add_argument("--bench-only", action="store_true", help="Skip correctness test")
    parser.add_argument("--sweep", action="store_true", help="Sweep thread counts")
    args = parser.parse_args()

    print(f"MLX version: {mx.__version__}")
    print(f"Spatial: {args.spatial}x{args.spatial}")
    print(f"Groups: {NUM_GROUPS}, Channels: {NUM_CHANNELS}, GroupSize: {GROUP_SIZE}")
    elems_per_group = args.spatial * args.spatial * GROUP_SIZE
    print(f"Elements per group: {elems_per_group:,}")
    print(f"Approach: single kernel, shared-memory tree reduction, no atomics")

    if not args.bench_only:
        passed_small = test_correctness(64, args.threads)
        if not passed_small:
            print("\nFAILED at 64x64 — aborting")
            return

        if args.spatial != 64:
            passed = test_correctness(args.spatial, args.threads)
            if not passed:
                print(f"\nFAILED at {args.spatial}x{args.spatial}")
                return

    if args.sweep:
        sweep_threads(args.spatial)
    else:
        bench_comparison(args.spatial, args.threads)


if __name__ == "__main__":
    main()
