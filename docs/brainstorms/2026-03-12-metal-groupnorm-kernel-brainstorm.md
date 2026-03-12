# Custom Metal GroupNorm Kernel — Brainstorm

**Date:** 2026-03-12
**Status:** CLOSED — approach definitively disproven
**Context:** Final investigation into whether custom Metal kernel can eliminate GroupNorm transpose overhead (50% of refiner time at 1024)

---

## What We Explored

Whether `mx.fast.metal_kernel()` can implement GroupNorm without the transpose+layer_norm pattern that `nn.GroupNorm(pytorch_compatible=True)` uses internally.

## Why This Approach

GroupNorm is 50% of refiner time at 1024 (~107ms of 215ms). The internal transpose produces non-contiguous views, causing Metal copy kernels. All Python-level optimization approaches were exhausted in exp 41 (ContiguousCopyGN, TransposedAffineGN, TwoPassFP32GN — all failed or regressed).

## Key Discovery: metal_kernel API Lacks Threadgroup Shared Memory

The `mx.fast.metal_kernel()` API does NOT expose threadgroup shared memory or barriers. It supports:
- `thread_position_in_grid` (element indexing)
- `simd_sum` (32-thread SIMD reduction)
- `atomic_fetch_add_explicit` + `atomic_outputs=True`
- Template types, custom headers

This eliminates single-pass GroupNorm kernels. The only viable approach within the API is simd_sum + atomic accumulation.

## Prototype Results

### Architecture
Two-kernel approach (no global barrier available):
- **Kernel A**: per-group sum + sum_sq via simd_sum -> atomic accumulate to fp32 stats buffer
- **Kernel B**: normalize + affine + optional ReLU, reading precomputed stats

Thread mapping restructured so consecutive threads stay within the same group (critical for correct simd_sum).

### Performance @ 1024x1024 (bf16, G=8, C=64)

| Method | Median | vs compiled GN |
|--------|--------|----------------|
| nn.GroupNorm (compiled) | 6.95ms | baseline |
| nn.GroupNorm + ReLU (compiled) | 8.01ms | baseline |
| Metal GN (2-kernel) | 9.82ms | **+41.3%** |
| Metal GN + fused ReLU (2-kernel) | 9.81ms | **+22.5%** |

### Correctness

| Spatial | Stats sum err | Stats sumsq err | Determinism | Unit w/b err | Rand w/b err |
|---------|--------------|-----------------|-------------|-------------|-------------|
| 64x64 | 9.2e-05 | 4.3e-02 | 1.2e-02 | 2.4e-04 | 6.3e-02 |
| 1024x1024 | 7.7e-02 | **315** | **2.0** | 7.6e-06 | 6.3e-02 |

At 1024: 262K atomic fp32 adds per group to the same 2 addresses. The accumulation is **non-deterministic** (stats drift by 2.0 between identical runs) and **imprecise** (sumsq error of 315 vs reference).

### Why It Failed

1. **Atomic contention**: 262K adds to the same address serializes GPU threads. The contention overhead exceeds the transpose copy savings.
2. **Non-determinism**: `memory_order_relaxed` with fp32 (non-associative) addition means different thread scheduling = different accumulated value. Not acceptable for inference.
3. **Two dispatches**: No global barrier means stats computation and normalization must be separate kernel launches. Each dispatch has overhead.
4. **Fused layer_norm is too good**: `mx.fast.layer_norm` is a single fused Metal kernel that computes mean+var+normalize in one pass over contiguous data with threadgroup shared memory (at C++ level). No combination of `mx.fast.metal_kernel` primitives can match it.

## Key Decisions

- **Custom Metal kernel for GroupNorm via `mx.fast.metal_kernel` is NOT viable** — API lacks threadgroup shared memory, and the simd_sum+atomics fallback is both slower and non-deterministic
- **GroupNorm cost is architectural** — it's the price of using the fused layer_norm kernel in NHWC layout with pytorch-compatible grouping semantics
- **This optimization line is closed** — all approaches exhausted (Python-level exp 41 + custom Metal prototype)

## Remaining Theoretical Paths (not pursued)

1. **Upstream MLX PR for `mx.fast.group_norm`** — proper C++/Metal implementation with threadgroup shared memory. Out of repo scope.
2. **Hierarchical reduction** — intermediate buffer to reduce atomics from 262K to ~1K. 3 dispatches total. Would fix determinism but likely still slower than transpose+layer_norm.
3. **Wait for MLX API expansion** — if `metal_kernel` adds threadgroup shared memory support in a future version.

## Files

- `scripts/test_metal_groupnorm.py` — prototype with both kernels + correctness/benchmark harness
