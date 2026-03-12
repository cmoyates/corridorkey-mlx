# Exp 41: Contiguous GroupNorm Variants

**Date**: 2026-03-12
**Verdict**: REVERT (all variants) — no improvement over baseline
**Experiment tag**: `contiguous-groupnorm-variants`

## Hypothesis

GroupNorm's internal transpose produces non-contiguous output, causing Metal copy kernels. Eliminating or mitigating the transpose should reduce refiner latency.

## What was tested

### Micro-benchmark: Isolated GN at 1024x1024, bf16, G=8, C=64

| Variant | Compiled median | vs baseline |
|---------|----------------|-------------|
| nn.GroupNorm (baseline) | 7.57ms | — |
| ContiguousCopyGN (+ 0.0) | 7.46ms | -1.5% (noise) |
| TransposedAffineGN | 9.12ms | +20.5% |
| TwoPassFP32GN (no transpose) | 36.85ms | +387% |

### Full refiner at 1024x1024, bf16, compiled

| Variant | Compiled median | vs baseline |
|---------|----------------|-------------|
| Baseline (nn.GroupNorm) | 215.8ms | — |
| ContiguousCopyGN | 215.6ms | -0.1% (noise) |
| TransposedAffineGN | 234.8ms | +8.8% |
| TwoPassFP32GN | 475.5ms | +120% |

### Parity (isolated GN vs nn.GroupNorm)

| Variant | max_abs_diff |
|---------|-------------|
| ContiguousCopyGN | 0.00e+00 (exact) |
| TransposedAffineGN | 0.00e+00 (exact) |
| TwoPassFP32GN | 4.88e-04 (within gate) |

## Variant descriptions

1. **ContiguousCopyGN**: Wraps standard GN, adds `+ 0.0` to force contiguous output. Idea: prevent downstream conv from triggering the copy. Result: no effect — mx.compile already handles this optimally.

2. **TransposedAffineGN**: Applies affine (weight*x+bias) in the transposed layout (B, G, HW, gs) before the second transpose. Saves one operation but the affine broadcast in 4D is slower than the standard 1D affine.

3. **TwoPassFP32GN**: Computes mean/var via mx.mean/mx.var with fp32 accumulation, no transpose at all. Eliminates both transposes but mx.mean/mx.var over axes (1,3) of a 4D tensor is 4.9x slower than mx.fast.layer_norm over the last dim of a 3D tensor.

## Key finding: GN is 50% of refiner time

From the first benchmark (`bench_groupnorm.py`):
- Refiner with GN: 214.8ms compiled
- Refiner without GN (identity): 107.4ms compiled
- **GN contribution: ~107ms = 50% of refiner time**

This is much larger than the 6.94% figure from the 512 Metal trace. At 1024, GN dominates because the reduction scales quadratically with spatial size (1024^2 vs 512^2 = 4x more elements per group).

## Why no variant helped

1. **mx.fast.layer_norm is irreplaceable**: It's a fused Metal kernel that computes mean+var+normalize in one pass over contiguous data. No combination of mx.mean + mx.var + element-wise ops can match it because each is a separate dispatch.

2. **The transposes ARE the cost of using layer_norm**: The data must be rearranged so each group's elements are contiguous in the last dim. There's no way around this in NHWC layout with pytorch-compatible grouping semantics.

3. **mx.compile already optimizes the copy**: Adding explicit contiguous hints doesn't help because the compiler already schedules the copy optimally.

4. **The affine is cheap**: Moving it before/after the transpose doesn't save enough to offset the broadcast overhead in 4D.

## Remaining path: custom Metal kernel

The only approach not tested is a custom Metal kernel via `mx.fast.metal_kernel()` that does the entire GroupNorm (mean, var, normalize, affine, optional relu) in one fused pass without any transpose. Challenges:
- 8M element reduction per group requires threadgroup shared memory + barriers
- `mx.fast.metal_kernel()` reduction capability unproven at this scale
- Custom kernel ops are NOT fusable by mx.compile — 9 fusion barriers could offset gains
- Must accumulate in fp32 (bf16 reduction of 8M elements = catastrophic precision loss)

Estimated effort: high. Estimated probability of net improvement: low-medium.

## Files

- `scripts/bench_groupnorm.py` — isolated GN + refiner with/without GN benchmark
- `scripts/bench_groupnorm_variants.py` — variant comparison benchmark
