# Dilated Conv Kernel Experiment — REVERT

**Date:** 2026-03-11
**Verdict:** REVERT (all 3 approaches failed)
**Classification:** mlx-portable, fundamental-constraint

## Hypothesis

Replacing im2col-based dilated convolution with either a custom Metal kernel or sub-pixel
decomposition would reduce latency 15-20% and peak memory 15-20%.

## Approaches Tested

### 1. Naive Metal kernel (1 thread per output element)

- Each thread computes one output pixel: loops over 9 kernel positions × 64 input channels = 576 serial FMAs
- **Result: 228ms vs 122ms baseline (1.87x SLOWER)**
- Root cause: serial inner loop cannot compete with GEMM's matrix hardware (AMX)

### 2. SIMD-cooperative Metal kernel (32 threads per output element)

- SIMD group splits C_IN accumulation across 32 lanes, uses `simd_sum` to reduce
- **Result: 342ms vs 122ms baseline (2.8x SLOWER)**
- Root cause: 32x more threads launched, each doing minimal work. Dispatch overhead dominates.

### 3. Sub-pixel decomposition (mathematically exact)

- Dilated conv(D) = D² standard convs on (H/D, W/D) sub-images, interleaved back
- Standard convs qualify for implicit GEMM (no im2col)
- Parity: **exact zero error** (identical computation, reordered)
- **Result: 126ms / 2336MB vs 120ms / 2143MB baseline**
- Root cause: reshape + transpose create intermediate copies that exceed im2col memory savings

## Key Learning

**The im2col memory inflation (9x) is the cost of using Apple's AMX matrix hardware.**

im2col + GEMM is not a "fallback" — it's a deliberately chosen computation strategy that
maps convolution to matrix multiplication, enabling hardware acceleration. Any approach that
bypasses this (custom kernels, decomposition) also bypasses AMX.

For C_in=C_out=64 with 3x3 kernels: the GEMM is compute-bound and fully saturates AMX.
The 9x memory overhead is real but unavoidable at this channel count.

## Implications

- **The refiner dilated conv is NOT optimizable via alternative dispatch paths.** The 15-20%
  improvement estimate from deep_dive_findings.md was based on the assumption that im2col
  was pure overhead. In reality, it's a necessary step to access the fast path.
- **Memory optimization must come from other directions:** tiled refiner inference (backbone once,
  tile refiner spatially), or reducing the number of refiner blocks.
- **The 2143MB peak memory baseline may be near-optimal** for this model architecture at 512x512
  on MLX, given the hardware constraint.
- **Custom Metal kernels via `mx.fast.metal_kernel()` are viable for element-wise and gather
  operations but NOT for compute-intensive operations (matmul, conv)** where AMX dominates.

## Positive Findings

- `mx.fast.metal_kernel()` works correctly inside `mx.compile()` — confirmed compatible
- Sub-pixel decomposition is mathematically exact (zero error) — useful technique if memory
  pressure becomes critical at higher resolutions
- Metal kernel parity with nn.Conv2d: max_abs 2.5e-5 — numerically faithful

## What NOT to Try Next

- Fused conv+GroupNorm+ReLU Metal kernel — same AMX constraint applies to the conv portion
- Winograd/FFT-based conv — MLX likely already uses these internally when beneficial
- Explicit im2col with smaller tiles — im2col IS the fast path, not the bottleneck
