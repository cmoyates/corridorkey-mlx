# Metal GPU Trace Findings — 512x512

Captured 2026-03-12 on M3 Max. Trace: `trace.gputrace`.

## Summary

- **Effective GPU time**: 109.5ms @ 512x512
- **Peak memory**: 1.85 GiB
- **GPU commands**: 644
- **Encoders**: 27
- **Bandwidth**: peaks ~353 GB/s (near M3 Max theoretical ~400 GB/s)

## Top Shaders by Cost

| Cost   | Kernel                          | Component              |
|--------|---------------------------------|------------------------|
| 22.73% | implicit_gemm_conv_2d_float     | Conv2d (refiner dilated convs) |
| 10.96% | steel_gemm_fused_nt_float       | Matmul (backbone attention/linear) |
| 9.88%  | gatherbfloat16uint32_3_3_int    | Gather (Hiera unroll/reroll) |
| 7.30%  | g3_copyfloat16float16           | Memory copies (layout transforms) |
| 5.78%  | layer_norm_looped_float16       | LayerNorm/GroupNorm    |
| 5.63%  | steel_gemm_splitk_nt_float3     | Matmul variant         |

## Component Timing Breakdown

| Component       | Time   | % of total | Notes                          |
|-----------------|--------|------------|--------------------------------|
| Backbone (Hiera)| ~20ms  | ~18%       | Dense, well-pipelined          |
| Decoder         | ~5ms   | ~5%        | Linear projections + upsample  |
| Refiner         | ~35ms  | ~32%       | Dilated convs dominate         |
| Overhead/copies | ~50ms  | ~45%       | Copies, dispatch gaps, gathers |

## Key Insights

1. **Backbone is efficient** — tightly packed gemm kernels, minimal gaps, no optimization needed
2. **Refiner is the #1 target** — dilated conv kernels (implicit_gemm) are the single largest cost
3. **Gather ops are expensive** — 9.88% for Hiera stage-transition unroll/reroll, serialized
4. **Copy ops are pure waste** — 7.3% on g3_copy (likely NCHW/NHWC transposes)
5. **CPU dispatch gaps visible** — especially in decoder/refiner phase, gaps between encoder blocks
6. **Memory-bandwidth bound** — 353/400 GB/s means compute opts (quantization) help less than traffic reduction
7. **Refiner pattern**: repeating `implicit_gemm (dilated conv) → layer_norm (GroupNorm)` per ResBlock

## Priority Queue (trace-informed)

### Eliminated
- **Half-resolution refiner** — DEAD. 10x REFINER_SCALE amplifies interpolation error through sigmoid; max_abs ~0.98 at both 0.5x and 0.75x
- **Gather ops (9.25%)** — DEAD. Already optimized form (replaced 3x reshape-transpose-reshape chains). Would require restructuring Hiera attention.
- **GroupNorm → fast.layer_norm (9.03%)** — DEAD. pytorch_compatible=True path already delegates to mx.fast.layer_norm internally. The 9% is the actual fused kernel cost.

### Active
1. **Eliminate copy ops (6.94%)** — g3_copyfloat16float16, 8.4M SIMD groups. Likely from GroupNorm internal transpose producing non-contiguous views before conv2d. Hard to eliminate without rewriting GroupNorm.
2. **mx.export_function** — AOT compiled graph, skip Python graph-building overhead
3. **block_softmax_float32 (3.95%)** — likely internal to SDPA, not separately optimizable
4. **Broadcast add (2.99%)** — residual connections, 4.2M SIMD groups

### Also eliminated
- **Refiner bf16 match decoder** — fp16 is ~1.9% faster than bf16 on M3 Max (better fp16 throughput). Copy ops are NOT from dtype mismatch.
- **compile_forward=True** — 15.6% SLOWER than per-component compile. Monolithic graph holds all buffers simultaneously; stage_gc sync points are beneficial for memory pressure.
- **CPU dispatch gaps** — these are inter-compile boundaries between backbone/decoder/refiner. compile_forward eliminates them but increases memory pressure, net negative.
- **mx.export_function** — 20.5% SLOWER. Same root cause as compile_forward: monolithic graph holds all buffers, no stage_gc sync points for buffer reuse. The per-component compile + stage_gc strategy is genuinely optimal for this model's memory profile.
