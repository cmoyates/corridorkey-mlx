# Deep Dive Findings — Breaking 120ms Plateau

Researched 2026-03-11. Sources: Gemini deep research report, ChatGPT M3 Max recommendations.

---

## CRITICAL: Dilated Convolutions Force im2col Fallback

**Source**: MLX PR #3147, deep dive section 3
**Classification**: mlx-portable — ROOT CAUSE of plateau
**Finding**: Dilated convolutions (dilation 1,2,4,8 in refiner) are EXCLUDED from MLX implicit GEMM. They fall back to explicit im2col which inflates activation memory by 9x (kernel_size^2). This explains:
- Why peak memory is 2143MB (im2col inflating refiner activations)
- Why cache-limit-256mb caused +15% regression (constant GC of massive im2col matrices)
- Why refiner fp16 failed fidelity (numerical instability + massive memory pressure)
**Impact**: This is likely THE dominant bottleneck. Fixing it requires either:
1. Replace dilated convs with stride-2 downsample + standard conv + bilinear upsample (ASPP restructuring)
2. Custom Metal kernel via mx.fast.metal_kernel (bypasses im2col entirely)
**Fidelity risk**: ASPP restructuring = HIGH (needs retraining). Custom kernel = LOW (same math).
**Expected impact**: 15-20% latency, 15-20% peak memory reduction

## HIGH: BF16 Full Pipeline (Backbone + Refiner)

**Source**: Deep dive section 4
**Classification**: mlx-portable
**Finding**: Model is memory-bandwidth bound, not compute-bound. Weight loading (180MB / 350GB/s = 0.5ms) is <1% of latency. The remaining 119.4ms is activation memory traffic. BF16 has same 8-bit exponent as FP32 (preserves dynamic range) but halves bandwidth. Previous FP16 refiner failed because FP16 has only 5 exponent bits — overflows near sigmoid saturation boundaries.
**Key insight**: BF16 backbone is safe because BF16 preserves the dynamic range that matters. Only final sigmoid needs FP32.
**Expected impact**: 20-30% latency reduction by halving activation traffic
**Fidelity risk**: LOW (BF16 prevents the overflow issues seen with FP16)

## HIGH: Hiera Unroll/Reroll Contiguity Audit

**Source**: Deep dive section 5
**Classification**: mlx-portable
**Finding**: Hiera's MaskUnitAttention uses 6D tensor unroll/reroll + transpose operations. MLX updates stride metadata (no physical copy) on transpose, but when the non-contiguous tensor hits a Linear/SDPA kernel, MLX inserts IMPLICIT memory copies to make it contiguous. Across 24 blocks, these hidden copies accumulate silently.
**Action**: Audit unroll/reroll to minimize non-contiguous-to-contiguous transitions. Reroll only when absolutely necessary (stage transitions with strided conv).
**Expected impact**: 8-12% latency
**Fidelity risk**: None

## MEDIUM: Dual-Stream Decoder Dispatch

**Source**: Deep dive section 6
**Classification**: mlx-portable
**Finding**: Alpha and foreground decoder heads are independent until final fusion. Can dispatch to parallel GPU command queues via `with mx.stream(mx.gpu(stream_id)):`.
**Caveat**: Only works when NOT using compile_forward (mx.compile captures a single graph). Would need per-component compilation mode.
**Expected impact**: 3-5% latency
**Fidelity risk**: None

## MEDIUM: Metal Capture Profiling

**Source**: ChatGPT recommendations section 8, deep dive section 1
**Classification**: mlx-portable
**Finding**: Can use `mx.metal.start_capture()` + Xcode Metal debugger to get per-kernel timing. Would answer definitively: which kernels dominate? Is the GPU saturated or starving?
**Action**: Profile before next optimization round. This is the highest-ROI diagnostic step.
**Commands**: `MTL_CAPTURE_ENABLED=1` env var, `mx.metal.start_capture(path="trace.gputrace")`

## LOW: Backbone Structured Pruning

**Source**: Deep dive section 7
**Classification**: concept-only (needs training)
**Finding**: For green-screen matting, global semantic understanding (stages 3-4) is over-parameterized. Pruning 30% of attention heads in final 12 blocks + brief fine-tuning could give linear latency reduction.
**Expected impact**: ~30% latency reduction (proportional to pruned compute)
**Fidelity risk**: Medium (needs fine-tuning, out of scope)

## ELIMINATED by Deep Dive

- **CoreML/ANE**: ANE requires 5D max tensors. Hiera uses 6D unrolling. Would need complete architecture rewrite. Out of scope.
- **Dynamic resolution 384**: Already have decoupled resolution (Opt Phase 3). The deep dive confirms this is viable but needs quality validation with real samples.
- **C++ dispatch**: Eliminates Python overhead but mx.compile already handles most of this. Marginal gain for massive engineering cost.

## Revised Priority Order for Loop

1. **BF16 full pipeline** — backbone bf16 + refiner bf16 + fp32 sigmoid only (search: selective-precision)
2. **Dilated conv restructuring** — either ASPP or custom Metal kernel (search: fused-metal-kernels OR new area)
3. **Unroll/reroll contiguity audit** — minimize implicit copies in Hiera blocks (search: tensor-layout-staging)
4. **Metal capture profiling** — diagnostic, not an experiment (manual step)
5. **Dual-stream decoders** — parallel alpha+fg heads (search: stream-pipelining)
6. **Token routing** — skip 80%+ of attention for easy tokens (search: token-routing)
