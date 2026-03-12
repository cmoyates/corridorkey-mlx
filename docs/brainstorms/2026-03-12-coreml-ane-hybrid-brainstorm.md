# Plan B: CoreML/ANE Hybrid Execution

**Date**: 2026-03-12
**Status**: DEFERRED (try Plan A first)
**Target**: Eliminate unroll/reroll entirely via ANE's 5D relayed partitioning
**Expected gain**: 10-20% latency reduction (speculative)

## What We're Building

Split inference: Hiera backbone on Apple Neural Engine (CoreML), decoder + refiner on GPU (MLX). Exploit ANE's compiler-level window partitioning to eliminate physical token permutations.

## Why This Approach

- Apple's own research shows 5D relayed partitioning eliminates unroll/reroll at zero memory cost
- SAM-2 Hiera-Large successfully ported to ANE (proven precedent)
- ANE + GPU can pipeline (frame N refiner on GPU while frame N+1 backbone on ANE)
- Targets the 9.25% gather overhead that's hardest to fix in MLX

## Why It's Deferred

- Currently out of scope per CLAUDE.md ("CoreML/ANE = out of scope")
- No coremltools dependency exists
- Significant engineering lift (1-2 weeks)
- Interop overhead (CoreML NCHW <-> MLX NHWC) could negate gains
- Deep research suggests only 5-10% net speedup after overhead
- Plan A targets same bottlenecks with lower risk

---

## Architecture

```
Frame N:
  [ANE] Hiera backbone (512x512 -> 4 feature maps)
     |  zero-copy via unified memory
     v
  [GPU/MLX] Decoder heads (features -> alpha/fg logits)
     |
  [GPU/MLX] Refiner (RGB + coarse -> final alpha/fg)

Frame N+1 (pipelined):
  [ANE] Hiera backbone (overlapped with GPU refiner of frame N)
```

### Key Technical Details

- **CoreML export**: PyTorch checkpoint -> coremltools -> .mlpackage
  - NOT from MLX (no direct MLX->CoreML path)
  - Must use original PyTorch Hiera from timm
  - Export at fixed img_size (separate model per resolution)

- **5D Relayed Partitioning** (Apple research):
  - CoreML compiler maps window partition as virtual stride (no physical permutation)
  - Replaces unroll: `(B, H, W, C)` -> 6D tensor -> 5D relay -> windowed attention
  - Zero intermediate memory allocation

- **Interop bridge**:
  - CoreML outputs NCHW numpy arrays
  - Convert to MLX NHWC: `mx.array(np.transpose(out, (0, 2, 3, 1)))`
  - Or use `ct.predict()` with MLMultiArray -> shared memory

---

## Implementation Phases

### Phase B1: Feasibility Proof (2-3 days)

1. Add `coremltools` to dev dependencies
2. Load PyTorch Hiera from timm (same checkpoint, `features_only=True`)
3. Export backbone-only to CoreML with `ct.convert()`
4. Run CoreML backbone on sample input
5. Compare outputs to MLX backbone outputs (parity check)
6. Measure ANE vs GPU latency for backbone alone

**Go/no-go gate**: If backbone-only CoreML is >50ms @512 on ANE, abandon.

### Phase B2: Interop Layer (2-3 days)

1. Build `CoreMLBackbone` wrapper: loads .mlpackage, runs predict
2. Build bridge: CoreML NCHW outputs -> MLX NHWC arrays
3. Modify GreenFormer to accept external backbone features
4. Test: CoreML backbone -> MLX decoder/refiner -> compare to all-MLX path
5. Parity gate: max_abs_error < 5e-3 on all outputs

### Phase B3: Pipeline + Benchmark (2-3 days)

1. Implement frame pipelining (ANE backbone + GPU refiner overlap)
2. Full benchmark suite: latency, p95, peak memory
3. Compare to all-MLX baseline
4. Score via existing scoring infrastructure

### Phase B4: Multi-Resolution Support (1-2 days)

1. Export CoreML models at 512 and 1024
2. Resolution selection in pipeline
3. Pos_embed handling (baked into CoreML model at export time)

---

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Interop overhead negates gains | HIGH | Measure backbone-only first (Phase B1) |
| CoreML export fails on Hiera ops | HIGH | Use timm's PyTorch Hiera, not our MLX port |
| ANE falls back to GPU silently | MEDIUM | Check `computeUnits` logs, use `.cpuAndNeuralEngine` |
| Parity regression from NCHW<->NHWC | MEDIUM | Strict golden.npz comparison |
| Dilated conv inefficient on ANE | LOW | Refiner stays on GPU regardless |
| Per-resolution models bloat disk | LOW | Only 2 models (512, 1024) |

## Prerequisites

- Scope approval to add CoreML/ANE work
- PyTorch + timm available for export (not needed at runtime)
- coremltools >= 8.0

## Open Questions

- Does `ct.predict()` on M3 Max actually use ANE for Hiera, or fall back to GPU?
- Zero-copy CoreML -> MLX: does IOSurface sharing work through coremltools Python API?
- 5D relayed partition: does coremltools auto-apply this, or manual op construction needed?
- ANE throughput for Hiera base_plus (24 blocks, 112-896 dims): any benchmarks?
- Pipeline overlap: can CoreML predict() run async while MLX uses GPU?
