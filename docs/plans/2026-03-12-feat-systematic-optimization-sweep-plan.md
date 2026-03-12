---
title: "feat: Systematic Optimization Sweep — All Untried Ideas"
type: feat
date: 2026-03-12
---

# Systematic Optimization Sweep

## Overview

Execute every untried optimization idea identified across 28 experiments, 12 compound findings, 6 brainstorms, 5 deep research docs, and upstream mining. Organized in 4 tiers by risk/effort. Each experiment follows lab protocol: plan, implement, benchmark, score, keep/revert, record.

## Baselines

| Res | Median | p95 | Peak Memory | Best Experiment |
|-----|--------|-----|-------------|-----------------|
| 512 | 117ms | 118ms | 2143MB | qkv-split-first-windowed-attn |
| 1024 | 422ms | 423ms | 3319MB | decoder-bf16-weights-load-time |

## Fidelity Budget

| Tensor | @512 max_abs | @1024 max_abs | Threshold | Headroom @1024 |
|--------|-------------|---------------|-----------|----------------|
| alpha_final | 0.00013 | 0.035 | 0.050 | 30% |
| fg_final | 0.00014 | 0.043 | 0.050 | **14%** |

**Critical**: fg_final @1024 has only 14% headroom. Any experiment adding fidelity error must be tested at 1024 first.

---

## Phase 1: Zero-Risk Sweeps (< 1 hour total, no code changes)

### ~~Exp 29: MLX_MAX_MB_PER_BUFFER + MLX_MAX_OPS_PER_BUFFER Sweep~~ ✅ DONE

**Hypothesis**: Default buffer limits cause premature command buffer splits, adding dispatch overhead.

**Protocol**:
```bash
# Baseline
uv run python scripts/run_research_experiment.py --name env-baseline

# MB sweep
for MB in 16 32 64 128 256 512 1024 1000000; do
  MLX_MAX_MB_PER_BUFFER=$MB uv run python scripts/bench_mlx.py
done

# OPS sweep at best MB
for OPS in 2 4 8 16 32; do
  MLX_MAX_MB_PER_BUFFER=BEST MLX_MAX_OPS_PER_BUFFER=$OPS uv run python scripts/bench_mlx.py
done
```

**Files**: None modified
**Rollback**: N/A (env vars only)
**Time**: 30 min

### ~~Exp 30: mx.set_wired_limit() Sweep~~ ✅ DONE

**Hypothesis**: Pinning model weights in physical RAM reduces p95 variance and may improve steady-state latency.

**Result**: REVERT — No benefit. Higher limits increase latency (+5-10%), peak memory (+500-700MB), and variance. Default (0/disabled) is optimal. Unified memory on Apple Silicon doesn't benefit from wired limits.

**Protocol**:
```python
# In bench script or smoke_engine.py
import mlx.core as mx
for limit_mb in [0, 512, 1024, 1536, 2048, 3072, 4096]:
    mx.set_wired_limit(limit_mb * 1024 * 1024)
    # benchmark ...
```

**Files**: None permanently modified (benchmark wrapper only)
**Rollback**: N/A
**Time**: 15 min

### ~~Exp 31: Fidelity Budget Audit~~ ✅ DONE

**Hypothesis**: One specific bf16 conversion consumes disproportionate fidelity budget. Identifying it allows targeted revert for headroom while keeping other gains.

**Result**: Confirmed. Current fg_final headroom is only 8.2% (critical). Error breakdown:
- **Backbone bf16** → 21.3% of fg_final error. Disabling raises fg headroom to 27.8%. Best lever if headroom becomes blocking.
- **Refiner bf16** → 52.2% of alpha_final error, only 6.5% of fg_final. Safe to keep.
- **Decoder bf16** → 5.0% alpha, 7.6% fg. Minor contributor.
- **All fp32** → 95%+ headroom (0.002/0.006 max_abs). Floor reference.

**Protocol**:
1. Bisect bf16 conversions: revert each individually, measure max_abs_error @1024
2. Candidates: backbone stages 1-3 bf16 (exp 27), refiner bf16 (exp 2), decoder bf16 weights (exp 28), deferred fp32 cast (exp 17), bf16 sigmoid (exp 18)
3. Rank by error contribution

**Files**: Temporary modifications to `greenformer.py` dtype casts
**Rollback**: Restore current state after audit
**Time**: 1-2 hours (5 benchmark runs)

---

## Phase 2: Low-Code, High-Confidence (1-4 hours each)

### ~~Exp 32: GroupNorm pytorch_compatible=False + Weight Remapping~~ ❌ FAILED

**Hypothesis**: Dropping `pytorch_compatible=True` eliminates 10 transpose+copy kernels in refiner (6.94% GPU time, ~8ms @512, ~29ms @1024).

**Result**: REVERT — Catastrophic fidelity failure (alpha=0.987, fg=0.973). Native MLX GroupNorm computes different normalization statistics (non-zero group means). Not a weight permutation issue — fundamentally different reduction axes. `pytorch_compatible=True` is required for correctness.

**Details**: See `docs/brainstorms/2026-03-12-mlx-native-kernel-optimizations-brainstorm.md` Phase A2.

**Protocol**:
1. Run `scripts/test_groupnorm_parity.py` to compare modes for C=64, G=8
2. If outputs differ: compute weight permutation, modify converter
3. Modify `src/corridorkey_mlx/model/refiner.py`: remove `pytorch_compatible=True`
4. If needed: modify `src/corridorkey_mlx/convert/converter.py` for weight remap
5. Run `scripts/compare_reference.py` + `scripts/run_research_experiment.py`

**Files**: `refiner.py`, possibly `converter.py`
**Target**: 5-7% latency reduction
**Risk**: LOW
**Time**: 2-4 hours

### ~~Exp 33: 1x1 Conv2d to nn.Linear in Decoder~~ ⏭️ ALREADY DONE

**Hypothesis**: MLX `conv_general` is 3-5x slower than MPS (issue #1409). Decoder 1x1 conv fusion can use `nn.Linear` + reshape instead.

**Result**: Already implemented. `fold_bn()` precomputes 2D weights; forward uses `@ .T` and `mx.addmm` — no `conv2d` dispatch in hot path.

**Protocol**:
1. Identify 1x1 conv layers in `src/corridorkey_mlx/model/decoder.py`
2. Replace with `nn.Linear` + reshape (NHWC: flatten spatial, linear, unflatten)
3. Load weights: 1x1 conv weight `(O,1,1,I)` to Linear weight `(O,I)` by squeezing spatial dims
4. Fidelity check + benchmark

**Files**: `decoder.py`, possibly `converter.py`
**Target**: 5-15% decoder latency improvement
**Risk**: LOW (mathematically equivalent)
**Time**: 2-3 hours

### ~~Exp 34: Phased Model Deletion (del backbone after features)~~ ❌ FAILED

**Hypothesis**: Deleting backbone weights after feature extraction frees ~500MB before decoder/refiner run. Different from exp001 (which deleted features, not backbone).

**Result**: REVERT — Zero savings. Peak memory (3319.1MB) is set during backbone forward pass (attention intermediates), not from weight coexistence. Deleting weights afterward doesn't lower the high-water mark.

**Protocol**:
1. After `features = self.backbone(x)`, insert `del self.backbone; mx.clear_cache()`
2. Benchmark peak memory reduction
3. Caveat: breaks subsequent calls. Only viable for single-shot inference or lazy reload

**Files**: `greenformer.py` (inference path)
**Target**: ~500MB peak memory reduction
**Risk**: LOW for memory, but breaks multi-call usage
**Time**: 1 hour

### ~~Exp 35: Edge-Aware Tile Blend Weights~~ ⏭️ ALREADY DONE

**Hypothesis**: Current linear ramp blending applies to image boundaries, causing alpha darkening. Only ramp edges overlapping adjacent tiles; full weight at image boundaries.

**Result**: Already implemented. `_make_blend_weights_2d()` position tuple uses `yi > 0` / `xi > 0` to detect neighbors — logically equivalent to EZ-CorridorKey's `at_top`/`at_left` boundary flags. Image boundary edges get full weight, internal overlaps get linear ramps. Test `test_corner_tile` confirms `w[0,0]==1.0` at boundary.

**Files**: `inference/tiling.py`
**Target**: Quality fix (no speed change expected)
**Risk**: LOW
**Time**: N/A

---

## Phase 3: Medium Effort, Speculative (2-8 hours each)

### Exp 36: Refiner-Only Tiling at >1024

**Hypothesis**: At 2K/4K, run backbone+decoder once at full res, tile only the CNN refiner. Lower peak memory than full-model tiling.

**Protocol**:
1. Implement `TiledCNNRefiner` wrapper (reference: CorridorKey-Engine impl)
2. In `greenformer.py` inference: backbone at full res, decoder at full res, tile refiner only
3. Benchmark at 2048 with tile sizes 256, 384, 512

**Files**: New class in `inference/tiled.py`, modify `greenformer.py`
**Target**: Enable 2K+ inference with bounded memory
**Risk**: MEDIUM (refiner blending at tile boundaries needs care)
**Time**: 4-6 hours

### ~~Exp 37: GEMM Tile Shape Alignment (Weight Pre-Packing)~~ ❌ FAILED

**Hypothesis**: Metal GEMM dispatch selects between `steel_gemm_fused_nt` and `steel_gemm_splitk_nt` based on matrix dims. Padding weight matrices to multiples of 32/64 may trigger faster dispatch.

**Result**: REVERT — 1.9% latency regression (430.58ms vs 422.46ms). Padded stage 0 K-dim 112→128 for 6 matmuls (QKV+proj+fc1 in blocks 0-1). Runtime `mx.pad` overhead negates any GEMM tile benefit. Stage 0 is only 2 of 24 blocks — too small to offset padding cost. +140MB peak memory from padded weight copies.

**Protocol**:
1. Profile current GEMM kernel selections via Metal trace
2. Identify layers with non-aligned dims (backbone stage 0: dim=112)
3. Pad weights at load time, adjust slicing after matmul
4. Benchmark

**Files**: Weight loading in `greenformer.py`
**Target**: Unknown (speculative)
**Risk**: MEDIUM (padding overhead may negate gains)
**Time**: 3-4 hours

### ~~Exp 38: Interleaved Backbone to Decoder Pipeline~~ ❌ DISPROVEN

**Hypothesis**: Feed stage 0 features to decoder projection immediately while backbone continues stages 1-3. Overlaps backbone compute with decoder feature projection.

**Result**: DISPROVEN via micro-benchmark. MLX streams show ZERO GPU-GPU overlap on Apple Silicon. Decoder-scale: 0.979x, backbone-scale: 0.990x, 4-matmul/4-stream: 0.961x. Streams add 2-4% dispatch overhead. Apple Silicon command queues don't parallelize compute workloads — only CPU-GPU overlap is possible (already exploited via async_eval).

**Protocol**:
1. Refactor backbone to yield features per-stage instead of collecting all 4
2. Start decoder feature projection as backbone stages complete
3. Use `mx.async_eval()` at stage boundaries
4. Benchmark latency overlap

**Files**: `hiera.py` (feature emission), `decoder.py` (incremental processing), `greenformer.py` (pipeline)
**Target**: Unknown (depends on compute/memory overlap)
**Risk**: MEDIUM-HIGH (complex refactor, may break mx.compile)
**Time**: 6-8 hours

### ~~Exp 39: addmm Fused Linear Retry~~ ❌ FAILED

**Hypothesis**: Exp 11 (`addmm-fused-linear-backbone`) showed 2x regression, likely wrong implementation. `mx.addmm(bias, x, W.T)` should be faster than `x @ W.T + bias`.

**Result**: REVERT — 4.3% regression (440.81ms vs 422.46ms baseline). Replaced `x @ _fc_wt + bias` with `mx.addmm(bias, x, _fc_wt)` in all 24 Hiera MLP blocks. MLX compiler already fuses contiguous matmul+add; explicit addmm adds dispatch overhead. Confirms exp 11 finding was real, not a bug.

**Files**: `hiera.py`
**Target**: ~2-5% per linear-heavy component
**Risk**: LOW
**Time**: 30 min

### ~~Exp 40: Unified bf16 Safetensors Checkpoint~~ ❌ FAILED

**Hypothesis**: Pre-converting weights to bf16 for layers that run in bf16 halves checkpoint load time and peak memory during loading.

**Result**: REVERT — Negligible savings. Decoder (5.6MB) + refiner (1.2MB) are tiny vs backbone (392MB). Only 3.4MB saved (0.8% of checkpoint). Load time/peak memory unchanged. Switched load path to `mx.load` (bf16-native) as minor infra improvement.

**Files**: `converter.py`, `corridorkey.py` (load path)
**Target**: ~50% faster load, ~200-300MB less peak during load
**Risk**: LOW
**Time**: 45 min

---

## Phase 4: Waiting on Upstream / High Effort

### Monitor: MLX PR #3120 (Split-K Quantized Matmul)

**Status**: Approved, awaiting merge
**Impact**: 25-30% faster quantized linear layers
**Action**: Bump MLX version when merged, re-benchmark quantized backbone

### Monitor: MLX PR #3026 (Quantized SDPA)

**Status**: Under review
**Impact**: ~40% faster attention with quantized KV
**Action**: Once merged, test with quantized backbone attention

### Future: PTQ4VM-Style Calibrated 4-bit Quantization

**Paper**: arXiv:2506.10840
**Impact**: ~50% backbone latency reduction
**Blocker**: Need calibration pipeline + dataset
**Effort**: HIGH (1-2 weeks)

---

## Execution Order

```
Phase 1 (day 1, morning):
  29: env var sweep (30 min)
  30: wired_limit sweep (15 min)
  31: fidelity audit (1-2 hr)

Phase 2 (day 1 afternoon + day 2):
  32: GroupNorm remap (2-4 hr)
  33: 1x1 conv to Linear (2-3 hr)
  34: del backbone (1 hr)
  35: edge-aware blend (1-2 hr)

Phase 3 (days 3-5):
  39: addmm retry (2-3 hr) -- quick win if exp 11 was just buggy
  40: bf16 checkpoint (2-3 hr)
  37: GEMM alignment (3-4 hr)
  36: refiner-only tiling (4-6 hr)
  38: interleaved pipeline (6-8 hr) -- last, most complex

Phase 4: async -- monitor PRs, act when merged
```

## Success Metrics

- **Primary**: Latency reduction @1024 (current: 422ms)
- **Secondary**: Peak memory reduction @1024 (current: 3319MB)
- **Constraint**: fg_final max_abs_error < 0.050 @1024

## Recording

Each experiment produces entry in `research/experiments.jsonl` + compound note if reusable finding.

## Unresolved Questions

- Does nn.Linear already use matmul internally (making exp 33 redundant)?
- GroupNorm C=64 G=8: is channel grouping actually different between pytorch_compatible modes?
- mx.set_wired_limit: what's the max safe value on user's machine?
- addmm: what exactly went wrong in exp 11?
- Interleaved pipeline: does yielding per-stage break mx.compile graph boundaries?
- Fidelity headroom: if audit reveals one conversion dominates, is it worth reverting for safety margin?
