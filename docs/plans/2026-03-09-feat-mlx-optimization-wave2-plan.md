---
title: "feat: MLX optimization wave 2 — SDPA, stage GC, slim forward"
type: feat
date: 2026-03-09
---

# MLX Optimization Wave 2

Second round of optimizations informed by PyTorch CorridorKey forks ([PR #104](https://github.com/nikopueringer/CorridorKey/pull/104), [Raiden129/CorridorKey_Test](https://github.com/Raiden129/CorridorKey_Test)) and [deep research on MLX ViT optimization](../MLX%20Vision%20Transformer%20Optimization%20Techniques.md).

**Brainstorm:** `docs/brainstorms/2026-03-09-pytorch-optimization-learnings-brainstorm.md`
**Wave 1 plan:** `docs/plans/2026-03-08-feat-mlx-memory-optimizations-plan.md` (bf16, fused decode, tiled GC — all shipped)

## Overview

Six targeted optimizations, ordered by effort/impact ratio. Wave 1 achieved 12x peak memory reduction via tiled inference + deterministic GC. Wave 2 focuses on: fixing cleanup gaps, reducing computation graph size, and migrating to optimized kernels.

## Technical Approach

### Phase 1: Engine Cleanup Fixes (trivial, 15 min)

**Item 6: Add `mx.clear_cache()` to engine cleanup**

`engine.py:210` does `gc.collect()` but never calls `mx.clear_cache()`. The tiling module already does this (`tiling.py:164`), showing the pattern is known but wasn't applied to the non-tiled cleanup.

**Critical placement detail:** `mx.clear_cache()` must go **after postprocessing**, not at line 210. At line 210, `alpha_out`/`fg_out` are still live MLX arrays used by `postprocess_alpha/foreground` (which calls `np.array()` to transfer to CPU). Clearing cache while those references exist is pointless.

```python
# engine.py — after postprocessing extracts numpy arrays (~line 225)
del alpha_out, fg_out
gc.collect()
mx.clear_cache()  # NEW: release Metal buffer cache
```

**Acceptance criteria:**
- [x] `mx.clear_cache()` called after all MLX arrays are consumed by postprocessing
- [x] `del` references to MLX arrays before cache clear
- [x] All existing tests pass (`uv run pytest`)

**Files:** `engine.py`

---

### Phase 2: Slim Forward Mode (low effort, 30 min)

**Item 3: Skip returning unused intermediate tensors**

`GreenFormer.forward()` (`corridorkey.py:103-113`) returns 9 dict keys. The engine uses at most 4 (`alpha_coarse`, `fg_coarse`, `alpha_final`, `fg_final`). The unused 5 (`alpha_logits`, `fg_logits`, `alpha_logits_up`, `fg_logits_up`, `delta_logits`) hold references that prevent MLX from freeing underlying buffers.

**Important clarification:** This is about **reference lifetime**, not computation skipping. All intermediates must still be computed to produce the final outputs. The savings come from not keeping references in the returned dict, allowing MLX to reclaim buffers sooner.

**Design decision:** Add `slim: bool = False` parameter to `GreenFormer.__init__`, not `__call__`. This avoids mx.compile recompilation from a runtime boolean changing the output dict shape. The `pipeline.infer()` API always returns the full dict for debugging. Only the engine sets `slim=True`.

```python
# corridorkey.py
class GreenFormer(nn.Module):
    def __init__(self, ..., slim: bool = False):
        self.slim = slim
        ...

    def __call__(self, x):
        ...
        if self.slim:
            return {
                "alpha_coarse": alpha_coarse,
                "fg_coarse": fg_coarse,
                "alpha_final": alpha_final,
                "fg_final": fg_final,
            }
        return {  # full dict for debugging/testing
            "alpha_logits": ...,
            ...
        }
```

**Acceptance criteria:**
- [x] `slim=True` returns 4-key dict, `slim=False` returns full 9-key dict
- [x] Engine passes `slim=True` via `load_model()`
- [x] `pipeline.infer()` keeps `slim=False` (preserves debugging API)
- [x] Parity tests run with `slim=False` (validate intermediate keys unchanged)
- [x] Add test: slim output matches corresponding keys from full output

**Files:** `corridorkey.py`, `engine.py`, `pipeline.py`, new test in `tests/`

---

### Phase 3: Non-Tiled Stage-Boundary Memory Management (medium effort, 1 hr)

**Item 1: Add intermediate graph materialization between encoder/decoder/refiner**

The non-tiled path builds one massive computation graph across all 24 Hiera blocks + 2 decoders + refiner before any materialization. Inserting `mx.eval()` at stage boundaries lets MLX free intermediate graph nodes.

**mx.compile interaction (CRITICAL):**
`mx.compile` wraps the entire `GreenFormer.__call__`. Inserting `mx.eval()` inside a compiled function breaks the compilation graph. Solution: add a `_compiled: bool` flag set by `load_model()` when compile is enabled. Skip stage-boundary GC when compiled.

```python
# corridorkey.py — inside __call__
features = self.backbone(x)

if not self._compiled:
    mx.eval(features)      # materialize backbone output
    gc.collect()
    mx.clear_cache()       # free backbone intermediate graph

# ... decoders ...
coarse = {...}

if not self._compiled:
    mx.eval(coarse)
    gc.collect()
    mx.clear_cache()

# ... refiner ...
```

**Key question answered:** Is this worth it for full-frame? The benchmark shows 27.6 GB peak for full-frame. The backbone's intermediate computation graph (24 blocks of attention + MLP, accumulated lazily) likely dominates. Breaking it into 3 smaller graphs (backbone, decoders, refiner) should reduce peak significantly by allowing MLX to reclaim backbone intermediates before running decoders.

**Acceptance criteria:**
- [x] `_compiled` flag set by `compile_model()` when compile=True
- [x] Stage-boundary GC only runs when `_compiled=False`
- [ ] Full-frame peak memory measurably decreases (measure with `mx.metal.get_peak_memory()`)
- [x] Compiled path behavior unchanged (no mx.eval calls inside graph)
- [x] All parity tests pass

**Files:** `corridorkey.py`, `pipeline.py`

---

### Phase 4: SDPA Migration (medium effort, 1-2 hr)

**Item 2: Replace manual attention with `mx.fast.scaled_dot_product_attention`**

Current (`hiera.py:288-290`):
```python
attn = (q * self.scale) @ mx.transpose(k, ...)
attn = mx.softmax(attn, axis=-1)
x = attn @ v
```

This materializes the full `(B, heads, windows, tokens, tokens)` attention matrix. `mx.fast.scaled_dot_product_attention` fuses this into a single Metal kernel.

**Pre-requisites (spike required, 15 min):**

Before implementing, verify with a quick spike:

1. **5D input handling:** Current Q/K/V are 5D `(B, heads, num_windows, tokens_per_window, head_dim)`. SDPA expects 4D `(batch, heads, seq_len, head_dim)`. Must fold `num_windows` into batch dim: reshape to `(B * num_windows, heads, tokens_per_window, head_dim)`, call SDPA, reshape back.

2. **Asymmetric Q/K/V lengths:** At stride blocks (indices 2, 5, 21), Q is max-pooled to fewer tokens than K/V. Verify `mx.fast.scaled_dot_product_attention` supports `q_len != kv_len` (standard for cross-attention, should work).

3. **Scale factor application:** Current code does `(q * scale) @ k.T`. SDPA does `(q @ k.T) * scale` internally. Mathematically equivalent in fp32 but different rounding in bf16. Measure parity impact.

**Implementation:**

```python
# hiera.py — MaskUnitAttention.__call__
# Fold windows into batch for SDPA
B, H, W, T, D = q.shape
q_4d = q.reshape(B * W, H, -1, D)  # (B*windows, heads, tokens, head_dim)
k_4d = k.reshape(B * W, H, T, D)
v_4d = v.reshape(B * W, H, T, D)

x = mx.fast.scaled_dot_product_attention(
    q_4d, k_4d, v_4d, scale=self.scale
)

x = x.reshape(B, H, W, -1, D)  # unfold windows
```

**Attention matrix sizes (reference):**

| Stage | Blocks | mask_unit_attn | Tokens per window | Attention matrix |
|---|---|---|---|---|
| 0 | 0-1 | True | 64 | 64x64 |
| 1 | 2-4 | True | 16 | 16x16 |
| 2 | 5-20 | False | 256 | 256x256 |
| 3 | 21-23 | False | 64 | 64x64 |

Sizes are modest thanks to Hiera's unrolling. Memory savings per SDPA call are small (~4MB for 256x256). Primary benefit is **kernel fusion** — one Metal dispatch vs three (matmul + softmax + matmul).

**Acceptance criteria:**
- [x] Spike confirms SDPA supports: 4D input, asymmetric Q/K, bf16
- [x] All attention calls use `mx.fast.scaled_dot_product_attention`
- [x] Window dim folded into batch before SDPA, unfolded after
- [x] Stride blocks (Q pooling) produce correct output with asymmetric lengths
- [x] E2E parity within `1e-3` tolerance (existing `PARITY_TOL_E2E`)
- [x] Backbone stage parity regressions documented if any

**Files:** `hiera.py`

---

### Phase 5: GPU Preprocessing (medium effort, 1-2 hr)

**Item 5: Move ImageNet normalize + resize from numpy/PIL to MLX**

Current flow (`io/image.py:48-68`, `engine.py:153-181`):
1. `image.astype(np.float32) / 255.0` — numpy
2. PIL bicubic resize — numpy/PIL
3. ImageNet normalize + concat — numpy
4. Single `mx.array()` call — boundary

**Scope limitation:** Full-frame path only. For tiled inference, the full-res image must stay accessible for tile slicing. Moving full-res preprocessing to GPU would allocate the entire image on GPU before tiling begins — counterproductive.

**Implementation approach:**

```python
# New: io/preprocess_mlx.py
def preprocess_mlx(image_uint8: np.ndarray, mask: np.ndarray, img_size: int) -> mx.array:
    """GPU-side preprocessing for full-frame inference."""
    # Convert to MLX early
    img = mx.array(image_uint8).astype(mx.float32) / 255.0  # (H, W, 3)
    mask = mx.array(mask).astype(mx.float32)                 # (H, W, 1)

    # Resize using nn.Upsample (bilinear, not bicubic)
    # Note: scale_factor = img_size / current_size
    img = mx.expand_dims(img, axis=0)  # (1, H, W, 3) NHWC
    img = upsample(img, target_size=img_size)
    mask = mx.expand_dims(mask, axis=0)
    mask = upsample(mask, target_size=img_size)

    # ImageNet normalize
    mean = mx.array([0.485, 0.456, 0.406])  # (3,)
    std = mx.array([0.229, 0.224, 0.225])
    img = (img - mean) / std

    # Concat
    return mx.concatenate([img, mask], axis=-1)  # (1, H, W, 4) NHWC
```

**Caveat — resize quality:** PIL `BICUBIC` and MLX `nn.Upsample(mode="cubic")` use different cubic kernels. May introduce parity differences. Run a measurement spike: compare PIL bicubic vs MLX cubic on the golden test image, measure max absolute difference.

**Acceptance criteria:**
- [x] Full-frame normalize+concat runs on GPU (resize stays CPU/PIL)
- [x] Tiled path unchanged (keeps numpy preprocessing)
- [x] Resize quality spike: PIL bicubic vs MLX cubic max diff = 0.22 (upscale), 0.59 (downscale)
- [x] Resize diff >> 1e-3: kept PIL resize, moved only normalize+concat to GPU
- [x] E2E parity within tolerance (preprocess_mlx vs preprocess: 0.0 diff)
- [ ] Optional: wrap in `@mx.compile` for static input sizes

**Files:** New `io/preprocess_mlx.py`, `engine.py`

---

### Phase 6: GPU Tensor Accumulators in Tiling (nice-to-have, deferred)

**Item 4: Replace numpy accumulators with MLX arrays**

Current (`tiling.py:120-122`): Three numpy float32 arrays at full resolution. Per-tile results are transferred via `np.array(out["alpha_final"][0])`.

**Why deferred:**
1. MLX lacks in-place slice assignment (`arr[y:y_end, x:x_end] += tile`)
2. `mx.scatter_add` or equivalent needs API verification
3. GPU->CPU transfer per tile is ~1MB at 512x512 — microseconds on unified memory
4. Wave 1 plan already flagged this as "best-effort, 30min timebox" with numpy fallback
5. Total transfer for 16 tiles (2048x2048) is ~16MB — negligible vs compute time

**If pursued later:** Investigate `mx.scatter_add` API, or accumulate using full-size zero tensors with padded tiles (wasteful but avoids slice assignment).

---

## Alternative Approaches Considered

### Token routing (from Raiden129)
Route "easy" tokens to lightweight LTRM module, skip expensive attention. **Rejected for now:** MLX lacks dynamic boolean indexing. `mx.where` + padding doesn't reduce FLOPs. NumPy CPU fallback per-block (16x in stage 2) would destroy async pipeline. Net benefit uncertain. Revisit if profiling shows attention is the bottleneck.

### Int8/MXFP4 quantization
50-75% weight memory reduction. **Deferred:** Matting requires sub-pixel alpha precision. Int8 likely safe (<1% drop) but needs quality validation on real footage. Leave Conv2d, LayerNorm, softmax in full precision. Profile as separate future work.

### Full weight-level bf16 casting
Cast all model weights to bf16 at load time (halves parameter memory). **Not yet explored:** Currently only activations use bf16. Would need to verify backbone parity in bf16 weights (16 blocks = most drift risk). Lower priority than computation graph optimizations.

## Acceptance Criteria

### Functional Requirements
- [ ] All 94 existing tests pass
- [ ] E2E parity within `1e-3` tolerance
- [ ] No regression in tiled inference behavior
- [ ] Engine cleanup releases all Metal buffers after postprocessing

### Non-Functional Requirements
- [ ] Full-frame peak memory measurably decreases with stage-boundary GC (measure with `mx.metal.get_peak_memory()`)
- [ ] No latency regression > 5% in any mode
- [ ] `uv run ruff check .` passes
- [ ] `uv run ruff format --check .` passes
- [ ] `uv run ty check` passes

### Quality Gates
- [ ] SDPA spike completed before Phase 4 implementation
- [ ] Resize quality spike completed before Phase 5 implementation
- [ ] Each phase merged separately with passing CI

## Implementation Order & Dependencies

```
Phase 1 (engine cleanup) ─── no deps ──────────────────── 15 min
Phase 2 (slim forward)  ─── no deps ──────────────────── 30 min
Phase 3 (stage GC)      ─── depends on Phase 2 (slim changes what's in dict) ── 1 hr
Phase 4 (SDPA)          ─── no deps (backbone only) ──── spike 15m + impl 1-2 hr
Phase 5 (GPU preprocess) ── no deps (I/O only) ────────── spike 15m + impl 1-2 hr
Phase 6 (GPU accum)     ─── deferred ─────────────────── (skip)
```

Phases 1-2 and Phase 4 can run in parallel. Phase 3 should follow Phase 2.

## References

### Internal
- Wave 1 plan: `docs/plans/2026-03-08-feat-mlx-memory-optimizations-plan.md`
- Brainstorm: `docs/brainstorms/2026-03-09-pytorch-optimization-learnings-brainstorm.md`
- Deep research: `docs/MLX Vision Transformer Optimization Techniques.md`
- Attention impl: `src/corridorkey_mlx/model/hiera.py:267-297`
- Forward pass: `src/corridorkey_mlx/model/corridorkey.py:56-113`
- Engine cleanup: `src/corridorkey_mlx/engine.py:187-210`
- Tiling accumulators: `src/corridorkey_mlx/inference/tiling.py:120-171`
- Preprocessing: `src/corridorkey_mlx/io/image.py:48-68`
- Pipeline defaults: `src/corridorkey_mlx/inference/pipeline.py:31-58`

### External
- PR #104: https://github.com/nikopueringer/CorridorKey/pull/104
- Raiden129 fork: https://github.com/Raiden129/CorridorKey_Test
- MLX SDPA docs: https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html
- MLX FlashAttention issue: https://github.com/ml-explore/mlx/issues/129
- MLX memory management: https://github.com/ml-explore/mlx/issues/742

## Open Questions

- Stage-boundary GC: measurable peak memory reduction in full-frame, or negligible?
- SDPA spike: does MLX SDPA handle 5D->4D reshape + asymmetric Q/K correctly?
- Resize quality: PIL bicubic vs MLX cubic — within parity tolerance?
- Weight-level bf16: safe for Hiera backbone (16 blocks of drift)?
- Int8 quantization: acceptable alpha precision threshold for matting?
