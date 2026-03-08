---
title: "feat: MLX Memory Optimizations (bf16 + Fused Decode + Deterministic GC)"
type: feat
date: 2026-03-08
---

# MLX Memory Optimizations

## Overview

Three sequential UMA-aware optimizations to reduce peak Metal memory and improve throughput on 8GB Apple Silicon. Each step is gated by `uv run pytest tests/test_parity.py` before proceeding.

**Branch:** `experiment/mlx-memory-optimizations` (off `main`)

## Problem Statement

- Full fp32 forward at 2048x2048 consumes significant unified memory
- Tiled inference accumulates MLX graph references across iterations, fragmenting UMA cache and causing OOM on 8GB Macs
- Two separate decoder upsamples (alpha 1ch + fg 3ch) generate redundant Metal dispatch

## Proposed Solution

| Step | Optimization | Priority | Risk |
|------|-------------|----------|------|
| 1 | Selective bfloat16 (backbone fp32, decoders+refiner bf16) | High | Medium |
| 2 | Batched decoder upsampling (fused resize) | Low | Medium |
| 3 | Deterministic GC pipeline in tile loop | **Critical** | Low |

---

## Phase 0: Prerequisite Spikes (Before Coding)

Three unknowns must be resolved before implementation begins. Each is a 10-minute spike.

### Spike 0a: mx.compile + bf16 compatibility

```python
# spike_compile_bf16.py — run interactively
import mlx.core as mx

def mixed_fn(x):
    y = x.astype(mx.bfloat16)
    z = y @ y.T
    return z.astype(mx.float32)

compiled = mx.compile(mixed_fn)
x = mx.random.normal((4, 4))
out = compiled(x)
mx.eval(out)
print(out.dtype, out)  # expect float32, no error
```

**If fails:** Step 1 must disable mx.compile when bf16 is active, or scope compile to backbone-only.

### Spike 0b: mx.metal.clear_cache() API existence

```python
import mlx.core as mx
mx.metal.clear_cache()  # does this exist?
# Also check: mx.metal.get_cache_memory(), mx.metal.set_cache_limit()
```

**If missing:** Check MLX version, find equivalent API, or skip cache clearing (rely on del + gc only).

### Spike 0c: Why was fp16 mixed precision reverted?

```bash
git log --all --oneline --grep="fp16\|float16\|mixed.prec\|precision" -- src/
```

Understanding the revert reason gates whether bf16 hits the same wall.

---

## Phase 1: Selective bfloat16 Mixed-Precision

### 1.1 Design Decisions (from SpecFlow gaps)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Activation-only or weight+activation bf16? | **Activation-only** initially | Simpler, no checkpoint changes, weights stay fp32 in safetensors. Revisit weight casting if memory savings insufficient. |
| Where does cast happen? | In `GreenFormer.__call__`, after `self.backbone(features)` | Single cast point, decoders receive bf16 features, matmuls auto-promote with fp32 weights |
| Sigmoid in bf16 or fp32? | **fp32** — cast back before sigmoid | Sigmoid saturation at boundaries affects alpha matte quality |
| Which output keys are fp32? | **All 9 keys** cast to fp32 before returning | Preserves existing API contract, simplifies test updates |
| Opt-in/opt-out? | Add `dtype: mx.Dtype = mx.float32` param to `GreenFormer.__init__` | Default fp32 = zero behavior change; `mx.bfloat16` enables mixed precision |

### 1.2 Implementation

**File: `src/corridorkey_mlx/model/corridorkey.py`**

```python
# GreenFormer.__init__ — add dtype param
def __init__(self, img_size: int = 512, use_sdpa: bool = True,
             dtype: mx.Dtype = mx.float32):
    ...
    self._compute_dtype = dtype

# GreenFormer.__call__ — cast after backbone, cast back before return
def __call__(self, x: mx.array) -> dict[str, mx.array]:
    # Backbone always fp32
    features = self.backbone(x)

    # Cast features to compute dtype for decoders
    if self._compute_dtype != mx.float32:
        features = [f.astype(self._compute_dtype) for f in features]

    alpha_logits = self.alpha_decoder(features)
    fg_logits = self.fg_decoder(features)

    # Cast back to fp32 before sigmoid (precision at saturation boundaries)
    alpha_logits_up = self._logit_upsampler(alpha_logits).astype(mx.float32)
    fg_logits_up = self._logit_upsampler(fg_logits).astype(mx.float32)

    alpha_coarse = mx.sigmoid(alpha_logits_up)
    fg_coarse = mx.sigmoid(fg_logits_up)

    # Refiner receives fp32 coarse predictions
    ...
    # All output dict values explicitly fp32
    return {k: v.astype(mx.float32) for k, v in outputs.items()}
```

### 1.3 Guardrails

- [x] Verify all `nn.GroupNorm` layers retain `pytorch_compatible=True` after any decoder modifications
- [x] Verify `nn.BatchNorm` in decoder fusion layer handles bf16 inputs correctly (running stats are fp32 in eval mode — matmul should auto-promote)
- [x] If mx.compile + bf16 fails (Spike 0a), add `if self._compute_dtype != mx.float32: self._compiled = False` guard — N/A, spike passed

### 1.4 Test Updates

**File: `tests/test_model_contract.py`**

```python
# Parameterize dtype test
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_output_dtype(dtype):
    model = GreenFormer(img_size=256, dtype=dtype)
    ...
    for key, arr in outputs.items():
        assert arr.dtype == mx.float32  # always fp32 output regardless of compute dtype
```

**File: `tests/test_parity.py`**

- Keep existing fp32 tolerances as-is (TIGHT=1e-4, E2E=1e-3)
- Add separate bf16 parity test with relaxed tolerances (measure actual drift in spike, expect ~1e-3 for intermediates, ~1e-2 for backbone-coupled outputs)
- If bf16 parity exceeds 5e-2 on any key, investigate before loosening further

### 1.5 Gate

```bash
uv run pytest tests/test_parity.py tests/test_model_contract.py -v
```

All existing fp32 tests must still pass unchanged. New bf16 tests must pass at relaxed tolerances.

---

## Phase 2: Batched Decoder Upsampling (Best-Effort)

### 2.1 Approach

The real bandwidth target: both `alpha_decoder` and `fg_decoder` independently resize 3 projected feature maps (2x, 4x, 8x) at identical spatial sizes. Batch them:

1. Run alpha and fg linear projections independently (preserves checkpoint keys)
2. At each stage (c2, c3, c4): concatenate alpha_proj and fg_proj along channel axis (`axis=-1`) — NHWC convention
3. Run single `nn.Upsample` on the fused tensor (2x, 4x, or 8x as appropriate)
4. Split back along channel axis before final 1x1 conv + BN

### 2.2 Architecture Change

This requires refactoring `DecoderHead` to expose intermediate projections, or creating a `FusedDecoderPair` wrapper that orchestrates both heads.

**Option A: FusedDecoderPair wrapper** (preferred — non-invasive)

```python
# New class in decoder.py
class FusedDecoderPair(nn.Module):
    """Runs two DecoderHeads with batched upsampling."""
    def __init__(self, alpha_head: DecoderHead, fg_head: DecoderHead):
        self.alpha_head = alpha_head
        self.fg_head = fg_head
        # Reuse alpha_head's pre-allocated upsamplers (same scale factors)

    def __call__(self, features):
        # Project independently (preserves weights)
        alpha_projs = [mlp(f) for mlp, f in zip(self.alpha_head.linear_projections, features)]
        fg_projs = [mlp(f) for mlp, f in zip(self.fg_head.linear_projections, features)]

        # Batch upsample at each scale
        for i, (a_proj, f_proj, upsampler) in enumerate(...):
            fused = mx.concatenate([a_proj, f_proj], axis=-1)  # NHWC
            fused_up = upsampler(fused)
            a_proj, f_proj = mx.split(fused_up, [256], axis=-1)  # split at embed_dim

        # Independent fusion + classification
        alpha_logits = self.alpha_head.classifier(self.alpha_head.fuse_bn(...))
        fg_logits = self.fg_head.classifier(self.fg_head.fuse_bn(...))
        return alpha_logits, fg_logits
```

### 2.3 Constraints

- [x] `nn.Upsample` instances stay pre-allocated in `__init__` (Phase 6 guardrail)
- [x] Concatenation on `axis=-1` only (NHWC)
- [x] Checkpoint loading unaffected (same `alpha_decoder.*` / `fg_decoder.*` keys)
- [x] mx.compile shape tracing: channel dim changes from 256 to 512 in fused path — verify compile handles this

### 2.4 Fallback Criteria

**Abandon Step 2 if ANY of:**
- Parity degrades beyond existing E2E tolerance (1e-3) vs unfused baseline
- `FusedDecoderPair` exceeds ~80 lines of structural change
- mx.compile fails to trace the variable-channel upsamples
- Implementation takes >2 hours

**If abandoned:** Revert all Step 2 changes, proceed directly to Step 3.

### 2.5 Gate

```bash
uv run pytest tests/test_parity.py tests/test_model_contract.py -v
```

---

## Phase 3: Deterministic GC Pipeline (CRITICAL — Highest Priority)

### 3.1 Scope Clarification

Two independent concerns, implemented together:

| Concern | Priority | Approach |
|---------|----------|----------|
| Per-tile memory cleanup (GC pipeline) | **Mandatory** | del + gc.collect + clear_cache |
| MLX-native accumulator (scatter-add) | Nice-to-have | Try briefly, fallback to numpy |

### 3.2 GC Pipeline Implementation

**File: `src/corridorkey_mlx/inference/tiling.py`**

```python
import gc

def tiled_inference(model, image, mask, tile_size, overlap, ...):
    # Accumulators (numpy — proven, safe)
    alpha_accum = np.zeros((full_h, full_w, 1), dtype=np.float32)
    fg_accum = np.zeros((full_h, full_w, 3), dtype=np.float32)
    weight_accum = np.zeros((full_h, full_w, 1), dtype=np.float32)

    for y_start, x_start in tile_coords:
        # --- Tile forward ---
        tile_input = ...  # slice + pad
        out = model(tile_input)
        mx.eval(out)  # (1) Force lazy graph materialization

        # Extract only what we need to numpy
        alpha_tile = np.array(out["alpha_final"][0])
        fg_tile = np.array(out["fg_final"][0])

        # Accumulate in numpy (unchanged logic)
        alpha_accum[y:y_end, x:x_end] += alpha_tile * weight
        fg_accum[y:y_end, x:x_end] += fg_tile * weight
        weight_accum[y:y_end, x:x_end] += weight

        # --- Deterministic memory cleanup (MANDATORY) ---
        del out, tile_input, alpha_tile, fg_tile  # (2) Drop Python refs
        gc.collect()                               # (3) Fire C++ destructors
        if hasattr(mx, 'metal') and hasattr(mx.metal, 'clear_cache'):
            mx.metal.clear_cache()                 # (4) Release Metal pages

    ...
```

### 3.3 MLX-Native Accumulator (Best-Effort, 30min timebox)

```python
# Attempt scatter-add — if this syntax works, use it
accum = mx.zeros((1, full_h, full_w, channels))
accum = accum.at[0, y:y_end, x:x_end, :].add(weighted_tile)
```

**If `mx.array.at[].add()` raises AttributeError or compilation error:** immediately revert to numpy accumulator. The GC pipeline is the real win.

### 3.4 Also Apply GC to engine.py (SpecFlow Gap 12)

**File: `src/corridorkey_mlx/engine.py`**

After `process_frame` extracts needed keys from the output dict, delete unused keys:

```python
outputs = self._model(x)
mx.eval(outputs)
alpha_final = outputs["alpha_final"]
fg_final = outputs["fg_final"]
# ... extract what's needed ...
del outputs  # Release the 7 unused intermediate tensors
gc.collect()
```

### 3.5 Gate

```bash
uv run pytest tests/test_parity.py tests/test_tiling.py -v
```

---

## Acceptance Criteria

### Functional

- [ ] All existing fp32 parity tests pass unchanged
- [ ] bf16 forward produces outputs within measurable tolerance of fp32 golden references
- [ ] Tiled inference completes without OOM on representative input
- [ ] `GreenFormer(dtype=mx.float32)` is exact same behavior as current code (zero regression)
- [ ] Checkpoint loading unchanged — same safetensors keys

### Non-Functional

- [ ] Peak Metal memory measurably reduced (benchmark with `scripts/bench_mlx.py`)
- [ ] No new dependencies
- [ ] GroupNorm `pytorch_compatible=True` preserved in all instances

### Quality Gates

- [ ] `uv run pytest` — all tests pass
- [ ] `uv run ruff check .` — no lint errors
- [ ] `uv run ruff format .` — formatted
- [ ] `uv run ty check` — no type errors

---

## Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| mx.compile rejects bf16 casts in graph | Medium | High | Spike 0a gates Step 1; fallback: disable compile for bf16 |
| bf16 parity exceeds tolerance | Medium | Medium | Measure actual drift in spike; adjust tolerances or revert |
| mx.metal.clear_cache() doesn't exist | Low | Low | Guarded with hasattr; gc.collect alone may suffice |
| Step 2 FusedDecoderPair too complex | Medium | Low | Hard fallback criteria — skip and focus on Step 3 |
| Prior fp16 revert cause applies to bf16 | Low | High | Spike 0c — check git history |

---

## Open Questions

- bf16 parity tolerance — exact numbers depend on Spike measurement
- Weight-level bf16 casting — revisit after activation-only results
- Slim forward mode (skip unused dict keys) — separate optimization, complements GC
- Decoupled resolution + bf16 interaction — test if backbone_size is on main

---

## References

- Brainstorm: `docs/brainstorms/2026-03-08-mlx-memory-optimizations-brainstorm.md`
- Model: `src/corridorkey_mlx/model/corridorkey.py`
- Decoder: `src/corridorkey_mlx/model/decoder.py`
- Refiner: `src/corridorkey_mlx/model/refiner.py`
- Tiling: `src/corridorkey_mlx/inference/tiling.py`
- Engine: `src/corridorkey_mlx/engine.py`
- Parity tests: `tests/test_parity.py`
- Contract tests: `tests/test_model_contract.py`
- Tolerances: `tests/conftest.py:24-26`
