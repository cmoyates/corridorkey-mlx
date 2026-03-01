---
title: "feat: CorridorKey MLX Inference Port"
type: feat
date: 2026-03-01
---

# CorridorKey MLX Inference Port

## Overview

Port CorridorKey's inference pipeline from PyTorch to MLX for native Apple Silicon execution. The port is staged across 5 phases, each with its own parity gate against PyTorch reference outputs. No training code. No UI.

## Problem Statement

CorridorKey (alpha matting model) runs on PyTorch. On Apple Silicon, PyTorch uses MPS which is slower and less memory-efficient than MLX's native Metal backend. A clean MLX port enables fast local inference without PyTorch overhead.

## Architecture

```
RGB + coarse alpha hint (4ch)
        |
        v
+------------------+
|  Hiera backbone   |  timm, features_only=True
|  -> 4 multiscale  |  feature maps
+------------------+
        |
   +----+----+
   v         v
+-------+ +-------+
| Alpha | |  FG   |
| head  | | head  |
| (1ch) | | (3ch) |
+-------+ +-------+
   |         |
   +----+----+
        v
+------------------+
|   CNN Refiner     |  RGB + coarse preds (7ch)
|   -> delta logits |  -> sigmoid
+------------------+
        |
        v
  final alpha + fg
```

**Key model components:**
- **Backbone:** Hiera (hierarchical vision transformer) -- 4 multiscale feature outputs
- **Decoder heads:** Two parallel heads (alpha 1ch, foreground 3ch) consuming backbone features
- **CNN Refiner:** Takes RGB + coarse predictions (7ch), predicts additive delta logits, final sigmoid

**MLX-specific considerations:**
- MLX uses NHWC layout natively vs PyTorch's NCHW
- Conv weight transpose needed: `(O,I,H,W)` -> `(O,H,W,I)`
- GroupNorm behavior must match PyTorch exactly for parity
- First conv is patched to 4 input channels (RGB + alpha hint)

## Implementation Phases

### Phase 1: PyTorch Reference Harness

**Prompt:** `prompts/phase1-backbone.md`

**Goal:** Deterministic reference pipeline that dumps intermediate tensors for staged MLX parity testing.

**Deliverables:**

- [x] `scripts/dump_pytorch_reference.py` -- loads checkpoint via state_dict, runs forward pass, saves intermediates
- [x] `reference/fixtures/` -- sample inputs and golden outputs
- [x] `tests/test_reference_fixtures.py` -- validates fixture shapes and dtypes

**Tensors to dump:**
1. 4 backbone feature maps (multiscale)
2. Alpha coarse logits
3. FG coarse logits
4. Alpha coarse probs
5. FG coarse probs
6. Refiner delta logits
7. Final alpha
8. Final FG

**Files touched:**

| File | Action |
|------|--------|
| `scripts/dump_pytorch_reference.py` | implement |
| `reference/fixtures/*.npz` or `*.safetensors` | generate |
| `tests/test_reference_fixtures.py` | implement (unskip) |
| `README.md` | update fixture format docs |

**Constraints:**
- Load via `state_dict`, not pickle
- Deterministic (fixed seed, mode)
- One tiny golden example checked in
- Print shape report via rich

**Definition of done:**
- Fixture files exist with all 8+ tensor groups
- Shape contract tests pass
- Script is idempotent

---

### Phase 2: MLX Decoder and Refiner Blocks

**Prompt:** `prompts/phase2-mlx-blocks.md`

**Goal:** MLX implementations of non-backbone blocks with parity tests against Phase 1 fixtures.

**Components to implement:**

- [x] `MLP` -- feedforward block
- [x] `DecoderHead` -- consumes backbone features, produces coarse predictions
- [x] `RefinerBlock` -- single refiner stage
- [x] `CNNRefinerModule` -- full refiner consuming 7ch input

**Files touched:**

| File | Action |
|------|--------|
| `src/corridorkey_mlx/model/decoder.py` | implement MLP + DecoderHead |
| `src/corridorkey_mlx/model/refiner.py` | implement RefinerBlock + CNNRefinerModule |
| `src/corridorkey_mlx/utils/layout.py` | implement NCHW to NHWC transforms |
| `tests/test_decoder_parity.py` | implement (unskip) |
| `tests/test_refiner_parity.py` | implement (unskip) |

**Constraints:**
- NHWC throughout, layout conversions only in `utils/layout.py`
- GroupNorm must match PyTorch behavior exactly
- Parity tests load saved backbone features (from Phase 1 fixtures) as input
- Report max abs error and mean abs error

**Definition of done:**
- Decoder parity test passes (max abs err < 1e-5)
- Refiner parity test passes (max abs err < 1e-5)
- Modules wired into partial model path for test usage

---

### Phase 3: Checkpoint Conversion

**Prompt:** `prompts/phase3-conversion.md`

**Goal:** Robust conversion pipeline from PyTorch checkpoint to MLX-compatible weights.

**Deliverables:**

- [x] `src/corridorkey_mlx/convert/converter.py` -- key mapping + weight transforms
- [x] Conversion diagnostic report (source key -> dest key, shapes, transform applied)
- [x] Output as safetensors

**Files touched:**

| File | Action |
|------|--------|
| `src/corridorkey_mlx/convert/converter.py` | implement |
| `src/corridorkey_mlx/convert/__init__.py` | export converter |
| `tests/test_conversion.py` | implement (unskip) |
| `scripts/convert_weights.py` | create (CLI wrapper) |

**Key mapping concerns:**
- Explicit key-by-key mapping, no regex guessing
- Conv weights: `(O,I,H,W)` -> `(O,H,W,I)`
- Patched 4-channel first conv must be preserved exactly
- No silent fallbacks -- all mismatches are errors

**Diagnostic output per key:**
```
source_key -> dest_key | src_shape -> dst_shape | transform
```

**Definition of done:**
- Converter script exists and runs
- Mapping is explicit and auditable
- Shape validation passes for all completed modules
- No orphan keys (every source key maps or is explicitly skipped with reason)

---

### Phase 4: Full Inference Pipeline

**Prompt:** `prompts/phase4-inference-pipeline.md` (not yet written)

**Goal:** End-to-end inference matching PyTorch output.

**Deliverables:**

- [x] `src/corridorkey_mlx/model/backbone.py` -- Hiera MLX port
- [x] `src/corridorkey_mlx/model/corridorkey.py` -- full model composition
- [x] `src/corridorkey_mlx/inference/pipeline.py` -- load, preprocess, forward, postprocess, save
- [x] `src/corridorkey_mlx/io/image.py` -- PIL-based image I/O + preprocessing
- [x] End-to-end parity test against PyTorch golden output

**Files touched:**

| File | Action |
|------|--------|
| `src/corridorkey_mlx/model/backbone.py` | implement Hiera MLX |
| `src/corridorkey_mlx/model/corridorkey.py` | implement full model |
| `src/corridorkey_mlx/inference/pipeline.py` | implement |
| `src/corridorkey_mlx/io/image.py` | implement |
| `tests/test_e2e_parity.py` | create |
| `main.py` | wire CLI via typer |

**Definition of done:**
- `uv run python main.py --image input.png --output output.png` produces correct result
- E2e parity test passes within tolerance
- Memory usage is reasonable (no unnecessary copies)

---

### Phase 5: Optimization and Benchmarking

**Prompt:** `prompts/phase5-optimization.md` (not yet written)

**Goal:** Production-quality performance on Apple Silicon.

**Deliverables:**

- [x] `scripts/bench_mlx.py` -- latency, throughput, memory reporting
- [x] `scripts/compare_reference.py` -- side-by-side output comparison
- [x] Performance optimizations (compile, memory layout, batching)

**Potential optimizations:**
- `mx.compile()` on hot paths
- Memory-efficient attention if Hiera benefits
- Avoid unnecessary `mx.eval()` calls (lazy graph)
- Optimal image preprocessing (avoid numpy round-trips)

**Definition of done:**
- Benchmark script reports latency + peak memory
- Performance is competitive with PyTorch MPS on same hardware
- No correctness regression (parity tests still pass)

---

## Acceptance Criteria

### Functional Requirements

- [ ] All 5 phases complete with passing parity gates
- [ ] Single-image inference CLI works end-to-end
- [ ] Output visually matches PyTorch reference

### Quality Gates (per phase)

- [ ] Parity tests pass within documented tolerance
- [ ] No skipped tests without explicit phase rationale
- [ ] ruff + mypy clean
- [ ] Fixtures are deterministic and reproducible

## Dependencies and Prerequisites

| Dependency | Phase | Notes |
|-----------|-------|-------|
| Original CorridorKey checkpoint | 1 | needed for reference dump |
| Original CorridorKey source code | 1 | needed to understand architecture |
| PyTorch + timm (reference group) | 1 | `uv sync --group reference` |
| Phase 1 fixtures | 2, 3 | parity test inputs |
| Phase 2 + 3 complete | 4 | blocks + weights needed for full model |

## Risk Analysis

| Risk | Impact | Mitigation |
|------|--------|------------|
| Hiera has no existing MLX port | High -- Phase 4 blocker | Research early; may need full rewrite |
| GroupNorm numerical differences | Medium -- parity failures | Test tolerance tuning; document acceptable drift |
| 4ch first conv patching | Low -- conversion bug | Explicit test in Phase 3 |
| MLX API changes | Low -- version pinned | Pin `mlx>=0.31.0` in pyproject |

## Current Repo State

```
src/corridorkey_mlx/
  __init__.py                 [exists]
  model/
    backbone.py               [placeholder]
    decoder.py                [placeholder]
    refiner.py                [placeholder]
    corridorkey.py            [placeholder]
  convert/
    converter.py              [placeholder]
  inference/
    pipeline.py               [placeholder]
  io/
    image.py                  [placeholder]
  utils/
    layout.py                 [placeholder]

scripts/
  dump_pytorch_reference.py   [placeholder]
  compare_reference.py        [placeholder]
  bench_mlx.py                [placeholder]

tests/
  test_import.py              [passes]
  test_reference_fixtures.py  [skipped - Phase 1]
  test_decoder_parity.py      [skipped - Phase 2]
  test_refiner_parity.py      [skipped - Phase 2]
  test_conversion.py          [skipped - Phase 3]

prompts/
  phase1-backbone.md          [written]
  phase2-mlx-blocks.md        [written]
  phase3-conversion.md        [written]
  phase4-inference-pipeline.md  [empty]
  phase5-optimization.md        [empty]
```

## Open Questions

- Hiera port strategy: full rewrite vs adapting existing MLX vision transformer code?
- Acceptable e2e parity tolerance? (likely max abs < 1e-3 due to float32 Metal vs CUDA differences)
- Where is the original CorridorKey checkpoint hosted? (needed for Phase 1)
- Original CorridorKey source -- is it the nicehash/CorridorKey repo or a fork?

## References

- Original repo: nicehash/CorridorKey on GitHub
- MLX: ml-explore/mlx on GitHub
- timm Hiera: `timm.create_model("hiera_...", features_only=True)`
- Phase prompts: `prompts/phase{1-5}-*.md`
