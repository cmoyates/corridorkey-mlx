# MLX Memory Optimizations — Brainstorm

**Date:** 2026-03-08
**Branch:** `experiment/mlx-memory-optimizations` (off `main`)
**Goal:** Reduce peak memory + improve throughput on 8GB Apple Silicon via 3 UMA-aware optimizations.

## What We're Building

Three sequential optimizations, each gated by `pytest tests/test_parity.py`:

### Step 1: Selective bfloat16 Mixed-Precision
- Backbone (Hiera): stays fp32 — attention drift risk at lower precision
- After backbone features extracted: cast to bf16
- Decoders + Refiner: operate in bf16 (bf16 has 8-bit exponent — survives REFINER_SCALE=10.0 multiplication unlike fp16)
- Final outputs: cast back to fp32

### Step 2: Batched Decoder Upsampling (Best-Effort)
- Keep `alpha_decoder` / `fg_decoder` structurally separate (preserve checkpoint key compatibility)
- Each head independently projects 4 backbone features to embed_dim=256
- **After projection**: concatenate features from both heads, run 2x/4x/8x `mx.image.resize` on batched tensor
- Split back before final 1x1 convolutions
- **Fallback**: If architecturally complex or breaks parity, skip entirely. Do NOT implement simple logit-only fusion.

### Step 3: Deterministic GC Pipeline (CRITICAL — highest priority)
- **Try**: MLX-native accumulator with scatter-add
- **Fallback**: Keep numpy accumulator if MLX scatter API is problematic
- **Mandatory regardless of accumulator type** — strict per-tile memory lifecycle:
  1. `mx.eval()` — force lazy graph execution
  2. `del` all tile-local intermediates
  3. `gc.collect()` — fire C++ destructors
  4. `mx.metal.clear_cache()` — release Metal buffer pages

## Why This Approach

- bf16 over fp16: dynamic range preserved (8-bit exp), halves memory vs fp32
- Batched resize: single mx.compile fusion pass > two separate resize dispatches
- Deterministic GC: prevents UMA cache fragmentation that OOMs 8GB Macs in tile loops

## Key Decisions

1. **Backbone stays fp32** — deep ViT attention is precision-sensitive
2. **Step 2 is best-effort** — skip if messy; Step 3 is the real win for memory
3. **GC pipeline is mandatory** — even with numpy fallback accumulator
4. **Checkpoint compatibility preserved** — no weight key changes in decoder refactor
5. **Sequential gating** — each step must pass parity before proceeding

## Critical Architectural Guardrails (Do Not Violate)

1. **GroupNorm Parity:** If any `nn.GroupNorm` layers are modified during the bf16 refactor, they MUST retain the `pytorch_compatible=True` flag. MLX calculates epsilon differently, and dropping this flag will instantly break golden parity.
2. **Upsampler Pre-allocation:** For Step 2, the `nn.Upsample` instances must continue to be pre-allocated in `__init__` (as done in Optimization Phase 6). Do not instantiate them dynamically inside the `forward` pass.
3. **NHWC Memory Layout:** The MLX port processes arrays natively in NHWC (Batch, Height, Width, Channel) format. When concatenating the alpha and fg projections in Step 2, you must concatenate along the last axis (`axis=-1`).

## Mismatches Found (vs Original Spec)

| Assumption | Reality |
|---|---|
| DecoderHead does separate alpha/fg resize | Two separate DecoderHead instances; internal per-feature upsampling (2x/4x/8x), then shared logit upsampler |
| backbone_size doesn't exist | Already implemented in Opt Phase 3 (may not be on main) |
| scatter-add syntax works | JAX-style syntax; MLX support unverified — plan with numpy fallback |

## Open Questions

- Does MLX support `.at[slice].add()` scatter syntax? (verify at impl time)
- bf16 parity tolerance vs golden fp32 references — what's acceptable threshold?
- Is batched resize actually faster under mx.compile, or does concat/split overhead negate gains?

## Validation

- Each step: `uv run pytest tests/test_parity.py`
- Final: `scripts/bench_mlx.py` — peak memory + median latency vs baseline
