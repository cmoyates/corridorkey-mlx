# Upstream Research 5 — Post-2048 Switch Audit

**Date:** 2026-03-13
**Sources:** MLX framework, CorridorKey (main), CorridorKey-Engine, Marcel's fork, EZ-CorridorKey, MLX examples

---

## Critical Finding: metal_kernel DOES support threadgroup shared memory

Our brainstorm doc (2026-03-12) stated `mx.fast.metal_kernel()` lacks threadgroup shared memory. **This was wrong.**

BTCA analysis of `mlx/backend/metal/custom_kernel.cpp` (lines 148-149) shows the user-provided `source` string is embedded directly into a Metal kernel function body. Any valid MSL is legal:
- `threadgroup float shared[N]` with N via template args (lines 42-62)
- `threadgroup_barrier(mem_flags::mem_threadgroup)` auto-detected as attribute
- MLX's own `layer_norm.metal` uses exactly this pattern (line 22)

Exp 42 went straight to simd_sum + atomics because we assumed shared memory wasn't available. A proper shared-memory reduction GroupNorm was never attempted. **Reopens the GroupNorm optimization path (50% of refiner time).**

→ Issue #14

## EZ-CorridorKey v1.6.0 integrates corridorkey-mlx

EZ-CorridorKey auto-detects and uses corridorkey-mlx as its Apple Silicon backend, claiming 1.5-2x faster than PyTorch MPS. Validates our API surface.

→ Issue #15

## Refiner tiling overlap: 64px is dangerously close to 65px receptive field

All upstream forks use 128px overlap (2x safety margin). EZ-CorridorKey claims >157 dB PSNR at 128px. Our 64px leaves <1px margin — may have subtle edge artifacts.

→ Issue #16

## Raiden129/CorridorKey_Test — new BTCA resource

Added to btca.config.jsonc. Key findings:

### Sparse Tiled Refiner (confirms V3/V6 direction)
- Skips tiles where alpha max < 0.05 (bg) or min > 0.95 (fg)
- Zeroes ALL delta channels (alpha + fg) on skip — same as our V3
- Claims 50-70% skip rate on typical footage
- **Why it works for them but not us:** PyTorch `model.eval()` freezes BN/GN running stats by default. Their GroupNorm uses frozen stats for free. Our MLX GroupNorm recomputes spatial stats per tile, causing the tiling artifact (exp 46).
- **Implication:** If #14 (shared-memory GroupNorm) succeeds, we could implement a FrozenGroupNorm that's actually fast, unblocking tile skip.

### In-place decode-and-refine pipeline
- `add_`, `sigmoid_`, `mul_` with explicit `del` between stages
- Classification: **pytorch-only** (MLX has no in-place ops)
- We already have mx.eval barriers between stages for similar effect

### refiner_scale multiplier
- User-controllable scaling of delta logits (0.0-3.0)
- Classification: **mlx-portable** — trivial to add, matches EZ-CorridorKey's feature

## Other findings (no new issues needed)

| Finding | Source | Classification |
|---------|--------|----------------|
| No new CorridorKey-Engine token routing changes | Engine | Confirmed unchanged |
| Marcel's fork stable, no new commits | Marcel | No action |
| Configurable worker pool counts | Engine v1.6 | concept-only |
| GELU approx="fast" in MLX examples | MLX examples | Low priority (#17) |
| mxfp8/nvfp4 quantization available | MLX framework | Low priority (not our bottleneck) |
| Winograd conv blocked by 64ch < 256 threshold | MLX framework | Not actionable |
| No new mx.compile breakthroughs | MLX framework | No action |
| class_predicate mixed quant API | MLX examples | Cleaner but equivalent to safe_quantize |
| Humility clamping removed upstream | CorridorKey main | Need to check our port |

## New Issues Created

| # | Title | Priority |
|---|-------|----------|
| 14 | Re-attempt custom Metal GroupNorm with threadgroup shared memory | HIGH |
| 15 | Upstream MLX backend: check for alignment and techniques | MEDIUM |
| 16 | Increase refiner tile overlap from 64px to 128px | MEDIUM |
| 17 | Test GELU approx="fast" in Hiera MLP | LOW |
