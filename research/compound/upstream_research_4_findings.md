# Upstream Research Round 4 — 2026-03-12

## Summary
Fourth upstream mining round. Application-level optimization confirmed exhausted at 40 experiments. Focus shifted to framework-level bottlenecks.

## MLX framework (v0.31.0 → v0.31.1)
- v0.31.1: M5 Pro/Max kernel tuning (no M3 benefit), 3/5/6-bit QMV kernels, load_weights strict fix
- mxfp8 and nvfp4 on Metal since v0.30.3 — potential for stage 0 (dim=112) quantization
- mx.fast.metal_kernel confirmed: works for element-wise ops, NOT for AMX-bound compute (conv, matmul)
- Split-K GEMM (v0.30.4) for large-K matmuls — may already be auto-selected by MLX heuristics
- PR #3120 (split-K quantized matmul, 25-30% faster) still open — would help quantized backbone
- PR #3247 (per-stream locking) still open — concurrent inference potential

## CorridorKey upstream
- No new optimization PRs since 2026-03-10
- PR #126 (BiRefNet integration) merged — different model, not relevant
- Issue #107 (distilled model) closed — no smaller checkpoint exists
- Issue #144 (quantization) closed — QAT not planned upstream
- Community repos (99oblivius/Engine, edenaion/EZ) — no new techniques beyond what we've tested

## Key finding: "Writing Fast MLX" guide audit
- Type promotion: `mx.array(scalar)` upcasts to fp32; Python scalars preserve input precision
- Our hot paths look clean: REFINER_SCALE=10.0 is Python float, ImageNet constants are preprocessing-only
- `mx.compile` closure hazard: captured `mx.array` values force retracing — our per-component approach avoids this
- Matmul ordering `x @ W.T` faster than `W.T @ x` — already applied

## Identified next experiment: Contiguous GroupNorm
- GroupNorm pytorch_compatible transpose copies = 6.94% GPU time (Metal trace)
- 10 instances in refiner (stem + 2 per ResBlock × 4)
- NOT the same as exp32 (which dropped pytorch_compatible and failed fidelity)
- Goal: same math, no non-contiguous intermediate
- Potential: ~29ms @1024 theoretical savings
- See brainstorm: `docs/brainstorms/2026-03-12-contiguous-groupnorm-brainstorm.md`

## Lower-priority untried ideas
1. mxfp8 for backbone stage 0 (dim=112, can't do int4/int8)
2. 5-bit quantization for stages 1-3 (MLX 0.31.1 QMV kernels)
3. mx.clear_cache() between pipeline stages (different from del backbone)
4. MLX 0.31.1 upgrade (free, minor)
