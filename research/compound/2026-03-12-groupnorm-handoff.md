# GroupNorm Optimization — Handoff Doc

**Date**: 2026-03-12
**Status**: Python-level approaches exhausted. Custom Metal kernel is the only remaining path.

---

## Context

40+ experiments over 4 upstream research rounds. Application-level optimization confirmed exhausted. Metal trace at 512 identified GroupNorm transpose copies as 6.94% GPU time. This session investigated whether that could be improved.

## Key Discovery: GN is 50% of refiner at 1024

| Metric | Value |
|--------|-------|
| Refiner with GN (compiled) | 214.8ms |
| Refiner without GN (identity) | 107.4ms |
| **GN contribution** | **~107ms (50%)** |
| Isolated GN (compiled) | 7.57ms x 9 = ~68ms |
| Copy/overhead | ~39ms (107 - 68) |

The 6.94% figure from the 512 Metal trace dramatically understates the problem at 1024. GN reduction scales quadratically with spatial size (4x more elements per group at 1024 vs 512).

## What was tried and failed

### Approach 1: ContiguousCopyGN
- Add `+ 0.0` after GN output to force contiguous
- Result: -0.1% (noise). mx.compile already handles this.

### Approach 2: TransposedAffineGN
- Apply affine (weight*x+bias) in transposed layout before the second transpose
- Result: +8.8% regression. 4D broadcast affine is slower than 1D.

### Approach 3: TwoPassFP32GN
- Compute mean/var via mx.mean/mx.var with fp32 accumulation, no transpose
- Result: +120% regression. mx.mean/mx.var over axes (1,3) of 4D tensor is 4.9x slower than mx.fast.layer_norm.

### Why nothing worked
- `mx.fast.layer_norm` is a single fused Metal kernel (mean+var+normalize in one pass over contiguous data)
- No combination of separate MLX primitives can match its throughput
- The 2 transposes (9.12ms overhead vs 7.57ms baseline in TransposedAffineGN) are the unavoidable cost of making data contiguous for layer_norm
- mx.compile already optimizes the copy propagation — no Python-level tricks help

## What remains: Custom Metal kernel

A `mx.fast.metal_kernel()` that computes grouped normalization in-place on NHWC data without transpose. This is the only unexplored path.

### Challenges (from SpecFlow analysis)
1. **8M element reduction per group** at 1024 — needs threadgroup shared memory + barriers
2. **`mx.fast.metal_kernel()` reduction capability unproven** at this scale
3. **Custom kernels are NOT fusable** by mx.compile — 9 dispatch barriers could offset gains
4. **Must accumulate in fp32** — bf16 over 8M elements = catastrophic precision loss
5. **Two kernel variants needed** — 5x GN+ReLU, 4x GN-only

### If someone picks this up
1. **First**: prototype a minimal Metal kernel that computes mean of (1, 1M, 8) tensor via `mx.fast.metal_kernel()` with threadgroup reductions. This proves/disproves the API's capability.
2. **If feasible**: implement full GroupNorm kernel (two-pass: reduce for stats, then normalize+affine+relu)
3. **If not feasible**: consider upstream MLX PR to add `mx.fast.group_norm` (or improve `nn.GroupNorm` to output contiguous)
4. **Fallback**: accept GN cost as architectural — it's the price of using the fused layer_norm kernel

### Estimated prize
- Best case: eliminate the ~39ms copy overhead (107ms GN total - 68ms layer_norm compute)
- Realistic: 10-20ms savings after accounting for custom kernel dispatch overhead
- At 1024: 215ms -> ~195-205ms = 5-10% refiner speedup, ~2-5% end-to-end

## Current best (unchanged)
- **422.46ms** median @1024, 3319MB peak (1407MB with buffer env vars)
- 41 experiments logged

## Files created this session
- `docs/brainstorms/2026-03-12-contiguous-groupnorm-brainstorm.md`
- `docs/plans/2026-03-12-feat-contiguous-groupnorm-refiner-plan.md`
- `research/compound/upstream_research_4_findings.md`
- `research/compound/exp041_contiguous_groupnorm_variants.md`
- `scripts/bench_groupnorm.py` — isolated GN + refiner with/without GN
- `scripts/bench_groupnorm_variants.py` — variant comparison (3 approaches)
- Updated: `research/program.md` (Phase 4 search areas), `research/experiments.jsonl` (exp 41)
