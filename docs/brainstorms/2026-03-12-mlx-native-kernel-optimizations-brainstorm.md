# Plan A: MLX-Native Kernel Optimizations

**Date**: 2026-03-12
**Status**: ACTIVE
**Target**: 16.19% of GPU time (GroupNorm 6.94% + Gather 9.25%)
**Expected gain**: 10-15% latency reduction (~83ms -> ~70-75ms @512)

## What We're Building

Three surgical optimizations targeting the two largest non-compute bottlenecks identified in Metal trace analysis, all staying within MLX framework.

## Why This Approach

- Targets specific, measured bottlenecks (not speculative)
- No new dependencies (no coremltools, no mlx-mfa yet)
- Each experiment is independent, reversible, and measurable
- Stays within project scope (CLAUDE.md: no CoreML/ANE)
- Fidelity gate via golden.npz remains unchanged

---

## Phase A1: Environment Variable Sweep (30 min)

### Hypothesis
`MLX_MAX_MB_PER_BUFFER` and `MLX_MAX_OPS_PER_BUFFER` defaults may cause premature command buffer splits, adding dispatch overhead.

### Experiment
```bash
# Baseline
uv run python scripts/bench_mlx.py

# Sweep
for MB in 16 32 64 128 256 512 1024 1000000; do
  MLX_MAX_MB_PER_BUFFER=$MB uv run python scripts/bench_mlx.py
done

# Combined with OPS tuning
MLX_MAX_MB_PER_BUFFER=1000000 MLX_MAX_OPS_PER_BUFFER=8 uv run python scripts/bench_mlx.py
```

### Success criteria
- Any measurable latency improvement (>2%) without fidelity regression
- Note: our per-component compile + stage_gc may already mitigate this

### Rollback
No code changes — env vars only.

---

## Phase A2: GroupNorm Weight Remapping (2-4 hours)

### Hypothesis
Dropping `pytorch_compatible=True` and remapping affine weights to match NHWC native grouping eliminates 10 transpose+copy kernels in refiner (6.94% GPU time).

### Current state
- 3 GroupNorm instances in `refiner.py` (stem + 4 blocks x 2 = 10 calls)
- All: `nn.GroupNorm(8, 64, pytorch_compatible=True)`
- C=64, G=8, C//G=8

### Weight remapping math
PyTorch groups channels as `[0..7], [8..15], ..., [56..63]` in NCHW.
MLX native (NHWC) groups channels sequentially in last dim — same ordering.

**Key insight**: With C=64 and G=8, the channel grouping is identical in both layouts because grouping is along the channel dim, not spatial. The `pytorch_compatible` flag may be unnecessary here.

### Experiment steps
1. Create `scripts/test_groupnorm_parity.py` — compare `pytorch_compatible=True` vs `False` outputs for refiner-shaped tensors
2. If outputs differ, compute the weight permutation: `perm = rearrange_for_native_grouping(G, C)`
3. Modify `refiner.py`: drop `pytorch_compatible=True`
4. Modify weight loading in `convert/converter.py` to apply permutation to refiner GroupNorm weights
5. Run `scripts/compare_reference.py` against golden.npz
6. Benchmark via `scripts/run_research_experiment.py`

### Success criteria
- Parity: max_abs_error < 5e-3 vs golden.npz
- Speed: measurable latency reduction (target: ~5-7% = 4-6ms @512)

### Rollback
Revert refiner.py + converter.py changes.

### Risk: LOW
- GroupNorm with C=64, G=8 — channel grouping may already be identical between pytorch_compatible and native modes
- If not, offline weight permutation is a well-understood transform

---

## Phase A3: mx.take vs Fancy Indexing (1 hour)

### Hypothesis
`x[:, perm, :]` dispatches JAX-style bound-checked fancy indexing. `mx.take(x, perm, axis=1)` dispatches a leaner dedicated gather kernel.

### Current state
- `hiera.py:634`: `x = x[:, self._unroll_perm, :]` (unroll)
- `hiera.py:649`: `feat = x[:, perm, :].reshape(...)` (reroll, 4x)

### Experiment
Replace both with `mx.take`:
```python
# Line 634
x = mx.take(x, self._unroll_perm, axis=1)
# Line 649
feat = mx.take(x, perm, axis=1).reshape(...)
```

### Success criteria
- Parity: exact match (this should be mathematically identical)
- Speed: any measurable improvement

### Rollback
Revert 2 lines in hiera.py.

### Risk: MINIMAL
- `mx.take` is documented as semantically equivalent for read operations
- Caveat: issue #3201 warns about `shapeless=True` + gather — we don't use shapeless

---

## Phase A4: Block-Sparse Attention Mask (1-2 days)

### Hypothesis
Replace physical unroll/reroll token permutation (9.25% GPU time) with block-sparse attention masking. Tokens stay in spatial order; mask enforces local windows.

### Why this could work
- Stages 0-1 (5 blocks): windowed attention currently implemented via reshape-batch
- Stages 2-3 (19 blocks): global attention (no windowing needed)
- Only stages 0-1 need masking; stages 2-3 are already optimal

### Design
1. Precompute block-sparse mask at init: `(1, 1, N, N)` bool/float tensor
   - For each token position, mark which positions are in the same mask unit window
   - `-inf` for out-of-window, `0` for in-window
2. Pass mask to `mx.fast.scaled_dot_product_attention(q, k, v, scale=s, mask=mask)`
3. Remove unroll call before blocks, remove reroll calls at stage ends
4. Adjust MaskUnitAttention to always use global-style attention with mask

### Challenges
- Mask is `(N, N)` where N=16384 @512 for stage 0 — that's 1GB in float32
  - Need to verify SDPA supports sparse/bool masks
  - Alternative: use additive mask in bfloat16 (512MB)
  - Alternative: keep windowed-batch approach for stages 0-1, only eliminate reroll gathers
- Q-pooling changes sequence length mid-block — mask shape must account for this
- Stage transitions change token count — separate masks per stage

### Fallback approach
If full mask replacement is too complex:
- Keep current unroll/reroll but switch to `mx.take` (Phase A3)
- Investigate mlx-mfa STEEL kernel as drop-in SDPA replacement (separate experiment)

### Success criteria
- Parity: max_abs_error < 5e-3 vs golden.npz
- Speed: measurable gather kernel reduction (target: 5-9% = 4-8ms @512)

### Rollback
Revert hiera.py changes (unroll/reroll + MaskUnitAttention).

### Risk: MEDIUM-HIGH
- Mask memory for early stages could be prohibitive
- Attention pattern must exactly match windowed behavior
- Q-pooling + mask interaction is complex

---

## Execution Order

1. **A1** (env var sweep) — free, 30 min, sets baseline
2. **A3** (mx.take) — trivial, 1 hour, may improve gathers
3. **A2** (GroupNorm remap) — moderate, 2-4 hours, targets 6.94%
4. **A4** (block-sparse mask) — complex, 1-2 days, targets 9.25%

Each is a standalone experiment following the research lab protocol:
plan -> implement -> benchmark -> score -> keep/revert -> record

## Open Questions

- GroupNorm C=64 G=8: is channel grouping actually different between pytorch_compatible modes?
- mx.take: does MLX dispatch a genuinely different Metal kernel vs fancy index?
- SDPA mask: does MLX support bool masks or only additive float masks?
- SDPA mask: what's the memory cost at N=16384?
- mlx-mfa: compatible with our MLX version (>=0.31.0)?
- Block-sparse mask + Q-pooling: how does mask shape change when Q is pooled?
