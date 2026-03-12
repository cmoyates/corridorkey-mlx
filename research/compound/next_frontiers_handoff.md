# Next Frontiers — Handoff Doc

Metal trace analysis (2026-03-12) concluded application-level optimization is exhausted. Remaining gains require framework-level or algorithmic changes. This doc captures the specific opportunities with enough context to pick up later.

---

## 1. GroupNorm transpose overhead (6.94% of GPU time)

**What**: `g3_copyfloat16float16` — 8.4M SIMD groups of same-type contiguity copies. Root cause: `nn.GroupNorm(pytorch_compatible=True)` reshapes + transposes internally, producing non-contiguous views. Subsequent conv2d/relu ops force a contiguous copy.

**Where**: Refiner has 10 GroupNorm calls (stem + 2 per ResBlock x4). Each triggers a copy after the internal transpose-back.

**Possible approaches**:
- Custom `FusedGroupNorm` that keeps data contiguous throughout — normalize in (B*H*W, G, C//G) layout, apply per-channel affine via reshaped weight (G, C//G), reshape back once. Challenge: mx.fast.layer_norm weight must match last dim (C//G=8), but each group needs its own weight slice.
- Upstream MLX PR: GroupNorm could apply affine BEFORE transposing back, or output a contiguous array. File: `mlx/nn/layers/normalization.py` in ml-explore/mlx.
- Skip pytorch_compatible and revalidate parity — the non-pytorch path avoids transpose entirely (uses mean/var directly). Would need to recheck fidelity against golden.npz.

**Measurement**: trace the `g3_copyfloat16float16` kernel count before/after. Should drop from ~10 instances to ~0 if successful.

---

## 2. Gather kernel overhead (9.25% of GPU time)

**What**: `gatherbfloat16uint32_3_3_int` — 3.1M SIMD groups across 5 dispatches. These are `x[:, perm, :]` indexed gathers for Hiera's unroll (1x) and reroll (4x at stage ends).

**Where**: `src/corridorkey_mlx/model/hiera.py` lines 634 (unroll) and 649 (reroll). Precomputed permutation indices in `self._unroll_perm` and `self._reroll_perms`.

**Why it's hard**: Gathers replaced 3x reshape-transpose-reshape chains that were even slower. The permutation is non-trivial (interleaves spatial positions for mask-unit attention windows).

**Possible approaches**:
- Restructure Hiera attention to operate on spatial layout directly (no unroll/reroll). Major refactor — would change every block's input layout. Timm's Hiera implementation uses the same unroll pattern, so no reference exists for a spatial-native variant.
- `mx.take(x, perm, axis=1)` instead of `x[:, perm, :]` — might dispatch a different (faster) kernel. Quick test.
- Fuse gather with subsequent operation — if the gather feeds into a matmul, fold permutation into weight matrix. Only works for the first op after gather; doesn't apply cleanly since gather feeds into attention blocks with multiple ops.
- Reduce gather count — currently 4 rerolls (one per stage end). Could defer reroll to only the stages needed by the decoder (all 4 are used, so no savings here).
- Smaller gathers at later stages — stages 2-3 have fewer tokens (1024, 256 vs 16384). Check if the 5 dispatches have genuinely identical sizes or if Xcode is grouping them. If later stages are smaller, the gather cost is front-loaded on stage 0-1.

**Measurement**: profile with `mx.take` vs fancy indexing; count kernel dispatches.

---

## 3. Softmax in attention (3.95%)

**What**: `block_softmax_float32` — 1.1M SIMD groups. Likely internal to `mx.fast.scaled_dot_product_attention`, not separately optimizable from application code.

**Possible approach**: Check if SDPA has a fused softmax path on M3 Max or if it always dispatches a separate softmax kernel. If separate, this is an MLX framework optimization opportunity.

---

## 4. Broadcast add / residual connections (2.99%)

**What**: `Cf2IBroadcastBDf2OAddAC` — 4.2M SIMD groups. These are elementwise adds in residual connections (both Hiera blocks and refiner ResBlocks).

**Possible approach**: Verify these aren't redundant (e.g., adding zero bias). Otherwise, this is fundamental compute — 2.99% is the cost of doing residual connections.

---

## 5. MLX_MAX_MB_PER_BUFFER sweep

**What**: Environment variable controlling Metal command buffer commit frequency. Never tested. Lower = more frequent commits (lower peak memory, more dispatch overhead). Higher = larger batches.

**Quick test**: `MLX_MAX_MB_PER_BUFFER=32 uv run python scripts/bench_mlx.py` (try 16, 32, 64, 128, 256). Compare median latency and peak memory.

---

## Reference

- Trace file: `trace.gputrace` (512x512, ~105ms GPU time)
- Full trace analysis: `research/compound/metal_trace_512_findings.md`
- Local optimum finding: `research/compound/2026-03-12-metal-trace-local-optimum.md`
- Current best: 83ms @512 per-component, 422ms @1024
- M3 Max bandwidth: ~353 GB/s measured (400 theoretical)
