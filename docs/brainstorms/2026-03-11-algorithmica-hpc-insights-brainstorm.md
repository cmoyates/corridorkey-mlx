# Algorithmica HPC Insights for CorridorKey MLX

**Date:** 2026-03-11
**Source:** https://en.algorithmica.org/hpc/
**Status:** Brainstorm

## Context

Current best: **117.18ms latency, 2143.2MB peak memory** (qkv-split-first-windowed-attn, 512x512).
19 experiments run. Key bottleneck: refiner dilated convolutions (im2col fallback, 9x activation memory inflation).

## Relevant Algorithmica Findings

### 1. Cache-Line Counting > Operation Counting

**Source:** Cache Lines, Memory Bandwidth chapters

**Principle:** Performance on memory-bound workloads correlates with *cache lines accessed*, not total operations. Strided access (stride=8+) fetches the same cache lines as sequential access despite doing 16x less arithmetic — no speedup beyond L1.

**Applicability to corridorkey-mlx:**
- The refiner's im2col fallback *explodes* spatial locality — gathering dilated conv inputs scatters reads across 9 non-contiguous positions per output pixel
- A fused Metal kernel that computes dilated conv in-place (reading neighbors directly) would access ~9x fewer cache lines than the materialized im2col buffer
- **This is the theoretical justification for why the custom Metal kernel approach (program.md Phase 3) should yield 15-20% improvement** — it's not just about fewer ops, it's about fewer cache line fetches

### 2. Hierarchical Blocking / Tiling for Matmul

**Source:** Matrix Multiplication chapter

**Principle:** Three-level blocking (L1/L2/L3) eliminated memory bottleneck entirely in matmul, achieving 75% of theoretical peak (24/32 GFLOPS). Key insight: keep one operand in L1, the other in L2, with asymmetric treatment based on access pattern (broadcast vs sequential).

**Applicability:**
- MLX's implicit GEMM for standard convolutions already does this internally
- But the im2col fallback for dilated convolutions defeats this by materializing a huge intermediate buffer that blows through L1/L2
- If writing a custom Metal kernel for dilated conv: tile the output spatially so that input reads for each tile fit in threadgroup memory (Apple GPU equivalent of L1)
- The matmul chapter's 6x16 micro-kernel pattern (12 vector registers as accumulators) maps to Metal's SIMD-group matrix operations

### 3. AoS vs SoA — NHWC Is Already Correct

**Source:** AoS and SoA chapter

**Principle:** AoS (grouping related fields together) wins when accessing multiple fields of the same element sequentially. SoA wins for columnar scans of a single field.

**Applicability:**
- NHWC layout (MLX default) = AoS for spatial positions — channels are contiguous per pixel
- This is correct for convolutions: each output pixel reads all C channels of each neighbor
- NCHW (PyTorch default) = SoA — better for channel-wise ops but worse for conv spatial access
- **No action needed** — MLX's NHWC is already the cache-friendly choice for this workload
- One caveat: LayerNorm and GroupNorm operate per-channel — verify these are using fused kernels that handle NHWC efficiently (program.md item 21.1)

### 4. Multiple Accumulators for Throughput Saturation

**Source:** SIMD Reductions, Throughput Computing chapters

**Principle:** When an instruction has latency L and throughput T, you need L*T independent accumulators to fully saturate execution units. A single accumulator creates a dependency chain that wastes cycles.

**Applicability:**
- Directly relevant to LayerNorm/GroupNorm: computing mean and variance requires reduction over spatial dimensions
- If `nn.LayerNorm` dispatches to `mx.fast.layer_norm` (fused kernel), this is handled internally
- If NOT fused: the reduction is a serial dependency chain — verifying this dispatch is high-value, low-effort (program.md Phase 2, item 1)
- Also relevant to any custom Metal kernel: structure reductions with multiple accumulators per SIMD lane

### 5. Non-Temporal Writes for Large Outputs

**Source:** Memory Bandwidth chapter

**Principle:** Write-heavy workloads on large buffers benefit from non-temporal (streaming) stores that bypass cache, avoiding read-for-ownership overhead. Achieved >2x throughput for large datasets.

**Applicability:**
- The decoder upsampling path writes large output tensors (full resolution alpha + fg maps)
- If these outputs are only consumed once downstream (by refiner), non-temporal writes would avoid polluting L2/L3 with data that won't be reused
- **Limited direct control in MLX** — this would require Metal kernel level control
- But relevant if writing custom refiner kernel: output writes should use `device.store()` without cache, since refiner output is final

### 6. Bandwidth vs Compute Bound Classification

**Source:** Memory Bandwidth chapter

**Principle:** Performance drops dramatically when working set exceeds cache capacity. L1: ~16 GFLOPS, RAM: ~2 GFLOPS (8x drop). Identifying which regime each operation lives in determines which optimizations matter.

**Applicability:**
- **Backbone attention (19 global blocks):** Likely compute-bound at 512x512 (Q*K^T matmul dominates). Already uses SDPA — limited further gain from memory optimization.
- **Backbone MLP:** bandwidth-bound if weights don't fit in cache. 8-bit quantization tested, no speedup — confirms weights already fit or MLX kernel is already optimal.
- **Refiner (4 dilated ResBlocks):** Memory-bound due to im2col inflation. The 9x memory expansion pushes working set well beyond cache — this is why it's THE bottleneck.
- **Decoder upsampling:** Bandwidth-bound (bilinear interpolation = low arithmetic intensity). Already fused in pairs — further gain unlikely.

### 7. Arithmetic Intensity as Decision Framework

**Source:** Bandwidth chapter, Matmul chapter

**Principle:** Arithmetic intensity = FLOPs / bytes transferred. Operations below the roofline ridge point are memory-bound; above are compute-bound. Optimization strategy differs completely between regimes.

**Applicability — per-component analysis:**

| Component | Est. Arithmetic Intensity | Regime | Optimization Strategy |
|---|---|---|---|
| Backbone attention | High (matmul) | Compute | Already SDPA'd. Token routing = skip compute |
| Backbone MLP | Medium (linear) | Mixed | Quantization didn't help. Likely near-optimal |
| Backbone LayerNorm | Very low (reduction) | Memory | Verify fused dispatch. If not fused = easy win |
| Refiner conv (dilated) | Low (im2col kills it) | Memory | Custom Metal kernel = biggest opportunity |
| Refiner GroupNorm | Very low | Memory | Fuse with preceding conv in custom kernel |
| Decoder upsample | Very low | Memory | Already paired. Near-optimal |

## Prioritized Experiment Ideas (from Algorithmica insights)

### Tier 1 — High confidence, directly supported by theory

1. **Verify LayerNorm fused dispatch** (program.md 21.1)
   - Algorithmica justification: reduction operations need fused kernels to avoid serial dependency chains
   - Effort: 1 hour investigation
   - Expected: 0-5% latency if currently unfused

2. **Custom Metal kernel for dilated conv** (program.md Phase 3)
   - Algorithmica justification: im2col destroys spatial locality (cache line counting principle); fused kernel accesses 9x fewer cache lines and eliminates intermediate buffer
   - Effort: High (days)
   - Expected: 15-20% latency + 15-20% memory

### Tier 2 — Medium confidence, informed by theory

3. **Fused conv+GroupNorm+GELU per refiner block**
   - Algorithmica justification: eliminates intermediate tensor materialization between ops (bandwidth reduction); keeps data in registers/threadgroup memory through the full block
   - Prerequisite: Metal kernel from #2
   - Expected: Additional 5-10% on top of #2

4. **Matmul ordering audit (`x @ W.T` vs `x @ W`)**
   - Algorithmica justification: transposing one operand for sequential access yielded 30% speedup in their matmul benchmarks
   - Effort: 2-3 hours
   - Expected: 0-3% — MLX likely handles this internally, but worth verifying

### Tier 3 — Speculative, loosely connected

5. **Spatial tiling of refiner input** — process refiner in cache-friendly tiles rather than full-resolution
   - Algorithmica justification: hierarchical blocking principle; keep working set in fast cache
   - Complication: dilated convs need halo regions, overlap blending
   - Expected: Memory reduction at high res, minimal latency impact

## Key Takeaway

Algorithmica's core message maps cleanly to the existing research program: **the refiner's im2col fallback is a textbook memory-bound bottleneck**, and the custom Metal kernel approach is theoretically well-justified by cache-line counting and arithmetic intensity analysis. The secondary win (LayerNorm fused dispatch verification) is low-hanging fruit.

## Open Questions

- Does `mx.fast.layer_norm` actually dispatch for Hiera's LayerNorm calls? (48 instances)
- What's the threadgroup memory budget on M-series GPUs for a fused dilated conv kernel?
- Has MLX's implicit GEMM improved for dilated convs since PR #3147?
- Is `mx.fast.metal_kernel()` stable enough for production use in current MLX version?
