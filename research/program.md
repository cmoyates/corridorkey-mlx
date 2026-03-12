# Research Program — corridorkey-mlx Optimization

## Objective

Reduce steady-state inference latency and peak memory usage on Apple Silicon while preserving baseline fidelity (max abs error < 1e-3 vs golden reference).

## Phase 1 search areas (priority order)

### 1. MLX tile lifecycle and memory discipline
- mx materialization placement — materialize early to release graph nodes
- Delete temporary tensor refs explicitly
- gc.collect() + mx.clear_cache() timing between pipeline stages
- Avoid redundant allocations in tiled inference loop

### 2. Selective precision policy
- Refiner in float16/bfloat16 (most numerically stable component)
- Backbone stays float32 (accumulation-sensitive)
- Decoder: test bf16 with fp32 sigmoid (current approach)

### 3. Tiled inference heuristics
- Tile size sweep (256, 384, 512, 768)
- Overlap sweep (32, 64, 96, 128)
- Blending strategy (linear ramp vs cosine vs flat center)

### 4. Compile-path policy
- mx.compile for fixed-shape paths
- Warmup-aware benchmarking (exclude first N runs)
- Investigate shapeless=True for backbone subgraph only

### 5. Tensor layout / staging / contiguity
- Ensure contiguous tensors before compute-heavy ops
- Minimize NCHW-NHWC transitions
- Staging: avoid CPU-GPU roundtrips in tiled inference

## Phase 2 search areas (from upstream research 2026-03-11)

### 6. Backbone quantization (8-bit Linear layers)
- `nn.quantize(model.backbone, bits=8)` — one-liner
- 144 Linear layers in Hiera dominate compute
- Expected: ~50% weight memory reduction + potential speedup on memory-bound matmuls
- Risk: fidelity gate (must check max_abs vs golden)
- Source: MLX `nn.quantize()` API, upstream Issue #53 concept

### 7. MLX memory tuning APIs
- `mx.set_wired_limit(bytes)` — pin model weights in physical RAM, reduce p95 variance
- `mx.set_cache_limit(bytes)` — tune buffer reuse pool size
  - Tight limit -> lower peak memory, higher alloc overhead
  - Loose limit -> faster reuse, higher peak
- Both zero-risk, macOS 15+ (Darwin 25.3.0 = compatible)

### 8. Verify nn.LayerNorm -> mx.fast.layer_norm dispatch
- 48 LayerNorm calls in backbone (2 per block x 24 blocks)
- If `nn.LayerNorm` doesn't dispatch to fused `mx.fast.layer_norm`, manual swap = free speedup
- Zero fidelity risk (numerically equivalent)

### 9. Token routing — skip attention for easy tokens
- Route tokens where alpha hint near 0 or 1 to cheap O(N) path instead of O(N^2) attention
- Applied at Hiera stages 2-3 only
- Zero-init LTRM module = identity residual, works without fine-tuning
- Source: 99oblivius/CorridorKey-Engine `HintBasedTokenRouter`
- Risk: fidelity unknown — need to test identity-residual path quality

### 10. Refiner-only tiling
- Run backbone+decoder once at full res, tile only the CNN refiner
- Lower peak memory than full-model tiling (backbone intermediates allocated once)
- Clean standalone impl in CorridorKey-Engine `TiledCNNRefiner`
- Most impactful at resolutions > 512

### 11. Custom fused Metal kernels for refiner — CLOSED (2026-03-11)
- `mx.fast.metal_kernel()` works and is compatible with `mx.compile()`
- BUT: custom kernels cannot access AMX matrix hardware, so conv operations are always slower
- Viable for element-wise / gather ops, NOT for compute-intensive (matmul, conv)
- Fusing conv+GroupNorm+ReLU won't help since the conv portion is AMX-bound
- See: research/compound/dilated_conv_kernel_experiment.md

## Phase 2 experiment queue (priority order)

| Pri | Experiment | Search Area | Effort | Expected Impact |
|-----|-----------|-------------|--------|-----------------|
| 1 | `nn.quantize(backbone, bits=8)` | S6 | Low | High — weight mem + matmul speed |
| 2 | `set_wired_limit()` sweep | S7 | Low | p95 reduction |
| 3 | `set_cache_limit()` sweep | S7 | Low | Peak memory reduction |
| 4 | LayerNorm dispatch check + swap | S8 | Low | Free if not already fused |
| 5 | Token routing (identity LTRM) | S9 | Medium | 50-80% attention FLOP reduction |
| 6 | Refiner-only tiling | S10 | Medium | Memory at high res |
| 7 | Fused Metal refiner kernels | S11 | High | Bandwidth reduction |

## Phase 3 search areas (from plateau analysis 2026-03-11)

### 12. SDPA attention for Hiera
- Replace manual attention math with `mx.fast.scaled_dot_product_attention`
- Fused kernel avoids materializing full NxN attention matrix
- Applied in `MaskUnitAttention` — already windowed, SDPA handles the rest
- Risk: must match windowed attention semantics exactly

### 13. Graph materialization strategy
- Strategic materialization placement to reduce peak live tensor count
- Profile which tensors are kept alive longest, force materialization of consumed ones
- Different from stage_gc: finer granularity within backbone stages

### 14. Stream pipelining
- `mx.stream()` to overlap compute with memory operations
- Backbone stage N compute overlapped with stage N+1 weight prefetch
- MLX supports explicit stream scheduling on Apple GPU

### 15. Weight format optimization
- Convert weights to optimal memory layout at load time
- Ensure matmul operands are contiguous along the right axis
- May help with Metal shader vectorization

### 16. ELIMINATED — Operator fusion hints
- mx.compile already fuses element-wise ops (depth 11, 24 arrays max)
- Manual fusion won't beat the compiler for these ops

### 17. Matmul ordering
- `x @ W.T` faster than `x @ W` for vector-matrix (from "Writing Fast MLX" guide)
- Check attention projections and decoder linears for suboptimal ordering

### 18. mx.addmm fusion
- `mx.addmm(c, a, b)` = fused `a @ b + c` in single kernel
- Applicable to any linear layer with bias add

### 19. Dtype cast cleanup
- mx.fast functions accumulate in higher precision internally
- Remove unnecessary astype() before/after mx.fast.layer_norm, mx.fast.scaled_dot_product_attention
- Use Python scalars instead of mx.array for constants to avoid upcasting

### 20. Async pipeline
- mx.async_eval() returns immediately, CPU builds next graph while GPU executes
- Must run in separate stream via mx.new_stream(mx.gpu)
- For throughput (video), not single-image latency

### 21. Refiner dilated conv fix — CLOSED (2026-03-11)
- Dilated convolutions (dilation 2,4,8) are EXCLUDED from MLX implicit GEMM (PR #3147)
- Forces explicit im2col fallback: inflates activation memory by 9x (kernel_size^2)
- **TESTED AND DISPROVEN**: im2col+GEMM is NOT a bottleneck — it's the fast path
- Tested: naive Metal kernel (1.87x slower), SIMD Metal kernel (2.8x slower), sub-pixel decomposition (5% latency + 9% memory regression)
- Root cause: im2col enables AMX matrix hardware acceleration. Bypassing im2col = bypassing AMX.
- The 9x memory inflation is the unavoidable cost of hardware-optimized GEMM on Apple Silicon
- See: research/compound/dilated_conv_kernel_experiment.md

### 22. Edge-aware tile blend weights
- Only ramp edges that overlap with adjacent tiles, keep full weight at image boundaries
- Source: edenaion/EZ-CorridorKey `CNNRefinerModule._blend_weight()`
- Prevents alpha darkening at image edges in tiled inference
- Low effort, quality improvement (not speed)

### 23. Batched frame processing
- Process multiple frames in single forward pass (batch dimension)
- Parallel CPU postprocessing via multiprocessing pool
- Source: MarcelLieb/CorridorKey batch-processing branch
- Only relevant for video pipeline throughput, not single-image latency

## Phase 4 search areas (from upstream mining round 4, 2026-03-12)

### 24. Contiguous GroupNorm (refiner) — ACTIVE
- `nn.GroupNorm(pytorch_compatible=True)` produces non-contiguous output
- Metal trace: `g3_copyfloat16float16` = 6.94% GPU time (10 instances in refiner)
- Custom impl to avoid internal transpose while preserving pytorch-compatible semantics
- exp32 proved: native GroupNorm (no pytorch_compatible) = catastrophic fidelity failure
- Approach: rewrite normalization math to avoid reshape-transpose-reshape, or fused Metal kernel
- See: `docs/brainstorms/2026-03-12-contiguous-groupnorm-brainstorm.md`

### 25. mxfp8 quantization for backbone stage 0
- Stage 0 (dim=112) currently fp32 — can't use int4/int8 (not divisible by 32)
- mxfp8 (MLX 0.30.3+) has different alignment constraints, might work
- Would quantize the only unquantized backbone stage

### 26. 5-bit quantization for stages 1-3
- Currently int8 via safe_quantize
- MLX 0.31.1 adds 3/5/6-bit QMV kernels
- 5-bit = more compression than int8, need fidelity check

### 27. mx.clear_cache() between pipeline stages
- Different from `del backbone` (exp34 = no effect on peak memory)
- Targets Metal buffer cache specifically
- Could prevent cache bloat from backbone intermediates during decoder/refiner

### Upstream watch list (2026-03-12)
- MLX PR #3120: split-K quantized matmul (25-30% faster qmm for small M) — still open
- MLX PR #3247: per-stream locking for concurrent inference — still open
- CorridorKey #107: distilled model — closed, no checkpoint
- CorridorKey #144: quantization — closed, no action

## Out of scope (phase 1)

- Architecture redesign
- Training / finetuning
- CoreML / ANE export
- Temporal coherence / optical flow
- Alpha-hint replacement
- ROI crop-and-paste inference

## Evaluation cadence

Each experiment produces:
1. Structured JSON result in `research/artifacts/`
2. Score via `scripts/score_experiment.py`
3. Summary appended to `research/experiments.jsonl`
4. Compound note if the finding is reusable
