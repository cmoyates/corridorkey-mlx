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

### 11. Custom fused Metal kernels for refiner
- `mx.fast.metal_kernel()` — JIT custom Metal shaders
- Fuse conv+GroupNorm+GELU per refiner block to reduce memory traffic
- Dilated convs in refiner disqualified from implicit GEMM (explicit im2col fallback)
- High effort, high reward for bandwidth-bound refiner

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

### 21. Refiner dilated conv fix (ROOT CAUSE)
- Dilated convolutions (dilation 1,2,4,8) are EXCLUDED from MLX implicit GEMM (PR #3147)
- Forces explicit im2col fallback: inflates activation memory by 9x (kernel_size^2)
- This is likely THE dominant bottleneck for both latency and memory
- Option A: Replace with stride-2 downsample + standard conv + bilinear upsample (needs retraining)
- Option B: Custom Metal kernel via mx.fast.metal_kernel (same math, no im2col)
- Expected: 15-20% latency + 15-20% peak memory reduction

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
