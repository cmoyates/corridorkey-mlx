# PyTorch Optimization Learnings for MLX Port

**Date:** 2026-03-09
**Status:** Draft
**Sources:**
- [PR #104](https://github.com/nikopueringer/CorridorKey/pull/104) (MarcelLieb) — fp16/compile/GPU preprocessing
- [CorridorKey_Test](https://github.com/Raiden129/CorridorKey_Test) (Raiden129) — FlashAttention/tiled refiner/token routing/cache clearing
- Deep research: MLX Vision Transformer Optimization Techniques (local)

## What We're Exploring

Extract optimization techniques from PyTorch CorridorKey forks, assess MLX applicability, identify net-new improvements for our port.

## Source Analysis

### PR #104 (MarcelLieb) — VRAM: 18 GB -> 1.9 GB

| Technique | Description | MLX Status |
|---|---|---|
| `torch.compile()` on GreenFormer | Graph-level JIT fusion | **Done** — `mx.compile()` in pipeline.py |
| Full fp16 model weights | `model.to(model_precision)` | **Done** — bf16 decoders, fp32 backbone/refiner |
| Mixed precision toggle | Conditional `torch.autocast` | **Done** — per-component dtype in GreenFormer |
| `torch.inference_mode()` | Faster than `no_grad()` | **N/A** — MLX has no grad tracking in inference |
| `set_float32_matmul_precision("high")` | TF32 matmuls | **N/A** — Apple Silicon different numerics |
| GPU-side preprocessing | Moved normalize/resize from numpy/cv2 to torch tensors on device | **NEW** — worth investigating |

### Raiden129/CorridorKey_Test — VRAM: 9.8 GB -> 1.6 GB (84% reduction), 40% faster

| Technique | Description | MLX Status |
|---|---|---|
| FlashAttention patching | Squeeze 5D->4D Q/K/V for SDPA dispatch | **Different** — MLX has own heuristic (see below) |
| Tiled CNN refiner | 512x512 tiles, 128px overlap, linear blend | **Done** — tiling.py (but uses numpy accumulators) |
| cuDNN benchmark disable | Avoid workspace allocation | **N/A** — no cuDNN on Metal |
| Strategic cache clearing | `empty_cache()` between encoder/decoder/refiner | **Partial** — done in tiling, NOT in non-tiled path |
| Token routing (experimental) | Route easy tokens to LTRM, edge tokens to full attention | **NEW** — novel compute reduction |

## Net-New Opportunities

### 1. Verify MLX Attention Memory Behavior

**Problem:** MLX's `mx.fast.scaled_dot_product_attention` has a dynamic heuristic choosing between O(N) streaming (flash-like) and O(N^2) explicit materialization. We don't know which path our Hiera backbone triggers.

**Action:** Profile attention memory scaling empirically:
1. Vary sequence length N (simulate different resolutions)
2. Track `mx.metal.get_peak_memory()` after each attention call
3. Plot peak memory vs N — quadratic = materialization, linear = streaming

**Impact:** If O(N^2) path is triggered at our token counts (~16K at 2048x2048), we may need to ensure contiguous 4D tensors entering SDPA — similar to Raiden129's patch but for Metal.

**Key finding from research:** Mask dtype must match Q/K/V dtype or entire computation upcasts to float32, halving bandwidth. Verify our attention masks (if any) match bf16 precision.

### 2. GPU-Side Preprocessing

**Problem:** Our engine likely does ImageNet normalization and resizing in numpy before converting to MLX arrays. On unified memory this is less costly than PCIe, but still leaves GPU ALUs idle during preprocessing.

**Action:**
- Audit `engine.py` preprocessing path
- Move normalize/resize to MLX operations
- Wrap in `@mx.compile` for kernel fusion (requires static input shapes)

**Caveat:** MLX lacks built-in bicubic resize. Manual implementation needed via `mx.meshgrid` + bilinear interpolation. Must use `mx.minimum`/`mx.maximum` for bounds clamping (no dynamic boolean indexing in MLX).

**Impact:** Moderate — eliminates numpy->mlx conversion overhead and uses GPU for parallel pixel operations.

### 3. Tiled Refiner: GPU Tensor Accumulators

**Problem:** Our tiled inference uses numpy accumulators for blend-weight averaging. This forces GPU->CPU transfer per tile and CPU->GPU for final result.

**Action:** Replace numpy accumulator arrays with MLX arrays. Accumulate blend weights and tile outputs entirely on GPU. Only convert final result to numpy at output.

**Impact:** Eliminates per-tile roundtrip. Especially significant at high tile counts (large images).

### 4. Non-Tiled Path: Strategic Memory Management

**Problem:** We do `gc.collect()` + `mx.metal.clear_cache()` in the tiling loop, but the non-tiled inference path doesn't have stage-boundary memory management.

**Action:** Add three-step cleanup protocol between pipeline stages in non-tiled forward:
1. `mx.eval()` on stage output tensors (force computation)
2. `del` intermediate tensors from previous stage
3. `gc.collect()` + `mx.metal.clear_cache()`

Insert at: encoder->decoder boundary, decoder->refiner boundary.

**Research confirms:** Just calling `mx.eval()` is insufficient. MLX's caching allocator hoards freed Metal buffers. Must explicitly `clear_cache()` to return memory to OS. This strictly bounds peak memory to the largest single stage rather than accumulated total.

**Impact:** Could significantly reduce peak memory in non-tiled path. Especially important for large img_size.

### 5. Token Routing (Experimental, Deferred)

**Problem:** Stage 2 has 16 blocks of global attention — dominates backbone compute. Many tokens are "easy" (solid FG/BG per alpha hint) and don't need full O(N^2) attention.

**Raiden129's approach:** LTRM module (LayerNorm->Linear->GELU->DWConv->Linear->ECA) at O(N) cost. Route by thresholding downsampled alpha hint. Zero-init fc2 weights for checkpoint compatibility.

**MLX challenge:** Three sparse processing strategies, each with tradeoffs:

| Strategy | Pros | Cons | Best When |
|---|---|---|---|
| `mx.where` + padding | Static shapes, compile-friendly | Doesn't reduce actual FLOPs | Low sparsity, many layers |
| Scatter + overflow bin | GPU-resident, physical reduction | Complex index arithmetic | Fixed-ratio routing |
| NumPy boolean indexing | Dynamic shapes, simple | CPU sync breaks async pipeline | Early, high-ratio culling (>60%) |

**Recommendation:** For CorridorKey, alpha hints typically have ~60-70% easy tokens. But routing happens at every block (16x in stage 2), so per-block CPU sync would be devastating. Best approach: `mx.where` with attention masking — keeps shapes static, compile-friendly, but note SDPA still processes padded tokens. Net benefit uncertain without benchmarking.

**Decision:** Defer until other optimizations landed. Needs careful profiling to verify compute savings > overhead.

### 6. Quantization (Future)

**Research findings:**
- **Int8:** ~50% memory reduction, ~1.8x speedup, <1% accuracy drop. Safe for all linear layers.
- **Int4:** ~75% memory reduction, ~2.4x speedup, 2-5% accuracy drop. Risky for matting precision.
- **MXFP4:** Best of both — 75% reduction, >2.4x speedup, <1.5% drop. M3/M4 optimized.

**Critical:** Leave Conv2d patch projection, LayerNorm, and softmax in full precision. These are pathologically sensitive to quantization in ViTs.

**For CorridorKey specifically:** Matting requires sub-pixel alpha precision. Int8 likely safe, Int4/MXFP4 needs quality validation on real footage. `mlx.nn.quantize` targets Linear/Embedding by default — Conv2d naturally excluded.

**Decision:** Defer. Profile Int8 as first candidate after other optimizations land.

## Key Decisions

1. **Attention profiling first** — verify O(N) vs O(N^2) behavior before optimizing attention path
2. **GPU preprocessing** — move normalize/resize to MLX, wrap in `@mx.compile`
3. **GPU tensor accumulators** — replace numpy accums in tiling with MLX arrays
4. **Non-tiled cache clearing** — add stage-boundary memory management protocol
5. **Token routing deferred** — MLX sparse processing constraints make ROI uncertain
6. **Quantization deferred** — Int8 first candidate, needs quality validation

## Priority Order

1. Non-tiled cache clearing (low effort, high impact on peak memory)
2. GPU tensor accumulators in tiling (medium effort, eliminates roundtrips)
3. Attention memory profiling (research, informs future work)
4. GPU preprocessing (medium effort, moderate speedup)
5. Token routing (high effort, uncertain ROI on MLX)
6. Int8 quantization (medium effort, needs quality validation)

## Open Questions

- What's our actual attention kernel path — O(N) or O(N^2) at 2048 resolution?
- How much time is spent in numpy preprocessing vs inference?
- Does `mx.metal.clear_cache()` between stages actually reduce peak memory in practice, or does lazy graph already handle it?
- For token routing: what % of tokens are "easy" in typical green screen footage?
- Int8 quantization: acceptable alpha precision loss threshold?
