# CorridorKey MLX — Optimization Summary

This document covers 57 experiments optimizing CorridorKey's MLX inference backend on Apple Silicon. It's intended for anyone working on MLX inference performance for this model or similar architectures (ViT backbone + CNN refiner matting models).

## Model architecture

CorridorKey is a video matting model that separates foreground from a green screen:

- **Backbone**: Hiera-Base-Plus (24 transformer blocks, 4 stages) — extracts multiscale features from a 4-channel input (RGB + coarse alpha hint)
- **Decoder**: Two SegFormer-style heads (alpha + foreground) — fuse multiscale features via linear projections + bilinear upsampling
- **Refiner**: CNN with 4 dilated residual blocks + 9 GroupNorm layers — sharpens edges using full-resolution RGB + coarse predictions

The model was trained at 2048×2048. At inference, we use **tiled processing** (512px tiles, 128px overlap) to handle arbitrary resolutions without OOM. A 1920×1080 frame = 15 tiles.

## Baseline and results

| Configuration | 37-frame clip @ 1920×1080 |
|---|---|
| **PyTorch (MPS)** | **3:34** |
| **MLX (optimized)** | **2:04** |
| **Speedup** | **1.72×** |

The MLX model inference is near locally optimal. Remaining gains are in pipeline overhead (ffmpeg decode, numpy postprocessing, despill/despeckle, file I/O), which accounts for roughly half of wall-clock time.

---

## What worked

These optimizations are cumulative and all active in the current default configuration.

### Operator fusion and dispatch reduction

| Experiment | What it does | Why it helps |
|---|---|---|
| Folded BatchNorm | Precompute scale+offset at load time | 2 ops vs 5 per BN layer |
| Conv1×1 → addmm bypass | Use `mx.addmm` instead of `mx.conv2d` for 1×1 convolutions | Avoids conv dispatch overhead for pointwise ops |
| Split-fuse decoder | Process alpha and fg heads without intermediate concat | Eliminates a large temporary allocation |
| Einsum fused output projection | Fuse attention output reshape + projection | One op instead of reshape→transpose→matmul |
| Split QKV + pretranspose MLP weights | Precompute split Q/K/V and transposed MLP weights at load time | Contiguous memory access patterns in hot loops |
| Precomputed unroll/reroll via mx.take | Single gather operation for Hiera's token reshuffling | Replaces 3-step reshape→transpose→reshape chains |

### Precision management

| Experiment | What it does | Why it helps |
|---|---|---|
| Decoder BF16 weights | Load decoder weights in BF16 | Halves decoder memory bandwidth |
| BF16 coarse sigmoid path | Keep BF16 through sigmoid and final addition | Fewer dtype conversions |
| Deferred FP32 cast | Only cast to FP32 at output boundary | Avoids redundant precision upcasts |

**Important**: Backbone stage 0 (dim=112) is precision-sensitive and MUST stay FP32. BF16 activations in stage 0 cause fidelity regression.

### Memory and scheduling

| Experiment | What it does | Why it helps |
|---|---|---|
| Per-component mx.compile | Compile backbone, decoders, refiner individually | Eager async_eval between components enables better scheduling |
| Async eval at stage boundaries | `mx.async_eval` between backbone→decoder→refiner | CPU builds next graph while GPU executes current |
| Dual-stream decoder dispatch | Alpha decoder on default stream, FG decoder on secondary | Overlaps two independent decoder forward passes |
| Tile eval between tiles | `mx.eval` after each refiner tile | Frees im2col buffers (9× inflation from dilated convs) before next tile |
| Cache limit 1536MB | `mx.set_cache_limit(1536 * 1024 * 1024)` | Forces Metal buffer reuse, reduces peak memory |
| **MLX buffer env vars** | `MLX_MAX_MB_PER_BUFFER=2, MLX_MAX_OPS_PER_BUFFER=2` | **17% faster** in isolated benchmarks. Small buffers force frequent eval, preventing computation graph buildup that hurts tiled workloads. |

### Custom Metal kernels

| Experiment | What it does | Why it helps |
|---|---|---|
| Metal GroupNorm v2 | Two-kernel approach: shared-mem stats reduction + fully parallel normalize | -67% vs `nn.GroupNorm(pytorch_compatible=True)`. Eliminates NHWC↔NCHW transposes. |
| Frozen GN stats mode | Metal kernel accepts precomputed (mean, var) for tiled inference | **Eliminates tiling artifacts completely** — 0.0 error vs non-tiled reference. Per-tile stats cause 68/255 max error at tile boundaries. |

---

## What didn't work (and why)

Understanding *why* these failed is more valuable than knowing they failed. The failure modes fall into clear categories.

### Category 1: Matting is edge-sensitive

Matting models produce per-pixel alpha values where edges (hair, fingers, silhouettes) matter most. Any optimization that degrades spatial or temporal edge fidelity fails catastrophically, even if mean error is tiny.

| Experiment | Approach | Result | Why |
|---|---|---|---|
| V7: Backbone resolution decoupling | Run backbone at lower res, refiner at full res | Even 12% downscale → 91/255 max edge error | Backbone provides spatial features for edge localization. Downscaling destroys sub-pixel edge detail the refiner can't recover. |
| V5: Feature caching (S2-S3) | Cache deep backbone features across frames | S2+S3: 247/255 max error between consecutive frames | S2 features (stride-16, 448ch) change significantly between real frames. Only S3 (stride-32) is stable, but it's only 3/24 blocks → 1.6% speedup. |
| V1: Output-space EMA | Blend final outputs across frames | Fails fidelity at ALL blend values | Temporal lag on edges visible even at α=0.95 |
| Feature-space EMA | Blend decoder features before refiner | α=0.9 (90% current): 70.5/255 max error | Edge features in decoder outputs shift significantly between frames. Blending smears them. |
| Refiner strided subsample | Subsample refiner input spatially | Quality loss | Same principle — spatial detail lost in subsampling |

**Takeaway**: For matting, any optimization must be bit-exact or near-exact. Approximate techniques that work for classification/detection (where mean accuracy matters) fail here because max error on edges is what determines visual quality.

### Category 2: Apple Silicon architecture constraints

Apple Silicon's unified memory and single GPU change which optimizations are profitable.

| Experiment | Approach | Result | Why |
|---|---|---|---|
| **Int8 quantization** | Quantize backbone stages 1-3 (int8 weights) | **11% SLOWER** | Dequantize-multiply overhead exceeds bandwidth savings. Unified memory means memory bandwidth is already shared efficiently. |
| Stream overlap | Run two GPU streams in parallel | No GPU-GPU parallelism | Apple Silicon has one GPU. `mx.stream` provides scheduling hints but not parallel execution. |
| Wired memory limit | Pin memory as resident via `mx.set_wired_limit` | No latency benefit | Useful on systems with paging; Apple Silicon unified memory rarely pages for our model size. |
| GEMM pad stage 0 K=112→128 | Pad dim to power-of-2 for better GEMM tiling | Regression | Padding overhead > alignment benefit at this scale. |

**Takeaway**: Don't assume discrete-GPU optimizations transfer. Int8 quantization, multi-GPU parallelism, and memory pinning are either unhelpful or counterproductive on Apple Silicon.

### Category 3: Not a bottleneck at production scale

Some optimizations show large improvements in isolation but don't move the needle on the real pipeline.

| Experiment | Micro-bench | Pipeline impact | Why |
|---|---|---|---|
| Metal GroupNorm v2 | -67% on GroupNorm | 2% total pipeline | GroupNorm is ~5ms/tile out of 227ms/tile. 15 tiles × 5ms = 75ms out of 3400ms/frame. |
| GELU fast approx | 0ms | 0ms | GELU compute is negligible relative to attention and convolutions. |
| Batch frame processing | — | Linear scaling (B=4 = 4× B=1) | GPU already fully utilized at B=1. Batching just uses more memory. |
| RLT token deduplication | — | N/A (not implemented) | Hiera's global attention stages have only 1024 and 256 tokens at 512px tiles. Windowed stages (majority of compute) don't benefit from token reduction. |

**Takeaway**: Always profile at production resolution and tile size before investing in an optimization. Micro-benchmarks at 512px with batch size 1 overstate impact.

### Category 4: Correct but not profitable

| Experiment | What happened |
|---|---|
| V6: Tile skip + frozen GN | Frozen GN works perfectly (0.0 tiling error). But skip rate is 0% at production tile sizes — large subjects fill every tile. Only useful for footage with small subjects or lots of empty background. |
| V4: Frozen GroupNorm (Python path) | Correct but 22% slower due to Python-side mean/var computation. Fixed by routing through Metal kernel (exp 51). |

---

## Architecture profile

Per-tile timing at 512×512 (single tile, steady state):

| Component | Time (ms) | % |
|---|---|---|
| Backbone stage 0 (2 blocks, windowed attn, dim=112) | 4.6 | 5% |
| Backbone stage 1 (3 blocks, windowed attn, dim=224) | 5.1 | 6% |
| **Backbone stage 2 (16 blocks, global attn, dim=448)** | **23.2** | **27%** |
| Backbone stage 3 (3 blocks, global attn, dim=896) | 3.8 | 4% |
| Decoders (alpha + fg, dual-stream) | ~5 | 6% |
| Refiner (9 GroupNorms, 4 dilated ResBlocks) | ~30 | 35% |
| Per-tile overhead (eval, slice, concat) | ~15 | 17% |
| **Total** | **~87** | **100%** |

At 1920×1080: 15 tiles × ~87ms ≈ 1.3s model inference + ~0.7s pipeline overhead ≈ **2.0s/frame**.

Stage 2 (16 blocks of global attention at dim=448) and the refiner are the two dominant costs. All known optimizations for these have been exhausted or rejected.

---

## Where to look next

The MLX model inference is at a local optimum after 57 experiments. Remaining speedup opportunities:

### 1. Pipeline overhead (highest impact)
~50% of wall time is non-MLX: ffmpeg decode, numpy uint8↔float32, despill/despeckle (numpy/OpenCV), compositing, file I/O. Optimizing the CorridorKey pipeline itself has more headroom than any further model optimization.

### 2. Fewer tiles
1920×1080 with 512px tiles = 15 tiles/frame. Reducing tile count is the most direct path to faster inference:
- Larger tiles (768, 1024) — needs more peak memory but fewer backbone runs
- Non-square tiles optimized for common aspect ratios
- Adaptive tiling that skips background regions (requires frozen GN for correctness)

### 3. Native runtime
Bypass Python entirely. Upstream [CorridorKey-Runtime](https://github.com/99oblivius/CorridorKey-Runtime) implements a C++ native runtime. Eliminates Python overhead, numpy conversions, and GIL contention.

### 4. Optical flow feature warping (V8)
The only theoretically viable temporal optimization. Warp cached S2-S3 features using motion vectors (Apple Vision's `VNGenerateOpticalFlow` runs on ANE). High complexity, uncertain payoff, and requires solving feature warping at stride-16 resolution.

---

## Experiment log

Full structured data: `research/experiments.jsonl` (57 entries with timing, fidelity, and verdict).

Compound notes with detailed analysis: `research/compound/` (27 documents).

Golden references: `reference/fixtures/golden.npz` (512×512) and `reference/fixtures/golden_2048.npz` (2048×2048).
