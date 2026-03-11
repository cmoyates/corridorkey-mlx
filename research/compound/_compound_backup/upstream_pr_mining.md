# Upstream CorridorKey PR Mining

Mined from [nikopueringer/CorridorKey](https://github.com/nikopueringer/CorridorKey) on 2026-03-10.
Repo has 4,684 stars, very active community contribution since open-source release.

---

## Merged / High-Signal Findings

### 1. PR #104 -- torch.compile + FP16 model weights + FlashAttention Hiera (MERGED)

**Source:** PR #104 by @MarcelLieb, merged 2026-03-10
**Summary:** Adds `torch.compile()` on GreenFormer, `model_precision` param (fp16/fp32), `mixed_precision` toggle, patched Hiera for FlashAttention, and `torch.inference_mode()` replacing `torch.no_grad()`.

Key changes:
- `@torch.compile(disable=(...))` decorator on `GreenFormer` (disabled on non-Linux/Windows)
- `model_precision: torch.dtype` param -- cast entire model to fp16 for 1.91 GB peak (vs 4.57 GB fp32)
- `torch.set_float32_matmul_precision("high")` when using mixed/reduced precision
- `torch.inference_mode()` instead of `torch.no_grad()`
- Patched timm fork for FlashAttention in Hiera (`Raiden129/pytorch-image-models-fix`)

**Classification:** concept-only (torch.compile/FlashAttention are CUDA-specific; mx.compile is MLX equivalent)
**Relevance:** We already have `mx.compile()` support. The FP16 model weight casting maps to our existing bf16 mixed precision work. The `inference_mode` vs `no_grad` distinction is PyTorch-only. FlashAttention is handled natively by Metal's MPS/MLX attention kernels. No direct code to port, but validates our optimization direction.

---

### 2. PR #92 -- Default MLX backend to tiled inference (MERGED)

**Source:** PR #92 by @cmoyates (us), merged 2026-03-09
**Summary:** Sets `tile_size=512, overlap=64` as defaults for MLX backend in the upstream `backend.py` factory.

Key changes:
- `create_engine()` accepts `tile_size` and `overlap` params (MLX only)
- Default: tiled inference out of the box for MLX
- Benchmark: 3279 ms/frame tiled vs 5864 ms/frame full-frame (M1 Pro), 2.5 GB vs 26.9 GB peak Metal

**Classification:** mlx-portable (this IS our code, integrated upstream)
**Relevance:** Already implemented in corridorkey-mlx. Confirms upstream has adopted our tiling as default. No action needed.

---

### 3. PR #54 -- VRAM & inference optimizations with CLI flags (DRAFT, ours)

**Source:** PR #54 by @cmoyates (us), draft since 2026-03-07
**Summary:** Phased PyTorch optimization: FP16 weight casting, GPU-side color math, decoupled backbone/refiner resolutions, tiled CNN refiner with tent blending.

Key changes in upstream model_transformer.py:
- `backbone_size` param on GreenFormer -- encoder runs at lower res, decoder/refiner at full res
- `_tiled_refine()` -- processes refiner in tiles with tent (linear ramp) blending, CPU accumulators
- `_build_tent_weight()` -- static tent weight map construction
- `refiner_tile_size` / `refiner_tile_overlap` params on GreenFormer
- Backbone downsampling via `F.interpolate` before encoder, upsample decoder logits to full `input_size`

**Classification:** concept-only (tent blending concept portable; PyTorch-specific CPU offload pattern)
**Relevance:** We already implemented decoupled backbone/refiner resolution (Opt Phase 3). The tiled *refiner* approach is interesting -- we tile the full model, but tiling just the refiner (which is the memory-heavy CNN at full res) could be more efficient. Our current `_tiled_refine` equivalent doesn't exist. The tent weight blending is a simpler alternative to our cosine ramps -- worth benchmarking.

---

### 4. Commit `f35fffe3` -- Original FP16 autocast (initial commit)

**Source:** Commit f35fffe3 by @nikopueringer
**Summary:** First optimization: `torch.autocast(dtype=torch.float16)` wrapping model forward pass.

**Classification:** concept-only
**Relevance:** We already use bf16 in MLX (better than fp16 on Apple Silicon due to no subnormal flushing). Already implemented.

---

## Open PRs -- Significant

### 5. PR #130 -- Tiled inference v2 for PyTorch (CLOSED but informative)

**Source:** PR #130 by @blackandredbot, closed 2026-03-10
**Summary:** Full tiled inference for PyTorch GreenFormer with cosine blend ramps, VRAM auto-detection, guided filter alpha refinement, original pixel restoration, and resolution-scaled matte tightening.

Key technical details:
- Tile alignment to 224px (LCM of Hiera patch_embed stride=4, q_stride^3=8, patch_stride=7)
- Model initialized at tile resolution -- pos_embed matches without runtime interpolation
- Cosine blend ramps: `0.5 - 0.5 * cos(pi * t)` with min 65px overlap (refiner RF)
- `guided_filter_alpha()` -- He et al. guided filter (O(N)/pixel) for edge-aware alpha snap
- Original pixel restoration: blend despilled original full-res pixels using alpha^0.6 as weight
- Morphological erosion + Gaussian re-feather scaled by `1 - (tile_size / img_size)`
- VRAM auto-detection: >=24GB no tiling, 12-24 1344px, 8-12 896px, <8 672px
- A10G benchmark: 5802 MiB full-frame vs 1052 MiB tiled

**Classification:** concept-only (PyTorch tiling infra; concepts already in our codebase)
**Relevance:**
- **Guided filter alpha:** NEW concept not in corridorkey-mlx. Could significantly improve tiled output quality by snapping alpha edges to image luminance boundaries. Pure numpy/OpenCV -- trivially portable to MLX (box filter = conv2d).
- **Resolution-scaled matte tightening:** We don't do this. Compensates for soft alpha edges at lower tile sizes using morphological erosion + Gaussian blur. Simple post-processing, easy to add.
- **Original pixel restoration:** Blends original full-res pixels back where alpha is high (solid regions), keeping model FG only at edges. Power curve gamma=0.6 sharpens blend weight. Novel quality improvement for tiled mode.
- **Tile alignment to 224px:** We use different alignment (our Hiera port may have different stride requirements -- verify).

---

### 6. PR #93 -- Comprehensive VRAM optimization (OPEN)

**Source:** PR #93 by @dzavada, open 2026-03-09
**Summary:** `--img-size` CLI flag, `--low-vram` bundle (tiled inference, half-res refiner, fp16 weights, periodic cache clearing), `torch.inference_mode()`, aggressive `del` of intermediates.

Key changes:
- Raised-cosine blend for tile seams
- Half-resolution refiner (~1.5 GB saved)
- `cudnn.benchmark=True`
- Periodic `torch.cuda.empty_cache()` to prevent fragmentation

**Classification:** concept-only
**Relevance:** Half-resolution refiner is interesting -- distinct from our decoupled backbone res. Running the refiner at half the input resolution, then upsampling, could reduce memory further. The cache clearing maps loosely to `mx.metal.clear_cache()` which we already call.

---

### 7. PR #124 -- Multithreaded I/O optimization (OPEN)

**Source:** PR #124 by @SpaceMarty, open 2026-03-10
**Summary:** `RuntimeThreadPool` context manager to overlap I/O with GPU inference for high-core-count CPUs.

Key changes:
- Reusable thread pool in `device_utils.py`
- Applied to `clip_manager.py` and `backend/service.py`

**Classification:** concept-only (Python threading is framework-agnostic but this is pipeline-level, not model-level)
**Relevance:** Low -- this optimizes the frame processing loop, not the model inference. If we build a CLI pipeline, overlapping I/O with MLX compute is worth considering. Not relevant to current model optimization work.

---

### 8. PR #109 -- Knowledge distillation via torchdistill (OPEN)

**Source:** PR #109 by @roshkins, open 2026-03-09
**Summary:** Proof-of-concept distillation pipeline using torchdistill to create a smaller GreenFormer variant.

**Classification:** pytorch-only (training/distillation pipeline)
**Relevance:** If a distilled model is produced, we'd need to port the smaller architecture. No action now -- monitor for a released distilled checkpoint.

---

### 9. Issue #53 -- 8-bit Quantization Support (CLOSED)

**Source:** Issue #53, closed 2026-03-10
**Summary:** Proposed `bitsandbytes` INT8 quantization for Linear layers to fit in 8GB VRAM. Replace `nn.Linear` with `bnb.nn.Linear8bitLt`.

**Classification:** pytorch-only (bitsandbytes is CUDA-only)
**Relevance:** MLX has native quantization (`nn.QuantizedLinear`) with 4-bit and 8-bit support. We could quantize the Hiera backbone's linear layers. Weight-only quantization would reduce checkpoint size (~180MB -> ~90MB at 8-bit, ~45MB at 4-bit) and potentially speed up attention. Worth investigating as a future optimization.

---

### 10. PR #126 -- BiRefNet integration (OPEN)

**Source:** PR #126 by @Warwlock, open 2026-03-10
**Summary:** Integrates BiRefNet as an alternative mask hint generator with explicit GPU memory management.

**Classification:** pytorch-only (different model entirely)
**Relevance:** None for model optimization. Could be relevant if we want to support alternative hint generators on MLX.

---

## Upstream Codebase Structure -- Optimization Opportunities

### Current upstream model architecture (as of main branch HEAD)

```
GreenFormer (model_transformer.py)
  encoder: timm Hiera (features_only=True) @ img_size
  alpha_decoder: DecoderHead(feature_channels, 256, output_dim=1)
  fg_decoder: DecoderHead(feature_channels, 256, output_dim=3)
  refiner: CNNRefinerModule(in_channels=7, hidden=64, out_channels=4)

CorridorKeyEngine (inference_engine.py)
  model_precision: torch.dtype (fp16/fp32)
  mixed_precision: bool (autocast)
  process_frame(): numpy in/out, torch internally
```

### Divergences from our MLX port worth noting

1. **Upstream now uses `@torch.compile`** on GreenFormer (Linux/Windows only). We use `mx.compile()` with `shapeless=False`. Aligned conceptually.

2. **Upstream `model_precision` param** allows casting entire model to fp16 before inference. We do bf16 mixed precision at the layer level (decoders bf16, backbone+refiner fp32). Could explore casting everything to bf16.

3. **Upstream PR #54 puts tiled refiner in GreenFormer.forward()** with CPU offload accumulators. Our tiling is at the pipeline level (full model per tile). Both approaches are valid; refiner-only tiling uses less memory per tile since the backbone runs once at full res.

4. **Upstream PR #130's guided filter** is a pure algorithmic improvement not present in either codebase's main branch. Easy win for tiled quality.

---

## Action Items (sorted by expected impact, updated 2026-03-11)

| Priority | Item | Source | Effort | Expected Impact |
|----------|------|--------|--------|-----------------|
| 1 | Backbone Linear quantization (8-bit) | MLX nn.quantize + Issue #53 | Low | Weight mem + matmul speed (144 Linears) |
| 2 | mx.set_wired_limit() sweep | MLX API | Low | p95 variance reduction |
| 3 | mx.set_cache_limit() sweep | MLX API | Low | Peak memory tuning |
| 4 | nn.LayerNorm -> mx.fast.layer_norm check | MLX API | Low | Free speedup if not already fused |
| 5 | Token routing (skip attention for easy tokens) | CorridorKey-Engine | Medium | 50-80% attention FLOP reduction |
| 6 | Refiner-only tiling | PR #54 + Engine impl | Medium | Lower peak mem at high res |
| 7 | Guided filter alpha post-processing | PR #130 | Low | Tiled edge quality |
| 8 | Fused Metal refiner kernels | mx.fast.metal_kernel | High | Bandwidth reduction |
| 9 | Monitor distilled model checkpoint | PR #109, Issue #107 | None (wait) | Smaller model if released |
| 10 | Edge-aware tile blend weights (no ramp at image boundaries) | EZ-CorridorKey | Low | Quality: prevents alpha darkening at image edges |

---

## 99oblivius/CorridorKey-Engine

**Source:** [github.com/99oblivius/CorridorKey-Engine](https://github.com/99oblivius/CorridorKey-Engine) (4 stars, active as of 2026-03-11)
**Description:** Community fork — production-oriented inference engine with optimization config system, tiled refiner, FlashAttention patching, and token routing.

### Key components

#### OptimizedGreenFormer (`CorridorKeyModule/core/optimized_model.py`)
- Extends upstream `GreenFormer` with `OptimizationConfig` dataclass (frozen, profile-based)
- **Tiled CNN Refiner** (`TiledCNNRefiner`): processes refiner in overlapping 512x512 tiles with linear blend ramps, cached blend weights. Processes tiles independently through stem→res1-4→final. Mathematically lossless (128px overlap > 65px receptive field).
- **FlashAttention patching** (`_patch_hiera_global_attention`): monkey-patches global-attention Hiera blocks to produce contiguous 4D Q/K/V for SDPA. Without this, PyTorch falls back to math backend (materializes full NxN attention matrix). Patches only `use_mask_unit_attn=False` blocks.
- **Token routing** (`HintBasedTokenRouter` + `LTRM`): routes "easy" tokens (alpha hint near 0 or 1) to lightweight LTRM module (O(N)) instead of global attention (O(N^2)). Applied at stages 2-3 only. Zero-init on LTRM fc2 → identity residual at init (checkpoint-compatible). **Requires fine-tuning** — disabled by default.

#### LTRM (Lightweight Token Refinement Module)
- Architecture: LayerNorm → Linear expand → GELU → DWConv 5x5 → GELU → Linear project → ECA (Efficient Channel Attention) residual
- Zero-init fc2 weight+bias so output is zero at init → identity via residual
- ECA: global avg pool → adaptive 1D conv → sigmoid gate (channel attention)
- Handles ragged token subsets (skips DWConv when spatial layout is broken)

#### Optimization profiles
- `original`: no optimizations
- `optimized`: flash_attention + tiled_refiner + disable_cudnn_benchmark + cache_clearing + fp16 + TF32 matmul
- `experimental`: all above + token_routing
- `performance`: max throughput — no tiling, no cache clearing, max-autotune compile, CUDA graphs

#### 4K benchmark results (RTX 4060 Laptop 8GB, DCI 4K 4096x2160, 100 frames)
- Optimized vs baseline (flash-only): **-40.7% median frame time** (4979ms vs 8402ms)
- VRAM: **-84% reserved** (1582MB vs 9792MB, baseline spills to system RAM)
- Total: 786s vs 1119s (-30%)

### Relevance to corridorkey-mlx

| Feature | Status | Notes |
|---------|--------|-------|
| Tiled refiner | Partially implemented | Our tiling is full-model; their refiner-only approach (backbone once, tile refiner) is more memory-efficient at high res. `TiledCNNRefiner` is a clean standalone impl. |
| Token routing / LTRM | Not implemented | Most impactful for large images. Identity-init means checkpoint-compatible. Needs fidelity testing — they say it requires fine-tuning. |
| FlashAttention patching | Already done | We use `mx.fast.scaled_dot_product_attention` in MaskUnitAttention. |
| OptimizationConfig | Not needed | Our config is simpler (constructor params). PyTorch-specific toggles (cudnn, CUDA graphs, TensorRT) don't apply. |
| Linear blend ramps | Already implemented | Our tiled inference uses linear ramps. |

### Action items from this repo

| Priority | Item | Effort | Expected Impact |
|----------|------|--------|-----------------|
| 1 | Refiner-only tiling (backbone+decoder once, tile refiner only) | Medium | Lower peak memory at >512 res |
| 2 | Token routing with identity LTRM (stages 2-3) | Medium | 50-80% attention FLOP reduction (needs fidelity gate) |
| 3 | ECA channel attention in LTRM | Low | Quality improvement if token routing is adopted |

---

## New Upstream Activity (since 2026-03-10)

### 11. PR #131 -- Auto-stitch comp frames into MP4 (MERGED 2026-03-11)

**Source:** PR #131 by @blackandredbot
**Summary:** After inference, automatically stitches Output/Comp PNGs into MP4 via ffmpeg at source fps. Non-fatal if ffmpeg missing.

**Classification:** pipeline-only
**Relevance:** None for model optimization. CLI/pipeline feature.

---

### 12. PR #132 -- Piecewise sRGB gamma correction for EXR (OPEN)

**Source:** PR #132 by @blackandredbot
**Summary:** Fixes EXR sequences with `input_is_linear=False` — was silently treating as linear. Replaces naive `pow(1/2.2)` with IEC 61966-2-1 piecewise sRGB transfer function.

**Classification:** color-pipeline
**Relevance:** If we add EXR support, use proper piecewise sRGB (threshold at 0.0031308). Not relevant to current model optimization.

---

### 13. PR #133 -- Cross-platform uv.lock drift fix (OPEN)

**Source:** PR #133 by @blackandredbot
**Summary:** Eliminates `uv.lock` drift from platform-conditional torch sources using extras-based backends (`uv sync --extra cuda` / `--extra mlx`).

**Classification:** build-system
**Relevance:** Good pattern for our pyproject.toml if we ever add CUDA backend support alongside MLX.

---

### 14. PR #100 -- Mac UX improvements (OPEN)

**Source:** PR #100 by @shezmic
**Summary:** `--backend auto|torch|mlx` CLI flag, platform-specific ffmpeg hints, uv check in launcher, MPS troubleshooting docs.

**Classification:** CLI/docs
**Relevance:** None for model optimization.

---

### 15. MarcelLieb/CorridorKey — Batched frame processing (branch: batch-processing)

**Source:** [MarcelLieb/CorridorKey](https://github.com/MarcelLieb/CorridorKey) batch-processing branch
**Summary:** Processes multiple frames in a single forward pass (batch dimension). Multiprocessing pool for parallel CPU postprocessing.

Key changes:
- `batch_process_frames(images: [B,H,W,3], masks: [B,H,W])` — batched preprocess → model forward → parallel postprocess
- `torch.multiprocessing.Pool(num_workers)` with `starmap` for postprocessing (despill, composite, etc.)
- `torch.cuda.empty_cache()` between preprocess→inference and inference→postprocess
- `num_workers` defaults to `cpu_count() // 2`
- `torch.compile` moved to function call for flexibility

**Classification:** concept-only (batched inference requires VRAM for multiple frames; postprocess parallelism is framework-agnostic)
**Relevance:** Low for single-image optimization. If we build a video pipeline, batching MLX inference + parallel numpy postprocess could help throughput. MLX's lazy evaluation already overlaps compute/transfer somewhat.

---

### 16. edenaion/EZ-CorridorKey (426 stars)

**Source:** [github.com/edenaion/EZ-CorridorKey](https://github.com/edenaion/EZ-CorridorKey)
**Description:** Simplified "EZ" wrapper with GUI, SAM2 tracker integration, Hiera FlashAttention patch.

Key technical details:
- Contains `patches/hiera-flashattention-v1.patch` and `README-hiera-flashattention.md`
- `sam2_tracker/` — SAM2-based object tracking integration
- Tiled refiner with edge-aware blend weights (only ramps internal edges, not image boundaries)
- `torch.compile(dynamic=False, fullgraph=True)` on refiner tile kernel
- `@torch.compiler.disable` on tiled forward (dynamic tile iteration)

**Classification:** concept-only
**Relevance:**
- **Edge-aware blend weights**: Their `_blend_weight()` only ramps edges that overlap with adjacent tiles, keeping full weight at image boundaries. Our tiled inference ramps all edges. This is a quality improvement worth adopting — prevents alpha darkening at image edges.
- **Compiled tile kernel**: `torch.compile(dynamic=False, fullgraph=True)` on fixed-size tile processing while disabling compilation on the tile loop. Maps to our approach of `mx.compile` for fixed shapes.

---

### 17. Ahmed791996/ComfyUI-YAK

**Source:** [github.com/Ahmed791996/ComfyUI-YAK](https://github.com/Ahmed791996/ComfyUI-YAK)
**Description:** ComfyUI custom nodes wrapping CorridorKey CLI via subprocess.

**Classification:** integration-wrapper
**Relevance:** None — subprocess wrapper, no model changes.

---

### 18. DCRepublic/CorridorKey_Docker_GUI

**Source:** [github.com/DCRepublic/CorridorKey_Docker_GUI](https://github.com/DCRepublic/CorridorKey_Docker_GUI)
**Description:** Docker + web frontend for CorridorKey.

**Classification:** deployment
**Relevance:** None for model optimization.

See also: `compound/mlx_framework_findings.md`, `compound/community_repo_findings.md`

---

## Unresolved Questions

- nn.LayerNorm: already dispatches to mx.fast.layer_norm internally?
- nn.quantize: can apply to backbone submodule only?
- 8-bit backbone: how much fidelity budget consumed vs golden.npz?
- mx.depends: works inside mx.compile-d graphs?
- Token routing without LTRM fine-tuning: identity residual cause quality regression?
- What fraction of tokens are "easy" at typical green screen input?
- Dilated conv implicit GEMM: 2D dispatch conditions same as 3D?
- Guided filter on MLX: mx.conv2d box filters or numpy post-processing?
- Tile alignment: upstream 224px LCM -- same constraint in our Hiera port?
- Edge-aware blend: does our tiled inference already skip ramps at image boundaries?
- Batched inference: does Hiera pos_embed work with batch_size > 1?
