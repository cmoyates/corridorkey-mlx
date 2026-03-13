# Video Matting Upstream Research — Temporal Optimization & Benchmarks

Research date: 2026-03-13
Context: CorridorKey MLX port, 422ms/frame @1024, backbone skip (V2) rejected due to 4ch input

---

## 1. Video-Specific Optimizations in Matting/Segmentation Models

### 1.1 Recurrent Temporal State (RVM approach)

**Robust Video Matting (WACV 2022)** uses ConvGRU recurrent units at 1/16 resolution to propagate temporal state. Key design decisions:

- Backbone runs every frame but at **heavily downsampled resolution** (downsample_ratio=0.25 for HD, 0.125 for 4K)
- Recurrent state captures temporal context without explicit feature caching
- MobileNetV3 backbone for speed; ResNet50 for quality
- **FPS on RTX 3090 FP16**: 172 FPS @1080p, 154 FPS @4K (tensor throughput only)
- **Classification**: pytorch-only (ConvGRU requires training; cannot retrofit to CorridorKey)

**Why this matters for us**: RVM's core insight is that backbone downsampling + recurrent state is more robust than backbone skipping. Our decoupled resolution (backbone_size < img_size) already implements the downsample half. The recurrent half requires training.

### 1.2 Memory Bank Propagation (MatAnyone / SAM2 approach)

**MatAnyone (CVPR 2025)** and **MatAnyone 2 (CVPR 2026)** use memory-propagation backbones:

- FIFO memory bank stores foreground embeddings from initial + historical frames
- Cross-attention between current frame features and memory bank features
- First-frame recurrent refinement: repeat first frame multiple times to bootstrap memory
- Reference-frame selection for long-range appearance changes
- Patch dropout during training prevents overfitting to reference content
- **Classification**: concept-only (requires trained memory attention; architecture incompatible)

**SAM2** uses Hiera backbone (same as CorridorKey) with memory bank:

- Memory bank stores spatial feature maps from recent N frames + prompted M frames
- Memory encoder: downsample mask + element-wise sum with frame embedding + lightweight conv fusion
- Cross-attention at 64-dim projection (memory features) with 256-dim object pointer split into 4x64 tokens
- Only stride 16/32 features (stages 3-4) used in memory attention; stride 4/8 (stages 1-2) bypass memory and go directly to decoder upsampling
- **Classification**: concept-only (SAM2's memory module is trained end-to-end)

**Key architectural insight from SAM2**: High-level features (stages 3-4) benefit from cross-frame attention. Low-level features (stages 1-2) are frame-local and don't need temporal propagation. This validates that **early backbone stages are less cacheable** than later ones — the opposite of what naive feature reuse assumes.

### 1.3 Backbone Skip / Feature Reuse (DFF / Accel approach)

**Deep Feature Flow (CVPR 2017)**:

- Full backbone on keyframes; optical flow warp of cached features on non-keyframes
- 10x speedup but 4.4 mIoU drop (73.9 -> 69.5) on Cityscapes
- Only works with 3ch RGB input (no hint channel)

**Accel (CVPR 2019)** improves on DFF:

- Dual-network: full backbone on keyframes, lightweight backbone on non-keyframes
- Fuses warped cached features with lightweight fresh features via learned score
- At keyframe interval 5: Accel-101 achieves 75.5 mIoU vs DFF's 68.7
- At interval 8-10: Accel maintains 70-75 mIoU while DFF drops to ~60
- **Classification**: concept-only (score fusion network is trained)

**Our experiment (V2 skip2)**: REJECTED. CorridorKey's 4ch input (RGB + alpha hint) means cached features carry stale hint information. Per-pixel max_abs ~1.0 at motion boundaries. PSNR drops to 16-24 dB on skipped frames. See `2026-03-13-backbone-skip-rejected-hint-in-backbone.md`.

### 1.4 Run-Length Tokenization (NeurIPS 2024)

**"Don't Look Twice"** — content-aware token deduplication for video ViTs:

- Finds temporally repeated patches across frames; replaces runs with single token + positional encoding
- **35% throughput increase, 0.1% accuracy drop** (no training required)
- At 30 FPS, **100%+ speedup**; up to **80% token reduction** on long videos
- Works on standard ViTs without architectural changes
- **Classification**: concept-only for CorridorKey (requires multi-frame token batching; Hiera's windowed attention with mask units complicates ragged tensor handling in MLX)

**Practical assessment**: Most promising training-free approach. But implementation complexity in MLX is high — Hiera's unroll/reroll permutations assume fixed token count. Variable-length sequences would require significant refactoring.

### 1.5 ResidualViT I/P-Frame Encoding (ICCV 2025 Highlight)

- Full ViT on I-frames; learnable residual + token reduction on P-frames
- **2.5x faster frame encoding, 60% lower cost**, preserving accuracy
- Token reduction module significantly cuts P-frame tokens
- **Classification**: pytorch-only (residual computation path requires training)

### 1.6 Object-Aware Video Matting (OAVM, 2025)

- Pixel-level temporal features via cross-frame matching
- Instance queries via set prediction
- Memory bank with initial + historical foreground embeddings
- Dilated + binarized previous-frame masks as foreground localization prompts
- **Results @1080p**: MAD 4.23, MSE 0.31, dtSSD 1.31
- **Classification**: concept-only (trained cross-frame attention)

### 1.7 Generative Video Matting (SIGGRAPH 2025)

- Reformulates matting as conditional video generation (from Stable Video Diffusion)
- Flow-matching mechanism to reduce inference steps
- Hybrid supervision in latent + image space
- **Classification**: out-of-scope (entirely different architecture)

### 1.8 Commercial Tool Approaches

**DaVinci Resolve / After Effects / Nuke**:
- No ML-based matting in timeline (keying is traditional chroma key / luminance key)
- GPU acceleration for decode/encode + color grading + compositing
- Proxy mode for real-time editing (downscale preview); full-res for final render
- Render cache system: pre-render heavy processing, play from cache
- Alpha channel handling: premultiplied vs straight alpha workflows
- **Pattern**: Always offline batch render for quality; real-time only for preview

**Runway** (and similar AI video tools):
- Cloud GPU inference (A100/H100)
- Not optimized for edge/local inference
- No published architecture details for matting specifically

---

## 2. Benchmarks for Video Matting

### 2.1 Datasets

| Dataset | Frames | Resolution | Split | Notes |
|---------|--------|------------|-------|-------|
| VideoMatte240K | 240,709 | 4K/HD | 479:5 train:val | Most used; val split heavily imbalanced (5 videos) |
| VM108 | ~108 clips | various | varies | Older benchmark |
| YouTubeMatte | varies | varies | varies | Real-world footage |
| CRGNN | varies | varies | varies | Real-world benchmark |

**VideoMatte240K limitations**: Inaccurate semantic representation in core regions. Lacks fine boundary details. 5-video val split is too small for robust evaluation.

### 2.2 Quality Metrics (all lower = better)

| Metric | What it measures | Typical SOTA values |
|--------|-----------------|---------------------|
| MAD (Mean Absolute Difference) | Overall alpha accuracy | 4-6 (good), <4 (excellent) |
| MSE (Mean Squared Error) | Squared alpha error, penalizes large errors | 0.3-1.5 (good) |
| SAD (Sum of Absolute Differences) | Total alpha error (whole image) | varies by resolution |
| Grad (Spatial Gradient) | Edge/detail fidelity | 0.5-1.0 (good) |
| Conn (Connectivity) | Perceptual hole/island artifacts | 0.3-0.5 (good) |
| **dtSSD** | Temporal coherence (mean squared diff of temporal gradients) | 1.0-1.5 (good), <1.0 (excellent) |

**dtSSD is the critical metric for video**. It measures flicker/temporal instability. A model can have good per-frame MAD but terrible dtSSD if predictions jitter frame-to-frame.

### 2.3 FPS Targets

| Use case | Target FPS | Typical resolution | Notes |
|----------|------------|-------------------|-------|
| Real-time preview | 24-30 | 512x288 to 720p | Interactive editing |
| Near-real-time | 10-15 | 1080p | Comfortable editing |
| Offline batch (VFX) | 1-5 | 1080p-4K | Quality-first |
| Mobile real-time | 30 | 512x288 | RVM's target |

**CorridorKey's position**: Offline batch VFX. 2.4 FPS @1024 is acceptable. Target: 3-5 FPS effective (with temporal tricks).

### 2.4 SOTA Benchmark Numbers (2025-2026)

| Model | MAD | MSE | dtSSD | Resolution | FPS (GPU) |
|-------|-----|-----|-------|------------|-----------|
| OAVM | 4.23 | 0.31 | 1.31 | 1080p | not reported |
| RVM (MobileNetV3) | ~6 | ~1.5 | ~1.4 | 1080p | 172 (RTX 3090) |
| MatAnyone 2 | SOTA | SOTA | SOTA | varies | not reported |
| RVM FP32 baseline | 6.08 | 1.47 | 1.36 | 512x288 | ~100+ |

---

## 3. Error Thresholds and Tolerances

### 3.1 Quantization Precision: What The Research Says

**PTQ4VM paper (2025)** — first systematic PTQ study for video matting (tested on RVM):

| Precision | SAD | MSE | dtSSD | Quality |
|-----------|-----|-----|-------|---------|
| FP32 | 6.08 | 1.47 | 1.36 | Reference |
| W8A8 | 6.03 | 1.29 | 1.46 | **Near-identical** (~0.8% SAD change) |
| W4A8 | 10.77 | 4.54 | 2.51 | Noticeable degradation (~77% SAD increase) |
| W4A4 | 20.33 | 13.80 | 4.63 | Significant degradation (~234% SAD increase) |

**Key findings**:
- **W8A8 is safe** — performance matches or slightly exceeds FP32 in some metrics
- **W4A8 is the cliff** — quality drops sharply; ~77% SAD increase
- **W4A4 requires specialized techniques** (GAC + OFA) to remain usable
- Recurrent structures (ConvGRU) are **especially vulnerable** to quantization noise — errors accumulate across frames
- Optical Flow Assistance (OFA) component specifically preserves temporal coherence under quantization

### 3.2 Is max_abs_error < 5e-3 Strict or Lenient?

**Context**: CorridorKey's fidelity gate requires max_abs_error < 5e-3 per tensor vs golden.npz.

**Assessment**: This is **moderately strict** for inference-only:

- Alpha values are [0, 1] range. max_abs 5e-3 = 0.5% of full range per pixel
- In 8-bit output (0-255), 5e-3 maps to ~1.3 levels — **sub-pixel precision**, well below perceptual threshold
- Published papers report W8A8 quantized models achieving <1% SAD change — implying per-pixel errors mostly well under 1e-2
- However, max_abs is a worst-case metric. A single outlier pixel at 4.9e-3 with all others at 1e-6 still passes
- FP16 inference typically introduces max_abs errors of 1e-4 to 1e-3 (well within 5e-3)
- BF16 may introduce slightly larger errors (1e-3 to 5e-3) due to reduced mantissa

**Practical answer**: 5e-3 is appropriate for validating precision changes (FP16/BF16 mixed precision). It would be **too strict** for algorithmic changes (backbone skip, feature warping) where structural differences can cause max_abs of 0.1-1.0 at motion boundaries. For algorithmic changes, use perceptual metrics (PSNR, SSIM, dtSSD) instead.

### 3.3 Perceptual Quality Thresholds

| Metric | Excellent | Good | Acceptable | Poor |
|--------|-----------|------|------------|------|
| PSNR (alpha) | >45 dB | 35-45 dB | 30-35 dB | <30 dB |
| SSIM (alpha) | >0.99 | 0.95-0.99 | 0.90-0.95 | <0.90 |
| dtSSD | <1.0 | 1.0-1.5 | 1.5-2.5 | >2.5 |

**From our skip2 experiment**: Non-skipped frames had PSNR 44-48 dB (excellent). Skipped frames had PSNR 16-24 dB (catastrophic). This confirms backbone skip without flow compensation is unacceptable.

### 3.4 Alpha vs Foreground Thresholds

- **Alpha is more sensitive**: Human perception is very sensitive to edge artifacts, halos, and transparency errors. Alpha errors at boundaries are immediately visible in compositing.
- **Foreground is more forgiving**: Foreground color errors are masked by alpha — regions where alpha is near 0 (transparent) can have large foreground errors with zero visual impact.
- **Practical threshold**: Alpha tolerances should be ~2-5x tighter than foreground tolerances. Our fidelity gate treats them equally at 5e-3, which is conservative for foreground.

### 3.5 Precision Format Recommendations

| Format | Best for | Risk | Expected max_abs vs FP32 |
|--------|----------|------|--------------------------|
| FP32 | Reference / golden | None | 0 |
| BF16 | Backbone (preserves dynamic range) | Slight edge softening | 1e-3 to 5e-3 |
| FP16 | Decoders (limited range OK for small tensors) | Overflow near saturation | 1e-4 to 1e-3 |
| INT8 (W8A8) | Full model if calibrated | Needs PTQ calibration data | ~1% SAD change |
| INT4 | NOT RECOMMENDED for matting | Quality cliff | 77%+ SAD increase |

---

## 4. Frame-Level Optimization Patterns

### 4.1 Which Layers Are Safe to Run at Lower Frequency?

Based on SAM2's architecture and video segmentation literature:

| Component | Safe to cache/skip? | Evidence |
|-----------|---------------------|----------|
| Late backbone stages (stride 16/32) | PARTIALLY — if input is pure RGB | SAM2 uses stages 3-4 in memory attention; DFF/Accel cache these |
| Early backbone stages (stride 4/8) | NO — too frame-specific | SAM2 bypasses memory for stages 1-2; used directly in decoder |
| Backbone with hint channel | NO — stale hints corrupt all stages | Our V2 experiment proved this |
| Decoder (SegFormer-style) | NO — depends on backbone features | Must run every frame |
| Refiner | PARTIALLY — tiles with unchanged alpha can skip | Requires tile-level diff detection |

**Critical finding for CorridorKey**: The 4ch input (RGB + alpha hint) makes backbone caching fundamentally different from RGB-only models. Every backbone stage encodes hint information. This is unique to CorridorKey's architecture.

### 4.2 Feature Warping / Interpolation

**Optical flow warping** (DFF, Accel):
- Warp cached features using estimated optical flow
- Works for RGB-only backbones; questionable for RGB+hint
- MLX lacks `grid_sample` — would need custom bilinear warp
- Apple Vision `VNGenerateOpticalFlowRequest` runs on ANE (free from GPU)
- Estimated overhead: flow ~20-50ms (ANE) + warp ~10ms

**Feature interpolation** (between keyframes):
- Linear interpolation of features between keyframes
- Only works for slow, smooth motion
- Breaks at scene cuts and fast motion

**For CorridorKey specifically**: Even with flow warping, the hint channel problem remains. Warping the hint-encoded features by motion vectors doesn't fix the fact that the hint mask itself has changed. Would need to re-encode the hint at each frame — which means running the full backbone.

### 4.3 Adaptive Computation / Scene Change Detection

**Lightweight scene change detection**:
- Compare frame histograms or SSIM between consecutive frames
- Threshold to detect cuts (abrupt) and gradual transitions
- Overhead: negligible on CPU (histogram comparison ~1ms)
- Can use low-level perceptual cues (sharpness, luminance) — no GPU needed

**Adaptive strategies**:
- High motion / scene change -> full pipeline
- Low motion -> EMA blending of outputs, skip refiner tiles
- Static regions -> reuse previous alpha directly

**For CorridorKey**: Scene change detection is useful for:
1. Forcing keyframes at cuts (if backbone skip ever becomes viable)
2. Identifying static tiles for refiner skip
3. Adaptive EMA weight (stronger smoothing on low-motion frames)

### 4.4 Batch Processing Multiple Frames

**Current constraint**: MLX has single GPU command queue — no true frame-level parallelism.

**Viable batching patterns**:
- CPU-GPU overlap via `mx.async_eval`: GPU processes frame N while CPU loads frame N+1
- `mlx.data` prefetching: `prefetch(prefetch_size=4, num_threads=4)` for async I/O
- Batch multiple frames through backbone if memory allows (B>1) — but CorridorKey was designed for B=1

**Not viable**:
- True double-buffering (GPU-GPU) — single command queue
- Multi-stream inference — not supported in MLX

---

## 5. Synthesis: Actionable Recommendations for CorridorKey MLX

### What Works (No Retraining Required)

| Technique | Expected gain | Effort | Risk |
|-----------|---------------|--------|------|
| **Temporal EMA blending** on outputs | Flicker elimination, ~0 compute | Trivial | None — tunable alpha parameter |
| **Adaptive refiner tile skip** | 10-20% refiner savings (~5-10ms @1024) | Medium | None — worst case = full refiner |
| **Async frame pipeline** (CPU-GPU overlap) | 20-30% throughput from hiding I/O | Low | None — no quality impact |
| **Scene change detection** | Guards adaptive strategies | Low | None — metadata only |

### What Doesn't Work (Proven or Structural)

| Technique | Why not | Evidence |
|-----------|---------|----------|
| **Backbone skip** | 4ch input (RGB+hint) makes cached features stale | V2 experiment, PSNR 16-24 dB on skipped frames |
| **Feature warping of cached backbone** | Hint channel invalidates warped features | Architectural — hint encodes spatial guidance |
| **ConvGRU recurrent state** | Requires training | RVM architecture; can't retrofit |
| **Memory bank attention** | Requires training | MatAnyone/SAM2 architecture |
| **INT4 quantization** | Quality cliff (77%+ SAD increase) | PTQ4VM paper |

### What Might Work (Higher Effort, Uncertain Payoff)

| Technique | Expected gain | Effort | Open question |
|-----------|---------------|--------|---------------|
| **RLT (Run-Length Tokenization)** | 35% backbone throughput | High | Can Hiera's windowed attention handle ragged tokens in MLX? |
| **Hint-aware backbone skip** | 1.5-2x if feasible | High | Can we separate hint encoding from RGB encoding in backbone? |
| **Optical flow for refiner tile targeting** | Better tile skip hit rate | Medium | Apple Vision flow quality @1024? PyObjC overhead? |
| **Resolution-adaptive backbone** | 3.6x on "easy" frames | Medium | How to classify easy vs hard without running backbone? |

### Recommended Priority

1. **EMA blending** — trivial, immediate flicker improvement
2. **Async frame pipeline** — 20-30% throughput, low effort
3. **Refiner tile skip** — 5-10ms savings per frame, medium effort
4. **Resolution-adaptive** with scene change detection — leverage existing decoupled resolution
5. **RLT investigation** — high ceiling, high effort

---

## Unresolved Questions

- Hint channel separation: can Hiera's early conv (4->112) be decomposed into RGB path + hint path?
- If hint path is a simple additive/multiplicative modulation, could cache RGB features + recompute hint contribution?
- RLT token masking: does MLX `mx.fast.scaled_dot_product_attention` support attention masks efficiently?
- dtSSD measurement: what's CorridorKey's per-frame dtSSD on real video without any temporal tricks?
- Refiner tile skip hit rate: what % of tiles are unchanged on typical VFX footage?
- `mlx.data` prefetch: actual throughput gain vs manual `mx.async_eval`?

---

## Sources

### Papers
- [RVM: Robust High-Resolution Video Matting (WACV 2022)](https://arxiv.org/abs/2108.11515)
- [MatAnyone: Stable Video Matting (CVPR 2025)](https://arxiv.org/abs/2501.14677)
- [MatAnyone 2: Scaling Video Matting (CVPR 2026)](https://arxiv.org/html/2512.11782v1)
- [Object-Aware Video Matting (2025)](https://arxiv.org/html/2503.01262v1)
- [Generative Video Matting (SIGGRAPH 2025)](https://arxiv.org/html/2508.07905v1)
- [PTQ4VM: Post-Training Quantization for Video Matting (2025)](https://arxiv.org/html/2506.10840)
- [Deep Feature Flow (CVPR 2017)](https://arxiv.org/abs/1611.07715)
- [Accel: Corrective Fusion for Video Segmentation (CVPR 2019)](https://arxiv.org/abs/1807.06667)
- [Don't Look Twice: Run-Length Tokenization (NeurIPS 2024)](https://arxiv.org/abs/2411.05222)
- [ResidualViT: I/P-Frame ViTs (ICCV 2025)](https://arxiv.org/abs/2509.13255)
- [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/html/2408.00714v1)
- [VideoMatt: Real-Time Video Matting Baseline (CVPRW 2023)](https://openaccess.thecvf.com/content/CVPR2023W/MobileAI/papers/Li_VideoMatt_A_Simple_Baseline_for_Accessible_Real-Time_Video_Matting_CVPRW_2023_paper.pdf)
- [Scene Detection Policies and Keyframe Extraction (2025)](https://arxiv.org/html/2506.00667v1)

### Repos
- [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting) (9.2k stars)
- [MatAnyone](https://github.com/pq-yang/MatAnyone) (CVPR 2025)
- [MatAnyone2](https://github.com/pq-yang/MatAnyone2) (CVPR 2026)
- [RLT](https://github.com/rccchoudhury/rlt) (NeurIPS 2024)
- [Accel](https://github.com/SamvitJ/Accel) (CVPR 2019)

### Prior Project Research
- `research/compound/apple_silicon_video_inference_research.md`
- `research/compound/2026-03-13-backbone-skip-rejected-hint-in-backbone.md`
- `docs/brainstorms/2026-03-12-video-pipeline-optimization-brainstorm.md`
- `research/compound/next_frontiers_handoff.md`
