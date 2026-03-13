# Video Pipeline Optimization — Brainstorm

**Date:** 2026-03-12
**Context:** 42 single-frame experiments exhausted. Application-level optimization at plateau (422ms @1024, 117ms @512). CorridorKey is a video pipeline — temporal optimization is the next frontier.

## What We Know

### Current State
- **422ms/frame @1024x1024**, 117ms @512x512, 1407MB peak (with env var tuning)
- Backbone (Hiera, 24 blocks): ~70-80% of latency (~300-350ms @1024)
- Decoder (alpha+fg): ~5-10% (~30-50ms)
- Refiner (CNN, GroupNorm-heavy): ~15-20% (~60-80ms)
- All single-frame optimizations exhausted (precision, quantization, compilation, fusion, layout, tiling)
- Framework bottlenecks: GroupNorm transposes (7% GPU), gather (9% GPU), dilated conv im2col — all confirmed dead ends

### CorridorKey Upstream
- **Strictly frame-by-frame** — zero temporal logic in the model
- Temporal coherence comes from upstream hint generators (GVM/VideoMaMa)
- No v2 or temporal variant announced
- No forks add temporal processing either
- VFX workflow: offline batch, EXR output, DaVinci Resolve / Nuke integration

### MLX Framework (v0.31.1, 2026-03-12)
- No `mx.fast.group_norm` planned
- No threadgroup shared memory in `mx.fast.metal_kernel` — confirmed dead end
- No `grid_sample` — would need custom implementation for feature warping
- No video inference examples in mlx-examples
- `mx.async_eval` exists for CPU-GPU overlap
- NAX (M5 only, macOS 26.2+) — GEMM only, no vision ops yet

### Hardware Reality
- **Real-time @1024 (24fps = 41.6ms) is impossible** on any current Apple Silicon
- M4 Max best case: ~200-280ms/frame (3.6-5 FPS) for single-frame
- Thermal throttling: MacBook Pro drops 30-40% under sustained load; Mac Studio preferred
- Unified memory is already zero-copy — no optimization needed there

## Opportunity Tiers

### Tier 1: Video-Temporal (Highest Impact, No Retraining)

#### A. Backbone Skip + Feature Reuse
- **Idea:** Run Hiera every Nth frame, reuse cached features for intermediate frames. Decoder+refiner always run.
- **Cost model:** Backbone ~300ms, decoder+refiner ~120ms. Skip N=5 → effective per-frame cost = (300 + 5*120) / 5 = 180ms/frame (1.9x speedup)
- **Memory:** Cache 4 feature stages = ~13MB @512, ~50MB @1024. Negligible.
- **Quality:** Excellent for slow/medium motion. Degrades on fast motion, camera shake.
- **Mitigation:** Scene change detection forces keyframes; adaptive interval based on feature drift.
- **Implementation:** `TemporalGreenFormer` wrapper around existing model. No model code changes.
- **Risk:** Low. Pure inference-time scheduling. Fully reversible.
- **Papers:** Deep Feature Flow (CVPR 2017), Accel (CVPR 2019)

#### B. Adaptive Refiner Tile Skip
- **Idea:** Compare coarse alpha between frames; skip refiner tiles where alpha is unchanged.
- **Cost model:** If 50% of tiles skip → refiner cost drops from ~80ms to ~40ms @1024.
- **Prereq:** Tiling infrastructure already exists.
- **Quality:** Perfect on unchanged regions. Only recomputes where the matte actually changes.
- **Implementation:** Diff previous vs current coarse alpha per tile; threshold to skip.
- **Risk:** Low. Worst case: all tiles run (no regression).

#### C. Temporal EMA Blending
- **Idea:** Exponential moving average on outputs (or features) across frames.
- **Cost:** ~0 compute. Eliminates per-frame flicker.
- **Quality:** Smooths edges; introduces ~1 frame lag on fast motion. Feature-space EMA better than output-space.
- **Implementation:** Trivial — 1 multiply + 1 add per tensor.
- **Risk:** Negligible. Tunable weight parameter (0.6-0.8).

### Tier 2: Pipeline Architecture (Medium Impact)

#### D. Async Frame Pipeline
- **Idea:** Overlap frame N GPU inference with frame N+1 I/O (load + preprocess + save).
- **Gain:** ~20-30% throughput from hiding I/O latency.
- **Implementation:** `mx.async_eval` + double-buffered frame loading.
- **Risk:** Low. No quality impact.

#### E. Resolution-Adaptive Backbone
- **Idea:** Auto-detect "easy" frames (static, simple alpha) and run backbone at 512 instead of 1024.
- **Prereq:** Decoupled resolution infrastructure already built (Opt Phase 3).
- **Gain:** 3.6x faster backbone on easy frames.
- **Quality:** Content-dependent. Needs empirical validation.
- **Risk:** Medium. Classifier for "easy vs hard" needs tuning.

### Tier 3: Advanced Temporal (High Impact, Higher Effort)

#### F. Run-Length Tokenization (RLT)
- **Idea:** Compare Hiera patch tokens across frames; merge near-identical tokens.
- **Paper:** "Don't Look Twice" (NeurIPS 2024). 40% throughput increase, 0.1% accuracy drop on video. No retraining.
- **Catch:** Requires modifying Hiera forward pass for variable-length token sequences. MLX attention may not handle ragged tensors efficiently.
- **Implementation:** Moderate — diff tokens, create mask, modify attention.
- **Risk:** Medium. Token masking overhead could negate gains in MLX.

#### G. Apple Vision Optical Flow + Feature Warping
- **Idea:** Use `VNGenerateOpticalFlowRequest` (runs on ANE, free from GPU) to warp cached backbone features to current frame geometry.
- **Cost:** Flow ~20-50ms on ANE + warp ~10ms + decoder ~120ms = ~150-180ms vs 422ms full pipeline.
- **Catch:** MLX lacks `grid_sample`. Need custom bilinear warp implementation. PyObjC bridge overhead unknown.
- **Gain:** Higher quality than direct feature reuse on motion sequences.
- **Risk:** Medium-high. Custom grid_sample + PyObjC integration = significant new code.

#### H. ResidualViT I/P-Frame Encoding
- **Idea:** Full Hiera on keyframes (I-frames), lightweight residual computation on P-frames.
- **Paper:** ResidualViT (2025). 60% compute reduction, 2.5x speedup, 1-3% accuracy drop.
- **Catch:** Requires implementing residual computation path in Hiera. Non-trivial.
- **Risk:** Medium. Quality degradation on matting edges unclear.

### Tier 4: Out of Scope (Documented for Future Reference)

- **CoreML/ANE compilation:** Refiner is only ~50ms of 422ms — even halving it saves 6%. Not worth complexity.
- **Recurrent state (ConvGRU):** Requires training. RVM's approach is elegant but can't retrofit without fine-tuning.
- **TCVOM Temporal Attention Module:** Plug-in module for matting networks. Requires training data.
- **Architectural changes (ASPP restructure):** Needs retraining. Would bypass dilated conv im2col.
- **Upstream MLX PRs:** `mx.fast.group_norm`, dilated conv Metal kernel — high value but C++/Metal work.

## Key Decisions

1. **Video pipeline is the priority.** Single-frame optimization is exhausted; temporal reuse is where the big wins are.
2. **No retraining.** Inference-only constraint remains. All temporal tricks must work without model weight changes.
3. **Offline batch target, not real-time.** VFX workflows are batch-oriented. Target: maximize throughput for frame sequences.
4. **Backbone skip is the highest-value first experiment.** 70-80% of compute, trivially cacheable, existing architecture supports it.

## Recommended Implementation Order

| Phase | What | Expected Gain | Effort |
|-------|------|---------------|--------|
| V1 | Async frame pipeline (D) | 20-30% throughput | Low |
| V2 | Backbone skip + feature reuse (A) | 1.5-2x throughput | Low-Medium |
| V3 | Scene change detection + adaptive interval | Quality guard for V2 | Low |
| V4 | Temporal EMA blending (C) | Flicker elimination | Trivial |
| V5 | Adaptive refiner tile skip (B) | +10-20% on top of V2 | Medium |
| V6 | Optical flow warping (G) | Quality improvement for V2 | High |
| V7 | Run-Length Tokenization (F) | +40% backbone throughput | High |

**Estimated combined impact (V1-V5):** ~3-4x throughput → ~100-140ms effective per frame @1024.

## Open Questions

1. Backbone feature stability: how much do Hiera features change frame-to-frame on typical VFX footage? Needs empirical measurement.
2. Keyframe interval sensitivity: what's the quality degradation curve at skip=2, 4, 8?
3. Does `mx.async_eval` actually overlap with CPU preprocessing, or does unified memory serialize?
4. MLX `grid_sample`: implement from scratch, or find existing community impl?
5. PyObjC Vision framework: actual optical flow latency on M-series @1024?
6. Refiner tile skip hit rate: what % of tiles are unchanged between typical video frames?
7. RLT in MLX: can Hiera's windowed attention handle masked/variable-length tokens without reshape overhead?

## Cross-Reference Notes (Deep Research vs Agent Research)

### Corrections to deep research doc
1. **PR #3147 is conv3D only, NOT conv2D.** Implicit GEMM explicitly falls back to im2col for dilation. Does NOT help our dilated refiner convs.
2. **`mx.fast.metal_kernel` does NOT expose threadgroup memory/barriers.** Experimentally proven in exp 42. Custom GroupNorm kernel was +41% slower + non-deterministic.
3. **Real-time @1024 on M4 Max is NOT "probable."** Even with 512 backbone decoupling, refiner alone is ~60-80ms @1024 — exceeds 41.6ms budget.

### New leads from deep research
1. **Gimlet Labs AI-generated Metal kernels** — fused VisionAttention+GroupNorm achieving 18x speedup on non-contiguous memory. Different approach than our exp 42.
2. **SAM2's Hiera memory bank** — direct architectural precedent for cross-frame Hiera feature caching. Validates backbone skip approach.
3. **MUT3R implicit motion detection** — pretrained ViTs down-weight dynamic regions in self-attention. Could identify reusable tokens without explicit flow.
4. **`mlx.data` prefetching** — `prefetch(prefetch_size=4, num_threads=4)` for async frame loading, cleaner than manual `mx.async_eval`.

## References

### Papers
- Deep Feature Flow (CVPR 2017) — arxiv:1611.07715
- Accel: Adaptive Keyframe Selection (CVPR 2019) — github.com/SamvitJ/Accel
- ResidualViT: I/P-Frame ViTs (2025) — arxiv:2509.13255
- Run-Length Tokenization / Don't Look Twice (NeurIPS 2024) — arxiv:2411.05222
- EVAD: Keyframe-Centric Token Dropout — arxiv:2304.08451
- RVM: Robust Video Matting (WACV 2022) — arxiv:2108.11515
- MatAnyone (CVPR 2025) — arxiv:2501.14677
- FTP-VM (CVPR 2023) — github.com/csvt32745/FTP-VM
- VMFormer — github.com/SHI-Labs/VMFormer
- TCVOM — github.com/yunkezhang/TCVOM

### Repos
- RVM: github.com/PeterL1n/RobustVideoMatting (9.2k stars)
- MatAnyone: github.com/pq-yang/MatAnyone (1.5k stars)
- RLT: github.com/rccchoudhury/rlt
- EZ-CorridorKey: github.com/edenaion/EZ-CorridorKey (565 stars, GUI + SAM2)
- CorridorKey-Engine: github.com/99oblivius/CorridorKey-Engine (async multi-GPU pipeline)
