---
title: Video Temporal Optimization Experiments
type: feat
date: 2026-03-12
deepened: 2026-03-12
---

# Video Temporal Optimization Experiments

## Enhancement Summary

**Deepened on:** 2026-03-12
**Sections enhanced:** 7 experiments + architecture + benchmark + risks
**Research agents used:** performance-oracle, architecture-strategist, kieran-python-reviewer, code-simplicity-reviewer, spec-flow-analyzer, video-matting-temporal-researcher, mlx-async-patterns-researcher, codebase-explorer

### Key Improvements
1. **Architectural prerequisite identified:** GreenFormer needs decomposed forward methods (`run_backbone`, `run_decoders`, `run_refiner`) — V2 cannot work without this
2. **Experiment count reduced:** V1 merged into V0, V4 (tile skip) eliminated (ineffective at 1024 single-tile + broken interaction with V2), V6 removed (noise)
3. **Critical memory bug prevented:** Feature cache must materialize arrays before storing, or lazy graph holds backbone intermediates alive (~2-3x memory bloat)
4. **EMA must reset on scene cuts** — otherwise cross-cut ghosting guaranteed
5. **Don't gc.collect()/mx.clear_cache() per frame** in video loop — defeats MLX buffer recycling

### New Considerations Discovered
- Video pipeline should NOT live in engine.py — create standalone `VideoProcessor` class taking GreenFormer directly
- Overlapping PNG save (not decode) with inference is the real async I/O opportunity
- EMA error accumulates across keyframe boundaries — reset on keyframes
- Per-frame fidelity gate should also check keyframe 0 against golden.npz (catch preprocessing regressions)
- 37-frame clip insufficient for thermal characterization (onset ~60-90s)

### Simplification Opportunities Applied
- FeatureCache class -> bare variable (a class wrapping a list is over-engineering)
- V3 (scene detection) made conditional on skip>3 — at skip=2-3, brute force handles cuts
- Streaming generator output recommended over buffered

---

## Overview

Single-frame optimization is exhausted (42 experiments, 422ms @1024 plateau). The next frontier is video-temporal optimization — exploiting frame-to-frame redundancy to cut effective per-frame cost. This plan defines **3 core experiments** (V0, V2, V5) plus 1 conditional (V3), ordered by simplicity and expected impact.

## Problem Statement

CorridorKey is a video matting pipeline but currently processes every frame independently. The Hiera backbone (~70-80% of latency) recomputes identical or near-identical features for temporally adjacent frames. For a 37-frame 1080p clip at 422ms/frame, total processing is ~16s. Temporal reuse could cut this to ~5-8s.

## Design Decisions (Pre-resolved)

| Question | Decision | Rationale |
|----------|----------|-----------|
| Non-square input (1920x1080) | Resize to 1024x1024 with distortion | Simplest. Matches existing single-frame pipeline. Quality acceptable for matting. |
| Hint video | Frame-aligned 1:1 with input | Both are 37 frames, same FPS. Extract in parallel. |
| Video fidelity gate | Per-frame max_abs < 5e-3 vs **full-pipeline single-frame output** (not golden.npz) | Temporal tricks intentionally diverge from golden. Compare against "ourselves without skip." |
| EMA domain | Pixel space (post-sigmoid) | Simpler. Logit-space EMA changes sigmoid nonlinearity — harder to reason about. |
| Backbone skip ratios | Sweep [2, 3, 5] | Covers conservative to aggressive. Pick pareto-optimal. |
| Scene detection | Pixel-level MSE on downsampled frames | Cheap, no ML dependency. Threshold TBD empirically. |
| Output format | PNG sequence + optional ffmpeg reassembly | EXR for VFX, PNG for development. Audio passthrough via ffmpeg mux. |
| mx.compile + backbone skip | Use per-component compilation or forward_eager | Compiled __call__ fuses all stages — backbone skip requires separate calls. |
| Video benchmark script | New file (NOT protected surface) | Protected surfaces are single-frame benchmarks. Video bench is a new concern. |
| Experiment order | V0 -> V2 -> V3 (conditional) -> V5 (conditional) | Simplest viable path first. |
| Hint frame mismatch | Assert frame counts match, raise ValueError | Fail fast on user error. |
| Output mode | Generator (yield per frame) | Enables progress reporting, partial output on crash, bounded memory. |

## Technical Approach

### Architecture

```
Frame N from MP4
    |
    +-- [CPU] Decode + preprocess (resize 1024, normalize, concat hint)
    |
    +-- [GPU] Backbone (Hiera, 24 blocks) <-- SKIP if not keyframe
    |         +-- 4 feature maps cached: [112, 224, 448, 896] ch at strides [4,8,16,32]
    |         +-- materialize BEFORE caching (critical: prevents lazy graph memory bloat)
    |
    +-- [GPU] Decoders (alpha + fg) <-- ALWAYS run (cheap, ~30-50ms)
    |         +-- coarse alpha, coarse fg at full res
    |
    +-- [GPU] Refiner (CNN, dilated convs) <-- ALWAYS run
    |         +-- delta logits -> final alpha, fg
    |
    +-- [Post] Temporal EMA blend with frame N-1 (optional, V5)
    |         +-- reset on scene cuts
    |
    +-- [CPU/Thread] Postprocess + save PNG/EXR (overlap with next frame inference)
```

### Research Insights: Architecture

**Decomposed forward API (prerequisite for V2):**
The current `forward_eager` runs backbone+decoders+refiner as one call. Backbone skip requires calling them separately. Add explicit methods to GreenFormer:
```python
def run_backbone(self, x: mx.array) -> list[mx.array]: ...
def run_decoders(self, features: list[mx.array]) -> dict[str, mx.array]: ...
def run_refiner(self, x: mx.array, coarse: dict[str, mx.array]) -> dict[str, mx.array]: ...
```
This is the minimal architectural change for V2. Do NOT reuse `forward_eager` directly — its `del features` and `async_eval` patterns conflict with external caching.

**VideoProcessor vs engine.py:**
Don't add `process_video()` to the engine. Create a standalone `VideoProcessor` class in `inference/video.py` that takes `GreenFormer` directly. The engine is single-frame and stateless; video processing is multi-frame and stateful (cache, EMA, scheduling). Keep them separate.

**Memory management pattern for video loops:**
```python
# Do NOT gc.collect() + mx.clear_cache() per frame — defeats buffer recycling.
# Let CACHE_LIMIT_BYTES (set to 2.0 GiB for video) handle it automatically.
# Only clear_cache() after the entire video is processed.
# gc.collect() at most every 10 frames, not every frame.
```

**Relevant prior art (from research):**
- **Deep Feature Flow** (CVPR 2017): Keyframe + flow warp of backbone features. 2-4x speedup on ResNet-101. Quality drops ~1.9% mIoU on segmentation. Flow warping introduces artifacts at edges — risky for matting.
- **RobustVideoMatting**: ConvGRU in decoder for temporal state. Best quality but requires retraining (out of scope).
- **MatAnyone** (CVPR 2025): Region-adaptive memory — separate core (stable interior) vs boundary (changing edge). Inspirational for future refiner optimization.
- **Key insight from all three:** Temporal state belongs in the decoder, not the backbone. Simple feature reuse (our approach) is the lowest-effort starting point.

### File Touchpoints

| File | Role | Changes |
|------|------|---------|
| `src/corridorkey_mlx/model/corridorkey.py` | Add decomposed forward methods | `run_backbone`, `run_decoders`, `run_refiner` |
| `src/corridorkey_mlx/inference/video.py` | **NEW** — VideoProcessor class | Frame extraction, scheduling, assembly, generator output |
| `src/corridorkey_mlx/inference/temporal.py` | **NEW** — temporal utilities | Scene detection, EMA blending |
| `scripts/bench_video.py` | **NEW** — video benchmark | FPS, wall-clock, per-frame latency histogram, memory |
| `scripts/infer_video.py` | **NEW** — CLI for video inference | Takes input.mp4 + hint.mp4, outputs frames/mp4 |
| `research/decision.schema.json` | Add `"temporal"` to search_area enum | Enables experiment logging |

### Protected surfaces: NONE modified.

## Implementation Phases

### Exp V0: Baseline Video Loop (+ Async I/O Measurement)

**Goal:** Extract frames, run per-frame inference, establish throughput baseline. Measure decode time to determine if async I/O is worth pursuing.

**Hypothesis:** Total wall-clock ~ 37 x 422ms ~ 15.6s @1024. Frame decode overhead is negligible (<5ms/frame).

**Implementation:**
- `scripts/infer_video.py` — CLI entry point
  - ffmpeg frame extraction to temp dir (or cv2.VideoCapture in-memory)
  - Loop: load frame + hint -> preprocess -> model(frame) -> postprocess -> save PNG
  - Optional: ffmpeg reassembly to MP4
- `scripts/bench_video.py` — benchmark harness
  - Metrics: total wall-clock, per-frame median/p95, peak memory, FPS
  - Generate "single-frame reference" outputs for all 37 frames (fidelity baseline)
  - Output: `research/artifacts/video_baseline.json`
- `src/corridorkey_mlx/inference/video.py` — VideoProcessor class
  - Generator-based: `yield` per frame for progress reporting + bounded memory
  - Measure decode time separately (logged per-frame)
  - If decode > 10ms: add `ThreadPoolExecutor` for overlapping **PNG save** with inference (not decode — the real I/O opportunity is save, ~15-30ms for 1024x1024 RGBA PNG)

**Acceptance criteria:**
- [x] Process all 37 frames without OOM
- [x] Per-frame latency within 5% of single-frame benchmark (no regression from loop overhead)
- [x] Reference outputs saved for fidelity comparison in later experiments
- [x] Decode time measured separately per-frame
- [ ] Frame 0 also validated against golden.npz (catch preprocessing regressions)

**Rollback:** N/A — new files only.

### Research Insights: V0

**Memory management in video loop:**
- Do NOT call `gc.collect() + mx.clear_cache()` every frame. Each `gc.collect()` is stop-the-world in CPython (1-5ms). Each `mx.clear_cache()` forces buffer re-allocation. In a 37-frame loop, this adds 37 full GC sweeps + 37 re-allocation cycles.
- Instead: set `mx.metal.set_cache_limit(2 * 1024**3)` (2 GiB) and let MLX recycle buffers. Only `mx.clear_cache()` after the full video. Call `gc.collect()` at most every 10 frames.
- Monitor `mx.metal.get_peak_memory()` at frames 1, 10, 20, 30, 37 to detect monotonic growth (leak indicator).

**MLX buffer limit for video:**
- Set `MLX_MAX_MB_PER_BUFFER=2` and `MLX_MAX_OPS_PER_BUFFER=2` (env vars, before importing MLX). This reduces peak memory 58% with zero latency penalty (proven in exp29).

**Async I/O — save, not decode:**
- Frame decode at 1024x1024 is typically 2-5ms — overlap with 422ms inference is negligible.
- PNG encoding of 1024x1024 RGBA is 15-30ms. Overlapping save with next-frame inference via `ThreadPoolExecutor(max_workers=1)` is the real opportunity.
- Use bounded queue (`maxsize=2-3`) to cap memory at ~96MB of in-flight numpy arrays.
- Thread termination: sentinel value (`None`) in queue. Exception propagation: store exception in a shared variable, check after `queue.join()`.

**Benchmark warmup:**
- Video warmup strategy: 3 single-frame warmup calls (triggers mx.compile traces), then 1 timed full-video pass.
- Report frame-0 latency separately from frames 1-N (frame-0 includes any remaining compile cost).

**Thermal characterization:**
- 37 frames at 422ms = ~16s sustained load. Thermal throttling onset is typically 60-90s on M3 Max.
- Add a separate 150+ frame thermal benchmark (loop the 37-frame clip 4x) to establish the "thermal knee" — frame index where latency exceeds baseline+10%.

---

### Exp V2: Backbone Skip + Feature Reuse

**Goal:** Cache backbone features, reuse for non-keyframes. Largest expected gain.

**Hypothesis:** At skip=3, effective cost = (350 + 3x120) / 3 ~ 237ms/frame (1.8x speedup). At skip=5, ~ 190ms/frame (2.2x).

**Prerequisites (implement in V0 or as V2 sub-task):**
- [x] Add `run_backbone()`, `run_decoders()`, `run_refiner()` to GreenFormer
- [x] Verify per-component compiled callables work when called individually (bit-exact with __call__)

**Implementation:**
- Feature cache as bare variables (not a class — a class wrapping a list is over-engineering):
  ```python
  # In VideoProcessor:
  cached_features: list[mx.array] | None = None
  cached_frame_idx: int = -1
  ```
- `src/corridorkey_mlx/inference/video.py`:
  - Keyframe scheduler: `frame_idx % skip_interval == 0 -> run backbone`
  - Non-keyframes: feed cached features to decoder+refiner
  - **CRITICAL:** Materialize features before storing in cache. Without this, the lazy computation graph holds all backbone intermediate buffers alive (~2-3x peak memory by frame 3-4).
  - **CRITICAL:** Do NOT `del features` when using cached — the cache owns them. Build a dedicated `run_decoders(features)` that does not mutate.

**Sweep:** Run with skip_interval in [2, 3, 5] on the 37-frame clip.

**Acceptance criteria:**
- [x] ~~Per-frame max_abs < 5e-3 vs V0 reference outputs~~ FAILED — max_abs ~1.0 at motion boundaries
- [x] Report: max/mean/p95 error per skip ratio — done (skip 2/3/5 all fail fidelity)
- [x] Report: FPS per skip ratio — skip2=2.42, skip3=2.65, skip5=2.94 FPS
- [ ] ~~At least 1.5x throughput improvement at best skip ratio that passes fidelity~~ BLOCKED — no skip ratio passes fidelity
- [x] Peak memory monitored per-frame (stable at 3508MB, not growing)

**Result: REJECTED.** Backbone skip is not viable for CorridorKey. The 4ch input (RGB+hint) means cached backbone features carry stale alpha hint spatial info. The refiner (additive residual on fresh RGB) cannot correct the coarse mismatch. Mean per-pixel error 4.2% on high-motion frames; visually unacceptable. See `research/artifacts/video_v2_skip*.json`.

**Fidelity measurement:**
```python
for frame_idx in range(37):
    ref = load_reference(frame_idx)      # from V0
    temporal = load_temporal(frame_idx)   # from this experiment
    error = np.max(np.abs(ref - temporal))
    assert error < 5e-3, f"Frame {frame_idx}: {error}"
```

**Key risks:**
- Decoder receives stale features but refiner sees fresh RGB -> conflicting signals at edges
- First frame always requires full backbone (no cache yet)
- mx.compile incompatible — must use per-component or eager

**Rollback:** Remove feature cache, restore per-frame backbone.

### Research Insights: V2

**Feature cache memory budget (bf16):**

| Stage | Spatial | Channels | Bytes (bf16) |
|-------|---------|----------|-------------|
| 0 | 256x256 | 112 | 14.7 MB |
| 1 | 128x128 | 224 | 7.3 MB |
| 2 | 64x64 | 448 | 3.7 MB |
| 3 | 32x32 | 896 | 1.8 MB |
| **Total** | | | **27.5 MB** |

Store post-cast bf16 features (not fp32 originals). Halves cache memory.

**Lazy evaluation trap (most critical finding):**
MLX lazy arrays retain references to the entire computation graph. Storing un-materialized features in the cache means the backbone's attention intermediates, im2col buffers, and normalization temps (~400-800MB) remain live across multiple frames. By frame 3-4, you're holding 3-4 backbone graphs in memory simultaneously. The fix is one line: materialize features before storing. The sync cost (~0.5ms) is negligible.

**Prior art — Deep Feature Flow quality tradeoffs:**
- DFF (CVPR 2017): 1.9% mIoU drop on Cityscapes at skip=5, 3.7x speedup. But Cityscapes is coarse segmentation. Matting is pixel-precise — quality drop at boundaries could be worse.
- Conservative start: skip=2 (every other frame). Maximum staleness = 1 frame at 24fps = 42ms of real-world change. Most green screen shots are slow enough that this is imperceptible.

**EMA error propagation concern:**
With EMA active (V5), stale backbone errors compound across non-keyframe sequences. EMA stores blended output, so each frame carries exponentially decaying error from all prior frames. Mitigation: reset EMA on keyframes (fresh backbone = clean start).

**Verify mx.clear_cache() doesn't evict cached features:**
`mx.clear_cache()` releases *unused* Metal buffers, not referenced ones. Cached `mx.array` objects remain valid. Verify with a one-line test: cache features -> `mx.clear_cache()` -> read cached features -> assert values unchanged.

---

### Exp V3: Scene Change Detection (Conditional — only if skip > 3)

**Goal:** Force keyframes on cuts/fast motion to prevent quality collapse during backbone skip.

**Condition:** Only implement if V2 results show skip=5 is needed for target throughput AND skip=5 fails fidelity at scene boundaries. At skip=2-3, the maximum staleness is 1-2 frames — brute force handles cuts.

**Hypothesis:** Scene detection catches hard cuts with >99% recall; false positive rate <5% on static shots.

**Implementation:**
- `src/corridorkey_mlx/inference/temporal.py`:
  ```python
  def detect_scene_change(prev_frame: np.ndarray | None, curr_frame: np.ndarray,
                          threshold: float = 30.0) -> bool:
      """MSE on 8x downsampled frames. Cheap, runs on CPU.

      Assumes uint8 [0, 255] HWC input frames.
      Threshold 30.0 ~ 5.5 mean pixel difference per channel.
      Returns True (force keyframe) when prev_frame is None (first frame).
      """
      if prev_frame is None:
          return True
      if prev_frame.ndim != 3 or curr_frame.ndim != 3:
          raise ValueError(f"Expected HWC (ndim=3), got {prev_frame.ndim}, {curr_frame.ndim}")
      prev_small = prev_frame[::8, ::8].astype(np.float32)
      curr_small = curr_frame[::8, ::8].astype(np.float32)
      mse = np.mean((prev_small - curr_small) ** 2)
      return mse > threshold
  ```
- Integrate into keyframe scheduler: `is_keyframe = (frame_idx % skip == 0) or scene_changed`
- **On scene change:** Also reset EMA state (prevent cross-cut ghosting).

**Sweep:** Threshold in [10, 20, 30, 50, 100] on the 37-frame clip.

**Acceptance criteria:**
- [ ] Correctly detects any scene cuts in test clip (manual verification)
- [ ] False positive rate < 5% on static/slow-motion segments
- [ ] Detection cost < 1ms per frame
- [ ] Combined with V2: fidelity passes for all frames including around cuts

**Key risk:** Gradual dissolves may not trigger detection. For VFX green screen footage this is rare — cuts are usually hard.

**Rollback:** Remove scene detection, revert to fixed-interval keyframes.

### Research Insights: V3

**Better algorithms than raw MSE (if needed):**

| Algorithm | Speed | How | Threshold guidance |
|-----------|-------|-----|-------------------|
| HSV luma delta | ~1ms | Weighted sum of delta_hue + delta_sat + delta_lum | Default ~27.0 / 255 |
| Histogram correlation | <1ms | Compare color histograms via correlation | Correlation < 0.7 |
| Perceptual hash (pHash) | ~1-2ms | DCT-based 64-bit hash, Hamming distance | Distance > 10 bits |
| SSIM | ~5ms | Structural similarity | SSIM < 0.5 |

PySceneDetect's `ContentDetector` (HSV luma_only mode) is the cheapest effective option. For green screen (controlled environment), raw MSE on downsampled frames is likely sufficient.

**Secondary signal — alpha hint MSE:**
Add MSE on the alpha hint (coarse mask) in addition to RGB. If the hint changes significantly (subject moved fast), force a keyframe even if background is static. Cost: <1ms extra. Catches cases where subject moves fast against static green background.

---

### Exp V5: Temporal EMA Blending (Conditional — only if V2 output shows flicker)

**Goal:** Smooth outputs across frames to eliminate per-frame flicker.

**Condition:** Only implement if visual inspection of V2 output video shows visible per-frame jitter. If backbone skip at skip=2-3 produces smooth output, EMA is wasted effort.

**Hypothesis:** EMA with weight 0.7 reduces temporal jitter by ~50% with minimal edge lag.

**Implementation:**
- `src/corridorkey_mlx/inference/temporal.py`:
  ```python
  class TemporalEMA:
      def __init__(self, weight: float = 0.7) -> None:
          if not 0.0 < weight <= 1.0:
              raise ValueError(f"EMA weight must be in (0, 1], got {weight}")
          self.weight = weight
          self.prev_alpha: np.ndarray | None = None
          self.prev_fg: np.ndarray | None = None

      def blend(self, alpha: np.ndarray, fg: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray]:
          alpha = alpha.astype(np.float32)
          fg = fg.astype(np.float32)
          if self.prev_alpha is None:
              self.prev_alpha, self.prev_fg = alpha, fg
              return alpha, fg
          blended_alpha = self.weight * alpha + (1 - self.weight) * self.prev_alpha
          blended_fg = self.weight * fg + (1 - self.weight) * self.prev_fg
          self.prev_alpha = blended_alpha
          self.prev_fg = blended_fg
          return blended_alpha, blended_fg

      def reset(self) -> None:
          """Clear temporal state. Call on scene changes."""
          self.prev_alpha = None
          self.prev_fg = None
  ```

**Sweep:** EMA weight in [0.5, 0.6, 0.7, 0.8, 0.9]

**Acceptance criteria:**
- [ ] Visual inspection: reduced flicker on output video (subjective)
- [ ] Temporal stability metric: mean abs diff between consecutive output frames (lower = smoother)
- [ ] No visible ghosting/lag on fast motion segments
- [ ] EMA weight selected that balances smoothness vs responsiveness

**Fidelity note:** EMA intentionally deviates from single-frame output. Do NOT apply max_abs fidelity gate here. Instead measure:
- Temporal stability: `mean(|output_t - output_{t-1}|)` — lower is smoother
- Edge sharpness: preserved within 95% of single-frame

**Rollback:** Disable EMA (set weight=1.0).

### Research Insights: V5

**EMA error accumulation:**
Each blend stores the blended result as `prev_alpha`, creating an exponential moving average where errors from stale backbone features compound. With weight=0.7, after 5 non-keyframe frames: (1-0.7)^5 = 0.24% of frame 1's error persists. Small but cumulative.

**Mitigation: Reset on keyframes.**
On keyframes (fresh backbone), call `ema.reset()` or set `weight=1.0` for that frame. This gives a "clean start" every `skip_interval` frames, preventing error propagation across keyframe boundaries.

**Float32 cast requirement:**
If inputs are float16 (plausible in mixed-precision pipeline), repeated multiply-add degrades precision over hundreds of frames. Always cast to float32 before blending.

**Future upgrade path — Motion-Corrected Moving Average (MCMA):**
MCMA adds optical flow warping of the previous prediction before averaging. Eliminates ghosting on motion. Runs in ~3-8ms on GPU. Requires an optical flow estimator (none exists in MLX currently). Worth investigating if EMA ghosting is unacceptable.

**mx.array to np.ndarray sync point:**
EMA operates on numpy (CPU). The conversion `np.array(mx_tensor)` triggers a GPU sync. Place this sync immediately after array materialization completes, before any CPU post-processing. If done on a not-yet-materialized lazy tensor, it blocks the GPU pipeline.

---

## Removed Experiments

### ~~Exp V1: Async I/O Pipeline~~ -> Merged into V0
**Rationale:** Thread-based double-buffer is ~15 lines. The plan already said "skip if decode <5ms." Measure decode in V0; if >10ms, enable async save (not decode) with `ThreadPoolExecutor`. Not a separate research question.

### ~~Exp V4: Adaptive Refiner Tile Skip~~ -> Eliminated
**Rationale:** At 1024x1024 with `refiner_tile_size=1024`, there is only 1 tile — tile skip is meaningless. Reducing to 512 adds 20% overhead from overlap. More critically, when V2 (backbone skip) is active, coarse alpha from stale features barely changes between frames, causing tile diff to always fall below threshold (100% skip rate = completely stale refiner output). This is a correctness bug, not a quality issue. The refiner is ~30-50ms — even 50% tile skip saves only 15-25ms, marginal vs 200ms from backbone skip.

### ~~Exp V6: Resolution-Adaptive Backbone~~ -> Removed from plan
**Rationale:** Already deferred. Still occupied ~25 lines, risk table entries, and success criteria references. Pure cognitive load. If needed later, write a new plan.

---

## Benchmark Methodology

### Video-specific metrics

| Metric | Definition | Target |
|--------|-----------|--------|
| Total wall-clock | Time from first frame decode to last frame saved | Minimize |
| Effective FPS | 37 / total_wall_clock | Maximize |
| Per-frame latency | Individual frame inference time (excludes I/O) | Report distribution |
| Backbone hit rate | % of frames running full backbone | Lower = more reuse |
| Fidelity (spatial) | Per-frame max_abs vs single-frame reference | < 5e-3 |
| Fidelity (temporal) | Mean abs diff between consecutive outputs | Report, lower = smoother |
| Peak memory | Max across all 37 frames | <= single-frame peak + 28MB cache |
| Thermal stability | Latency slope over time (not just frame 1 vs 37) | < 10% drift over 37 frames |
| Frame 0 vs golden.npz | Keyframe parity with PyTorch reference | < 5e-3 (catch preprocessing regressions) |

### Research Insights: Benchmark

**Thermal characterization requires a separate long benchmark:**
- 37 frames = ~16s sustained load. Thermal onset is 60-90s on M3 Max.
- Add 150+ frame benchmark (4x loop of 37-frame clip) to find the "thermal knee."
- Report latency slope (not just endpoints) — detect throttling onset frame.

**Memory monitoring pattern:**
```python
for i, frame in enumerate(frames):
    result = process(frame)
    if i % 10 == 0:
        print(f"Frame {i}: peak_mem={mx.metal.get_peak_memory() / 1e6:.0f}MB")
# Monotonic growth = leak indicator
```

### Experiment logging

Append to `research/experiments.jsonl` with `search_area: "temporal"`. Add fields:
```json
{
  "experiment_name": "video-backbone-skip-3",
  "search_area": "temporal",
  "resolution": 1024,
  "video_frames": 37,
  "effective_fps": 4.2,
  "backbone_hit_rate": 0.33,
  "temporal_stability": 0.0012,
  "fidelity_max_abs": 0.0038
}
```

## Dependencies and Risks

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Backbone skip produces unacceptable edge artifacts | Blocks V2 | Measure at skip=2 first (conservative); quality acceptable for green screen at 24fps |
| Stale features + fresh RGB fight in refiner | Quality regression at edges | Quantify per-skip-ratio; refiner corrects most drift via fresh RGB input |
| mx.compile incompatible with backbone skip | Must use eager (slower per-frame) | Per-component compilation works; benchmark vs fully eager |
| Feature cache lazy graph memory bloat | 2-3x peak memory by frame 3-4 | **Materialize features before caching** (one line, non-negotiable) |
| gc.collect() per frame kills throughput | 37 stop-the-world GC cycles | Remove per-frame GC, rely on cache_limit for buffer recycling |
| Thermal throttling on long sequences | Latency drift >10% | 37 frames too short; separate 150+ frame thermal test |
| 37-frame clip too short for meaningful temporal stats | Weak statistical signal | Sufficient for development; larger clips for final validation |
| EMA ghosting across scene cuts | Visible cross-cut bleeding | Reset EMA on scene changes |
| No decomposed forward API on GreenFormer | V2 cannot work | Add run_backbone/run_decoders/run_refiner as V0/V2 prerequisite |

## Success Criteria

**Minimum viable:** V0 + V2 (backbone skip=2-3) passes fidelity at 1.5x throughput improvement.

**Target:** V0 + V2 + V5 (if needed) delivers ~2-3x throughput (~140-210ms effective/frame @1024, ~5-7 FPS).

## Simplified Experiment Sequence

1. **V0: Baseline video loop** — includes decode measurement, async PNG save if decode >10ms, reference output generation
2. **V2: Backbone skip + feature caching** — sweep skip=[2, 3, 5], bare variable cache, decomposed forward API
3. **V3: Scene detection** — conditional, only if skip>3 needed AND cuts cause quality collapse
4. **V5: Temporal EMA** — conditional, only if visual inspection of V2 output shows flicker

Three experiments minimum. Each isolated. Each with clear keep/revert criteria.

## References

### Internal
- Brainstorm: `docs/brainstorms/2026-03-12-video-pipeline-optimization-brainstorm.md`
- Deep research: `docs/MLX Video Matting Optimization Research.md`
- Engine API: `src/corridorkey_mlx/engine.py`
- Forward pass: `src/corridorkey_mlx/model/corridorkey.py`
- Tiling: `src/corridorkey_mlx/inference/tiling.py`
- Stream findings: `research/compound/2026-03-12-mlx-stream-no-gpu-parallelism.md`
- Memory patterns: `docs/solutions/runtime-errors/nchw-nhwc-transpose-and-tensor-lifecycle.md`
- Buffer limits: `research/compound/exp029_buffer_limits_memory_reduction.md`
- Apple Silicon video research: `research/compound/apple_silicon_video_inference_research.md`
- Local optimum finding: `research/compound/2026-03-12-metal-trace-local-optimum.md`

### External
- Deep Feature Flow (CVPR 2017): arxiv:1611.07715 — keyframe + flow warp, 2-4x speedup
- RobustVideoMatting (WACV 2022): arxiv:2108.11515 — ConvGRU decoder, 76 FPS @ 4K
- MatAnyone (CVPR 2025): arxiv:2501.14677 — region-adaptive memory, core vs boundary
- MCMA temporal smoothing: arxiv:2403.03120 — motion-corrected moving average, ~3-8ms
- PySceneDetect: scenedetect.com — HSV luma scene detection, ContentDetector
- Run-Length Tokenization (NeurIPS 2024): arxiv:2411.05222
- ResidualViT (2025): arxiv:2509.13255

## Unresolved Questions

- 37-frame test clip: does it contain scene cuts? If no, scene detection untestable
- Backbone skip fidelity: does Hiera feature drift differ from ResNet (DFF was ResNet-based)?
- EMA alpha coefficient: optimal value for 24fps vs 30fps green screen?
- Feature interpolation (future): linearly interpolate between keyframe features instead of reusing stale — worth the complexity?
- Longer test clip available for thermal + temporal validation?
- Per-component compiled callables: do they work correctly when called individually outside forward_eager?
