---
title: Video Temporal Optimization Experiments
type: feat
date: 2026-03-12
---

# Video Temporal Optimization Experiments

## Overview

Single-frame optimization is exhausted (42 experiments, 422ms @1024 plateau). The next frontier is video-temporal optimization — exploiting frame-to-frame redundancy to cut effective per-frame cost. This plan defines 7 sequential experiments ordered by simplicity and expected impact, each building on the last.

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
| Experiment order | 1-2-3-4-6-5-7 | EMA (5) after tile skip (6) — EMA blends outputs, tile skip compares pre-blend coarse alpha. Resolution-adaptive (7) last/optional. |

## Technical Approach

### Architecture

```
Frame N from MP4
    |
    +-- [CPU] Decode + preprocess (resize 1024, normalize, concat hint)
    |
    +-- [GPU] Backbone (Hiera, 24 blocks) <-- SKIP if not keyframe
    |         +-- 4 feature maps cached: [112, 224, 448, 896] ch at strides [4,8,16,32]
    |
    +-- [GPU] Decoders (alpha + fg) <-- ALWAYS run (cheap, ~30-50ms)
    |         +-- coarse alpha, coarse fg at full res
    |
    +-- [GPU] Refiner (CNN, dilated convs) <-- SKIP tiles where alpha unchanged
    |         +-- delta logits -> final alpha, fg
    |
    +-- [Post] Temporal EMA blend with frame N-1
    |
    +-- [CPU] Postprocess + save PNG/EXR
```

### File Touchpoints

| File | Role | Changes |
|------|------|---------|
| `src/corridorkey_mlx/inference/video.py` | **NEW** — video pipeline orchestrator | Frame extraction, scheduling, assembly |
| `src/corridorkey_mlx/inference/temporal.py` | **NEW** — temporal utilities | Scene detection, EMA, feature cache, tile diff |
| `src/corridorkey_mlx/engine.py` | Add `process_video()` method | Wraps video pipeline, exposes simple API |
| `scripts/bench_video.py` | **NEW** — video benchmark | FPS, wall-clock, per-frame latency histogram, memory |
| `scripts/infer_video.py` | **NEW** — CLI for video inference | Takes input.mp4 + hint.mp4, outputs frames/mp4 |
| `research/decision.schema.json` | Add `"temporal"` to search_area enum | Enables experiment logging |

### Protected surfaces: NONE modified.

## Implementation Phases

### Exp V0: Baseline Video Loop

**Goal:** Extract frames, run per-frame inference, establish throughput baseline.

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

**Acceptance criteria:**
- [ ] Process all 37 frames without OOM
- [ ] Per-frame latency within 5% of single-frame benchmark (no regression from loop overhead)
- [ ] Reference outputs saved for fidelity comparison in later experiments
- [ ] Decode time measured separately

**Rollback:** N/A — new files only.

---

### Exp V1: Async I/O Pipeline

**Goal:** Overlap CPU frame decode with GPU inference.

**Hypothesis:** If decode ~ 5-10ms and inference ~ 422ms, async saves ~1-2%. If decode ~ 30-50ms (EXR or high-res), saves ~7-12%.

**Implementation:**
- `src/corridorkey_mlx/inference/video.py`:
  - Thread-based producer (CPU decode) + consumer (GPU inference)
  - `mx.async_eval()` to kick off inference, collect on next iteration
  - Double-buffer: load frame N+1 while GPU processes frame N

**Acceptance criteria:**
- [ ] Total wall-clock <= baseline - decode_time (decode time fully hidden)
- [ ] Zero fidelity impact (bit-identical outputs to V0 baseline)
- [ ] Peak memory <= baseline + 1 frame (~32MB overhead for double-buffer)

**Rollback:** Revert to synchronous loop.

**Key risk:** If decode is <5ms, this is not worth the complexity. **Measure decode time in V0 first** — skip V1 if <5ms.

---

### Exp V2: Backbone Skip + Feature Reuse

**Goal:** Cache backbone features, reuse for non-keyframes. Largest expected gain.

**Hypothesis:** At skip=3, effective cost = (350 + 3x120) / 3 ~ 237ms/frame (1.8x speedup). At skip=5, ~ 190ms/frame (2.2x).

**Implementation:**
- `src/corridorkey_mlx/inference/temporal.py`:
  ```python
  class FeatureCache:
      features: list[mx.array] | None  # 4 backbone feature maps
      frame_idx: int                    # frame that produced these features

      def is_valid(self) -> bool: ...
      def store(self, features, idx): ...
      def get(self) -> list[mx.array]: ...
  ```
- `src/corridorkey_mlx/inference/video.py`:
  - Keyframe scheduler: `frame_idx % skip_interval == 0 -> run backbone`
  - Non-keyframes: feed cached features to decoder+refiner
  - Requires calling backbone and decode+refine as separate steps
  - Use `forward_eager` path (per-component, not compiled __call__)

**Sweep:** Run with skip_interval in [2, 3, 5] on the 37-frame clip.

**Acceptance criteria:**
- [ ] Per-frame max_abs < 5e-3 vs V0 reference outputs (across all 37 frames)
- [ ] Report: max/mean/p95 error per skip ratio
- [ ] Report: FPS per skip ratio
- [ ] At least 1.5x throughput improvement at best skip ratio that passes fidelity

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

**Rollback:** Remove FeatureCache, restore per-frame backbone.

---

### Exp V3: Scene Change Detection

**Goal:** Force keyframes on cuts/fast motion to prevent quality collapse during backbone skip.

**Hypothesis:** Scene detection catches hard cuts with >99% recall; false positive rate <5% on static shots.

**Implementation:**
- `src/corridorkey_mlx/inference/temporal.py`:
  ```python
  def detect_scene_change(prev_frame: np.ndarray, curr_frame: np.ndarray,
                          threshold: float = 30.0) -> bool:
      """MSE on 8x downsampled frames. Cheap, runs on CPU."""
      prev_small = prev_frame[::8, ::8].astype(np.float32)
      curr_small = curr_frame[::8, ::8].astype(np.float32)
      mse = np.mean((prev_small - curr_small) ** 2)
      return mse > threshold
  ```
- Integrate into keyframe scheduler: `is_keyframe = (frame_idx % skip == 0) or scene_changed`

**Sweep:** Threshold in [10, 20, 30, 50, 100] on the 37-frame clip.

**Acceptance criteria:**
- [ ] Correctly detects any scene cuts in test clip (manual verification)
- [ ] False positive rate < 5% on static/slow-motion segments
- [ ] Detection cost < 1ms per frame
- [ ] Combined with V2: fidelity passes for all frames including around cuts

**Key risk:** Gradual dissolves may not trigger detection. For VFX green screen footage this is rare — cuts are usually hard.

**Rollback:** Remove scene detection, revert to fixed-interval keyframes.

---

### Exp V4: Adaptive Refiner Tile Skip

**Goal:** Skip refiner tiles where coarse alpha is unchanged from previous frame.

**Hypothesis:** On typical green screen footage (static background, moving subject), 30-60% of tiles are unchanged -> 20-40ms saved @1024.

**Implementation:**
- `src/corridorkey_mlx/inference/temporal.py`:
  ```python
  def diff_coarse_alpha(prev_alpha: mx.array, curr_alpha: mx.array,
                        tile_coords: list, threshold: float = 0.01) -> list[bool]:
      """Return mask of tiles that need recomputation."""
      needs_update = []
      for (y0, y1), (x0, x1) in tile_coords:
          tile_diff = mx.max(mx.abs(
              prev_alpha[..., y0:y1, x0:x1, :] -
              curr_alpha[..., y0:y1, x0:x1, :]
          )).item()
          needs_update.append(tile_diff > threshold)
      return needs_update
  ```
- Modify refiner tiling to skip tiles where `needs_update[i] == False`, reuse previous output

**Prerequisite:** Only meaningful when multiple refiner tiles exist. At 1024x1024 with `refiner_tile_size=1024`, there's only 1 tile. Either:
- (a) Run at native 1920x1080 (non-square, 2 tiles horizontally), or
- (b) Reduce `refiner_tile_size` to 512 (4 tiles at 1024x1024), or
- (c) Skip this experiment if single-tile

**Acceptance criteria:**
- [ ] Tile skip rate measured per-frame (what % of tiles skipped)
- [ ] No visible seam artifacts at tile boundaries
- [ ] Per-frame max_abs < 5e-3 vs V0 reference
- [ ] Measurable latency reduction (>5%) when tile skip rate > 30%

**Interaction warning:** If V2 (backbone skip) is active, coarse alpha from stale features changes very little between frames -> tile skip becomes over-confident (always skips). Need to diff against **full-pipeline coarse alpha**, not cached-feature coarse alpha.

**Rollback:** Remove tile skip logic, always run all tiles.

---

### Exp V5: Temporal EMA Blending

**Goal:** Smooth outputs across frames to eliminate per-frame flicker.

**Hypothesis:** EMA with weight 0.7 reduces temporal jitter by ~50% with minimal edge lag.

**Implementation:**
- `src/corridorkey_mlx/inference/temporal.py`:
  ```python
  class TemporalEMA:
      def __init__(self, weight: float = 0.7):
          self.weight = weight
          self.prev_alpha: np.ndarray | None = None
          self.prev_fg: np.ndarray | None = None

      def blend(self, alpha: np.ndarray, fg: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray]:
          if self.prev_alpha is None:
              self.prev_alpha, self.prev_fg = alpha, fg
              return alpha, fg
          blended_alpha = self.weight * alpha + (1 - self.weight) * self.prev_alpha
          blended_fg = self.weight * fg + (1 - self.weight) * self.prev_fg
          self.prev_alpha = blended_alpha
          self.prev_fg = blended_fg
          return blended_alpha, blended_fg
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

---

### Exp V6: Resolution-Adaptive Backbone (Optional/Stretch)

**Goal:** Run backbone at 512 for "easy" frames, 1024 for complex ones.

**Hypothesis:** 30-50% of green screen frames are "easy" (static subject, clean edges). Running backbone at 512 on those saves ~300ms each (3.6x faster backbone).

**Implementation complexity: HIGH.** Requires:
- Frame difficulty classifier (variance of alpha hint, edge density, or motion magnitude)
- Dynamic pos_embed reinterpolation per-frame (currently fixed at init)
- Either two model instances or runtime rebuild
- Interaction with backbone skip (cached 512 features incompatible with 1024 decoder)
- Decoupled resolution infrastructure (exists on another branch, not merged here)

**Decision:** Defer unless V2-V5 combined throughput is insufficient. The complexity:benefit ratio is poor given that backbone skip (V2) already addresses backbone cost.

**Rollback:** Use fixed resolution.

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
| Peak memory | Max across all 37 frames | <= single-frame peak + cache overhead |
| Thermal stability | Latency of frame 1 vs frame 37 | < 10% drift |

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
| Backbone skip produces unacceptable edge artifacts | Blocks V2 | Measure at skip=2 first (conservative); scene detection (V3) guards worst cases |
| Stale features + fresh RGB fight in refiner | Quality regression at edges | Quantify per-skip-ratio; may need feature-space interpolation |
| mx.compile incompatible with backbone skip | Must use eager (slower per-frame) | Per-component compilation may work; benchmark both |
| Thermal throttling on long sequences | Latency drift >10% | Monitor; test on Mac Studio if available |
| 37-frame clip too short for meaningful temporal stats | Weak statistical signal | Sufficient for development; larger clips for final validation |
| Tile skip + backbone skip interaction | Over-confident tile skipping | Diff against full-pipeline coarse alpha, not cached-feature alpha |

## Success Criteria

**Minimum viable:** V0 + V2 (backbone skip=3) passes fidelity at 1.5x throughput improvement.

**Target:** V0-V5 combined delivers ~3x throughput (~140ms effective/frame @1024, ~8 FPS).

**Stretch:** V0-V6 delivers ~4x throughput with adaptive resolution.

## References

### Internal
- Brainstorm: `docs/brainstorms/2026-03-12-video-pipeline-optimization-brainstorm.md`
- Deep research: `docs/MLX Video Matting Optimization Research.md`
- Engine API: `src/corridorkey_mlx/engine.py`
- Forward pass: `src/corridorkey_mlx/model/corridorkey.py`
- Tiling: `src/corridorkey_mlx/inference/tiling.py`
- Stream findings: `research/compound/2026-03-12-mlx-stream-no-gpu-parallelism.md`
- Memory patterns: `docs/solutions/runtime-errors/nchw-nhwc-transpose-and-tensor-lifecycle.md`

### External
- Deep Feature Flow (CVPR 2017): arxiv:1611.07715
- Run-Length Tokenization (NeurIPS 2024): arxiv:2411.05222
- ResidualViT (2025): arxiv:2509.13255
- RVM: github.com/PeterL1n/RobustVideoMatting
- MatAnyone: github.com/pq-yang/MatAnyone
