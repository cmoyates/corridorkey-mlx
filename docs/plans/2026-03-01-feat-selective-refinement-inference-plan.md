---
title: "feat: Experimental Selective Refinement Inference"
type: feat
date: 2026-03-01
status: draft
---

# Experimental Selective Refinement Inference

## Overview

Add a coarse-to-fine inference mode that runs a cheap full-frame coarse pass, identifies uncertain regions via alpha gradient/transition analysis, then selectively re-infers only those regions at high resolution. Goal: reduce compute on easy regions (solid foreground/background) while preserving quality on hard edges (hair, blur, translucency).

## Problem Statement

Current MLX inference runs the full model at native resolution (2048x2048) for every pixel. Most of the frame is trivially foreground or background — only edge regions need high-res processing. At 2048, inference is slow (~1.1-1.2x vs PyTorch MPS), and the bottleneck is raw compute volume, not backend efficiency. Reducing the number of high-res pixels processed is the most direct path to speedup.

## Proposed Solution

Single-model, two-pass pipeline:

1. **Coarse pass** — resize full image to `coarse_size` (default 512), run `GreenFormer`, get `alpha_coarse`
2. **Uncertainty mask** — from coarse alpha, identify transition-band pixels + high-gradient edges, dilate
3. **Tile selection** — convert mask to tile grid, merge overlapping tiles
4. **Selective refinement** — extract tiles from full-res image, run model on each
5. **Stitching** — blend refined tiles into bicubic-upscaled coarse result

Key insight: use **one model instance** at `tile_size` (default 512) for both passes. Coarse pass = full frame resized to 512. Tiles = 512x512 crops from full-res image. No need to load two models or manage two pos_embed sizes.

## Technical Approach

### Architecture

```
Full-res image (e.g. 2048x2048)
    |
    +---> Resize to coarse_size (512x512) ---> Model ---> alpha_coarse
    |                                                         |
    |                                          Uncertainty mask (gradient + band)
    |                                                         |
    |                                          Tile selection (grid coords)
    |                                                         |
    +--- Extract tiles from full-res ---------> Model ------> refined tiles
    |                                                         |
    +--- Bicubic upsample coarse to full-res                  |
    |                                                         |
    +--- Blend: coarse_upsampled + refined_tiles -----------> final output
```

### New Files

| File | Purpose |
|---|---|
| `src/corridorkey_mlx/inference/selective_refine.py` | Core pipeline: uncertainty mask, tile selection, two-pass inference, stitching |
| `scripts/experiment_selective_refine.py` | CLI: run baseline vs selective, produce comparison report |
| `tests/test_selective_refine.py` | Unit tests for mask, tile selection, blending logic |

### Module: `selective_refine.py`

```python
# Public API

@dataclass
class SelectiveRefineConfig:
    coarse_size: int = 512          # full-frame coarse pass resolution
    tile_size: int = 512            # tile crop size (matches model img_size)
    tile_overlap: int = 64          # overlap between adjacent tiles
    uncertainty_low: float = 0.05   # alpha values in [low, high] = uncertain
    uncertainty_high: float = 0.95
    gradient_threshold: float = 0.02  # Sobel magnitude threshold
    dilation_radius: int = 32       # dilate uncertain regions before tiling
    min_tile_coverage: float = 0.05 # skip tiles with <5% uncertain pixels
    compile: bool = True

@dataclass
class SelectiveRefineResult:
    alpha_final: np.ndarray         # (H, W) uint8
    fg_final: np.ndarray            # (H, W, 3) uint8
    coarse_alpha: np.ndarray        # (H, W) uint8 — upscaled coarse
    uncertainty_mask: np.ndarray    # (H, W) bool — refinement mask
    tile_coords: list[tuple]        # selected tile (y0, x0, y1, x1) in full-res
    stats: dict                     # timing, tile count, coverage fraction

def selective_refine(
    model: GreenFormer,
    image: np.ndarray,           # (H, W, 3) uint8
    alpha_hint: np.ndarray,      # (H, W) or (H, W, 1) uint8
    config: SelectiveRefineConfig = ...,
) -> SelectiveRefineResult: ...
```

#### Uncertainty Mask Generation

```python
def compute_uncertainty_mask(
    alpha_coarse: np.ndarray,   # (H, W) float32 in [0, 1]
    config: SelectiveRefineConfig,
) -> np.ndarray:
    """Returns (H, W) bool mask at coarse resolution."""
    # 1. Transition band: alpha in [uncertainty_low, uncertainty_high]
    band = (alpha_coarse > config.uncertainty_low) & (alpha_coarse < config.uncertainty_high)

    # 2. High-gradient edges (simple Sobel via numpy finite differences)
    dy = np.abs(np.diff(alpha_coarse, axis=0, prepend=alpha_coarse[:1, :]))
    dx = np.abs(np.diff(alpha_coarse, axis=1, prepend=alpha_coarse[:, :1]))
    gradient = np.sqrt(dy**2 + dx**2)
    edges = gradient > config.gradient_threshold

    # 3. Union + dilation
    mask = band | edges
    mask = binary_dilation(mask, radius=config.dilation_radius)
    return mask
```

Dilation: use simple box filter convolution (no scipy dependency) — iterate `radius` times with 3x3 max filter, or do a single pass with a square structuring element via `np.lib.stride_tricks`.

#### Tile Selection

Reuse `tiling._compute_tile_coords` for the full-res grid, then filter to tiles that overlap the upscaled uncertainty mask above `min_tile_coverage`. Merge overlapping selected tiles when adjacency would create redundant computation.

```python
def select_tiles(
    mask_fullres: np.ndarray,    # (H, W) bool at full resolution
    full_h: int, full_w: int,
    config: SelectiveRefineConfig,
) -> list[tuple[int, int, int, int]]:
    """Returns list of (y0, x0, y1, x1) tile coords in full-res space."""
```

#### Stitching / Blending

Reuse `tiling._make_blend_weights_2d` for overlap feathering between refined tiles. For the coarse-to-refined boundary, use the uncertainty mask as a soft blend weight (dilated + Gaussian-blurred edge) so refined regions fade smoothly into the upscaled coarse result.

```python
def stitch_results(
    coarse_upscaled: np.ndarray,   # (H, W, C) float32
    refined_tiles: list[...],       # per-tile alpha + fg
    tile_coords: list[tuple],
    mask_fullres: np.ndarray,       # (H, W) float32 blend weight [0,1]
    config: SelectiveRefineConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (alpha, fg) at full resolution."""
```

### Script: `experiment_selective_refine.py`

```
Usage:
  uv run python scripts/experiment_selective_refine.py \
    --image samples/sample.png \
    --hint samples/hint.png \
    --checkpoint checkpoints/corridorkey_mlx.safetensors \
    --coarse-size 512 \
    --output-dir output/selective_refine/

Output:
  output/selective_refine/
    baseline_alpha.png
    baseline_fg.png
    selective_alpha.png
    selective_fg.png
    uncertainty_mask.png
    tile_overlay.png          # source image with tile rects drawn
    diff_alpha.png            # |baseline - selective| heatmap
    report.txt                # timing, tile count, coverage, diff stats
```

Report format:
```
=== Selective Refinement Experiment ===
Image: 2048x2048
Coarse size: 512
Tile size: 512, overlap: 64

Baseline:
  Time: 4200 ms

Selective:
  Coarse pass: 180 ms
  Tiles selected: 12 / 49 (24.5% of frame)
  Tile refinement: 1100 ms
  Stitching: 20 ms
  Total: 1300 ms
  Speedup: 3.2x

Quality (vs baseline):
  Alpha max_abs_diff: 0.012
  Alpha mean_abs_diff: 0.0003
  FG max_abs_diff: 0.008
  FG mean_abs_diff: 0.0002

Warnings: none
```

### Implementation Phases

#### Phase 1: Uncertainty mask + tile selection (no model needed)

- [x] `compute_uncertainty_mask()` with synthetic alpha
- [x] `select_tiles()` with grid logic
- [x] Binary dilation (numpy-only)
- [x] Unit tests with deterministic synthetic data (19 tests)

**Files:** `selective_refine.py` (mask + tile funcs), `test_selective_refine.py`

#### Phase 2: Two-pass pipeline

- [x] Coarse pass: resize + preprocess + model + postprocess
- [x] Tile extraction from full-res
- [x] Per-tile model forward pass
- [x] Return raw results (no blending yet)
- [x] 7 pipeline tests (type, shapes, dtypes, stats, edge cases)

**Files:** `selective_refine.py` (main pipeline func)

#### Phase 3: Stitching + blending

- [x] Bicubic upsample coarse to full-res
- [x] Accumulate refined tiles with blend weights (_make_blend_weights_2d)
- [x] Soft boundary blending (Gaussian-blurred uncertainty mask)
- [x] Final compositing integrated into selective_refine()
- [x] 10 new tests (Gaussian blur + stitch_results + pipeline stitch_ms)

**Files:** `selective_refine.py` (stitch func)

#### Phase 4: Script + benchmark + comparison

- CLI script with all knobs exposed
- Baseline vs selective timing
- Quality diff metrics + artifact saves
- Report generation

**Files:** `scripts/experiment_selective_refine.py`

#### Phase 5: Tests + docs

- Add narrow unit tests for mask/tile/blend
- Update README with experimental section
- Mark all new code as experimental

**Files:** `tests/test_selective_refine.py`, `README.md`

## Acceptance Criteria

### Functional

- [ ] `selective_refine()` produces alpha + fg at same resolution as input
- [ ] Baseline path (`pipeline.infer`, `engine.process_frame`) unchanged
- [ ] Uncertainty mask correctly identifies transition-band + edge pixels
- [ ] Tile selection covers all uncertain regions (false positives OK, false negatives bad)
- [ ] Blending produces no visible seams at tile boundaries
- [ ] Boundary between refined and coarse regions is smooth
- [ ] Script produces timing + quality comparison report
- [ ] Works with real checkpoint at 2048 resolution

### Non-Functional

- [ ] No new external dependencies (numpy-only for mask/dilation)
- [ ] All new code clearly labeled experimental
- [ ] Existing 94 tests still pass
- [ ] New tests run in default suite (no checkpoint needed)

### Quality Gates

- [ ] Alpha diff vs baseline < 0.05 max_abs for conservative defaults
- [ ] Speedup measurable (>1.5x) when <50% of tiles selected
- [ ] Warning emitted when mask covers >90% or <1% of frame

## Design Decisions

### Single model vs two models

**Chosen: single model at tile_size.** Both coarse (full frame resized to 512) and tiles (512 crops from full-res) use the same `GreenFormer(img_size=512)` instance. Avoids loading weights twice, simplifies pos_embed, halves memory.

**Tradeoff:** coarse pass at 512 may lose detail vs running at 1024. If coarse quality matters, can add a `coarse_size != tile_size` mode later with two model instances.

### Dilation implementation

**Chosen: numpy-only box dilation.** Iterate N times with 3x3 max-filter, or single pass with `np.maximum` over shifted arrays. No scipy dependency.

### Blend strategy at coarse/refined boundary

**Chosen: soft mask blending.** Gaussian-blur the binary uncertainty mask to create a smooth [0,1] blend weight. `result = coarse * (1 - weight) + refined * weight`. Avoids hard seams without complex feathering.

### Tile-to-tile overlap

**Chosen: reuse existing `tiling._make_blend_weights_2d`.** Already tested and working for tile-to-tile seams. Only need to add the coarse-to-refined boundary blending on top.

## Dependencies & Risks

- **Risk:** Coarse pass at 512 produces systematically different uncertainty than a 2048 pass — transition band may shift. Mitigated by conservative defaults (wide uncertainty band, generous dilation).
- **Risk:** Model trained at 2048 may underperform on 512 crops from arbitrary positions (pos_embed mismatch, different effective receptive field). Mitigated: model already works at 512 with interpolated pos_embed (existing tests confirm this).
- **Risk:** Stitching artifacts at boundary between refined and coarse regions. Mitigated by soft blend mask + generous dilation.
- **Risk:** Tile selection overhead may dominate for images where most of the frame is uncertain (e.g. all hair). Expected: in worst case, falls back to ~full-frame tiled inference speed, not slower than baseline.

## Edge Cases & Decisions

### Zero tiles selected (empty mask)
Valid result. Return bicubic-upscaled coarse as final output. Emit `logging.warning("no tiles selected — returning upscaled coarse")`. Return metadata `stats["tiles_selected"] = 0`.

### 100% tiles selected (full coverage)
Emit warning when >80% tiles selected. Still run normally (no auto-fallback to `tiled_inference`) — the coarse pass overhead is small. Let user interpret.

### Alpha hint for tile passes
Use original user-supplied alpha hint, cropped and resized to tile coords. NOT the coarse output — avoids feedback loop not seen during training. Requires keeping original full-res alpha hint in memory.

### Compiled model constraint
`mx.compile(shapeless=False)` traces for one shape. Coarse (512) and tiles (512 crops) use the SAME shape → single compiled model works. If `coarse_size != tile_size` is added later, will need eager coarse or two models.

### Non-square inputs
Coarse pass squash-resizes to `coarse_size x coarse_size` (same as current engine). Tile grid operates in original-resolution pixel space (non-square OK — `_compute_tile_coords` runs independently on H and W). Uncertainty mask from coarse pass is upsampled to original aspect ratio via bicubic before tile selection.

### Boundary tiles
Continue using zero-padding (consistent with existing `tiled_inference`). Known minor artifact at image edges. Document as caveat.

### Coordinate spaces
Three spaces: (a) coarse `coarse_size x coarse_size`, (b) original `H x W`, (c) tile `tile_size x tile_size`. Mask generated in (a), upsampled to (b) for tile selection, tiles extracted from (b), each tile preprocessed independently in (c).

### Blend seam between coarse and refined
Gaussian-blur the binary refinement mask (sigma = dilation_radius / 2) to create smooth [0,1] weight. Blend: `final = coarse_up * (1 - w) + refined * w`. Dilation ensures the seam falls entirely within the refined region.

### Output contract
Return `SelectiveRefineResult` dataclass (not a dict matching `tiled_inference`). This is a standalone experimental API, not a drop-in replacement. Engine integration deferred.

## Open Questions

1. Acceptable alpha diff threshold for "quality preserving"? 0.01? 0.05? Needs empirical data.
2. Include fg gradient in uncertainty mask, or alpha-only sufficient?
3. Always produce visualization artifacts, or gate behind `--visualize`?
4. Worth supporting `coarse_size != tile_size` in v1, or defer?

## References

- `src/corridorkey_mlx/inference/tiling.py` — existing tile coord + blend weight utilities
- `src/corridorkey_mlx/inference/pipeline.py` — `load_model`, `infer` entry points
- `src/corridorkey_mlx/engine.py` — `CorridorKeyMLXEngine.process_frame` resize logic
- `src/corridorkey_mlx/io/image.py` — preprocess/postprocess helpers
- `src/corridorkey_mlx/utils/profiling.py` — `warmup_and_bench`, `time_fn`
- `scripts/bench_mlx.py` — benchmark script pattern
- `scripts/smoke_2048.py` — 2048 smoke test pattern
- `tests/test_tiling_consistency.py` — test pattern for tile logic
