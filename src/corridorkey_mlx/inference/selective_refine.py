"""Experimental selective refinement inference.

Coarse-to-fine pipeline: run a cheap full-frame coarse pass, identify uncertain
regions via alpha gradient/transition analysis, then selectively re-infer only
those regions at high resolution. Reduces compute on easy regions (solid
foreground/background) while preserving quality on hard edges.

WARNING: This module is experimental and not part of the stable API.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np
from PIL import Image

from corridorkey_mlx.inference.tiling import (
    _compute_tile_coords,
    _make_blend_weights_2d,
)
from corridorkey_mlx.io.image import normalize_rgb

if TYPE_CHECKING:
    from corridorkey_mlx.model.corridorkey import GreenFormer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration & result types
# ---------------------------------------------------------------------------


@dataclass
class SelectiveRefineConfig:
    """Configuration for selective refinement inference."""

    coarse_size: int = 512
    """Full-frame coarse pass resolution (square)."""

    tile_size: int = 512
    """Tile crop size — must match model img_size."""

    tile_overlap: int = 64
    """Overlap between adjacent tiles in pixels."""

    uncertainty_low: float = 0.05
    """Alpha values in [low, high] are considered uncertain."""

    uncertainty_high: float = 0.95
    """Alpha values in [low, high] are considered uncertain."""

    gradient_threshold: float = 0.02
    """Sobel magnitude threshold for edge detection."""

    dilation_radius: int = 32
    """Dilate uncertain regions by this many pixels before tiling."""

    min_tile_coverage: float = 0.05
    """Skip tiles with less than this fraction of uncertain pixels."""

    compile: bool = True
    """Whether to use mx.compile for model forward passes."""


@dataclass
class SelectiveRefineResult:
    """Result of selective refinement inference."""

    alpha_final: np.ndarray
    """(H, W) uint8 — final alpha matte."""

    fg_final: np.ndarray
    """(H, W, 3) uint8 — final foreground."""

    coarse_alpha: np.ndarray
    """(H, W) uint8 — bicubic-upscaled coarse alpha."""

    uncertainty_mask: np.ndarray
    """(H, W) bool — regions selected for refinement."""

    tile_coords: list[tuple[int, int, int, int]]
    """Selected tile (y0, x0, y1, x1) coords in full-res space."""

    stats: dict[str, object] = field(default_factory=dict)
    """Timing, tile count, coverage fraction, etc."""


# ---------------------------------------------------------------------------
# Binary dilation (numpy-only, no scipy)
# ---------------------------------------------------------------------------


def _binary_dilation(mask: np.ndarray, radius: int) -> np.ndarray:
    """Dilate a 2D boolean mask by `radius` pixels using iterative 3x3 max.

    Args:
        mask: (H, W) boolean array.
        radius: Number of dilation iterations.

    Returns:
        (H, W) boolean array — dilated mask.
    """
    if radius <= 0:
        return mask.copy()

    result = mask.astype(np.uint8)
    for _ in range(radius):
        padded = np.pad(result, 1, mode="constant", constant_values=0)
        # 3x3 max filter via shifted slices
        dilated = padded[:-2, :-2]  # top-left
        dilated = np.maximum(dilated, padded[:-2, 1:-1])  # top
        dilated = np.maximum(dilated, padded[:-2, 2:])  # top-right
        dilated = np.maximum(dilated, padded[1:-1, :-2])  # left
        dilated = np.maximum(dilated, padded[1:-1, 1:-1])  # center
        dilated = np.maximum(dilated, padded[1:-1, 2:])  # right
        dilated = np.maximum(dilated, padded[2:, :-2])  # bottom-left
        dilated = np.maximum(dilated, padded[2:, 1:-1])  # bottom
        dilated = np.maximum(dilated, padded[2:, 2:])  # bottom-right
        result = dilated
    return result.astype(bool)


# ---------------------------------------------------------------------------
# Uncertainty mask
# ---------------------------------------------------------------------------


def compute_uncertainty_mask(
    alpha_coarse: np.ndarray,
    config: SelectiveRefineConfig,
) -> np.ndarray:
    """Compute binary uncertainty mask from coarse alpha prediction.

    Identifies pixels in the transition band (neither clearly FG nor BG)
    and high-gradient edges, then dilates the union.

    Args:
        alpha_coarse: (H, W) float32 in [0, 1].
        config: Selective refinement configuration.

    Returns:
        (H, W) boolean mask at coarse resolution.
    """
    # 1. Transition band
    band = (alpha_coarse > config.uncertainty_low) & (
        alpha_coarse < config.uncertainty_high
    )

    # 2. High-gradient edges (finite-difference Sobel approximation)
    dy = np.abs(np.diff(alpha_coarse, axis=0, prepend=alpha_coarse[:1, :]))
    dx = np.abs(np.diff(alpha_coarse, axis=1, prepend=alpha_coarse[:, :1]))
    gradient = np.sqrt(dy**2 + dx**2)
    edges = gradient > config.gradient_threshold

    # 3. Union + dilation
    mask = band | edges
    mask = _binary_dilation(mask, config.dilation_radius)

    coverage = float(np.mean(mask))
    if coverage > 0.90:
        logger.warning(
            "Uncertainty mask covers %.1f%% of frame — "
            "selective refinement may not provide speedup",
            coverage * 100,
        )
    elif coverage < 0.01:
        logger.warning(
            "Uncertainty mask covers %.1f%% of frame — "
            "returning upscaled coarse may be sufficient",
            coverage * 100,
        )

    return mask


# ---------------------------------------------------------------------------
# Tile selection
# ---------------------------------------------------------------------------


def select_tiles(
    mask_fullres: np.ndarray,
    full_h: int,
    full_w: int,
    config: SelectiveRefineConfig,
) -> list[tuple[int, int, int, int]]:
    """Select tiles from full-res grid that overlap uncertain regions.

    Args:
        mask_fullres: (H, W) boolean mask at full resolution.
        full_h, full_w: Full-resolution image dimensions.
        config: Selective refinement configuration.

    Returns:
        List of (y0, x0, y1, x1) tile coordinates in full-res pixel space.
    """
    y_coords = _compute_tile_coords(full_h, config.tile_size, config.tile_overlap)
    x_coords = _compute_tile_coords(full_w, config.tile_size, config.tile_overlap)

    total_tiles = len(y_coords) * len(x_coords)
    selected: list[tuple[int, int, int, int]] = []

    for y_start, y_end in y_coords:
        for x_start, x_end in x_coords:
            tile_mask = mask_fullres[y_start:y_end, x_start:x_end]
            tile_pixels = tile_mask.size
            if tile_pixels == 0:
                continue
            coverage = float(np.sum(tile_mask)) / tile_pixels
            if coverage >= config.min_tile_coverage:
                selected.append((y_start, x_start, y_end, x_end))

    logger.info(
        "Tile selection: %d / %d tiles (%.1f%% of frame)",
        len(selected),
        total_tiles,
        len(selected) / max(total_tiles, 1) * 100,
    )
    return selected


# ---------------------------------------------------------------------------
# Gaussian blur (numpy-only, no scipy)
# ---------------------------------------------------------------------------


def _gaussian_kernel_1d(sigma: float) -> np.ndarray:
    """Create a 1D Gaussian kernel with given sigma."""
    radius = int(np.ceil(3 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def _gaussian_blur(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Apply separable Gaussian blur to a 2D float32 array.

    Args:
        arr: (H, W) float32 array.
        sigma: Standard deviation of the Gaussian kernel.

    Returns:
        (H, W) float32 blurred array.
    """
    if sigma <= 0:
        return arr.copy()

    kernel = _gaussian_kernel_1d(sigma)
    radius = len(kernel) // 2

    # Horizontal pass (pad + convolve via shifted slices)
    padded = np.pad(arr, ((0, 0), (radius, radius)), mode="edge")
    result = np.zeros_like(arr)
    for i, w in enumerate(kernel):
        result += w * padded[:, i : i + arr.shape[1]]

    # Vertical pass
    padded = np.pad(result, ((radius, radius), (0, 0)), mode="edge")
    output = np.zeros_like(arr)
    for i, w in enumerate(kernel):
        output += w * padded[i : i + arr.shape[0], :]

    return output


# ---------------------------------------------------------------------------
# Stitching / blending
# ---------------------------------------------------------------------------


def stitch_results(
    coarse_alpha: np.ndarray,
    coarse_fg: np.ndarray,
    tile_results: list[dict[str, np.ndarray]],
    tile_coords: list[tuple[int, int, int, int]],
    mask_fullres: np.ndarray,
    config: SelectiveRefineConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend refined tiles into upscaled coarse result.

    Two-layer blending:
    1. Tile-to-tile: weighted accumulation using _make_blend_weights_2d
       (linear ramps in overlap regions).
    2. Coarse-to-refined: soft blend using Gaussian-blurred uncertainty mask.

    Args:
        coarse_alpha: (H, W) float32 in [0, 1] — bicubic-upscaled coarse alpha.
        coarse_fg: (H, W, 3) float32 in [0, 1] — bicubic-upscaled coarse fg.
        tile_results: Per-tile dicts with 'alpha' (ts, ts) and 'fg' (ts, ts, 3).
        tile_coords: (y0, x0, y1, x1) per tile in full-res pixel space.
        mask_fullres: (H, W) bool — uncertainty mask at full resolution.
        config: Selective refinement configuration.

    Returns:
        (alpha, fg) as float32 arrays at full resolution.
        alpha: (H, W) float32 in [0, 1].
        fg: (H, W, 3) float32 in [0, 1].
    """
    full_h, full_w = coarse_alpha.shape[:2]

    if len(tile_coords) == 0:
        return coarse_alpha.copy(), coarse_fg.copy()

    # -- Step 1: Accumulate refined tiles with blend weights --
    refined_alpha = np.zeros((full_h, full_w), dtype=np.float32)
    refined_fg = np.zeros((full_h, full_w, 3), dtype=np.float32)
    weight_accum = np.zeros((full_h, full_w), dtype=np.float32)

    # Pre-compute tile grid indices for neighbor detection
    y_coords = _compute_tile_coords(full_h, config.tile_size, config.tile_overlap)
    x_coords = _compute_tile_coords(full_w, config.tile_size, config.tile_overlap)

    # Build a set of selected tile grid positions for neighbor lookup
    selected_grid_pos: set[tuple[int, int]] = set()
    for y0, x0, _y1, _x1 in tile_coords:
        for yi, (ys, _ye) in enumerate(y_coords):
            for xi, (xs, _xe) in enumerate(x_coords):
                if ys == y0 and xs == x0:
                    selected_grid_pos.add((yi, xi))

    for tile_idx, (y0, x0, y1, x1) in enumerate(tile_coords):
        actual_h = y1 - y0
        actual_w = x1 - x0
        tile_data = tile_results[tile_idx]

        # Resize tile output to actual crop size if needed
        tile_alpha = tile_data["alpha"]
        tile_fg = tile_data["fg"]
        if tile_alpha.shape[0] != actual_h or tile_alpha.shape[1] != actual_w:
            tile_alpha_pil = Image.fromarray(
                (np.clip(tile_alpha, 0, 1) * 255).astype(np.uint8), mode="L"
            )
            tile_alpha = (
                np.asarray(
                    tile_alpha_pil.resize((actual_w, actual_h), Image.BICUBIC),
                    dtype=np.float32,
                )
                / 255.0
            )
            tile_fg_pil = Image.fromarray(
                (np.clip(tile_fg, 0, 1) * 255).astype(np.uint8), mode="RGB"
            )
            tile_fg = (
                np.asarray(
                    tile_fg_pil.resize((actual_w, actual_h), Image.BICUBIC),
                    dtype=np.float32,
                )
                / 255.0
            )

        # Find grid position for this tile
        yi_pos = xi_pos = 0
        for yi, (ys, _ye) in enumerate(y_coords):
            for xi, (xs, _xe) in enumerate(x_coords):
                if ys == y0 and xs == x0:
                    yi_pos, xi_pos = yi, xi
                    break

        # Only apply blend ramps toward neighbors that are also selected
        has_top = (yi_pos - 1, xi_pos) in selected_grid_pos
        has_bottom = (yi_pos + 1, xi_pos) in selected_grid_pos
        has_left = (yi_pos, xi_pos - 1) in selected_grid_pos
        has_right = (yi_pos, xi_pos + 1) in selected_grid_pos

        w = _make_blend_weights_2d(
            actual_h, actual_w, config.tile_overlap, (has_top, has_bottom, has_left, has_right)
        )

        refined_alpha[y0:y1, x0:x1] += tile_alpha * w
        refined_fg[y0:y1, x0:x1, :] += tile_fg * w[:, :, np.newaxis]
        weight_accum[y0:y1, x0:x1] += w

    # Normalize by accumulated weights (only where tiles were placed)
    has_tiles = weight_accum > 1e-8
    refined_alpha[has_tiles] /= weight_accum[has_tiles]
    for c in range(3):
        refined_fg[has_tiles, c] /= weight_accum[has_tiles]

    # -- Step 2: Soft blend between coarse and refined --
    blur_sigma = max(config.dilation_radius / 2.0, 1.0)
    blend_weight = _gaussian_blur(
        mask_fullres.astype(np.float32), sigma=blur_sigma
    )
    blend_weight = np.clip(blend_weight, 0.0, 1.0)

    alpha_out = coarse_alpha * (1.0 - blend_weight) + refined_alpha * blend_weight
    fg_out = coarse_fg * (1.0 - blend_weight[:, :, np.newaxis]) + (
        refined_fg * blend_weight[:, :, np.newaxis]
    )

    return alpha_out, fg_out


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------


def _resize_to_square(arr: np.ndarray, size: int) -> np.ndarray:
    """Resize HW or HWC array to (size, size) using PIL bicubic."""
    if arr.ndim == 2:
        pil = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), mode="L")
        pil = pil.resize((size, size), Image.BICUBIC)
        return np.asarray(pil, dtype=np.float32) / 255.0
    # HWC
    pil = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    pil = pil.resize((size, size), Image.BICUBIC)
    return np.asarray(pil, dtype=np.float32) / 255.0


def _build_input(rgb_f32: np.ndarray, hint_f32: np.ndarray) -> mx.array:
    """Build (1, H, W, 4) NHWC input from RGB and alpha hint arrays.

    Args:
        rgb_f32: (H, W, 3) float32 in [0, 1].
        hint_f32: (H, W) or (H, W, 1) float32 in [0, 1].
    """
    normalized = normalize_rgb(rgb_f32)
    if hint_f32.ndim == 2:
        hint_f32 = hint_f32[:, :, np.newaxis]
    combined = np.concatenate([normalized, hint_f32], axis=-1)
    return mx.array(combined[np.newaxis])


def _run_model(model: GreenFormer, x: mx.array) -> dict[str, mx.array]:
    """Run model and materialize outputs."""
    outputs = model(x)
    # NOTE: mx.eval is MLX array materialization, not Python eval()
    mx.eval(outputs)  # noqa: S307
    return outputs


# ---------------------------------------------------------------------------
# Two-pass selective refinement pipeline
# ---------------------------------------------------------------------------


def selective_refine(
    model: GreenFormer,
    image: np.ndarray,
    alpha_hint: np.ndarray,
    config: SelectiveRefineConfig | None = None,
) -> SelectiveRefineResult:
    """Run two-pass selective refinement inference.

    1. Coarse pass: resize full image to coarse_size, run model, get alpha_coarse
    2. Uncertainty mask from coarse alpha
    3. Upsample mask to full resolution
    4. Select tiles overlapping uncertain regions
    5. Run model on each selected tile from full-res image
    6. Return raw results (blending/stitching in phase 3)

    Args:
        model: Loaded GreenFormer. img_size must equal config.tile_size.
        image: (H, W, 3) uint8 RGB input.
        alpha_hint: (H, W) or (H, W, 1) uint8 coarse alpha hint.
        config: Pipeline configuration. Uses defaults if None.

    Returns:
        SelectiveRefineResult with coarse + per-tile outputs.
    """
    if config is None:
        config = SelectiveRefineConfig()

    full_h, full_w = image.shape[:2]
    stats: dict[str, object] = {}

    # -- Convert to float32 [0, 1] --
    rgb_f32 = image.astype(np.float32) / 255.0
    hint_f32 = alpha_hint.astype(np.float32) / 255.0
    if hint_f32.ndim == 3:
        hint_f32 = hint_f32[:, :, 0]

    # ---------------------------------------------------------------
    # Step 1: Coarse pass
    # ---------------------------------------------------------------
    t0 = time.perf_counter()

    coarse_rgb = _resize_to_square(rgb_f32, config.coarse_size)
    coarse_hint = _resize_to_square(hint_f32, config.coarse_size)
    coarse_input = _build_input(coarse_rgb, coarse_hint)

    coarse_out = _run_model(model, coarse_input)

    coarse_alpha_f32 = np.array(coarse_out["alpha_final"][0, :, :, 0])  # (cs, cs)

    stats["coarse_ms"] = (time.perf_counter() - t0) * 1000

    # -- Upsample coarse alpha to full res for output --
    coarse_alpha_pil = Image.fromarray(
        (np.clip(coarse_alpha_f32, 0, 1) * 255).astype(np.uint8), mode="L"
    )
    coarse_alpha_upscaled = np.asarray(
        coarse_alpha_pil.resize((full_w, full_h), Image.BICUBIC), dtype=np.uint8
    )

    # ---------------------------------------------------------------
    # Step 2: Uncertainty mask (at coarse resolution)
    # ---------------------------------------------------------------
    mask_coarse = compute_uncertainty_mask(coarse_alpha_f32, config)

    # Upsample mask to full resolution
    mask_pil = Image.fromarray(mask_coarse.astype(np.uint8) * 255, mode="L")
    mask_fullres = (
        np.asarray(mask_pil.resize((full_w, full_h), Image.NEAREST), dtype=np.uint8)
        > 127
    )
    stats["mask_coverage"] = float(np.mean(mask_fullres))

    # ---------------------------------------------------------------
    # Step 3: Tile selection
    # ---------------------------------------------------------------
    tile_coords = select_tiles(mask_fullres, full_h, full_w, config)
    stats["tiles_selected"] = len(tile_coords)
    stats["tiles_total"] = len(
        _compute_tile_coords(full_h, config.tile_size, config.tile_overlap)
    ) * len(_compute_tile_coords(full_w, config.tile_size, config.tile_overlap))

    # ---------------------------------------------------------------
    # Step 4: Per-tile refinement
    # ---------------------------------------------------------------
    t1 = time.perf_counter()

    tile_results: list[dict[str, np.ndarray]] = []
    for y0, x0, y1, x1 in tile_coords:
        # Extract tile from full-res image
        tile_rgb = rgb_f32[y0:y1, x0:x1, :]
        tile_hint = hint_f32[y0:y1, x0:x1]

        actual_h, actual_w = tile_rgb.shape[:2]

        # Resize tile to model input size if needed
        if actual_h != config.tile_size or actual_w != config.tile_size:
            tile_rgb = _resize_to_square(tile_rgb, config.tile_size)
            tile_hint = _resize_to_square(tile_hint, config.tile_size)

        tile_input = _build_input(tile_rgb, tile_hint)
        tile_out = _run_model(model, tile_input)

        # Store as numpy at tile_size resolution
        tile_results.append({
            "alpha": np.array(tile_out["alpha_final"][0, :, :, 0]),  # (ts, ts)
            "fg": np.array(tile_out["fg_final"][0]),  # (ts, ts, 3)
        })

    stats["tile_refine_ms"] = (time.perf_counter() - t1) * 1000

    # ---------------------------------------------------------------
    # Step 5: Upsample coarse fg + stitch
    # ---------------------------------------------------------------
    coarse_fg_f32 = np.array(coarse_out["fg_final"][0])  # (cs, cs, 3)
    coarse_fg_pil = Image.fromarray(
        (np.clip(coarse_fg_f32, 0, 1) * 255).astype(np.uint8), mode="RGB"
    )
    coarse_fg_up_f32 = (
        np.asarray(
            coarse_fg_pil.resize((full_w, full_h), Image.BICUBIC), dtype=np.float32
        )
        / 255.0
    )
    coarse_alpha_up_f32 = coarse_alpha_upscaled.astype(np.float32) / 255.0

    t2 = time.perf_counter()

    if len(tile_coords) == 0:
        logger.warning("No tiles selected — returning upscaled coarse result")
        alpha_out_f32 = coarse_alpha_up_f32
        fg_out_f32 = coarse_fg_up_f32
    else:
        alpha_out_f32, fg_out_f32 = stitch_results(
            coarse_alpha_up_f32,
            coarse_fg_up_f32,
            tile_results,
            tile_coords,
            mask_fullres,
            config,
        )

    stats["stitch_ms"] = (time.perf_counter() - t2) * 1000
    stats["total_ms"] = (
        stats["coarse_ms"] + stats["tile_refine_ms"] + stats["stitch_ms"]
    )
    stats["tiles_selected"] = len(tile_coords)

    # Convert to uint8
    alpha_final = (np.clip(alpha_out_f32, 0, 1) * 255).astype(np.uint8)
    fg_final = (np.clip(fg_out_f32, 0, 1) * 255).astype(np.uint8)

    return SelectiveRefineResult(
        alpha_final=alpha_final,
        fg_final=fg_final,
        coarse_alpha=coarse_alpha_upscaled,
        uncertainty_mask=mask_fullres,
        tile_coords=tile_coords,
        stats=stats,
    )
