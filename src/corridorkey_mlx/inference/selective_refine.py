"""Experimental selective refinement inference.

Coarse-to-fine pipeline: run a cheap full-frame coarse pass, identify uncertain
regions via alpha gradient/transition analysis, then selectively re-infer only
those regions at high resolution. Reduces compute on easy regions (solid
foreground/background) while preserving quality on hard edges.

WARNING: This module is experimental and not part of the stable API.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from corridorkey_mlx.inference.tiling import _compute_tile_coords

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
