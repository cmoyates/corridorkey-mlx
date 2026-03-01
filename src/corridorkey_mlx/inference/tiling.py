"""Tiled inference for large images.

Splits large images into overlapping tiles, runs model on each tile,
then blends results using linear ramp weights in the overlap region.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np

if TYPE_CHECKING:
    from corridorkey_mlx.model.corridorkey import GreenFormer

DEFAULT_TILE_SIZE = 512
DEFAULT_OVERLAP = 64


def _compute_tile_coords(
    image_size: int,
    tile_size: int,
    overlap: int,
) -> list[tuple[int, int]]:
    """Compute (start, end) positions for tiles along one axis.

    Tiles overlap by `overlap` pixels. Last tile is clamped to image boundary.
    """
    if image_size <= tile_size:
        return [(0, image_size)]

    stride = tile_size - overlap
    coords: list[tuple[int, int]] = []
    start = 0
    while start < image_size:
        end = min(start + tile_size, image_size)
        # Ensure last tile is full-sized by shifting start back
        if end - start < tile_size and start > 0:
            start = max(0, end - tile_size)
        coords.append((start, end))
        if end == image_size:
            break
        start += stride
    return coords


def _make_blend_weights_2d(
    tile_h: int,
    tile_w: int,
    overlap: int,
    position: tuple[bool, bool, bool, bool],
) -> np.ndarray:
    """Create 2D linear blend weights for a tile.

    Args:
        tile_h, tile_w: Tile spatial dimensions.
        overlap: Overlap size in pixels.
        position: (has_top_neighbor, has_bottom_neighbor, has_left_neighbor, has_right_neighbor)

    Returns:
        (tile_h, tile_w) float32 weight array with linear ramps in overlap regions.
    """
    weights = np.ones((tile_h, tile_w), dtype=np.float32)
    has_top, has_bottom, has_left, has_right = position

    if overlap <= 0:
        return weights

    ramp = np.linspace(0.0, 1.0, overlap, dtype=np.float32)

    if has_top and overlap <= tile_h:
        weights[:overlap, :] *= ramp[:, None]
    if has_bottom and overlap <= tile_h:
        weights[-overlap:, :] *= ramp[::-1, None]
    if has_left and overlap <= tile_w:
        weights[:, :overlap] *= ramp[None, :]
    if has_right and overlap <= tile_w:
        weights[:, -overlap:] *= ramp[None, ::-1]

    return weights


def tiled_inference(
    model: GreenFormer,
    x: mx.array,
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> dict[str, mx.array]:
    """Run model on overlapping tiles and blend the results.

    Args:
        model: Loaded GreenFormer (must accept tile_size x tile_size input).
        x: Full-resolution input (1, H, W, 4) NHWC.
        tile_size: Size of each square tile. Must match model.backbone.img_size.
        overlap: Overlap in pixels between adjacent tiles.

    Returns:
        Dict with 'alpha_final' and 'fg_final' blended at full resolution.
    """
    if x.shape[0] != 1:
        msg = f"Tiled inference only supports batch_size=1, got {x.shape[0]}"
        raise ValueError(msg)

    _, full_h, full_w, _ = x.shape

    # If image fits in one tile, just run normally
    if full_h <= tile_size and full_w <= tile_size:
        return model(x)

    y_coords = _compute_tile_coords(full_h, tile_size, overlap)
    x_coords = _compute_tile_coords(full_w, tile_size, overlap)

    # Accumulators for weighted blending (numpy for simplicity)
    alpha_accum = np.zeros((full_h, full_w, 1), dtype=np.float32)
    fg_accum = np.zeros((full_h, full_w, 3), dtype=np.float32)
    weight_accum = np.zeros((full_h, full_w, 1), dtype=np.float32)

    for yi, (y_start, y_end) in enumerate(y_coords):
        for xi, (x_start, x_end) in enumerate(x_coords):
            tile = x[:, y_start:y_end, x_start:x_end, :]

            # Pad to tile_size if needed (edge tiles may be smaller)
            pad_h = tile_size - (y_end - y_start)
            pad_w = tile_size - (x_end - x_start)
            if pad_h > 0 or pad_w > 0:
                tile = mx.pad(tile, [(0, 0), (0, pad_h), (0, pad_w), (0, 0)])

            out = model(tile)
            # NOTE: mx.eval is MLX array materialization, not Python eval()
            mx.eval(out)  # noqa: S307

            alpha_tile = np.array(out["alpha_final"][0])  # (tile_h, tile_w, 1)
            fg_tile = np.array(out["fg_final"][0])  # (tile_h, tile_w, 3)

            # Crop padding
            actual_h = y_end - y_start
            actual_w = x_end - x_start
            alpha_tile = alpha_tile[:actual_h, :actual_w, :]
            fg_tile = fg_tile[:actual_h, :actual_w, :]

            # Blend weights
            position = (
                yi > 0,  # has top neighbor
                yi < len(y_coords) - 1,  # has bottom neighbor
                xi > 0,  # has left neighbor
                xi < len(x_coords) - 1,  # has right neighbor
            )
            w = _make_blend_weights_2d(actual_h, actual_w, overlap, position)
            w3d = w[:, :, None]  # (H, W, 1)

            alpha_accum[y_start:y_end, x_start:x_end, :] += alpha_tile * w3d
            fg_accum[y_start:y_end, x_start:x_end, :] += fg_tile * w3d
            weight_accum[y_start:y_end, x_start:x_end, :] += w3d

    # Normalize by accumulated weights
    weight_accum = np.maximum(weight_accum, 1e-8)
    alpha_final = mx.array(alpha_accum / weight_accum)[None]  # (1, H, W, 1)
    fg_final = mx.array(fg_accum / weight_accum)[None]  # (1, H, W, 3)

    return {"alpha_final": alpha_final, "fg_final": fg_final}
