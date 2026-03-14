"""Tiled inference for large images.

Splits large images into overlapping tiles, runs model on each tile,
then blends results using linear ramp weights in the overlap region.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np

if TYPE_CHECKING:
    from corridorkey_mlx.model.corridorkey import GreenFormer

logger = logging.getLogger(__name__)

DEFAULT_TILE_SIZE = 768
DEFAULT_OVERLAP = 128
BBOX_THRESHOLD = 0.01

# Tiles where the alpha hint is uniformly below this (background) or above
# 1-this (foreground) skip inference entirely — the refiner contributes
# nothing when the coarse prediction is already confident.
TILE_SKIP_CONFIDENCE = 0.01


def _compute_tile_coords(
    image_size: int,
    tile_size: int,
    overlap: int,
) -> list[tuple[int, int]]:
    """Compute (start, end) positions for tiles along one axis.

    Tiles overlap by `overlap` pixels. Last tile is clamped to image boundary.
    """
    if overlap >= tile_size:
        msg = f"overlap ({overlap}) must be less than tile_size ({tile_size})"
        raise ValueError(msg)

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


def _find_subject_bbox(
    mask: mx.array,
    margin: int,
) -> tuple[int, int, int, int] | None:
    """Find bounding box of non-zero pixels in mask with margin.

    Args:
        mask: (H, W) alpha hint values.
        margin: Pixels to expand bbox on each side.

    Returns:
        (y_start, y_end, x_start, x_end) or None if mask is all-zero.
    """
    # Find non-zero rows and columns
    row_any = mx.any(mask > BBOX_THRESHOLD, axis=1)  # (H,)
    col_any = mx.any(mask > BBOX_THRESHOLD, axis=0)  # (W,)
    # NOTE: mx.eval is MLX array materialization, not Python eval()
    mx.eval(row_any, col_any)  # noqa: S307

    row_any_np = np.array(row_any)
    col_any_np = np.array(col_any)

    if not np.any(row_any_np):
        return None

    row_indices = np.nonzero(row_any_np)[0]
    col_indices = np.nonzero(col_any_np)[0]

    h, w = mask.shape
    y_start = max(0, int(row_indices[0]) - margin)
    y_end = min(h, int(row_indices[-1]) + 1 + margin)
    x_start = max(0, int(col_indices[0]) - margin)
    x_end = min(w, int(col_indices[-1]) + 1 + margin)

    return y_start, y_end, x_start, x_end


def _single_tile_inference(
    model: GreenFormer,
    x: mx.array,
    bbox: tuple[int, int, int, int],
    tile_size: int,
) -> dict[str, mx.array]:
    """Run inference on a single cropped tile and paste into full-res output.

    Args:
        model: Loaded GreenFormer.
        x: Full-resolution input (1, H, W, 4) NHWC.
        bbox: (y_start, y_end, x_start, x_end) subject bounding box.
        tile_size: Model tile size for padding.

    Returns:
        Dict with 'alpha_final' and 'fg_final' at full resolution.
    """
    _, full_h, full_w, _ = x.shape
    y_start, y_end, x_start, x_end = bbox
    crop_h = y_end - y_start
    crop_w = x_end - x_start

    # Crop input to bbox
    tile = x[:, y_start:y_end, x_start:x_end, :]

    # Pad to tile_size
    pad_h = tile_size - crop_h
    pad_w = tile_size - crop_w
    if pad_h > 0 or pad_w > 0:
        tile = mx.pad(tile, [(0, 0), (0, pad_h), (0, pad_w), (0, 0)])

    out = model(tile)
    # NOTE: mx.eval is MLX array materialization, not Python eval()
    mx.eval(out)  # noqa: S307

    # Crop padding and paste into full-res output
    alpha_crop = np.array(out["alpha_final"][0, :crop_h, :crop_w, :])
    fg_crop = np.array(out["fg_final"][0, :crop_h, :crop_w, :])

    alpha_full = np.zeros((full_h, full_w, 1), dtype=np.float32)
    fg_full = np.zeros((full_h, full_w, 3), dtype=np.float32)

    alpha_full[y_start:y_end, x_start:x_end, :] = alpha_crop
    fg_full[y_start:y_end, x_start:x_end, :] = fg_crop

    return {
        "alpha_final": mx.array(alpha_full)[None],  # (1, H, W, 1)
        "fg_final": mx.array(fg_full)[None],  # (1, H, W, 3)
    }


# Margin around subject bbox (pixels) — prevents edge clipping artifacts
SINGLE_TILE_MARGIN = 64


def tiled_inference(
    model: GreenFormer,
    x: mx.array,
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> dict[str, mx.array]:
    """Run model on overlapping tiles and blend the results.

    If the subject (non-zero alpha hint) fits within a single tile_size x tile_size
    region, runs a single inference instead of the full tiling grid.

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

    # Dynamic single-tile: if subject bbox fits in one tile, skip full grid
    bbox = _find_subject_bbox(x[0, :, :, 3], margin=SINGLE_TILE_MARGIN)
    if bbox is not None:
        y_start, y_end, x_start, x_end = bbox
        bbox_h = y_end - y_start
        bbox_w = x_end - x_start
        if bbox_h <= tile_size and bbox_w <= tile_size:
            logger.debug(
                "Single-tile shortcut: subject bbox %dx%d fits in %dx%d tile",
                bbox_w, bbox_h, tile_size, tile_size,
            )
            return _single_tile_inference(model, x, bbox, tile_size)

    y_coords = _compute_tile_coords(full_h, tile_size, overlap)
    x_coords = _compute_tile_coords(full_w, tile_size, overlap)

    # Accumulators for weighted blending (numpy for simplicity)
    alpha_accum = np.zeros((full_h, full_w, 1), dtype=np.float32)
    fg_accum = np.zeros((full_h, full_w, 3), dtype=np.float32)
    weight_accum = np.zeros((full_h, full_w, 1), dtype=np.float32)

    tiles_skipped = 0
    tiles_total = len(y_coords) * len(x_coords)

    for yi, (y_start, y_end) in enumerate(y_coords):
        for xi, (x_start, x_end) in enumerate(x_coords):
            actual_h = y_end - y_start
            actual_w = x_end - x_start

            # Alpha-hint guided tile skipping: check channel 3 (mask) before inference
            mask_tile = x[0, y_start:y_end, x_start:x_end, 3]
            mask_min = float(mx.min(mask_tile).item())
            mask_max = float(mx.max(mask_tile).item())

            if mask_max < TILE_SKIP_CONFIDENCE:
                # Pure background — skip inference, fill zeros
                alpha_tile = np.zeros((actual_h, actual_w, 1), dtype=np.float32)
                fg_tile = np.zeros((actual_h, actual_w, 3), dtype=np.float32)
                tiles_skipped += 1
            elif mask_min > (1.0 - TILE_SKIP_CONFIDENCE):
                # Pure foreground — skip inference, fill ones + input RGB
                alpha_tile = np.ones((actual_h, actual_w, 1), dtype=np.float32)
                fg_tile = np.array(mx.sigmoid(x[0, y_start:y_end, x_start:x_end, :3]))
                tiles_skipped += 1
            else:
                # Mixed content — run full inference
                tile = x[:, y_start:y_end, x_start:x_end, :]

                # Pad to tile_size if needed (edge tiles may be smaller)
                pad_h = tile_size - actual_h
                pad_w = tile_size - actual_w
                if pad_h > 0 or pad_w > 0:
                    tile = mx.pad(tile, [(0, 0), (0, pad_h), (0, pad_w), (0, 0)])

                out = model(tile)
                mx.eval(out)  # materialize before np.array conversion  # noqa: S307
                alpha_tile = np.array(out["alpha_final"][0])
                fg_tile = np.array(out["fg_final"][0])

                # Crop padding
                alpha_tile = alpha_tile[:actual_h, :actual_w, :]
                fg_tile = fg_tile[:actual_h, :actual_w, :]

            # Blend weights
            position = (
                yi > 0,  # has top neighbor
                yi < len(y_coords) - 1,  # has bottom neighbor
                xi > 0,  # has left neighbor
                xi < len(x_coords) - 1,  # has right neighbor
            )
            blend_weights = _make_blend_weights_2d(actual_h, actual_w, overlap, position)
            blend_weights_3d = blend_weights[:, :, None]  # (H, W, 1)

            alpha_accum[y_start:y_end, x_start:x_end, :] += alpha_tile * blend_weights_3d
            fg_accum[y_start:y_end, x_start:x_end, :] += fg_tile * blend_weights_3d
            weight_accum[y_start:y_end, x_start:x_end, :] += blend_weights_3d

            # Lightweight cleanup — delete refs but skip expensive gc/cache flush
            del alpha_tile, fg_tile, blend_weights, blend_weights_3d

    if tiles_skipped > 0:
        skip_pct = 100.0 * tiles_skipped / tiles_total
        logger.debug("Tiles skipped: %d/%d (%.0f%%)", tiles_skipped, tiles_total, skip_pct)

    # Normalize by accumulated weights
    weight_accum = np.maximum(weight_accum, 1e-8)
    alpha_final = mx.array(alpha_accum / weight_accum)[None]  # (1, H, W, 1)
    fg_final = mx.array(fg_accum / weight_accum)[None]  # (1, H, W, 3)

    return {"alpha_final": alpha_final, "fg_final": fg_final}
