"""Tests for experimental selective refinement — Phase 1.

Covers uncertainty mask generation, binary dilation, and tile selection.
All tests use deterministic synthetic data and require no checkpoint.
"""

from __future__ import annotations

import numpy as np
import pytest

from corridorkey_mlx.inference.selective_refine import (
    SelectiveRefineConfig,
    _binary_dilation,
    compute_uncertainty_mask,
    select_tiles,
)

# ---------------------------------------------------------------------------
# Binary dilation
# ---------------------------------------------------------------------------


class TestBinaryDilation:
    def test_zero_radius_returns_copy(self) -> None:
        mask = np.array([[False, True, False], [False, False, False]])
        result = _binary_dilation(mask, radius=0)
        np.testing.assert_array_equal(result, mask)
        # Must be a copy, not the same object
        assert result is not mask

    def test_single_pixel_radius_one(self) -> None:
        """Single True pixel dilated by 1 should expand to 3x3 block."""
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 2] = True
        result = _binary_dilation(mask, radius=1)
        # Center 3x3 should be True
        assert result[1:4, 1:4].all()
        # Corners should remain False
        assert not result[0, 0]
        assert not result[4, 4]

    def test_radius_two_expands_further(self) -> None:
        mask = np.zeros((7, 7), dtype=bool)
        mask[3, 3] = True
        result = _binary_dilation(mask, radius=2)
        # Center pixel + 2 iterations should cover 5x5 block
        assert result[1:6, 1:6].all()

    def test_full_mask_stays_full(self) -> None:
        mask = np.ones((10, 10), dtype=bool)
        result = _binary_dilation(mask, radius=5)
        assert result.all()

    def test_empty_mask_stays_empty(self) -> None:
        mask = np.zeros((10, 10), dtype=bool)
        result = _binary_dilation(mask, radius=5)
        assert not result.any()

    def test_preserves_shape(self) -> None:
        mask = np.zeros((13, 17), dtype=bool)
        mask[6, 8] = True
        result = _binary_dilation(mask, radius=3)
        assert result.shape == (13, 17)


# ---------------------------------------------------------------------------
# Uncertainty mask
# ---------------------------------------------------------------------------


class TestComputeUncertaintyMask:
    def test_solid_foreground_no_uncertainty(self) -> None:
        """All-white alpha → no uncertain pixels."""
        alpha = np.ones((64, 64), dtype=np.float32)
        config = SelectiveRefineConfig(dilation_radius=0)
        mask = compute_uncertainty_mask(alpha, config)
        assert not mask.any()

    def test_solid_background_no_uncertainty(self) -> None:
        """All-black alpha → no uncertain pixels."""
        alpha = np.zeros((64, 64), dtype=np.float32)
        config = SelectiveRefineConfig(dilation_radius=0)
        mask = compute_uncertainty_mask(alpha, config)
        assert not mask.any()

    def test_transition_band_detected(self) -> None:
        """Pixels in [low, high] should be flagged as uncertain."""
        alpha = np.full((64, 64), 0.5, dtype=np.float32)
        config = SelectiveRefineConfig(dilation_radius=0)
        mask = compute_uncertainty_mask(alpha, config)
        assert mask.all()

    def test_sharp_edge_detected(self) -> None:
        """Sharp 0→1 edge should trigger gradient detection."""
        alpha = np.zeros((64, 64), dtype=np.float32)
        alpha[:, 32:] = 1.0
        config = SelectiveRefineConfig(dilation_radius=0, gradient_threshold=0.02)
        mask = compute_uncertainty_mask(alpha, config)
        # Column 32 should be flagged (sharp edge)
        assert mask[:, 32].all()
        # Far-away columns should not
        assert not mask[:, 0].any()
        assert not mask[:, 63].any()

    def test_dilation_expands_mask(self) -> None:
        """Dilated mask should be larger than undilated."""
        alpha = np.zeros((64, 64), dtype=np.float32)
        alpha[:, 32:] = 1.0

        config_no_dil = SelectiveRefineConfig(dilation_radius=0)
        config_dil = SelectiveRefineConfig(dilation_radius=4)

        mask_small = compute_uncertainty_mask(alpha, config_no_dil)
        mask_large = compute_uncertainty_mask(alpha, config_dil)

        assert np.sum(mask_large) > np.sum(mask_small)

    def test_output_shape_matches_input(self) -> None:
        alpha = np.random.default_rng(42).random((100, 80)).astype(np.float32)
        config = SelectiveRefineConfig(dilation_radius=2)
        mask = compute_uncertainty_mask(alpha, config)
        assert mask.shape == (100, 80)
        assert mask.dtype == bool


# ---------------------------------------------------------------------------
# Tile selection
# ---------------------------------------------------------------------------


class TestSelectTiles:
    def test_no_uncertain_pixels_returns_empty(self) -> None:
        """Empty mask → no tiles selected."""
        mask = np.zeros((512, 512), dtype=bool)
        config = SelectiveRefineConfig()
        tiles = select_tiles(mask, 512, 512, config)
        assert tiles == []

    def test_full_mask_selects_all_tiles(self) -> None:
        """Full mask → every tile selected."""
        mask = np.ones((1024, 1024), dtype=bool)
        config = SelectiveRefineConfig(tile_size=512, tile_overlap=64)
        tiles = select_tiles(mask, 1024, 1024, config)
        # Should select all tiles in the grid
        assert len(tiles) > 0
        # All tiles should be present
        from corridorkey_mlx.inference.tiling import _compute_tile_coords

        y_coords = _compute_tile_coords(1024, 512, 64)
        x_coords = _compute_tile_coords(1024, 512, 64)
        assert len(tiles) == len(y_coords) * len(x_coords)

    def test_partial_mask_filters_tiles(self) -> None:
        """Only tiles overlapping the mask should be selected."""
        mask = np.zeros((1024, 1024), dtype=bool)
        # Mark a region in the top-left (>5% of tile area = 512*512*0.05 ≈ 13107 px)
        mask[:200, :200] = True
        config = SelectiveRefineConfig(tile_size=512, tile_overlap=64)
        tiles = select_tiles(mask, 1024, 1024, config)
        # Should select at least the top-left tile
        assert len(tiles) >= 1
        # The first tile should cover (0, 0)
        assert any(y0 == 0 and x0 == 0 for y0, x0, _, _ in tiles)

    def test_min_coverage_filters_low_overlap(self) -> None:
        """Tiles with coverage below threshold should be excluded."""
        mask = np.zeros((512, 512), dtype=bool)
        # Set just 1% of pixels — below default 5% threshold
        mask[:5, :5] = True  # 25 pixels out of 262144 ~ 0.01%
        config = SelectiveRefineConfig(tile_size=512, tile_overlap=64, min_tile_coverage=0.05)
        tiles = select_tiles(mask, 512, 512, config)
        assert tiles == []

    def test_tile_coords_are_valid(self) -> None:
        """All returned tile coords should be within image bounds."""
        mask = np.ones((800, 600), dtype=bool)
        config = SelectiveRefineConfig(tile_size=512, tile_overlap=64)
        tiles = select_tiles(mask, 800, 600, config)
        for y0, x0, y1, x1 in tiles:
            assert 0 <= y0 < y1 <= 800
            assert 0 <= x0 < x1 <= 600

    def test_single_tile_image(self) -> None:
        """Image smaller than tile_size should produce at most 1 tile."""
        mask = np.ones((256, 256), dtype=bool)
        config = SelectiveRefineConfig(tile_size=512, tile_overlap=64)
        tiles = select_tiles(mask, 256, 256, config)
        assert len(tiles) == 1
        assert tiles[0] == (0, 0, 256, 256)


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------


class TestSelectiveRefineConfig:
    def test_default_values(self) -> None:
        config = SelectiveRefineConfig()
        assert config.coarse_size == 512
        assert config.tile_size == 512
        assert config.tile_overlap == 64
        assert config.uncertainty_low == pytest.approx(0.05)
        assert config.uncertainty_high == pytest.approx(0.95)
        assert config.gradient_threshold == pytest.approx(0.02)
        assert config.dilation_radius == 32
        assert config.min_tile_coverage == pytest.approx(0.05)
        assert config.compile is True
