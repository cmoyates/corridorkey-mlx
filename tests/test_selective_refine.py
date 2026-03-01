"""Tests for experimental selective refinement.

Phase 1: uncertainty mask, binary dilation, tile selection (synthetic data).
Phase 2: two-pass pipeline with real model (random weights, small resolution).
Phase 3: stitching, Gaussian blur, blended output.
Phase 5: narrow unit tests for mask/tile/blend edge cases.
"""

from __future__ import annotations

import logging

import mlx.core as mx
import numpy as np
import pytest

from corridorkey_mlx.inference.selective_refine import (
    SelectiveRefineConfig,
    SelectiveRefineResult,
    _binary_dilation,
    _build_input,
    _gaussian_blur,
    _gaussian_kernel_1d,
    _resize_to_square,
    compute_uncertainty_mask,
    select_tiles,
    selective_refine,
    stitch_results,
)
from corridorkey_mlx.model.corridorkey import GreenFormer

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


# ---------------------------------------------------------------------------
# Phase 2: Two-pass pipeline
# ---------------------------------------------------------------------------

SMALL_SIZE = 256


@pytest.fixture()
def model() -> GreenFormer:
    """Small model with random weights for pipeline tests."""
    m = GreenFormer(img_size=SMALL_SIZE)
    m.eval()
    # NOTE: mx.eval materializes lazy MLX arrays, not Python's eval()
    mx.eval(m.parameters())  # noqa: S307
    return m


class TestSelectiveRefinePipeline:
    def test_returns_correct_type(self, model: GreenFormer) -> None:
        """Pipeline returns SelectiveRefineResult."""
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
        hint = rng.integers(0, 256, (512, 512), dtype=np.uint8)
        config = SelectiveRefineConfig(
            coarse_size=SMALL_SIZE,
            tile_size=SMALL_SIZE,
            tile_overlap=32,
            dilation_radius=4,
        )

        result = selective_refine(model, image, hint, config)
        assert isinstance(result, SelectiveRefineResult)

    def test_output_shapes_match_input(self, model: GreenFormer) -> None:
        """Output arrays match input spatial dimensions."""
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, (400, 600, 3), dtype=np.uint8)
        hint = rng.integers(0, 256, (400, 600), dtype=np.uint8)
        config = SelectiveRefineConfig(
            coarse_size=SMALL_SIZE,
            tile_size=SMALL_SIZE,
            tile_overlap=32,
            dilation_radius=4,
        )

        result = selective_refine(model, image, hint, config)

        assert result.alpha_final.shape == (400, 600)
        assert result.fg_final.shape == (400, 600, 3)
        assert result.coarse_alpha.shape == (400, 600)
        assert result.uncertainty_mask.shape == (400, 600)

    def test_output_dtypes(self, model: GreenFormer) -> None:
        """Output arrays have expected dtypes."""
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
        hint = rng.integers(0, 256, (512, 512), dtype=np.uint8)
        config = SelectiveRefineConfig(
            coarse_size=SMALL_SIZE,
            tile_size=SMALL_SIZE,
            tile_overlap=32,
            dilation_radius=4,
        )

        result = selective_refine(model, image, hint, config)

        assert result.alpha_final.dtype == np.uint8
        assert result.fg_final.dtype == np.uint8
        assert result.coarse_alpha.dtype == np.uint8
        assert result.uncertainty_mask.dtype == bool

    def test_stats_populated(self, model: GreenFormer) -> None:
        """Stats dict contains expected timing keys."""
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
        hint = rng.integers(0, 256, (512, 512), dtype=np.uint8)
        config = SelectiveRefineConfig(
            coarse_size=SMALL_SIZE,
            tile_size=SMALL_SIZE,
            tile_overlap=32,
            dilation_radius=4,
        )

        result = selective_refine(model, image, hint, config)

        assert "coarse_ms" in result.stats
        assert "tile_refine_ms" in result.stats
        assert "stitch_ms" in result.stats
        assert "total_ms" in result.stats
        assert "tiles_selected" in result.stats
        assert "tiles_total" in result.stats
        assert "mask_coverage" in result.stats
        assert result.stats["coarse_ms"] > 0

    def test_zero_tiles_valid_result(self, model: GreenFormer) -> None:
        """Pipeline with no uncertain regions still returns valid result."""
        # All-white image + all-white hint
        image = np.full((SMALL_SIZE, SMALL_SIZE, 3), 255, dtype=np.uint8)
        hint = np.full((SMALL_SIZE, SMALL_SIZE), 255, dtype=np.uint8)
        config = SelectiveRefineConfig(
            coarse_size=SMALL_SIZE,
            tile_size=SMALL_SIZE,
            tile_overlap=32,
            dilation_radius=2,
        )

        result = selective_refine(model, image, hint, config)

        # With random weights we can't guarantee zero tiles, but result is valid
        assert result.alpha_final.shape == (SMALL_SIZE, SMALL_SIZE)
        assert result.stats["tiles_selected"] >= 0

    def test_hint_3d_accepted(self, model: GreenFormer) -> None:
        """Alpha hint with shape (H, W, 1) should work."""
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
        hint = rng.integers(0, 256, (512, 512, 1), dtype=np.uint8)
        config = SelectiveRefineConfig(
            coarse_size=SMALL_SIZE,
            tile_size=SMALL_SIZE,
            tile_overlap=32,
            dilation_radius=4,
        )

        result = selective_refine(model, image, hint, config)
        assert result.alpha_final.shape == (512, 512)

    def test_stitch_ms_in_stats(self, model: GreenFormer) -> None:
        """Stats should include stitch timing."""
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
        hint = rng.integers(0, 256, (512, 512), dtype=np.uint8)
        config = SelectiveRefineConfig(
            coarse_size=SMALL_SIZE,
            tile_size=SMALL_SIZE,
            tile_overlap=32,
            dilation_radius=4,
        )

        result = selective_refine(model, image, hint, config)
        assert "stitch_ms" in result.stats
        assert result.stats["stitch_ms"] >= 0


# ---------------------------------------------------------------------------
# Phase 3: Gaussian blur + stitching
# ---------------------------------------------------------------------------


class TestGaussianBlur:
    def test_zero_sigma_returns_copy(self) -> None:
        arr = np.ones((10, 10), dtype=np.float32) * 0.5
        result = _gaussian_blur(arr, sigma=0)
        np.testing.assert_array_equal(result, arr)
        assert result is not arr

    def test_uniform_unchanged(self) -> None:
        """Blurring a uniform array should return the same values."""
        arr = np.full((20, 20), 0.7, dtype=np.float32)
        result = _gaussian_blur(arr, sigma=3.0)
        np.testing.assert_allclose(result, 0.7, atol=1e-5)

    def test_preserves_shape(self) -> None:
        arr = np.random.default_rng(42).random((30, 40)).astype(np.float32)
        result = _gaussian_blur(arr, sigma=2.0)
        assert result.shape == (30, 40)

    def test_smooths_step_edge(self) -> None:
        """Step edge should be smoothed into a gradient."""
        arr = np.zeros((1, 100), dtype=np.float32)
        arr[:, 50:] = 1.0
        result = _gaussian_blur(arr, sigma=5.0)
        # At the edge, values should be intermediate (not 0 or 1)
        assert 0.1 < result[0, 50] < 0.9
        # Far from edge, values should be near original
        assert result[0, 0] < 0.05
        assert result[0, 99] > 0.95

    def test_output_range(self) -> None:
        """Output should stay in [0, 1] for [0, 1] input."""
        arr = np.random.default_rng(42).random((50, 50)).astype(np.float32)
        result = _gaussian_blur(arr, sigma=4.0)
        assert result.min() >= -1e-6
        assert result.max() <= 1.0 + 1e-6


class TestStitchResults:
    def test_no_tiles_returns_coarse(self) -> None:
        """Empty tile list → returns coarse unchanged."""
        coarse_alpha = np.full((100, 100), 0.5, dtype=np.float32)
        coarse_fg = np.full((100, 100, 3), 0.3, dtype=np.float32)
        config = SelectiveRefineConfig()

        alpha, fg = stitch_results(
            coarse_alpha, coarse_fg, [], [], np.zeros((100, 100), dtype=bool), config
        )
        np.testing.assert_array_equal(alpha, coarse_alpha)
        np.testing.assert_array_equal(fg, coarse_fg)

    def test_single_tile_full_coverage(self) -> None:
        """Single tile covering the whole image — result should be mostly refined."""
        size = 64
        coarse_alpha = np.full((size, size), 0.2, dtype=np.float32)
        coarse_fg = np.full((size, size, 3), 0.2, dtype=np.float32)
        tile_alpha = np.full((size, size), 0.8, dtype=np.float32)
        tile_fg = np.full((size, size, 3), 0.8, dtype=np.float32)
        mask = np.ones((size, size), dtype=bool)
        config = SelectiveRefineConfig(tile_size=size, tile_overlap=8, dilation_radius=4)
        tile_coords = [(0, 0, size, size)]
        tile_results = [{"alpha": tile_alpha, "fg": tile_fg}]

        alpha, fg = stitch_results(
            coarse_alpha, coarse_fg, tile_results, tile_coords, mask, config
        )

        # Center should be dominated by refined values
        center = size // 2
        assert alpha[center, center] > 0.6
        assert fg[center, center, 0] > 0.6

    def test_output_shapes(self) -> None:
        """Output shapes should match input."""
        h, w = 80, 120
        coarse_alpha = np.zeros((h, w), dtype=np.float32)
        coarse_fg = np.zeros((h, w, 3), dtype=np.float32)
        mask = np.ones((h, w), dtype=bool)
        config = SelectiveRefineConfig(tile_size=64, tile_overlap=8, dilation_radius=2)

        from corridorkey_mlx.inference.tiling import _compute_tile_coords

        y_coords = _compute_tile_coords(h, 64, 8)
        x_coords = _compute_tile_coords(w, 64, 8)
        tile_coords = [(ys, xs, ye, xe) for ys, ye in y_coords for xs, xe in x_coords]
        tile_results = [
            {
                "alpha": np.full((ye - ys, xe - xs), 0.5, dtype=np.float32),
                "fg": np.full((ye - ys, xe - xs, 3), 0.5, dtype=np.float32),
            }
            for ys, xs, ye, xe in tile_coords
        ]

        alpha, fg = stitch_results(
            coarse_alpha, coarse_fg, tile_results, tile_coords, mask, config
        )

        assert alpha.shape == (h, w)
        assert fg.shape == (h, w, 3)

    def test_blend_weight_smoothness(self) -> None:
        """Blend weight at mask boundary should be smooth, not binary."""
        size = 100
        mask = np.zeros((size, size), dtype=bool)
        mask[20:80, 20:80] = True

        # Test the Gaussian blur step directly
        blend = _gaussian_blur(mask.astype(np.float32), sigma=5.0)

        # At the mask edge, blend should be intermediate
        assert 0.1 < blend[20, 50] < 0.9
        # Inside mask, should be near 1
        assert blend[50, 50] > 0.9
        # Outside mask, should be near 0
        assert blend[5, 5] < 0.1


# ---------------------------------------------------------------------------
# Phase 5: Narrow unit tests
# ---------------------------------------------------------------------------


class TestGaussianKernel1D:
    def test_sums_to_one(self) -> None:
        k = _gaussian_kernel_1d(sigma=2.0)
        assert abs(k.sum() - 1.0) < 1e-6

    def test_symmetric(self) -> None:
        k = _gaussian_kernel_1d(sigma=3.0)
        np.testing.assert_allclose(k, k[::-1], atol=1e-7)

    def test_peak_at_center(self) -> None:
        k = _gaussian_kernel_1d(sigma=1.5)
        center = len(k) // 2
        assert k[center] == k.max()

    def test_length_odd(self) -> None:
        for sigma in [0.5, 1.0, 2.0, 5.0]:
            k = _gaussian_kernel_1d(sigma)
            assert len(k) % 2 == 1


class TestResizeToSquare:
    def test_2d_grayscale(self) -> None:
        arr = np.ones((64, 64), dtype=np.float32) * 0.5
        result = _resize_to_square(arr, 32)
        assert result.shape == (32, 32)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, 0.5, atol=0.02)

    def test_3d_rgb(self) -> None:
        arr = np.ones((64, 64, 3), dtype=np.float32) * 0.3
        result = _resize_to_square(arr, 16)
        assert result.shape == (16, 16, 3)
        assert result.dtype == np.float32

    def test_non_square_input(self) -> None:
        arr = np.zeros((100, 50), dtype=np.float32)
        result = _resize_to_square(arr, 32)
        assert result.shape == (32, 32)

    def test_values_clipped(self) -> None:
        """Values outside [0,1] are clipped before resize."""
        arr = np.full((32, 32), 1.5, dtype=np.float32)
        result = _resize_to_square(arr, 16)
        assert result.max() <= 1.0 + 1e-3


class TestBuildInput:
    def test_output_shape(self) -> None:
        rgb = np.ones((64, 64, 3), dtype=np.float32) * 0.5
        hint = np.ones((64, 64), dtype=np.float32) * 0.5
        result = _build_input(rgb, hint)
        assert result.shape == (1, 64, 64, 4)

    def test_hint_3d(self) -> None:
        rgb = np.ones((32, 32, 3), dtype=np.float32)
        hint = np.ones((32, 32, 1), dtype=np.float32)
        result = _build_input(rgb, hint)
        assert result.shape == (1, 32, 32, 4)

    def test_dtype_float32(self) -> None:
        rgb = np.zeros((16, 16, 3), dtype=np.float32)
        hint = np.zeros((16, 16), dtype=np.float32)
        result = _build_input(rgb, hint)
        assert result.dtype == mx.float32


class TestUncertaintyMaskEdgeCases:
    def test_gradient_only_detection(self) -> None:
        """Sharp edge at 0.0 → 0.03: outside transition band but gradient detected."""
        alpha = np.zeros((64, 64), dtype=np.float32)
        alpha[:, 32:] = 0.03  # Below uncertainty_low (0.05) — no band detection
        config = SelectiveRefineConfig(
            dilation_radius=0,
            gradient_threshold=0.01,
            uncertainty_low=0.05,
            uncertainty_high=0.95,
        )
        mask = compute_uncertainty_mask(alpha, config)
        # Column 32 edge should still be detected via gradient
        assert mask[:, 32].any()

    def test_non_square_input(self) -> None:
        alpha = np.random.default_rng(7).random((40, 80)).astype(np.float32)
        config = SelectiveRefineConfig(dilation_radius=1)
        mask = compute_uncertainty_mask(alpha, config)
        assert mask.shape == (40, 80)

    def test_warning_high_coverage(self, caplog: pytest.LogCaptureFixture) -> None:
        """Mask covering >90% should emit warning."""
        alpha = np.full((64, 64), 0.5, dtype=np.float32)
        config = SelectiveRefineConfig(dilation_radius=4)
        with caplog.at_level(logging.WARNING):
            compute_uncertainty_mask(alpha, config)
        assert any("speedup" in r.message for r in caplog.records)

    def test_warning_low_coverage(self, caplog: pytest.LogCaptureFixture) -> None:
        """Mask covering <1% should emit warning."""
        alpha = np.zeros((64, 64), dtype=np.float32)
        # One tiny gradient spot
        alpha[32, 32] = 0.5
        config = SelectiveRefineConfig(dilation_radius=0)
        with caplog.at_level(logging.WARNING):
            compute_uncertainty_mask(alpha, config)
        assert any("coarse may be sufficient" in r.message for r in caplog.records)


class TestSelectTilesEdgeCases:
    def test_non_square_image(self) -> None:
        mask = np.ones((300, 800), dtype=bool)
        config = SelectiveRefineConfig(tile_size=256, tile_overlap=32)
        tiles = select_tiles(mask, 300, 800, config)
        assert len(tiles) > 0
        for y0, x0, y1, x1 in tiles:
            assert 0 <= y0 < y1 <= 300
            assert 0 <= x0 < x1 <= 800

    def test_exact_tile_boundary(self) -> None:
        """Image exactly divisible by tile_size with no overlap."""
        mask = np.ones((512, 512), dtype=bool)
        config = SelectiveRefineConfig(tile_size=256, tile_overlap=0)
        tiles = select_tiles(mask, 512, 512, config)
        assert len(tiles) == 4  # 2x2 grid

    def test_coverage_just_above_threshold(self) -> None:
        """Tile with coverage barely above threshold is kept."""
        tile_area = 512 * 512
        threshold = 0.05
        needed = int(tile_area * threshold) + 1
        mask = np.zeros((512, 512), dtype=bool)
        mask.ravel()[:needed] = True
        config = SelectiveRefineConfig(tile_size=512, tile_overlap=0, min_tile_coverage=threshold)
        tiles = select_tiles(mask, 512, 512, config)
        assert len(tiles) == 1


class TestStitchEdgeCases:
    def test_tile_resized_to_actual_crop(self) -> None:
        """Tile output at tile_size is resized to actual crop dimensions."""
        h, w = 200, 300
        coarse_alpha = np.full((h, w), 0.3, dtype=np.float32)
        coarse_fg = np.full((h, w, 3), 0.3, dtype=np.float32)
        mask = np.ones((h, w), dtype=bool)
        config = SelectiveRefineConfig(tile_size=512, tile_overlap=0, dilation_radius=2)

        # Tile coords smaller than tile_size
        tile_coords = [(0, 0, h, w)]
        tile_results = [
            {
                "alpha": np.full((512, 512), 0.9, dtype=np.float32),
                "fg": np.full((512, 512, 3), 0.9, dtype=np.float32),
            }
        ]

        alpha, fg = stitch_results(
            coarse_alpha, coarse_fg, tile_results, tile_coords, mask, config
        )
        assert alpha.shape == (h, w)
        # Center should be close to refined value (0.9)
        assert alpha[h // 2, w // 2] > 0.6

    def test_values_in_valid_range(self) -> None:
        """Output should be clipped to [0, 1] range."""
        size = 64
        coarse_alpha = np.ones((size, size), dtype=np.float32)
        coarse_fg = np.ones((size, size, 3), dtype=np.float32)
        tile_alpha = np.ones((size, size), dtype=np.float32)
        tile_fg = np.ones((size, size, 3), dtype=np.float32)
        mask = np.ones((size, size), dtype=bool)
        config = SelectiveRefineConfig(tile_size=size, tile_overlap=0, dilation_radius=2)

        alpha, fg = stitch_results(
            coarse_alpha,
            coarse_fg,
            [{"alpha": tile_alpha, "fg": tile_fg}],
            [(0, 0, size, size)],
            mask,
            config,
        )
        assert alpha.min() >= -1e-6
        assert alpha.max() <= 1.0 + 1e-6
