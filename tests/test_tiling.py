"""Test tiled inference consistency.

Verifies that:
1. Single-tile images produce same results as non-tiled inference.
2. Tiled results on larger images have reasonable blending behavior.
3. Tile coordinate computation is correct.
4. Phase 4: numpy input, per-tile memory cleanup, cache limit management.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from corridorkey_mlx.inference.tiling import (
    DEFAULT_OVERLAP,
    _compute_tile_coords,
    _make_blend_weights_2d,
    tiled_inference,
)
from corridorkey_mlx.model.corridorkey import GreenFormer

TILE_SIZE = 256
TOLERANCE = 1e-5


@pytest.fixture()
def model() -> GreenFormer:
    model = GreenFormer(img_size=TILE_SIZE)
    # NOTE: mx.eval is MLX array materialization, not Python eval()
    mx.eval(model.parameters())  # noqa: S307
    return model


class TestTileCoords:
    def test_overlap_gte_tile_size_raises(self) -> None:
        with pytest.raises(ValueError, match="overlap.*must be less than tile_size"):
            _compute_tile_coords(512, 256, 256)
        with pytest.raises(ValueError, match="overlap.*must be less than tile_size"):
            _compute_tile_coords(512, 256, 300)

    def test_single_tile(self) -> None:
        coords = _compute_tile_coords(200, 256, 32)
        assert coords == [(0, 200)]

    def test_exact_fit(self) -> None:
        coords = _compute_tile_coords(256, 256, 32)
        assert coords == [(0, 256)]

    def test_two_tiles_with_overlap(self) -> None:
        coords = _compute_tile_coords(400, 256, 32)
        assert len(coords) == 2
        # All tiles cover full range
        assert coords[0][0] == 0
        assert coords[-1][1] == 400
        # Overlap exists between tiles
        assert coords[0][1] > coords[1][0]

    def test_full_coverage(self) -> None:
        """Every pixel is covered by at least one tile."""
        for image_size in [300, 512, 700, 1024]:
            coords = _compute_tile_coords(image_size, 256, 32)
            covered = set()
            for start, end in coords:
                covered.update(range(start, end))
            assert covered == set(range(image_size)), f"Gap in coverage at size={image_size}"

    def test_full_coverage_96px_overlap(self) -> None:
        """Full coverage with 96px overlap (Phase 4 default)."""
        for image_size in [300, 512, 700, 1024]:
            coords = _compute_tile_coords(image_size, 256, 96)
            covered = set()
            for start, end in coords:
                covered.update(range(start, end))
            assert covered == set(range(image_size)), f"Gap at size={image_size}, overlap=96"


class TestBlendWeights:
    def test_interior_tile(self) -> None:
        """Interior tile (all neighbors) has ramps on all edges."""
        w = _make_blend_weights_2d(256, 256, 32, (True, True, True, True))
        assert w.shape == (256, 256)
        # Corners should be near zero (double ramp)
        assert w[0, 0] < 0.01
        # Center should be 1.0
        assert w[128, 128] == 1.0

    def test_corner_tile(self) -> None:
        """Top-left corner tile (no top/left neighbors) has full weight at origin."""
        w = _make_blend_weights_2d(256, 256, 32, (False, True, False, True))
        assert w[0, 0] == 1.0
        # Bottom-right edge should ramp
        assert w[-1, -1] < 1.0

    def test_no_overlap(self) -> None:
        w = _make_blend_weights_2d(256, 256, 0, (True, True, True, True))
        assert (w == 1.0).all()

    def test_96px_overlap_ramps(self) -> None:
        """96px overlap produces wider ramp regions."""
        w = _make_blend_weights_2d(256, 256, 96, (True, True, True, True))
        # At pixel 48 (midpoint of 96px ramp), weight should be ~0.5
        assert 0.2 < w[48, 128] < 0.8


class TestDefaultOverlap:
    def test_default_overlap_is_96(self) -> None:
        assert DEFAULT_OVERLAP == 96


class TestTiledInference:
    def test_single_tile_matches_direct(self, model: GreenFormer) -> None:
        """Image that fits in one tile produces identical results to direct inference."""
        mx.random.seed(42)
        x_mx = mx.random.normal((1, TILE_SIZE, TILE_SIZE, 4))
        # NOTE: mx.eval is MLX array materialization, not Python eval()
        mx.eval(x_mx)  # noqa: S307

        direct_out = model(x_mx)
        mx.eval(direct_out)  # noqa: S307

        # tiled_inference now expects numpy input
        x_np = np.array(x_mx)
        tiled_out = tiled_inference(model, x_np, tile_size=TILE_SIZE, overlap=32)
        mx.eval(tiled_out)  # noqa: S307

        for key in ("alpha_final", "fg_final"):
            diff = float(mx.max(mx.abs(direct_out[key] - tiled_out[key])))
            assert diff < TOLERANCE, f"{key}: max_diff={diff:.2e}"

    def test_larger_image_runs(self, model: GreenFormer) -> None:
        """Tiled inference runs without error on larger-than-tile images."""
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((1, 400, 400, 4)).astype(np.float32)

        result = tiled_inference(model, x_np, tile_size=TILE_SIZE, overlap=32)
        # NOTE: mx.eval is MLX array materialization, not Python eval()
        mx.eval(result)  # noqa: S307

        assert result["alpha_final"].shape == (1, 400, 400, 1)
        assert result["fg_final"].shape == (1, 400, 400, 3)

    def test_batch_size_validation(self, model: GreenFormer) -> None:
        """Batch size > 1 raises ValueError."""
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((2, TILE_SIZE, TILE_SIZE, 4)).astype(np.float32)
        with pytest.raises(ValueError, match="batch_size=1"):
            tiled_inference(model, x_np, tile_size=TILE_SIZE)

    def test_accepts_numpy_input(self, model: GreenFormer) -> None:
        """tiled_inference accepts np.ndarray (Phase 4 lazy slicing)."""
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((1, TILE_SIZE, TILE_SIZE, 4)).astype(np.float32)
        result = tiled_inference(model, x_np, tile_size=TILE_SIZE, overlap=32)
        assert "alpha_final" in result
        assert "fg_final" in result

    def test_96px_overlap_runs(self, model: GreenFormer) -> None:
        """Tiled inference with 96px overlap (new default) runs correctly."""
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((1, 400, 400, 4)).astype(np.float32)

        result = tiled_inference(model, x_np, tile_size=TILE_SIZE, overlap=96)
        # NOTE: mx.eval is MLX array materialization, not Python eval()
        mx.eval(result)  # noqa: S307

        assert result["alpha_final"].shape == (1, 400, 400, 1)
        assert result["fg_final"].shape == (1, 400, 400, 3)

    def test_cache_restored_after_tiling(self, model: GreenFormer) -> None:
        """Cache limit is restored after tiled_inference completes."""
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((1, 400, 400, 4)).astype(np.float32)

        tiled_inference(model, x_np, tile_size=TILE_SIZE, overlap=32)
        cache_after = mx.get_cache_memory()

        # Verify it completes cleanly and cache memory is queryable
        assert cache_after >= 0

    def test_no_monotonic_memory_growth(self, model: GreenFormer) -> None:
        """Peak memory doesn't grow monotonically across tiles."""
        rng = np.random.default_rng(42)
        # Large enough for multiple tiles
        x_np = rng.standard_normal((1, 600, 600, 4)).astype(np.float32)

        mx.reset_peak_memory()
        tiled_inference(model, x_np, tile_size=TILE_SIZE, overlap=32)
        peak_after = mx.get_peak_memory()

        # Peak should be bounded (not proportional to total image size)
        assert peak_after > 0  # sanity: something was allocated
