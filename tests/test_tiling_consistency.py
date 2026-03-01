"""Test tiled inference consistency.

Verifies that:
1. Single-tile images produce same results as non-tiled inference.
2. Tiled results on larger images have reasonable blending behavior.
3. Tile coordinate computation is correct.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from corridorkey_mlx.inference.tiling import (
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
    model.eval()
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


class TestTiledInference:
    def test_single_tile_matches_direct(self, model: GreenFormer) -> None:
        """Image that fits in one tile produces identical results to direct inference."""
        mx.random.seed(42)
        x = mx.random.normal((1, TILE_SIZE, TILE_SIZE, 4))
        mx.eval(x)  # noqa: S307

        direct_out = model(x)
        mx.eval(direct_out)  # noqa: S307

        tiled_out = tiled_inference(model, x, tile_size=TILE_SIZE, overlap=32)
        mx.eval(tiled_out)  # noqa: S307

        for key in ("alpha_final", "fg_final"):
            diff = float(mx.max(mx.abs(direct_out[key] - tiled_out[key])))
            assert diff < TOLERANCE, f"{key}: max_diff={diff:.2e}"

    def test_larger_image_runs(self, model: GreenFormer) -> None:
        """Tiled inference runs without error on larger-than-tile images."""
        mx.random.seed(42)
        x = mx.random.normal((1, 400, 400, 4))
        mx.eval(x)  # noqa: S307

        result = tiled_inference(model, x, tile_size=TILE_SIZE, overlap=32)
        mx.eval(result)  # noqa: S307

        assert result["alpha_final"].shape == (1, 400, 400, 1)
        assert result["fg_final"].shape == (1, 400, 400, 3)

    def test_batch_size_validation(self, model: GreenFormer) -> None:
        """Batch size > 1 raises ValueError."""
        mx.random.seed(42)
        x = mx.random.normal((2, TILE_SIZE, TILE_SIZE, 4))
        with pytest.raises(ValueError, match="batch_size=1"):
            tiled_inference(model, x, tile_size=TILE_SIZE)
