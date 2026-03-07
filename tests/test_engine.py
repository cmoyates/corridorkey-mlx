"""Tests for CorridorKeyMLXEngine contract."""

from __future__ import annotations

import numpy as np
import pytest

from corridorkey_mlx.engine import (
    CorridorKeyMLXEngine,
    _validate_engine_params,
    _validate_image,
    _validate_mask,
)

from .conftest import MLX_CHECKPOINT_PATH, has_checkpoint

# ---------------------------------------------------------------------------
# Input validation (no checkpoint needed)
# ---------------------------------------------------------------------------


class TestValidateImage:
    def test_rejects_non_ndarray(self) -> None:
        with pytest.raises(TypeError, match="numpy ndarray"):
            _validate_image("not_an_array")  # type: ignore[arg-type]

    def test_rejects_wrong_dtype(self) -> None:
        img = np.zeros((64, 64, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="uint8"):
            _validate_image(img)

    def test_rejects_wrong_shape(self) -> None:
        img = np.zeros((64, 64), dtype=np.uint8)
        with pytest.raises(ValueError, match="\\(H, W, 3\\)"):
            _validate_image(img)

    def test_rejects_wrong_channels(self) -> None:
        img = np.zeros((64, 64, 4), dtype=np.uint8)
        with pytest.raises(ValueError, match="\\(H, W, 3\\)"):
            _validate_image(img)

    def test_accepts_valid_image(self) -> None:
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        _validate_image(img)  # should not raise


class TestValidateMask:
    def test_rejects_non_ndarray(self) -> None:
        with pytest.raises(TypeError, match="numpy ndarray"):
            _validate_mask("not_an_array")  # type: ignore[arg-type]

    def test_rejects_wrong_dtype(self) -> None:
        mask = np.zeros((64, 64), dtype=np.float32)
        with pytest.raises(ValueError, match="uint8"):
            _validate_mask(mask)

    def test_accepts_hw(self) -> None:
        mask = np.zeros((64, 64), dtype=np.uint8)
        _validate_mask(mask)  # should not raise

    def test_accepts_hw1(self) -> None:
        mask = np.zeros((64, 64, 1), dtype=np.uint8)
        _validate_mask(mask)  # should not raise

    def test_rejects_hw3(self) -> None:
        mask = np.zeros((64, 64, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="\\(H, W\\) or \\(H, W, 1\\)"):
            _validate_mask(mask)


# ---------------------------------------------------------------------------
# Engine init
# ---------------------------------------------------------------------------


class TestEngineInit:
    def test_missing_checkpoint_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            CorridorKeyMLXEngine(checkpoint_path="/nonexistent/weights.safetensors")


# ---------------------------------------------------------------------------
# Engine integration (requires checkpoint)
# ---------------------------------------------------------------------------


@has_checkpoint
class TestEngineIntegration:
    """Integration tests that load the real model."""

    @pytest.fixture(scope="class")
    def engine(self) -> CorridorKeyMLXEngine:
        return CorridorKeyMLXEngine(
            checkpoint_path=MLX_CHECKPOINT_PATH,
            img_size=512,
            compile=False,
        )

    def test_output_keys(self, engine: CorridorKeyMLXEngine) -> None:
        image = np.random.default_rng(42).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        mask = np.random.default_rng(42).integers(0, 256, (64, 64), dtype=np.uint8)
        result = engine.process_frame(image, mask)
        assert set(result.keys()) == {"alpha", "fg", "comp", "processed"}

    def test_output_shapes(self, engine: CorridorKeyMLXEngine) -> None:
        h, w = 100, 150
        image = np.random.default_rng(42).integers(0, 256, (h, w, 3), dtype=np.uint8)
        mask = np.random.default_rng(42).integers(0, 256, (h, w), dtype=np.uint8)
        result = engine.process_frame(image, mask)
        assert result["alpha"].shape == (h, w)
        assert result["fg"].shape == (h, w, 3)
        assert result["comp"].shape == (h, w, 3)
        assert result["processed"].shape == (h, w, 3)

    def test_output_dtypes(self, engine: CorridorKeyMLXEngine) -> None:
        image = np.random.default_rng(42).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        mask = np.random.default_rng(42).integers(0, 256, (64, 64), dtype=np.uint8)
        result = engine.process_frame(image, mask)
        for key, arr in result.items():
            assert arr.dtype == np.uint8, f"{key} dtype is {arr.dtype}"

    def test_mask_hw1_accepted(self, engine: CorridorKeyMLXEngine) -> None:
        image = np.random.default_rng(42).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        mask = np.random.default_rng(42).integers(0, 256, (64, 64, 1), dtype=np.uint8)
        result = engine.process_frame(image, mask)
        assert "alpha" in result

    def test_refiner_scale_zero_returns_coarse(self, engine: CorridorKeyMLXEngine) -> None:
        image = np.random.default_rng(42).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        mask = np.random.default_rng(42).integers(0, 256, (64, 64), dtype=np.uint8)
        result = engine.process_frame(image, mask, refiner_scale=0.0)
        assert result["alpha"].shape == (64, 64)


# ---------------------------------------------------------------------------
# Full 2048 inference (slow, requires checkpoint)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@has_checkpoint
def test_smoke_2048_full() -> None:
    """Full 2048 inference with real checkpoint."""
    engine = CorridorKeyMLXEngine(
        checkpoint_path=MLX_CHECKPOINT_PATH,
        img_size=2048,
        compile=False,
    )

    rng = np.random.default_rng(42)
    rgb = rng.integers(0, 256, (2048, 2048, 3), dtype=np.uint8)
    mask = rng.integers(0, 256, (2048, 2048), dtype=np.uint8)

    result = engine.process_frame(rgb, mask)

    assert result["alpha"].shape == (2048, 2048)
    assert result["alpha"].dtype == np.uint8
    assert result["fg"].shape == (2048, 2048, 3)
    assert result["comp"].shape == (2048, 2048, 3)

    for key in ("alpha", "fg", "comp"):
        arr = result[key]
        assert not np.isnan(arr).any(), f"{key} contains NaN"
        assert not np.isinf(arr).any(), f"{key} contains Inf"

    assert result["alpha"].min() != result["alpha"].max(), "alpha is constant"


# ---------------------------------------------------------------------------
# Parameter validation (no checkpoint needed)
# ---------------------------------------------------------------------------


class TestValidateEngineParams:
    def test_backbone_size_not_divisible_by_4(self) -> None:
        with pytest.raises(ValueError, match="divisible by 4"):
            _validate_engine_params(512, backbone_size=511, tile_size=None, tile_overlap=96)

    def test_backbone_size_exceeds_img_size(self) -> None:
        with pytest.raises(ValueError, match="<= img_size"):
            _validate_engine_params(512, backbone_size=1024, tile_size=None, tile_overlap=96)

    def test_tile_size_not_divisible_by_4(self) -> None:
        with pytest.raises(ValueError, match="divisible by 4"):
            _validate_engine_params(512, backbone_size=None, tile_size=511, tile_overlap=96)

    def test_tile_overlap_exceeds_tile_size(self) -> None:
        with pytest.raises(ValueError, match="< tile_size"):
            _validate_engine_params(512, backbone_size=None, tile_size=256, tile_overlap=256)

    def test_valid_params_accepted(self) -> None:
        _validate_engine_params(2048, backbone_size=1024, tile_size=512, tile_overlap=96)

    def test_none_params_accepted(self) -> None:
        _validate_engine_params(512, backbone_size=None, tile_size=None, tile_overlap=96)


# ---------------------------------------------------------------------------
# Engine new params (requires checkpoint)
# ---------------------------------------------------------------------------


@has_checkpoint
class TestEngineFp16:
    @pytest.fixture(scope="class")
    def engine_fp16(self) -> CorridorKeyMLXEngine:
        return CorridorKeyMLXEngine(
            checkpoint_path=MLX_CHECKPOINT_PATH,
            img_size=512,
            compile=False,
            fp16=True,
        )

    @pytest.fixture(scope="class")
    def engine_fp32(self) -> CorridorKeyMLXEngine:
        return CorridorKeyMLXEngine(
            checkpoint_path=MLX_CHECKPOINT_PATH,
            img_size=512,
            compile=False,
            fp16=False,
        )

    def test_fp16_produces_valid_output(self, engine_fp16: CorridorKeyMLXEngine) -> None:
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        mask = rng.integers(0, 256, (64, 64), dtype=np.uint8)
        result = engine_fp16.process_frame(image, mask)
        assert result["alpha"].shape == (64, 64)
        assert result["alpha"].dtype == np.uint8
        assert not np.isnan(result["alpha"]).any()

    def test_fp32_backward_compat(self, engine_fp32: CorridorKeyMLXEngine) -> None:
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        mask = rng.integers(0, 256, (64, 64), dtype=np.uint8)
        result = engine_fp32.process_frame(image, mask)
        assert set(result.keys()) == {"alpha", "fg", "comp", "processed"}


@has_checkpoint
class TestEngineBackboneSize:
    def test_backbone_size_produces_valid_output(self) -> None:
        engine = CorridorKeyMLXEngine(
            checkpoint_path=MLX_CHECKPOINT_PATH,
            img_size=512,
            compile=False,
            fp16=False,
            backbone_size=256,
        )
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        mask = rng.integers(0, 256, (64, 64), dtype=np.uint8)
        result = engine.process_frame(image, mask)
        assert result["alpha"].shape == (64, 64)
        assert not np.isnan(result["alpha"]).any()


@has_checkpoint
class TestEngineTiling:
    @pytest.fixture(scope="class")
    def engine_tiled(self) -> CorridorKeyMLXEngine:
        return CorridorKeyMLXEngine(
            checkpoint_path=MLX_CHECKPOINT_PATH,
            img_size=512,
            compile=False,
            fp16=False,
            tile_size=256,
            tile_overlap=64,
        )

    def test_tiled_produces_valid_output(self, engine_tiled: CorridorKeyMLXEngine) -> None:
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
        mask = rng.integers(0, 256, (512, 512), dtype=np.uint8)
        result = engine_tiled.process_frame(image, mask)
        assert result["alpha"].shape == (512, 512)
        assert result["fg"].shape == (512, 512, 3)
        assert result["alpha"].dtype == np.uint8
        assert not np.isnan(result["alpha"]).any()

    def test_tiled_output_keys(self, engine_tiled: CorridorKeyMLXEngine) -> None:
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
        mask = rng.integers(0, 256, (512, 512), dtype=np.uint8)
        result = engine_tiled.process_frame(image, mask)
        assert set(result.keys()) == {"alpha", "fg", "comp", "processed"}

    def test_tiled_small_image_no_tiling(self, engine_tiled: CorridorKeyMLXEngine) -> None:
        """Image smaller than tile_size should still work (single tile)."""
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        mask = rng.integers(0, 256, (64, 64), dtype=np.uint8)
        result = engine_tiled.process_frame(image, mask)
        assert result["alpha"].shape == (64, 64)
