"""Regression tests for decoupled backbone/refiner resolutions (Phase 3).

Not parity tests vs golden — validates outputs are structurally correct
and that backbone_size=None preserves exact existing behavior.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from corridorkey_mlx.model.corridorkey import GreenFormer

from .conftest import IMG_SIZE, MLX_CHECKPOINT_PATH, has_checkpoint

BACKBONE_SIZE = IMG_SIZE // 2  # 256 for IMG_SIZE=512


# ---------------------------------------------------------------------------
# Structural tests (random weights, no checkpoint needed)
# ---------------------------------------------------------------------------


class TestDecoupledStructure:
    """Verify shapes, dtypes, and flag behavior with random weights."""

    @pytest.fixture(scope="class")
    def decoupled_model(self) -> GreenFormer:
        model = GreenFormer(img_size=IMG_SIZE, backbone_size=BACKBONE_SIZE)
        # mx.eval is MLX array materialization, not Python eval
        mx.eval(model.parameters())  # noqa: S307
        return model

    @pytest.fixture(scope="class")
    def coupled_model(self) -> GreenFormer:
        model = GreenFormer(img_size=IMG_SIZE)
        # mx.eval is MLX array materialization, not Python eval
        mx.eval(model.parameters())  # noqa: S307
        return model

    def test_decoupled_flag_set(self, decoupled_model: GreenFormer) -> None:
        assert decoupled_model._decoupled is True

    def test_coupled_flag_unset(self, coupled_model: GreenFormer) -> None:
        assert coupled_model._decoupled is False

    def test_backbone_size_none_not_decoupled(self) -> None:
        model = GreenFormer(img_size=IMG_SIZE, backbone_size=None)
        assert model._decoupled is False

    def test_backbone_size_equals_img_size_not_decoupled(self) -> None:
        model = GreenFormer(img_size=IMG_SIZE, backbone_size=IMG_SIZE)
        assert model._decoupled is False

    def test_output_shapes_match_full_res(
        self, decoupled_model: GreenFormer
    ) -> None:
        """Decoupled outputs must have same spatial dims as full-res input."""
        rng = np.random.default_rng(42)
        x = mx.array(rng.standard_normal((1, IMG_SIZE, IMG_SIZE, 4)).astype(np.float32))
        outputs = decoupled_model(x)
        # mx.eval is MLX array materialization, not Python eval
        mx.eval(outputs)  # noqa: S307

        assert outputs["alpha_coarse"].shape == (1, IMG_SIZE, IMG_SIZE, 1)
        assert outputs["fg_coarse"].shape == (1, IMG_SIZE, IMG_SIZE, 3)
        assert outputs["alpha_final"].shape == (1, IMG_SIZE, IMG_SIZE, 1)
        assert outputs["fg_final"].shape == (1, IMG_SIZE, IMG_SIZE, 3)

    def test_output_shapes_slim(self, decoupled_model: GreenFormer) -> None:
        rng = np.random.default_rng(42)
        x = mx.array(rng.standard_normal((1, IMG_SIZE, IMG_SIZE, 4)).astype(np.float32))
        outputs = decoupled_model(x, slim=True)
        # mx.eval is MLX array materialization, not Python eval
        mx.eval(outputs)  # noqa: S307

        assert set(outputs.keys()) == {
            "alpha_coarse", "fg_coarse", "alpha_final", "fg_final"
        }
        assert outputs["alpha_final"].shape == (1, IMG_SIZE, IMG_SIZE, 1)

    def test_no_nan_inf(self, decoupled_model: GreenFormer) -> None:
        rng = np.random.default_rng(42)
        x = mx.array(rng.standard_normal((1, IMG_SIZE, IMG_SIZE, 4)).astype(np.float32))
        outputs = decoupled_model(x, slim=True)
        # mx.eval is MLX array materialization, not Python eval
        mx.eval(outputs)  # noqa: S307

        for key, val in outputs.items():
            arr = np.array(val)
            assert not np.isnan(arr).any(), f"{key} contains NaN"
            assert not np.isinf(arr).any(), f"{key} contains Inf"

    def test_alpha_in_valid_range(self, decoupled_model: GreenFormer) -> None:
        """Alpha outputs should be in [0, 1] (sigmoid output)."""
        rng = np.random.default_rng(42)
        x = mx.array(rng.standard_normal((1, IMG_SIZE, IMG_SIZE, 4)).astype(np.float32))
        outputs = decoupled_model(x, slim=True)
        # mx.eval is MLX array materialization, not Python eval
        mx.eval(outputs)  # noqa: S307

        for key in ("alpha_coarse", "alpha_final"):
            arr = np.array(outputs[key])
            assert arr.min() >= 0.0, f"{key} min={arr.min()}"
            assert arr.max() <= 1.0, f"{key} max={arr.max()}"

    def test_full_output_keys(self, decoupled_model: GreenFormer) -> None:
        """Non-slim output should have all 9 keys."""
        rng = np.random.default_rng(42)
        x = mx.array(rng.standard_normal((1, IMG_SIZE, IMG_SIZE, 4)).astype(np.float32))
        outputs = decoupled_model(x)
        # mx.eval is MLX array materialization, not Python eval
        mx.eval(outputs)  # noqa: S307

        expected_keys = {
            "alpha_logits", "fg_logits",
            "alpha_logits_up", "fg_logits_up",
            "alpha_coarse", "fg_coarse",
            "delta_logits",
            "alpha_final", "fg_final",
        }
        assert set(outputs.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Backward compatibility (requires checkpoint)
# ---------------------------------------------------------------------------


@has_checkpoint
class TestBackwardCompatibility:
    """backbone_size=None must produce identical output to the old code path."""

    @pytest.fixture(scope="class")
    def reference_output(self) -> dict[str, np.ndarray]:
        """Output from model without backbone_size (original behavior)."""
        model = GreenFormer(img_size=IMG_SIZE)
        model.load_checkpoint(MLX_CHECKPOINT_PATH)
        rng = np.random.default_rng(99)
        x = mx.array(rng.standard_normal((1, IMG_SIZE, IMG_SIZE, 4)).astype(np.float32))
        outputs = model(x, slim=True)
        # mx.eval is MLX array materialization, not Python eval
        mx.eval(outputs)  # noqa: S307
        return {k: np.array(v) for k, v in outputs.items()}

    @pytest.fixture(scope="class")
    def explicit_none_output(self) -> dict[str, np.ndarray]:
        """Output from model with backbone_size=None (should be identical)."""
        model = GreenFormer(img_size=IMG_SIZE, backbone_size=None)
        model.load_checkpoint(MLX_CHECKPOINT_PATH)
        rng = np.random.default_rng(99)
        x = mx.array(rng.standard_normal((1, IMG_SIZE, IMG_SIZE, 4)).astype(np.float32))
        outputs = model(x, slim=True)
        # mx.eval is MLX array materialization, not Python eval
        mx.eval(outputs)  # noqa: S307
        return {k: np.array(v) for k, v in outputs.items()}

    def test_exact_match(
        self,
        reference_output: dict[str, np.ndarray],
        explicit_none_output: dict[str, np.ndarray],
    ) -> None:
        for key in reference_output:
            np.testing.assert_array_equal(
                reference_output[key],
                explicit_none_output[key],
                err_msg=f"{key} differs with backbone_size=None vs omitted",
            )


# ---------------------------------------------------------------------------
# Decoupled with checkpoint (requires checkpoint)
# ---------------------------------------------------------------------------


@has_checkpoint
class TestDecoupledWithCheckpoint:
    """Validate decoupled inference with real weights produces valid output."""

    @pytest.fixture(scope="class")
    def decoupled_output(self) -> dict[str, np.ndarray]:
        model = GreenFormer(img_size=IMG_SIZE, backbone_size=BACKBONE_SIZE)
        model.load_checkpoint(MLX_CHECKPOINT_PATH)
        rng = np.random.default_rng(42)
        x = mx.array(rng.standard_normal((1, IMG_SIZE, IMG_SIZE, 4)).astype(np.float32))
        outputs = model(x, slim=True)
        # mx.eval is MLX array materialization, not Python eval
        mx.eval(outputs)  # noqa: S307
        return {k: np.array(v) for k, v in outputs.items()}

    def test_no_nan_inf(self, decoupled_output: dict[str, np.ndarray]) -> None:
        for key, arr in decoupled_output.items():
            assert not np.isnan(arr).any(), f"{key} contains NaN"
            assert not np.isinf(arr).any(), f"{key} contains Inf"

    def test_alpha_range(self, decoupled_output: dict[str, np.ndarray]) -> None:
        for key in ("alpha_coarse", "alpha_final"):
            arr = decoupled_output[key]
            assert arr.min() >= 0.0, f"{key} min={arr.min()}"
            assert arr.max() <= 1.0, f"{key} max={arr.max()}"

    def test_fg_range(self, decoupled_output: dict[str, np.ndarray]) -> None:
        for key in ("fg_coarse", "fg_final"):
            arr = decoupled_output[key]
            assert arr.min() >= 0.0, f"{key} min={arr.min()}"
            assert arr.max() <= 1.0, f"{key} max={arr.max()}"

    def test_shapes(self, decoupled_output: dict[str, np.ndarray]) -> None:
        assert decoupled_output["alpha_final"].shape == (1, IMG_SIZE, IMG_SIZE, 1)
        assert decoupled_output["fg_final"].shape == (1, IMG_SIZE, IMG_SIZE, 3)

    def test_alpha_not_constant(self, decoupled_output: dict[str, np.ndarray]) -> None:
        arr = decoupled_output["alpha_final"]
        assert arr.min() != arr.max(), "alpha_final is constant"
