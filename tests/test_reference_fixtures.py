"""Parity tests: PyTorch reference fixtures (Phase 1).

These tests validate that dump_pytorch_reference.py produces
fixtures with expected shapes and dtypes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

FIXTURE_PATH = Path("reference/fixtures/golden.npz")

IMG_SIZE = 512
BATCH = 1

# Expected shapes for each tensor in the fixture file
EXPECTED_SHAPES: dict[str, tuple[int, ...]] = {
    "input": (BATCH, 4, IMG_SIZE, IMG_SIZE),
    # Backbone features at strides 4, 8, 16, 32
    "encoder_feature_0": (BATCH, 112, IMG_SIZE // 4, IMG_SIZE // 4),
    "encoder_feature_1": (BATCH, 224, IMG_SIZE // 8, IMG_SIZE // 8),
    "encoder_feature_2": (BATCH, 448, IMG_SIZE // 16, IMG_SIZE // 16),
    "encoder_feature_3": (BATCH, 896, IMG_SIZE // 32, IMG_SIZE // 32),
    # Decoder outputs at H/4 resolution
    "alpha_logits": (BATCH, 1, IMG_SIZE // 4, IMG_SIZE // 4),
    "fg_logits": (BATCH, 3, IMG_SIZE // 4, IMG_SIZE // 4),
    # Upsampled to full resolution
    "alpha_logits_up": (BATCH, 1, IMG_SIZE, IMG_SIZE),
    "fg_logits_up": (BATCH, 3, IMG_SIZE, IMG_SIZE),
    # Coarse predictions (after sigmoid)
    "alpha_coarse": (BATCH, 1, IMG_SIZE, IMG_SIZE),
    "fg_coarse": (BATCH, 3, IMG_SIZE, IMG_SIZE),
    # Refiner
    "delta_logits": (BATCH, 4, IMG_SIZE, IMG_SIZE),
    # Final outputs
    "alpha_final": (BATCH, 1, IMG_SIZE, IMG_SIZE),
    "fg_final": (BATCH, 3, IMG_SIZE, IMG_SIZE),
}


@pytest.fixture()
def fixtures() -> dict[str, np.ndarray]:
    if not FIXTURE_PATH.exists():
        pytest.skip(f"Fixture file not found: {FIXTURE_PATH}")
    return dict(np.load(FIXTURE_PATH))


class TestFixtureCompleteness:
    """All expected tensors exist in the fixture file."""

    def test_all_keys_present(self, fixtures: dict[str, np.ndarray]) -> None:
        missing = set(EXPECTED_SHAPES.keys()) - set(fixtures.keys())
        assert not missing, f"Missing fixture keys: {missing}"

    def test_no_extra_keys(self, fixtures: dict[str, np.ndarray]) -> None:
        extra = set(fixtures.keys()) - set(EXPECTED_SHAPES.keys())
        assert not extra, f"Unexpected fixture keys: {extra}"


class TestBackboneFeaturesShape:
    """4 feature maps from Hiera backbone have expected shapes."""

    @pytest.mark.parametrize("idx", range(4))
    def test_feature_shape(self, fixtures: dict[str, np.ndarray], idx: int) -> None:
        key = f"encoder_feature_{idx}"
        assert fixtures[key].shape == EXPECTED_SHAPES[key]

    @pytest.mark.parametrize("idx", range(4))
    def test_feature_dtype(self, fixtures: dict[str, np.ndarray], idx: int) -> None:
        key = f"encoder_feature_{idx}"
        assert fixtures[key].dtype == np.float32


class TestCoarsePredictionsShape:
    """Alpha (1ch) and foreground (3ch) coarse logits/probs exist with correct shapes."""

    @pytest.mark.parametrize(
        "key",
        [
            "alpha_logits",
            "fg_logits",
            "alpha_logits_up",
            "fg_logits_up",
            "alpha_coarse",
            "fg_coarse",
        ],
    )
    def test_shape(self, fixtures: dict[str, np.ndarray], key: str) -> None:
        assert fixtures[key].shape == EXPECTED_SHAPES[key]

    @pytest.mark.parametrize(
        "key",
        [
            "alpha_logits",
            "fg_logits",
            "alpha_logits_up",
            "fg_logits_up",
            "alpha_coarse",
            "fg_coarse",
        ],
    )
    def test_dtype(self, fixtures: dict[str, np.ndarray], key: str) -> None:
        assert fixtures[key].dtype == np.float32

    @pytest.mark.parametrize("key", ["alpha_coarse", "fg_coarse"])
    def test_sigmoid_range(self, fixtures: dict[str, np.ndarray], key: str) -> None:
        """Coarse predictions (post-sigmoid) should be in [0, 1]."""
        assert fixtures[key].min() >= 0.0
        assert fixtures[key].max() <= 1.0


class TestRefinerOutputsShape:
    """Delta logits and final alpha/fg have expected shapes."""

    @pytest.mark.parametrize("key", ["delta_logits", "alpha_final", "fg_final"])
    def test_shape(self, fixtures: dict[str, np.ndarray], key: str) -> None:
        assert fixtures[key].shape == EXPECTED_SHAPES[key]

    @pytest.mark.parametrize("key", ["delta_logits", "alpha_final", "fg_final"])
    def test_dtype(self, fixtures: dict[str, np.ndarray], key: str) -> None:
        assert fixtures[key].dtype == np.float32

    @pytest.mark.parametrize("key", ["alpha_final", "fg_final"])
    def test_final_sigmoid_range(self, fixtures: dict[str, np.ndarray], key: str) -> None:
        """Final predictions (post-sigmoid) should be in [0, 1]."""
        assert fixtures[key].min() >= 0.0
        assert fixtures[key].max() <= 1.0
