"""Shared test fixtures, paths, tolerances, and skip markers."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from corridorkey_mlx.model.corridorkey import GreenFormer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GOLDEN_PATH = Path("reference/fixtures/golden.npz")
GOLDEN_WEIGHTS_PATH = Path("reference/fixtures/golden_weights.npz")
MLX_CHECKPOINT_PATH = Path("checkpoints/corridorkey_mlx.safetensors")
PT_CHECKPOINT_PATH = Path("checkpoints/CorridorKey_v1.0.pth")

# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------
PARITY_TOL_TIGHT = 1e-4
PARITY_TOL_BACKBONE = 2e-2
PARITY_TOL_E2E = 1e-3

# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------
has_golden = pytest.mark.skipif(not GOLDEN_PATH.exists(), reason="golden.npz not found")
has_checkpoint = pytest.mark.skipif(
    not MLX_CHECKPOINT_PATH.exists(), reason="MLX checkpoint not found"
)
has_pt_checkpoint = pytest.mark.skipif(
    not PT_CHECKPOINT_PATH.exists(), reason="PyTorch checkpoint not found"
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
IMG_SIZE = 512
SMALL_IMG_SIZE = 256


@pytest.fixture(scope="module")
def golden_fixtures() -> dict[str, np.ndarray]:
    """Load golden.npz fixture file (PyTorch NCHW format)."""
    if not GOLDEN_PATH.exists():
        pytest.skip("golden.npz not found")
    return dict(np.load(GOLDEN_PATH))


@pytest.fixture(scope="module")
def random_model() -> GreenFormer:
    """GreenFormer with random weights at 512."""
    model = GreenFormer(img_size=IMG_SIZE)
    model.eval()
    # mx.eval is MLX array materialization, not Python eval()
    mx.eval(model.parameters())  # noqa: S307
    return model


@pytest.fixture(scope="module")
def loaded_model() -> GreenFormer:
    """GreenFormer with checkpoint weights at 512."""
    if not MLX_CHECKPOINT_PATH.exists():
        pytest.skip("MLX checkpoint not found")
    model = GreenFormer(img_size=IMG_SIZE)
    model.load_checkpoint(MLX_CHECKPOINT_PATH)
    return model
