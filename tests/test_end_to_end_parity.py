"""End-to-end parity tests: MLX GreenFormer vs PyTorch golden reference (Phase 5).

Loads golden.npz (PyTorch NCHW), runs same input through MLX model,
compares all intermediate and final outputs.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from corridorkey_mlx.model.corridorkey import GreenFormer
from corridorkey_mlx.utils.layout import nchw_to_nhwc_np, nhwc_to_nchw_np

FIXTURE_PATH = Path("reference/fixtures/golden.npz")
CHECKPOINT_PATH = Path("checkpoints/corridorkey_mlx.safetensors")
IMG_SIZE = 512

# Tolerance tiers — coarse path inherits backbone drift, refiner adds more.
# Backbone stages have up to ~0.01 max abs err (16 sequential blocks).
# Final outputs pass through sigmoid which compresses errors.
TOLERANCES: dict[str, float] = {
    "alpha_logits": 1e-3,
    "fg_logits": 1e-3,
    "alpha_logits_up": 1e-3,
    "fg_logits_up": 1e-3,
    "alpha_coarse": 1e-4,
    "fg_coarse": 1e-4,
    "delta_logits": 2e-3,
    "alpha_final": 2e-4,
    "fg_final": 2e-4,
}


def _skip_if_missing() -> None:
    if not FIXTURE_PATH.exists():
        pytest.skip("golden.npz not found — run dump_pytorch_reference.py first")
    if not CHECKPOINT_PATH.exists():
        pytest.skip("Checkpoint not found — run scripts/convert_weights.py first")


@pytest.fixture(scope="module")
def model_outputs_and_fixtures() -> tuple[dict[str, mx.array], dict[str, np.ndarray]]:
    """Load model, run forward pass with golden input, return (mlx_outputs, fixtures)."""
    _skip_if_missing()

    fixtures = dict(np.load(FIXTURE_PATH))

    # Load input: NCHW -> NHWC
    input_nhwc = mx.array(nchw_to_nhwc_np(fixtures["input"]))

    model = GreenFormer(img_size=IMG_SIZE)
    model.load_checkpoint(CHECKPOINT_PATH)

    outputs = model(input_nhwc)
    # materialize — mx.eval is MLX lazy graph evaluation, not Python eval
    mx.eval(outputs)  # noqa: S307

    return outputs, fixtures


@pytest.mark.parametrize("key", list(TOLERANCES.keys()))
def test_parity(
    key: str,
    model_outputs_and_fixtures: tuple[dict[str, mx.array], dict[str, np.ndarray]],
) -> None:
    """MLX output matches PyTorch golden reference within tolerance."""
    outputs, fixtures = model_outputs_and_fixtures

    assert key in outputs, f"Missing output key: {key}"
    assert key in fixtures, f"Missing fixture key: {key}"

    # MLX output is NHWC, fixture is NCHW — convert MLX to NCHW for comparison
    mlx_nchw = nhwc_to_nchw_np(np.array(outputs[key]))
    expected_nchw = fixtures[key]

    assert mlx_nchw.shape == expected_nchw.shape, (
        f"{key} shape mismatch: MLX {mlx_nchw.shape} vs PyTorch {expected_nchw.shape}"
    )

    abs_err = np.abs(mlx_nchw - expected_nchw)
    max_abs = float(np.max(abs_err))
    mean_abs = float(np.mean(abs_err))
    tol = TOLERANCES[key]

    print(
        f"\n{key:<20s} | shape={mlx_nchw.shape} | "
        f"max_abs={max_abs:.6e} | mean_abs={mean_abs:.6e} | tol={tol:.1e}"
    )

    assert max_abs < tol, f"{key}: max abs error {max_abs:.6e} exceeds tolerance {tol:.1e}"
