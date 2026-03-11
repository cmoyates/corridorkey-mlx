"""Consolidated golden reference parity tests.

Merged from: test_end_to_end_parity.py, test_decoder_parity.py,
test_refiner_parity.py, test_hiera_stage_parity.py, test_reference_fixtures.py.

Loads full model with checkpoint, runs forward on golden input,
compares all outputs against PyTorch golden.npz references.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from corridorkey_mlx.model.corridorkey import GreenFormer
from corridorkey_mlx.utils.layout import nchw_to_nhwc_np, nhwc_to_nchw_np

from .conftest import (
    GOLDEN_PATH,
    IMG_SIZE,
    MLX_CHECKPOINT_PATH,
    PARITY_TOL_BACKBONE,
)

NUM_BACKBONE_STAGES = 4

# Per-output tolerances — backbone drift cascades through decoder/refiner.
# Tuned for 1024x1024 (higher resolution = more accumulated FP32 drift).
OUTPUT_TOLERANCES: dict[str, float] = {
    "alpha_logits": 3e-2,
    "fg_logits": 3e-2,
    "alpha_logits_up": 3e-2,
    "fg_logits_up": 3e-2,
    "alpha_coarse": 2e-3,
    "fg_coarse": 5e-3,
    "delta_logits": 1e-1,
    "alpha_final": 5e-3,
    "fg_final": 1.5e-2,
}


def _skip_if_missing() -> None:
    if not GOLDEN_PATH.exists():
        pytest.skip("golden.npz not found")
    if not MLX_CHECKPOINT_PATH.exists():
        pytest.skip("MLX checkpoint not found")


@pytest.fixture(scope="module")
def model_outputs_and_fixtures() -> tuple[dict[str, mx.array], dict[str, np.ndarray]]:
    """Load model, run forward with golden input, return (mlx_outputs, fixtures)."""
    _skip_if_missing()

    fixtures = dict(np.load(GOLDEN_PATH))

    # NCHW -> NHWC for MLX
    input_nhwc = mx.array(nchw_to_nhwc_np(fixtures["input"]))

    model = GreenFormer(img_size=IMG_SIZE)
    model.load_checkpoint(MLX_CHECKPOINT_PATH)

    outputs = model(input_nhwc)
    # mx.eval is MLX array materialization, not Python eval()
    mx.eval(outputs)  # noqa: S307

    return outputs, fixtures


# ---------------------------------------------------------------------------
# Backbone stage parity (4 stages)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def backbone_features() -> tuple[list[mx.array], dict[str, np.ndarray]]:
    """Run backbone alone to get intermediate stage features."""
    _skip_if_missing()

    fixtures = dict(np.load(GOLDEN_PATH))
    input_nhwc = mx.array(nchw_to_nhwc_np(fixtures["input"]))

    from corridorkey_mlx.model.hiera import HieraBackbone

    backbone = HieraBackbone(img_size=IMG_SIZE)
    backbone.load_checkpoint(MLX_CHECKPOINT_PATH)

    features = backbone(input_nhwc)
    mx.eval(features)  # noqa: S307

    return features, fixtures


@pytest.mark.parametrize("stage_idx", range(NUM_BACKBONE_STAGES))
def test_backbone_stage_parity(
    stage_idx: int,
    backbone_features: tuple[list[mx.array], dict[str, np.ndarray]],
) -> None:
    """MLX backbone stage output matches PyTorch within tolerance."""
    features, fixtures = backbone_features

    expected_nchw = fixtures[f"encoder_feature_{stage_idx}"]
    result_nchw = nhwc_to_nchw_np(np.array(features[stage_idx]))

    assert result_nchw.shape == expected_nchw.shape, (
        f"Stage {stage_idx} shape mismatch: {result_nchw.shape} vs {expected_nchw.shape}"
    )

    max_abs_err = float(np.max(np.abs(result_nchw - expected_nchw)))
    assert max_abs_err < PARITY_TOL_BACKBONE, (
        f"Stage {stage_idx} max_abs={max_abs_err:.2e} > {PARITY_TOL_BACKBONE}"
    )


# ---------------------------------------------------------------------------
# E2E output parity (decoder + refiner outputs)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", list(OUTPUT_TOLERANCES.keys()))
def test_output_parity(
    key: str,
    model_outputs_and_fixtures: tuple[dict[str, mx.array], dict[str, np.ndarray]],
) -> None:
    """MLX output matches PyTorch golden reference within tolerance."""
    outputs, fixtures = model_outputs_and_fixtures

    assert key in outputs, f"Missing MLX output key: {key}"
    assert key in fixtures, f"Missing fixture key: {key}"

    mlx_nchw = nhwc_to_nchw_np(np.array(outputs[key]))
    expected_nchw = fixtures[key]

    assert mlx_nchw.shape == expected_nchw.shape, (
        f"{key} shape mismatch: {mlx_nchw.shape} vs {expected_nchw.shape}"
    )

    max_abs = float(np.max(np.abs(mlx_nchw - expected_nchw)))
    tol = OUTPUT_TOLERANCES[key]

    assert max_abs < tol, f"{key}: max_abs={max_abs:.2e} > tol={tol:.1e}"
