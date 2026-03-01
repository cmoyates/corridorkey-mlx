"""Parity tests: MLX Hiera backbone vs PyTorch reference (Phase 4).

Loads golden input + encoder features from fixtures, runs MLX backbone
with checkpoint weights, compares each stage output.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from corridorkey_mlx.model.hiera import HieraBackbone
from corridorkey_mlx.utils.layout import nchw_to_nhwc_np, nhwc_to_nchw_np

FIXTURE_PATH = Path("reference/fixtures/golden.npz")
CHECKPOINT_PATH = Path("checkpoints/corridorkey_mlx.safetensors")
IMG_SIZE = 512
NUM_STAGES = 4

# Stage 2 runs 16 consecutive blocks — float32 drift accumulates on Metal vs CPU.
# Mean error stays < 1e-4; max outliers can reach ~0.01 in the deepest stage.
MAX_ABS_TOL = 2e-2


def _skip_if_missing() -> None:
    if not FIXTURE_PATH.exists():
        pytest.skip("Fixture files not found — run dump_pytorch_reference.py first")
    if not CHECKPOINT_PATH.exists():
        pytest.skip("Checkpoint not found — run scripts/convert_weights.py first")


@pytest.fixture(scope="module")
def backbone_and_fixtures() -> (
    tuple[list[mx.array], dict[str, np.ndarray]]
):
    """Load backbone once, return (mlx_features, fixtures)."""
    _skip_if_missing()

    fixtures = dict(np.load(FIXTURE_PATH))

    # Load input: NCHW -> NHWC
    input_nchw = fixtures["input"]
    input_nhwc = mx.array(nchw_to_nhwc_np(input_nchw))

    backbone = HieraBackbone(img_size=IMG_SIZE)
    backbone.load_checkpoint(CHECKPOINT_PATH)

    features = backbone(input_nhwc)
    # materialize all features — mx.eval is MLX lazy evaluation, not Python eval
    mx.eval(features)  # noqa: S307

    return features, fixtures


@pytest.mark.parametrize("stage_idx", range(NUM_STAGES))
def test_stage_parity(
    stage_idx: int,
    backbone_and_fixtures: tuple[list[mx.array], dict[str, np.ndarray]],
) -> None:
    """MLX backbone stage output matches PyTorch within tolerance."""
    features, fixtures = backbone_and_fixtures

    expected_nchw = fixtures[f"encoder_feature_{stage_idx}"]
    result_nhwc = features[stage_idx]
    result_nchw = nhwc_to_nchw_np(np.array(result_nhwc))

    assert result_nchw.shape == expected_nchw.shape, (
        f"Stage {stage_idx} shape mismatch: {result_nchw.shape} vs {expected_nchw.shape}"
    )

    abs_err = np.abs(result_nchw - expected_nchw)
    max_abs_err = float(np.max(abs_err))
    mean_abs_err = float(np.mean(abs_err))
    print(
        f"\nStage {stage_idx} parity — "
        f"max_abs: {max_abs_err:.6e}, mean_abs: {mean_abs_err:.6e}"
    )

    assert max_abs_err < MAX_ABS_TOL, (
        f"Stage {stage_idx} max abs error {max_abs_err:.6e} exceeds tolerance {MAX_ABS_TOL}"
    )
