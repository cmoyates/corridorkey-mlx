"""FP16 vs FP32 parity tests.

Compares FP16 MLX output against FP32 MLX output (NOT PyTorch golden).
Looser tolerance than FP32-vs-PT because of FP16 rounding + refiner 10x scale.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from corridorkey_mlx.inference.pipeline import load_model
from corridorkey_mlx.utils.layout import nchw_to_nhwc_np

from .conftest import GOLDEN_PATH, IMG_SIZE, MLX_CHECKPOINT_PATH

FP16_TOLERANCE = 1e-3

ENGINE_OUTPUT_KEYS = ["alpha_coarse", "fg_coarse", "alpha_final", "fg_final"]


def _skip_if_missing() -> None:
    if not GOLDEN_PATH.exists():
        pytest.skip("golden.npz not found")
    if not MLX_CHECKPOINT_PATH.exists():
        pytest.skip("MLX checkpoint not found")


@pytest.fixture(scope="module")
def fp32_and_fp16_outputs() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Run forward pass with FP32 and FP16 models, return numpy outputs."""
    _skip_if_missing()

    fixtures = dict(np.load(GOLDEN_PATH))
    input_nhwc = mx.array(nchw_to_nhwc_np(fixtures["input"]))

    # FP32 baseline
    model_fp32 = load_model(MLX_CHECKPOINT_PATH, img_size=IMG_SIZE)
    out_fp32 = model_fp32(input_nhwc)
    # mx.eval materializes lazy MLX arrays (not Python eval)
    mx.eval(out_fp32)  # noqa: S307
    np_fp32 = {k: np.array(v) for k, v in out_fp32.items()}

    # FP16 (mixed precision: backbone FP32, decoder+refiner FP16)
    # Input stays FP32 — backbone needs full precision
    model_fp16 = load_model(MLX_CHECKPOINT_PATH, img_size=IMG_SIZE, fp16=True)
    out_fp16 = model_fp16(input_nhwc)
    mx.eval(out_fp16)  # noqa: S307
    np_fp16 = {k: np.array(v) for k, v in out_fp16.items()}

    return np_fp32, np_fp16


@pytest.mark.parametrize("key", ENGINE_OUTPUT_KEYS)
def test_fp16_vs_fp32_parity(
    key: str,
    fp32_and_fp16_outputs: tuple[dict[str, np.ndarray], dict[str, np.ndarray]],
) -> None:
    """FP16 output matches FP32 output within tolerance for engine-relevant keys."""
    np_fp32, np_fp16 = fp32_and_fp16_outputs

    assert key in np_fp32, f"Missing FP32 key: {key}"
    assert key in np_fp16, f"Missing FP16 key: {key}"

    fp32_val = np_fp32[key].astype(np.float32)
    fp16_val = np_fp16[key].astype(np.float32)

    assert fp32_val.shape == fp16_val.shape, (
        f"{key} shape mismatch: {fp32_val.shape} vs {fp16_val.shape}"
    )

    max_abs = float(np.max(np.abs(fp32_val - fp16_val)))
    mean_abs = float(np.mean(np.abs(fp32_val - fp16_val)))

    assert max_abs < FP16_TOLERANCE, (
        f"{key}: max_abs={max_abs:.2e} > tol={FP16_TOLERANCE:.1e} (mean={mean_abs:.2e})"
    )


def test_fp16_outputs_valid(
    fp32_and_fp16_outputs: tuple[dict[str, np.ndarray], dict[str, np.ndarray]],
) -> None:
    """FP16 outputs have no NaN/Inf and alpha/fg are in [0, 1]."""
    _, np_fp16 = fp32_and_fp16_outputs

    for key in ENGINE_OUTPUT_KEYS:
        val = np_fp16[key].astype(np.float32)
        assert not np.any(np.isnan(val)), f"{key} has NaN"
        assert not np.any(np.isinf(val)), f"{key} has Inf"
        assert float(np.min(val)) >= 0.0, f"{key} min={np.min(val):.4f} < 0"
        assert float(np.max(val)) <= 1.0, f"{key} max={np.max(val):.4f} > 1"


def test_fp32_path_unaffected() -> None:
    """load_model without fp16 returns FP32 parameters (opt-in check)."""
    if not MLX_CHECKPOINT_PATH.exists():
        pytest.skip("MLX checkpoint not found")

    model = load_model(MLX_CHECKPOINT_PATH, img_size=IMG_SIZE, fp16=False)
    params = model.parameters()
    first_param = next(iter(_flatten_params(params)))
    assert first_param.dtype == mx.float32, (
        f"Expected float32 params when fp16=False, got {first_param.dtype}"
    )


def _flatten_params(params: dict | list | mx.array) -> list[mx.array]:
    """Recursively flatten nested parameter dict/list to list of arrays."""
    if isinstance(params, mx.array):
        return [params]
    if isinstance(params, dict):
        result = []
        for v in params.values():
            result.extend(_flatten_params(v))
        return result
    if isinstance(params, list):
        result = []
        for v in params:
            result.extend(_flatten_params(v))
        return result
    return []
