"""Parity tests: MLX refiner vs PyTorch reference (Phase 2).

Uses saved coarse predictions + RGB -> runs MLX refiner ->
compares against saved PyTorch delta logits and final outputs.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from corridorkey_mlx.model.refiner import CNNRefinerModule
from corridorkey_mlx.utils.layout import conv_weight_pt_to_mlx, nchw_to_nhwc_np, nhwc_to_nchw_np

FIXTURE_PATH = Path("reference/fixtures/golden.npz")
WEIGHTS_PATH = Path("reference/fixtures/golden_weights.npz")

MAX_ABS_TOL = 1e-4


def _skip_if_missing() -> None:
    if not FIXTURE_PATH.exists() or not WEIGHTS_PATH.exists():
        pytest.skip("Fixture files not found — run dump_pytorch_reference.py first")


def _load_refiner_weights(weights: dict[str, np.ndarray]) -> list[tuple[str, mx.array]]:
    """Convert PyTorch refiner state_dict to MLX weight list.

    Handles:
    - Conv weight transpose (O,I,H,W) -> (O,H,W,I)
    - Key mapping from PyTorch Sequential indices to named attributes
    """
    prefix = "refiner."
    weight_list: list[tuple[str, mx.array]] = []

    # Map PyTorch stem Sequential keys to our named attributes
    stem_key_map = {
        "stem.0.weight": "stem_conv.weight",
        "stem.0.bias": "stem_conv.bias",
        "stem.1.weight": "stem_gn.weight",
        "stem.1.bias": "stem_gn.bias",
    }

    for pt_key, value in weights.items():
        if not pt_key.startswith(prefix):
            continue
        mlx_key = pt_key[len(prefix) :]

        # Remap stem keys
        if mlx_key in stem_key_map:
            mlx_key = stem_key_map[mlx_key]

        # Conv2d weights: (O,I,H,W) -> (O,H,W,I)
        if mlx_key.endswith(".weight") and _is_conv_weight(value):
            value = conv_weight_pt_to_mlx(value)

        weight_list.append((mlx_key, mx.array(value)))

    return weight_list


def _is_conv_weight(value: np.ndarray) -> bool:
    """Check if a weight tensor is a conv weight (4D with spatial dims)."""
    return value.ndim == 4


def _build_and_load_refiner(weights: dict[str, np.ndarray]) -> CNNRefinerModule:
    """Build an MLX CNNRefinerModule and load converted weights."""
    refiner = CNNRefinerModule()
    weight_list = _load_refiner_weights(weights)
    refiner.load_weights(weight_list)
    refiner.eval()
    mx.eval(refiner.parameters())  # noqa: S307 — mx.eval is MLX's compute trigger
    return refiner


def test_refiner_delta_parity() -> None:
    """MLX refiner delta logits match PyTorch within tolerance."""
    _skip_if_missing()

    fixtures = dict(np.load(FIXTURE_PATH))
    weights = dict(np.load(WEIGHTS_PATH))

    # RGB: first 3 channels of input
    rgb_nchw = fixtures["input"][:, :3]  # (1, 3, H, W)
    rgb_nhwc = mx.array(nchw_to_nhwc_np(rgb_nchw))

    # Coarse predictions: alpha_coarse (1ch) + fg_coarse (3ch) = 4ch
    alpha_coarse_nhwc = nchw_to_nhwc_np(fixtures["alpha_coarse"])
    fg_coarse_nhwc = nchw_to_nhwc_np(fixtures["fg_coarse"])
    coarse_pred = mx.array(np.concatenate([alpha_coarse_nhwc, fg_coarse_nhwc], axis=-1))

    refiner = _build_and_load_refiner(weights)
    result_nhwc = refiner(rgb_nhwc, coarse_pred)
    mx.eval(result_nhwc)  # noqa: S307

    result_nchw = nhwc_to_nchw_np(np.array(result_nhwc))
    expected = fixtures["delta_logits"]

    max_abs_err = float(np.max(np.abs(result_nchw - expected)))
    mean_abs_err = float(np.mean(np.abs(result_nchw - expected)))
    print(f"\nRefiner delta parity — max_abs: {max_abs_err:.6e}, mean_abs: {mean_abs_err:.6e}")

    assert result_nchw.shape == expected.shape
    assert max_abs_err < MAX_ABS_TOL, (
        f"Max abs error {max_abs_err:.6e} exceeds tolerance {MAX_ABS_TOL}"
    )


def test_refiner_final_output_parity() -> None:
    """MLX final alpha and fg match PyTorch within tolerance.

    Tests the full residual + sigmoid path using MLX refiner output.
    """
    _skip_if_missing()

    fixtures = dict(np.load(FIXTURE_PATH))
    weights = dict(np.load(WEIGHTS_PATH))

    # Run refiner
    rgb_nchw = fixtures["input"][:, :3]
    rgb_nhwc = mx.array(nchw_to_nhwc_np(rgb_nchw))
    alpha_coarse_nhwc = nchw_to_nhwc_np(fixtures["alpha_coarse"])
    fg_coarse_nhwc = nchw_to_nhwc_np(fixtures["fg_coarse"])
    coarse_pred = mx.array(np.concatenate([alpha_coarse_nhwc, fg_coarse_nhwc], axis=-1))

    refiner = _build_and_load_refiner(weights)
    delta_nhwc = refiner(rgb_nhwc, coarse_pred)
    mx.eval(delta_nhwc)  # noqa: S307

    # Apply residual in logit space + sigmoid (matching PyTorch forward pass)
    alpha_logits_up = mx.array(nchw_to_nhwc_np(fixtures["alpha_logits_up"]))
    fg_logits_up = mx.array(nchw_to_nhwc_np(fixtures["fg_logits_up"]))

    alpha_final = mx.sigmoid(alpha_logits_up + delta_nhwc[..., 0:1])
    fg_final = mx.sigmoid(fg_logits_up + delta_nhwc[..., 1:4])
    mx.eval(alpha_final, fg_final)  # noqa: S307

    # Compare
    for name, result, expected_key in [
        ("alpha_final", alpha_final, "alpha_final"),
        ("fg_final", fg_final, "fg_final"),
    ]:
        result_nchw = nhwc_to_nchw_np(np.array(result))
        expected = fixtures[expected_key]
        max_abs_err = float(np.max(np.abs(result_nchw - expected)))
        mean_abs_err = float(np.mean(np.abs(result_nchw - expected)))
        print(f"\n{name} parity — max_abs: {max_abs_err:.6e}, mean_abs: {mean_abs_err:.6e}")
        assert max_abs_err < MAX_ABS_TOL, (
            f"{name}: max abs error {max_abs_err:.6e} exceeds tolerance {MAX_ABS_TOL}"
        )
