"""Parity tests: MLX decoder heads vs PyTorch reference (Phase 2).

Uses saved backbone features -> runs MLX decoder -> compares
against saved PyTorch coarse predictions.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from corridorkey_mlx.model.decoder import DecoderHead
from corridorkey_mlx.utils.layout import conv_weight_pt_to_mlx, nchw_to_nhwc_np, nhwc_to_nchw_np

FIXTURE_PATH = Path("reference/fixtures/golden.npz")
WEIGHTS_PATH = Path("reference/fixtures/golden_weights.npz")

BACKBONE_CHANNELS = [112, 224, 448, 896]
EMBED_DIM = 256
MAX_ABS_TOL = 1e-4  # relaxed slightly for float32 Metal vs CPU differences


def _skip_if_missing() -> None:
    if not FIXTURE_PATH.exists() or not WEIGHTS_PATH.exists():
        pytest.skip("Fixture files not found — run dump_pytorch_reference.py first")


def _load_decoder_weights(
    prefix: str,
    weights: dict[str, np.ndarray],
) -> dict[str, mx.array]:
    """Convert PyTorch decoder state_dict to MLX parameter dict.

    Maps PyTorch keys (e.g. 'alpha_decoder.linear_c1.proj.weight') to
    MLX nested keys (e.g. 'linear_c1.proj.weight'), transposing conv weights.
    """
    params: dict[str, mx.array] = {}
    for pt_key, value in weights.items():
        if not pt_key.startswith(prefix):
            continue
        mlx_key = pt_key[len(prefix) :]

        # Conv2d weights: (O,I,H,W) -> (O,H,W,I)
        if "linear_fuse.weight" in mlx_key or "classifier.weight" in mlx_key:
            value = conv_weight_pt_to_mlx(value)

        # BatchNorm: running_mean -> running_mean, running_var -> running_var
        # PyTorch uses num_batches_tracked which MLX doesn't need
        if "num_batches_tracked" in mlx_key:
            continue

        params[mlx_key] = mx.array(value)

    return params


def _params_to_nested(flat: dict[str, mx.array]) -> dict:
    """Convert flat dot-separated keys to nested dict for mlx load_weights."""
    nested: dict = {}
    for key, value in flat.items():
        parts = key.split(".")
        current = nested
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return nested


def _build_and_load_decoder(
    output_dim: int,
    prefix: str,
    weights: dict[str, np.ndarray],
) -> DecoderHead:
    """Build an MLX DecoderHead and load converted weights."""
    decoder = DecoderHead(BACKBONE_CHANNELS, EMBED_DIM, output_dim)
    flat_params = _load_decoder_weights(prefix, weights)
    nested_params = _params_to_nested(flat_params)
    decoder.load_weights(list(_flatten_nested(nested_params)))
    decoder.eval()  # use running stats for BatchNorm (not batch stats)
    mx.eval(decoder.parameters())  # noqa: S307 — mx.eval is MLX lazy eval, not Python eval
    return decoder


def _flatten_nested(d: dict, prefix: str = "") -> list[tuple[str, mx.array]]:
    """Flatten nested dict to list of (dotted_key, array) pairs."""
    items: list[tuple[str, mx.array]] = []
    for k, v in d.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.extend(_flatten_nested(v, full_key))
        else:
            items.append((full_key, v))
    return items


def _run_decoder_parity(
    output_dim: int,
    prefix: str,
    expected_key: str,
) -> None:
    """Run a decoder parity test."""
    _skip_if_missing()

    fixtures = dict(np.load(FIXTURE_PATH))
    weights = dict(np.load(WEIGHTS_PATH))

    # Load features in NHWC
    features_nhwc = [mx.array(nchw_to_nhwc_np(fixtures[f"encoder_feature_{i}"])) for i in range(4)]

    decoder = _build_and_load_decoder(output_dim, prefix, weights)
    result_nhwc = decoder(features_nhwc)
    mx.eval(result_nhwc)  # noqa: S307 — mx.eval is MLX lazy eval, not Python eval

    # Convert result back to NCHW for comparison
    result_nchw = nhwc_to_nchw_np(np.array(result_nhwc))
    expected = fixtures[expected_key]

    max_abs_err = float(np.max(np.abs(result_nchw - expected)))
    mean_abs_err = float(np.mean(np.abs(result_nchw - expected)))
    print(f"\n{prefix} parity — max_abs: {max_abs_err:.6e}, mean_abs: {mean_abs_err:.6e}")

    assert result_nchw.shape == expected.shape, (
        f"Shape mismatch: {result_nchw.shape} vs {expected.shape}"
    )
    assert max_abs_err < MAX_ABS_TOL, (
        f"Max abs error {max_abs_err:.6e} exceeds tolerance {MAX_ABS_TOL}"
    )


def test_alpha_decoder_parity() -> None:
    """MLX alpha decoder matches PyTorch within tolerance."""
    _run_decoder_parity(
        output_dim=1,
        prefix="alpha_decoder.",
        expected_key="alpha_logits",
    )


def test_fg_decoder_parity() -> None:
    """MLX foreground decoder matches PyTorch within tolerance."""
    _run_decoder_parity(
        output_dim=3,
        prefix="fg_decoder.",
        expected_key="fg_logits",
    )
