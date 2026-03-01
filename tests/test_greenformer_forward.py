"""Forward pass shape and smoke tests for GreenFormer (Phase 5).

Uses random weights — no checkpoint needed.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from corridorkey_mlx.model.corridorkey import GreenFormer

IMG_SIZE = 512
BATCH = 1


@pytest.fixture(scope="module")
def model_and_output() -> tuple[GreenFormer, dict[str, mx.array]]:
    """Build model with random weights and run forward pass once."""
    model = GreenFormer(img_size=IMG_SIZE)
    x = mx.random.normal((BATCH, IMG_SIZE, IMG_SIZE, 4))
    out = model(x)
    # materialize all outputs — mx.eval is MLX lazy graph evaluation, not Python eval
    mx.eval(out)
    return model, out


EXPECTED_SHAPES: dict[str, tuple[int, ...]] = {
    "alpha_logits": (BATCH, IMG_SIZE // 4, IMG_SIZE // 4, 1),
    "fg_logits": (BATCH, IMG_SIZE // 4, IMG_SIZE // 4, 3),
    "alpha_logits_up": (BATCH, IMG_SIZE, IMG_SIZE, 1),
    "fg_logits_up": (BATCH, IMG_SIZE, IMG_SIZE, 3),
    "alpha_coarse": (BATCH, IMG_SIZE, IMG_SIZE, 1),
    "fg_coarse": (BATCH, IMG_SIZE, IMG_SIZE, 3),
    "delta_logits": (BATCH, IMG_SIZE, IMG_SIZE, 4),
    "alpha_final": (BATCH, IMG_SIZE, IMG_SIZE, 1),
    "fg_final": (BATCH, IMG_SIZE, IMG_SIZE, 3),
}


@pytest.mark.parametrize("key,expected_shape", EXPECTED_SHAPES.items())
def test_output_shapes(
    key: str,
    expected_shape: tuple[int, ...],
    model_and_output: tuple[GreenFormer, dict[str, mx.array]],
) -> None:
    _, out = model_and_output
    assert key in out, f"Missing output key: {key}"
    assert out[key].shape == expected_shape, (
        f"{key}: expected {expected_shape}, got {out[key].shape}"
    )


def test_all_keys_present(
    model_and_output: tuple[GreenFormer, dict[str, mx.array]],
) -> None:
    _, out = model_and_output
    assert set(out.keys()) == set(EXPECTED_SHAPES.keys())


def test_coarse_probs_in_range(
    model_and_output: tuple[GreenFormer, dict[str, mx.array]],
) -> None:
    """Sigmoid outputs must be in [0, 1]."""
    _, out = model_and_output
    for key in ("alpha_coarse", "fg_coarse", "alpha_final", "fg_final"):
        arr = out[key]
        assert float(mx.min(arr)) >= 0.0, f"{key} has values < 0"
        assert float(mx.max(arr)) <= 1.0, f"{key} has values > 1"


def test_output_dtype(
    model_and_output: tuple[GreenFormer, dict[str, mx.array]],
) -> None:
    _, out = model_and_output
    for key, arr in out.items():
        assert arr.dtype == mx.float32, f"{key}: expected float32, got {arr.dtype}"
