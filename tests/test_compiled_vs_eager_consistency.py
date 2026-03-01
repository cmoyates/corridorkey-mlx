"""Test that mx.compile() produces numerically consistent results vs eager mode."""

from __future__ import annotations

import mlx.core as mx
import pytest

from corridorkey_mlx.model.corridorkey import GreenFormer

IMG_SIZE = 256
TOLERANCE = 1e-4
OUTPUT_KEYS = ("alpha_final", "fg_final", "alpha_coarse", "fg_coarse", "delta_logits")


@pytest.fixture()
def model() -> GreenFormer:
    model = GreenFormer(img_size=IMG_SIZE)
    model.eval()
    # NOTE: mx.eval is MLX array materialization, not Python eval()
    mx.eval(model.parameters())  # noqa: S307
    return model


@pytest.fixture()
def dummy_input() -> mx.array:
    mx.random.seed(42)
    x = mx.random.normal((1, IMG_SIZE, IMG_SIZE, 4))
    mx.eval(x)  # noqa: S307
    return x


def test_compiled_matches_eager(model: GreenFormer, dummy_input: mx.array) -> None:
    """Fixed-shape compiled output matches eager output within tolerance."""
    eager_out = model(dummy_input)
    mx.eval(eager_out)  # noqa: S307

    compiled_fn = mx.compile(model.__call__)
    compiled_out = compiled_fn(dummy_input)
    mx.eval(compiled_out)  # noqa: S307

    for key in OUTPUT_KEYS:
        diff = float(mx.max(mx.abs(eager_out[key] - compiled_out[key])))
        assert diff < TOLERANCE, f"{key}: max_abs_diff={diff:.2e} > {TOLERANCE}"


def test_compiled_deterministic(model: GreenFormer, dummy_input: mx.array) -> None:
    """Compiled model produces identical results across consecutive calls."""
    compiled_fn = mx.compile(model.__call__)

    out1 = compiled_fn(dummy_input)
    mx.eval(out1)  # noqa: S307
    out2 = compiled_fn(dummy_input)
    mx.eval(out2)  # noqa: S307

    for key in OUTPUT_KEYS:
        diff = float(mx.max(mx.abs(out1[key] - out2[key])))
        assert diff == 0.0, f"{key}: non-deterministic, max_diff={diff:.2e}"
