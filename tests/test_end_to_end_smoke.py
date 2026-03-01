"""End-to-end smoke test for GreenFormer (Phase 5).

Verifies the full pipeline works with random weights — no checkpoint needed.
Tests model construction, forward pass, and basic output sanity.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from corridorkey_mlx.io.image import (
    postprocess_alpha,
    postprocess_foreground,
    preprocess,
)
from corridorkey_mlx.model.corridorkey import GreenFormer

IMG_SIZE = 256  # smaller for speed


@pytest.fixture(scope="module")
def model() -> GreenFormer:
    return GreenFormer(img_size=IMG_SIZE)


def test_forward_with_preprocessed_input(model: GreenFormer) -> None:
    """Full pipeline: numpy image -> preprocess -> model -> postprocess."""
    rgb = np.random.rand(IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
    alpha_hint = np.random.rand(IMG_SIZE, IMG_SIZE, 1).astype(np.float32)

    x = preprocess(rgb, alpha_hint)
    assert x.shape == (1, IMG_SIZE, IMG_SIZE, 4)

    out = model(x)
    # materialize — mx.eval is MLX's lazy graph evaluation, not Python eval
    mx.eval(out)

    alpha = postprocess_alpha(out["alpha_final"])
    fg = postprocess_foreground(out["fg_final"])

    assert alpha.shape == (IMG_SIZE, IMG_SIZE)
    assert alpha.dtype == np.uint8
    assert fg.shape == (IMG_SIZE, IMG_SIZE, 3)
    assert fg.dtype == np.uint8


def test_deterministic_output(model: GreenFormer) -> None:
    """Same input produces same output."""
    mx.random.seed(123)
    x = mx.random.normal((1, IMG_SIZE, IMG_SIZE, 4))
    # materialize — mx.eval is MLX's lazy graph evaluation, not Python eval
    mx.eval(x)

    out1 = model(x)
    mx.eval(out1)
    out2 = model(x)
    mx.eval(out2)

    for key in out1:
        np.testing.assert_array_equal(
            np.array(out1[key]),
            np.array(out2[key]),
            err_msg=f"{key} not deterministic",
        )
