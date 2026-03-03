"""Model contract tests: imports, shapes, outputs, determinism.

Consolidated from: test_import.py, test_greenformer_forward.py,
test_hiera_stage_shapes.py, test_end_to_end_smoke.py, test_smoke_2048.py (wiring).

No checkpoint needed — uses random weights only.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from corridorkey_mlx.model.corridorkey import GreenFormer
from corridorkey_mlx.model.hiera import HieraBackbone, HieraPatchEmbed

from .conftest import IMG_SIZE, SMALL_IMG_SIZE

# ---------------------------------------------------------------------------
# Package imports
# ---------------------------------------------------------------------------


def test_package_imports() -> None:
    import corridorkey_mlx
    import corridorkey_mlx.convert  # noqa: F401
    import corridorkey_mlx.inference  # noqa: F401
    import corridorkey_mlx.io  # noqa: F401
    import corridorkey_mlx.model  # noqa: F401
    import corridorkey_mlx.utils  # noqa: F401

    assert corridorkey_mlx.__version__ == "0.1.0"


# ---------------------------------------------------------------------------
# GreenFormer output contract (random weights)
# ---------------------------------------------------------------------------

EXPECTED_SHAPES: dict[str, tuple[int, ...]] = {
    "alpha_logits": (1, IMG_SIZE // 4, IMG_SIZE // 4, 1),
    "fg_logits": (1, IMG_SIZE // 4, IMG_SIZE // 4, 3),
    "alpha_logits_up": (1, IMG_SIZE, IMG_SIZE, 1),
    "fg_logits_up": (1, IMG_SIZE, IMG_SIZE, 3),
    "alpha_coarse": (1, IMG_SIZE, IMG_SIZE, 1),
    "fg_coarse": (1, IMG_SIZE, IMG_SIZE, 3),
    "delta_logits": (1, IMG_SIZE, IMG_SIZE, 4),
    "alpha_final": (1, IMG_SIZE, IMG_SIZE, 1),
    "fg_final": (1, IMG_SIZE, IMG_SIZE, 3),
}


@pytest.fixture(scope="module")
def model_output() -> dict[str, mx.array]:
    """Forward pass with random weights, materialized once."""
    model = GreenFormer(img_size=IMG_SIZE)
    x = mx.random.normal((1, IMG_SIZE, IMG_SIZE, 4))
    out = model(x)
    # mx.eval is MLX array materialization, not Python eval()
    mx.eval(out)  # noqa: S307
    return out


def test_all_keys_present(model_output: dict[str, mx.array]) -> None:
    assert set(model_output.keys()) == set(EXPECTED_SHAPES.keys())


@pytest.mark.parametrize("key,expected_shape", EXPECTED_SHAPES.items())
def test_output_shapes(
    key: str,
    expected_shape: tuple[int, ...],
    model_output: dict[str, mx.array],
) -> None:
    assert model_output[key].shape == expected_shape, (
        f"{key}: expected {expected_shape}, got {model_output[key].shape}"
    )


def test_output_dtype_float32(model_output: dict[str, mx.array]) -> None:
    for key, arr in model_output.items():
        assert arr.dtype == mx.float32, f"{key}: expected float32, got {arr.dtype}"


def test_sigmoid_outputs_in_range(model_output: dict[str, mx.array]) -> None:
    """Post-sigmoid outputs must be in [0, 1]."""
    for key in ("alpha_coarse", "fg_coarse", "alpha_final", "fg_final"):
        arr = model_output[key]
        assert float(mx.min(arr)) >= 0.0, f"{key} has values < 0"
        assert float(mx.max(arr)) <= 1.0, f"{key} has values > 1"


# ---------------------------------------------------------------------------
# Backbone shape contract
# ---------------------------------------------------------------------------

BACKBONE_SHAPES = [
    (1, 128, 128, 112),  # stride 4
    (1, 64, 64, 224),  # stride 8
    (1, 32, 32, 448),  # stride 16
    (1, 16, 16, 896),  # stride 32
]


def test_patch_embed_output_shape() -> None:
    patch_embed = HieraPatchEmbed()
    x = mx.zeros((1, IMG_SIZE, IMG_SIZE, 4))
    out = patch_embed(x)
    expected_n = (IMG_SIZE // 4) * (IMG_SIZE // 4)
    assert out.shape == (1, expected_n, 112)


def test_backbone_returns_four_features() -> None:
    backbone = HieraBackbone(img_size=IMG_SIZE)
    x = mx.zeros((1, IMG_SIZE, IMG_SIZE, 4))
    features = backbone(x)
    assert len(features) == 4


@pytest.mark.parametrize("stage_idx", range(4))
def test_backbone_feature_shapes(stage_idx: int) -> None:
    backbone = HieraBackbone(img_size=IMG_SIZE)
    x = mx.zeros((1, IMG_SIZE, IMG_SIZE, 4))
    features = backbone(x)
    assert features[stage_idx].shape == BACKBONE_SHAPES[stage_idx]


# ---------------------------------------------------------------------------
# Full pipeline roundtrip (preprocess -> model -> postprocess)
# ---------------------------------------------------------------------------


def test_pipeline_roundtrip() -> None:
    """numpy -> preprocess -> model -> postprocess -> uint8."""
    from corridorkey_mlx.io.image import postprocess_alpha, postprocess_foreground, preprocess

    model = GreenFormer(img_size=SMALL_IMG_SIZE)
    rgb = np.random.rand(SMALL_IMG_SIZE, SMALL_IMG_SIZE, 3).astype(np.float32)
    alpha_hint = np.random.rand(SMALL_IMG_SIZE, SMALL_IMG_SIZE, 1).astype(np.float32)

    x = preprocess(rgb, alpha_hint)
    assert x.shape == (1, SMALL_IMG_SIZE, SMALL_IMG_SIZE, 4)

    out = model(x)
    mx.eval(out)  # noqa: S307

    alpha = postprocess_alpha(out["alpha_final"])
    fg = postprocess_foreground(out["fg_final"])

    assert alpha.shape == (SMALL_IMG_SIZE, SMALL_IMG_SIZE)
    assert alpha.dtype == np.uint8
    assert fg.shape == (SMALL_IMG_SIZE, SMALL_IMG_SIZE, 3)
    assert fg.dtype == np.uint8


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_deterministic_output() -> None:
    """Same input -> same output."""
    model = GreenFormer(img_size=SMALL_IMG_SIZE)
    mx.random.seed(123)
    x = mx.random.normal((1, SMALL_IMG_SIZE, SMALL_IMG_SIZE, 4))
    mx.eval(x)  # noqa: S307

    out1 = model(x)
    mx.eval(out1)  # noqa: S307
    out2 = model(x)
    mx.eval(out2)  # noqa: S307

    for key in out1:
        np.testing.assert_array_equal(
            np.array(out1[key]), np.array(out2[key]), err_msg=f"{key} not deterministic"
        )


# ---------------------------------------------------------------------------
# No NaN/Inf (wiring check)
# ---------------------------------------------------------------------------


def test_no_nan_inf() -> None:
    """Random-weight forward produces finite outputs."""
    model = GreenFormer(img_size=SMALL_IMG_SIZE)
    x = mx.random.normal((1, SMALL_IMG_SIZE, SMALL_IMG_SIZE, 4))
    mx.eval(x)  # noqa: S307
    out = model(x)
    mx.eval(out)  # noqa: S307

    for key in ("alpha_final", "fg_final"):
        arr = np.array(out[key])
        assert not np.isnan(arr).any(), f"{key} contains NaN"
        assert not np.isinf(arr).any(), f"{key} contains Inf"
