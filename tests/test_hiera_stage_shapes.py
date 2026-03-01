"""Shape contract tests for Hiera backbone (Phase 4).

No checkpoint needed — verifies structural correctness with random weights.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from corridorkey_mlx.model.hiera import HieraBackbone, HieraPatchEmbed

IMG_SIZE = 512
NUM_FEATURES = 4
EXPECTED_SHAPES = [
    (1, 128, 128, 112),  # stride 4
    (1, 64, 64, 224),  # stride 8
    (1, 32, 32, 448),  # stride 16
    (1, 16, 16, 896),  # stride 32
]


def test_patch_embed_output_shape() -> None:
    """PatchEmbed produces (B, N, C) with N = (H/4)*(W/4)."""
    patch_embed = HieraPatchEmbed()
    x = mx.zeros((1, IMG_SIZE, IMG_SIZE, 4))
    out = patch_embed(x)
    expected_n = (IMG_SIZE // 4) * (IMG_SIZE // 4)
    assert out.shape == (1, expected_n, 112)


def test_backbone_returns_four_features() -> None:
    """Backbone returns exactly 4 feature maps."""
    backbone = HieraBackbone(img_size=IMG_SIZE)
    x = mx.zeros((1, IMG_SIZE, IMG_SIZE, 4))
    features = backbone(x)
    assert len(features) == NUM_FEATURES


@pytest.mark.parametrize("stage_idx", range(NUM_FEATURES))
def test_feature_map_shape(stage_idx: int) -> None:
    """Each stage feature map has correct (B, H, W, C) shape."""
    backbone = HieraBackbone(img_size=IMG_SIZE)
    x = mx.zeros((1, IMG_SIZE, IMG_SIZE, 4))
    features = backbone(x)
    assert features[stage_idx].shape == EXPECTED_SHAPES[stage_idx]
