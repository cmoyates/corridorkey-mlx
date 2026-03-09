"""GPU-side preprocessing for full-frame inference.

Moves ImageNet normalization and channel concatenation to MLX while
keeping PIL bicubic resize on CPU (MLX cubic kernel differs too much).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx

if TYPE_CHECKING:
    import numpy as np

# ImageNet normalization constants (GPU-resident)
_IMAGENET_MEAN = mx.array([0.485, 0.456, 0.406])
_IMAGENET_STD = mx.array([0.229, 0.224, 0.225])


def preprocess_mlx(rgb_f32: np.ndarray, mask_f32: np.ndarray) -> mx.array:
    """GPU-side normalize + concat for full-frame inference.

    Assumes resize already happened on CPU via PIL bicubic.

    Args:
        rgb_f32: (H, W, 3) float32 in [0, 1], already resized.
        mask_f32: (H, W, 1) float32 in [0, 1], already resized.

    Returns:
        (1, H, W, 4) mx.array — ImageNet-normalized RGB + raw alpha hint.
    """
    img = mx.array(rgb_f32)
    mask = mx.array(mask_f32)

    # ImageNet normalize on GPU
    img = (img - _IMAGENET_MEAN) / _IMAGENET_STD

    # Concat and add batch dim
    combined = mx.concatenate([img, mask], axis=-1)  # (H, W, 4)
    return mx.expand_dims(combined, axis=0)  # (1, H, W, 4)
