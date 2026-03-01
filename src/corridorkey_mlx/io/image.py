"""Image loading, saving, and preprocessing for CorridorKey inference.

All preprocessing produces NHWC tensors suitable for the MLX model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from pathlib import Path

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_image(path: str | Path) -> np.ndarray:
    """Load image as RGB float32 array in [0, 1] range, shape (H, W, 3)."""
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


def load_alpha_hint(path: str | Path) -> np.ndarray:
    """Load alpha hint as grayscale float32 array in [0, 1], shape (H, W, 1)."""
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float32)[:, :, np.newaxis] / 255.0


def normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    """Apply ImageNet normalization to (H, W, 3) float32 RGB in [0, 1]."""
    return (rgb - IMAGENET_MEAN) / IMAGENET_STD


def preprocess(
    rgb: np.ndarray,
    alpha_hint: np.ndarray,
) -> mx.array:
    """Build 4-channel NHWC input tensor from RGB and alpha hint.

    Args:
        rgb: (H, W, 3) float32 in [0, 1]
        alpha_hint: (H, W, 1) float32 in [0, 1]

    Returns:
        (1, H, W, 4) mx.array — ImageNet-normalized RGB + raw alpha hint.
    """
    normalized_rgb = normalize_rgb(rgb)
    combined = np.concatenate([normalized_rgb, alpha_hint], axis=-1)  # (H, W, 4)
    return mx.array(combined[np.newaxis])  # (1, H, W, 4)


def postprocess_alpha(alpha: mx.array) -> np.ndarray:
    """Convert model alpha output to uint8 numpy array.

    Args:
        alpha: (1, H, W, 1) probabilities in [0, 1]

    Returns:
        (H, W) uint8 array.
    """
    arr = np.array(alpha[0, :, :, 0])
    return (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)


def postprocess_foreground(fg: mx.array) -> np.ndarray:
    """Convert model foreground output to uint8 numpy array.

    Args:
        fg: (1, H, W, 3) probabilities in [0, 1]

    Returns:
        (H, W, 3) uint8 array.
    """
    arr = np.array(fg[0])
    return (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)


def save_alpha(alpha: np.ndarray, path: str | Path) -> None:
    """Save alpha matte as grayscale PNG."""
    Image.fromarray(alpha, mode="L").save(path)


def save_foreground(fg: np.ndarray, path: str | Path) -> None:
    """Save foreground as RGB PNG."""
    Image.fromarray(fg, mode="RGB").save(path)
