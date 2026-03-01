"""Tensor layout conversion utilities.

Centralizes NCHW <-> NHWC transforms and conv weight transposes.
All layout conversions in the project go through this module.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np


def nchw_to_nhwc(x: mx.array) -> mx.array:
    """Convert (N, C, H, W) -> (N, H, W, C)."""
    return mx.transpose(x, axes=(0, 2, 3, 1))


def nhwc_to_nchw(x: mx.array) -> mx.array:
    """Convert (N, H, W, C) -> (N, C, H, W)."""
    return mx.transpose(x, axes=(0, 3, 1, 2))


def conv_weight_pt_to_mlx(w: np.ndarray) -> np.ndarray:
    """Transpose PyTorch conv weight (O, I, H, W) -> MLX conv weight (O, H, W, I)."""
    return np.transpose(w, axes=(0, 2, 3, 1))


def nchw_to_nhwc_np(x: np.ndarray) -> np.ndarray:
    """Convert numpy (N, C, H, W) -> (N, H, W, C)."""
    return np.transpose(x, axes=(0, 2, 3, 1))


def nhwc_to_nchw_np(x: np.ndarray) -> np.ndarray:
    """Convert numpy (N, H, W, C) -> (N, C, H, W)."""
    return np.transpose(x, axes=(0, 3, 1, 2))
