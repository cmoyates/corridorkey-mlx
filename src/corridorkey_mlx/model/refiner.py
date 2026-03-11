"""CNN refiner — MLX port.

Input: RGB + coarse predictions (7ch total) in NHWC.
Output: additive delta logits (4ch) in NHWC.

Architecture mirrors nikopueringer/CorridorKey CNNRefinerModule.
"""

from __future__ import annotations

import numpy as np

import mlx.core as mx
import mlx.nn as nn

REFINER_CHANNELS = 64
REFINER_GROUPS = 8
REFINER_SCALE = 10.0

# Dilations for refiner residual blocks (indexed by block number 1-4)
REFINER_DILATIONS = {1: 1, 2: 2, 3: 4, 4: 8}


def inflate_dilated_kernel(weight: mx.array, dilation: int) -> mx.array:
    """Inflate a (O, kH, kW, I) conv kernel by inserting zeros to eliminate dilation.

    A 3x3 kernel with dilation d is mathematically equivalent to a
    (2d+1)x(2d+1) kernel with dilation 1, where the original weights sit at
    strided positions and zeros fill the gaps.

    This lets MLX dispatch the conv via implicit GEMM (no im2col fallback),
    trading extra FLOPs for dramatically less memory bandwidth.
    """
    if dilation <= 1:
        return weight
    w = np.array(weight)
    out_ch, kh, kw, in_ch = w.shape
    eff_h = dilation * (kh - 1) + 1
    eff_w = dilation * (kw - 1) + 1
    inflated = np.zeros((out_ch, eff_h, eff_w, in_ch), dtype=w.dtype)
    inflated[:, ::dilation, ::dilation, :] = w
    return mx.array(inflated)


class RefinerBlock(nn.Module):
    """Dilated residual block with GroupNorm (NHWC).

    When inflate=True and dilation>1, replaces dilated conv with an
    equivalently-sized standard conv (no dilation parameter). This avoids
    the im2col fallback in MLX and enables implicit GEMM dispatch.
    """

    def __init__(self, channels: int, dilation: int, inflate: bool = False) -> None:
        super().__init__()
        if inflate and dilation > 1:
            # Inflated kernel: effective size = dilation*(3-1)+1 = 2*dilation+1
            effective_kernel = dilation * 2 + 1
            self.conv1 = nn.Conv2d(
                channels, channels, kernel_size=effective_kernel, padding=dilation
            )
            self.conv2 = nn.Conv2d(
                channels, channels, kernel_size=effective_kernel, padding=dilation
            )
        else:
            self.conv1 = nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            )
            self.conv2 = nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            )
        self.gn1 = nn.GroupNorm(REFINER_GROUPS, channels, pytorch_compatible=True)
        self.gn2 = nn.GroupNorm(REFINER_GROUPS, channels, pytorch_compatible=True)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        out = nn.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return nn.relu(out + residual)


class CNNRefinerModule(nn.Module):
    """CNN refiner: stem + 4 dilated ResBlocks + 1x1 projection (NHWC).

    Takes RGB (3ch) and coarse predictions (4ch) concatenated along channel dim.
    Returns delta logits (4ch) scaled by 10x.
    """

    def __init__(self, inflate_dilated: bool = True) -> None:
        super().__init__()
        refiner_input_channels = 7  # RGB (3) + coarse_pred (4: alpha + fg)
        self.stem_conv = nn.Conv2d(
            refiner_input_channels, REFINER_CHANNELS, kernel_size=3, padding=1
        )
        self.stem_gn = nn.GroupNorm(REFINER_GROUPS, REFINER_CHANNELS, pytorch_compatible=True)

        self.res1 = RefinerBlock(REFINER_CHANNELS, dilation=1, inflate=inflate_dilated)
        self.res2 = RefinerBlock(REFINER_CHANNELS, dilation=2, inflate=inflate_dilated)
        self.res3 = RefinerBlock(REFINER_CHANNELS, dilation=4, inflate=inflate_dilated)
        self.res4 = RefinerBlock(REFINER_CHANNELS, dilation=8, inflate=inflate_dilated)

        refiner_output_channels = 4  # delta for alpha (1) + delta for fg (3)
        self.final = nn.Conv2d(REFINER_CHANNELS, refiner_output_channels, kernel_size=1)

    def __call__(self, rgb: mx.array, coarse_pred: mx.array) -> mx.array:
        """Forward pass.

        Args:
            rgb: (B, H, W, 3) — RGB input in NHWC
            coarse_pred: (B, H, W, 4) — concatenated alpha_coarse + fg_coarse in NHWC

        Returns:
            Delta logits: (B, H, W, 4) in NHWC, scaled by 10x
        """
        x = mx.concatenate([rgb, coarse_pred], axis=-1)  # (B, H, W, 7)
        x = nn.relu(self.stem_gn(self.stem_conv(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        return self.final(x) * REFINER_SCALE
