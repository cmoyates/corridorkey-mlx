"""CNN refiner — MLX port.

Input: RGB + coarse predictions (7ch total) in NHWC.
Output: additive delta logits (4ch) in NHWC.

Architecture mirrors nikopueringer/CorridorKey CNNRefinerModule.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

REFINER_CHANNELS = 64
REFINER_GROUPS = 8
REFINER_SCALE = 10.0


class RefinerBlock(nn.Module):
    """Dilated residual block with GroupNorm (NHWC)."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.gn1 = nn.GroupNorm(REFINER_GROUPS, channels, pytorch_compatible=True)
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
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

    def __init__(self) -> None:
        super().__init__()
        refiner_input_channels = 7  # RGB (3) + coarse_pred (4: alpha + fg)
        self.stem_conv = nn.Conv2d(
            refiner_input_channels, REFINER_CHANNELS, kernel_size=3, padding=1
        )
        self.stem_gn = nn.GroupNorm(REFINER_GROUPS, REFINER_CHANNELS, pytorch_compatible=True)

        self.res1 = RefinerBlock(REFINER_CHANNELS, dilation=1)
        self.res2 = RefinerBlock(REFINER_CHANNELS, dilation=2)
        self.res3 = RefinerBlock(REFINER_CHANNELS, dilation=4)
        self.res4 = RefinerBlock(REFINER_CHANNELS, dilation=8)

        refiner_output_channels = 4  # delta for alpha (1) + delta for fg (3)
        self.final = nn.Conv2d(REFINER_CHANNELS, refiner_output_channels, kernel_size=1)
        # Precomputed 2D weight for 1x1 conv bypass (set in prepare_inference)
        self._final_weight_2d: mx.array | None = None

    def prepare_inference(self) -> None:
        """Precompute 2D weight for 1x1 final conv to bypass mx.conv2d dispatch.

        Uses mx.addmm for fused bias+matmul. Call after weights are loaded.
        """
        c_out, _, _, c_in = self.final.weight.shape
        self._final_weight_2d = self.final.weight.reshape(c_out, c_in)
        # materialize reshaped weight — mx.eval is MLX array materialization
        mx.eval(self._final_weight_2d)  # noqa: S307

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
        # 1x1 conv as addmm (fused bias+matmul, bypass conv2d dispatch)
        if self._final_weight_2d is not None:
            return mx.addmm(self.final.bias, x, self._final_weight_2d.T) * REFINER_SCALE
        return self.final(x) * REFINER_SCALE
