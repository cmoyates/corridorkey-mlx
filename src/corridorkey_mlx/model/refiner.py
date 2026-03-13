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


class FrozenGroupNorm(nn.Module):
    """GroupNorm with frozen-stats support for tiled inference.

    Three modes:
    - Normal: delegates to mx.fast.layer_norm (same perf as nn.GroupNorm)
    - Collecting: computes mean/var manually, saves stats, returns normal output
    - Frozen: uses pre-collected stats instead of computing from input
    """

    def __init__(self, num_groups: int, dims: int) -> None:
        super().__init__()
        self.num_groups = num_groups
        self.dims = dims
        self.eps = 1e-5
        self.weight = mx.ones((dims,))
        self.bias = mx.zeros((dims,))
        self._frozen_stats: tuple[mx.array, mx.array] | None = None
        self._collecting = False
        self._collected_stats: tuple[mx.array, mx.array] | None = None

    def __call__(self, x: mx.array) -> mx.array:
        if self._frozen_stats is None and not self._collecting:
            return self._fast_forward(x)
        return self._custom_forward(x)

    def _fast_forward(self, x: mx.array) -> mx.array:
        """Normal path — identical to nn.GroupNorm(pytorch_compatible=True)."""
        batch, *rest, dims = x.shape
        group_size = dims // self.num_groups
        x = x.reshape(batch, -1, self.num_groups, group_size)
        x = x.transpose(0, 2, 1, 3).reshape(batch, self.num_groups, -1)
        x = mx.fast.layer_norm(x, eps=self.eps, weight=None, bias=None)
        x = x.reshape(batch, self.num_groups, -1, group_size)
        x = x.transpose(0, 2, 1, 3).reshape(batch, *rest, dims)
        return x * self.weight + self.bias

    def _custom_forward(self, x: mx.array) -> mx.array:
        """Collecting/frozen path — manual mean/var computation."""
        batch, *rest, dims = x.shape
        group_size = dims // self.num_groups
        x = x.reshape(batch, -1, self.num_groups, group_size)
        x = x.transpose(0, 2, 1, 3).reshape(batch, self.num_groups, -1)

        if self._frozen_stats is not None:
            mean, var = self._frozen_stats
        else:
            # Compute stats in fp32 for numerical stability — at full 2048x2048
            # resolution, float16 variance overflows (33M elements per group)
            x_f32 = x.astype(mx.float32)
            mean = x_f32.mean(axis=-1, keepdims=True)
            var = x_f32.var(axis=-1, keepdims=True)
            if self._collecting:
                self._collected_stats = (mean, var)

        # Normalize in input dtype to preserve activation precision
        input_dtype = x.dtype
        x = (x - mean.astype(input_dtype)) * mx.rsqrt(var.astype(input_dtype) + self.eps)

        x = x.reshape(batch, self.num_groups, -1, group_size)
        x = x.transpose(0, 2, 1, 3).reshape(batch, *rest, dims)
        return x * self.weight + self.bias


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
        self.gn1 = FrozenGroupNorm(REFINER_GROUPS, channels)
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.gn2 = FrozenGroupNorm(REFINER_GROUPS, channels)

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
        self.stem_gn = FrozenGroupNorm(REFINER_GROUPS, REFINER_CHANNELS)

        self.res1 = RefinerBlock(REFINER_CHANNELS, dilation=1)
        self.res2 = RefinerBlock(REFINER_CHANNELS, dilation=2)
        self.res3 = RefinerBlock(REFINER_CHANNELS, dilation=4)
        self.res4 = RefinerBlock(REFINER_CHANNELS, dilation=8)

        refiner_output_channels = 4  # delta for alpha (1) + delta for fg (3)
        self.final = nn.Conv2d(REFINER_CHANNELS, refiner_output_channels, kernel_size=1)
        # Precomputed 2D weight for 1x1 conv bypass (set in prepare_inference)
        self._final_weight_2d: mx.array | None = None

    def _all_groupnorms(self) -> list[FrozenGroupNorm]:
        """Return all 9 FrozenGroupNorm instances in forward order."""
        return [
            self.stem_gn,
            self.res1.gn1,
            self.res1.gn2,
            self.res2.gn1,
            self.res2.gn2,
            self.res3.gn1,
            self.res3.gn2,
            self.res4.gn1,
            self.res4.gn2,
        ]

    def collect_groupnorm_stats(self, rgb: mx.array, coarse_pred: mx.array) -> None:
        """Full-image forward to collect GroupNorm statistics.

        Block-level mx.eval bounds peak memory to ~6GB per conv im2col.
        """
        for gn in self._all_groupnorms():
            gn._collecting = True
            gn._collected_stats = None

        x = mx.concatenate([rgb, coarse_pred], axis=-1)
        x = nn.relu(self.stem_gn(self.stem_conv(x)))
        # mx.eval: MLX array materialization (not Python eval)
        mx.eval(x)  # noqa: S307
        x = self.res1(x)
        mx.eval(x)  # noqa: S307
        x = self.res2(x)
        mx.eval(x)  # noqa: S307
        x = self.res3(x)
        mx.eval(x)  # noqa: S307
        x = self.res4(x)
        mx.eval(x)  # noqa: S307
        del x  # discard output — only stats matter

        # Materialize collected stats
        all_arrays: list[mx.array] = []
        for gn in self._all_groupnorms():
            assert gn._collected_stats is not None
            all_arrays.extend(gn._collected_stats)
            gn._collecting = False
        # mx.eval: MLX array materialization (not Python eval)
        mx.eval(*all_arrays)  # noqa: S307

    def freeze_groupnorm_stats(self) -> None:
        """Copy collected stats to frozen stats on each GroupNorm."""
        for gn in self._all_groupnorms():
            assert gn._collected_stats is not None
            gn._frozen_stats = gn._collected_stats

    def unfreeze_groupnorm_stats(self) -> None:
        """Clear frozen and collected stats on each GroupNorm."""
        for gn in self._all_groupnorms():
            gn._frozen_stats = None
            gn._collected_stats = None

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
