"""CNN refiner — MLX port.

Input: RGB + coarse predictions (7ch total) in NHWC.
Output: additive delta logits (4ch) in NHWC.

Architecture mirrors nikopueringer/CorridorKey CNNRefinerModule.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from corridorkey_mlx.utils.metal_groupnorm import metal_groupnorm

REFINER_CHANNELS = 64
REFINER_GROUPS = 8
REFINER_SCALE = 10.0


def _pixel_unshuffle_nhwc(x: mx.array, d: int) -> mx.array:
    """Space-to-depth for NHWC: (B,H,W,C) → (B,H//d,W//d,C*d²).

    Packs spatially distant pixels into contiguous channels so that a
    subsequent grouped conv can replace a dilated conv with contiguous
    memory access patterns.
    """
    B, H, W, C = x.shape
    x = x.reshape(B, H // d, d, W // d, d, C)
    x = x.transpose(0, 1, 3, 2, 4, 5)  # (B, H//d, W//d, d, d, C)
    return x.reshape(B, H // d, W // d, d * d * C)


def _pixel_shuffle_nhwc(x: mx.array, d: int) -> mx.array:
    """Depth-to-space for NHWC: (B,H//d,W//d,C*d²) → (B,H,W,C)."""
    B, Hd, Wd, Cdd = x.shape
    C = Cdd // (d * d)
    x = x.reshape(B, Hd, Wd, d, d, C)
    x = x.transpose(0, 1, 3, 2, 4, 5)  # (B, Hd, d, Wd, d, C)
    return x.reshape(B, Hd * d, Wd * d, C)


class FrozenGroupNorm(nn.Module):
    """GroupNorm with frozen-stats support for tiled inference.

    Three modes:
    - Normal: custom Metal kernel (no transposes, ~67% faster than nn.GroupNorm)
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
        if self._frozen_stats is not None:
            # Frozen path — Metal kernel with precomputed stats (no transposes)
            return metal_groupnorm(x, self.weight, self.bias, frozen_stats=self._frozen_stats)
        if self._collecting:
            return self._collecting_forward(x)
        # Normal path — Metal kernel computes its own stats
        return metal_groupnorm(x, self.weight, self.bias)

    def _collecting_forward(self, x: mx.array) -> mx.array:
        """Collecting path — compute mean/var, save stats, return normal output."""
        batch, *rest, dims = x.shape
        group_size = dims // self.num_groups
        x_grouped = x.reshape(batch, -1, self.num_groups, group_size)
        x_grouped = x_grouped.transpose(0, 2, 1, 3).reshape(batch, self.num_groups, -1)

        # Compute stats in fp32 for numerical stability
        x_f32 = x_grouped.astype(mx.float32)
        mean = x_f32.mean(axis=-1, keepdims=True)
        var = x_f32.var(axis=-1, keepdims=True)
        self._collected_stats = (mean, var)

        # Use Metal kernel for the actual normalization (fast path)
        return metal_groupnorm(x, self.weight, self.bias)


class RefinerBlock(nn.Module):
    """Dilated residual block with GroupNorm (NHWC).

    Supports optional Space-to-Depth (SPD) mode for dilated convolutions.
    SPD replaces scattered dilated memory access with contiguous grouped
    conv, improving cache utilization on Apple Silicon.
    """

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.dilation = dilation
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
        # SPD conv replacements (populated by prepare_spd)
        self._use_spd = False
        self._spd_conv1: nn.Conv2d | None = None
        self._spd_conv2: nn.Conv2d | None = None

    def prepare_spd(self) -> None:
        """Convert dilated convs to SPD + grouped standard conv.

        Mathematically lossless: dilated conv with dilation=d is equivalent
        to pixel_unshuffle(d) -> grouped conv(groups=d^2) -> pixel_shuffle(d).
        Weights are tiled d^2 times to replicate the same kernel per sub-image.
        """
        d = self.dilation
        if d <= 1:
            return
        d2 = d * d
        C = self.conv1.weight.shape[0]  # REFINER_CHANNELS

        # Create grouped conv layers (standard 3x3, no dilation)
        self._spd_conv1 = nn.Conv2d(
            C * d2, C * d2, kernel_size=3, padding=1, groups=d2, bias=True,
        )
        self._spd_conv2 = nn.Conv2d(
            C * d2, C * d2, kernel_size=3, padding=1, groups=d2, bias=True,
        )

        # Tile weights: each of d^2 groups uses identical original weights
        self._spd_conv1.weight = mx.tile(self.conv1.weight, (d2, 1, 1, 1))
        self._spd_conv1.bias = mx.tile(self.conv1.bias, (d2,))
        self._spd_conv2.weight = mx.tile(self.conv2.weight, (d2, 1, 1, 1))
        self._spd_conv2.bias = mx.tile(self.conv2.bias, (d2,))

        # Materialize tiled weights — mx.eval is MLX array materialization
        mx.eval(self._spd_conv1.weight, self._spd_conv1.bias,  # noqa: S307
                self._spd_conv2.weight, self._spd_conv2.bias)
        self._use_spd = True

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        if self._use_spd:
            d = self.dilation
            # Conv1: unshuffle -> grouped conv -> shuffle -> GroupNorm -> ReLU
            out = _pixel_unshuffle_nhwc(x, d)
            out = self._spd_conv1(out)
            out = _pixel_shuffle_nhwc(out, d)
            out = nn.relu(self.gn1(out))
            # Conv2: unshuffle -> grouped conv -> shuffle -> GroupNorm
            out = _pixel_unshuffle_nhwc(out, d)
            out = self._spd_conv2(out)
            out = _pixel_shuffle_nhwc(out, d)
            out = self.gn2(out)
        else:
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

    def prepare_inference(self, use_spd: bool = False) -> None:
        """Precompute optimized weights for inference.

        - 1x1 final conv -> addmm bypass
        - Dilated convs -> SPD grouped conv (if use_spd=True)
        """
        c_out, _, _, c_in = self.final.weight.shape
        self._final_weight_2d = self.final.weight.reshape(c_out, c_in)
        # materialize reshaped weight — mx.eval is MLX array materialization
        mx.eval(self._final_weight_2d)  # noqa: S307

        # Convert dilated convs to SPD + grouped conv for contiguous access
        if use_spd:
            self.res2.prepare_spd()  # dilation=2
            self.res3.prepare_spd()  # dilation=4
            self.res4.prepare_spd()  # dilation=8

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
