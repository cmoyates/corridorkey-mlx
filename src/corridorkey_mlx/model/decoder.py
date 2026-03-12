"""Decoder heads — MLX port.

Two heads: alpha (1ch) and foreground (3ch).
Consume multiscale backbone features (NHWC), upsample and fuse to produce predictions.

Architecture mirrors nikopueringer/CorridorKey DecoderHead (SegFormer-style).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn

if TYPE_CHECKING:
    from collections.abc import Sequence


class MLP(nn.Module):
    """Single linear projection: input_dim -> embed_dim."""

    def __init__(self, input_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.proj(x)


class DecoderHead(nn.Module):
    """SegFormer-style multiscale feature fusion head (NHWC).

    Takes 4 multi-scale feature maps in NHWC format, projects each to embed_dim,
    upsamples to the largest spatial resolution, fuses, and classifies.
    """

    def __init__(
        self,
        in_channels: list[int],
        embed_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.linear_c1 = MLP(in_channels[0], embed_dim)
        self.linear_c2 = MLP(in_channels[1], embed_dim)
        self.linear_c3 = MLP(in_channels[2], embed_dim)
        self.linear_c4 = MLP(in_channels[3], embed_dim)

        # Pre-build upsamplers for feature maps at strides 2x, 4x, 8x
        # relative to the first (stride-4) feature map.
        self._upsampler_2x = nn.Upsample(
            scale_factor=(2.0, 2.0), mode="linear", align_corners=False
        )
        self._upsampler_4x = nn.Upsample(
            scale_factor=(4.0, 4.0), mode="linear", align_corners=False
        )
        self._upsampler_8x = nn.Upsample(
            scale_factor=(8.0, 8.0), mode="linear", align_corners=False
        )

        fused_channels = embed_dim * len(in_channels)
        self.linear_fuse = nn.Conv2d(fused_channels, embed_dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm(embed_dim)
        self.classifier = nn.Conv2d(embed_dim, output_dim, kernel_size=1)

        # Folded BN params (precomputed after weight loading)
        self._bn_folded = False
        self._bn_scale: mx.array | None = None
        self._bn_offset: mx.array | None = None

    def fold_bn(self) -> None:
        """Fold BatchNorm into precomputed scale+offset for inference.

        Reduces BN from 5 element-wise ops (sub, rsqrt, mul, mul, add) to 2
        (mul, add), simplifying the compiled Metal kernel graph.
        Also precomputes 2D weights for 1x1 Conv2d layers to bypass conv2d
        dispatch and enable mx.addmm fusion for bias addition.
        Must be called after weights are loaded and model is in eval mode.
        """
        eps = self.bn.eps
        inv_std = mx.rsqrt(self.bn.running_var + eps)
        self._bn_scale = self.bn.weight * inv_std
        self._bn_offset = self.bn.bias - self.bn.running_mean * self._bn_scale

        # Precompute 2D weights from 1x1 Conv2d layers: (C_out, 1, 1, C_in) -> (C_out, C_in)
        # Avoids mx.conv2d dispatch overhead; classifier uses mx.addmm for fused bias+matmul
        c_out_f, _, _, c_in_f = self.linear_fuse.weight.shape
        self._fuse_weight_2d = self.linear_fuse.weight.reshape(c_out_f, c_in_f)
        c_out_c, _, _, c_in_c = self.classifier.weight.shape
        self._classifier_weight_2d = self.classifier.weight.reshape(c_out_c, c_in_c)

        # materialize folded params — mx.eval is MLX array materialization
        mx.eval(  # noqa: S307
            self._bn_scale, self._bn_offset,
            self._fuse_weight_2d, self._classifier_weight_2d,
        )
        self._bn_folded = True

    def _apply_bn(self, x: mx.array) -> mx.array:
        """Apply BatchNorm — folded (2 ops) or standard (5 ops)."""
        if self._bn_folded:
            return x * self._bn_scale + self._bn_offset
        return self.bn(x)

    def __call__(self, features: list[mx.array]) -> mx.array:
        """Forward pass.

        Args:
            features: 4 feature maps in NHWC format.
                      [0]: (B, H/4,  W/4,  C1)
                      [1]: (B, H/8,  W/8,  C2)
                      [2]: (B, H/16, W/16, C3)
                      [3]: (B, H/32, W/32, C4)

        Returns:
            Logits in NHWC: (B, H/4, W/4, output_dim)
        """
        c1, c2, c3, c4 = features
        upsamplers = [None, self._upsampler_2x, self._upsampler_4x, self._upsampler_8x]

        projected = []
        for feat, linear, up in zip(
            [c1, c2, c3, c4],
            [self.linear_c1, self.linear_c2, self.linear_c3, self.linear_c4],
            upsamplers,
            strict=True,
        ):
            b, h, w, _c = feat.shape
            # NHWC: reshape to (B, H*W, C), project, reshape back
            x = feat.reshape(b, h * w, _c)
            x = linear(x)  # (B, H*W, embed_dim)
            x = x.reshape(b, h, w, -1)  # (B, H, W, embed_dim)
            if up is not None:
                x = up(x)
            projected.append(x)

        # Concatenate in c4, c3, c2, c1 order to match trained weight layout
        fused = mx.concatenate(projected[::-1], axis=-1)  # (B, H/4, W/4, embed_dim*4)
        # 1x1 conv as matmul (bypass mx.conv2d dispatch); linear_fuse has no bias
        fused = fused @ self._fuse_weight_2d.T
        fused = self._apply_bn(fused)
        fused = nn.relu(fused)
        # 1x1 conv as addmm (fused bias+matmul); classifier has bias
        return mx.addmm(self.classifier.bias, fused, self._classifier_weight_2d.T)

    def _project_features(self, features: list[mx.array]) -> list[mx.array]:
        """Project features without upsampling or fusion."""
        linears = [self.linear_c1, self.linear_c2, self.linear_c3, self.linear_c4]
        projected = []
        for feat, linear in zip(features, linears, strict=True):
            b, h, w, _c = feat.shape
            x = feat.reshape(b, h * w, _c)
            x = linear(x)
            x = x.reshape(b, h, w, -1)
            projected.append(x)
        return projected

    def _fuse_and_classify(self, projected: list[mx.array]) -> mx.array:
        """Fuse pre-upsampled projections and classify."""
        fused = mx.concatenate(projected[::-1], axis=-1)
        fused = fused @ self._fuse_weight_2d.T
        fused = self._apply_bn(fused)
        fused = nn.relu(fused)
        return mx.addmm(self.classifier.bias, fused, self._classifier_weight_2d.T)


class FusedDecoderPair(nn.Module):
    """Runs two DecoderHeads with batched upsampling.

    Concatenates alpha+fg projections along channel axis before each upsample,
    reducing 6 Metal dispatch calls to 3.
    """

    def __init__(self, alpha_head: DecoderHead, fg_head: DecoderHead) -> None:
        super().__init__()
        self.alpha_head = alpha_head
        self.fg_head = fg_head
        # Reuse alpha_head's pre-allocated upsamplers
        self._upsamplers: Sequence[nn.Upsample | None] = [
            None,
            alpha_head._upsampler_2x,
            alpha_head._upsampler_4x,
            alpha_head._upsampler_8x,
        ]

    def __call__(self, features: list[mx.array]) -> tuple[mx.array, mx.array]:
        """Forward pass with batched upsampling.

        Returns:
            (alpha_logits, fg_logits) both in NHWC.
        """
        alpha_projs = self.alpha_head._project_features(features)
        fg_projs = self.fg_head._project_features(features)

        alpha_up = []
        fg_up = []
        for a_proj, f_proj, up in zip(alpha_projs, fg_projs, self._upsamplers, strict=True):
            if up is not None:
                # Batch upsample: concat along channel axis (NHWC)
                fused = mx.concatenate([a_proj, f_proj], axis=-1)
                fused = up(fused)
                embed_dim = a_proj.shape[-1]
                a_proj = fused[:, :, :, :embed_dim]
                f_proj = fused[:, :, :, embed_dim:]
            alpha_up.append(a_proj)
            fg_up.append(f_proj)

        alpha_logits = self.alpha_head._fuse_and_classify(alpha_up)
        fg_logits = self.fg_head._fuse_and_classify(fg_up)
        return alpha_logits, fg_logits
