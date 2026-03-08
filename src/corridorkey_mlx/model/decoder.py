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
        fused = self.linear_fuse(fused)
        fused = self.bn(fused)
        fused = nn.relu(fused)
        # No dropout at inference (eval mode)
        return self.classifier(fused)

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
        fused = self.linear_fuse(fused)
        fused = self.bn(fused)
        fused = nn.relu(fused)
        return self.classifier(fused)


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

    def __call__(
        self, features: list[mx.array]
    ) -> tuple[mx.array, mx.array]:
        """Forward pass with batched upsampling.

        Returns:
            (alpha_logits, fg_logits) both in NHWC.
        """
        alpha_projs = self.alpha_head._project_features(features)
        fg_projs = self.fg_head._project_features(features)

        alpha_up = []
        fg_up = []
        for a_proj, f_proj, up in zip(
            alpha_projs, fg_projs, self._upsamplers, strict=True
        ):
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
