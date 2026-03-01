"""Decoder heads — MLX port.

Two heads: alpha (1ch) and foreground (3ch).
Consume multiscale backbone features (NHWC), upsample and fuse to produce predictions.

Architecture mirrors nikopueringer/CorridorKey DecoderHead (SegFormer-style).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


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
        target_h, target_w = c1.shape[1], c1.shape[2]  # H/4, W/4

        projected = []
        for feat, linear in zip(
            [c1, c2, c3, c4],
            [self.linear_c1, self.linear_c2, self.linear_c3, self.linear_c4],
            strict=True,
        ):
            b, h, w, _c = feat.shape
            # NHWC: reshape to (B, H*W, C), project, reshape back
            x = feat.reshape(b, h * w, _c)
            x = linear(x)  # (B, H*W, embed_dim)
            x = x.reshape(b, h, w, -1)  # (B, H, W, embed_dim)
            # Upsample to target spatial size
            if h != target_h or w != target_w:
                scale_h = target_h / h
                scale_w = target_w / w
                upsampler = nn.Upsample(
                    scale_factor=(scale_h, scale_w),
                    mode="linear",
                    align_corners=False,
                )
                x = upsampler(x)
            projected.append(x)

        # Concatenate along channel dim (last dim in NHWC)
        fused = mx.concatenate(projected, axis=-1)  # (B, H/4, W/4, embed_dim*4)
        fused = self.linear_fuse(fused)
        fused = self.bn(fused)
        fused = nn.relu(fused)
        # No dropout at inference (eval mode)
        return self.classifier(fused)
