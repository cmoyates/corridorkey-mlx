"""Top-level CorridorKey model (GreenFormer) — MLX port.

Composes Hiera backbone + dual decoder heads + CNN refiner.
All internal operations use NHWC layout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn
from safetensors import safe_open

from corridorkey_mlx.model.backbone import HieraBackbone
from corridorkey_mlx.model.decoder import DecoderHead
from corridorkey_mlx.model.hiera import ENCODER_KEY_PREFIX, _interpolate_pos_embed, _prod
from corridorkey_mlx.model.refiner import CNNRefinerModule

if TYPE_CHECKING:
    from pathlib import Path

BACKBONE_CHANNELS = [112, 224, 448, 896]
EMBED_DIM = 256


class GreenFormer(nn.Module):
    """CorridorKey: Hiera encoder + dual decoder heads + CNN refiner.

    Input:  (B, H, W, 4) NHWC — ImageNet-normalized RGB + alpha hint [0,1]
    Output: dict with coarse/final alpha and foreground maps in NHWC.
    """

    def __init__(
        self, img_size: int = 512, backbone_size: int | None = None
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.backbone_size = backbone_size

        # Backbone runs at backbone_size if set, else img_size
        backbone_res = backbone_size if backbone_size is not None else img_size
        self._decoupled = backbone_res != img_size

        self.backbone = HieraBackbone(img_size=backbone_res)
        self.alpha_decoder = DecoderHead(BACKBONE_CHANNELS, EMBED_DIM, output_dim=1)
        self.fg_decoder = DecoderHead(BACKBONE_CHANNELS, EMBED_DIM, output_dim=3)
        self.refiner = CNNRefinerModule()

        # Decoder outputs at stride-4 (H/4, W/4); upsampler is always 4x
        self._logit_upsampler = nn.Upsample(
            scale_factor=(4.0, 4.0), mode="linear", align_corners=False
        )

        # Resamplers for decoupled resolution (backbone_size != img_size)
        if self._decoupled:
            down_scale = backbone_res / img_size
            self._backbone_downsampler = nn.Upsample(
                scale_factor=(down_scale, down_scale),
                mode="linear",
                align_corners=False,
            )
            # Upsample coarse logits from backbone_size back to img_size
            up_scale = img_size / backbone_res
            self._fullres_upsampler = nn.Upsample(
                scale_factor=(up_scale, up_scale),
                mode="linear",
                align_corners=False,
            )

    def __call__(self, x: mx.array, *, slim: bool = False) -> dict[str, mx.array]:
        """Forward pass.

        Args:
            x: (B, H, W, 4) NHWC — ImageNet-normalized RGB + alpha hint.
            slim: If True, only return the 4 engine-relevant outputs
                (alpha_coarse, fg_coarse, alpha_final, fg_final), skipping
                5 intermediate tensors to save VRAM.

        Returns:
            Dict with coarse/final alpha and foreground maps in NHWC.
            When slim=False (default), also includes intermediate logits.
        """
        # Keep full-res input for refiner when resolutions are decoupled
        x_full = x

        # Downsample to backbone resolution if decoupled
        if self._decoupled:
            x = self._backbone_downsampler(x)

        # Backbone -> 4 multiscale feature maps in NHWC
        features = self.backbone(x)

        # Decoder heads -> logits at backbone_H/4 resolution
        alpha_logits = self.alpha_decoder(features)  # (B, bH/4, bW/4, 1)
        fg_logits = self.fg_decoder(features)  # (B, bH/4, bW/4, 3)

        # Upsample logits to backbone resolution (4x from stride-4 decoder)
        alpha_logits_up = self._logit_upsampler(alpha_logits)  # (B, bH, bW, 1)
        fg_logits_up = self._logit_upsampler(fg_logits)  # (B, bH, bW, 3)

        # If decoupled, upsample coarse logits from backbone_size to full res
        if self._decoupled:
            alpha_logits_up = self._fullres_upsampler(alpha_logits_up)
            fg_logits_up = self._fullres_upsampler(fg_logits_up)

        # Coarse predictions via sigmoid
        alpha_coarse = mx.sigmoid(alpha_logits_up)
        fg_coarse = mx.sigmoid(fg_logits_up)

        # Refiner: full-res RGB + coarse predictions -> delta logits
        rgb = x_full[:, :, :, :3]  # (B, H, W, 3)
        coarse_pred = mx.concatenate([alpha_coarse, fg_coarse], axis=-1)  # (B, H, W, 4)
        delta_logits = self.refiner(rgb, coarse_pred)  # (B, H, W, 4)

        # Final predictions: additive residual in logit space, then sigmoid
        alpha_final = mx.sigmoid(alpha_logits_up + delta_logits[:, :, :, 0:1])
        fg_final = mx.sigmoid(fg_logits_up + delta_logits[:, :, :, 1:4])

        if slim:
            return {
                "alpha_coarse": alpha_coarse,
                "fg_coarse": fg_coarse,
                "alpha_final": alpha_final,
                "fg_final": fg_final,
            }

        return {
            "alpha_logits": alpha_logits,
            "fg_logits": fg_logits,
            "alpha_logits_up": alpha_logits_up,
            "fg_logits_up": fg_logits_up,
            "alpha_coarse": alpha_coarse,
            "fg_coarse": fg_coarse,
            "delta_logits": delta_logits,
            "alpha_final": alpha_final,
            "fg_final": fg_final,
        }

    def load_checkpoint(self, path: str | Path) -> None:
        """Load all weights from converted safetensors checkpoint.

        Handles:
        - Backbone keys (encoder.model.* prefix) with pos_embed interpolation
        - Decoder keys (alpha_decoder.*, fg_decoder.*)
        - Refiner keys (refiner.*)
        """
        target_tokens = _prod(self.backbone.tokens_spatial_shape)
        weight_pairs: list[tuple[str, mx.array]] = []

        with safe_open(str(path), framework="numpy") as f:
            for full_key in f.keys():  # noqa: SIM118
                tensor = mx.array(f.get_tensor(full_key))

                if full_key.startswith(ENCODER_KEY_PREFIX):
                    # Backbone: strip encoder.model. prefix
                    mlx_key = "backbone." + full_key[len(ENCODER_KEY_PREFIX) :]
                    if mlx_key == "backbone.pos_embed":
                        tensor = _interpolate_pos_embed(tensor, target_tokens)
                        # materialize interpolated embedding
                        mx.eval(tensor)
                else:
                    # Decoder/refiner: use key as-is
                    mlx_key = full_key

                weight_pairs.append((mlx_key, tensor))

        self.load_weights(weight_pairs)
        self.eval()
        # materialize all parameters
        mx.eval(self.parameters())
