"""Top-level CorridorKey model (GreenFormer) — MLX port.

Composes Hiera backbone + dual decoder heads + CNN refiner.
All internal operations use NHWC layout.
"""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn
from safetensors import safe_open

from corridorkey_mlx.model.backbone import HieraBackbone
from corridorkey_mlx.model.decoder import DecoderHead, FusedDecoderPair
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
        self,
        img_size: int = 512,
        dtype: mx.Dtype = mx.float32,
        fused_decode: bool = False,
        slim: bool = False,
        use_sdpa: bool = True,
        stage_gc: bool = True,
    ) -> None:
        super().__init__()
        self._compute_dtype = dtype
        self._fused_decode = fused_decode
        self._slim = slim
        self._stage_gc = stage_gc
        self._compiled = False
        self.backbone = HieraBackbone(img_size=img_size, use_sdpa=use_sdpa)
        self.alpha_decoder = DecoderHead(BACKBONE_CHANNELS, EMBED_DIM, output_dim=1)
        self.fg_decoder = DecoderHead(BACKBONE_CHANNELS, EMBED_DIM, output_dim=3)
        self.refiner = CNNRefinerModule()

        if fused_decode:
            self._fused_pair = FusedDecoderPair(self.alpha_decoder, self.fg_decoder)

        # Decoder outputs at stride-4 (H/4, W/4); upsampler is always 4x
        self._logit_upsampler = nn.Upsample(
            scale_factor=(4.0, 4.0), mode="linear", align_corners=False
        )

    def __call__(self, x: mx.array) -> dict[str, mx.array]:
        """Forward pass.

        Args:
            x: (B, H, W, 4) NHWC — ImageNet-normalized RGB + alpha hint.

        Returns:
            Dict with keys: alpha_logits, fg_logits, alpha_logits_up, fg_logits_up,
            alpha_coarse, fg_coarse, delta_logits, alpha_final, fg_final.
            All tensors in NHWC format.
        """
        # Backbone always runs in fp32
        features = self.backbone(x)

        # Materialize backbone output so MLX can free intermediate graph nodes.
        # NOTE: mx.eval is MLX array materialization, not Python eval()
        if self._stage_gc and not self._compiled:
            mx.eval(features)  # noqa: S307
            gc.collect()
            mx.clear_cache()

        # Cast features to compute dtype for decoders (bf16 saves memory)
        if self._compute_dtype != mx.float32:
            features = [f.astype(self._compute_dtype) for f in features]

        # Decoder heads -> logits at H/4 resolution
        if self._fused_decode:
            alpha_logits, fg_logits = self._fused_pair(features)
        else:
            alpha_logits = self.alpha_decoder(features)  # (B, H/4, W/4, 1)
            fg_logits = self.fg_decoder(features)  # (B, H/4, W/4, 3)

        # Free backbone feature maps — decoders consumed them,
        # refiner uses rgb+coarse_pred only
        del features

        # Upsample logits to full input resolution (4x from stride-4 decoder)
        alpha_logits_up = self._logit_upsampler(alpha_logits)  # (B, H, W, 1)
        fg_logits_up = self._logit_upsampler(fg_logits)  # (B, H, W, 3)

        # Cast back to fp32 before sigmoid (precision at saturation boundaries)
        if self._compute_dtype != mx.float32:
            alpha_logits_up = alpha_logits_up.astype(mx.float32)
            fg_logits_up = fg_logits_up.astype(mx.float32)

        # Coarse predictions via sigmoid (always fp32)
        alpha_coarse = mx.sigmoid(alpha_logits_up)
        fg_coarse = mx.sigmoid(fg_logits_up)

        # Materialize decoder output so MLX can free decoder graph nodes.
        # NOTE: mx.eval is MLX array materialization, not Python eval()
        if self._stage_gc and not self._compiled:
            mx.eval(alpha_coarse, fg_coarse, alpha_logits_up, fg_logits_up)  # noqa: S307
            gc.collect()
            mx.clear_cache()

        # Refiner receives fp32 coarse predictions
        rgb = x[:, :, :, :3]  # (B, H, W, 3)
        coarse_pred = mx.concatenate([alpha_coarse, fg_coarse], axis=-1)  # (B, H, W, 4)
        delta_logits = self.refiner(rgb, coarse_pred)  # (B, H, W, 4)

        # Final predictions: additive residual in logit space, then sigmoid
        alpha_final = mx.sigmoid(alpha_logits_up + delta_logits[:, :, :, 0:1])
        fg_final = mx.sigmoid(fg_logits_up + delta_logits[:, :, :, 1:4])

        if self._slim:
            return {
                "alpha_coarse": alpha_coarse,
                "fg_coarse": fg_coarse,
                "alpha_final": alpha_final,
                "fg_final": fg_final,
            }

        return {
            "alpha_logits": alpha_logits.astype(mx.float32),
            "fg_logits": fg_logits.astype(mx.float32),
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
