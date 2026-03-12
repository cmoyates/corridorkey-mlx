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
        refiner_dtype: mx.Dtype | None = mx.bfloat16,
        compile_refiner: bool = True,
        compile_decoders: bool = True,
        compile_backbone: bool = True,
        compile_forward: bool = False,
        refiner_tile_size: int | None = 512,
    ) -> None:
        super().__init__()
        self._compute_dtype = dtype
        self._fused_decode = fused_decode
        self._slim = slim
        self._stage_gc = stage_gc
        self._compiled = False
        self._refiner_dtype = refiner_dtype
        self._compile_refiner = compile_refiner
        self._compile_decoders = compile_decoders
        self._compile_backbone = compile_backbone
        self._compile_forward = compile_forward
        self._refiner_tile_size = refiner_tile_size
        self.backbone = HieraBackbone(img_size=img_size, use_sdpa=use_sdpa)
        self.alpha_decoder = DecoderHead(BACKBONE_CHANNELS, EMBED_DIM, output_dim=1)
        self.fg_decoder = DecoderHead(BACKBONE_CHANNELS, EMBED_DIM, output_dim=3)
        self.refiner = CNNRefinerModule()
        self._compiled_refiner_call = None
        self._compiled_alpha_decoder_call = None
        self._compiled_fg_decoder_call = None
        self._compiled_fused_pair_call = None
        self._compiled_backbone_call = None

        # Secondary GPU stream for parallel decoder dispatch
        self._fg_stream = mx.new_stream(mx.gpu)

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
        backbone_fn = self._compiled_backbone_call or self.backbone
        features = backbone_fn(x)

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
            fused_fn = self._compiled_fused_pair_call or self._fused_pair
            alpha_logits, fg_logits = fused_fn(features)
            del features

            # Sequential upsample + sigmoid (fused path has no stream benefit)
            alpha_logits_up = self._logit_upsampler(alpha_logits)
            fg_logits_up = self._logit_upsampler(fg_logits)
            if self._compute_dtype != mx.float32:
                alpha_logits_up = alpha_logits_up.astype(mx.float32)
                fg_logits_up = fg_logits_up.astype(mx.float32)
            alpha_coarse = mx.sigmoid(alpha_logits_up)
            fg_coarse = mx.sigmoid(fg_logits_up)
        else:
            alpha_fn = self._compiled_alpha_decoder_call or self.alpha_decoder
            fg_fn = self._compiled_fg_decoder_call or self.fg_decoder
            # Alpha pipeline on default stream
            alpha_logits = alpha_fn(features)  # (B, H/4, W/4, 1)
            # FG pipeline on secondary stream — decoder + upsample + sigmoid
            with mx.stream(self._fg_stream):
                fg_logits = fg_fn(features)  # (B, H/4, W/4, 3)
                fg_logits_up = self._logit_upsampler(fg_logits)  # (B, H, W, 3)
                if self._compute_dtype != mx.float32:
                    fg_logits_up = fg_logits_up.astype(mx.float32)
                fg_coarse = mx.sigmoid(fg_logits_up)

            del features

            # Alpha upsample + sigmoid on default stream (parallel with fg stream)
            alpha_logits_up = self._logit_upsampler(alpha_logits)  # (B, H, W, 1)
            if self._compute_dtype != mx.float32:
                alpha_logits_up = alpha_logits_up.astype(mx.float32)
            alpha_coarse = mx.sigmoid(alpha_logits_up)

        # Skip decoder→refiner materialization barrier: decoder outputs are
        # small (~8MB at 512²) so keeping them lazy lets MLX fuse sigmoid +
        # concatenation + refiner stem conv into fewer Metal dispatches.
        # The backbone barrier above is sufficient to reclaim large feature maps.

        # Refiner receives coarse predictions (optionally in reduced precision)
        rgb = x[:, :, :, :3]  # (B, H, W, 3)
        coarse_pred = mx.concatenate([alpha_coarse, fg_coarse], axis=-1)  # (B, H, W, 4)
        refiner_fn = self._compiled_refiner_call or self.refiner

        # Prepare refiner inputs (optionally cast to reduced precision)
        if self._refiner_dtype is not None:
            rgb_r = rgb.astype(self._refiner_dtype)
            coarse_r = coarse_pred.astype(self._refiner_dtype)
        else:
            rgb_r, coarse_r = rgb, coarse_pred

        # Tiled refiner reduces peak im2col memory for dilated convolutions
        ts = self._refiner_tile_size
        if ts is not None and (rgb_r.shape[1] > ts or rgb_r.shape[2] > ts):
            delta_logits = self._refiner_tiled(refiner_fn, rgb_r, coarse_r)
        else:
            delta_logits = refiner_fn(rgb_r, coarse_r)

        if self._refiner_dtype is not None:
            delta_logits = delta_logits.astype(mx.float32)  # (B, H, W, 4)
            del rgb_r, coarse_r

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

    def _refiner_tiled(
        self,
        refiner_fn: object,
        rgb: mx.array,
        coarse: mx.array,
    ) -> mx.array:
        """Run refiner on spatial tiles to reduce peak im2col memory.

        Tiles overlap by 32px (refiner receptive field radius = 31px) to
        ensure exact equivalence with full-image processing.
        """
        ts = self._refiner_tile_size
        assert ts is not None
        overlap = 32  # RF radius: stem(1) + res1(2) + res2(4) + res3(8) + res4(16) = 31
        H = rgb.shape[1]
        W = rgb.shape[2]

        tile_rows: list[mx.array] = []
        y = 0
        while y < H:
            tile_cols: list[mx.array] = []
            x = 0
            while x < W:
                # Extract tile with overlap padding (clamped to image bounds)
                y0 = max(0, y - overlap)
                x0 = max(0, x - overlap)
                y1 = min(H, y + ts + overlap)
                x1 = min(W, x + ts + overlap)

                delta_tile = refiner_fn(
                    rgb[:, y0:y1, x0:x1, :],
                    coarse[:, y0:y1, x0:x1, :],
                )

                # Crop to owned region (remove overlap padding)
                own_y = y - y0
                own_x = x - x0
                own_h = min(ts, H - y)
                own_w = min(ts, W - x)
                tile_cols.append(
                    delta_tile[:, own_y : own_y + own_h, own_x : own_x + own_w, :]
                )
                x += ts

            if len(tile_cols) > 1:
                tile_rows.append(mx.concatenate(tile_cols, axis=2))
            else:
                tile_rows.append(tile_cols[0])
            y += ts

        if len(tile_rows) > 1:
            return mx.concatenate(tile_rows, axis=1)
        return tile_rows[0]

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

        # Cast refiner weights to reduced precision at load time
        if self._refiner_dtype is not None:
            weight_pairs = [
                (k, v.astype(self._refiner_dtype)) if k.startswith("refiner.") else (k, v)
                for k, v in weight_pairs
            ]

        self.load_weights(weight_pairs)
        self.eval()
        # materialize all parameters — mx.eval is MLX array materialization
        mx.eval(self.parameters())  # noqa: S307  -- mx.eval, not Python eval

        # Fold BatchNorm into precomputed scale+offset (2 ops vs 5)
        # Also precomputes 2D weights for 1x1 conv addmm bypass
        self.alpha_decoder.fold_bn()
        self.fg_decoder.fold_bn()
        # Precompute refiner 1x1 conv weight for addmm bypass
        self.refiner.prepare_inference()

        if self._compile_forward:
            # Whole-forward compilation: fuses backbone + decoders + upsample +
            # sigmoid + refiner + final sigmoid into a single compiled graph.
            # Eliminates eager Metal dispatches between components and the
            # stage_gc sync point.
            self._compiled = True
            self.__call__ = mx.compile(self.__call__)  # type: ignore[method-assign]
        else:
            # Per-component compilation (legacy path)
            if self._compile_backbone:
                self._compiled_backbone_call = mx.compile(self.backbone.__call__)

            if self._compile_refiner:
                self._compiled_refiner_call = mx.compile(self.refiner.__call__)

            if self._compile_decoders:
                if self._fused_decode:
                    self._compiled_fused_pair_call = mx.compile(self._fused_pair.__call__)
                else:
                    self._compiled_alpha_decoder_call = mx.compile(self.alpha_decoder.__call__)
                    self._compiled_fg_decoder_call = mx.compile(self.fg_decoder.__call__)
