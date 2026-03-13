"""Top-level CorridorKey model (GreenFormer) — MLX port.

Composes Hiera backbone + dual decoder heads + CNN refiner.
All internal operations use NHWC layout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn

from corridorkey_mlx.model.backbone import HieraBackbone
from corridorkey_mlx.model.decoder import DecoderHead, FusedDecoderPair
from corridorkey_mlx.model.hiera import ENCODER_KEY_PREFIX, _interpolate_pos_embed, _prod
from corridorkey_mlx.model.refiner import CNNRefinerModule
from corridorkey_mlx.utils.quantize import safe_quantize

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
        refiner_tile_size: int | None = 1024,
        quantize_backbone_stages: bool = True,
        backbone_bf16_stages123: bool = True,
        decoder_dtype: mx.Dtype | None = mx.bfloat16,
        refiner_skip_confidence: float | None = None,
        refiner_frozen_gn: bool = False,
    ) -> None:
        super().__init__()
        self._compute_dtype = dtype
        self._quantize_backbone_stages = quantize_backbone_stages
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
        self._decoder_dtype = decoder_dtype
        self._refiner_skip_confidence = refiner_skip_confidence
        self._refiner_frozen_gn = refiner_frozen_gn
        self._tiles_skipped = 0  # diagnostic counter, reset per forward
        self._tiles_total = 0
        self.backbone = HieraBackbone(
            img_size=img_size, use_sdpa=use_sdpa, bf16_stages123=backbone_bf16_stages123
        )
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

    @property
    def tile_skip_stats(self) -> tuple[int, int]:
        """(tiles_skipped, tiles_total) from last _refiner_tiled call."""
        return self._tiles_skipped, self._tiles_total

    @property
    def compiled(self) -> bool:
        """Whether the model is running inside a compiled graph."""
        return self._compiled

    @compiled.setter
    def compiled(self, value: bool) -> None:
        self._compiled = value

    def __call__(self, x: mx.array) -> dict[str, mx.array]:
        """Forward pass — compile-safe (no async_eval / stream switches).

        This method can be safely wrapped with ``mx.compile()``.  Eager-only
        optimizations (async_eval, multi-stream dispatch) live in
        ``forward_eager`` and are used by the per-component compilation path.

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

        # Cast features to compute dtype for decoders (bf16 saves memory)
        if self._compute_dtype != mx.float32:
            features = [f.astype(self._compute_dtype) for f in features]

        # Decoder heads -> logits at H/4 resolution
        if self._fused_decode:
            fused_fn = self._compiled_fused_pair_call or self._fused_pair
            alpha_logits, fg_logits = fused_fn(features)
            del features

            alpha_logits_up = self._logit_upsampler(alpha_logits)
            fg_logits_up = self._logit_upsampler(fg_logits)
            alpha_coarse = mx.sigmoid(alpha_logits_up)
            fg_coarse = mx.sigmoid(fg_logits_up)
        else:
            alpha_fn = self._compiled_alpha_decoder_call or self.alpha_decoder
            fg_fn = self._compiled_fg_decoder_call or self.fg_decoder
            alpha_logits = alpha_fn(features)  # (B, H/4, W/4, 1)
            fg_logits = fg_fn(features)  # (B, H/4, W/4, 3)
            del features

            alpha_logits_up = self._logit_upsampler(alpha_logits)  # (B, H, W, 1)
            alpha_coarse = mx.sigmoid(alpha_logits_up)
            fg_logits_up = self._logit_upsampler(fg_logits)  # (B, H, W, 3)
            fg_coarse = mx.sigmoid(fg_logits_up)

        # Refiner receives coarse predictions (optionally in reduced precision)
        rgb = x[:, :, :, :3]  # (B, H, W, 3)
        coarse_pred = mx.concatenate([alpha_coarse, fg_coarse], axis=-1)  # (B, H, W, 4)
        refiner_fn = self._compiled_refiner_call or self.refiner

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
            del rgb_r, coarse_r

        # Final predictions: additive residual in logit space, then sigmoid
        alpha_final = mx.sigmoid(alpha_logits_up + delta_logits[:, :, :, 0:1])
        fg_final = mx.sigmoid(fg_logits_up + delta_logits[:, :, :, 1:4])

        if self._slim:
            return {
                "alpha_coarse": alpha_coarse.astype(mx.float32),
                "fg_coarse": fg_coarse.astype(mx.float32),
                "alpha_final": alpha_final.astype(mx.float32),
                "fg_final": fg_final.astype(mx.float32),
            }

        return {
            "alpha_logits": alpha_logits.astype(mx.float32),
            "fg_logits": fg_logits.astype(mx.float32),
            "alpha_logits_up": alpha_logits_up.astype(mx.float32),
            "fg_logits_up": fg_logits_up.astype(mx.float32),
            "alpha_coarse": alpha_coarse.astype(mx.float32),
            "fg_coarse": fg_coarse.astype(mx.float32),
            "delta_logits": delta_logits.astype(mx.float32),
            "alpha_final": alpha_final.astype(mx.float32),
            "fg_final": fg_final.astype(mx.float32),
        }

    def run_backbone(self, x: mx.array) -> list[mx.array]:
        """Run backbone only, return multiscale features.

        Uses compiled callable when available. Casts features to compute
        dtype if configured (e.g. bf16 for decoder bandwidth savings).

        Args:
            x: (B, H, W, 4) NHWC input.

        Returns:
            List of 4 feature maps at strides [4, 8, 16, 32].
        """
        backbone_fn = self._compiled_backbone_call or self.backbone
        features = backbone_fn(x)
        if self._compute_dtype != mx.float32:
            features = [f.astype(self._compute_dtype) for f in features]
        return features

    def run_decoders(self, features: list[mx.array]) -> dict[str, mx.array]:
        """Run decoder heads on backbone features. Does NOT consume features.

        Returns coarse alpha/fg and upsampled logits needed by refiner.

        Args:
            features: 4 multiscale feature maps from run_backbone().

        Returns:
            Dict with alpha_logits_up, fg_logits_up, alpha_coarse, fg_coarse.
        """
        if self._fused_decode:
            fused_fn = self._compiled_fused_pair_call or self._fused_pair
            alpha_logits, fg_logits = fused_fn(features)
        else:
            alpha_fn = self._compiled_alpha_decoder_call or self.alpha_decoder
            fg_fn = self._compiled_fg_decoder_call or self.fg_decoder
            alpha_logits = alpha_fn(features)
            with mx.stream(self._fg_stream):
                fg_logits = fg_fn(features)

        alpha_logits_up = self._logit_upsampler(alpha_logits)
        fg_logits_up = self._logit_upsampler(fg_logits)
        alpha_coarse = mx.sigmoid(alpha_logits_up)
        fg_coarse = mx.sigmoid(fg_logits_up)

        return {
            "alpha_logits_up": alpha_logits_up,
            "fg_logits_up": fg_logits_up,
            "alpha_coarse": alpha_coarse,
            "fg_coarse": fg_coarse,
        }

    def run_refiner(self, x: mx.array, coarse: dict[str, mx.array]) -> dict[str, mx.array]:
        """Run refiner on fresh RGB + coarse predictions from decoders.

        Produces final alpha and foreground via additive residual in logit
        space. Supports tiled execution and reduced-precision refiner.

        Args:
            x: (B, H, W, 4) NHWC input (RGB extracted from channels 0:3).
            coarse: Dict from run_decoders() with logits_up and coarse maps.

        Returns:
            Dict with alpha_coarse, fg_coarse, alpha_final, fg_final (fp32).
            If not slim, also includes intermediate logits/delta.
        """
        alpha_logits_up = coarse["alpha_logits_up"]
        fg_logits_up = coarse["fg_logits_up"]
        alpha_coarse = coarse["alpha_coarse"]
        fg_coarse = coarse["fg_coarse"]

        rgb = x[:, :, :, :3]
        coarse_pred = mx.concatenate([alpha_coarse, fg_coarse], axis=-1)
        refiner_fn = self._compiled_refiner_call or self.refiner

        if self._refiner_dtype is not None:
            rgb_r = rgb.astype(self._refiner_dtype)
            coarse_r = coarse_pred.astype(self._refiner_dtype)
        else:
            rgb_r, coarse_r = rgb, coarse_pred

        ts = self._refiner_tile_size
        if ts is not None and (rgb_r.shape[1] > ts or rgb_r.shape[2] > ts):
            delta_logits = self._refiner_tiled(refiner_fn, rgb_r, coarse_r)
        else:
            delta_logits = refiner_fn(rgb_r, coarse_r)

        if self._refiner_dtype is not None:
            del rgb_r, coarse_r

        alpha_final = mx.sigmoid(alpha_logits_up + delta_logits[:, :, :, 0:1])
        fg_final = mx.sigmoid(fg_logits_up + delta_logits[:, :, :, 1:4])

        if self._slim:
            return {
                "alpha_coarse": alpha_coarse.astype(mx.float32),
                "fg_coarse": fg_coarse.astype(mx.float32),
                "alpha_final": alpha_final.astype(mx.float32),
                "fg_final": fg_final.astype(mx.float32),
            }

        return {
            "alpha_logits_up": alpha_logits_up.astype(mx.float32),
            "fg_logits_up": fg_logits_up.astype(mx.float32),
            "alpha_coarse": alpha_coarse.astype(mx.float32),
            "fg_coarse": fg_coarse.astype(mx.float32),
            "delta_logits": delta_logits.astype(mx.float32),
            "alpha_final": alpha_final.astype(mx.float32),
            "fg_final": fg_final.astype(mx.float32),
        }

    def forward_eager(self, x: mx.array) -> dict[str, mx.array]:
        """Eager forward with async materialization and multi-stream dispatch.

        Used by the per-component compilation path (compile_forward=False).
        NOT safe for wrapping with mx.compile — use __call__ instead.
        """
        # Backbone always runs in fp32
        backbone_fn = self._compiled_backbone_call or self.backbone
        features = backbone_fn(x)

        # Async-materialize backbone features so the CPU can build the
        # decoder graph while the GPU finishes backbone work.
        # NOTE: mx.async_eval is MLX async materialization, not Python eval()
        if self._stage_gc:
            mx.async_eval(*features)  # noqa: S307  -- mx.async_eval, not builtins

        # Cast features to compute dtype for decoders (bf16 saves memory)
        if self._compute_dtype != mx.float32:
            features = [f.astype(self._compute_dtype) for f in features]

        # Decoder heads -> logits at H/4 resolution
        if self._fused_decode:
            fused_fn = self._compiled_fused_pair_call or self._fused_pair
            alpha_logits, fg_logits = fused_fn(features)
            del features

            alpha_logits_up = self._logit_upsampler(alpha_logits)
            fg_logits_up = self._logit_upsampler(fg_logits)
            alpha_coarse = mx.sigmoid(alpha_logits_up)
            fg_coarse = mx.sigmoid(fg_logits_up)
        else:
            alpha_fn = self._compiled_alpha_decoder_call or self.alpha_decoder
            fg_fn = self._compiled_fg_decoder_call or self.fg_decoder
            # Alpha pipeline on default stream
            alpha_logits = alpha_fn(features)  # (B, H/4, W/4, 1)
            # FG pipeline on secondary stream
            with mx.stream(self._fg_stream):
                fg_logits = fg_fn(features)  # (B, H/4, W/4, 3)

            del features

            alpha_logits_up = self._logit_upsampler(alpha_logits)  # (B, H, W, 1)
            alpha_coarse = mx.sigmoid(alpha_logits_up)
            fg_logits_up = self._logit_upsampler(fg_logits)  # (B, H, W, 3)
            fg_coarse = mx.sigmoid(fg_logits_up)

        # Async-materialize decoder outputs before refiner to free decoder
        # intermediate buffers (im2col inflation in refiner).
        # NOTE: mx.async_eval is MLX async materialization, not Python eval()
        if self._stage_gc:
            mx.async_eval(alpha_logits_up, fg_logits_up, alpha_coarse, fg_coarse)  # noqa: S307  -- mx.async_eval, not Python eval

        # Refiner receives coarse predictions (optionally in reduced precision)
        rgb = x[:, :, :, :3]  # (B, H, W, 3)
        coarse_pred = mx.concatenate([alpha_coarse, fg_coarse], axis=-1)  # (B, H, W, 4)
        refiner_fn = self._compiled_refiner_call or self.refiner

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
            del rgb_r, coarse_r

        # Final predictions: additive residual in logit space, then sigmoid
        alpha_final = mx.sigmoid(alpha_logits_up + delta_logits[:, :, :, 0:1])
        fg_final = mx.sigmoid(fg_logits_up + delta_logits[:, :, :, 1:4])

        if self._slim:
            return {
                "alpha_coarse": alpha_coarse.astype(mx.float32),
                "fg_coarse": fg_coarse.astype(mx.float32),
                "alpha_final": alpha_final.astype(mx.float32),
                "fg_final": fg_final.astype(mx.float32),
            }

        return {
            "alpha_logits": alpha_logits.astype(mx.float32),
            "fg_logits": fg_logits.astype(mx.float32),
            "alpha_logits_up": alpha_logits_up.astype(mx.float32),
            "fg_logits_up": fg_logits_up.astype(mx.float32),
            "alpha_coarse": alpha_coarse.astype(mx.float32),
            "fg_coarse": fg_coarse.astype(mx.float32),
            "delta_logits": delta_logits.astype(mx.float32),
            "alpha_final": alpha_final.astype(mx.float32),
            "fg_final": fg_final.astype(mx.float32),
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

        When refiner_skip_confidence is set, tiles where coarse alpha is
        uniformly confident (all near 0 or all near 1) are skipped — the
        refiner delta is zero, so final = coarse for those regions.

        When refiner_frozen_gn is enabled, precomputes GroupNorm stats on
        the full image before tiling so per-tile stats match full-image.
        """
        ts = self._refiner_tile_size
        assert ts is not None
        overlap = 32  # RF radius: stem(1) + res1(2) + res2(4) + res3(8) + res4(16) = 31
        H = rgb.shape[1]
        W = rgb.shape[2]
        skip_thresh = self._refiner_skip_confidence

        # Frozen GroupNorm: collect full-image stats, then freeze for tiled pass
        if self._refiner_frozen_gn:
            self.refiner.collect_groupnorm_stats(rgb, coarse)
            self.refiner.freeze_groupnorm_stats()
            # Bypass compiled refiner — compiled graph doesn't see frozen stats
            refiner_fn = self.refiner

        self._tiles_skipped = 0
        self._tiles_total = 0

        try:
            return self._refiner_tile_loop(refiner_fn, rgb, coarse, ts, overlap, H, W, skip_thresh)
        finally:
            if self._refiner_frozen_gn:
                self.refiner.unfreeze_groupnorm_stats()

    def _refiner_tile_loop(
        self,
        refiner_fn: object,
        rgb: mx.array,
        coarse: mx.array,
        ts: int,
        overlap: int,
        H: int,
        W: int,
        skip_thresh: float | None,
    ) -> mx.array:
        """Inner tile loop — extracted for try/finally clarity."""
        tile_rows: list[mx.array] = []
        y = 0
        while y < H:
            tile_cols: list[mx.array] = []
            x = 0
            while x < W:
                own_h = min(ts, H - y)
                own_w = min(ts, W - x)
                self._tiles_total += 1

                # Adaptive skip: check coarse alpha confidence on owned region
                if skip_thresh is not None:
                    tile_alpha = coarse[:, y : y + own_h, x : x + own_w, 0]
                    # Materialize min/max for the branch decision
                    alpha_min = mx.min(tile_alpha)
                    alpha_max = mx.max(tile_alpha)
                    # mx.eval: MLX array materialization (not Python eval)
                    mx.eval(alpha_min, alpha_max)  # noqa: S307
                    if float(alpha_max) < skip_thresh or float(alpha_min) > (1.0 - skip_thresh):
                        # Tile is uniformly confident — refiner delta = 0
                        zeros = mx.zeros((rgb.shape[0], own_h, own_w, 4), dtype=rgb.dtype)
                        tile_cols.append(zeros)
                        self._tiles_skipped += 1
                        x += ts
                        continue

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
                cropped = delta_tile[:, own_y : own_y + own_h, own_x : own_x + own_w, :]
                # Materialize cropped tile and discard full delta_tile so MLX
                # can free the im2col buffers (~9x activation inflation from
                # dilated convolutions) before the next tile starts.
                # mx.eval: MLX array materialization (not Python eval)
                mx.eval(cropped)  # noqa: S307
                del delta_tile
                tile_cols.append(cropped)
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
        - Mixed-precision checkpoints (bf16 decoder/refiner weights loaded natively)
        """
        target_tokens = _prod(self.backbone.tokens_spatial_shape)
        weight_pairs: list[tuple[str, mx.array]] = []

        # mx.load handles bf16 safetensors natively (safe_open+numpy cannot)
        raw_weights = mx.load(str(path))

        for full_key, tensor in raw_weights.items():
            if full_key.startswith(ENCODER_KEY_PREFIX):
                # Backbone: strip encoder.model. prefix
                mlx_key = "backbone." + full_key[len(ENCODER_KEY_PREFIX) :]
                if mlx_key == "backbone.pos_embed":
                    tensor = _interpolate_pos_embed(tensor, target_tokens)
                    # NOTE: mx.eval is MLX array materialization, not Python eval()
                    mx.eval(tensor)  # noqa: S307  -- mx.eval, not Python eval
            else:
                # Decoder/refiner: use key as-is
                mlx_key = full_key

            weight_pairs.append((mlx_key, tensor))

        # Cast refiner weights to reduced precision (skip if already correct dtype)
        if self._refiner_dtype is not None:
            weight_pairs = [
                (k, v.astype(self._refiner_dtype))
                if k.startswith("refiner.") and v.dtype != self._refiner_dtype
                else (k, v)
                for k, v in weight_pairs
            ]

        # Cast decoder weights to reduced precision (skip if already correct dtype)
        if self._decoder_dtype is not None:
            decoder_prefixes = ("alpha_decoder.", "fg_decoder.")
            weight_pairs = [
                (k, v.astype(self._decoder_dtype))
                if k.startswith(decoder_prefixes) and v.dtype != self._decoder_dtype
                else (k, v)
                for k, v in weight_pairs
            ]

        self.load_weights(weight_pairs)
        self.eval()
        # materialize all parameters — mx.eval is MLX array materialization
        mx.eval(self.parameters())  # noqa: S307  -- mx.eval, not Python eval()

        # Quantize backbone stages 1-3 (dims 224,448,896 — all divisible by 32).
        # Stage 0 (dim=112) is skipped by safe_quantize. Int8 weights halve
        # memory bandwidth for 22/24 blocks' Linear layers (attn qkv/proj + MLP).
        if self._quantize_backbone_stages:
            # stage_ends = [1, 4, 20, 23]; stages 1-3 = blocks 2..23
            stage1_start = self.backbone.stage_ends[0] + 1  # block 2
            for blk in self.backbone.blocks[stage1_start:]:
                safe_quantize(blk, group_size=32, bits=8)
            # materialize quantized parameters — mx.eval is MLX graph materialization
            mx.eval(self.backbone.parameters())  # noqa: S307  -- mx.eval, not Python eval()

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
            # Per-component compilation: each component individually compiled,
            # with eager async_eval + multi-stream dispatch between them.
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

            # NOTE: forward_eager() provides async_eval + multi-stream dispatch
            # but is NOT safe for mx.compile wrapping.  Callers that need eager
            # optimizations without whole-forward compile should call
            # model.forward_eager(x) directly.
