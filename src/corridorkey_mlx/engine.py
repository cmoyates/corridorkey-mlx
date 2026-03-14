"""CorridorKey MLX engine — drop-in backend for the main CorridorKey app.

Wraps the MLX GreenFormer model behind a stable ``process_frame`` API that
mirrors the Torch engine contract expected by the main CorridorKey repository.
"""

from __future__ import annotations

import gc
import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np
from PIL import Image

from corridorkey_mlx.inference.pipeline import load_model
from corridorkey_mlx.inference.tiling import (
    DEFAULT_OVERLAP,
    tiled_inference,
)
from corridorkey_mlx.io.image import (
    postprocess_alpha,
    postprocess_foreground,
    preprocess,
)
from corridorkey_mlx.io.preprocess_mlx import preprocess_mlx

if TYPE_CHECKING:
    from corridorkey_mlx.model.corridorkey import GreenFormer

logger = logging.getLogger(__name__)

PRODUCTION_IMG_SIZE = 2048


class CorridorKeyMLXEngine:
    """MLX inference engine compatible with the main CorridorKey backend contract.

    Args:
        checkpoint_path: Absolute or resolvable path to converted MLX
            ``.safetensors`` checkpoint. Required — no default.
        device: Ignored on MLX (Apple Silicon uses unified memory).
            Accepted for API compatibility with the Torch engine.
        img_size: Internal model resolution (square). The model was trained
            at 2048. Use 512 for fast dev iteration. When tiling is enabled,
            this is ignored and the model runs at ``tile_size``.
        use_refiner: If True, return refined alpha/fg. If False, return
            coarse predictions (skips refiner in postprocessing, not forward pass).
        compile: If True, wrap model forward with ``mx.compile`` for faster
            repeated inference at the same resolution.
        tile_size: If set, enable tiled inference — split full-res input into
            overlapping tiles of this size. Model loads at tile_size instead of
            img_size. Set to 0 or None to disable.
        overlap: Overlap in pixels between adjacent tiles (default 128).

    Example::

        from corridorkey_mlx import CorridorKeyMLXEngine

        # Full-frame (default)
        engine = CorridorKeyMLXEngine("/path/to/corridorkey_mlx.safetensors")

        # Tiled — 12x less memory at 2048x2048
        engine = CorridorKeyMLXEngine("/path/to/ckpt.safetensors", tile_size=512, overlap=128)

        result = engine.process_frame(rgb_uint8, mask_uint8)
        # result["alpha"]     — (H, W) uint8
        # result["fg"]        — (H, W, 3) uint8
        # result["comp"]      — (H, W, 3) uint8
        # result["processed"] — (H, W, 3) uint8
    """

    _despill_warned: bool = False
    _despeckle_warned: bool = False

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | None = None,
        img_size: int = PRODUCTION_IMG_SIZE,
        use_refiner: bool = True,
        compile: bool = True,
        tile_size: int | None = None,
        overlap: int = DEFAULT_OVERLAP,
        backbone_size: int | None = None,
    ) -> None:
        checkpoint = Path(checkpoint_path)
        if not checkpoint.exists():
            msg = f"Checkpoint not found: {checkpoint}"
            raise FileNotFoundError(msg)

        # Tune MLX buffer commit frequency.
        # Without compilation: small buffers force frequent materialization,
        # preventing graph buildup (17% faster for tiled workloads).
        # With compilation: the compiler handles fusion, so larger buffers
        # let the graph optimizer see more operations to fuse.
        import os
        if not compile:
            os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "2")
            os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "2")

        if device is not None:
            logger.info("device=%r ignored on MLX (unified memory)", device)

        self._use_refiner = use_refiner
        self._tiled = bool(tile_size)
        self._tile_size = tile_size or img_size
        self._overlap = overlap

        if self._tiled:
            # Tiled: model runs at tile_size, input stays full-res
            self._img_size = self._tile_size
            self._model: GreenFormer = load_model(
                checkpoint, img_size=self._tile_size, compile=False, slim=True,
                backbone_size=backbone_size,
                refiner_dtype=mx.float16,
                stage_gc=False,
            )
            logger.info(
                "Tiled inference: tile_size=%d, overlap=%d", self._tile_size, self._overlap
            )
        else:
            # Full-frame: resize input to img_size
            self._img_size = img_size
            self._model = load_model(
                checkpoint, img_size=img_size, compile=compile, slim=True,
                backbone_size=backbone_size,
                refiner_dtype=mx.float16,
            )

    def process_frame(
        self,
        image: np.ndarray,
        mask_linear: np.ndarray,
        refiner_scale: float = 1.0,
        input_is_linear: bool = False,
        fg_is_straight: bool = True,
        despill_strength: float = 1.0,
        auto_despeckle: bool = True,
        despeckle_size: int = 400,
    ) -> dict[str, np.ndarray]:
        """Run inference on a single frame.

        Args:
            image: RGB input, uint8 ``(H, W, 3)``.
            mask_linear: Coarse alpha hint, uint8 ``(H, W)`` or ``(H, W, 1)``.
            refiner_scale: Blend factor between coarse and refined output.
                1.0 = fully refined, 0.0 = fully coarse.
            input_is_linear: Accepted for compat; currently a no-op.
                ImageNet normalization assumes sRGB inputs.
            fg_is_straight: If True, foreground uses straight alpha for compositing.
            despill_strength: Stub — not yet implemented.
            auto_despeckle: Stub — not yet implemented.
            despeckle_size: Stub — not yet implemented.

        Returns:
            Dict with uint8 numpy arrays:
            - ``alpha``: ``(H, W)`` alpha matte
            - ``fg``: ``(H, W, 3)`` foreground
            - ``comp``: ``(H, W, 3)`` foreground composited over black
            - ``processed``: ``(H, W, 3)`` same as ``fg`` (placeholder)
        """
        # -- input validation --
        _validate_image(image)
        _validate_mask(mask_linear)

        original_h, original_w = image.shape[:2]

        # -- to float32 [0, 1] --
        rgb_f32 = image.astype(np.float32) / 255.0
        mask_f32 = mask_linear.astype(np.float32) / 255.0
        if mask_f32.ndim == 2:
            mask_f32 = mask_f32[:, :, np.newaxis]

        if self._tiled:
            # Tiled: keep full-res, preprocess, run tiled_inference
            x = preprocess(rgb_f32, mask_f32)
            outputs = tiled_inference(
                self._model, x, tile_size=self._tile_size, overlap=self._overlap
            )
            # NOTE: mx.eval is MLX array materialization, not Python eval()
            mx.eval(outputs)  # noqa: S307
        else:
            # Full-frame: resize to model resolution
            if rgb_f32.shape[0] != self._img_size or rgb_f32.shape[1] != self._img_size:
                rgb_pil = Image.fromarray(image).resize(
                    (self._img_size, self._img_size), Image.Resampling.BICUBIC
                )
                rgb_f32 = np.asarray(rgb_pil, dtype=np.float32) / 255.0

                mask_u8 = mask_linear if mask_linear.ndim == 2 else mask_linear[:, :, 0]
                mask_pil = Image.fromarray(mask_u8, mode="L").resize(
                    (self._img_size, self._img_size), Image.Resampling.BICUBIC
                )
                mask_f32 = np.asarray(mask_pil, dtype=np.float32)[:, :, np.newaxis] / 255.0

            x = preprocess_mlx(rgb_f32, mask_f32)
            outputs = self._model(x)
            # NOTE: mx.eval is MLX array materialization, not Python eval()
            mx.eval(outputs)  # noqa: S307

        # -- select coarse vs refined --
        if self._tiled:
            # tiled_inference only returns final (refined) outputs
            alpha_out = outputs["alpha_final"]
            fg_out = outputs["fg_final"]
        else:
            alpha_coarse = outputs["alpha_coarse"]
            fg_coarse = outputs["fg_coarse"]
            alpha_refined = outputs["alpha_final"]
            fg_refined = outputs["fg_final"]

            if not self._use_refiner or refiner_scale == 0.0:
                alpha_out = alpha_coarse
                fg_out = fg_coarse
            elif refiner_scale == 1.0:
                alpha_out = alpha_refined
                fg_out = fg_refined
            else:
                s = refiner_scale
                alpha_out = alpha_coarse * (1.0 - s) + alpha_refined * s
                fg_out = fg_coarse * (1.0 - s) + fg_refined * s

        del outputs

        # -- postprocess to uint8 --
        alpha_u8 = postprocess_alpha(alpha_out)
        fg_u8 = postprocess_foreground(fg_out)
        del alpha_out, fg_out

        # -- resize back to original --
        if alpha_u8.shape[0] != original_h or alpha_u8.shape[1] != original_w:
            target = (original_w, original_h)
            alpha_u8 = np.asarray(
                Image.fromarray(alpha_u8, mode="L").resize(target, Image.Resampling.BICUBIC),
                dtype=np.uint8,
            )
            fg_u8 = np.asarray(
                Image.fromarray(fg_u8, mode="RGB").resize(target, Image.Resampling.BICUBIC),
                dtype=np.uint8,
            )

        # -- stubs: despill / despeckle --
        if despill_strength > 0.0 and not CorridorKeyMLXEngine._despill_warned:
            warnings.warn(
                "despill not yet implemented in MLX backend; strength ignored",
                stacklevel=2,
            )
            CorridorKeyMLXEngine._despill_warned = True

        if auto_despeckle and not CorridorKeyMLXEngine._despeckle_warned:
            warnings.warn(
                "despeckle not yet implemented in MLX backend; ignored",
                stacklevel=2,
            )
            CorridorKeyMLXEngine._despeckle_warned = True

        # -- composite over black --
        alpha_3ch = alpha_u8[:, :, np.newaxis].astype(np.float32) / 255.0
        fg_float = fg_u8.astype(np.float32)
        comp = (fg_float * alpha_3ch).astype(np.uint8) if fg_is_straight else fg_u8.copy()

        return {
            "alpha": alpha_u8,
            "fg": fg_u8,
            "comp": comp,
            "processed": fg_u8,
        }


def _validate_image(image: np.ndarray) -> None:
    """Validate image is uint8 HWC RGB."""
    if not isinstance(image, np.ndarray):
        msg = f"image must be numpy ndarray, got {type(image).__name__}"
        raise TypeError(msg)
    if image.dtype != np.uint8:
        msg = f"image must be uint8, got {image.dtype}"
        raise ValueError(msg)
    if image.ndim != 3 or image.shape[2] != 3:
        msg = f"image must be (H, W, 3), got {image.shape}"
        raise ValueError(msg)


def _validate_mask(mask: np.ndarray) -> None:
    """Validate mask is uint8 HW or HW1."""
    if not isinstance(mask, np.ndarray):
        msg = f"mask must be numpy ndarray, got {type(mask).__name__}"
        raise TypeError(msg)
    if mask.dtype != np.uint8:
        msg = f"mask must be uint8, got {mask.dtype}"
        raise ValueError(msg)
    if mask.ndim == 2:
        return
    if mask.ndim == 3 and mask.shape[2] == 1:
        return
    msg = f"mask must be (H, W) or (H, W, 1), got {mask.shape}"
    raise ValueError(msg)
