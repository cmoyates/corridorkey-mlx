"""CorridorKey MLX engine — drop-in backend for the main CorridorKey app.

Wraps the MLX GreenFormer model behind a stable ``process_frame`` API that
mirrors the Torch engine contract expected by the main CorridorKey repository.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from corridorkey_mlx.inference.pipeline import load_model
from corridorkey_mlx.inference.tiling import tiled_inference

if TYPE_CHECKING:
    from corridorkey_mlx.model.corridorkey import GreenFormer

logger = logging.getLogger(__name__)

PRODUCTION_IMG_SIZE = 2048

# ImageNet normalization constants as MLX arrays for zero-copy preprocessing
_IMAGENET_MEAN = mx.array([0.485, 0.456, 0.406], dtype=mx.float32)
_IMAGENET_STD = mx.array([0.229, 0.224, 0.225], dtype=mx.float32)


class CorridorKeyMLXEngine:
    """MLX inference engine compatible with the main CorridorKey backend contract.

    Args:
        checkpoint_path: Absolute or resolvable path to converted MLX
            ``.safetensors`` checkpoint. Required — no default.
        device: Ignored on MLX (Apple Silicon uses unified memory).
            Accepted for API compatibility with the Torch engine.
        img_size: Internal model resolution (square). The model was trained
            at 2048. Use 512 for fast dev iteration.
        use_refiner: If True, return refined alpha/fg. If False, return
            coarse predictions (skips refiner in postprocessing, not forward pass).
        compile: If True, wrap model forward with ``mx.compile`` for faster
            repeated inference at the same resolution. Auto-disabled when
            backbone_size differs from img_size.
        fp16: If True, cast decoder weights to float16 (mixed precision).
            Backbone + refiner stay FP32 for numerical stability.
        backbone_size: Backbone runs at this resolution while refiner
            runs at img_size. None = same as img_size. Must be <= img_size
            and divisible by 4 (patch stride).
        tile_size: If set, split input into overlapping tiles of this size.
            None = no tiling (single-shot inference). Must be divisible by 4.
        tile_overlap: Overlap in pixels between adjacent tiles.
            Only used when tile_size is set.

    Example::

        from corridorkey_mlx import CorridorKeyMLXEngine

        engine = CorridorKeyMLXEngine("/path/to/corridorkey_mlx.safetensors")
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
        fp16: bool = True,
        backbone_size: int | None = None,
        tile_size: int | None = None,
        tile_overlap: int = 96,
    ) -> None:
        checkpoint = Path(checkpoint_path)
        if not checkpoint.exists():
            msg = f"Checkpoint not found: {checkpoint}"
            raise FileNotFoundError(msg)

        if device is not None:
            logger.info("device=%r ignored on MLX (unified memory)", device)

        # -- parameter validation --
        _validate_engine_params(img_size, backbone_size, tile_size, tile_overlap)

        # Auto-disable compile when backbone_size differs from img_size
        effective_compile = compile
        if compile and backbone_size is not None and backbone_size != img_size:
            logger.warning(
                "mx.compile disabled: backbone_size=%d != img_size=%d "
                "(dual-resolution forward has varying internal shapes)",
                backbone_size,
                img_size,
            )
            effective_compile = False

        self._img_size = img_size
        self._use_refiner = use_refiner
        self._tile_size = tile_size
        self._tile_overlap = tile_overlap

        # Memory safety rail: cap MLX at 80% of system memory
        _set_memory_limit_safety_rail()

        # When tiling, model runs at tile_size; backbone_size ignored per design
        # (tiles are already small, downsampling further is wasteful)
        model_res = tile_size if tile_size is not None else img_size
        model_backbone_size = None if tile_size is not None else backbone_size

        self._model: GreenFormer = load_model(
            checkpoint,
            img_size=model_res,
            compile=effective_compile,
            fp16=fp16,
            backbone_size=model_backbone_size,
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

        All preprocessing, inference, and postprocessing run as MLX ops on GPU.
        A single ``np.array()`` sync happens at the end to return numpy results.

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

        # -- convert to mx.array immediately (single CPU->GPU transfer) --
        rgb_mx = mx.array(image).astype(mx.float32) / 255.0  # (H, W, 3)
        mask_mx = mx.array(mask_linear).astype(mx.float32) / 255.0
        if mask_mx.ndim == 2:
            mask_mx = mask_mx[:, :, None]  # (H, W, 1)

        # -- resize to model resolution (MLX-native, no PIL) --
        needs_resize = original_h != self._img_size or original_w != self._img_size
        if needs_resize:
            scale_h = self._img_size / original_h
            scale_w = self._img_size / original_w
            resizer = nn.Upsample(
                scale_factor=(scale_h, scale_w), mode="linear", align_corners=False
            )
            # nn.Upsample expects NHWC
            rgb_mx = resizer(rgb_mx[None])[0]  # (img_size, img_size, 3)
            mask_mx = resizer(mask_mx[None])[0]  # (img_size, img_size, 1)

        # -- preprocess: ImageNet norm + alpha concat (all MLX ops) --
        rgb_norm = (rgb_mx - _IMAGENET_MEAN) / _IMAGENET_STD
        x = mx.concatenate([rgb_norm, mask_mx], axis=-1)[None]  # (1, H, W, 4)

        # -- forward: tiled or single-shot --
        if self._tile_size is not None:
            # Tiled path: full image stays as numpy, tiles sent to GPU on demand
            # mx.eval is MLX array materialization, not Python eval
            mx.eval(x)  # noqa: S307
            x_np = np.array(x)
            outputs = tiled_inference(
                self._model,
                x_np,
                tile_size=self._tile_size,
                overlap=self._tile_overlap,
            )
            # Tiled inference only returns alpha_final/fg_final
            alpha_out = outputs["alpha_final"]
            fg_out = outputs["fg_final"]
        else:
            # Single-shot path (slim=True skips 5 unused intermediate tensors)
            outputs = self._model(x, slim=True)

            # -- select coarse vs refined (MLX ops) --
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

        # -- resize back to original (MLX-native) --
        if needs_resize:
            inv_scale_h = original_h / self._img_size
            inv_scale_w = original_w / self._img_size
            inv_resizer = nn.Upsample(
                scale_factor=(inv_scale_h, inv_scale_w),
                mode="linear",
                align_corners=False,
            )
            alpha_out = inv_resizer(alpha_out)  # (1, orig_h, orig_w, 1)
            fg_out = inv_resizer(fg_out)  # (1, orig_h, orig_w, 3)

        # -- postprocess to uint8 (MLX ops) --
        alpha_out = mx.clip(alpha_out[0, :, :, 0], 0.0, 1.0)  # (H, W)
        fg_out = mx.clip(fg_out[0], 0.0, 1.0)  # (H, W, 3)

        alpha_u8_mx = (alpha_out * 255.0).astype(mx.uint8)
        fg_u8_mx = (fg_out * 255.0).astype(mx.uint8)

        # -- composite over black (MLX ops) --
        if fg_is_straight:
            comp_mx = (fg_out * alpha_out[:, :, None] * 255.0).astype(mx.uint8)
        else:
            comp_mx = fg_u8_mx

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

        # -- single GPU->CPU sync: materialize all outputs at once --
        # mx.eval is MLX array materialization, not Python eval
        mx.eval(alpha_u8_mx, fg_u8_mx, comp_mx)  # noqa: S307
        alpha_u8 = np.array(alpha_u8_mx)
        fg_u8 = np.array(fg_u8_mx)
        comp = np.array(comp_mx)

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


PATCH_STRIDE = 4
MEMORY_LIMIT_FRACTION = 0.8


def _set_memory_limit_safety_rail() -> None:
    """Set MLX memory limit to 80% of system memory if detectable."""
    try:
        import subprocess

        result = subprocess.run(  # noqa: S603, S607
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            check=True,
        )
        total_bytes = int(result.stdout.strip())
        limit = int(total_bytes * MEMORY_LIMIT_FRACTION)
        mx.set_memory_limit(limit)
        logger.debug(
            "MLX memory limit set to %.0f MB (%.0f%% of %d MB)",
            limit / 1e6,
            MEMORY_LIMIT_FRACTION * 100,
            total_bytes / 1e6,
        )
    except Exception:
        logger.debug("Could not detect system memory; skipping memory limit")


def _validate_engine_params(
    img_size: int,
    backbone_size: int | None,
    tile_size: int | None,
    tile_overlap: int,
) -> None:
    """Validate engine constructor parameters."""
    if backbone_size is not None:
        if backbone_size % PATCH_STRIDE != 0:
            msg = f"backbone_size ({backbone_size}) must be divisible by {PATCH_STRIDE}"
            raise ValueError(msg)
        if backbone_size > img_size:
            msg = f"backbone_size ({backbone_size}) must be <= img_size ({img_size})"
            raise ValueError(msg)

    if tile_size is not None:
        if tile_size % PATCH_STRIDE != 0:
            msg = f"tile_size ({tile_size}) must be divisible by {PATCH_STRIDE}"
            raise ValueError(msg)
        if tile_overlap >= tile_size:
            msg = f"tile_overlap ({tile_overlap}) must be < tile_size ({tile_size})"
            raise ValueError(msg)
