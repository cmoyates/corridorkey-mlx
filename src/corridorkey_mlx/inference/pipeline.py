"""Inference pipeline.

Orchestrates: load weights -> load image -> preprocess -> model forward -> postprocess -> save.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
from mlx.utils import tree_map

from corridorkey_mlx.io.image import (
    load_alpha_hint,
    load_image,
    postprocess_alpha,
    postprocess_foreground,
    preprocess,
    save_alpha,
    save_foreground,
)
from corridorkey_mlx.model.corridorkey import GreenFormer

if TYPE_CHECKING:
    import numpy as np

DEFAULT_CHECKPOINT = Path("checkpoints/corridorkey_mlx.safetensors")
DEFAULT_IMG_SIZE = 512


def load_model(
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    img_size: int = DEFAULT_IMG_SIZE,
    compile: bool = False,
    shapeless: bool = False,
    fp16: bool = False,
    backbone_size: int | None = None,
) -> GreenFormer:
    """Build GreenFormer and load weights from safetensors checkpoint.

    Args:
        checkpoint: Path to converted MLX safetensors weights.
        img_size: Input resolution (square). Must match at inference time.
        compile: If True, wrap forward pass with mx.compile for fused execution.
            Automatically disabled when backbone_size differs from img_size
            (dual-resolution forward has varying internal shapes).
        shapeless: If True, use shapeless=True with mx.compile. Only safe when
            no shape-dependent logic varies across calls. The Hiera backbone
            uses shape-dependent reshapes, so shapeless is NOT recommended
            unless all inputs share the same spatial dimensions.
        fp16: If True, cast decoder weights to float16 (mixed precision).
            Backbone + refiner stay FP32 for numerical stability.
        backbone_size: If set, backbone runs at this resolution while refiner
            runs at img_size. Must be <= img_size and divisible by 4.
    """
    model = GreenFormer(img_size=img_size, backbone_size=backbone_size)
    model.load_checkpoint(checkpoint)
    if fp16:
        _cast_model_fp16(model)
    if compile:
        if model._decoupled:
            import logging

            logging.getLogger(__name__).warning(
                "mx.compile disabled: backbone_size=%d != img_size=%d "
                "(dual-resolution forward has varying internal shapes)",
                backbone_size,
                img_size,
            )
        else:
            model = compile_model(model, shapeless=shapeless)
    return model


def _cast_model_fp16(model: GreenFormer) -> None:
    """Cast decoder parameters to float16, keep backbone + refiner FP32.

    Mixed precision strategy:
    - Backbone FP32: avoids cumulative drift across 24 Hiera blocks
    - Decoders FP16: biggest parameter count, safe after sigmoid clamping
    - Refiner FP32: REFINER_SCALE=10.0 amplifies FP16 rounding errors
      beyond acceptable tolerance (2.3e-3 max_abs vs 1e-3 target)
    """

    def to_fp16(x: mx.array) -> mx.array:
        return x.astype(mx.float16)

    for submodule in (model.alpha_decoder, model.fg_decoder):
        submodule.update(tree_map(to_fp16, submodule.parameters()))
    # materialize FP16 parameters — mx.eval is MLX array materialization
    mx.eval(model.parameters())  # noqa: S307


def compile_model(model: GreenFormer, shapeless: bool = False) -> GreenFormer:
    """Wrap model forward pass with mx.compile for fused execution.

    Fixed-shape compile (shapeless=False) is safe for all inputs of the
    same resolution. Shapeless compile is experimental — the backbone uses
    shape-dependent reshapes that may trigger recompilation.
    """
    model.__call__ = mx.compile(model.__call__, shapeless=shapeless)  # type: ignore[method-assign]
    return model


def infer(
    model: GreenFormer,
    image_path: str | Path,
    alpha_hint_path: str | Path,
) -> dict[str, mx.array]:
    """Run single-image inference.

    Args:
        model: Loaded GreenFormer model.
        image_path: Path to RGB input image.
        alpha_hint_path: Path to coarse alpha hint (grayscale).

    Returns:
        Model output dict with all intermediate and final tensors.
    """
    img_size = model.backbone.img_size
    rgb = load_image(image_path, img_size=img_size)
    alpha_hint = load_alpha_hint(alpha_hint_path, img_size=img_size)
    x = preprocess(rgb, alpha_hint)
    outputs = model(x)
    # materialize all outputs
    mx.eval(outputs)  # noqa: S307
    return outputs


def infer_and_save(
    model: GreenFormer,
    image_path: str | Path,
    alpha_hint_path: str | Path,
    output_dir: str | Path,
) -> dict[str, np.ndarray]:
    """Run inference and save alpha + foreground PNGs.

    Returns:
        Dict with 'alpha' and 'foreground' as uint8 numpy arrays.
    """
    outputs = infer(model, image_path, alpha_hint_path)

    alpha_arr = postprocess_alpha(outputs["alpha_final"])
    fg_arr = postprocess_foreground(outputs["fg_final"])

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    save_alpha(alpha_arr, out / "alpha.png")
    save_foreground(fg_arr, out / "foreground.png")

    return {"alpha": alpha_arr, "foreground": fg_arr}
