"""Inference pipeline.

Orchestrates: load weights -> load image -> preprocess -> model forward -> postprocess -> save.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx

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
) -> GreenFormer:
    """Build GreenFormer and load weights from safetensors checkpoint."""
    model = GreenFormer(img_size=img_size)
    model.load_checkpoint(checkpoint)
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
    rgb = load_image(image_path)
    alpha_hint = load_alpha_hint(alpha_hint_path)
    x = preprocess(rgb, alpha_hint)
    # materialize input
    mx.eval(x)  # noqa: S307
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
