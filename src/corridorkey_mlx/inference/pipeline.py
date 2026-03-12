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


WIRED_LIMIT_BYTES = 512 * 1024 * 1024  # 512 MiB — covers model weights + working set
CACHE_LIMIT_BYTES = 1536 * 1024 * 1024  # 1.5 GiB — forces buffer reuse, reduces peak memory


def load_model(
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    img_size: int = DEFAULT_IMG_SIZE,
    compile: bool = False,
    shapeless: bool = False,
    dtype: mx.Dtype = mx.bfloat16,
    fused_decode: bool = True,
    slim: bool = False,
    use_sdpa: bool = True,
    stage_gc: bool = True,
    refiner_dtype: mx.Dtype | None = mx.float16,
    wired_limit_bytes: int | None = WIRED_LIMIT_BYTES,
    cache_limit_bytes: int | None = CACHE_LIMIT_BYTES,
) -> GreenFormer:
    """Build GreenFormer and load weights from safetensors checkpoint.

    Args:
        checkpoint: Path to converted MLX safetensors weights.
        img_size: Input resolution (square). Must match at inference time.
        compile: If True, wrap forward pass with mx.compile for fused execution.
        shapeless: If True, use shapeless=True with mx.compile. Only safe when
            no shape-dependent logic varies across calls. The Hiera backbone
            uses shape-dependent reshapes, so shapeless is NOT recommended
            unless all inputs share the same spatial dimensions.
        dtype: Compute dtype for decoder activations. bfloat16 reduces memory;
            backbone and sigmoid always stay fp32. All outputs are fp32.
        fused_decode: If True, batch alpha+fg decoder upsamples to reduce
            Metal dispatch calls. Bit-exact with unfused path.
        slim: If True, forward returns only 4 final keys (drops intermediates).
            Reduces reference lifetime so MLX can reclaim buffers sooner.
        use_sdpa: If True, use mx.fast.scaled_dot_product_attention in backbone.
        stage_gc: If True, materialize + GC at backbone/decoder/refiner boundaries.
        refiner_dtype: Dtype for refiner weights+activations. float16 halves
            bandwidth at full resolution. None = same as backbone (fp32).
        wired_limit_bytes: Pin this many bytes as wired/resident Metal memory.
            Prevents OS paging and reduces p95 latency variance.
            None = no wiring (MLX default). 512 MiB covers model weights.
        cache_limit_bytes: Metal buffer cache size limit. Forces buffer reuse
            when cache exceeds this, reducing peak memory. None = unlimited
            (MLX default). 1.5 GiB balances reuse overhead vs peak memory.
    """
    if wired_limit_bytes is not None:
        mx.set_wired_limit(wired_limit_bytes)
    if cache_limit_bytes is not None:
        mx.set_cache_limit(cache_limit_bytes)

    model = GreenFormer(
        img_size=img_size,
        dtype=dtype,
        fused_decode=fused_decode,
        slim=slim,
        use_sdpa=use_sdpa,
        stage_gc=stage_gc,
        refiner_dtype=refiner_dtype,
    )
    model.load_checkpoint(checkpoint)
    if compile:
        model = compile_model(model, shapeless=shapeless)
    return model


def compile_model(model: GreenFormer, shapeless: bool = False) -> GreenFormer:
    """Wrap model forward pass with mx.compile for fused execution.

    Fixed-shape compile (shapeless=False) is safe for all inputs of the
    same resolution. Shapeless compile is experimental — the backbone uses
    shape-dependent reshapes that may trigger recompilation.
    """
    model._compiled = True
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
