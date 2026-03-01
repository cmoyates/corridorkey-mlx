#!/usr/bin/env python3
"""Single-image inference with CorridorKey MLX.

Usage:
    uv run python scripts/infer.py --image input.png --hint alpha_hint.png
    uv run python scripts/infer.py --image input.png --hint alpha_hint.png --output-dir results/
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mlx.core as mx

from corridorkey_mlx.inference.pipeline import DEFAULT_CHECKPOINT, DEFAULT_IMG_SIZE, load_model
from corridorkey_mlx.io.image import (
    load_alpha_hint,
    load_image,
    postprocess_alpha,
    postprocess_foreground,
    preprocess,
    save_alpha,
    save_foreground,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="CorridorKey MLX inference")
    parser.add_argument("--image", type=Path, required=True, help="RGB input image")
    parser.add_argument("--hint", type=Path, required=True, help="Alpha hint (grayscale)")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="MLX safetensors checkpoint",
    )
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE, help="Model input size")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Output directory")
    args = parser.parse_args()

    if not args.image.exists():
        print(f"Image not found: {args.image}")
        raise SystemExit(1)
    if not args.hint.exists():
        print(f"Alpha hint not found: {args.hint}")
        raise SystemExit(1)
    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Run: uv run python scripts/convert_weights.py")
        raise SystemExit(1)

    print(f"Loading model (img_size={args.img_size})...")
    t0 = time.perf_counter()
    model = load_model(args.checkpoint, args.img_size)
    print(f"  Model loaded in {time.perf_counter() - t0:.2f}s")

    print(f"Preprocessing {args.image}...")
    rgb = load_image(args.image)
    alpha_hint = load_alpha_hint(args.hint)
    x = preprocess(rgb, alpha_hint)
    # materialize input — mx.eval is MLX lazy graph eval, not Python eval
    mx.eval(x)  # noqa: S307

    print("Running inference...")
    t0 = time.perf_counter()
    outputs = model(x)
    # materialize outputs — mx.eval is MLX lazy graph eval, not Python eval
    mx.eval(outputs)  # noqa: S307
    print(f"  Inference in {time.perf_counter() - t0:.2f}s")

    # Print summary
    print("\nOutput tensors:")
    for key, arr in outputs.items():
        print(
            f"  {key:<20s} shape={arr.shape} "
            f"min={float(mx.min(arr)):.4f} max={float(mx.max(arr)):.4f}"
        )

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    alpha_arr = postprocess_alpha(outputs["alpha_final"])
    fg_arr = postprocess_foreground(outputs["fg_final"])

    alpha_path = args.output_dir / "alpha.png"
    fg_path = args.output_dir / "foreground.png"
    save_alpha(alpha_arr, alpha_path)
    save_foreground(fg_arr, fg_path)
    print(f"\nSaved: {alpha_path}, {fg_path}")


if __name__ == "__main__":
    main()
