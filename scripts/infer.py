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

from corridorkey_mlx.inference.pipeline import (
    DEFAULT_CHECKPOINT,
    DEFAULT_IMG_SIZE,
    infer_and_save,
    load_model,
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

    print(f"Running inference on {args.image}...")
    t0 = time.perf_counter()
    results = infer_and_save(model, args.image, args.hint, args.output_dir)
    print(f"  Inference + save in {time.perf_counter() - t0:.2f}s")

    print(f"\nSaved: {args.output_dir / 'alpha.png'}, {args.output_dir / 'foreground.png'}")
    print(f"  Alpha shape: {results['alpha'].shape}")
    print(f"  Foreground shape: {results['foreground'].shape}")


if __name__ == "__main__":
    main()
