"""Smoke test for the CorridorKeyMLXEngine integration surface.

Instantiates the engine, runs one frame, prints output shapes and value ranges.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from corridorkey_mlx import CorridorKeyMLXEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test: CorridorKeyMLXEngine")
    parser.add_argument("--image", type=Path, required=True, help="RGB input image")
    parser.add_argument("--hint", type=Path, required=True, help="Grayscale alpha hint")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/corridorkey_mlx.safetensors"),
    )
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    print(f"Loading engine (img_size={args.img_size})...")
    engine = CorridorKeyMLXEngine(
        checkpoint_path=args.checkpoint,
        img_size=args.img_size,
        compile=False,
    )

    rgb = np.asarray(Image.open(args.image).convert("RGB"))
    mask = np.asarray(Image.open(args.hint).convert("L"))
    print(f"Input image: {rgb.shape} {rgb.dtype}")
    print(f"Input mask:  {mask.shape} {mask.dtype}")

    result = engine.process_frame(rgb, mask)

    for key, arr in result.items():
        print(f"  {key}: shape={arr.shape} dtype={arr.dtype} range=[{arr.min()}, {arr.max()}]")

    if args.output_dir is not None:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result["alpha"], mode="L").save(out / "alpha.png")
        Image.fromarray(result["fg"], mode="RGB").save(out / "fg.png")
        Image.fromarray(result["comp"], mode="RGB").save(out / "comp.png")
        print(f"Saved outputs to {out}")

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
