#!/usr/bin/env python3
"""PyTorch reference inference on real images.

Usage:
    uv run --group reference python scripts/infer_pytorch.py \
        --image samples/sample.png --hint samples/hint.png
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Reuse the GreenFormer and loading logic from dump script
from dump_pytorch_reference import IMG_SIZE, GreenFormer, load_checkpoint
from PIL import Image

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)


def preprocess(
    image_path: Path,
    hint_path: Path,
    img_size: int,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Load and preprocess image + hint into a 4ch input tensor."""
    rgb = Image.open(image_path).convert("RGB")
    hint = Image.open(hint_path).convert("L")
    original_size = rgb.size[::-1]  # (H, W)

    rgb_resized = rgb.resize((img_size, img_size), Image.BILINEAR)
    hint_resized = hint.resize((img_size, img_size), Image.BILINEAR)

    rgb_t = torch.from_numpy(np.asarray(rgb_resized)).float().permute(2, 0, 1) / 255.0
    hint_t = torch.from_numpy(np.asarray(hint_resized)).float().unsqueeze(0) / 255.0

    # ImageNet normalize RGB
    rgb_t = (rgb_t.unsqueeze(0) - IMAGENET_MEAN) / IMAGENET_STD
    hint_t = hint_t.unsqueeze(0)

    return torch.cat([rgb_t, hint_t], dim=1), original_size


def main() -> None:
    parser = argparse.ArgumentParser(description="PyTorch CorridorKey inference")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--hint", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/CorridorKey_v1.0.pth"),
    )
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--output-dir", type=Path, default=Path("samples/output_pytorch"))
    args = parser.parse_args()

    torch.set_grad_enabled(False)

    print(f"Building model (img_size={args.img_size})...")
    model = GreenFormer(img_size=args.img_size)
    model.eval()

    print(f"Loading checkpoint: {args.checkpoint}")
    load_checkpoint(model, args.checkpoint)

    print("Preprocessing...")
    input_tensor, original_size = preprocess(args.image, args.hint, args.img_size)
    print(f"  Input shape: {input_tensor.shape}")

    print("Running inference...")
    t0 = time.perf_counter()
    outputs = model(input_tensor)
    elapsed = time.perf_counter() - t0
    print(f"  Inference: {elapsed:.2f}s")

    # Extract final predictions
    alpha = outputs["alpha_final"]  # [1, 1, H, W]
    fg = outputs["fg_final"]  # [1, 3, H, W]

    # Upsample to original resolution
    alpha = F.interpolate(alpha, size=original_size, mode="bilinear", align_corners=False)
    fg = F.interpolate(fg, size=original_size, mode="bilinear", align_corners=False)

    # To uint8 numpy
    alpha_np = (alpha[0, 0].clamp(0, 1).numpy() * 255).astype(np.uint8)
    fg_np = (fg[0].clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    # Composite over green
    alpha_f = alpha_np.astype(np.float32) / 255.0
    green = np.array([0, 177, 64], dtype=np.float32)
    comp_np = fg_np.astype(np.float32) * alpha_f[..., None] + green * (1 - alpha_f[..., None])
    comp_np = comp_np.clip(0, 255).astype(np.uint8)

    # Save
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(alpha_np, mode="L").save(out / "alpha.png")
    Image.fromarray(fg_np, mode="RGB").save(out / "fg.png")
    Image.fromarray(comp_np, mode="RGB").save(out / "comp.png")

    print(f"Saved to {out}/")
    for name, arr in [("alpha", alpha_np), ("fg", fg_np), ("comp", comp_np)]:
        print(f"  {name}: {arr.shape} range=[{arr.min()}, {arr.max()}]")


if __name__ == "__main__":
    main()
