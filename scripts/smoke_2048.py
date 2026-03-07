"""2048 smoke test — validates end-to-end MLX inference at native resolution.

Loads a real checkpoint, runs inference at 2048x2048 (CorridorKey's training
resolution), reports timing, peak memory, output diagnostics. Uses sample
images from samples/ by default; falls back to synthetic if unavailable.

This is an execution/stability check, not a parity campaign.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from corridorkey_mlx import CorridorKeyMLXEngine
from corridorkey_mlx.utils.profiling import memory_snapshot, reset_peak

DEFAULT_CHECKPOINT = Path("checkpoints/corridorkey_mlx.safetensors")
DEFAULT_IMG_SIZE = 2048
DEFAULT_OUTPUT_DIR = Path("output/smoke_2048")
DEFAULT_SEED = 42
DEFAULT_SAMPLE_IMAGE = Path("samples/sample.png")
DEFAULT_SAMPLE_HINT = Path("samples/hint.png")


def generate_synthetic_inputs(img_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate deterministic synthetic RGB + alpha hint inputs.

    RGB: uniform random uint8.
    Hint: radial gradient (bright center, dark edges) — more realistic than noise.
    """
    rng = np.random.default_rng(seed)
    rgb = rng.integers(0, 256, (img_size, img_size, 3), dtype=np.uint8)

    # Radial gradient hint: bright center fading to dark edges
    y, x = np.mgrid[:img_size, :img_size]
    center = img_size / 2.0
    distance = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    max_distance = np.sqrt(2) * center
    gradient = 1.0 - (distance / max_distance)
    mask = (gradient * 255).clip(0, 255).astype(np.uint8)

    return rgb, mask


def load_user_inputs(image_path: Path, hint_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load user-supplied RGB image and alpha hint."""
    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    mask = np.asarray(Image.open(hint_path).convert("L"))
    return rgb, mask


def report_diagnostics(result: dict[str, np.ndarray]) -> bool:
    """Print output diagnostics. Returns True if outputs look healthy."""
    healthy = True
    for key in ("alpha", "fg", "comp"):
        arr = result[key]
        has_nan = bool(np.isnan(arr).any())
        has_inf = bool(np.isinf(arr).any())
        print(
            f"  {key:10s}: shape={arr.shape}  dtype={arr.dtype}  range=[{arr.min()}, {arr.max()}]"
        )
        if has_nan:
            print(f"  WARNING: {key} contains NaN!")
            healthy = False
        if has_inf:
            print(f"  WARNING: {key} contains Inf!")
            healthy = False

    # Flag suspicious alpha patterns
    alpha = result["alpha"]
    if alpha.min() == alpha.max():
        print(f"  WARNING: alpha is constant ({alpha.min()}) — suspicious")
        healthy = False
    if alpha.min() == 0 and alpha.max() == 0:
        print("  WARNING: alpha is all-zeros")
        healthy = False
    if alpha.min() == 255 and alpha.max() == 255:
        print("  WARNING: alpha is all-ones (255)")
        healthy = False

    return healthy


def main() -> None:
    parser = argparse.ArgumentParser(
        description="2048 smoke test: CorridorKeyMLXEngine at native resolution"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="MLX safetensors checkpoint",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=DEFAULT_IMG_SIZE,
        help="Model input resolution (default: 2048)",
    )
    parser.add_argument("--image", type=Path, default=None, help="RGB input image")
    parser.add_argument("--hint", type=Path, default=None, help="Grayscale alpha hint")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for saved PNGs",
    )
    parser.add_argument(
        "--save-outputs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save output PNGs (default: True)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use mx.compile (default: True)",
    )
    args = parser.parse_args()

    # -- validate checkpoint --
    if not args.checkpoint.exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        print("Run scripts/convert_weights.py first.")
        sys.exit(1)

    # -- resolve inputs: explicit args > samples/ > synthetic --
    image_path = args.image
    hint_path = args.hint

    if image_path is None and DEFAULT_SAMPLE_IMAGE.exists():
        image_path = DEFAULT_SAMPLE_IMAGE
    if hint_path is None and image_path is not None and DEFAULT_SAMPLE_HINT.exists():
        hint_path = DEFAULT_SAMPLE_HINT

    if image_path is not None:
        if hint_path is None:
            print("ERROR: --hint required when --image is provided (or place samples/hint.png)")
            sys.exit(1)
        print(f"Loading inputs: image={image_path}, hint={hint_path}")
        rgb, mask = load_user_inputs(image_path, hint_path)
        using_synthetic = False
    else:
        print(f"Generating synthetic {args.img_size}x{args.img_size} inputs (seed={args.seed})...")
        rgb, mask = generate_synthetic_inputs(args.img_size, args.seed)
        using_synthetic = True

    print(f"Input RGB:  {rgb.shape} {rgb.dtype}")
    print(f"Input mask: {mask.shape} {mask.dtype}")

    # -- load engine --
    print(f"Loading engine (img_size={args.img_size}, compile={args.compile})...")
    try:
        engine = CorridorKeyMLXEngine(
            checkpoint_path=args.checkpoint,
            img_size=args.img_size,
            compile=args.compile,
        )
    except Exception as exc:
        print(f"ERROR loading engine: {exc}")
        sys.exit(1)

    # -- run inference --
    print("Running inference...")
    reset_peak()
    start = time.perf_counter()

    try:
        result = engine.process_frame(rgb, mask)
    except (RuntimeError, MemoryError) as exc:
        elapsed = time.perf_counter() - start
        mem = memory_snapshot()
        print(f"\nFAILED after {elapsed:.1f}s")
        print(f"Peak memory: {mem.peak_mb:.0f} MB")
        print(f"Error: {exc}")
        print("\nPossible causes:")
        print("  - Insufficient unified memory for 2048 inference")
        print("  - Try --no-compile to reduce memory overhead")
        print("  - Try --img-size 1024 to halve resolution")
        sys.exit(1)

    elapsed = time.perf_counter() - start
    mem = memory_snapshot()

    # -- report --
    print(f"\nInference completed in {elapsed:.2f}s")
    print(f"Peak memory:   {mem.peak_mb:.0f} MB")
    print(f"Active memory: {mem.active_mb:.0f} MB")
    print(f"Cache memory:  {mem.cache_mb:.0f} MB")
    source_label = "synthetic" if using_synthetic else f"loaded ({image_path})"
    print(f"Input: {source_label}, model res={args.img_size}x{args.img_size}")
    print("\nOutputs:")
    healthy = report_diagnostics(result)

    # -- save outputs --
    if args.save_outputs:
        out = args.output_dir
        out.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result["alpha"], mode="L").save(out / "alpha.png")
        Image.fromarray(result["fg"], mode="RGB").save(out / "fg.png")
        Image.fromarray(result["comp"], mode="RGB").save(out / "comp.png")
        print(f"\nSaved outputs to {out}/")

    # -- verdict --
    if healthy:
        print("\n2048 smoke test PASSED.")
    else:
        print("\n2048 smoke test completed with WARNINGS (see above).")
        sys.exit(1)


if __name__ == "__main__":
    main()
