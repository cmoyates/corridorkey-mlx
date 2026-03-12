#!/usr/bin/env python3
"""Capture a Metal GPU trace of one CorridorKey MLX inference pass.

Produces a .gputrace bundle that can be opened in Xcode for kernel-level profiling.

Usage:
    MTL_CAPTURE_ENABLED=1 uv run python scripts/metal_trace.py
    MTL_CAPTURE_ENABLED=1 uv run python scripts/metal_trace.py --img-size 512
    MTL_CAPTURE_ENABLED=1 uv run python scripts/metal_trace.py --output my_trace.gputrace
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlx.core as mx

# Ensure package is importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from corridorkey_mlx.inference.pipeline import load_model

DEFAULT_IMG_SIZE = 1024
DEFAULT_CHECKPOINT = Path("checkpoints/corridorkey_mlx.safetensors")
DEFAULT_OUTPUT = "trace.gputrace"
WARMUP_RUNS = 3


def make_dummy_input(img_size: int) -> mx.array:
    """4-channel NHWC input (RGB + alpha hint)."""
    mx.random.seed(42)
    return mx.random.normal((1, img_size, img_size, 4))


def materialize(*args: mx.array | dict) -> None:
    """Force MLX lazy arrays to materialize on GPU."""
    mx.eval(*args)  # noqa: S307


def main() -> None:
    parser = argparse.ArgumentParser(description="Metal GPU trace for CorridorKey MLX")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="MLX safetensors checkpoint",
    )
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Output .gputrace path",
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"ERROR: checkpoint not found at {args.checkpoint}")
        sys.exit(1)

    trace_path = args.output

    print(f"Loading model (img_size={args.img_size})...")
    model = load_model(
        checkpoint=args.checkpoint,
        img_size=args.img_size,
        compile=False,  # eager — trace shows individual kernels
    )

    x = make_dummy_input(args.img_size)
    materialize(x)

    # --- Warmup ---
    print(f"Running {WARMUP_RUNS} warmup iterations...")
    for i in range(WARMUP_RUNS):
        out = model(x)
        materialize(out)
        print(f"  warmup {i + 1}/{WARMUP_RUNS} done")

    # --- Capture ---
    print(f"Starting Metal GPU capture -> {trace_path}")
    mx.metal.start_capture(trace_path)

    out = model(x)
    materialize(out)  # force all GPU work to complete inside capture
    mx.synchronize()  # ensure Metal command buffers are fully committed

    mx.metal.stop_capture()
    print("Capture complete.\n")

    # --- Instructions ---
    abs_path = str(Path(trace_path).resolve())
    print("=" * 60)
    print("HOW TO OPEN THE TRACE")
    print("=" * 60)
    print()
    print(f"  open {abs_path}")
    print()
    print("Or from Xcode:")
    print("  File -> Open... -> select the .gputrace bundle")
    print()
    print("In the GPU trace viewer, look for:")
    print("  - Longest-running compute kernels (sort by duration)")
    print("  - Memory bandwidth bottlenecks")
    print("  - Gaps between kernel dispatches (CPU-side overhead)")
    print("  - Kernel occupancy and thread utilization")
    print()


if __name__ == "__main__":
    main()
