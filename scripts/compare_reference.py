#!/usr/bin/env python3
"""Compare MLX outputs against PyTorch reference fixtures.

Reports max abs error and mean abs error per tensor.

Usage:
    uv run python scripts/compare_reference.py
    uv run python scripts/compare_reference.py --fixture reference/fixtures/golden.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from corridorkey_mlx.model.corridorkey import GreenFormer
from corridorkey_mlx.utils.layout import nchw_to_nhwc_np, nhwc_to_nchw_np

console = Console()

FIXTURE_DIR = Path("reference/fixtures")
DEFAULT_FIXTURE = FIXTURE_DIR / "golden.npz"
FIXTURE_2048 = FIXTURE_DIR / "golden_2048.npz"
DEFAULT_CHECKPOINT = Path("checkpoints/corridorkey_mlx.safetensors")

# Map fixture tensor names to model output keys
TENSOR_MAP = {
    "alpha_logits": "alpha_logits",
    "fg_logits": "fg_logits",
    "alpha_coarse": "alpha_coarse",
    "fg_coarse": "fg_coarse",
    "delta_logits": "delta_logits",
    "alpha_final": "alpha_final",
    "fg_final": "fg_final",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare MLX vs PyTorch reference")
    parser.add_argument("--fixture", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--img-size", type=int, default=512)
    args = parser.parse_args()

    # Auto-select fixture based on resolution if not explicitly provided
    if args.fixture is None:
        args.fixture = FIXTURE_2048 if args.img_size >= 2048 else DEFAULT_FIXTURE

    if not args.fixture.exists():
        console.print(f"[red]Fixture not found: {args.fixture}[/red]")
        console.print("Run: uv run python scripts/dump_pytorch_reference.py")
        raise SystemExit(1)
    if not args.checkpoint.exists():
        console.print(f"[red]Checkpoint not found: {args.checkpoint}[/red]")
        console.print("Run: uv run python scripts/convert_weights.py")
        raise SystemExit(1)

    # Load fixture
    ref = np.load(str(args.fixture))
    console.print(f"[bold]Loaded reference:[/bold] {args.fixture}")
    console.print(f"  Keys: {list(ref.keys())}")

    # Load model and run inference with fixture input
    model = GreenFormer(img_size=args.img_size)
    model.load_checkpoint(args.checkpoint)

    if "input" in ref:
        # Golden fixture stores NCHW (PyTorch); model expects NHWC
        x = mx.array(nchw_to_nhwc_np(ref["input"]))
    else:
        console.print("[yellow]No 'input' key in fixture, using random input[/yellow]")
        mx.random.seed(42)
        x = mx.random.normal((1, args.img_size, args.img_size, 4))

    outputs = model(x)
    # NOTE: mx.eval is MLX array materialization, not Python eval()
    mx.eval(outputs)  # noqa: S307

    # Compare
    table = Table(title="MLX vs PyTorch Reference")
    table.add_column("Tensor", justify="left")
    table.add_column("Shape", justify="left")
    table.add_column("Max Abs Err", justify="right")
    table.add_column("Mean Abs Err", justify="right")
    table.add_column("Status", justify="center")

    for ref_key, mlx_key in TENSOR_MAP.items():
        if ref_key not in ref:
            continue
        if mlx_key not in outputs:
            table.add_row(ref_key, "", "", "", "[yellow]MISSING[/yellow]")
            continue

        ref_tensor = ref[ref_key]
        # MLX outputs are NHWC; golden refs are NCHW — convert for comparison
        mlx_tensor = nhwc_to_nchw_np(np.array(outputs[mlx_key]))

        if ref_tensor.shape != mlx_tensor.shape:
            table.add_row(
                ref_key,
                f"{ref_tensor.shape} vs {mlx_tensor.shape}",
                "",
                "",
                "[red]SHAPE MISMATCH[/red]",
            )
            continue

        diff = np.abs(ref_tensor - mlx_tensor)
        max_err = float(np.max(diff))
        mean_err = float(np.mean(diff))
        status = "[green]OK[/green]" if max_err < 1e-3 else "[red]DRIFT[/red]"

        table.add_row(
            ref_key,
            str(ref_tensor.shape),
            f"{max_err:.2e}",
            f"{mean_err:.2e}",
            status,
        )

    console.print(table)


if __name__ == "__main__":
    main()
