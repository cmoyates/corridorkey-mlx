#!/usr/bin/env python3
"""Convert CorridorKey PyTorch checkpoint to MLX safetensors.

Usage:
    uv run --group reference python scripts/convert_weights.py \
        --checkpoint checkpoints/CorridorKey_v1.0.pth \
        --output checkpoints/corridorkey_mlx.safetensors
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table

from corridorkey_mlx.convert.converter import convert_checkpoint

console = Console()

DEFAULT_CHECKPOINT = Path("checkpoints/CorridorKey_v1.0.pth")
DEFAULT_OUTPUT = Path("checkpoints/corridorkey_mlx.safetensors")


def print_diagnostics(diagnostics: list) -> None:
    """Print conversion diagnostic table."""
    table = Table(title="Weight Conversion Report")
    table.add_column("Source Key", style="cyan", max_width=50)
    table.add_column("Dest Key", style="green", max_width=50)
    table.add_column("Src Shape", style="yellow")
    table.add_column("Dst Shape", style="yellow")
    table.add_column("Transform", style="magenta")

    for record in diagnostics:
        table.add_row(
            record.source_key,
            record.dest_key,
            str(record.source_shape),
            str(record.dest_shape),
            record.transform,
        )

    console.print(table)

    # Summary stats
    total = len(diagnostics)
    transposed = sum(1 for r in diagnostics if "conv_transpose" in r.transform)
    remapped = sum(1 for r in diagnostics if r.source_key != r.dest_key)
    console.print(f"\n[bold]Total keys:[/bold] {total}")
    console.print(f"[bold]Conv transposed:[/bold] {transposed}")
    console.print(f"[bold]Keys remapped:[/bold] {remapped}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert CorridorKey PyTorch → MLX weights")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to PyTorch .pth checkpoint",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output .safetensors path",
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        console.print(f"[red]Checkpoint not found: {args.checkpoint}[/red]")
        raise SystemExit(1)

    console.print(f"[bold]Converting:[/bold] {args.checkpoint}")
    console.print(f"[bold]Output:[/bold] {args.output}")

    diagnostics = convert_checkpoint(args.checkpoint, args.output)
    print_diagnostics(diagnostics)

    console.print(f"\n[green]Saved to {args.output}[/green]")


if __name__ == "__main__":
    main()
