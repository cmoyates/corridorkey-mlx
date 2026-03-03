"""CLI for downloading model weights from GitHub Releases.

Usage:
    python -m corridorkey_mlx.weights download [flags]
    corridorkey-weights download [flags]
"""

from __future__ import annotations

import argparse
import sys

from corridorkey_mlx.weights import download_weights


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="corridorkey-weights",
        description="Download corridorkey-mlx model weights from GitHub Releases.",
    )
    sub = parser.add_subparsers(dest="command")

    dl = sub.add_parser("download", help="Download weights from a GitHub Release")
    dl.add_argument(
        "--tag",
        default=None,
        help='Release tag (default: "latest", or $CORRIDORKEY_MLX_WEIGHTS_TAG)',
    )
    dl.add_argument(
        "--asset",
        default=None,
        dest="asset_name",
        help="Asset filename override (default: corridorkey_mlx.safetensors)",
    )
    dl.add_argument(
        "--out",
        default=None,
        help="Output directory (default: platform cache dir)",
    )
    dl.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if cached",
    )
    dl.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip SHA256 verification",
    )
    dl.add_argument(
        "--print-path",
        action="store_true",
        help="Print the resolved local path and exit (for scripting)",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "download":
        from pathlib import Path

        out = Path(args.out) if args.out else None
        path = download_weights(
            tag=args.tag,
            asset_name=args.asset_name,
            out=out,
            force=args.force,
            verify=not args.no_verify,
        )

        if args.print_path:
            # print to stdout (not stderr) for scripting
            print(path)
        else:
            from rich.console import Console

            Console(stderr=True).print(f"[green]Weights ready:[/green] {path}")


if __name__ == "__main__":
    main()
