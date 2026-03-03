"""Entry point for `python -m corridorkey_mlx`.

Routes subcommands:
    python -m corridorkey_mlx weights download [flags]
"""

from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "weights":
        from corridorkey_mlx.weights_cli import main as weights_main

        weights_main(sys.argv[2:])
    else:
        print("Usage: python -m corridorkey_mlx weights download [flags]")
        print()
        print("Subcommands:")
        print("  weights download   Download model weights from GitHub Releases")
        sys.exit(1)


if __name__ == "__main__":
    main()
