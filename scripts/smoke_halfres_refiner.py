#!/usr/bin/env python3
"""Smoke test: half-res refiner fidelity impact.

Compares alpha_final/fg_final between full-res and half-res refiner
against golden.npz to measure actual error introduced.

Usage:
    uv run python scripts/smoke_halfres_refiner.py
    uv run python scripts/smoke_halfres_refiner.py --scale 0.75
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from corridorkey_mlx.model.corridorkey import GreenFormer
from corridorkey_mlx.utils.layout import nchw_to_nhwc_np, nhwc_to_nchw_np

CHECKPOINT = Path("checkpoints/corridorkey_mlx.safetensors")
GOLDEN = Path("reference/fixtures/golden.npz")
IMG_SIZE = 1024  # golden fixture is 1024x1024

CHECK_TENSORS = ["alpha_coarse", "fg_coarse", "alpha_final", "fg_final"]


def _materialize(*args: mx.array | dict) -> None:
    """Force MLX lazy arrays to materialize on GPU."""
    mx.eval(*args)  # noqa: S307  -- mx.eval, not Python eval()


def run_model(refiner_scale: float) -> dict[str, np.ndarray]:
    """Run model with given refiner_scale, return NCHW numpy outputs."""
    model = GreenFormer(img_size=IMG_SIZE, slim=True, refiner_scale=refiner_scale)
    model.load_checkpoint(CHECKPOINT)

    ref = np.load(str(GOLDEN))
    x = mx.array(nchw_to_nhwc_np(ref["input"]))

    start = time.perf_counter()
    out = model(x)
    _materialize(out)
    elapsed = (time.perf_counter() - start) * 1000.0

    print(f"  refiner_scale={refiner_scale}: {elapsed:.1f}ms")

    return {k: nhwc_to_nchw_np(np.array(out[k])) for k in CHECK_TENSORS if k in out}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=float, default=0.5)
    args = parser.parse_args()

    if not CHECKPOINT.exists():
        print(f"ERROR: checkpoint not found at {CHECKPOINT}")
        sys.exit(1)
    if not GOLDEN.exists():
        print(f"ERROR: golden fixture not found at {GOLDEN}")
        sys.exit(1)

    ref = np.load(str(GOLDEN))

    print("Running full-res baseline (refiner_scale=1.0)...")
    full = run_model(1.0)

    print(f"Running half-res (refiner_scale={args.scale})...")
    half = run_model(args.scale)

    print("\n=== Fidelity vs golden.npz ===")
    print(f"{'tensor':<16} {'full max_err':>12} {'half max_err':>12} {'half/full':>10}")
    print("-" * 54)

    all_pass = True
    for key in CHECK_TENSORS:
        if key not in ref:
            continue
        ref_t = ref[key]
        full_err = float(np.max(np.abs(ref_t - full[key])))
        half_err = float(np.max(np.abs(ref_t - half[key])))
        ratio = half_err / full_err if full_err > 0 else float("inf")
        status = "PASS" if half_err < 1e-1 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"{key:<16} {full_err:>12.6f} {half_err:>12.6f} {ratio:>9.1f}x  {status}")

    print(f"\n=== Half-res vs full-res (direct) ===")
    print(f"{'tensor':<16} {'max_abs_diff':>12} {'mean_abs_diff':>14}")
    print("-" * 44)
    for key in CHECK_TENSORS:
        if key not in full or key not in half:
            continue
        diff = np.abs(full[key] - half[key])
        print(f"{key:<16} {np.max(diff):>12.6f} {np.mean(diff):>14.8f}")

    print(f"\nOverall: {'PASS' if all_pass else 'FAIL'} (threshold=1e-1)")


if __name__ == "__main__":
    main()
