#!/usr/bin/env python3
"""Exp 34: Benchmark single_shot mode peak memory.

Compares peak memory with and without backbone deletion after feature
extraction. single_shot=True breaks subsequent calls, so each measurement
uses a fresh model instance.

Usage:
    uv run python scripts/bench_single_shot.py
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from corridorkey_mlx.model.corridorkey import GreenFormer
from corridorkey_mlx.utils.layout import nchw_to_nhwc_np, nhwc_to_nchw_np

CHECKPOINT = Path("checkpoints/corridorkey_mlx.safetensors")
FIXTURE = Path("reference/fixtures/golden.npz")
RESOLUTION = 1024
FIDELITY_THRESHOLD = 0.050


def _materialize(*arrays: mx.array) -> None:
    """Force MLX array materialization (NOT Python builtin)."""
    mx.eval(*arrays)  # noqa: S307  -- mx.eval, not builtins


def measure_single_run(single_shot: bool) -> dict:
    """Measure peak memory and latency for a single forward pass."""
    gc.collect()
    mx.clear_cache()

    model = GreenFormer(
        img_size=RESOLUTION,
        slim=True,
        single_shot=single_shot,
        compile_backbone=False,
        compile_decoders=False,
        compile_refiner=False,
    )
    if CHECKPOINT.exists():
        model.load_checkpoint(CHECKPOINT)
    else:
        model.eval()
        _materialize(model.parameters())

    # Load golden fixture input
    ref = np.load(str(FIXTURE))
    if "input" in ref:
        x = mx.array(nchw_to_nhwc_np(ref["input"]))
    else:
        mx.random.seed(42)
        x = mx.random.normal((1, RESOLUTION, RESOLUTION, 4))
    _materialize(x)

    mx.reset_peak_memory()

    start = time.perf_counter()
    outputs = model(x)
    _materialize(outputs)
    latency_ms = (time.perf_counter() - start) * 1000.0

    peak_mb = round(mx.get_peak_memory() / (1024 * 1024), 1)

    # Fidelity check
    fidelity_ok = True
    max_errors = {}
    if FIXTURE.exists():
        for key in ["alpha_final", "fg_final"]:
            if key in ref and key in outputs:
                ref_t = ref[key]
                mlx_t = nhwc_to_nchw_np(np.array(outputs[key]))
                if ref_t.shape == mlx_t.shape:
                    max_err = float(np.max(np.abs(ref_t - mlx_t)))
                    max_errors[key] = round(max_err, 6)
                    if max_err >= FIDELITY_THRESHOLD:
                        fidelity_ok = False

    del model, outputs, x
    gc.collect()
    mx.clear_cache()

    return {
        "single_shot": single_shot,
        "peak_memory_mb": peak_mb,
        "latency_ms": round(latency_ms, 1),
        "fidelity_ok": fidelity_ok,
        "max_errors": max_errors,
    }


def main() -> None:
    print("Exp 34: Phased Model Deletion (single_shot mode)")
    print(f"Resolution: {RESOLUTION}x{RESOLUTION}")
    print()

    # Run multiple trials for stability
    trials = 3
    results = {"normal": [], "single_shot": []}

    for trial in range(trials):
        print(f"Trial {trial + 1}/{trials}:")

        r_normal = measure_single_run(single_shot=False)
        print(f"  normal:      peak={r_normal['peak_memory_mb']}MB  latency={r_normal['latency_ms']}ms  fidelity={'PASS' if r_normal['fidelity_ok'] else 'FAIL'}")
        results["normal"].append(r_normal)

        r_single = measure_single_run(single_shot=True)
        print(f"  single_shot: peak={r_single['peak_memory_mb']}MB  latency={r_single['latency_ms']}ms  fidelity={'PASS' if r_single['fidelity_ok'] else 'FAIL'}")
        results["single_shot"].append(r_single)

    # Summary
    normal_peak = min(r["peak_memory_mb"] for r in results["normal"])
    single_peak = min(r["peak_memory_mb"] for r in results["single_shot"])
    savings = normal_peak - single_peak

    print(f"\n{'='*60}")
    print(f"Normal peak memory:      {normal_peak} MB")
    print(f"Single-shot peak memory: {single_peak} MB")
    print(f"Savings:                 {savings} MB ({savings/normal_peak*100:.1f}%)")
    print(f"{'='*60}")

    # Save
    out_dir = Path("research/artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"exp34_single_shot_{int(time.time())}.json"
    out_path.write_text(json.dumps({
        "experiment_name": "exp34-phased-model-deletion",
        "resolution": RESOLUTION,
        "results": results,
        "normal_peak_mb": normal_peak,
        "single_shot_peak_mb": single_peak,
        "savings_mb": savings,
    }, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
