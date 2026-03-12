#!/usr/bin/env python3
"""Exp 31: Fidelity Budget Audit.

Bisect bf16 conversions: disable each individually and measure
max_abs_error @1024 vs golden.npz to identify which conversion
consumes the most fidelity headroom.

Usage:
    uv run python scripts/audit_fidelity_budget.py
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

PARITY_TENSORS = ["alpha_final", "fg_final"]
FIDELITY_THRESHOLD = 0.050  # hard limit from plan

# Each config: (name, description, constructor overrides)
# "current" = all bf16 enabled (baseline error)
# Each variant disables ONE conversion to see how much error drops
CONFIGS = [
    (
        "all-bf16-enabled",
        "Current state -- all bf16 active",
        {},  # defaults
    ),
    (
        "no-backbone-bf16",
        "Disable backbone stages 1-3 bf16 cast",
        {"backbone_bf16_stages123": False},
    ),
    (
        "no-refiner-bf16",
        "Disable refiner bf16 (weights + inputs stay fp32)",
        {"refiner_dtype": None},
    ),
    (
        "no-decoder-bf16",
        "Disable decoder weight bf16 cast",
        {"decoder_dtype": None},
    ),
    (
        "no-refiner-no-decoder-bf16",
        "Disable both refiner and decoder bf16",
        {"refiner_dtype": None, "decoder_dtype": None},
    ),
    (
        "all-fp32",
        "Everything in fp32 -- maximum fidelity reference",
        {"backbone_bf16_stages123": False, "refiner_dtype": None, "decoder_dtype": None},
    ),
]

RESOLUTION = 1024


def _materialize(*arrays: mx.array) -> None:
    """Force MLX array materialization (NOT Python eval / builtin)."""
    mx.eval(*arrays)  # noqa: S307


def measure_fidelity(config_name: str, desc: str, overrides: dict) -> dict:
    """Build model with overrides, run golden input, measure error."""
    gc.collect()
    mx.clear_cache()

    print(f"\n--- {config_name}: {desc} ---")

    # Build model with overrides
    kwargs = {
        "img_size": RESOLUTION,
        "slim": True,
        "compile_backbone": False,
        "compile_decoders": False,
        "compile_refiner": False,
    }
    kwargs.update(overrides)
    model = GreenFormer(**kwargs)

    if CHECKPOINT.exists():
        model.load_checkpoint(CHECKPOINT)
    else:
        print("  WARNING: no checkpoint, using random weights")
        model.eval()
        _materialize(model.parameters())

    # Load golden fixture
    ref = np.load(str(FIXTURE))
    if "input" in ref:
        x = mx.array(nchw_to_nhwc_np(ref["input"]))
    else:
        mx.random.seed(42)
        x = mx.random.normal((1, RESOLUTION, RESOLUTION, 4))

    _materialize(x)

    # Run inference
    start = time.perf_counter()
    outputs = model(x)
    _materialize(outputs)
    latency_ms = (time.perf_counter() - start) * 1000.0

    # Measure error per tensor
    results = {"config": config_name, "description": desc, "latency_ms": round(latency_ms, 1)}
    for key in PARITY_TENSORS:
        if key not in ref or key not in outputs:
            results[key] = {"status": "MISSING"}
            continue

        ref_tensor = ref[key]
        mlx_tensor = nhwc_to_nchw_np(np.array(outputs[key]))

        if ref_tensor.shape != mlx_tensor.shape:
            results[key] = {"status": "SHAPE_MISMATCH"}
            continue

        diff = np.abs(ref_tensor - mlx_tensor)
        max_err = float(np.max(diff))
        mean_err = float(np.mean(diff))
        p99_err = float(np.percentile(diff, 99))

        results[key] = {
            "max_abs_error": round(max_err, 6),
            "mean_abs_error": round(mean_err, 8),
            "p99_abs_error": round(p99_err, 6),
            "within_budget": max_err < FIDELITY_THRESHOLD,
        }

        status = "OK" if max_err < FIDELITY_THRESHOLD else "OVER BUDGET"
        headroom_pct = round((1 - max_err / FIDELITY_THRESHOLD) * 100, 1)
        print(f"  {key}: max={max_err:.6f}  mean={mean_err:.8f}  p99={p99_err:.6f}  [{status}, {headroom_pct}% headroom]")

    del model, outputs, x
    gc.collect()
    mx.clear_cache()

    return results


def main() -> None:
    print("Exp 31: Fidelity Budget Audit")
    print(f"Resolution: {RESOLUTION}x{RESOLUTION}")
    print(f"Threshold: {FIDELITY_THRESHOLD}")
    print(f"Configs to test: {len(CONFIGS)}")

    all_results = []
    for name, desc, overrides in CONFIGS:
        result = measure_fidelity(name, desc, overrides)
        all_results.append(result)

    # Summary table
    print("\n" + "=" * 100)
    print(f"{'Config':<30} {'alpha max_abs':>14} {'fg max_abs':>14} {'alpha headroom':>16} {'fg headroom':>14}")
    print("-" * 100)

    for r in all_results:
        alpha_err = r.get("alpha_final", {}).get("max_abs_error", "N/A")
        fg_err = r.get("fg_final", {}).get("max_abs_error", "N/A")

        if isinstance(alpha_err, float):
            alpha_hr = f"{(1 - alpha_err / FIDELITY_THRESHOLD) * 100:.1f}%"
        else:
            alpha_hr = "N/A"

        if isinstance(fg_err, float):
            fg_hr = f"{(1 - fg_err / FIDELITY_THRESHOLD) * 100:.1f}%"
        else:
            fg_hr = "N/A"

        print(f"{r['config']:<30} {str(alpha_err):>14} {str(fg_err):>14} {alpha_hr:>16} {fg_hr:>14}")

    print("=" * 100)

    # Error contribution analysis
    if len(all_results) >= 2:
        baseline = all_results[0]  # all-bf16-enabled
        print("\n--- Error Contribution Analysis ---")
        print("(Error reduction when disabling each conversion vs all-bf16-enabled baseline)")
        print()

        for r in all_results[1:]:
            for key in PARITY_TENSORS:
                base_err = baseline.get(key, {}).get("max_abs_error")
                this_err = r.get(key, {}).get("max_abs_error")
                if isinstance(base_err, float) and isinstance(this_err, float):
                    reduction = base_err - this_err
                    pct = (reduction / base_err * 100) if base_err > 0 else 0
                    print(f"  {r['config']:<30} {key}: {reduction:+.6f} ({pct:+.1f}%)")

    # Save results
    out_dir = Path("research/artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"exp31_fidelity_audit_{int(time.time())}.json"
    out_path.write_text(json.dumps({
        "experiment_name": "exp31-fidelity-budget-audit",
        "resolution": RESOLUTION,
        "threshold": FIDELITY_THRESHOLD,
        "results": all_results,
    }, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
