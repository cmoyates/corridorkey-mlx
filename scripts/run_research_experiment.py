#!/usr/bin/env python3
"""Run a research experiment: benchmark + parity check + structured output.

Wraps the repo's existing benchmark and parity surfaces into a single
structured JSON result for the autoresearch loop.

Usage:
    uv run python scripts/run_research_experiment.py
    uv run python scripts/run_research_experiment.py --experiment-name "refiner-bf16"
    uv run python scripts/run_research_experiment.py --resolution 512 --bench-runs 15
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from corridorkey_mlx.model.corridorkey import GreenFormer
from corridorkey_mlx.utils.layout import nchw_to_nhwc_np, nhwc_to_nchw_np

ARTIFACTS_DIR = Path("research/artifacts")
DEFAULT_CHECKPOINT = Path("checkpoints/corridorkey_mlx.safetensors")
DEFAULT_FIXTURE = Path("reference/fixtures/golden.npz")
DEFAULT_RESOLUTION = 512
DEFAULT_WARMUP = 3
DEFAULT_BENCH = 10

FIDELITY_THRESHOLD = 1e-3

PARITY_TENSORS = [
    "alpha_logits",
    "fg_logits",
    "alpha_coarse",
    "fg_coarse",
    "delta_logits",
    "alpha_final",
    "fg_final",
]


def get_peak_memory_mb() -> float:
    """Read peak Metal memory in MB."""
    try:
        return mx.get_peak_memory() / (1024 * 1024)
    except AttributeError:
        pass
    try:
        return mx.metal.get_peak_memory() / (1024 * 1024)
    except Exception:
        return 0.0


def reset_peak_memory() -> None:
    """Reset peak memory counter."""
    try:
        mx.reset_peak_memory()
    except AttributeError:
        pass
    try:
        mx.metal.reset_peak_memory()
    except Exception:
        pass


def _materialize(*arrays: mx.array) -> None:
    """Force MLX array materialization (NOT Python eval)."""
    mx.eval(*arrays)  # noqa: S307


def run_parity(
    model: GreenFormer, fixture_path: Path, img_size: int
) -> dict[str, dict[str, float | str]]:
    """Run parity check against golden reference. Returns per-tensor results."""
    if not fixture_path.exists():
        return {"error": "fixture not found"}

    ref = np.load(str(fixture_path))

    if "input" in ref:
        # Golden fixture stores NCHW (PyTorch); model expects NHWC
        x = mx.array(nchw_to_nhwc_np(ref["input"]))
    else:
        mx.random.seed(42)
        x = mx.random.normal((1, img_size, img_size, 4))

    outputs = model(x)
    _materialize(outputs)

    results = {}
    for key in PARITY_TENSORS:
        if key not in ref or key not in outputs:
            results[key] = {"status": "MISSING"}
            continue

        ref_tensor = ref[key]
        # MLX outputs are NHWC; golden refs are NCHW — convert for comparison
        mlx_tensor = nhwc_to_nchw_np(np.array(outputs[key]))

        if ref_tensor.shape != mlx_tensor.shape:
            results[key] = {
                "status": "SHAPE_MISMATCH",
                "ref_shape": list(ref_tensor.shape),
                "mlx_shape": list(mlx_tensor.shape),
            }
            continue

        diff = np.abs(ref_tensor - mlx_tensor)
        max_err = float(np.max(diff))
        mean_err = float(np.mean(diff))
        passed = max_err < FIDELITY_THRESHOLD

        results[key] = {
            "max_abs_error": max_err,
            "mean_abs_error": mean_err,
            "passed": passed,
            "status": "PASS" if passed else "FAIL",
        }

    return results


def run_benchmark(
    model: GreenFormer,
    img_size: int,
    warmup_runs: int,
    bench_runs: int,
) -> dict[str, float | list[float]]:
    """Run latency benchmark. Returns timing results."""
    mx.random.seed(42)
    x = mx.random.normal((1, img_size, img_size, 4))
    _materialize(x)

    def run() -> None:
        out = model(x)
        _materialize(out)

    # Warmup
    warmup_start = time.perf_counter()
    run()
    warmup_ms = (time.perf_counter() - warmup_start) * 1000.0

    for _ in range(warmup_runs - 1):
        run()

    # Benchmark
    times: list[float] = []
    for _ in range(bench_runs):
        start = time.perf_counter()
        run()
        times.append((time.perf_counter() - start) * 1000.0)

    times.sort()
    median_ms = times[len(times) // 2]
    p95_idx = max(0, int(len(times) * 0.95) - 1)
    p95_ms = times[p95_idx]
    min_ms = times[0]

    return {
        "warmup_ms": round(warmup_ms, 2),
        "median_ms": round(median_ms, 2),
        "p95_ms": round(p95_ms, 2),
        "min_ms": round(min_ms, 2),
        "all_times_ms": [round(t, 2) for t in times],
    }


def measure_peak_memory(checkpoint: Path | None, img_size: int) -> float:
    """Measure peak memory on a fresh model instance."""
    gc.collect()
    mx.clear_cache()

    model = GreenFormer(img_size=img_size, slim=True)
    if checkpoint and checkpoint.exists():
        model.load_checkpoint(checkpoint)
    else:
        model.eval()
        _materialize(model.parameters())

    mx.random.seed(42)
    x = mx.random.normal((1, img_size, img_size, 4))
    _materialize(x)

    reset_peak_memory()
    out = model(x)
    _materialize(out)

    peak_mb = get_peak_memory_mb()

    del out, model, x
    gc.collect()
    mx.clear_cache()

    return round(peak_mb, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run research experiment")
    parser.add_argument("--experiment-name", default="unnamed")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--warmup-runs", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--bench-runs", type=int, default=DEFAULT_BENCH)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    ckpt = args.checkpoint if args.checkpoint.exists() else None
    timestamp = datetime.now(timezone.utc).isoformat()

    print(f"Experiment: {args.experiment_name}")
    print(f"Resolution: {args.resolution}x{args.resolution}")
    print(f"Checkpoint: {ckpt or 'random weights'}")
    print()

    # Build model
    model = GreenFormer(img_size=args.resolution, slim=True)
    if ckpt:
        model.load_checkpoint(ckpt)
    else:
        model.eval()
        _materialize(model.parameters())

    # Parity check
    print("Running parity check...")
    parity = run_parity(model, args.fixture, args.resolution)
    fidelity_passed = all(
        v.get("passed", False) for v in parity.values() if isinstance(v, dict) and "passed" in v
    )
    print(f"  Fidelity: {'PASS' if fidelity_passed else 'FAIL'}")

    # Benchmark
    print("Running benchmark...")
    bench = run_benchmark(model, args.resolution, args.warmup_runs, args.bench_runs)
    print(f"  Median: {bench['median_ms']}ms, P95: {bench['p95_ms']}ms")

    # Peak memory (fresh instance)
    print("Measuring peak memory...")
    del model
    gc.collect()
    mx.clear_cache()
    peak_mb = measure_peak_memory(ckpt, args.resolution)
    print(f"  Peak: {peak_mb}MB")

    # Assemble result
    result = {
        "experiment_name": args.experiment_name,
        "timestamp": timestamp,
        "resolution": args.resolution,
        "warmup_runs": args.warmup_runs,
        "bench_runs": args.bench_runs,
        "checkpoint": str(ckpt) if ckpt else None,
        "fidelity_passed": fidelity_passed,
        "parity": parity,
        "benchmark": bench,
        "peak_memory_mb": peak_mb,
    }

    # Write output
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.output:
        out_path = args.output
    else:
        safe_name = args.experiment_name.replace(" ", "_").replace("/", "_")
        out_path = ARTIFACTS_DIR / f"{safe_name}_{int(time.time())}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nResult written to {out_path}")

    if not fidelity_passed:
        print("\nFIDELITY FAILED — candidate must be reverted")
        sys.exit(1)


if __name__ == "__main__":
    main()
