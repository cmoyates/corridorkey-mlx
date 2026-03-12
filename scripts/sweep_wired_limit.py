#!/usr/bin/env python3
"""Exp 30: Sweep mx.set_wired_limit() values and benchmark.

Tests whether pinning model weights in physical RAM reduces p95 variance
and/or improves steady-state latency.

Usage:
    uv run python scripts/sweep_wired_limit.py
    uv run python scripts/sweep_wired_limit.py --resolution 512
"""

from __future__ import annotations

import argparse
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

DEFAULT_CHECKPOINT = Path("checkpoints/corridorkey_mlx.safetensors")
DEFAULT_FIXTURE = Path("reference/fixtures/golden.npz")

WIRED_LIMIT_MB_VALUES = [0, 512, 1024, 1536, 2048, 3072, 4096]

WARMUP_RUNS = 5
BENCH_RUNS = 20

PARITY_TENSORS = ["alpha_final", "fg_final"]
FIDELITY_THRESHOLD = 1e-1


def _materialize(*arrays: mx.array) -> None:
    """Force MLX array materialization (NOT Python eval)."""
    mx.eval(*arrays)  # noqa: S307


def run_single_benchmark(
    wired_limit_mb: int,
    checkpoint: Path,
    fixture: Path,
    resolution: int,
) -> dict:
    """Run benchmark at a specific wired_limit value."""
    # Clean slate
    gc.collect()
    mx.clear_cache()

    # Set wired limit
    mx.set_wired_limit(wired_limit_mb * 1024 * 1024)

    # Build and load model
    model = GreenFormer(img_size=resolution, slim=True)
    if checkpoint.exists():
        model.load_checkpoint(checkpoint)
    else:
        model.eval()
        _materialize(model.parameters())

    # Input
    mx.random.seed(42)
    x = mx.random.normal((1, resolution, resolution, 4))
    _materialize(x)

    def run() -> dict[str, mx.array]:
        return model(x)

    # Warmup
    warmup_start = time.perf_counter()
    out = run()
    _materialize(out)
    warmup_ms = (time.perf_counter() - warmup_start) * 1000.0

    for _ in range(WARMUP_RUNS - 1):
        out = run()
        _materialize(out)

    # Benchmark
    times: list[float] = []
    for _ in range(BENCH_RUNS):
        start = time.perf_counter()
        out = run()
        _materialize(out)
        times.append((time.perf_counter() - start) * 1000.0)

    times.sort()
    median_ms = times[len(times) // 2]
    p95_idx = max(0, int(len(times) * 0.95) - 1)
    p95_ms = times[p95_idx]
    min_ms = times[0]
    stddev_ms = float(np.std(times))

    # Peak memory (fresh instance)
    del model, out, x
    gc.collect()
    mx.clear_cache()

    mx.set_wired_limit(wired_limit_mb * 1024 * 1024)
    model2 = GreenFormer(img_size=resolution, slim=True)
    if checkpoint.exists():
        model2.load_checkpoint(checkpoint)
    else:
        model2.eval()
        _materialize(model2.parameters())

    mx.random.seed(42)
    x2 = mx.random.normal((1, resolution, resolution, 4))
    _materialize(x2)

    mx.reset_peak_memory()
    out2 = model2(x2)
    _materialize(out2)
    peak_mb = round(mx.get_peak_memory() / (1024 * 1024), 1)

    # Quick fidelity check
    fidelity_ok = True
    if fixture.exists():
        ref = np.load(str(fixture))
        if "input" in ref:
            xf = mx.array(nchw_to_nhwc_np(ref["input"]))
        else:
            mx.random.seed(42)
            xf = mx.random.normal((1, resolution, resolution, 4))

        fid_out = model2(xf)
        _materialize(fid_out)

        for key in PARITY_TENSORS:
            if key in ref and key in fid_out:
                ref_t = ref[key]
                mlx_t = nhwc_to_nchw_np(np.array(fid_out[key]))
                if ref_t.shape == mlx_t.shape:
                    max_err = float(np.max(np.abs(ref_t - mlx_t)))
                    if max_err >= FIDELITY_THRESHOLD:
                        fidelity_ok = False

    del model2, out2, x2
    gc.collect()
    mx.clear_cache()

    # Reset wired limit
    mx.set_wired_limit(0)

    return {
        "wired_limit_mb": wired_limit_mb,
        "warmup_ms": round(warmup_ms, 2),
        "median_ms": round(median_ms, 2),
        "p95_ms": round(p95_ms, 2),
        "min_ms": round(min_ms, 2),
        "stddev_ms": round(stddev_ms, 2),
        "peak_memory_mb": peak_mb,
        "fidelity_ok": fidelity_ok,
        "all_times_ms": [round(t, 2) for t in times],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp 30: mx.set_wired_limit() sweep")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument(
        "--limits",
        type=int,
        nargs="+",
        default=WIRED_LIMIT_MB_VALUES,
        help="Wired limit values in MB to sweep",
    )
    args = parser.parse_args()

    print(f"Exp 30: mx.set_wired_limit() sweep")
    print(f"Resolution: {args.resolution}x{args.resolution}")
    print(f"Limits (MB): {args.limits}")
    print(f"Warmup: {WARMUP_RUNS}, Bench: {BENCH_RUNS}")
    print()

    results = []
    for limit_mb in args.limits:
        print(f"--- wired_limit = {limit_mb} MB ---")
        result = run_single_benchmark(limit_mb, args.checkpoint, args.fixture, args.resolution)
        results.append(result)
        print(
            f"  median={result['median_ms']}ms  p95={result['p95_ms']}ms  "
            f"stddev={result['stddev_ms']}ms  peak_mem={result['peak_memory_mb']}MB  "
            f"fidelity={'PASS' if result['fidelity_ok'] else 'FAIL'}"
        )
        print()

    # Summary table
    print("=" * 80)
    print(f"{'Wired Limit':>12} {'Median':>10} {'P95':>10} {'StdDev':>10} {'Peak Mem':>10} {'Fidelity':>10}")
    print("-" * 80)
    baseline_median = results[0]["median_ms"]  # 0 MB = baseline
    for r in results:
        delta = ((r["median_ms"] - baseline_median) / baseline_median * 100) if baseline_median > 0 else 0
        sign = "+" if delta > 0 else ""
        print(
            f"{r['wired_limit_mb']:>9} MB "
            f"{r['median_ms']:>8.1f}ms "
            f"{r['p95_ms']:>8.1f}ms "
            f"{r['stddev_ms']:>8.2f}ms "
            f"{r['peak_memory_mb']:>7.0f}MB "
            f"{'PASS' if r['fidelity_ok'] else 'FAIL':>10}"
            f"  ({sign}{delta:.1f}%)"
        )
    print("=" * 80)

    # Find best
    valid = [r for r in results if r["fidelity_ok"]]
    if valid:
        best = min(valid, key=lambda r: r["median_ms"])
        print(f"\nBest: wired_limit={best['wired_limit_mb']}MB -> {best['median_ms']}ms median")

    # Save results
    out_dir = Path("research/artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"exp30_wired_limit_sweep_{int(time.time())}.json"
    out_path.write_text(json.dumps({
        "experiment_name": "exp30-wired-limit-sweep",
        "resolution": args.resolution,
        "warmup_runs": WARMUP_RUNS,
        "bench_runs": BENCH_RUNS,
        "sweep_results": results,
    }, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
