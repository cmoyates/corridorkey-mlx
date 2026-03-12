#!/usr/bin/env python3
"""Micro-benchmark: Do MLX streams overlap GPU work?

Tests whether two independent matmuls on separate GPU streams execute
faster than sequentially on one stream. Answers whether Apple Silicon
actually parallelizes across Metal command queues for compute.

Usage:
    uv run python scripts/bench_stream_overlap.py
"""

from __future__ import annotations

import time

import mlx.core as mx

WARMUP_RUNS = 5
BENCH_RUNS = 20

# Decoder-scale: (16384, 256) @ (256, 256)
SMALL_N = 16384
SMALL_DIM = 256

# Backbone-scale: (16384, 896) @ (896, 896)
BIG_N = 16384
BIG_DIM = 896


def _materialize(*arrays: mx.array) -> None:
    """Force MLX array materialization (NOT Python eval)."""
    mx.eval(*arrays)  # noqa: S307


def bench_median(fn: object, warmup: int = WARMUP_RUNS, runs: int = BENCH_RUNS) -> float:
    """Warmup + benchmark, return median ms."""
    for _ in range(warmup):
        fn()  # type: ignore[operator]
    times: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()  # type: ignore[operator]
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        times.append(elapsed_ms)
    times.sort()
    return times[len(times) // 2]


def test_stream_overlap(n: int, dim: int, label: str) -> None:
    """Compare sequential vs parallel stream dispatch for two independent matmuls."""
    mx.random.seed(42)
    a = mx.random.normal((n, dim))
    b = mx.random.normal((dim, dim))
    c = mx.random.normal((n, dim))
    d = mx.random.normal((dim, dim))
    _materialize(a, b, c, d)

    stream2 = mx.new_stream(mx.gpu)

    def sequential() -> None:
        out1 = a @ b
        out2 = c @ d
        _materialize(out1, out2)

    def parallel() -> None:
        out1 = a @ b
        with mx.stream(stream2):
            out2 = c @ d
        _materialize(out1, out2)

    seq_ms = bench_median(sequential)
    par_ms = bench_median(parallel)
    speedup = seq_ms / par_ms if par_ms > 0 else 0.0

    print(f"\n--- {label} ({n}x{dim} @ {dim}x{dim}) ---")
    print(f"  Sequential: {seq_ms:.2f} ms")
    print(f"  Parallel:   {par_ms:.2f} ms")
    print(f"  Speedup:    {speedup:.3f}x")

    if speedup > 1.10:
        print("  -> OVERLAP DETECTED (>10% speedup)")
    elif speedup > 1.05:
        print("  -> MARGINAL overlap (5-10%)")
    else:
        print("  -> NO meaningful overlap (<5%)")


def test_four_matmuls(n: int, dim: int) -> None:
    """Test 4 independent matmuls: 1 stream vs 2 streams vs 4 streams."""
    mx.random.seed(42)
    mats = [(mx.random.normal((n, dim)), mx.random.normal((dim, dim))) for _ in range(4)]
    _materialize(*[m for pair in mats for m in pair])

    streams = [mx.new_stream(mx.gpu) for _ in range(3)]

    def one_stream() -> None:
        outs = [a @ b for a, b in mats]
        _materialize(*outs)

    def two_streams() -> None:
        out0 = mats[0][0] @ mats[0][1]
        out1 = mats[1][0] @ mats[1][1]
        with mx.stream(streams[0]):
            out2 = mats[2][0] @ mats[2][1]
            out3 = mats[3][0] @ mats[3][1]
        _materialize(out0, out1, out2, out3)

    def four_streams() -> None:
        outs = []
        for i, (a, b) in enumerate(mats):
            if i == 0:
                outs.append(a @ b)
            else:
                with mx.stream(streams[i - 1]):
                    outs.append(a @ b)
        _materialize(*outs)

    one_ms = bench_median(one_stream)
    two_ms = bench_median(two_streams)
    four_ms = bench_median(four_streams)

    print(f"\n--- 4 matmuls ({n}x{dim}) ---")
    print(f"  1 stream:  {one_ms:.2f} ms")
    print(f"  2 streams: {two_ms:.2f} ms (speedup: {one_ms / two_ms:.3f}x)")
    print(f"  4 streams: {four_ms:.2f} ms (speedup: {one_ms / four_ms:.3f}x)")


def main() -> None:
    print("MLX Stream Overlap Benchmark")
    print("=" * 40)

    # Test at decoder scale
    test_stream_overlap(SMALL_N, SMALL_DIM, "Decoder-scale")

    # Test at backbone scale
    test_stream_overlap(BIG_N, BIG_DIM, "Backbone-scale")

    # Test with 4 matmuls (simulates alpha+fg decoder pair)
    test_four_matmuls(SMALL_N, SMALL_DIM)

    print("\n" + "=" * 40)
    print("Decision: speedup > 1.10 = proceed with Exp 38")
    print("          speedup < 1.05 = abandon Exp 38")


if __name__ == "__main__":
    main()
