"""Profiling utilities for MLX inference benchmarking.

Provides timing helpers that force mx.eval() for accurate measurement,
since MLX uses lazy evaluation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import mlx.core as mx


@dataclass
class TimingResult:
    """Result from a timed inference run."""

    elapsed_ms: float
    label: str

    def __repr__(self) -> str:
        return f"{self.label}: {self.elapsed_ms:.1f} ms"


def time_fn(fn: object, *args: object, label: str = "run", **kwargs: object) -> TimingResult:
    """Time a function that returns MLX arrays, forcing evaluation.

    Forces mx.eval on the result to ensure the computation graph
    is fully executed before stopping the timer.
    """
    start = time.perf_counter()
    result = fn(*args, **kwargs)  # type: ignore[operator]

    # Force evaluation of all MLX arrays in the result
    # NOTE: mx.eval is MLX's array materialization, not Python eval()
    if isinstance(result, mx.array):
        mx.eval(result)  # noqa: S307
    elif isinstance(result, dict):
        arrays = [v for v in result.values() if isinstance(v, mx.array)]
        if arrays:
            mx.eval(*arrays)  # noqa: S307
    elif isinstance(result, (list, tuple)):
        arrays = [v for v in result if isinstance(v, mx.array)]
        if arrays:
            mx.eval(*arrays)  # noqa: S307

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return TimingResult(elapsed_ms=elapsed_ms, label=label)


def warmup_and_bench(
    fn: object,
    *args: object,
    warmup_runs: int = 3,
    bench_runs: int = 10,
    label: str = "bench",
    **kwargs: object,
) -> tuple[TimingResult, TimingResult, list[float]]:
    """Run warmup iterations then benchmark, reporting both.

    Returns:
        (warmup_first_run, steady_state_median, all_bench_times_ms)
    """
    # Warmup — first run captures compile cost if using mx.compile
    warmup_first = time_fn(fn, *args, label=f"{label}/warmup_first", **kwargs)
    for _ in range(warmup_runs - 1):
        time_fn(fn, *args, label="warmup", **kwargs)

    # Benchmark runs
    times: list[float] = []
    for _ in range(bench_runs):
        result = time_fn(fn, *args, label=label, **kwargs)
        times.append(result.elapsed_ms)

    times.sort()
    median_ms = times[len(times) // 2]
    steady = TimingResult(elapsed_ms=median_ms, label=f"{label}/steady_median")
    return warmup_first, steady, times
