#!/usr/bin/env python3
"""Score an experiment result against the current best.

Reads structured JSON from run_research_experiment.py, enforces fidelity gates,
computes a weighted score, and recommends KEEP / REVERT / INCONCLUSIVE.

Usage:
    uv run python scripts/score_experiment.py --result research/artifacts/latest.json
    uv run python scripts/score_experiment.py --result research/artifacts/latest.json --baseline research/best_result.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BEST_RESULT_PATH = Path("research/best_result.json")

# Scoring weights (must sum to 1.0)
WEIGHT_MEDIAN_LATENCY = 0.6
WEIGHT_P95_LATENCY = 0.1
WEIGHT_PEAK_MEMORY = 0.3

# Noise thresholds for keep/revert
LATENCY_IMPROVEMENT_THRESHOLD = 0.02  # 2%
MEMORY_IMPROVEMENT_THRESHOLD = 0.05  # 5%


def load_json(path: Path) -> dict:
    """Load JSON file."""
    return json.loads(path.read_text())


def compute_score(result: dict, baseline: dict | None) -> float:
    """Compute weighted score. Higher is better. Baseline = 1.0."""
    if baseline is None:
        return 1.0

    base_median = baseline["benchmark"]["median_ms"]
    base_p95 = baseline["benchmark"]["p95_ms"]
    base_mem = baseline["peak_memory_mb"]

    cand_median = result["benchmark"]["median_ms"]
    cand_p95 = result["benchmark"]["p95_ms"]
    cand_mem = result["peak_memory_mb"]

    # Avoid division by zero
    if cand_median <= 0 or cand_p95 <= 0 or cand_mem <= 0:
        return 0.0

    score = (
        WEIGHT_MEDIAN_LATENCY * (base_median / cand_median)
        + WEIGHT_P95_LATENCY * (base_p95 / cand_p95)
        + WEIGHT_PEAK_MEMORY * (base_mem / cand_mem)
    )
    return round(score, 4)


def recommend(result: dict, baseline: dict | None) -> str:
    """Return KEEP, REVERT, or INCONCLUSIVE."""
    # Hard gate: fidelity
    if not result.get("fidelity_passed", False):
        return "REVERT"

    if baseline is None:
        return "KEEP"

    base_median = baseline["benchmark"]["median_ms"]
    base_mem = baseline["peak_memory_mb"]
    cand_median = result["benchmark"]["median_ms"]
    cand_mem = result["peak_memory_mb"]

    latency_ratio = (base_median - cand_median) / base_median if base_median > 0 else 0
    memory_ratio = (base_mem - cand_mem) / base_mem if base_mem > 0 else 0

    latency_improved = latency_ratio >= LATENCY_IMPROVEMENT_THRESHOLD
    memory_improved = memory_ratio >= MEMORY_IMPROVEMENT_THRESHOLD
    latency_regressed = latency_ratio <= -LATENCY_IMPROVEMENT_THRESHOLD
    memory_regressed = memory_ratio <= -MEMORY_IMPROVEMENT_THRESHOLD

    if latency_regressed and memory_regressed:
        return "REVERT"
    if latency_improved or memory_improved:
        return "KEEP"
    return "INCONCLUSIVE"


def main() -> None:
    parser = argparse.ArgumentParser(description="Score experiment result")
    parser.add_argument("--result", type=Path, required=True, help="Experiment result JSON")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=BEST_RESULT_PATH,
        help="Baseline result JSON (default: research/best_result.json)",
    )
    args = parser.parse_args()

    if not args.result.exists():
        print(f"Result not found: {args.result}")
        sys.exit(1)

    result = load_json(args.result)

    # Fidelity gate
    if not result.get("fidelity_passed", False):
        print("VERDICT: REVERT")
        print("Reason: fidelity gate failed")
        failed = [
            k
            for k, v in result.get("parity", {}).items()
            if isinstance(v, dict) and v.get("status") == "FAIL"
        ]
        if failed:
            print(f"Failed tensors: {', '.join(failed)}")
        sys.exit(1)

    # Load baseline
    baseline = None
    if args.baseline.exists():
        baseline = load_json(args.baseline)

    # Score
    score = compute_score(result, baseline)
    verdict = recommend(result, baseline)

    print(f"Experiment: {result.get('experiment_name', 'unknown')}")
    print(f"Resolution: {result.get('resolution')}x{result.get('resolution')}")
    print(f"Fidelity:   PASS")
    print(f"Score:      {score}")

    if baseline:
        base_score = compute_score(baseline, baseline)
        print(f"Baseline:   {base_score}")
        print(
            f"Latency:    {result['benchmark']['median_ms']}ms "
            f"(baseline: {baseline['benchmark']['median_ms']}ms)"
        )
        print(
            f"Memory:     {result['peak_memory_mb']}MB "
            f"(baseline: {baseline['peak_memory_mb']}MB)"
        )
    else:
        print("Baseline:   none (this becomes baseline)")
        print(f"Latency:    {result['benchmark']['median_ms']}ms")
        print(f"Memory:     {result['peak_memory_mb']}MB")

    print(f"\nVERDICT: {verdict}")

    # Output machine-readable summary
    summary = {
        "experiment_name": result.get("experiment_name"),
        "score": score,
        "verdict": verdict,
        "fidelity_passed": True,
        "median_ms": result["benchmark"]["median_ms"],
        "p95_ms": result["benchmark"]["p95_ms"],
        "peak_memory_mb": result["peak_memory_mb"],
    }
    print(f"\n{json.dumps(summary)}")


if __name__ == "__main__":
    main()
