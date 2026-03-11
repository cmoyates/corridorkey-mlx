#!/usr/bin/env python3
"""Summarize an experiment result and append to experiment log.

Creates a short markdown summary, appends a JSONL entry to
research/experiments.jsonl, and proposes the next experiment area.

Usage:
    uv run python scripts/summarize_experiment.py --result research/artifacts/latest.json
    uv run python scripts/summarize_experiment.py --result research/artifacts/latest.json --verdict KEEP
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXPERIMENTS_LOG = Path("research/experiments.jsonl")
BEST_RESULT_PATH = Path("research/best_result.json")

# Priority search areas for suggesting next experiment
SEARCH_AREAS = [
    "memory discipline (materialization, gc, cache clearing)",
    "selective precision (refiner/decoder bf16)",
    "tiled inference heuristics (tile size, overlap)",
    "compile-path policy (mx.compile scope)",
    "tensor layout / contiguity",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize experiment")
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--decision", type=Path, default=None, help="Decision JSON (for search_area extraction)")
    parser.add_argument("--verdict", choices=["KEEP", "WEAK_KEEP", "REVERT", "INCONCLUSIVE"], default=None)
    parser.add_argument("--notes", type=str, default="")
    args = parser.parse_args()

    if not args.result.exists():
        print(f"Result not found: {args.result}")
        sys.exit(1)

    result = json.loads(args.result.read_text())
    name = result.get("experiment_name", "unknown")
    timestamp = result.get("timestamp", "")

    # Extract search_area from decision.json if available
    search_area = ""
    if args.decision and args.decision.exists():
        decision = json.loads(args.decision.read_text())
        search_area = decision.get("search_area", "")

    verdict = args.verdict
    if verdict is None:
        verdict = "KEEP" if result.get("fidelity_passed", False) else "REVERT"

    # Build JSONL entry
    entry = {
        "experiment_name": name,
        "timestamp": timestamp,
        "resolution": result.get("resolution"),
        "search_area": search_area,
        "verdict": verdict,
        "fidelity_passed": result.get("fidelity_passed"),
        "median_ms": result.get("benchmark", {}).get("median_ms"),
        "p95_ms": result.get("benchmark", {}).get("p95_ms"),
        "peak_memory_mb": result.get("peak_memory_mb"),
        "notes": args.notes,
        "artifact": str(args.result),
    }

    # Append to log
    EXPERIMENTS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with EXPERIMENTS_LOG.open("a") as f:
        f.write(json.dumps(entry) + "\n")

    # If KEEP and better than current best, update best
    if verdict in ("KEEP", "WEAK_KEEP"):
        if BEST_RESULT_PATH.exists():
            best = json.loads(BEST_RESULT_PATH.read_text())
            best_median = best.get("benchmark", {}).get("median_ms", float("inf"))
            cand_median = result.get("benchmark", {}).get("median_ms", float("inf"))
            if cand_median < best_median:
                BEST_RESULT_PATH.write_text(json.dumps(result, indent=2))
                print(f"New best result: {cand_median}ms (was {best_median}ms)")
            else:
                print(f"Kept but not new best ({cand_median}ms >= {best_median}ms)")
        else:
            BEST_RESULT_PATH.write_text(json.dumps(result, indent=2))
            print(f"First result saved as baseline: {result.get('benchmark', {}).get('median_ms')}ms")

    # Print summary
    print(f"\n--- Experiment Summary ---")
    print(f"Name:     {name}")
    print(f"Verdict:  {verdict}")
    print(f"Fidelity: {'PASS' if result.get('fidelity_passed') else 'FAIL'}")
    print(f"Latency:  {result.get('benchmark', {}).get('median_ms')}ms median")
    print(f"Memory:   {result.get('peak_memory_mb')}MB peak")
    if args.notes:
        print(f"Notes:    {args.notes}")
    print(f"Logged:   {EXPERIMENTS_LOG}")

    # Suggest next area
    log_count = 0
    if EXPERIMENTS_LOG.exists():
        log_count = sum(1 for _ in EXPERIMENTS_LOG.open())

    area_idx = log_count % len(SEARCH_AREAS)
    print(f"\nSuggested next area: {SEARCH_AREAS[area_idx]}")


if __name__ == "__main__":
    main()
