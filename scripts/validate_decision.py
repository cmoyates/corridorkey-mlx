#!/usr/bin/env python3
"""Validate artifacts/latest_decision.json against the decision schema.

Exits 0 if valid, 1 if invalid or missing. Diagnostics to stderr.

Usage:
    uv run python scripts/validate_decision.py
    uv run python scripts/validate_decision.py --decision path/to/decision.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCHEMA_PATH = Path("research/decision.schema.json")
DEFAULT_DECISION_PATH = Path("artifacts/latest_decision.json")

VALID_SEARCH_AREAS = {
    "tile-lifecycle-memory",
    "selective-precision",
    "tiled-inference-heuristics",
    "compile-path-policy",
    "tensor-layout-staging",
    "backbone-quantization",
    "mlx-memory-tuning",
    "layernorm-fusion",
    "token-routing",
    "refiner-only-tiling",
    "fused-metal-kernels",
}

REQUIRED_FIELDS = {"experiment_name", "hypothesis", "files_changed", "search_area"}
OPTIONAL_FIELDS = {"next_hypothesis", "notes"}
ALL_FIELDS = REQUIRED_FIELDS | OPTIONAL_FIELDS


def validate(decision: dict) -> list[str]:
    """Return list of validation errors (empty = valid)."""
    errors: list[str] = []

    # Check required fields
    for field in REQUIRED_FIELDS:
        if field not in decision:
            errors.append(f"missing required field: {field}")

    if errors:
        return errors

    # Check no extra fields
    extra = set(decision.keys()) - ALL_FIELDS
    if extra:
        errors.append(f"unexpected fields: {', '.join(sorted(extra))}")

    # Type checks
    if not isinstance(decision["experiment_name"], str) or len(decision["experiment_name"]) < 1:
        errors.append("experiment_name must be a non-empty string")

    if not isinstance(decision["hypothesis"], str) or len(decision["hypothesis"]) < 10:
        errors.append("hypothesis must be a string with at least 10 characters")

    if not isinstance(decision["files_changed"], list) or len(decision["files_changed"]) < 1:
        errors.append("files_changed must be a non-empty list")
    elif not all(isinstance(f, str) for f in decision["files_changed"]):
        errors.append("files_changed items must be strings")

    if decision["search_area"] not in VALID_SEARCH_AREAS:
        errors.append(
            f"search_area must be one of: {', '.join(sorted(VALID_SEARCH_AREAS))}; "
            f"got: {decision['search_area']}"
        )

    # Optional field type checks
    for field in OPTIONAL_FIELDS:
        if field in decision and not isinstance(decision[field], str):
            errors.append(f"{field} must be a string")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate decision JSON")
    parser.add_argument(
        "--decision",
        type=Path,
        default=DEFAULT_DECISION_PATH,
        help=f"Path to decision JSON (default: {DEFAULT_DECISION_PATH})",
    )
    args = parser.parse_args()

    if not args.decision.exists():
        print(f"Decision file not found: {args.decision}", file=sys.stderr)
        sys.exit(1)

    try:
        decision = json.loads(args.decision.read_text())
    except json.JSONDecodeError as exc:
        print(f"Invalid JSON in {args.decision}: {exc}", file=sys.stderr)
        sys.exit(1)

    errors = validate(decision)
    if errors:
        print(f"Validation failed for {args.decision}:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        sys.exit(1)

    print(f"Valid: {decision['experiment_name']} [{decision['search_area']}]")


if __name__ == "__main__":
    main()
