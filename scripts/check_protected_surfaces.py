#!/usr/bin/env python3
"""Check that protected surfaces have not been modified vs HEAD.

Exits 0 if no protected files were touched, 1 if any were modified.
Modified file list printed to stderr on failure.

Usage:
    uv run python scripts/check_protected_surfaces.py
"""

from __future__ import annotations

import subprocess
import sys

# Protected files — must not be modified by Claude during the loop
PROTECTED_FILES = [
    "scripts/bench_mlx.py",
    "scripts/compare_reference.py",
    "scripts/smoke_2048.py",
    "scripts/bench_optimizations.py",
    "scripts/score_experiment.py",
    "scripts/run_research_experiment.py",
    "scripts/validate_decision.py",
    "scripts/check_protected_surfaces.py",
    "research/benchmark_spec.md",
    "loop.sh",
    "research/decision.schema.json",
]

# Protected directories — any file under these is protected
PROTECTED_DIRS = [
    "reference/fixtures/",
    "tests/",
]


def get_changed_files() -> set[str]:
    """Get files changed vs HEAD (staged + unstaged)."""
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD"],
        capture_output=True,
        text=True,
    )
    staged = set(result.stdout.strip().splitlines()) if result.stdout.strip() else set()

    # Also check untracked new files
    result = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        capture_output=True,
        text=True,
    )
    untracked = set(result.stdout.strip().splitlines()) if result.stdout.strip() else set()

    return staged | untracked


def check_protected(changed: set[str]) -> list[str]:
    """Return list of protected files that were modified."""
    violations: list[str] = []

    for filepath in changed:
        # Check exact file matches
        if filepath in PROTECTED_FILES:
            violations.append(filepath)
            continue

        # Check directory matches
        for protected_dir in PROTECTED_DIRS:
            if filepath.startswith(protected_dir):
                violations.append(filepath)
                break

    return sorted(violations)


def main() -> None:
    changed = get_changed_files()
    if not changed:
        print("No files changed — protected surfaces intact.")
        sys.exit(0)

    violations = check_protected(changed)
    if violations:
        print("Protected surface violations detected:", file=sys.stderr)
        for filepath in violations:
            print(f"  - {filepath}", file=sys.stderr)
        sys.exit(1)

    print(f"OK: {len(changed)} files changed, none protected.")


if __name__ == "__main__":
    main()
