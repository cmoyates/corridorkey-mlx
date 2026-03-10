#!/usr/bin/env python3
"""PreToolUse hook: block edits to protected benchmark/scoring/baseline files.

Reads tool_name and file_path from CLAUDE_TOOL_USE_* env vars.
Exits 2 (block) if the target is a protected surface, 0 otherwise.
"""

import json
import os
import sys
from pathlib import Path

PROTECTED_FILES = {
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
}

PROTECTED_DIRS = {
    "reference/fixtures/",
}


def main() -> None:
    tool_input = os.environ.get("CLAUDE_TOOL_USE_INPUT", "{}")
    try:
        params = json.loads(tool_input)
    except json.JSONDecodeError:
        sys.exit(0)

    file_path = params.get("file_path", "") or params.get("path", "")
    if not file_path:
        sys.exit(0)

    # Normalize to repo-relative path — try CLAUDE_PROJECT_DIR, then script location
    repo_root = os.environ.get("CLAUDE_PROJECT_DIR", "")
    if not repo_root:
        # Derive from hook location: .claude/hooks/ -> repo root
        repo_root = str(Path(__file__).resolve().parent.parent.parent)
    if file_path.startswith(repo_root):
        rel_path = file_path[len(repo_root) :].lstrip("/")
    else:
        rel_path = file_path

    # Check protected files
    if rel_path in PROTECTED_FILES:
        print(f"BLOCKED: {rel_path} is a protected research surface. Do not modify.", file=sys.stderr)
        sys.exit(2)

    # Check protected dirs
    for pdir in PROTECTED_DIRS:
        if rel_path.startswith(pdir):
            print(f"BLOCKED: {rel_path} is in protected dir {pdir}. Do not modify.", file=sys.stderr)
            sys.exit(2)

    sys.exit(0)


if __name__ == "__main__":
    main()
