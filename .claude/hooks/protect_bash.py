#!/usr/bin/env python3
"""PreToolUse hook for Bash: block destructive commands and tampering with protected files."""

import json
import os
import re
import sys

PROTECTED_PATTERNS = [
    r"scripts/bench_mlx\.py",
    r"scripts/compare_reference\.py",
    r"scripts/smoke_2048\.py",
    r"scripts/bench_optimizations\.py",
    r"scripts/score_experiment\.py",
    r"scripts/run_research_experiment\.py",
    r"research/benchmark_spec\.md",
    r"reference/fixtures/",
]

DESTRUCTIVE_PATTERNS = [
    r"\brm\s+-rf\s+[/.]",
    r"\bgit\s+reset\s+--hard\b",
    r"\bgit\s+clean\s+-fd\b",
    r"\bgit\s+checkout\s+--\s+\.",
]


def main() -> None:
    tool_input = os.environ.get("CLAUDE_TOOL_USE_INPUT", "{}")
    try:
        params = json.loads(tool_input)
    except json.JSONDecodeError:
        sys.exit(0)

    command = params.get("command", "")
    if not command:
        sys.exit(0)

    # Check destructive commands
    for pattern in DESTRUCTIVE_PATTERNS:
        if re.search(pattern, command):
            print(f"BLOCKED: destructive command detected: {command[:80]}", file=sys.stderr)
            sys.exit(2)

    # Check for tampering with protected files via shell
    for pattern in PROTECTED_PATTERNS:
        # Look for sed/awk/tee/cat>/ targeting protected files
        if re.search(pattern, command) and re.search(
            r"\b(sed|awk|tee|truncate|>|>>|mv|cp\s.*\s)\b", command
        ):
            print(
                f"BLOCKED: shell write to protected file detected: {command[:80]}",
                file=sys.stderr,
            )
            sys.exit(2)

    sys.exit(0)


if __name__ == "__main__":
    main()
