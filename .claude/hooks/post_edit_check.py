#!/usr/bin/env python3
"""PostToolUse hook (async): run cheap syntax check on modified Python files."""

import json
import os
import subprocess
import sys


def main() -> None:
    tool_input = os.environ.get("CLAUDE_TOOL_USE_INPUT", "{}")
    try:
        params = json.loads(tool_input)
    except json.JSONDecodeError:
        sys.exit(0)

    file_path = params.get("file_path", "") or params.get("path", "")
    if not file_path or not file_path.endswith(".py"):
        sys.exit(0)

    if not os.path.exists(file_path):
        sys.exit(0)

    # Quick syntax check via py_compile
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", file_path],
        capture_output=True,
        text=True,
        timeout=10,
    )

    if result.returncode != 0:
        print(f"SYNTAX ERROR in {file_path}:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)

    # Quick JSON schema check for research artifacts
    if file_path.endswith(".json") and "research/" in file_path:
        try:
            with open(file_path) as f:
                json.load(f)
        except json.JSONDecodeError as e:
            print(f"INVALID JSON in {file_path}: {e}", file=sys.stderr)
            sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
