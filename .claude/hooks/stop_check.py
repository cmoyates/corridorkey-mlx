#!/usr/bin/env python3
"""Stop hook: warn if stopping without experiment artifacts or documentation.

Checks for recent artifacts in research/artifacts/ or a documented reason.
Non-blocking (exit 0) but prints a reminder.
"""

import os
import sys
import time
from pathlib import Path


def main() -> None:
    artifacts_dir = Path("research/artifacts")
    if not artifacts_dir.exists():
        print(
            "REMINDER: No research/artifacts/ dir found. "
            "If you made code changes, run the experiment loop before stopping.",
            file=sys.stderr,
        )
        sys.exit(0)

    # Check for artifacts modified in the last 30 minutes
    recent_threshold = time.time() - 1800
    recent = [
        f
        for f in artifacts_dir.glob("*.json")
        if f.stat().st_mtime > recent_threshold
    ]

    if not recent:
        print(
            "REMINDER: No recent experiment artifacts found. "
            "If you made code changes, consider running:\n"
            "  uv run python scripts/run_research_experiment.py\n"
            "  uv run python scripts/score_experiment.py --result <path>",
            file=sys.stderr,
        )

    sys.exit(0)


if __name__ == "__main__":
    main()
