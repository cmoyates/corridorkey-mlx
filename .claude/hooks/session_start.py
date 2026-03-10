#!/usr/bin/env python3
"""SessionStart hook: inject current lab context."""

import json
import sys
from pathlib import Path


def main() -> None:
    lines = ["=== corridorkey-mlx autoresearch lab ==="]

    # Current best
    best_path = Path("research/best_result.json")
    if best_path.exists():
        try:
            best = json.loads(best_path.read_text())
            lines.append(
                f"Best: {best.get('experiment_name', '?')} — "
                f"{best.get('benchmark', {}).get('median_ms', '?')}ms, "
                f"{best.get('peak_memory_mb', '?')}MB"
            )
        except Exception:
            lines.append("Best: (could not read)")
    else:
        lines.append("Best: no baseline yet — run first experiment")

    # Experiment count
    log_path = Path("research/experiments.jsonl")
    if log_path.exists():
        count = sum(1 for _ in log_path.open())
        lines.append(f"Experiments: {count} logged")
    else:
        lines.append("Experiments: none yet")

    lines.append("See: research/CLAUDE_AUTORESEARCH.md")

    print("\n".join(lines))
    sys.exit(0)


if __name__ == "__main__":
    main()
