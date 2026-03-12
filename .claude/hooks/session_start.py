#!/usr/bin/env python3
"""SessionStart hook: inject current lab context."""

import json
import sys
from pathlib import Path


def main() -> None:
    lines = ["=== corridorkey-mlx autoresearch lab ==="]

    # Current best (per-resolution)
    best_files = sorted(Path("research").glob("best_result_*.json"))
    if best_files:
        for bp in best_files:
            try:
                best = json.loads(bp.read_text())
                res = best.get("resolution", "?")
                lines.append(
                    f"Best @{res}: {best.get('experiment_name', '?')} — "
                    f"{best.get('benchmark', {}).get('median_ms', '?')}ms, "
                    f"{best.get('peak_memory_mb', '?')}MB"
                )
            except Exception:
                lines.append(f"Best: (could not read {bp.name})")
    else:
        # Fallback to legacy best_result.json
        legacy = Path("research/best_result.json")
        if legacy.exists():
            try:
                best = json.loads(legacy.read_text())
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
