#!/usr/bin/env python3
"""Generate a compound learning note from experiment result + decision.

Writes a markdown file to research/compound/ capturing what was tried,
what happened, and why — so the proposer can learn from it.

Usage:
    uv run python scripts/compound_note.py \
        --result artifacts/runs/result-3.json \
        --decision artifacts/runs/decision-3.json \
        --verdict REVERT
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

COMPOUND_DIR = Path("research/compound")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate compound note")
    parser.add_argument("--result", type=Path, required=False, default=None)
    parser.add_argument("--decision", type=Path, required=False, default=None)
    parser.add_argument("--verdict", required=True, choices=["KEEP", "WEAK_KEEP", "REVERT", "INCONCLUSIVE", "ERROR"])
    parser.add_argument("--score", type=float, default=None)
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--error-log", type=Path, default=None, help="Path to error log for runtime failures")
    parser.add_argument("--experiment-name", type=str, default=None, help="Experiment name (when decision.json unavailable)")
    parser.add_argument("--search-area", type=str, default=None, help="Search area (when decision.json unavailable)")
    args = parser.parse_args()

    # Load result and decision if available
    result = {}
    decision = {}
    error_text = ""

    if args.result and args.result.exists():
        result = json.loads(args.result.read_text())
    if args.decision and args.decision.exists():
        decision = json.loads(args.decision.read_text())
    if args.error_log and args.error_log.exists():
        full_log = args.error_log.read_text()
        # Keep last 40 lines to stay concise
        error_text = "\n".join(full_log.strip().splitlines()[-40:])

    name = args.experiment_name or decision.get("experiment_name", "unknown")
    hypothesis = decision.get("hypothesis", "(no hypothesis)")
    search_area = args.search_area or decision.get("search_area", "unknown")
    files_changed = decision.get("files_changed", [])

    fidelity_passed = result.get("fidelity_passed", False)
    benchmark = result.get("benchmark", {})
    median_ms = benchmark.get("median_ms")
    p95_ms = benchmark.get("p95_ms")
    peak_mb = result.get("peak_memory_mb")
    parity = result.get("parity", {})
    max_abs_errors = parity.get("max_abs_errors", {})

    # Determine failure reason
    if args.verdict == "ERROR":
        failure_reason = "Runtime error — experiment crashed before producing results"
    elif not fidelity_passed and result:
        failure_reason = "Fidelity gate failed"
        if max_abs_errors:
            worst_key = max(max_abs_errors, key=max_abs_errors.get)
            failure_reason += f" — worst: {worst_key}={max_abs_errors[worst_key]:.2e}"
    elif args.verdict == "REVERT":
        failure_reason = "Performance regression"
    elif args.verdict == "INCONCLUSIVE":
        failure_reason = "Within measurement noise"
    elif args.verdict in ("KEEP", "WEAK_KEEP"):
        failure_reason = None

    # Count existing notes for numbering
    COMPOUND_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(COMPOUND_DIR.glob("exp*.md"))
    next_num = len(existing) + 1

    # Build note
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    slug = name.replace(" ", "-").lower()
    filename = f"exp{next_num:03d}_{slug}.md"

    lines = [
        f"# Compound: {name}",
        "",
        f"**Date:** {date}",
        f"**Search area:** {search_area}",
        f"**Verdict:** {args.verdict}" + (f" (score {args.score})" if args.score else ""),
        "",
        "## Hypothesis",
        "",
        hypothesis,
        "",
        "## Result",
        "",
    ]

    if not fidelity_passed:
        lines.append(f"**FIDELITY FAILURE** — changes break numerical parity with golden reference.")
        if max_abs_errors:
            lines.append("")
            lines.append("Worst errors per tensor:")
            for key, val in sorted(max_abs_errors.items(), key=lambda x: -x[1])[:5]:
                lines.append(f"- `{key}`: {val:.2e} (threshold: 1e-3)")
        lines.append("")
        lines.append("### Takeaway")
        lines.append("")
        lines.append(f"This approach ({search_area}) is NOT safe for the modified files.")
        lines.append("Do NOT retry the same change without a different mitigation strategy.")
    else:
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        if median_ms is not None:
            lines.append(f"| Median latency | {median_ms:.2f}ms |")
        if p95_ms is not None:
            lines.append(f"| P95 latency | {p95_ms:.2f}ms |")
        if peak_mb is not None:
            lines.append(f"| Peak memory | {peak_mb:.1f}MB |")
        if args.score is not None:
            lines.append(f"| Score | {args.score} |")

    lines.extend([
        "",
        "## Files changed",
        "",
    ])
    for f in files_changed:
        lines.append(f"- `{f}`")

    if failure_reason:
        lines.extend([
            "",
            "## Why it failed",
            "",
            failure_reason,
        ])

    if error_text:
        lines.extend([
            "",
            "## Error log",
            "",
            "```",
            error_text,
            "```",
            "",
            "### Takeaway",
            "",
            "This runtime error must be addressed before retrying this approach.",
            "The proposer should read this traceback and fix the root cause.",
        ])

    if args.notes:
        lines.extend([
            "",
            "## Notes",
            "",
            args.notes,
        ])

    note_path = COMPOUND_DIR / filename
    note_path.write_text("\n".join(lines) + "\n")
    print(f"Compound note: {note_path}")


if __name__ == "__main__":
    main()
