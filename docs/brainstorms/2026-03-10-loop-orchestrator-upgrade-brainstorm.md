# Brainstorm: loop.sh Orchestrator Upgrade

**Date:** 2026-03-10
**Status:** Ready for planning

## What We're Building

Upgrade `loop.sh` from a thin wrapper around Claude into a proper ephemeral shell-driven orchestrator, implementing the architecture from `docs/Claude Code Autonomous Optimization Loop.md`.

Core shift: Claude becomes a **stateless mutation proposer**. The orchestrator owns benchmarking, fidelity gating, and keep/revert decisions.

## Why This Approach

The current loop.sh delegates everything to Claude — it runs benchmarks, interprets results, and writes decisions. This creates trust issues (Claude could hallucinate results) and context bloat (benchmark output fills the context window). By moving benchmarks to the orchestrator:

- **Determinism**: Benchmark results are never filtered through LLM interpretation
- **Context efficiency**: Claude's prompt stays small — just codebase + recent history
- **Debuggability**: Every iteration's raw benchmark output is captured on disk
- **Safety**: Fidelity gate can't be bypassed by prompt engineering

## Key Decisions

1. **Orchestrator runs benchmarks** — Claude proposes code changes only. Shell runs `compare_reference.py` (fidelity gate) and `bench_mlx.py` (performance gate) after Claude exits.

2. **Fully dynamic prompt** — Shell constructs prompt each iteration from: CLAUDE.md, program.md, best_result.json, last 5 experiments from experiments.jsonl, active_hypothesis.md, compound notes index.

3. **Baseline capture pre-loop** — Run bench_mlx.py + compare_reference.py once before loop starts, write `artifacts/benchmark_baseline.json`. Loop compares against this.

4. **Structured JSON output** — Use `claude -p --output-format json --no-session-persistence`. Claude writes decision.json to disk. Shell parses stdout JSON as well.

5. **JSON schema validation** — Create `artifacts/decision.schema.json`. Validate Claude's output with python jsonschema before acting on it.

6. **Full cleanup between iterations** — `git clean -fd`, purge `__pycache__`, clear MLX caches. Ensures sterile benchmark environment.

7. **Keep bash** — Bash orchestrator with small Python helpers for JSON parsing/validation. Transparent and auditable.

## Scope of Changes

### New files
- `artifacts/decision.schema.json` — output contract for Claude
- Python helper for schema validation (or inline in loop.sh)

### Modified files
- `loop.sh` — major rewrite:
  - Baseline capture phase
  - Dynamic prompt construction
  - `--no-session-persistence --output-format json` flags
  - Post-Claude benchmark execution
  - Schema validation
  - Full cleanup between iterations
  - Richer stall tracking (fidelity vs performance vs noise)
  - Descriptive commit messages (pull from decision.json)

### Claude's new contract
Claude's job per iteration:
1. Read codebase state + injected context
2. Choose one optimization from allowed search areas
3. Implement the code change
4. Write `artifacts/latest_decision.json` with: hypothesis, files_modified, expected_impact, status=experiment_proposed
5. Exit

Claude does NOT:
- Run benchmarks
- Evaluate fidelity
- Decide keep/revert (orchestrator does this)

## Open Questions

- Scoring thresholds for keep/revert — reuse existing `score_experiment.py` or embed simplified logic in bash?
- Should prompt include full file contents of target source files, or just let Claude read them via tools?
- `--output-format json` wraps Claude's entire response — need to test if decision.json file writing still works alongside it
- Max token budget per iteration via `--max-tokens`?
