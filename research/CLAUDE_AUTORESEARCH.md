# Claude Autoresearch Lab — corridorkey-mlx

## Purpose

Structured optimization research loop for MLX inference performance on Apple Silicon. Optimizes latency + peak memory while gating on fidelity.

## Experiment loop

```
plan → implement → benchmark → score → keep/revert → record
```

1. **Plan**: Write hypothesis, target files, expected impact, rollback criteria
2. **Implement**: Minimal change, one variable at a time
3. **Benchmark**: `uv run python scripts/run_research_experiment.py [--experiment-name NAME]`
4. **Score**: `uv run python scripts/score_experiment.py [--result PATH]`
5. **Keep/revert**: Score output says KEEP, REVERT, or INCONCLUSIVE
6. **Record**: `uv run python scripts/summarize_experiment.py [--result PATH]`

## Benchmark truth sources

| Surface | Command | Measures |
|---------|---------|----------|
| Latency (eager+compiled) | `uv run python scripts/bench_mlx.py` | warmup, steady-state, min, speedup |
| PyTorch parity | `uv run python scripts/compare_reference.py` | max/mean abs error per tensor |
| 2048 smoke | `uv run python scripts/smoke_2048.py` | timing, peak memory, output health |
| Opt matrix | `uv run python scripts/bench_optimizations.py` | toggle ablation across opt flags |

These scripts are **protected**. Do not modify them.

## Keep/revert behavior

A candidate is KEPT when:
- All fidelity gates pass (max abs error < 5e-3 per tensor vs golden reference)
- Latency improved OR memory improved (with no regression on the other metric beyond noise)
- Score > current best score

A candidate is REVERTED when:
- Any fidelity gate fails (hard failure, non-negotiable)
- Both latency and memory regressed
- Score did not improve

INCONCLUSIVE when within noise margins on both metrics.

## Hook behavior

- **PreToolUse (Edit|Write)**: Blocks edits to protected files
- **PreToolUse (Bash)**: Blocks destructive commands (`rm -rf`, `git reset --hard`, etc.) and shell tampering with protected files
- **PostToolUse (Edit|Write)**: Runs syntax check on modified Python files (async, cheap)
- **Stop**: Warns if stopping without producing experiment artifacts or documenting why
- **SessionStart**: Prints current lab status

## Skill behavior

- `/upstream-research` — Read-only codebase exploration (forked context)
- `/plan-experiment` — Manual trigger to plan next experiment
- `/benchmark-review` — Read latest result, recommend keep/revert
- `/compound-finding` — Record a learning note

## How to continue later

1. Claude loads `CLAUDE.md` automatically, which references this file
2. Check `research/best_result.json` for current best
3. Check `research/experiments.jsonl` for history
4. Check `research/program.md` for the research program
5. Run `/plan-experiment` to pick the next experiment
6. The experiment loop is self-contained — follow it

## File layout

```
research/
  CLAUDE_AUTORESEARCH.md   — this file
  program.md               — research program (priorities, search areas)
  benchmark_spec.md        — benchmark spec (metrics, thresholds, rules)
  experiments.jsonl         — append-only experiment log
  best_result.json          — current best result
  compound/                 — learning notes
    upstream_pr_mining.md   — upstream CorridorKey PR analysis
```
