---
title: "feat: Upgrade loop.sh to Ephemeral Shell-Driven Orchestrator"
type: feat
date: 2026-03-10
---

# Upgrade loop.sh to Ephemeral Shell-Driven Orchestrator

## Overview

Rewrite `loop.sh` so the orchestrator owns benchmarking, fidelity gating, and keep/revert decisions. Claude becomes a stateless mutation proposer — it reads context, modifies code, writes a structured decision file, and exits. The shell handles everything else.

## Problem Statement

Current loop.sh delegates everything to Claude: running benchmarks, interpreting results, deciding keep/revert. This creates:

- **Trust gap**: Claude could hallucinate benchmark results or misinterpret them
- **Context bloat**: Benchmark output fills Claude's context window
- **No independent verification**: The loop trusts Claude's self-reported decision
- **Missing flags**: No `--no-session-persistence` or `--output-format json`
- **No baseline capture**: No pre-loop measurement to compare against
- **ROOT path bug**: Line 4 resolves to parent of repo, not repo root
- **Artifact path mismatch**: loop.sh expects `artifacts/` but scripts write to `research/artifacts/`

## Proposed Solution

Ephemeral shell-driven pipeline per the design doc. Each iteration:

1. Shell assembles dynamic prompt from filesystem state
2. `claude -p --no-session-persistence --output-format json` proposes one code mutation
3. Shell validates decision.json against schema
4. Shell checks no protected surfaces were modified
5. Shell runs `run_research_experiment.py` (fidelity + benchmark + peak memory)
6. Shell runs `score_experiment.py` for verdict
7. KEEP → commit + update best_result.json | REVERT → restore + log
8. Full cleanup between iterations

**Key insight from spec-flow**: Use `run_research_experiment.py` as the single gate instead of separate `compare_reference.py` + `bench_mlx.py`. It already combines parity + benchmark + peak memory into structured JSON and exits nonzero on fidelity failure. This eliminates 4 gaps (no JSON output from protected scripts, no peak memory from bench_mlx, resolution ambiguity).

## Technical Approach

### Phase 1: Foundation (decision schema + helpers)

#### 1a. Create `artifacts/decision.schema.json`

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["experiment_name", "hypothesis", "files_changed", "search_area"],
  "properties": {
    "experiment_name": { "type": "string", "minLength": 1 },
    "hypothesis": { "type": "string", "minLength": 10 },
    "files_changed": {
      "type": "array",
      "items": { "type": "string" },
      "minItems": 1
    },
    "search_area": {
      "type": "string",
      "enum": [
        "tile-lifecycle-memory",
        "selective-precision",
        "tiled-inference-heuristics",
        "compile-path-policy",
        "tensor-layout-staging"
      ]
    },
    "next_hypothesis": { "type": "string" },
    "notes": { "type": "string" }
  },
  "additionalProperties": false
}
```

Claude writes this to `artifacts/latest_decision.json`. No status/verdict — the orchestrator determines that independently.

#### 1b. Create `scripts/validate_decision.py`

Small Python helper:
- Reads `artifacts/latest_decision.json`
- Validates against `artifacts/decision.schema.json` using `json` + manual checks (no jsonschema dep)
- Exits 0 on valid, 1 on invalid with diagnostic to stderr

#### 1c. Create `scripts/check_protected_surfaces.py`

Diffs protected files against HEAD:
- `bench_mlx.py`, `compare_reference.py`, `smoke_2048.py`, `bench_optimizations.py`, `score_experiment.py`, `run_research_experiment.py`, `benchmark_spec.md`, `reference/fixtures/*`
- Exits 0 if none modified, 1 if any touched (with file list to stderr)
- This is the belt to the hooks' suspenders — ensures integrity even if hooks don't load with `--no-session-persistence`

### Phase 2: Core orchestrator rewrite (`loop.sh`)

#### 2a. Fix ROOT path + remove cd

Replace broken `cd` logic with absolute path resolution:
```bash
ROOT="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
```
All subsequent commands use `$ROOT/` prefix instead of relying on cwd.

#### 2b. Add flock guard

```bash
LOCKFILE="$ROOT/.loop.lock"
exec 200>"$LOCKFILE"
flock -n 200 || { echo "Another loop instance running."; exit 1; }
```

#### 2c. Add trap handler

```bash
cleanup_on_exit() {
  if ! git -C "$ROOT" diff --quiet; then
    echo "Interrupted — reverting uncommitted changes."
    git -C "$ROOT" checkout -- .
    git -C "$ROOT" clean -fd
  fi
}
trap cleanup_on_exit EXIT INT TERM
```

#### 2d. Baseline capture phase

Before loop starts:
```bash
if [[ ! -f "$ROOT/artifacts/benchmark_baseline.json" ]]; then
  echo "Capturing baseline..."
  uv run python "$ROOT/scripts/run_research_experiment.py" \
    --experiment-name baseline \
    --output "$ROOT/artifacts/benchmark_baseline.json"
  # Abort if fidelity fails on baseline
fi
```

#### 2e. Dynamic prompt construction

Build prompt each iteration from filesystem state:

```bash
build_prompt() {
  local last_experiments
  last_experiments=$(tail -5 "$ROOT/research/experiments.jsonl" 2>/dev/null || echo "[]")

  local best_result
  best_result=$(cat "$ROOT/research/best_result.json" 2>/dev/null || echo "{}")

  local compound_index
  compound_index=$(ls "$ROOT/research/compound/"*.md 2>/dev/null | head -10 | xargs -I{} basename {} || echo "none")

  cat <<PROMPT
You are a stateless MLX optimization engineer. You will propose exactly one
bounded code mutation for corridorkey-mlx. You do NOT run benchmarks — the
orchestrator does that after you exit.

## Your contract
1. Read the codebase context below and the source files you need.
2. Choose one experiment from the allowed search areas in program.md.
3. Implement the minimal code change (one variable at a time).
4. Write artifacts/latest_decision.json conforming to the schema.
5. Exit. Do NOT run bench_mlx.py, compare_reference.py, or any benchmark scripts.

## Decision schema (artifacts/latest_decision.json)
Required fields: experiment_name, hypothesis, files_changed, search_area
Optional fields: next_hypothesis, notes
Do NOT include status/verdict — the orchestrator determines that.

## Current best result
$best_result

## Last 5 experiments
$last_experiments

## Compound notes index
$compound_index

## Allowed search areas (from program.md)
1. tile-lifecycle-memory — del refs, gc timing, avoid redundant allocs
2. selective-precision — refiner fp16, backbone fp32, decoder bf16
3. tiled-inference-heuristics — tile size/overlap sweeps, blending
4. compile-path-policy — mx.compile for fixed shapes
5. tensor-layout-staging — contiguity, NCHW-NHWC minimization

## Rules
- Modify only files in src/corridorkey_mlx/, scripts/infer.py, scripts/smoke_engine.py
- Do NOT modify any file in scripts/bench_*, scripts/compare_*, scripts/score_*, scripts/run_research_*, reference/fixtures/, tests/
- One optimization variable per experiment
- If you cannot find a viable experiment, write decision.json with notes explaining why
PROMPT
}
```

#### 2f. Main loop

```bash
for ((i=1; i<=ITERATIONS; i++)); do
  echo "=== Iteration $i ==="

  # Checkpoint
  git -C "$ROOT" add -A
  git -C "$ROOT" commit -m "checkpoint: before iteration $i" --allow-empty -q

  # Build + invoke Claude
  PROMPT="$(build_prompt)"
  timeout "${CLAUDE_TIMEOUT:-600}" claude -p "$PROMPT" \
    --no-session-persistence \
    --output-format json \
    --allowedTools "Read,Edit,Write,Bash,Grep,Glob" \
    > "$ROOT/artifacts/runs/claude-output-$i.json" 2>&1 || true

  # Validate decision
  if ! uv run python "$ROOT/scripts/validate_decision.py"; then
    echo "Invalid or missing decision.json. Reverting."
    git -C "$ROOT" checkout -- . && git -C "$ROOT" clean -fd
    STALLS=$((STALLS + 1)); STALL_TYPE="invalid_output"
    log_iteration "$i" "REVERT" "invalid_output" ""
    continue_or_break; continue
  fi

  # Protected surface check
  if ! uv run python "$ROOT/scripts/check_protected_surfaces.py"; then
    echo "Protected surface modified! Reverting."
    git -C "$ROOT" checkout -- . && git -C "$ROOT" clean -fd
    STALLS=$((STALLS + 1)); STALL_TYPE="protected_surface"
    log_iteration "$i" "REVERT" "protected_surface_violation" ""
    continue_or_break; continue
  fi

  # Read experiment name for logging
  EXP_NAME="$(read_json_field artifacts/latest_decision.json experiment_name)"

  # Fidelity + benchmark gate (single script)
  RESULT_FILE="$ROOT/artifacts/runs/result-$i.json"
  if ! uv run python "$ROOT/scripts/run_research_experiment.py" \
    --experiment-name "$EXP_NAME" \
    --output "$RESULT_FILE"; then
    echo "Fidelity gate FAILED. Reverting."
    git -C "$ROOT" checkout -- . && git -C "$ROOT" clean -fd
    STALLS=$((STALLS + 1)); STALL_TYPE="fidelity"
    log_iteration "$i" "REVERT" "fidelity_failure" "$EXP_NAME"
    continue_or_break; continue
  fi

  # Score
  SCORE_OUTPUT="$(uv run python "$ROOT/scripts/score_experiment.py" \
    --result "$RESULT_FILE")"
  VERDICT="$(echo "$SCORE_OUTPUT" | tail -1 | python3 -c 'import json,sys; print(json.load(sys.stdin)["verdict"])')"
  SCORE="$(echo "$SCORE_OUTPUT" | tail -1 | python3 -c 'import json,sys; print(json.load(sys.stdin)["score"])')"

  case "$VERDICT" in
    KEEP)
      echo "KEEP — score=$SCORE"
      git -C "$ROOT" add -A
      git -C "$ROOT" commit -m "exp: $EXP_NAME [score=$SCORE, verdict=KEEP]" -q
      uv run python "$ROOT/scripts/summarize_experiment.py" \
        --result "$RESULT_FILE" --verdict KEEP
      STALLS=0
      ;;
    REVERT)
      echo "REVERT — score=$SCORE"
      git -C "$ROOT" checkout -- . && git -C "$ROOT" clean -fd
      uv run python "$ROOT/scripts/summarize_experiment.py" \
        --result "$RESULT_FILE" --verdict REVERT
      STALLS=$((STALLS + 1)); STALL_TYPE="performance"
      ;;
    *)
      echo "INCONCLUSIVE — score=$SCORE"
      git -C "$ROOT" checkout -- . && git -C "$ROOT" clean -fd
      uv run python "$ROOT/scripts/summarize_experiment.py" \
        --result "$RESULT_FILE" --verdict INCONCLUSIVE
      STALLS=$((STALLS + 1)); STALL_TYPE="inconclusive"
      ;;
  esac

  # Archive decision + Claude output
  cp "$ROOT/artifacts/latest_decision.json" "$ROOT/artifacts/runs/decision-$i.json" 2>/dev/null || true

  # Cleanup
  find "$ROOT" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
  git -C "$ROOT" clean -fd -q 2>/dev/null || true

  continue_or_break
done
```

#### 2g. Stall tracking helpers

```bash
MAX_STALLS="${MAX_STALLS:-3}"
STALLS=0
STALL_TYPE=""

continue_or_break() {
  if (( STALLS >= MAX_STALLS )); then
    echo "Stopping: $STALLS consecutive failures (type: $STALL_TYPE)"
    break
  fi
}

log_iteration() {
  local iter="$1" verdict="$2" reason="$3" name="$4"
  echo "{\"iteration\":$iter,\"verdict\":\"$verdict\",\"reason\":\"$reason\",\"experiment\":\"$name\",\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" \
    >> "$ROOT/artifacts/runs/loop-log.jsonl"
}
```

### Phase 3: Polish

#### 3a. Unify artifact paths

- All loop artifacts → `artifacts/` (root level, gitignored)
- Research artifacts (experiment JSONs) → `research/artifacts/` (committed)
- Add `artifacts/` to `.gitignore` if not already there

#### 3b. Add `artifacts/` to .gitignore

Loop artifacts are ephemeral — only `research/` artifacts are committed.

#### 3c. Update CLAUDE.md

Add `loop.sh` and its helpers to the appropriate surface classification:
- `loop.sh` → protected (orchestrator)
- `scripts/validate_decision.py` → protected
- `scripts/check_protected_surfaces.py` → protected
- `artifacts/decision.schema.json` → read-only

## Files Changed

| File | Action | Classification |
|------|--------|----------------|
| `loop.sh` | Rewrite | Protected (orchestrator) |
| `artifacts/decision.schema.json` | New | Read-only |
| `scripts/validate_decision.py` | New | Protected |
| `scripts/check_protected_surfaces.py` | New | Protected |
| `CLAUDE.md` | Update | Add new protected surfaces |
| `.gitignore` | Update | Add `artifacts/` |

## Acceptance Criteria

- [x] `loop.sh` invokes `claude -p --no-session-persistence --output-format json`
- [x] Claude does NOT run any benchmark scripts
- [x] Orchestrator runs `run_research_experiment.py` for fidelity + benchmark gate
- [x] Orchestrator runs `score_experiment.py` for verdict
- [x] `decision.schema.json` exists and `validate_decision.py` enforces it
- [x] Protected surface check runs before benchmarks
- [x] Baseline captured pre-loop if missing
- [x] Dynamic prompt includes best_result, last 5 experiments, compound notes index
- [x] `flock` prevents concurrent instances (mkdir-based, macOS compatible)
- [x] `trap` handler reverts on SIGINT/SIGTERM
- [x] Full cleanup between iterations (`git clean -fd`, `__pycache__` purge)
- [x] Each iteration archived to `artifacts/runs/`
- [x] `artifacts/runs/loop-log.jsonl` records every iteration's verdict + reason
- [x] KEEP commits include experiment name and score in message
- [x] Stall counter resets on KEEP, halts at MAX_STALLS

## Open Questions

- `--no-session-persistence` + hooks: do `.claude/settings.json` hooks still load? If not, `check_protected_surfaces.py` is the critical safety net — verify this
- `score_experiment.py` last-line JSON parsing — confirm exact output format
- `summarize_experiment.py` on REVERT path — does it handle REVERT verdict correctly? (appends to JSONL but should NOT update best_result.json)
- Should `best_result.json` update use composite score instead of raw `median_ms`?
- Claude timeout — 600s (10min) default reasonable?
- Should prompt include `research/program.md` contents inline or let Claude read it?

## References

- Design doc: `docs/Claude Code Autonomous Optimization Loop.md`
- Brainstorm: `docs/brainstorms/2026-03-10-loop-orchestrator-upgrade-brainstorm.md`
- Benchmark spec: `research/benchmark_spec.md`
- Research program: `research/program.md`
- Current loop: `loop.sh`
