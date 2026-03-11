#!/usr/bin/env bash
# Ephemeral shell-driven optimization loop for corridorkey-mlx.
#
# Claude is a stateless mutation proposer. This orchestrator owns:
#   - Dynamic prompt construction
#   - Fidelity + benchmark gating (run_research_experiment.py)
#   - Scoring + keep/revert decisions (score_experiment.py)
#   - Git state management
#   - Experiment logging (summarize_experiment.py)
#
# Usage:
#   ./loop.sh [iterations]          # default: 10
#   MAX_STALLS=5 ./loop.sh 20       # custom stall limit
#   CLAUDE_TIMEOUT=900 ./loop.sh    # custom timeout (seconds)

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths (absolute — no cd, zoxide-safe)
# ---------------------------------------------------------------------------
ROOT="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"

ITERATIONS="${1:-10}"
MAX_STALLS="${MAX_STALLS:-3}"
CLAUDE_TIMEOUT="${CLAUDE_TIMEOUT:-600}"

STALLS=0
STALL_TYPE=""
LAST_ERROR=""

ARTIFACTS_DIR="$ROOT/artifacts"
RUNS_DIR="$ARTIFACTS_DIR/runs"
DECISION_PATH="$ARTIFACTS_DIR/latest_decision.json"
BASELINE_PATH="$ARTIFACTS_DIR/benchmark_baseline.json"
LOOP_LOG="$RUNS_DIR/loop-log.jsonl"

# ---------------------------------------------------------------------------
# Lock — prevent concurrent instances (mkdir is atomic on all Unix)
# ---------------------------------------------------------------------------
LOCKDIR="$ROOT/.loop.lock"
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  echo "Another loop instance is running (lock: $LOCKDIR)."
  echo "If stale, remove manually: rm -rf $LOCKDIR"
  exit 1
fi

# ---------------------------------------------------------------------------
# Cleanup + trap
# ---------------------------------------------------------------------------
cleanup() {
  local exit_code=$?
  # Revert any uncommitted changes
  if ! git -C "$ROOT" diff --quiet 2>/dev/null; then
    echo "Cleaning up uncommitted changes..."
    git -C "$ROOT" checkout -- . 2>/dev/null || true
    git -C "$ROOT" clean -fd -q 2>/dev/null || true
  fi
  # Release lock
  rm -rf "$LOCKDIR"
  exit $exit_code
}
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
read_json_field() {
  python3 - "$1" "$2" <<'PY'
import json, sys
path, key = sys.argv[1], sys.argv[2]
with open(path) as f:
    data = json.load(f)
print(data.get(key, ""))
PY
}

log_iteration() {
  local iter="$1" verdict="$2" reason="$3" name="$4"
  local ts
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "{\"iteration\":$iter,\"verdict\":\"$verdict\",\"reason\":\"$reason\",\"experiment\":\"$name\",\"timestamp\":\"$ts\"}" \
    >> "$LOOP_LOG"
}

revert_changes() {
  # Preserve experiment log + best result across reverts
  cp "$ROOT/research/experiments.jsonl" /tmp/_exp_jsonl_backup 2>/dev/null || true
  cp "$ROOT/research/best_result.json" /tmp/_best_result_backup 2>/dev/null || true
  cp -r "$ROOT/research/compound" /tmp/_compound_backup 2>/dev/null || true
  git -C "$ROOT" checkout -- . 2>/dev/null || true
  git -C "$ROOT" clean -fd -q 2>/dev/null || true
  cp /tmp/_exp_jsonl_backup "$ROOT/research/experiments.jsonl" 2>/dev/null || true
  cp /tmp/_best_result_backup "$ROOT/research/best_result.json" 2>/dev/null || true
  cp -r /tmp/_compound_backup "$ROOT/research/compound" 2>/dev/null || true
}

iter_cleanup() {
  # Purge __pycache__ and untracked files
  find "$ROOT" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
  git -C "$ROOT" clean -fd -q 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------
require_clean_tree() {
  if ! git -C "$ROOT" diff --quiet || ! git -C "$ROOT" diff --cached --quiet; then
    echo "Working tree is not clean. Commit or stash changes first."
    exit 1
  fi
}

require_clean_tree
mkdir -p "$RUNS_DIR"

# ---------------------------------------------------------------------------
# Baseline capture (once, before loop starts)
# ---------------------------------------------------------------------------
if [[ ! -f "$BASELINE_PATH" ]]; then
  echo "========================================"
  echo "Capturing baseline benchmark..."
  echo "========================================"
  if ! uv run python "$ROOT/scripts/run_research_experiment.py" \
    --experiment-name "baseline" \
    --output "$BASELINE_PATH"; then
    echo "Baseline capture failed (fidelity check). Cannot start loop."
    exit 1
  fi
  echo "Baseline saved: $BASELINE_PATH"
fi

# ---------------------------------------------------------------------------
# Dynamic prompt construction
# ---------------------------------------------------------------------------
build_prompt() {
  local last_experiments best_result compound_index last_error_block

  # Error context from previous iteration
  if [[ -n "$LAST_ERROR" ]]; then
    last_error_block="THE PREVIOUS ITERATION FAILED. You MUST read this error and avoid repeating it:
\`\`\`
$LAST_ERROR
\`\`\`
Adapt your approach based on this error. Do NOT retry the exact same experiment."
  else
    last_error_block="(none)"
  fi

  # Last 5 experiments from log
  last_experiments="$(tail -5 "$ROOT/research/experiments.jsonl" 2>/dev/null || echo "(no experiments yet)")"

  # Current best result
  best_result="$(cat "$ROOT/research/best_result.json" 2>/dev/null || echo "(no best result yet)")"

  # Compound notes — last 3 full contents (most recent learnings)
  local compound_notes=""
  if ls "$ROOT/research/compound/"*.md 1>/dev/null 2>&1; then
    compound_notes="$(ls -t "$ROOT/research/compound/"*.md | head -3 | while read -r f; do
      echo "### $(basename "$f")"
      cat "$f"
      echo ""
    done)"
  else
    compound_notes="(no compound notes yet)"
  fi

  cat <<PROMPT
You are a stateless MLX optimization engineer. You will propose exactly one
bounded code mutation for corridorkey-mlx. You do NOT run benchmarks — the
orchestrator does that after you exit.

CRITICAL REQUIREMENT: You MUST use the Write tool to create the file
artifacts/latest_decision.json before you finish. The loop WILL FAIL if this
file does not exist. This is your primary deliverable — code changes without
this file are worthless.

## Your contract
1. Read CLAUDE.md and research/program.md to understand the project.
2. Read relevant source files you plan to modify.
3. Choose one experiment from the allowed search areas below.
4. Implement the minimal code change (one variable at a time).
5. MANDATORY: Write artifacts/latest_decision.json (see schema below).
6. Exit immediately. Do NOT run any benchmark or scoring scripts.

## Decision schema — MUST write to artifacts/latest_decision.json
Use the Write tool to create this file with valid JSON containing:

Required fields:
  - experiment_name: string (kebab-case, descriptive)
  - hypothesis: string (>= 10 chars, what + why)
  - files_changed: array of file paths you modified
  - search_area: one of the enum values below
Optional fields:
  - next_hypothesis: string (what to try next)
  - notes: string (caveats, context)

Do NOT include status, verdict, score, or benchmark results.

Example (artifacts/latest_decision.json):
{
  "experiment_name": "refiner-bf16-weights",
  "hypothesis": "Cast refiner weights to bf16 to halve memory bandwidth at full resolution",
  "files_changed": ["src/corridorkey_mlx/model/corridorkey.py"],
  "search_area": "selective-precision"
}

## Current best result
$best_result

## Recent experiments (last 5)
$last_experiments

## Compound learnings (MUST READ — do not repeat failed approaches)
$compound_notes

## Allowed search areas
1. tile-lifecycle-memory — del refs, gc timing, avoid redundant allocs in tiled loops
2. selective-precision — refiner fp16, backbone fp32, decoder bf16
3. tiled-inference-heuristics — tile size/overlap sweeps, blending strategies
4. compile-path-policy — mx.compile for fixed shapes, warmup-aware
5. tensor-layout-staging — contiguity, minimize NCHW-NHWC transitions
6. backbone-quantization — nn.quantize(backbone, bits=8), 144 Linear layers dominate compute
7. mlx-memory-tuning — mx.set_wired_limit() for p95, mx.set_cache_limit() for peak memory
8. layernorm-fusion — check if nn.LayerNorm dispatches to mx.fast.layer_norm, swap if not
9. token-routing — skip attention for easy tokens (alpha hint near 0/1), identity LTRM at stages 2-3
10. refiner-only-tiling — backbone+decoder once at full res, tile only CNN refiner
11. fused-metal-kernels — mx.fast.metal_kernel() for conv+GN+GELU in refiner blocks

## Previous iteration error (if any)
$last_error_block

## Rules
- Modify ONLY files in: src/corridorkey_mlx/, scripts/infer.py, scripts/smoke_engine.py
- Do NOT modify: scripts/bench_*, scripts/compare_*, scripts/score_*,
  scripts/run_research_*, scripts/validate_*, scripts/check_protected_*,
  reference/fixtures/, tests/, loop.sh
- Do NOT run: bench_mlx.py, compare_reference.py, run_research_experiment.py,
  score_experiment.py, or any benchmark/scoring script
- One optimization variable per experiment
- If no viable experiment remains, write decision.json with notes explaining why

REMINDER: Your FINAL action before exiting MUST be writing artifacts/latest_decision.json
using the Write tool. Without this file, your entire session is wasted.
PROMPT
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo "Starting optimization loop"
echo "  Iterations: $ITERATIONS"
echo "  Max stalls: $MAX_STALLS"
echo "  Timeout:    ${CLAUDE_TIMEOUT}s"
echo "  Baseline:   $BASELINE_PATH"
echo "========================================"

for ((i=1; i<=ITERATIONS; i++)); do
  echo ""
  echo "========================================"
  echo "Iteration $i / $ITERATIONS  (stalls: $STALLS/$MAX_STALLS)"
  echo "========================================"

  # Checkpoint current state
  git -C "$ROOT" add -A
  git -C "$ROOT" commit -m "checkpoint: before iteration $i" --allow-empty -q

  # Build dynamic prompt and invoke Claude
  PROMPT="$(build_prompt)"
  echo "[claude] Invoking stateless mutation proposer..."
  claude -p "$PROMPT" \
    --no-session-persistence \
    --output-format json \
    --allowedTools "Read,Edit,Write,Bash,Grep,Glob" \
    --max-turns 25 \
    > "$RUNS_DIR/claude-output-$i.json" 2>&1 &
  CLAUDE_PID=$!

  # Timeout guard (macOS has no `timeout` command)
  (
    sleep "$CLAUDE_TIMEOUT"
    if kill -0 "$CLAUDE_PID" 2>/dev/null; then
      echo "[timeout] Claude exceeded ${CLAUDE_TIMEOUT}s — killing."
      kill "$CLAUDE_PID" 2>/dev/null
    fi
  ) &
  TIMER_PID=$!

  wait "$CLAUDE_PID" 2>/dev/null || true
  kill "$TIMER_PID" 2>/dev/null || true
  wait "$TIMER_PID" 2>/dev/null || true

  # --- Gate 1: Validate decision schema ---
  echo "[gate] Validating decision.json..."
  GATE1_ERR=""
  if ! GATE1_ERR="$(uv run python "$ROOT/scripts/validate_decision.py" --decision "$DECISION_PATH" 2>&1)"; then
    echo "$GATE1_ERR"
    echo "[FAIL] Invalid or missing decision.json. Reverting."
    LAST_ERROR="decision.json validation: $GATE1_ERR"
    revert_changes
    STALLS=$((STALLS + 1)); STALL_TYPE="invalid_output"
    log_iteration "$i" "REVERT" "invalid_decision" ""
    iter_cleanup
    if (( STALLS >= MAX_STALLS )); then
      echo "Stopping: $STALLS consecutive failures (type: $STALL_TYPE)"
      break
    fi
    continue
  fi

  EXP_NAME="$(read_json_field "$DECISION_PATH" experiment_name)"
  echo "[info] Experiment: $EXP_NAME"

  # --- Gate 2: Protected surface check ---
  echo "[gate] Checking protected surfaces..."
  GATE2_ERR=""
  if ! GATE2_ERR="$(uv run python "$ROOT/scripts/check_protected_surfaces.py" 2>&1)"; then
    echo "$GATE2_ERR"
    echo "[FAIL] Protected surface modified! Reverting."
    LAST_ERROR="protected surface violation: $GATE2_ERR"
    revert_changes
    STALLS=$((STALLS + 1)); STALL_TYPE="protected_surface"
    log_iteration "$i" "REVERT" "protected_surface_violation" "$EXP_NAME"
    iter_cleanup
    if (( STALLS >= MAX_STALLS )); then
      echo "Stopping: $STALLS consecutive failures (type: $STALL_TYPE)"
      break
    fi
    continue
  fi

  # --- Gate 3: Fidelity + benchmark ---
  RESULT_FILE="$RUNS_DIR/result-$i.json"
  GATE3_LOG="$RUNS_DIR/gate3-$i.log"
  echo "[gate] Running fidelity + benchmark..."
  set +o pipefail
  uv run python "$ROOT/scripts/run_research_experiment.py" \
    --experiment-name "$EXP_NAME" \
    --output "$RESULT_FILE" 2>&1 | tee "$GATE3_LOG"
  GATE3_EXIT="${PIPESTATUS[0]}"
  set -o pipefail
  if [[ "$GATE3_EXIT" -ne 0 ]]; then
    echo ""
    echo "FIDELITY FAILED — candidate must be reverted"
    # Capture last 30 lines of error output for next iteration
    LAST_ERROR="experiment '$EXP_NAME' failed (fidelity/runtime error):
$(tail -30 "$GATE3_LOG")"
    # Archive decision before revert destroys it
    cp "$DECISION_PATH" "$RUNS_DIR/decision-$i.json" 2>/dev/null || true
    # Log fidelity failure to experiment history (result file exists even on failure)
    if [[ -f "$RESULT_FILE" ]]; then
      uv run python "$ROOT/scripts/summarize_experiment.py" \
        --result "$RESULT_FILE" --verdict REVERT --notes "fidelity failure, loop iteration $i"
    fi
    revert_changes
    # Compound: capture learning from failure (fidelity or runtime error)
    if [[ -f "$RESULT_FILE" ]]; then
      COMPOUND_VERDICT="REVERT"
    else
      COMPOUND_VERDICT="ERROR"
    fi
    COMPOUND_ARGS=(--verdict "$COMPOUND_VERDICT" --error-log "$GATE3_LOG" --experiment-name "$EXP_NAME" --notes "loop iteration $i")
    [[ -f "$RESULT_FILE" ]] && COMPOUND_ARGS+=(--result "$RESULT_FILE")
    [[ -f "$RUNS_DIR/decision-$i.json" ]] && COMPOUND_ARGS+=(--decision "$RUNS_DIR/decision-$i.json")
    uv run python "$ROOT/scripts/compound_note.py" "${COMPOUND_ARGS[@]}" || true
    STALLS=$((STALLS + 1)); STALL_TYPE="fidelity"
    log_iteration "$i" "REVERT" "fidelity_failure" "$EXP_NAME"
    iter_cleanup
    if (( STALLS >= MAX_STALLS )); then
      echo "Stopping: $STALLS consecutive failures (type: $STALL_TYPE)"
      break
    fi
    continue
  fi

  # --- Gate 4: Score and decide ---
  echo "[gate] Scoring experiment..."
  SCORE_OUTPUT="$(uv run python "$ROOT/scripts/score_experiment.py" \
    --result "$RESULT_FILE" 2>&1)" || {
    echo "[FAIL] Scoring failed. Reverting."
    LAST_ERROR="scoring failed for '$EXP_NAME': $SCORE_OUTPUT"
    revert_changes
    STALLS=$((STALLS + 1)); STALL_TYPE="scoring_error"
    log_iteration "$i" "REVERT" "scoring_error" "$EXP_NAME"
    iter_cleanup
    if (( STALLS >= MAX_STALLS )); then
      echo "Stopping: $STALLS consecutive failures (type: $STALL_TYPE)"
      break
    fi
    continue
  }

  # Parse machine-readable JSON from last line of score output
  SCORE_JSON="$(echo "$SCORE_OUTPUT" | tail -1)"
  VERDICT="$(echo "$SCORE_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["verdict"])')"
  SCORE="$(echo "$SCORE_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["score"])')"

  echo "$SCORE_OUTPUT"
  echo ""

  # --- Apply decision ---
  case "$VERDICT" in
    KEEP)
      echo "[KEEP] score=$SCORE — committing changes"
      # Commit BEFORE cleanup (preserves new files)
      git -C "$ROOT" add -A
      git -C "$ROOT" commit -m "exp: $EXP_NAME [score=$SCORE, verdict=KEEP]" -q
      # Log to experiment history
      uv run python "$ROOT/scripts/summarize_experiment.py" \
        --result "$RESULT_FILE" --verdict KEEP --notes "loop iteration $i"
      STALLS=0
      STALL_TYPE=""
      LAST_ERROR=""
      log_iteration "$i" "KEEP" "score=$SCORE" "$EXP_NAME"
      ;;
    REVERT)
      echo "[REVERT] score=$SCORE — reverting changes"
      # Log BEFORE revert (result file is in artifacts/runs/, survives clean)
      uv run python "$ROOT/scripts/summarize_experiment.py" \
        --result "$RESULT_FILE" --verdict REVERT --notes "loop iteration $i"
      revert_changes
      LAST_ERROR="experiment '$EXP_NAME' reverted (score=$SCORE, performance regression)"
      STALLS=$((STALLS + 1)); STALL_TYPE="performance"
      log_iteration "$i" "REVERT" "performance_regression score=$SCORE" "$EXP_NAME"
      ;;
    *)
      echo "[INCONCLUSIVE] score=$SCORE — reverting (within noise)"
      uv run python "$ROOT/scripts/summarize_experiment.py" \
        --result "$RESULT_FILE" --verdict INCONCLUSIVE --notes "loop iteration $i"
      revert_changes
      LAST_ERROR="experiment '$EXP_NAME' inconclusive (score=$SCORE, within noise). Try a different approach."
      STALLS=$((STALLS + 1)); STALL_TYPE="inconclusive"
      log_iteration "$i" "INCONCLUSIVE" "within_noise score=$SCORE" "$EXP_NAME"
      ;;
  esac

  # Archive decision
  cp "$DECISION_PATH" "$RUNS_DIR/decision-$i.json" 2>/dev/null || true

  # --- Compound: capture learning from this iteration ---
  if [[ -f "$RESULT_FILE" ]] && [[ -f "$RUNS_DIR/decision-$i.json" ]]; then
    uv run python "$ROOT/scripts/compound_note.py" \
      --result "$RESULT_FILE" --decision "$RUNS_DIR/decision-$i.json" \
      --verdict "$VERDICT" --score "$SCORE" --notes "loop iteration $i"
  fi

  # Cleanup between iterations
  iter_cleanup

  # Stall check
  if (( STALLS >= MAX_STALLS )); then
    echo ""
    echo "Stopping: $STALLS consecutive non-winning iterations (type: $STALL_TYPE)"
    break
  fi
done

echo ""
echo "========================================"
echo "Loop complete"
echo "  Iterations run: $i"
echo "  Final stalls:   $STALLS"
echo "  Log:            $LOOP_LOG"
echo "========================================"
