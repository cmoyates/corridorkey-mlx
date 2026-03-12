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
RESOLUTION="${RESOLUTION:-1024}"

STALLS=0
STALL_TYPE=""
LAST_ERROR=""

ARTIFACTS_DIR="$ROOT/artifacts"
RUNS_DIR="$ARTIFACTS_DIR/runs"
DECISION_PATH="$ARTIFACTS_DIR/latest_decision.json"
BASELINE_PATH="$ARTIFACTS_DIR/benchmark_baseline_${RESOLUTION}.json"
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
  cp "$ROOT"/research/best_result_*.json /tmp/ 2>/dev/null || true
  cp -r "$ROOT/research/compound" /tmp/_compound_backup 2>/dev/null || true
  git -C "$ROOT" checkout -- . 2>/dev/null || true
  git -C "$ROOT" clean -fd -q 2>/dev/null || true
  cp /tmp/_exp_jsonl_backup "$ROOT/research/experiments.jsonl" 2>/dev/null || true
  cp /tmp/_best_result_backup "$ROOT/research/best_result.json" 2>/dev/null || true
  cp /tmp/best_result_*.json "$ROOT/research/" 2>/dev/null || true
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
    --resolution "$RESOLUTION" \
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
  local baseline_median_ms baseline_memory_mb

  # Extract baseline numbers for dynamic prompt
  baseline_median_ms="$(python3 -c "
import json
with open('$BASELINE_PATH') as f:
    print(json.load(f)['benchmark']['median_ms'])
" 2>/dev/null || echo "?")"
  baseline_memory_mb="$(python3 -c "
import json
with open('$BASELINE_PATH') as f:
    print(json.load(f)['peak_memory_mb'])
" 2>/dev/null || echo "?")"

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

  # ALL experiments from log (proposer must see full history to avoid repeats)
  last_experiments="$(cat "$ROOT/research/experiments.jsonl" 2>/dev/null || echo "(no experiments yet)")"

  # Extract unique tried experiment names for explicit dedup
  local tried_names
  tried_names="$(python3 -c "
import json, sys
names = set()
for line in open('$ROOT/research/experiments.jsonl'):
    line = line.strip()
    if not line or line.startswith('{') is False: continue
    try: names.add(json.loads(line)['experiment_name'])
    except: pass
for n in sorted(names): print(f'  - {n}')
" 2>/dev/null || echo "  (none)")"

  # Current best result (per-resolution, fallback to legacy)
  if [[ -f "$ROOT/research/best_result_${RESOLUTION}.json" ]]; then
    best_result="$(cat "$ROOT/research/best_result_${RESOLUTION}.json")"
  else
    best_result="$(cat "$ROOT/research/best_result.json" 2>/dev/null || echo "(no best result yet)")"
  fi

  # Steering: compute which priority tier to target this iteration
  local steering_directive
  steering_directive="$(python3 -c "
import json, sys
TOP_AREAS = ['selective-precision', 'tensor-layout-staging', 'refiner-dilated-conv-fix']
MED_AREAS = ['token-routing', 'stream-pipelining', 'fused-metal-kernels', 'matmul-ordering', 'addmm-fusion', 'dtype-cast-cleanup']
tried = set()
area_attempts = {}
try:
    for line in open('$ROOT/research/experiments.jsonl'):
        line = line.strip()
        if not line: continue
        e = json.loads(line)
        sa = e.get('search_area', '')
        if sa: area_attempts[sa] = area_attempts.get(sa, 0) + 1
except: pass

# Find top-priority areas with < 2 attempts
untried_top = [a for a in TOP_AREAS if area_attempts.get(a, 0) < 2]
untried_med = [a for a in MED_AREAS if area_attempts.get(a, 0) < 2]

if untried_top:
    target = untried_top[0]
    print(f'MANDATORY: You MUST target search area \"{target}\" this iteration. This is a TOP PRIORITY area that has not been adequately explored. Do NOT choose a different area.')
elif untried_med:
    target = untried_med[0]
    print(f'STRONGLY RECOMMENDED: Target search area \"{target}\" — all top-priority areas have been explored.')
else:
    print('All priority areas explored. Choose any allowed area with a novel approach.')
" 2>/dev/null || echo "")"

  # Compound notes — build compact index (title + verdict per file)
  local compound_index=""
  if ls "$ROOT/research/compound/"*.md 1>/dev/null 2>&1; then
    compound_index="$(ls -t "$ROOT/research/compound/"*.md | while read -r f; do
      local title verdict
      title="$(head -1 "$f" | sed 's/^# //')"
      verdict="$(grep -m1 '^\*\*Verdict:\*\*' "$f" 2>/dev/null | sed 's/\*\*Verdict:\*\* //' || echo "")"
      if [[ -n "$verdict" ]]; then
        echo "- $(basename "$f"): $title [$verdict]"
      else
        echo "- $(basename "$f"): $title"
      fi
    done)"
  else
    compound_index="(no compound notes yet)"
  fi

  cat <<PROMPT
You are a stateless MLX optimization engineer. You will propose exactly one
bounded code mutation for corridorkey-mlx. You do NOT run benchmarks — the
orchestrator does that after you exit.

## STEERING DIRECTIVE (from orchestrator)
$steering_directive

CRITICAL REQUIREMENT: You MUST use the Write tool to create the file
artifacts/latest_decision.json before you finish. The loop WILL FAIL if this
file does not exist. This is your primary deliverable — code changes without
this file are worthless.

## Your contract
1. Read CLAUDE.md and research/program.md to understand the project.
2. Search research/compound/ for learnings related to your planned experiment.
   Use Grep to find relevant notes, then Read the matching files.
3. Read relevant source files you plan to modify.
4. Choose one experiment from the allowed search areas below.
5. Implement the minimal code change (one variable at a time).
6. MANDATORY: Write artifacts/latest_decision.json (see schema below).
7. Exit immediately. Do NOT run any benchmark or scoring scripts.

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

## ALL previous experiments (full history)
$last_experiments

## ALREADY TRIED — DO NOT REPEAT these experiment names or approaches:
$tried_names
You MUST choose an experiment that is NOT in the list above. If you propose a
duplicate name, the loop will reject it. Try a genuinely different approach.

## Compound learnings (do not repeat failed approaches)
The directory research/compound/ contains hard-won lessons from previous
experiments. BEFORE choosing your experiment, you MUST:
1. Use Grep to search research/compound/ for keywords related to your planned
   search area (e.g., "quantiz", "bf16", "refiner", "compile", "fidelity")
2. Use Read to read any matching files in full
3. Do NOT repeat any approach marked as failed/reverted/error

Available notes:
$compound_index

## KEY INSIGHT — ROOT CAUSE OF PLATEAU
The model is MEMORY-BANDWIDTH BOUND, not compute-bound. Baseline: ${baseline_median_ms}ms median, ${baseline_memory_mb}MB peak.
The latency is dominated by:
1. Dilated convolutions in refiner force im2col fallback (9x activation memory inflation)
2. FP32 backbone activations double bandwidth vs BF16
3. Hidden non-contiguous tensor copies from Hiera's unroll/reroll across 24 blocks

Read research/compound/deep_dive_findings.md for full analysis.

## Allowed search areas (PRIORITY ORDER — try highest impact first)
### TOP PRIORITY — highest expected impact
2. selective-precision — BACKBONE BF16 (not just decoders). BF16 has same 8-bit exponent as FP32 (preserves dynamic range, unlike FP16 which failed). Only final sigmoid needs FP32. Expected: 20-30% latency by halving activation traffic.
5. tensor-layout-staging — AUDIT Hiera unroll/reroll for hidden non-contiguous copies. MLX inserts implicit memory copies when strided tensors hit Linear/SDPA kernels. Minimize reroll frequency. Expected: 8-12% latency.
21. refiner-dilated-conv-fix — Dilated convs force im2col fallback (excluded from implicit GEMM per MLX PR #3147). Replace with stride-2 downsample + standard conv + bilinear upsample per block. Expected: 15-20% latency + memory.

### MEDIUM PRIORITY
9. token-routing — skip attention for easy tokens (alpha hint near 0/1), identity LTRM at stages 2-3. Expected: 50-80% attention FLOP reduction at 512x512 green screen.
14. stream-pipelining — dispatch alpha and fg decoder heads to parallel GPU command queues via mx.stream(). Only works without compile_forward. Expected: 3-5%.
11. fused-metal-kernels — mx.fast.metal_kernel() for refiner conv+GN+GELU. NOTE: custom kernels break mx.compile fusion, so net effect is unclear.
17. matmul-ordering — ensure x @ W.T pattern (faster than x @ W) in attention projections and decoder linears.
18. addmm-fusion — use mx.addmm for fused add+matmul in decoder/refiner linear layers.
19. dtype-cast-cleanup — remove unnecessary astype() calls around mx.fast functions (they accumulate in higher precision internally).

### LOWER PRIORITY
1. tile-lifecycle-memory — del refs, gc timing, avoid redundant allocs in tiled loops
3. tiled-inference-heuristics — tile size/overlap sweeps, blending strategies
4. compile-path-policy — mx.compile for fixed shapes, warmup-aware
6. backbone-quantization — ONLY stages 1-3 (dims 224,448,896 divisible by 32). Stage 0 (dim=112) must stay fp32.
7. mlx-memory-tuning — mx.set_wired_limit() for p95, mx.set_cache_limit() for peak memory
10. refiner-only-tiling — backbone+decoder once at full res, tile only CNN refiner
12. sdpa-attention — ALREADY APPLIED. mx.fast.scaled_dot_product_attention is already used in MaskUnitAttention.
13. graph-materialization — strategic materialization to reduce peak live tensor count
15. weight-format — convert weights to contiguous/optimal layout at load time
20. async-pipeline — mx.async_eval + mx.new_stream for overlapping preprocessing with GPU inference
22. edge-aware-blend — only ramp tile edges that overlap with adjacent tiles, keep full weight at image boundaries (quality fix, not speed). Source: EZ-CorridorKey.

### ELIMINATED — DO NOT ATTEMPT
8. ELIMINATED — nn.LayerNorm already dispatches to mx.fast.layer_norm.
16. ELIMINATED — mx.compile already fuses element-wise ops (depth 11, 24 arrays).

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
echo "  Resolution: ${RESOLUTION}x${RESOLUTION}"
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

  # --- Gate 1.5: Duplicate experiment detection ---
  if grep -q "\"experiment_name\": \"$EXP_NAME\"" "$ROOT/research/experiments.jsonl" 2>/dev/null; then
    echo "[DUPLICATE] Experiment '$EXP_NAME' was already tried. Reverting."
    LAST_ERROR="DUPLICATE EXPERIMENT: '$EXP_NAME' was already tried and logged in experiments.jsonl. You MUST choose a completely different experiment name and approach. Read the full experiment history carefully."
    revert_changes
    STALLS=$((STALLS + 1)); STALL_TYPE="duplicate"
    log_iteration "$i" "REVERT" "duplicate_experiment" "$EXP_NAME"
    iter_cleanup
    if (( STALLS >= MAX_STALLS )); then
      echo "Stopping: $STALLS consecutive failures (type: $STALL_TYPE)"
      break
    fi
    continue
  fi

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
    --resolution "$RESOLUTION" \
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
        --result "$RESULT_FILE" --decision "$DECISION_PATH" --verdict REVERT --notes "fidelity failure, loop iteration $i"
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
    --result "$RESULT_FILE" \
    --baseline "$BASELINE_PATH" 2>&1)" || {
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
    KEEP|WEAK_KEEP)
      echo "[$VERDICT] score=$SCORE — committing changes"
      # Commit BEFORE cleanup (preserves new files)
      git -C "$ROOT" add -A
      git -C "$ROOT" commit -m "exp: $EXP_NAME [score=$SCORE, verdict=$VERDICT]" -q
      # Log to experiment history
      uv run python "$ROOT/scripts/summarize_experiment.py" \
        --result "$RESULT_FILE" --decision "$DECISION_PATH" --verdict KEEP --notes "loop iteration $i"
      STALLS=0
      STALL_TYPE=""
      LAST_ERROR=""
      log_iteration "$i" "$VERDICT" "score=$SCORE" "$EXP_NAME"
      ;;
    REVERT)
      echo "[REVERT] score=$SCORE — reverting changes"
      # Log BEFORE revert (result file is in artifacts/runs/, survives clean)
      uv run python "$ROOT/scripts/summarize_experiment.py" \
        --result "$RESULT_FILE" --decision "$DECISION_PATH" --verdict REVERT --notes "loop iteration $i"
      revert_changes
      LAST_ERROR="experiment '$EXP_NAME' reverted (score=$SCORE, performance regression)"
      STALLS=$((STALLS + 1)); STALL_TYPE="performance"
      log_iteration "$i" "REVERT" "performance_regression score=$SCORE" "$EXP_NAME"
      ;;
    *)
      echo "[INCONCLUSIVE] score=$SCORE — reverting (within noise)"
      uv run python "$ROOT/scripts/summarize_experiment.py" \
        --result "$RESULT_FILE" --decision "$DECISION_PATH" --verdict INCONCLUSIVE --notes "loop iteration $i"
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
