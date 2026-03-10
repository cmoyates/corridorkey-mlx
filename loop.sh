#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ITERATIONS="${1:-10}"
MAX_STALLS="${MAX_STALLS:-3}"
STALLS=0

require_clean_tree() {
  if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Working tree is not clean. Commit or stash changes first."
    exit 1
  fi
}

read_json_field() {
  python3 - "$1" "$2" <<'PY'
import json, sys
path, key = sys.argv[1], sys.argv[2]
with open(path) as f:
    data = json.load(f)
print(data.get(key, ""))
PY
}

require_clean_tree

mkdir -p artifacts/runs

for ((i=1; i<=ITERATIONS; i++)); do
  echo
  echo "=============================="
  echo "Iteration $i"
  echo "=============================="

  git add -A
  git commit -m "checkpoint: before iteration $i" --allow-empty >/dev/null

  PROMPT=$(cat <<'EOF'
Run exactly one bounded optimization iteration for corridorkey-mlx.

Rules:
- Start by reading CLAUDE.md, research/program.md, research/benchmark_spec.md,
  research/best_result.json, research/experiments.jsonl, and the latest notes in research/compound/.
- Fidelity is a regression gate, not an optimization target.
- Choose exactly one experiment in an allowed phase-1 search area.
- Modify only allowed files.
- Reuse the structured benchmark scripts and existing repo benchmark surfaces.
- Write:
  - artifacts/latest_result.json
  - artifacts/latest_summary.md
  - artifacts/latest_decision.json
- Update:
  - research/experiments.jsonl
  - the relevant research/compound/*.md note
- latest_decision.json must contain:
  - status: keep | revert | inconclusive
  - reason: short string
  - next_experiment: short string
- If fidelity fails, status must be revert.
- If there is no meaningful latency or peak-memory improvement, use revert or inconclusive.
- Propose exactly one next experiment.
EOF
)

  claude -p "$PROMPT" \
    --allowedTools "Read,Edit,Write,Bash,Grep,Glob" || true

  if [[ ! -f artifacts/latest_decision.json ]]; then
    echo "No decision file produced. Reverting."
    git reset --hard HEAD~1 >/dev/null
    exit 1
  fi

  STATUS="$(read_json_field artifacts/latest_decision.json status)"
  REASON="$(read_json_field artifacts/latest_decision.json reason)"
  NEXT="$(read_json_field artifacts/latest_decision.json next_experiment)"

  echo "Decision: $STATUS"
  echo "Reason:   $REASON"
  echo "Next:     $NEXT"

  case "$STATUS" in
    keep)
      git add -A
      git commit -m "exp: iteration $i" >/dev/null
      STALLS=0
      ;;
    revert)
      git reset --hard HEAD~1 >/dev/null
      STALLS=$((STALLS + 1))
      ;;
    inconclusive|*)
      git reset --hard HEAD~1 >/dev/null
      STALLS=$((STALLS + 1))
      ;;
  esac

  cp artifacts/latest_decision.json "artifacts/runs/decision-$i.json" || true
  cp artifacts/latest_result.json "artifacts/runs/result-$i.json" || true
  cp artifacts/latest_summary.md "artifacts/runs/summary-$i.md" || true

  if (( STALLS >= MAX_STALLS )); then
    echo "Stopping after $STALLS non-winning iterations."
    break
  fi
done