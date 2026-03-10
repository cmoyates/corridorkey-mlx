# benchmark-review

Read the latest experiment result and recommend keep/revert.

## When to use
- After running an experiment
- When reviewing accumulated results
- Before deciding whether to commit a change

## Behavior
1. Find the most recent JSON in `research/artifacts/`
2. Run scoring logic mentally (or invoke `scripts/score_experiment.py`)
3. Compare against `research/best_result.json`
4. Recommend: KEEP, REVERT, or INCONCLUSIVE

## Checklist
- [ ] Fidelity gates all pass?
- [ ] Median latency improved by >= 2%?
- [ ] Peak memory improved by >= 5%?
- [ ] No metric regressed beyond noise?
- [ ] Score > baseline score?

## Output
```
VERDICT: [KEEP/REVERT/INCONCLUSIVE]
Reason: [one sentence]
Score: [X.XXXX] (baseline: 1.0000)
Next: [suggested action]
```
