# plan-experiment

Plan a single bounded optimization experiment.

## When to use
- Before implementing any optimization change
- When choosing between multiple experiment ideas
- Manual trigger only — not auto-invoked

## Output format

```markdown
## Experiment: [name]

### Hypothesis
[One sentence: what you expect to improve and why]

### Target files
- [file1.py]: [what changes]
- [file2.py]: [what changes]

### Benchmark commands
```bash
uv run python scripts/run_research_experiment.py --experiment-name "[name]"
uv run python scripts/score_experiment.py --result research/artifacts/[name]_*.json
```

### Fidelity risks
- [What could break fidelity and why]

### Rollback criteria
- Revert if: [specific conditions]
- Keep if: [specific conditions]

### Estimated scope
- Files touched: N
- Lines changed: ~N
- Risk level: low/medium/high
```

## Rules
- One variable per experiment
- Max 3 files touched
- Must include rollback criteria
- Must identify fidelity risks
- Must reference specific benchmark commands
