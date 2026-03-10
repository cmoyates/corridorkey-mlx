# compound-finding

Record a concise learning note from an experiment.

## When to use
- After completing an experiment (keep or revert)
- When discovering a reusable insight about MLX performance
- When finding a gotcha or anti-pattern

## Behavior
1. Write a short markdown file to `research/compound/`
2. Filename: `YYYY-MM-DD-[topic].md`
3. Keep it concise — max 20 lines

## Template

```markdown
# [Topic]

**Context**: [What experiment/situation produced this insight]
**Finding**: [The actual insight, 1-2 sentences]
**Evidence**: [Metric or observation that supports it]
**Implication**: [How this should influence future experiments]
```

## Rules
- One finding per file
- Must be grounded in evidence (not speculative)
- Must be actionable for future experiments
- Maps to `/ce:compound` if Compound Engineering plugin is available
