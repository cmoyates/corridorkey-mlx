# upstream-research

Read-only research skill for exploring upstream CorridorKey, MLX source, and MLX examples.

## When to use
- Before planning an experiment, to check if upstream has relevant changes
- When investigating MLX API patterns or optimizations
- When looking for prior art in CorridorKey

## Behavior
- Read-only: do not modify any files
- Search local code, git history, and BTCA resources if available
- Classify findings as: `mlx-portable`, `concept-only`, `pytorch-only`

## BTCA resources (if available)
Check `btca.config.jsonc` for configured repos:
- `mlx` — MLX framework source
- `mlxExamples` — MLX examples repo
- `corridorKey` — upstream CorridorKey repo (nikopueringer)
- `corridorKeyEngine` — CorridorKey Engine (99oblivius)
- `corridorKeyMarcel` — Marcel Lieb's CorridorKey fork
- `ezCorridorKey` — EZ-CorridorKey (edenaion)

## Without BTCA
Fall back to:
- Local git log analysis
- Grep/glob on local files
- Web search for MLX documentation

## Output format
For each finding, report:
- **Source**: repo/PR/file
- **Summary**: one sentence
- **Classification**: mlx-portable / concept-only / pytorch-only
- **Relevance**: how it could inform an experiment
