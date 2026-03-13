# CLAUDE.md — corridorkey-mlx

MLX inference port of CorridorKey for Apple Silicon.

## Architecture

- Input: 4ch (RGB + coarse alpha hint)
- Backbone: Hiera (timm, features_only=True) → 4 multiscale features
- Decoder heads: alpha (1ch) + foreground (3ch), upsampled to full res
- Refiner: CNN over RGB + coarse preds (7ch) → additive delta logits → sigmoid

## Repo layout

- `src/corridorkey_mlx/` — main package
  - `model/` — MLX model definitions
  - `convert/` — PyTorch→MLX weight conversion
  - `inference/` — inference pipeline
  - `io/` — image loading, saving, preprocessing
  - `utils/` — shared helpers, layout transforms
- `scripts/` — CLI tools (dump reference, compare, bench)
- `research/` — autoresearch lab (experiments, benchmarks, learnings)
- `prompts/` — phased port instructions
- `reference/` — PyTorch reference harness outputs
- `tests/` — parity and unit tests

## Conventions

- Python 3.12+, uv for deps
- ruff for lint/format, ty for types, pytest for tests
- MLX uses NHWC — centralize layout transforms in `utils/`
- All non-trivial changes need a validation path
- Inference only — no training code
- Preserve PyTorch behavior before optimizing

## Shell

- Do not use `cd` — zoxide overrides it and breaks non-interactive shells. Use absolute paths instead.

## Commands

```bash
uv run pytest                # run tests
uv run ruff check .          # lint
uv run ruff format .         # format
uv run ty check              # type check
```

## Research Lab — Optimization Workflow

### Fidelity policy
- Fidelity is NOT an optimization target
- Fidelity is a regression gate ONLY
- Any candidate failing fidelity thresholds is rejected, regardless of speed gains
- Correctness dominates speed — always

### Mutable surfaces (safe to modify)
- `src/corridorkey_mlx/` — model, inference, io, utils code
- `scripts/infer.py`, `scripts/smoke_engine.py`
- `research/experiments.jsonl` — append-only experiment log
- `research/compound/` — learning notes
- `research/best_result.json` — current best

### Protected surfaces (do NOT modify without explicit approval)
- `scripts/bench_mlx.py` — benchmark truth source
- `scripts/compare_reference.py` — parity truth source
- `scripts/smoke_2048.py` — stability truth source
- `scripts/bench_optimizations.py` — optimization matrix truth source
- `scripts/score_experiment.py` — scoring logic
- `scripts/run_research_experiment.py` — experiment runner
- `scripts/validate_decision.py` — decision schema validator
- `scripts/check_protected_surfaces.py` — protected surface guard
- `loop.sh` — orchestrator (shell-driven loop)
- `research/decision.schema.json` — decision output contract
- `research/benchmark_spec.md` — benchmark spec
- `reference/fixtures/golden.npz` — golden reference
- `tests/` — existing parity/unit tests

### Benchmark discipline
1. Never benchmark without warmup (min 3 runs)
2. Report median, not mean (outlier-resistant)
3. Always measure steady-state separately from cold-start
4. Peak memory must be measured on a fresh model instance
5. Parity check against golden reference required for every experiment
6. All results go to structured JSON, not free-form logs

### Experiment tracking
- Experiments are tracked as GitHub issues on `cmoyates/corridorkey-mlx`
- Before starting an experiment, check issues for existing context and related work
- New experiment ideas → create a GitHub issue with hypothesis, approach, and expected impact
- After completing an experiment → update the issue with results and close if resolved
- Cross-reference related issues (e.g., blockers, dependencies)

### Experiment loop
1. Plan: hypothesis + target files + benchmark commands + rollback criteria
2. Implement: minimal change, one variable at a time
3. Benchmark: `uv run python scripts/run_research_experiment.py`
4. Score: `uv run python scripts/score_experiment.py`
5. Decide: keep / revert based on scoring output
6. Record: append to `research/experiments.jsonl` + write compound note + update GitHub issue

### Quantization
- NEVER use `nn.quantize(block, ...)` directly — Hiera stage 0 has dim=112, not divisible by 32, and will crash
- ALWAYS use `from corridorkey_mlx.utils.quantize import safe_quantize` — it skips incompatible layers automatically

### Scope discipline
- Do not widen scope casually
- One optimization variable per experiment
- If an experiment touches >3 files, reconsider scope
- Architecture redesign, training, CoreML/ANE, temporal coherence = out of scope
