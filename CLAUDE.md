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
