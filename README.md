# corridorkey-mlx

MLX inference port of [CorridorKey](https://github.com/nikopueringer/CorridorKey) for Apple Silicon.

## Architecture

```
RGB image + coarse alpha hint (4ch)
        │
        ▼
┌──────────────────┐
│  Hiera backbone   │  (timm, features_only)
│  → 4 multiscale   │
│    feature maps    │
└──────────────────┘
        │
   ┌────┴────┐
   ▼         ▼
┌───────┐ ┌───────┐
│ Alpha │ │  FG   │
│ head  │ │ head  │
│ (1ch) │ │ (3ch) │
└───────┘ └───────┘
   │         │
   └────┬────┘
        ▼
┌──────────────────┐
│   CNN Refiner     │  RGB + coarse preds (7ch)
│   → delta logits  │  → sigmoid
└──────────────────┘
        │
        ▼
  final alpha + fg
```

## Phased Roadmap

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | PyTorch reference harness + fixture dump | **In progress** |
| 2 | MLX decoder/refiner blocks + parity tests | Not started |
| 3 | Checkpoint conversion (PyTorch → MLX) | Not started |
| 4 | Full inference pipeline | Not started |
| 5 | Optimization + benchmarking | Not started |

See `prompts/` for detailed phase instructions.

## Setup

```bash
uv sync --group dev
```

For PyTorch reference work:
```bash
uv sync --group reference
```

## Development

```bash
uv run pytest              # tests
uv run ruff check .        # lint
uv run ruff format .       # format
uv run mypy src/           # type check
```

## Reference Fixtures

Phase 1 generates golden reference tensors from PyTorch for MLX parity testing.

**Format:** single `reference/fixtures/golden.npz` (numpy compressed archive)

**Generate:**
```bash
uv run --group reference python scripts/dump_pytorch_reference.py \
    --checkpoint checkpoints/CorridorKey_v1.0.pth
```

**Contents (all float32, NCHW, batch=1, img_size=512):**

| Key | Shape | Description |
|-----|-------|-------------|
| `input` | (1, 4, 512, 512) | Random input (seed=42) |
| `encoder_feature_0` | (1, 112, 128, 128) | Backbone stride-4 |
| `encoder_feature_1` | (1, 224, 64, 64) | Backbone stride-8 |
| `encoder_feature_2` | (1, 448, 32, 32) | Backbone stride-16 |
| `encoder_feature_3` | (1, 896, 16, 16) | Backbone stride-32 |
| `alpha_logits` | (1, 1, 128, 128) | Alpha decoder output (H/4) |
| `fg_logits` | (1, 3, 128, 128) | FG decoder output (H/4) |
| `alpha_logits_up` | (1, 1, 512, 512) | Alpha logits upsampled |
| `fg_logits_up` | (1, 3, 512, 512) | FG logits upsampled |
| `alpha_coarse` | (1, 1, 512, 512) | sigmoid(alpha_logits_up) |
| `fg_coarse` | (1, 3, 512, 512) | sigmoid(fg_logits_up) |
| `delta_logits` | (1, 4, 512, 512) | Refiner output (10x scaled) |
| `alpha_final` | (1, 1, 512, 512) | Final alpha prediction |
| `fg_final` | (1, 3, 512, 512) | Final FG prediction |

## Current Status

Phase 1 in progress — reference harness and fixture dump implemented.
