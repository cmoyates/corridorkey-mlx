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
| 1 | PyTorch reference harness + fixture dump | Done |
| 2 | MLX decoder/refiner blocks + parity tests | Done |
| 3 | Checkpoint conversion (PyTorch → MLX) | Done |
| 4 | Hiera backbone port | Done |
| 5 | Full model assembly + e2e parity | Done |
| 6 | Optimization + benchmarking | Done |

See `prompts/` for detailed phase instructions.

## Usage

### Setup

```bash
uv sync --group dev
```

### Convert weights

Convert the PyTorch checkpoint to MLX safetensors (one-time):

```bash
uv run python scripts/convert_weights.py \
    --checkpoint checkpoints/CorridorKey_v1.0.pth \
    --output checkpoints/corridorkey_mlx.safetensors
```

### Single-image inference

```bash
uv run python scripts/infer.py \
    --image input.png \
    --hint alpha_hint.png \
    --output-dir output/
```

Outputs `output/alpha.png` (alpha matte) and `output/foreground.png` (foreground).

Options:
- `--checkpoint PATH` — MLX safetensors file (default: `checkpoints/corridorkey_mlx.safetensors`)
- `--img-size N` — model input resolution (default: 512)
- `--output-dir DIR` — output directory (default: `output/`)

### Python API

```python
from corridorkey_mlx.inference.pipeline import load_model, infer_and_save

model = load_model("checkpoints/corridorkey_mlx.safetensors", img_size=512)
results = infer_and_save(model, "input.png", "alpha_hint.png", "output/")
```

## Development

```bash
uv run pytest              # tests
uv run ruff check .        # lint
uv run ruff format .       # format
uv run mypy src/           # type check
```

For PyTorch reference work:
```bash
uv sync --group reference
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

## Parity Results

End-to-end parity vs PyTorch reference (512×512, float32):

| Tensor | Max Abs Error | Mean Abs Error |
|--------|--------------|----------------|
| alpha_logits | 8.8e-05 | 1.6e-05 |
| fg_logits | 1.5e-04 | 7.2e-06 |
| alpha_coarse | 9.7e-06 | 1.1e-06 |
| fg_coarse | 6.7e-06 | 1.1e-06 |
| delta_logits | 1.1e-04 | 4.3e-06 |
| alpha_final | 2.6e-05 | 8.7e-08 |
| fg_final | 9.5e-06 | 1.1e-06 |

## Performance

### Compiled inference

Use `compile=True` for fused execution on fixed-resolution inputs:

```python
model = load_model("checkpoints/corridorkey_mlx.safetensors", img_size=512, compile=True)
```

The first call incurs a one-time compilation cost. Subsequent calls at the same
resolution run faster. Shapeless compilation (`shapeless=True`) is **not recommended**
due to shape-dependent reshapes in the Hiera backbone.

### Benchmarking

```bash
uv run python scripts/bench_mlx.py
uv run python scripts/bench_mlx.py --resolutions 256 512 1024 --bench-runs 20
```

Reports eager vs compiled latency, warmup cost, and parity check per resolution.

### Large images (tiled inference)

For images larger than the model's input resolution, use tiled inference with
overlap blending:

```python
from corridorkey_mlx.inference.tiling import tiled_inference

model = load_model("checkpoints/corridorkey_mlx.safetensors", img_size=512)
x = preprocess(rgb, alpha_hint)  # full-resolution (1, H, W, 4)
result = tiled_inference(model, x, tile_size=512, overlap=64)
```

### Recommended settings for Apple Silicon

| Setting | Value | Notes |
|---------|-------|-------|
| `img_size` | 512 | Good speed/quality balance |
| `compile` | True | ~1.5–2x faster after warmup |
| `tile_size` | 512 | Match `img_size` for tiling |
| `overlap` | 64 | Smooth blending at tile boundaries |

### Comparing against PyTorch reference

```bash
uv run python scripts/compare_reference.py
```

## Current Status

Phases 1–6 complete. Full model assembly with end-to-end parity verified.
Optimization, benchmarking, and tiled inference available.
