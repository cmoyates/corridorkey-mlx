# Exp 29: MLX Buffer Limits — 58% Peak Memory Reduction

## Finding

Setting `MLX_MAX_MB_PER_BUFFER` and `MLX_MAX_OPS_PER_BUFFER` to small values
dramatically reduces peak Metal memory with zero latency penalty.

## Best Config

```bash
MLX_MAX_MB_PER_BUFFER=2 MLX_MAX_OPS_PER_BUFFER=2
```

## Results @1024

| Config | Median | Peak Memory | vs Default |
|--------|--------|-------------|-----------|
| Default | ~450ms | 3319MB | — |
| MB=2, OPS=2 | ~437ms | 1407MB | **-58%** |

## Memory Scaling

| MB | Peak Memory |
|-----|-------------|
| 1 | 1400MB |
| 2 | 1380-1565MB |
| 4 | 1496MB |
| 8 | 2099MB |
| 16 | 2709MB |
| 32 | 3319MB (=default) |
| 64 | 3319MB |
| 128 | 4706MB |

OPS at MB=2: OPS=2→1407MB, OPS=4→1407MB, OPS=8→1597MB, OPS=32→1725MB.

## Why It Works

Smaller command buffers force MLX to flush GPU work more frequently,
allowing intermediate Metal allocations to be freed sooner. This reduces
the high-water mark of concurrent allocations without affecting throughput
because the GPU is compute-bound, not dispatch-bound.

## Caveats

- Latency difference is within noise — confirmed via thermal-controlled
  alternating A/B test (3 reps each, interleaved)
- Fidelity: PASS at all buffer sizes (outputs are identical)
- `mx.compile` fails with "Not allowed inside a graph transformation" when
  buffer limits are set — need to investigate if this is a pre-existing issue
- These are env vars, not code changes. Can be set in inference scripts or
  as process-level defaults.

## Application

Add to inference entry points:
```python
import os
os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "2")
os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "2")
# must be set BEFORE importing mlx
```

Or document as recommended launch config:
```bash
MLX_MAX_MB_PER_BUFFER=2 MLX_MAX_OPS_PER_BUFFER=2 uv run python scripts/infer.py ...
```
