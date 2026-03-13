# MLX Buffer Env Var Tuning — 17% Isolated, 2s Pipeline

## Finding
`MLX_MAX_MB_PER_BUFFER=2, MLX_MAX_OPS_PER_BUFFER=2` is 17% faster in isolated MLX benchmarks, translates to ~2s real pipeline improvement (2:06 → 2:04).

## Evidence
| Config | Isolated (ms) | Peak Memory |
|---|---|---|
| MB=2 OPS=2 | 1519 | 762MB |
| default | 1832 | 762MB |

Real pipeline: 2:06 → 2:04 (37 frames @ 1920x1080).

## Why it works
Small buffers force MLX to evaluate frequently, preventing computation graph buildup. Tiled inference naturally benefits — each tile's eval is a clean break. Large buffers cause MLX to accumulate cross-tile graphs that are expensive to dispatch.

## Why pipeline gain is smaller
Non-MLX overhead (ffmpeg decode, numpy postprocessing, despill/despeckle, file I/O) is ~50% of pipeline wall time and unaffected by buffer tuning.
