# Handoff: Next Optimization Session

## Where we are

**Torch 3:34 → MLX 0:53 = 3.96× faster** (37 frames @ 1920×1080)

### Per-frame breakdown (768px tiles, 6 tiles/frame)

| Phase | ms/frame | % | Bottleneck? |
|---|---|---|---|
| Read | 4.7 | 0.2% | No |
| **Infer** | **1429** | **67%** | Model floor (~218ms/tile × 6 + 120ms tiling overhead) |
| Postprocess | 89.8 | 4.2% | Minor |
| **Write** | **604.6** | **28.4%** | **YES — biggest non-inference target** |
| **Total** | **2132** | 100% | |

### What changed this session
- Tile size 512→768: **2:04 → 0:53** (massive — fewer tiles is the single biggest win)
- Frozen GN enabled by default (perfect tiling correctness, 0 cost at 768px)
- Dead code removed (compile_forward, forward_eager, quantize_backbone_stages)
- Fidelity: alpha_final 2.78e-3, fg_final 4.04e-3 (both pass <5e-3 gate)

### What was tested and didn't help
- Overlap 128→64: no difference at 768px (overlap is small fraction of tile)
- GPU preprocessing (preprocess_mlx): +4s slower (extra copy overhead)
- Whole-forward mx.compile: no benefit over per-component compilation
- Refiner FP16 toggle: no difference
- Frozen GN overhead: 0ms at 768px (amortized)
- 1024px tiles: regressed (memory pressure)

## Current branch
`feat/misc-optimizations` on `cmoyates/corridorkey-mlx`, branched from main.

## Files modified this session
- `src/corridorkey_mlx/inference/tiling.py` — DEFAULT_TILE_SIZE 512→768
- `src/corridorkey_mlx/inference/pipeline.py` — refiner_frozen_gn default True
- `src/corridorkey_mlx/model/corridorkey.py` — removed compile_forward, forward_eager, quantize_backbone_stages
- `src/corridorkey_mlx/engine.py` — compile=False for tiled path (per-component sufficient)

## How to benchmark

Always benchmark through the main CorridorKey repo, not isolated MLX scripts:

```bash
# Install latest corridorkey-mlx into CorridorKey venv
VIRTUAL_ENV=../CorridorKey/.venv uv pip install --reinstall --python ../CorridorKey/.venv/bin/python "corridorkey-mlx @ git+file:///path/to/corridorkey-mlx"

# Or from remote
VIRTUAL_ENV=../CorridorKey/.venv uv pip install --reinstall --python ../CorridorKey/.venv/bin/python "corridorkey-mlx @ git+ssh://git@github.com/cmoyates/corridorkey-mlx.git@feat/misc-optimizations"

# Run benchmark (shows per-phase timing)
../CorridorKey/.venv/bin/python ../CorridorKey/corridorkey_cli.py run-inference --backend mlx --linear --despill 5 --despeckle --refiner 1.0
```

The progress bar shows total time. Per-phase timing (read/infer/postprocess/write) is printed in the log.

**Test clip**: `ClipsForInference/BetterGreenScreenTest_BASE/` — 37 frames @ 1920×1080.

**Baseline**: 0:53 (2132ms/frame)

## Project board

https://github.com/users/cmoyates/projects/2

22 open issues, prioritized by tier. Work in tier order (Tier 0 first).

### Tier 0 — Write I/O elimination (>200ms/frame each)

| # | Title | Expected savings | Approach |
|---|---|---|---|
| **34** | FFmpeg stdin pipe | >500ms | Pipe raw frames to ffmpeg subprocess, eliminate per-frame file I/O |
| **33** | Deferred raw binary output | >400ms | numpy.tofile() during inference, background EXR conversion |
| **37** | Dynamic single-tile inference | >1000ms (conditional) | If subject fits in 768px bbox, skip tiling entirely |
| **35** | OpenEXR 3.4.4 ZIP compression | >200ms | Replace cv2.imwrite with compiled OpenEXR bindings |

**Start with #34 or #33** — they attack the 605ms/frame write bottleneck. #34 is most radical (pipe to ffmpeg), #33 is simplest (raw dump + background convert). #37 is content-dependent but gives the biggest win when applicable.

### Tier 1 — Inference / tiling optimization (50-80ms/frame)

| # | Title | Expected savings |
|---|---|---|
| **32** | MLX scatter_add tile blending | 60-80ms |
| **36** | Overlap-free tiling + seam correction | >70ms |
| **27** | Async I/O pipeline | 30-40% overall |
| **39** | Entropy-guided token pruning | >50ms |

### Tier 2 — Medium wins (10-50ms/frame)

| # | Title | Expected savings |
|---|---|---|
| **40** | Sub-resolution refiner | >40ms |
| **41** | Stage 2 weight similarity | 20-30ms |
| **42** | Speculative frame pipelining | 10-15% throughput |
| **25** | Configurable outputs | 15-20% |
| **45** | Zero-copy numpy views | 10-20ms |

### Tier 3 — Investigation / long shots

#24, #28, #30, #31, #38, #43, #44, #46, #47, #48

## Fidelity gates

| Change type | Threshold |
|---|---|
| Precision changes | alpha_final, fg_final max_abs < 5e-3 vs golden |
| Pipeline-only changes | Byte-identical output files |
| Algorithmic changes | Alpha PSNR >35dB, SSIM >0.97, dtSSD <1.5 |

Golden references: `reference/fixtures/golden.npz` (512) and `golden_2048.npz` (2048). Regenerate with `scripts/dump_pytorch_reference.py`.

Fidelity check: `uv run python scripts/compare_reference.py --img-size 512`

## Key context

- The upstream CorridorKey repo (`../CorridorKey`) already has `DEFAULT_MLX_TILE_SIZE = 768` and per-phase timing instrumentation in `clip_manager.py`
- The CorridorKey backend.py `_wrap_mlx_output` does despill/despeckle/composite in numpy — that's the 90ms postprocess cost
- Write I/O is 4 files per frame: FG (EXR), Matte (EXR), Comp (PNG), Processed (EXR)
- `cv2.imwrite` for EXR uses slow zlib compression and blocks the main thread
- The MLX engine sets `MLX_MAX_MB_PER_BUFFER=2, MLX_MAX_OPS_PER_BUFFER=2` via `os.environ.setdefault` — don't override
- Model cost per tile is 218ms at 768px — this is the architectural floor
- Tiling overhead is 120ms/frame (numpy blending + mx.eval per tile)

## Deep research available

Full 38-vector analysis at `research/Optimizing MLX Video Matting on Apple Silicon.md` — covers MLX/Metal tricks, tiling strategies, Hiera backbone, refiner, I/O, and unorthodox approaches. Each finding has feasibility, impact, risk, and complexity ratings.

## What NOT to try (proven dead ends from 57 prior experiments)

- Temporal anything (EMA, caching, blending, skip) — matting edges too volatile
- Backbone downscaling — edge degradation at all ratios
- Int8 quantization — 11% slower on Apple Silicon
- Batch B>1 — linear scaling, no amortization
- GPU stream parallelism — Apple Silicon = one GPU
- Token dedup/RLT — windowed attention, small global token counts
- GELU fast approx — 0ms difference
- GPU preprocessing for tiled path — adds copy overhead

See `research/OPTIMIZATION_SUMMARY.md` for the full 57-experiment breakdown.
