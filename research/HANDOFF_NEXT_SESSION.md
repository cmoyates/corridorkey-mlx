# Handoff: Next Optimization Session

## Where we are

**Torch 3:34 → MLX 0:53 = 3.96× faster** (37 frames @ 1920×1080)

### Corrected per-frame breakdown (768px tiles, 6 tiles/frame)

| Phase | ms/frame (measured) | Wall-time impact | Notes |
|---|---|---|---|
| Read | 4.7 | 0ms | Async reader thread, fully overlapped |
| **Infer** | **1429** | **1429ms** | **THE bottleneck — 218ms/tile × 6 + 120ms overhead** |
| Postprocess | 89.8 | 0ms | Runs in writer thread, overlapped with next infer |
| Write | 604.6 | 0ms | Async writer thread, fully overlapped (605 < 1429) |
| **Wall time** | — | **~1429ms** | **37 × 1.429 = 52.9s ≈ actual 53s** |

**Key insight**: The timing summary in clip_manager sums per-phase times, but read/postprocess/write all run in parallel threads. Only inference determines wall time.

### What changed this session
- **Dynamic single-tile inference (#37)**: if subject bbox fits in one 768px tile, runs 1 tile instead of 6 → ~5x faster on applicable frames
- Tile size 512→768: **2:04 → 0:53** (previous session)
- Frozen GN enabled by default (previous session)
- service.py: added fast_exr support to OutputConfig + _write_image (consistent with clip_manager)

### Issue board: 22 → 1 open
All issues triaged and resolved except #46 (Apple Instruments profiling — requires GUI).

## What was resolved

### Implemented
| # | Issue | Result |
|---|---|---|
| **37** | Dynamic single-tile inference | **Implemented** — _find_subject_bbox + _single_tile_inference in tiling.py |

### Already done (discovered during triage)
| # | Issue | Finding |
|---|---|---|
| 27 | Async I/O pipeline | Already in clip_manager.py (reader + writer threads, queue depth 2) |
| 25 | Configurable outputs | Already in upstream (enabled_outputs, --outputs CLI flag) |
| 34, 33, 35 | Write I/O optimization | **Moot** — writes fully overlap with inference (605ms < 1429ms) |

### Closed as not actionable
| # | Issue | Why |
|---|---|---|
| 32 | MLX scatter_add blending | numpy blending is only ~24ms; mx.eval sync is the real cost |
| 36 | Overlap-free tiling | Same tile count with overlap=0; prior test showed no difference |
| 39 | Token pruning | Hiera uses windowed attention, not global O(N²) |
| 48 | mx.compile shapeless | Per-component compile already optimal |
| 47 | Adaptive tile placement | Duplicate of #37 |
| 42 | Speculative pipelining | Refiner needs frame N's RGB — sequential dependency |
| 40 | Sub-res refiner | V7 test rejected — edge degradation at all ratios |
| 41 | Stage 2 weight similarity | V5 feature caching catastrophic (247/255 error) |
| 45 | Zero-copy numpy views | Copy unavoidable at GPU→CPU boundary |
| 44 | GGML quantization | Int8 already 11% slower; Q4 worse |
| 43 | CoreML/ANE hybrid | MaskUnitAttention not convertible to CoreML |
| 38 | Profile Stage 2 | No actionable optimization without retraining |
| 31 | Numba JIT postprocess | Already vectorized numpy, <2ms |
| 28 | HW video decode | Read is 0.2% of time (4.7ms) |
| 30 | Content-adaptive tile skip | Already exists (alpha hint check) + #37 |
| 24 | GPU despill/despeckle | Runs in writer thread, fully overlapped |

## Current state

### Bottleneck: inference only
The ONLY path to faster wall time is reducing per-tile inference cost (currently 218ms/tile at 768px). Everything else is overlapped by the async pipeline.

### Remaining optimization surface
1. **Reduce tile count** — dynamic single-tile (#37) helps when subject is small
2. **Reduce per-tile cost** — model architecture limits (218ms = backbone + decoder + refiner)
3. **Only #46 remains open** — Apple Instruments Metal System Trace profiling (manual)

### What NOT to try (expanded from 57 → 79 experiments)
All items from previous handoff plus:
- Write I/O optimization — writes already fully overlapped with inference
- MLX scatter_add — copy-on-write semantics, no gain over numpy
- Overlap-free tiling — same tile count, no compute savings
- Token pruning — Hiera windowed attention, not applicable
- Frame pipelining — refiner sequential dependency on RGB
- GPU postprocessing — already overlapped in writer thread

## Current branch
`feat/misc-optimizations` on `cmoyates/corridorkey-mlx`, branched from main.

## Files modified this session
- `src/corridorkey_mlx/inference/tiling.py` — dynamic single-tile inference (bbox analysis + single-tile path)
- `tests/test_tiling.py` — 5 new tests for bbox + single-tile

## How to benchmark

```bash
# Install latest corridorkey-mlx into CorridorKey venv
VIRTUAL_ENV=../CorridorKey/.venv uv pip install --reinstall --python ../CorridorKey/.venv/bin/python "corridorkey-mlx @ git+file:///path/to/corridorkey-mlx"

# Run benchmark (shows per-phase timing)
../CorridorKey/.venv/bin/python ../CorridorKey/corridorkey_cli.py run-inference --backend mlx --linear --despill 5 --despeckle --refiner 1.0

# With fast EXR (uncompressed, ~10x faster writes — though writes are already overlapped)
../CorridorKey/.venv/bin/python ../CorridorKey/corridorkey_cli.py run-inference --backend mlx --linear --despill 5 --despeckle --refiner 1.0 --fast-exr
```

**Test clip**: `ClipsForInference/BetterGreenScreenTest_BASE/` — 37 frames @ 1920×1080.

## Fidelity gates

| Change type | Threshold |
|---|---|
| Precision changes | alpha_final, fg_final max_abs < 5e-3 vs golden |
| Pipeline-only changes | Byte-identical output files |
| Algorithmic changes | Alpha PSNR >35dB, SSIM >0.97, dtSSD <1.5 |

## Deep research available

Full 38-vector analysis at `research/Optimizing MLX Video Matting on Apple Silicon.md`.
