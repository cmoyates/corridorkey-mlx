# Handoff: Final Optimization Session — Hail Mary Pass

## Mission

This is the last optimization session. Everything obvious has been tried. 79 experiments across 6 phases have been run. 22 issues were triaged and closed. The async pipeline hides all I/O. The only thing left that moves the needle is **reducing per-tile inference time** (currently 218ms/tile × 6 tiles = 1308ms + 120ms overhead = 1429ms/frame wall time).

**Your job**: run a deep research pass looking for weird, stupid, unconventional, bottom-of-the-barrel ideas that might shave time off per-tile inference. Things nobody would normally try. Abuse MLX internals, exploit Apple Silicon quirks, find framework bugs that happen to be faster, whatever. If it passes fidelity and saves >10ms/tile, ship it. If it doesn't, log it and move on.

**Goal**: any measurable wall-time improvement on the 37-frame benchmark (currently 0:54).

## Current performance

**Torch 3:34 → MLX 0:54 = 3.96× faster** (37 frames @ 1920×1080, 768px tiles)

### Per-tile breakdown at 768px (estimated from 512px profile, scaled)

| Component | ~ms/tile | % | Notes |
|---|---|---|---|
| Backbone S0 (2 blocks, windowed, dim=112) | ~12 | 5% | FP32 required (dim=112 precision-sensitive) |
| Backbone S1 (3 blocks, windowed, dim=224) | ~13 | 6% | |
| **Backbone S2 (16 blocks, global, dim=448)** | **~60** | **27%** | Dominant. 4× spatial tokens vs 512px. |
| Backbone S3 (3 blocks, global, dim=896) | ~10 | 5% | |
| Decoders (alpha + fg, dual-stream) | ~13 | 6% | BF16 weights, fused upsample |
| **Refiner (9 GN, 4 dilated ResBlocks)** | **~75** | **35%** | Full-res. Dilated convs = scattered memory access. |
| Per-tile overhead (eval, slice, pad) | ~35 | 16% | mx.eval sync, numpy convert |
| **Total** | **~218** | **100%** | |

### Architecture constraints (non-negotiable)
- Backbone: Hiera-Base-Plus, 24 blocks, pretrained weights, no retraining
- Decoder: SegFormer dual-head, BF16
- Refiner: 4 dilated ResBlocks + 9 GroupNorm, FP32 (FP16 tested, no difference)
- Input: 4ch NHWC (RGB + alpha hint), ImageNet-normalized
- Tiling: 768px tiles, 128px overlap, frozen GN stats
- Per-component mx.compile active (backbone, decoders, refiner separately)
- MLX buffer env: MLX_MAX_MB_PER_BUFFER=2, MLX_MAX_OPS_PER_BUFFER=2

## What's been tried and failed (79 experiments)

### Temporal / cross-frame (all fail fidelity)
- Output EMA blending (all α values)
- Feature-space EMA (decoder features)
- S2/S3 feature caching across frames (247/255 max error)
- Backbone skip (every other frame) — visible motion artifacts
- Skip2 with interpolation — still artifacts

### Precision / quantization
- Int8 quantization — 11% SLOWER on Apple Silicon (dequant overhead)
- BF16 backbone stage 0 — fidelity regression (dim=112 sensitive)
- Refiner FP16 toggle — no difference
- GELU fast approx — 0ms difference

### Architecture / compute reduction
- Backbone resolution decoupling — edge degradation at all ratios (V7)
- Sub-resolution refiner — same edge issue
- Token dedup/RLT — windowed attention, small global token counts
- GEMM pad stage 0 K=112→128 — regression
- Batch B>1 — linear scaling, no amortization
- Enforced windowed attention in S2 — needs retraining

### Tiling / scheduling
- Overlap 128→64 — no difference at 768px
- Overlap 0 — same tile count (last tile shifts back)
- 1024px tiles — regressed (memory pressure)
- Whole-forward mx.compile — no benefit over per-component
- mx.compile shapeless=True — unsafe (Hiera shape-dependent reshapes)
- GPU preprocessing for tiled path — +4s slower
- mx.vmap for tiles — not tested but B>1 showed linear scaling
- Deferred eval across tiles — memory explosion risk

### I/O / pipeline
- Write I/O optimization — already fully overlapped by async pipeline
- GPU postprocessing — overlapped in writer thread
- HW video decode — read is 0.2% of time
- Zero-copy numpy views — copy unavoidable at GPU→CPU boundary
- MLX scatter_add blending — copy-on-write, no gain

### Metal / low-level
- Custom Metal GroupNorm — -67% micro but 0% pipeline impact
- mx.fast.metal_kernel() lacks shared memory / barriers for complex kernels
- GPU stream parallelism — Apple Silicon = one GPU
- Wired memory limit — no benefit on unified memory

## What to research

Think laterally. Here are seed directions — but don't limit yourself to these:

1. **MLX graph-level tricks**: Are there undocumented mx.compile flags, trace modes, or eval strategies? Can we hint the scheduler? Can we abuse `mx.disable_compile` selectively to avoid recompilation overhead on certain paths?

2. **Metal shader cache / PSO warming**: Does MLX cache Pipeline State Objects between runs? Can we pre-warm the shader cache on model load to eliminate first-tile compilation stalls?

3. **Memory layout tricks**: NHWC is MLX standard but are there operations where a temporary transpose to NCHW would give better Metal kernel tiling? Especially for the dilated convolutions in the refiner.

4. **Attention implementation**: The backbone uses `mx.fast.scaled_dot_product_attention` — are there faster paths for the specific head counts / dims in Hiera? Custom attention kernels?

5. **Refiner conv tricks**: The dilated convolutions are the biggest single cost. Can we replace dilation with strided conv + upsample (same receptive field, contiguous memory)? This doesn't need retraining if the weight mapping is exact.

6. **mx.eval granularity**: Current code evals per-tile. What if we eval per-component-per-tile (backbone, decoder, refiner separately within each tile)? Finer-grained eval might allow better buffer reuse.

7. **Compile boundaries**: Currently backbone/decoder/refiner are compiled separately. What about compiling backbone+decoder as one unit? The decoder is cheap but the compile boundary forces a sync.

8. **Apple Silicon memory bandwidth**: Are we bandwidth-bound or compute-bound per component? If bandwidth-bound, can we reduce data movement (smaller intermediates, in-place ops)?

9. **MLX version upgrades**: Has MLX gotten faster in recent versions? Are there new primitives (fused attention, better compile, new Metal kernels) we're not using?

10. **Numerical tricks**: Can we use lower precision for intermediate computations within compiled regions while keeping FP32 at boundaries? Mixed BF16/FP32 within a single compiled block?

## How to work

1. **Research first** — use deep research / web search to find MLX tricks, Apple Silicon optimization papers, Metal best practices
2. **Prototype fast** — make the change, run `uv run pytest tests/test_tiling.py -x -q` for correctness
3. **Benchmark** — install into CorridorKey venv and run the 37-frame benchmark
4. **Keep or revert** — if it saves wall time and passes fidelity, commit. Otherwise revert and log.
5. **Log everything** — even failures are valuable. Comment on relevant GitHub issues or create new ones.

## How to benchmark

```bash
# Install into CorridorKey venv
VIRTUAL_ENV=../CorridorKey/.venv uv pip install --reinstall --python ../CorridorKey/.venv/bin/python "corridorkey-mlx @ git+file:///Users/cristopheryates/Documents/Projects/Python/corridorkey-mlx"

# Run benchmark
../CorridorKey/.venv/bin/python ../CorridorKey/corridorkey_cli.py run-inference --backend mlx --linear --despill 5 --despeckle --refiner 1.0
```

**Baseline**: 0:54 (37 frames @ 1920×1080, median infer 1433ms/frame)

## Fidelity gates

| Change type | Threshold |
|---|---|
| Precision changes | alpha_final, fg_final max_abs < 5e-3 vs golden |
| Pipeline-only changes | Byte-identical output files |
| Algorithmic changes | Alpha PSNR >35dB, SSIM >0.97, dtSSD <1.5 |

```bash
uv run python scripts/compare_reference.py --img-size 512
```

## Key files

| File | What |
|---|---|
| `src/corridorkey_mlx/model/corridorkey.py` | GreenFormer — backbone + decoder + refiner assembly |
| `src/corridorkey_mlx/model/hiera.py` | Hiera backbone (24 blocks, MaskUnitAttention) |
| `src/corridorkey_mlx/model/decoder.py` | SegFormer dual-head decoder |
| `src/corridorkey_mlx/model/refiner.py` | Dilated ResBlock refiner + GroupNorm |
| `src/corridorkey_mlx/inference/tiling.py` | Tiled inference + single-tile shortcut |
| `src/corridorkey_mlx/inference/pipeline.py` | Model loading + compile config |
| `src/corridorkey_mlx/engine.py` | Engine API (process_frame) |
| `research/OPTIMIZATION_SUMMARY.md` | Full 57-experiment log |
| `research/Optimizing MLX Video Matting on Apple Silicon.md` | 38-vector deep research analysis |

## Rules

- **Fidelity is a gate, not a target** — any candidate failing thresholds is rejected
- **Don't try things from the "failed" list above** — they're proven dead ends
- **Measure wall time, not component time** — only the benchmark number matters
- **Ship small** — one optimization per commit, revertable
- **Log failures** — a documented dead end is still progress
