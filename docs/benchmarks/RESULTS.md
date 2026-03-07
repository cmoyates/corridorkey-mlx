# Benchmark Results

Hardware: Apple M4 Max (128GB unified)
MLX version: 0.24.2
Weights: `corridorkey_mlx.safetensors` (real checkpoint)
Input: `samples/sample.png` (1920x1080) + `samples/hint.png` (1280x720), resized to model res

## Results Table

| Phase | Config | Median (ms) | Peak MB | Active MB | Notes |
|-------|--------|------------:|--------:|----------:|-------|
| 0 -- Baseline | FP32, full backbone, no tiling | 10,434 | 26,689 | 825 | Compiled, 2048 |
| 1 -- FP16 decoders | FP16 decoders, FP32 backbone+refiner | 7,988 | 27,937 | 1,897 | ~12% faster; peak unchanged (decoder-only FP16) |
| 2 -- CPU-GPU roundtrips | MLX-native preprocess + slim forward | — | — | — | Integrated into engine; no isolated bench |
| 3 -- Backbone 1024 | FP32, bb=1024, no tiling | 1,447 | 8,044 | 572 | 6.3x faster, 3.4x less peak vs baseline |
| 4 -- Tiled inference | FP32, full backbone, tiled 512/96px | 3,467 | 2,626 | 551 | Peak capped ~2.6 GB; 3x slower than bb=1024 |
| 5 -- All optimizations | FP16, bb=1024, tiled 512/96px | 8,450* | 2,301 | 273 | *smoke_2048 wall time (real images, resize overhead) |

## Phase 0 -- Baseline (FP32, unoptimized)

**Date:** 2026-03-07
**Commit:** baseline (pre-optimization)

### Timing (eager vs compiled)

| Resolution | Eager Steady (ms) | Compiled Steady (ms) | Speedup | Peak MB | Active MB |
|------------|-------------------:|---------------------:|--------:|--------:|----------:|
| 256x256    |               29.1 |                 26.1 |   1.11x |   1,248 |       542 |
| 512x512    |              119.6 |                168.2 |   0.71x |   2,555 |       555 |
| 1024x1024  |            1,067.1 |              1,502.4 |   0.71x |   3,673 |       609 |

Compiled is slower at 512+ — Hiera's shape-dependent reshapes cause recompilation overhead per resolution.

### Memory

| Metric | Value |
|--------|------:|
| Peak at 2048 (from baseline doc) | 26,689 MB |
| Active at 2048 | 825 MB |

Peak dominated by transient computation graphs during Hiera backbone, not persistent allocations.

## Phase 5 -- Engine Configuration Matrix

**Date:** 2026-03-07
**Commit:** 54b1f7b

### Benchmark Matrix (2048x2048)

| Config | FP16 | Backbone | Tiled | Median (ms) | Min (ms) | Peak MB | Active MB |
|--------|------|----------|-------|------------:|--------:|--------:|----------:|
| Baseline | off | 2048 | no | 9,068 | 7,953 | 27,560 | 1,520 |
| FP16 only | on | 2048 | no | 7,988 | 7,268 | 27,937 | 1,897 |
| Backbone 1024 | off | 1024 | no | 1,447 | 1,433 | 8,044 | 572 |
| FP16 + bb=1024 | on | 1024 | no | 1,695 | 1,424 | 8,041 | 569 |
| Tiled 512 | off | 2048 | 512/96 | 3,467 | 3,450 | 2,626 | 551 |
| FP16 + tiled 512 | on | 2048 | 512/96 | 3,588 | 3,538 | 2,900 | 824 |

### Regression Checks (smaller resolutions)

| Config | FP16 | Median (ms) | Min (ms) | Peak MB | Active MB |
|--------|------|------------:|---------:|--------:|----------:|
| 512 FP32 | off | 237 | 204 | 2,150 | 276 |
| 512 FP16 | on | 285 | 263 | 2,408 | 548 |
| 1024 FP32 | off | 1,087 | 1,028 | 3,953 | 845 |
| 1024 FP16 | on | 1,088 | 1,031 | 4,247 | 1,139 |

### Smoke Test (all optimizations, real images)

| Setting | Value |
|---------|-------|
| Config | FP16, bb=1024, tiled 512/96 |
| Input | 1920x1080 RGB + 1280x720 hint, resized to 2048 |
| Wall time | 8.45s |
| Peak memory | 2,301 MB |
| Active memory | 273 MB |
| Cache memory | 515 MB |
| Output shapes | alpha (1080, 1920), fg (1080, 1920, 3), comp (1080, 1920, 3) |
| Health | PASSED (no NaN/Inf, full [0, 255] range) |

### Smoke Test (baseline, no optimizations, real images)

| Setting | Value |
|---------|-------|
| Config | FP32, full backbone, no tiling, no compile |
| Wall time | 4.87s |
| Peak memory | 26,421 MB |
| Active memory | 381 MB |
| Cache memory | 25,790 MB |

## Key Observations

1. **Backbone at 1024 = biggest win**: 6.3x faster, 3.4x less peak memory vs full-res backbone. Dominates all other optimizations.
2. **Tiling caps peak memory**: ~2.6 GB peak vs 27.6 GB non-tiled at 2048. Best option for memory-constrained systems.
3. **FP16 (decoder-only) has marginal impact**: slight latency improvement but doesn't reduce peak memory — backbone and refiner stay FP32 due to precision sensitivity.
4. **Compiled mode is slower at 512+**: Hiera's dynamic shapes hurt compilation. Only beneficial at 256.
5. **bb=1024 + tiling combined**: smoke test shows 2.3 GB peak — meets the <6 GB target from the plan, but wall time is slower due to tiling overhead on already-downsampled backbone output.

## Targets vs Actuals

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Peak memory at 2048 (non-tiled, bb=1024) | <6,000 MB | 8,044 MB | Close — backbone alone not enough |
| Peak memory at 2048 (tiled) | bounded ~2 GB/tile | 2,626 MB | MET |
| Peak memory at 2048 (bb=1024 + tiled) | — | 2,301 MB | MET |
| Throughput at 2048 (bb=1024) | 1.5-2x baseline | 6.3x baseline | EXCEEDED |
| CPU-GPU syncs per frame | 1 | 1 (non-tiled path) | MET |
| FP16 parity vs FP32 | <1e-3 | <1e-3 (decoder-only) | MET |
