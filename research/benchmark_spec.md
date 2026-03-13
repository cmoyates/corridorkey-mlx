# Benchmark Spec — corridorkey-mlx

## Optimization objectives

| Metric | Unit | Direction | Weight |
|--------|------|-----------|--------|
| Median latency (steady-state) | ms | lower is better | 0.6 |
| P95 latency (steady-state) | ms | lower is better | 0.1 |
| Peak memory | MB | lower is better | 0.3 |

## Regression gates (fidelity)

All fidelity checks are pass/fail. Any failure = hard reject.

### Tier 1 — Precision/numerical changes (FP16, BF16, quantization, compilation)

Max absolute error per tensor vs golden reference. Appropriate when the optimization
changes numerical precision but not the algorithm (same operations, different dtypes).

| Check | Threshold | Source |
|-------|-----------|--------|
| alpha_final max abs error vs golden | < 5e-3 | compare_reference.py |
| fg_final max abs error vs golden | < 5e-3 | compare_reference.py |
| alpha_coarse max abs error vs golden | < 5e-3 | compare_reference.py |
| fg_coarse max abs error vs golden | < 5e-3 | compare_reference.py |
| delta_logits max abs error vs golden | < 5e-3 | compare_reference.py |
| Output NaN/Inf check | none | smoke_2048.py |
| Alpha not all-zero or all-one | varies | smoke_2048.py |

### Tier 2 — Algorithmic/temporal changes (feature reuse, interpolation, EMA, skip)

Perceptual metrics for changes that alter the computation path. Max-abs breaks down
when errors concentrate at motion boundaries (e.g., skip2 had 0.996 max_abs but was
fine on static frames). Separate alpha/fg thresholds — alpha is 2-5x more sensitive.

| Check | Threshold | Source |
|-------|-----------|--------|
| alpha PSNR vs full-pipeline | > 35 dB | bench_video.py |
| fg PSNR vs full-pipeline | > 33 dB | bench_video.py |
| alpha SSIM vs full-pipeline | > 0.97 | bench_video.py |
| dtSSD (temporal coherence) | < 1.5 | bench_video.py |
| Output NaN/Inf check | none | smoke_2048.py |
| Alpha not all-zero or all-one | varies | smoke_2048.py |

**Calibration notes** (from deep research):
- PSNR > 35dB and SSIM > 0.95 are *baseline* indicators (not production grade)
- SOTA video matting models achieve dtSSD ~1.0-1.5 on HD datasets
- Alpha is 2-5x more sensitive than foreground — stricter alpha thresholds
- Metrics should be evaluated on semi-transparent boundary regions (not just solid core)
- PSNR/SSIM measured per-frame; dtSSD across consecutive frame pairs

## Dataset / fixture expectations

- Golden reference: `reference/fixtures/golden.npz` (512x512, seed=42, float32)
- Smoke inputs: `samples/sample.png` + `samples/hint.png` (or synthetic fallback)
- Checkpoint: `checkpoints/corridorkey_mlx.safetensors`

## Resolution buckets

| Bucket | Resolution | Purpose |
|--------|-----------|---------|
| dev | 256x256 | Fast iteration |
| standard | 512x512 | Primary benchmark + parity |
| production | 1024x1024 | Memory pressure test |
| native | 2048x2048 | Full production path |

## Measurement rules

### Cold-start vs steady-state
- Cold-start: First inference after model load (includes JIT, cache fill)
- Steady-state: Median of bench_runs after warmup_runs discarded
- Default: warmup_runs=3, bench_runs=10

### Latency
- Measured with time.perf_counter() bracketing model(x) + mx materialization
- Median of bench_runs = primary metric
- P95 = 95th percentile of bench_runs
- Min = best-case (useful for ceiling analysis)

### Peak memory
- Measured via mx.get_peak_memory()
- Reset before each measurement run
- Measured on a fresh model instance (not reused from latency benchmark)
- gc.collect() + mx.clear_cache() before reset

## Keep/revert rules

### KEEP when
- All fidelity gates pass AND
- (median_latency improved by >= 2% OR peak_memory improved by >= 5%)
- AND no metric regressed by more than the improvement margin on the other

### REVERT when
- Any fidelity gate fails (immediate, non-negotiable)
- Both latency and memory regressed

### INCONCLUSIVE when
- Fidelity passes but improvements are within noise (<2% latency, <5% memory)
- Re-run with more bench_runs before deciding

## Scoring formula

```
score = 0.6 * (baseline_latency / candidate_latency) +
        0.1 * (baseline_p95 / candidate_p95) +
        0.3 * (baseline_memory / candidate_memory)
```

Higher is better. Baseline score = 1.0.
Only computed for candidates that pass all fidelity gates.
