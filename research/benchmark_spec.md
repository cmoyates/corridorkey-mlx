# Benchmark Spec — corridorkey-mlx

## Optimization objectives

| Metric | Unit | Direction | Weight |
|--------|------|-----------|--------|
| Median latency (steady-state) | ms | lower is better | 0.6 |
| P95 latency (steady-state) | ms | lower is better | 0.1 |
| Peak memory | MB | lower is better | 0.3 |

## Regression gates (fidelity)

All fidelity checks are pass/fail. Any failure = hard reject.

| Check | Threshold | Source |
|-------|-----------|--------|
| alpha_final max abs error vs golden | < 5e-3 | compare_reference.py |
| fg_final max abs error vs golden | < 5e-3 | compare_reference.py |
| alpha_coarse max abs error vs golden | < 5e-3 | compare_reference.py |
| fg_coarse max abs error vs golden | < 5e-3 | compare_reference.py |
| delta_logits max abs error vs golden | < 5e-3 | compare_reference.py |
| Output NaN/Inf check | none | smoke_2048.py |
| Alpha not all-zero or all-one | varies | smoke_2048.py |

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
- Measured via mx.metal.get_peak_memory() (or mx.get_peak_memory())
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
