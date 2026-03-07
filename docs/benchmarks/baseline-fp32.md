# Baseline Benchmarks (FP32, no optimizations)

Date: 2026-03-07
Hardware: Apple M4 Max (128GB unified)
MLX version: 0.24.2
Weights: corridorkey_mlx.safetensors (real checkpoint)
Config: warmup=2, bench=3, batch=1, eager+compiled (isolated per resolution)
NOTE: Captured under load — latency numbers not representative. Re-run on idle system. Memory numbers should be stable.

## Results

| Resolution | Eager Steady (ms) | Compiled Steady (ms) | Speedup | Peak MB  | Active MB |
|------------|-------------------:|---------------------:|--------:|---------:|----------:|
| 512x512    |              219.7 |                173.8 |   1.26x |   2554.4 |     555.1 |
| 1024x1024  |              865.0 |                793.7 |   1.09x |   3673.1 |     609.1 |
| 2048x2048  |            10434.0 |               7076.7 |   1.47x |  26689.0 |     825.1 |

## Key Observations

- Peak memory at 2048: ~26.7 GB — well above the <6 GB target
- Active memory stays modest (~825 MB at 2048) — peak dominated by intermediate computation graphs
- Compiled speedup: 1.1-1.5x across resolutions
- 26.7 GB peak at 2048 suggests massive intermediate tensor allocation during Hiera backbone
- Active memory (what's alive after forward pass) is only ~825 MB — most peak usage is transient

## Targets (from plan)

| Metric | Current | Target |
|--------|--------:|-------:|
| Peak memory at 2048 (non-tiled) | 26689 MB | <6000 MB |
| Throughput at 2048 (compiled) | 7077 ms | ~3500-4700 ms (1.5-2x) |
