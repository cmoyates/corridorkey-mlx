# Pipeline Bottleneck Analysis — Real Frame Profiling

## Finding
Micro-benchmarks don't predict real pipeline impact. The dominant cost is tile count × backbone, not any single component.

## Evidence
- Metal GroupNorm v2: -67% micro-bench, 0% pipeline impact (2:06 → 2:06)
- 1920x1080 with tile_size=512, overlap=128 → 15 tiles/frame
- 227ms/tile total: ~117ms model inference + ~110ms overhead (I/O, blending, Python)
- GroupNorm saves ~5ms/tile = 75ms/frame out of 3400ms → ~2%
- Torch baseline: 3:34, MLX: 2:06 → 1.7x faster

## Implication
To move the needle past 2:06, must reduce either:
1. **Tile count** — larger tiles (needs more memory), or smarter tiling
2. **Per-tile backbone cost** — feature caching (V5), token reduction (V9/RLT)
3. **Pipeline overhead** — the ~110ms/tile non-inference overhead is 48% of per-tile time

Component-level optimizations (<5ms/tile) are in the noise.
