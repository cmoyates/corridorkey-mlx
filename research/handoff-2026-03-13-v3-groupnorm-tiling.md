# Handoff: V3 Adaptive Tile Skip + GroupNorm-Aware Tiling — 2026-03-13

## Where We Are

46 experiments. V3 tile skip logic is implemented and correct, but blocked by a GroupNorm tiling artifact that makes tile_size<image_size lossy. At the only lossless tile size (1024), the test video has 0% skip rate.

### V3 Results

| Config | Tile Size | Tiles | Skip Rate | Median (ms) | Fidelity |
|--------|-----------|-------|-----------|-------------|----------|
| V0 baseline @2048 | 1024 | 4 | — | 6041 | PASS |
| V3 thresh=0.02 @2048 | 1024 | 4 | 0% | 6047 | PASS |
| V3 thresh=0.01 @2048 | 512 | 16 | 32.4% | 5552 | FAIL |
| V3 thresh=0.02 @2048 | 512 | 16 | 32.8% | 5595 | FAIL |
| V3 thresh=0.05 @2048 | 512 | 16 | 33.1% | 5748 | FAIL |

### Root Cause of Fidelity Failure

**GroupNorm spatial statistics diverge when tiled.** The refiner has GroupNorm in every layer (stem + 4 ResBlocks = 9 GroupNorm layers). GroupNorm normalizes over (H, W) within each channel group. When the image is tiled:

- Full image: statistics computed over 2048x2048 = 4.2M pixels per group
- tile_size=1024: statistics over 1088x1088 = 1.2M pixels (with 32px overlap)
- tile_size=512: statistics over 576x576 = 331K pixels (with 32px overlap)

Smaller spatial extent → different mean/variance → different normalization → boundary artifacts. The 32px overlap handles the receptive field correctly, but GroupNorm's global spatial dependency is not addressed.

Evidence: all three V3 thresholds produce **identical** fidelity metrics despite different skip rates, proving the error is 100% from tiling, 0% from skip logic.

### What V3 Added (code changes in place)

- `GreenFormer.__init__`: `refiner_skip_confidence` param + `tile_skip_stats` property
- `_refiner_tiled`: confidence check per tile — if coarse alpha uniformly < thresh or > (1-thresh), emit zeros
- `VideoProcessor`: `tiles_skipped`/`tiles_total` in `FrameResult`
- `pipeline.py`: `refiner_skip_confidence` + `refiner_tile_size` params in `load_model`
- `bench_video.py`: `--tile-skip-threshold`, `--tile-skip-sweep`, `--tile-size` flags

All code is functional and tested (73/73 tests pass, pre-existing parity failure excluded).

## What To Do Next: GroupNorm-Aware Tiling

### The Fix

Precompute GroupNorm statistics on the full image, then tile with frozen stats. Two-pass approach:

#### Pass 1: Stats collection (full image, cheap)
- Run refiner stem + all GroupNorm layers in "stats-only" mode on the full image
- Collect per-group (mean, variance) at each GroupNorm layer
- This is cheap — GroupNorm stats are just mean/var reductions, no convolutions needed
- But the convolution outputs feed into GroupNorm, so we need the conv outputs too...

#### Alternative: Two-pass forward
1. **Pass 1 — full image forward, capture GroupNorm stats.** Run the full refiner forward on the complete image at reduced precision (or even downscaled) to collect the 9 sets of (mean, var) per GroupNorm layer. This is the same cost as one full refiner call.
2. **Pass 2 — tiled forward with frozen stats.** Run `_refiner_tiled` as today, but each GroupNorm uses the pre-collected full-image stats instead of computing per-tile stats.

#### Cleaner alternative: Frozen GroupNorm mode
- Add a `freeze_stats` mode to `RefinerBlock` / `CNNRefinerModule`
- When frozen, GroupNorm uses externally-provided (mean, var) instead of computing from input
- Stats pass: single full-image forward that records stats
- Tiled pass: tiles use recorded stats

This doubles the refiner calls (1 full + N tiled), so it's only worth it if tile skipping saves more than the stats pass costs. With 33% skip rate at tile_size=512, we run 1 + 0.67*16 ≈ 11.7 tile-equivalents vs 16 full tiles = 27% fewer tile-refiner calls, but we add 1 full-image stats pass.

**Net: stats pass ≈ 1 full refiner call. Tiled pass with skip = ~11 half-size calls ≈ 2.75 full calls. Total ≈ 3.75 full calls vs 4 full calls (tile_size=1024) = ~6% savings.** Marginal.

#### Better alternative: Spatial-mean GroupNorm
Instead of per-tile GroupNorm, compute statistics on a larger spatial context:
- Downscale the full image to compute GroupNorm statistics
- Apply those statistics during tiled processing
- Or: use InstanceNorm / LayerNorm which have different spatial properties

#### Simplest alternative: Overlapping statistics window
- Increase overlap from 32px (receptive field) to a much larger value (e.g., 256px)
- Larger overlap = more shared spatial context = stats closer to full-image
- Trade-off: less memory savings from tiling, but exact receptive field + better stats
- At 256px overlap on 512 tiles: effective tile = 1024px wide = same as current tile_size=1024

### Recommended Approach

**Frozen GroupNorm** is the cleanest and most general:

1. Add `CNNRefinerModule.collect_stats(rgb, coarse) -> list[tuple[mean, var]]` — full forward, captures GroupNorm stats at each layer, discards output
2. Add `CNNRefinerModule.set_frozen_stats(stats)` — subsequent forward calls use these stats
3. In `_refiner_tiled`: call `collect_stats` once on full image, then tile as normal with frozen stats
4. Tile skip still works — just skip tiles with confident alpha

**Implementation plan:**
1. Subclass or modify `nn.GroupNorm` to accept external (mean, var) — "frozen mode"
2. Add stats collection pass to `CNNRefinerModule`
3. Wire into `_refiner_tiled` — stats pass before tile loop
4. Benchmark: tile_size=512 + frozen stats + tile skip vs baseline

### Scope Check

- Files to modify: `refiner.py`, `corridorkey.py` (2-3 files, in scope)
- Risk: moderate — modifying GroupNorm behavior, but frozen stats are mathematically equivalent to full-image processing
- Validation: tile_size=512 with frozen stats should match tile_size=1024 output exactly

## Key Files

- Refiner module: `src/corridorkey_mlx/model/refiner.py`
- GreenFormer (tiled refiner): `src/corridorkey_mlx/model/corridorkey.py`
- Video processor: `src/corridorkey_mlx/inference/video.py`
- Pipeline: `src/corridorkey_mlx/inference/pipeline.py`
- Video benchmark: `scripts/bench_video.py` (now with `--tile-skip-threshold`, `--tile-skip-sweep`, `--tile-size`)
- Benchmark spec: `research/benchmark_spec.md`
- Previous handoff: `research/handoff-2026-03-13-post-v1v2.md`
- Experiments log: `research/experiments.jsonl` (46 entries)

## CLI Quick Reference

```bash
# V0 baseline at 2048 (generates reference)
uv run python scripts/bench_video.py --img-size 2048

# V3 tile skip sweep at 2048 (tile_size=1024, safe)
uv run python scripts/bench_video.py --img-size 2048 --async-decode --tile-skip-sweep 0.01 0.02 0.05 --no-save-reference

# V3 tile skip with smaller tiles (currently fails fidelity — needs frozen GN)
uv run python scripts/bench_video.py --img-size 2048 --tile-size 512 --async-decode --tile-skip-threshold 0.02 --no-save-reference

# V2 async baseline at 1024
uv run python scripts/bench_video.py --async-decode --no-save-reference
```

## Open Questions

- Can we avoid the full-image stats pass? (e.g., downscaled stats, or running stats from first tile)
- Should frozen GN be a separate `FrozenGroupNorm` class or a mode flag on existing GN?
- Does MLX `nn.GroupNorm` expose internals enough to inject external stats, or do we need a custom impl?
- tile_size=512 net benefit after stats pass overhead — is 6% enough to justify complexity?
- Could we use a larger overlap (256px) instead of frozen GN? Simpler but less memory-efficient
