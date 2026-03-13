# Handoff: V4 Frozen GroupNorm Results — 2026-03-13

## TL;DR

Frozen GroupNorm works — **eliminates GroupNorm tiling artifacts at any tile size** (max_abs=0.0 between tile_size=512 and tile_size=1024 when both use frozen stats). But it's slower, not faster. The stats pass costs more than tile skip saves.

## Benchmark Results (37 frames, 2048x2048)

V0 reference: unfrozen, tile_size=1024, no async.

| Config | Tile | Skip | Median (ms) | FPS | Fidelity vs V0 | Notes |
|--------|------|------|-------------|-----|----------------|-------|
| V0 baseline (unfrozen) | 1024 | — | 5799 | 0.17 | PASS (0.0) | reference |
| Frozen GN, no skip | 1024 | — | 7115 | 0.14 | α=0.314, fg=0.608 | stats pass overhead |
| Frozen GN, no skip | 512 | — | 7478 | 0.13 | α=0.314, fg=0.608 | more tiles + stats |
| Frozen GN + tileskip=0.02 | 512 | 33% | 7237 | 0.14 | α=0.012, fg=0.996 | fg skip artifact |
| Frozen GN 512 vs Frozen GN 1024 | — | — | — | — | **0.0** | tile size irrelevant |

### Key observations

1. **Frozen GN makes tile size irrelevant for GroupNorm** — frozen-512 and frozen-1024 produce bit-identical output (max_abs=0.0).

2. **Frozen ≠ unfrozen** — frozen GN (full-image stats) differs from unfrozen (per-tile stats) by α_max_abs=0.314. This is expected: different normalization statistics produce different output. Neither is "wrong" — frozen is mathematically closer to a single untiled pass.

3. **Stats pass kills the speed gain** — frozen GN adds ~1300ms/frame (one full refiner forward for stats collection). At tile_size=512 with 33% tile skip, we save ~3 tiles but add the stats pass. Net result: **slower than unfrozen tile_size=1024**.

4. **Tile skip has an fg artifact** — zeroing the refiner delta in confident-alpha tiles is correct for alpha (max_abs=0.012) but wrong for foreground (max_abs=0.996). The refiner's fg correction is non-zero even in confident-alpha regions. This is a V3 issue, not a frozen GN issue.

## What frozen GN gets us

### Pros
- Correctness: tile size no longer affects GroupNorm behavior
- Enables arbitrarily small tiles without normalization artifacts
- Could reduce peak per-tile memory on memory-constrained devices

### Cons
- **~22% slower** (stats pass overhead: 1 full refiner forward)
- No FPS improvement at 2048 vs unfrozen tile_size=1024
- Tile skip still broken for fg channel (separate issue)
- Changes output vs unfrozen reference (α=0.314 divergence)

## Cost model

At 2048x2048 with tile_size=512:
- Stats pass: ~1 full refiner call = ~1300ms
- 16 tiles × 67% (after skip) = ~11 tile calls
- Each 512 tile ≈ 1/16 of a full call → ~0.7 full equivalents
- Total: 1 + 0.7 = **1.7 full refiner equivalents**

Compare unfrozen tile_size=1024:
- 4 tiles, each ≈ 1/4 of full call = **1.0 full refiner equivalent**

Frozen GN is 1.7x the refiner cost for zero fidelity benefit at 1024.

## Verdict

**Frozen GN is a correct but unprofitable optimization at 2048.** The stats pass overhead exceeds any tile skip savings. It would only be valuable if:

1. Peak per-tile memory matters (e.g., 8GB device needing tile_size=256)
2. The tile skip fg artifact is fixed (so tile skip actually improves throughput)
3. The workload has higher skip rates than this test video (33%)

## What's still broken

1. **V3 tile skip fg artifact** — refiner fg delta is non-zero in confident-alpha regions. Fix options:
   - Skip only alpha channel delta, always compute fg delta
   - Use a per-channel confidence threshold
   - Accept fg error in background regions (may be invisible after compositing)

2. **Frozen GN vs unfrozen divergence** — α_max_abs=0.314 is non-trivial. If frozen GN becomes default, the V0 reference needs regeneration with frozen stats.

## Files changed

- `src/corridorkey_mlx/model/refiner.py` — `FrozenGroupNorm` class, stats API
- `src/corridorkey_mlx/model/corridorkey.py` — `refiner_frozen_gn` param, wiring in `_refiner_tiled`
- `src/corridorkey_mlx/inference/pipeline.py` — `refiner_frozen_gn` in `load_model()`
- `scripts/bench_video.py` — `--frozen-gn` flag
- `tests/test_frozen_groupnorm.py` — 7 tests (all pass)

## CLI

```bash
# Frozen GN at 512, no skip (correct, slower)
uv run python scripts/bench_video.py --img-size 2048 --tile-size 512 --frozen-gn --no-save-reference

# Frozen GN at 512 + tile skip (fg artifact)
uv run python scripts/bench_video.py --img-size 2048 --tile-size 512 --frozen-gn --tile-skip-threshold 0.02 --no-save-reference

# Unfrozen baseline (fastest, current best)
uv run python scripts/bench_video.py --img-size 2048
```
