---
title: "Experiment 001: Tile Lifecycle Memory Discipline"
type: feat
status: completed
date: 2026-03-10
origin: docs/brainstorms/2026-03-08-mlx-memory-optimizations-brainstorm.md
---

# Experiment 001: Tile Lifecycle Memory Discipline

## Hypothesis

Explicitly deleting consumed tensors at two pipeline points reduces peak Metal memory without affecting fidelity or latency. MLX's lazy evaluator holds buffer references until Python refs drop — earlier deletion = earlier buffer reclamation.

## Problem

Two tensor groups outlive their usefulness:

1. **`features` list in `GreenFormer.__call__`** — 4 backbone feature maps (~13MB at 512×512) stay alive through the refiner, which never uses them. With `stage_gc=True`, these are already materialized Metal buffers. Deleting them before the refiner lets Metal reclaim ~13MB per tile.

2. **`w` / `w3d` in `tiling.py` tile loop** — numpy blend weights (~2KB per tile) not in the `del` statement. Negligible individually but shows discipline and prevents accumulation across many tiles.

## Changes

### `src/corridorkey_mlx/model/corridorkey.py`

After both decoders consume `features` (line 93), delete the list before upsampling:

```python
        alpha_logits = self.alpha_decoder(features)
        fg_logits = self.fg_decoder(features)

+       # Free backbone feature maps — decoders have consumed them,
+       # refiner uses rgb+coarse_pred only
+       del features

        # Upsample logits to full input resolution
        alpha_logits_up = self._logit_upsampler(alpha_logits)
```

**Why this is safe:**
- `features` is not referenced after line 93 in `__call__`
- Refiner inputs are `rgb` (sliced from `x`) + `coarse_pred` (from decoder outputs)
- `alpha_logits` / `fg_logits` are separate arrays — decoder already produced them
- With `stage_gc=True` (always on for tiled inference), features are materialized buffers, so `del` frees real Metal memory
- Without `stage_gc` (compiled path), features are graph nodes — `del` has no effect since downstream ops hold transitive refs. Harmless.
- Non-slim return dict uses `alpha_logits`/`fg_logits`, not `features`

### `src/corridorkey_mlx/inference/tiling.py`

Add `w, w3d` to the cleanup statement:

```python
-           del out, tile, alpha_tile, fg_tile
+           del out, tile, alpha_tile, fg_tile, w, w3d
            gc.collect()
            mx.clear_cache()
```

## Acceptance Criteria

- [x] Fidelity: all 7 tensors pass max_abs_error < 1e-3 vs golden.npz
- [x] Peak memory: ≥5% reduction vs baseline (or at minimum no regression)
- [x] Median latency: no regression beyond noise (≤2% slower)
- [x] All existing tests pass (`uv run pytest`)

## Benchmark Commands

```bash
# 1. Run experiment
uv run python scripts/run_research_experiment.py \
  --experiment-name "tile-lifecycle-memory-discipline" \
  --output research/artifacts/exp001_tile_lifecycle.json

# 2. Score (first run = becomes baseline)
uv run python scripts/score_experiment.py \
  --result research/artifacts/exp001_tile_lifecycle.json

# 3. Existing tests
uv run pytest
```

## Rollback

Revert two files:
```bash
git checkout -- src/corridorkey_mlx/model/corridorkey.py src/corridorkey_mlx/inference/tiling.py
```

## Expected Impact

| Metric | Expectation | Reasoning |
|--------|-------------|-----------|
| Peak memory | ~5-10% reduction at 512×512 | ~13MB freed per tile from features deletion; cumulative benefit at higher res |
| Latency | Neutral | `del` is O(1), `gc.collect` already runs |
| Fidelity | Identical | No computation changes, only reference cleanup |

## Important Notes

- This is the **first experiment** — no baseline exists yet. The runner will establish baseline.
- `stage_gc=True` is the default for tiled inference (`_compiled=False`), so `features` are materialized buffers that `del` can actually free.
- The benchmark measures full-frame 512×512, not tiled. Tiled inference memory benefit would be larger but is not directly captured by `run_research_experiment.py`.

## Sources

- **Origin brainstorm:** [docs/brainstorms/2026-03-08-mlx-memory-optimizations-brainstorm.md](docs/brainstorms/2026-03-08-mlx-memory-optimizations-brainstorm.md) — Step 3 (Deterministic GC Pipeline) identified per-tile cleanup as critical
- **Research program:** [research/program.md](research/program.md) — Priority #1: MLX tile lifecycle and memory discipline
- **Current tiling impl:** `src/corridorkey_mlx/inference/tiling.py:161-164`
- **Current forward pass:** `src/corridorkey_mlx/model/corridorkey.py:63-142`
