---
title: "Frozen GroupNorm for Correct Tiled Refiner Inference"
type: feat
date: 2026-03-13
---

# Frozen GroupNorm for Correct Tiled Refiner Inference

## Overview

V3 tile skip logic works but is blocked by a GroupNorm tiling artifact. When `tile_size < image_size`, each tile computes GroupNorm statistics over a smaller spatial extent, diverging from full-image statistics and causing boundary artifacts. At `tile_size=512` on 2048 input, all V3 configs fail fidelity regardless of skip threshold — proving the error is 100% from tiling, 0% from skip logic.

Fix: precompute GroupNorm (mean, var) on the full image, then tile with frozen stats.

## Problem Statement

The refiner has 9 GroupNorm layers (1 stem + 4 ResBlocks x 2). GroupNorm normalizes over `(H, W, group_size)` per group. When tiled:

- Full image 2048x2048: stats over 4.2M pixels/group
- tile_size=1024: stats over 1.2M pixels (32px overlap) — **passes fidelity**
- tile_size=512: stats over 331K pixels — **fails fidelity** (max_abs=0.706)

## Proposed Solution

**Two-pass frozen GroupNorm:**

1. **Stats pass**: Run refiner forward on full image, collect (mean, var) at each of 9 GroupNorm layers. Use block-level `mx.eval` to bound peak memory (~6GB per conv im2col, freed between blocks).
2. **Tiled pass**: Run `_refiner_tiled` as today, but each GroupNorm uses the pre-collected full-image stats instead of computing per-tile stats.

### Why this works

Frozen stats are mathematically equivalent to full-image processing. The GroupNorm output is `(x - mean) / sqrt(var + eps) * weight + bias` — if mean/var come from the full image, tiles produce identical output to a single full-image pass.

### Cost model

Stats pass = 1 full refiner call. At tile_size=512 on 2048 with 33% skip rate:
- Without frozen GN: 16 tiles (fails fidelity)
- With frozen GN: 1 stats pass + ~11 tiles = ~3.75 full-call equivalents vs 4 at tile_size=1024
- **Net: ~6% savings** (marginal but unblocks V3 tile skip at smaller tiles)

## Technical Approach

### Step 1: `FrozenGroupNorm` class — `refiner.py`

Drop-in replacement for `nn.GroupNorm`. Three modes:

| Mode | When | Behavior |
|------|------|----------|
| Normal | `_frozen_stats=None`, `_collecting=False` | Delegates to `mx.fast.layer_norm` (same perf as `nn.GroupNorm`) |
| Collecting | `_collecting=True` | Computes mean/var manually, saves to `_collected_stats`, returns normal output |
| Frozen | `_frozen_stats=(mean, var)` | Uses provided stats instead of computing from input |

Must match MLX's `pytorch_compatible=True` reshape pattern exactly:
```
(B, *spatial, C) -> (B, spatial_flat, G, gs) -> transpose(0,2,1,3) -> (B, G, spatial_flat*gs)
```
Mean/var shape: `(B, G, 1)` — independent of spatial dims, so frozen stats from full image apply to any tile size.

```python
# refiner.py — new class
class FrozenGroupNorm(nn.Module):
    """GroupNorm with frozen-stats support for tiled inference."""

    def __init__(self, num_groups: int, dims: int) -> None:
        super().__init__()
        self.num_groups = num_groups
        self.dims = dims
        self.eps = 1e-5
        self.weight = mx.ones((dims,))
        self.bias = mx.zeros((dims,))
        self._frozen_stats: tuple[mx.array, mx.array] | None = None
        self._collecting = False
        self._collected_stats: tuple[mx.array, mx.array] | None = None

    def __call__(self, x: mx.array) -> mx.array:
        if self._frozen_stats is None and not self._collecting:
            return self._fast_forward(x)  # mx.fast.layer_norm path
        return self._custom_forward(x)    # manual mean/var path

    def _fast_forward(self, x):
        """Normal path — identical to nn.GroupNorm(pytorch_compatible=True)."""
        batch, *rest, dims = x.shape
        group_size = dims // self.num_groups
        x = x.reshape(batch, -1, self.num_groups, group_size)
        x = x.transpose(0, 2, 1, 3).reshape(batch, self.num_groups, -1)
        x = mx.fast.layer_norm(x, eps=self.eps, weight=None, bias=None)
        x = x.reshape(batch, self.num_groups, -1, group_size)
        x = x.transpose(0, 2, 1, 3).reshape(batch, *rest, dims)
        return x * self.weight + self.bias

    def _custom_forward(self, x):
        """Collecting/frozen path — manual mean/var computation."""
        batch, *rest, dims = x.shape
        group_size = dims // self.num_groups
        x = x.reshape(batch, -1, self.num_groups, group_size)
        x = x.transpose(0, 2, 1, 3).reshape(batch, self.num_groups, -1)

        if self._frozen_stats is not None:
            mean, var = self._frozen_stats
        else:
            mean = x.mean(axis=-1, keepdims=True)
            var = x.var(axis=-1, keepdims=True)
            if self._collecting:
                self._collected_stats = (mean, var)

        x = (x - mean) * mx.rsqrt(var + self.eps)

        x = x.reshape(batch, self.num_groups, -1, group_size)
        x = x.transpose(0, 2, 1, 3).reshape(batch, *rest, dims)
        return x * self.weight + self.bias
```

### Step 2: Update `RefinerBlock` + `CNNRefinerModule` — `refiner.py`

- [x] Replace all 9 `nn.GroupNorm(REFINER_GROUPS, REFINER_CHANNELS, pytorch_compatible=True)` with `FrozenGroupNorm(REFINER_GROUPS, REFINER_CHANNELS)`
- [x] Add `_all_groupnorms()` helper to `CNNRefinerModule` — returns list of all 9 `FrozenGroupNorm` instances
- [x] Add `collect_groupnorm_stats(rgb, coarse)` — full-image forward with `_collecting=True` on all GNs, block-level `mx.eval` to bound peak memory:

```python
def collect_groupnorm_stats(self, rgb: mx.array, coarse_pred: mx.array) -> None:
    """Full-image forward to collect GroupNorm statistics.

    Block-level mx.eval bounds peak memory to ~6GB per conv im2col.
    """
    for gn in self._all_groupnorms():
        gn._collecting = True
        gn._collected_stats = None

    x = mx.concatenate([rgb, coarse_pred], axis=-1)
    x = nn.relu(self.stem_gn(self.stem_conv(x)))
    # mx.eval: MLX array materialization (not Python eval)
    mx.eval(x)  # noqa: S307 -- free im2col buffer
    x = self.res1(x)
    mx.eval(x)  # noqa: S307
    x = self.res2(x)
    mx.eval(x)  # noqa: S307
    x = self.res3(x)
    mx.eval(x)  # noqa: S307
    x = self.res4(x)
    mx.eval(x)  # noqa: S307
    del x  # discard output — only stats matter

    # Materialize collected stats
    all_arrays = []
    for gn in self._all_groupnorms():
        assert gn._collected_stats is not None
        all_arrays.extend(gn._collected_stats)
        gn._collecting = False
    # mx.eval: MLX array materialization (not Python eval)
    mx.eval(*all_arrays)  # noqa: S307
```

- [x] Add `freeze_groupnorm_stats()` — copies `_collected_stats` to `_frozen_stats` on each GN
- [x] Add `unfreeze_groupnorm_stats()` — clears `_frozen_stats` and `_collected_stats` on each GN

### Step 3: Wire into `_refiner_tiled` — `corridorkey.py`

- [x] Add `refiner_frozen_gn: bool = False` param to `GreenFormer.__init__`, store as `self._refiner_frozen_gn`
- [x] In `_refiner_tiled`: if `self._refiner_frozen_gn`:
  1. Call `self.refiner.collect_groupnorm_stats(rgb, coarse)` on full image
  2. Call `self.refiner.freeze_groupnorm_stats()`
  3. Force `refiner_fn = self.refiner` (uncompiled — compiled graph doesn't see frozen stats)
  4. Run existing tile loop
  5. Call `self.refiner.unfreeze_groupnorm_stats()` in finally block

### Step 4: Thread param through pipeline — `pipeline.py`

- [x] Add `refiner_frozen_gn: bool = False` to `load_model()` kwargs
- [x] Pass through to `GreenFormer(refiner_frozen_gn=refiner_frozen_gn, ...)`

### Step 5: Add CLI flag — `bench_video.py`

- [x] Add `--frozen-gn` flag (store_true, default False)
- [x] Pass through to `load_model(refiner_frozen_gn=args.frozen_gn)`

### Step 6: Tests

- [x] **FrozenGroupNorm parity test**: normal forward vs `nn.GroupNorm(pytorch_compatible=True)` on same input — should be exact match (0.0 diff)
- [x] **Frozen stats identity test**: collect stats on input X, freeze, forward on same X — output matches unfrozen forward exactly
- [x] **Frozen stats cross-size test**: collect stats on (B, 64, 64, 64), freeze, forward on (B, 32, 32, 64) — no crash, output is valid
- [x] **Run existing 73 tests** — verify no regression (FrozenGroupNorm is drop-in compatible)

### Step 7: Validate + benchmark

- [ ] Run: `uv run python scripts/bench_video.py --img-size 2048 --tile-size 512 --frozen-gn --async-decode --tile-skip-threshold 0.02 --no-save-reference`
- [ ] Compare fidelity metrics against tile_size=1024 baseline (should now PASS)
- [ ] Record as experiment 47 in `research/experiments.jsonl`
- [ ] Update handoff doc with results

## Acceptance Criteria

- [ ] FrozenGroupNorm produces identical output to `nn.GroupNorm(pytorch_compatible=True)` in normal mode
- [ ] tile_size=512 with frozen GN passes fidelity gate (max_abs < 5e-3 vs golden.npz)
- [ ] tile_size=512 with frozen GN matches tile_size=1024 output within tolerance
- [ ] All 73+ existing tests pass
- [ ] No regression in tile_size=1024 behavior (frozen GN only activates when flag is set)

## Dependencies & Risks

**Risks:**
- Stats pass peak memory (~6GB per conv im2col) may be tight on 16GB machines at 2048 — but 2048 resolution likely requires >=32GB anyway
- Manual mean/var computation in frozen path may be slightly slower than `mx.fast.layer_norm` fused kernel — acceptable since it only runs during tiled inference
- Compiled refiner is bypassed when frozen GN is active — perf impact mitigated by per-tile `mx.eval`

**Dependencies:**
- V3 tile skip code already in place (experiment 43-45)
- No external dependencies — pure MLX operations

## Open Questions

- Stats pass at 2048 peak mem acceptable on target hardware?
- Could downscaled stats pass (run at 1024, apply at 2048) be "good enough" approximation?
- Worth auto-enabling frozen GN when tile_size < some threshold vs always opt-in?

## References

- Handoff doc: `research/handoff-2026-03-13-v3-groupnorm-tiling.md`
- GroupNorm bottleneck analysis: `research/compound/2026-03-12-groupnorm-handoff.md`
- MLX GroupNorm source: uses `mx.fast.layer_norm` on `(B, G, spatial*gs)` for `pytorch_compatible=True`
- Refiner module: `src/corridorkey_mlx/model/refiner.py`
- GreenFormer tiled refiner: `src/corridorkey_mlx/model/corridorkey.py:406-491`
- Pipeline param threading: `src/corridorkey_mlx/inference/pipeline.py`
- Buffer limits learning: `research/compound/exp029_buffer_limits_memory_reduction.md`
