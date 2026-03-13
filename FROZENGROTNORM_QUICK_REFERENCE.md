# FrozenGroupNorm Implementation — Quick Reference

## Critical Code Locations

### 1. Refiner GroupNorm (9 instances, all NHWC)
**File**: `src/corridorkey_mlx/model/refiner.py`

- **Line 31, 39**: RefinerBlock gn1, gn2 (2 per block × 4 blocks = 8 GN)
- **Line 61**: CNNRefinerModule stem_gn (1 GN)
- **Line 83-102**: CNNRefinerModule.__call__() forward pass

All use `nn.GroupNorm(REFINER_GROUPS, channels, pytorch_compatible=True)`
- REFINER_GROUPS = 8
- REFINER_CHANNELS = 64
- REFINER_SCALE = 10.0 (final output scaling)

### 2. Tiling Implementation
**File**: `src/corridorkey_mlx/model/corridorkey.py`

- **Line 47-48, 65**: GreenFormer.__init__ params: refiner_tile_size, refiner_skip_confidence
- **Line 164-168**: Forward method checks tiling conditions
- **Line 280-500** (approx): _refiner_tiled() method — WHERE TO ADD FROZEN STATS

Current _refiner_tiled logic:
1. Iterate over tiles
2. Extract tile from rgb + coarse_pred
3. Call refiner_fn(rgb_tile, coarse_tile)
4. Blend with overlap

Add frozen stats collection before tile loop.

### 3. Pipeline Parameter Threading
**File**: `src/corridorkey_mlx/inference/pipeline.py`

- **Line 50-51**: load_model() params: refiner_skip_confidence, refiner_tile_size
- **Line 85-96**: GreenFormer() initialization with these params

Add `refiner_frozen_gn: bool = False` parameter here.

### 4. Test Framework
**File**: `tests/conftest.py`

- **Line 24-26**: Parity tolerances (PARITY_TOL_TIGHT, PARITY_TOL_E2E)
- **Line 42-43**: IMG_SIZE, SMALL_IMG_SIZE constants
- **Line 46-72**: Fixture patterns (random_model, loaded_model, golden_fixtures)

Add frozen GN test fixtures here.

**File**: `tests/test_model_contract.py`

- **Line 53-64**: model_output fixture — test forward pass
- **Line 67-93**: Output shape + dtype + range assertions

Add tests for frozen stats collection correctness here.

### 5. Video Benchmark
**File**: `scripts/bench_video.py`

- Command-line flags: --img-size, --tile-size, --async-decode, --tile-skip-threshold
- Add: --refiner-frozen-gn (boolean)

### 6. Experiment Log
**File**: `research/experiments.jsonl`

- 46 entries currently
- Experiment V3 (#46) shows tiling artifact: tile_size=512 fidelity FAIL
- FrozenGroupNorm experiments will append new entries

---

## Critical Patterns to Follow

### Pattern 1: Parameter Threading
```python
# pipeline.py
def load_model(..., refiner_frozen_gn: bool = False) -> GreenFormer:
    model = GreenFormer(..., refiner_frozen_gn=refiner_frozen_gn)

# corridorkey.py
class GreenFormer(nn.Module):
    def __init__(self, ..., refiner_frozen_gn: bool = False):
        self._refiner_frozen_gn = refiner_frozen_gn
```

### Pattern 2: Precompute + Flag Check
See `DecoderHead.fold_bn()` (line 77-114) and `CNNRefinerModule.prepare_inference()` (line 73-81)

```python
def collect_frozen_stats(self, rgb, coarse_pred):
    """Full-image forward, capture GroupNorm stats at each layer."""
    # Run full refiner, intercept GroupNorm outputs
    # Return list of (mean, var) tuples (9 entries)
    
def set_frozen_stats(self, stats):
    """Store stats for use during tiled inference."""
    self._frozen_gn_stats = stats
    self._frozen_gn_enabled = True

def __call__(self, x):
    # GroupNorm layers check self._frozen_gn_enabled
    # If True, use provided stats instead of computing from input
```

### Pattern 3: Test Fixture
```python
@pytest.fixture(scope="module")
def model_with_frozen_stats():
    model = GreenFormer(img_size=IMG_SIZE, refiner_frozen_gn=True)
    model.load_checkpoint(MLX_CHECKPOINT_PATH)
    model.refiner.prepare_inference()
    return model
```

### Pattern 4: Tolerance-based Assertion
```python
# From test_parity.py
def test_frozen_gn_matches_full_image(model_with_frozen_stats):
    rgb = mx.random.normal((1, 2048, 2048, 3))
    coarse = mx.random.uniform((1, 2048, 2048, 4))
    
    # Full image reference
    result_full = model_with_frozen_stats(mx.concatenate([rgb, coarse], axis=-1))
    
    # Tiled with frozen stats (tile_size=512)
    result_tiled_512 = model_with_frozen_stats(mx.concatenate([rgb, coarse], axis=-1))
    
    # Should match within tight tolerance
    max_diff = float(mx.max(mx.abs(result_full["alpha_final"] - result_tiled_512["alpha_final"])))
    assert max_diff < 1e-5, f"Tile 512 diverged: {max_diff}"
```

---

## Implementation Checklist

### Step 1: Create FrozenGroupNorm Infrastructure
- [ ] Add `FrozenGroupNorm` class (inline in refiner.py or new normalization.py)
- [ ] Implement: collect_stats(), set_frozen_stats(), __call__(with flag check)
- [ ] Add type hints and docstrings (NHWC layout, shape expectations)

### Step 2: Integrate into Refiner
- [ ] Modify RefinerBlock to use FrozenGroupNorm (if separate class) or add flag support
- [ ] Modify CNNRefinerModule:
  - Add collect_frozen_stats(rgb, coarse_pred) method
  - Add set_frozen_stats(stats) method
  - Wire into _refiner_tiled()

### Step 3: Wire into GreenFormer
- [ ] Add refiner_frozen_gn parameter to __init__
- [ ] In _refiner_tiled(), before tile loop:
  - If self._refiner_frozen_gn:
    - Call self.refiner.collect_frozen_stats(rgb_r, coarse_r)
    - Call self.refiner.set_frozen_stats(stats)

### Step 4: Update Pipeline
- [ ] Add refiner_frozen_gn parameter to load_model()
- [ ] Pass through to GreenFormer()

### Step 5: Add Tests
- [ ] Unit: FrozenGroupNorm.collect_stats() shape validation
- [ ] Integration: tile_size=512 with frozen GN matches tile_size=1024
- [ ] Regression: All 94 existing tests pass
- [ ] Parity: Fidelity gates pass (Tier 1 + Tier 2)

### Step 6: Benchmark
- [ ] Add --refiner-frozen-gn flag to bench_video.py
- [ ] Test: tile_size=512 with frozen GN at 2048 resolution
- [ ] Compare: latency, memory, fidelity vs baseline

### Step 7: Log Results
- [ ] Append to research/experiments.jsonl
- [ ] Record: experiment_name, verdict, fidelity_passed, latency, memory
- [ ] Note: files modified, sweep_results if applicable

---

## Expected Outcomes

### Success Criteria
1. tile_size=512 with frozen GN passes fidelity (Tier 1: max_abs < 5e-3)
2. Output identical to tile_size=1024 (within numerical precision)
3. No regression on existing 94 tests
4. Latency improvement ≥ 5% vs baseline at tile_size=1024

### Failure Modes
1. Frozen stats don't capture tile variation → fidelity FAIL
2. Stats pass overhead negates tile skip savings → minimal latency gain
3. MLX GroupNorm internals don't allow stat injection → needs custom kernel
4. Memory savings insufficient to justify code complexity → REVERT

---

## Related Experiments
- **Exp #32**: GroupNorm pytorch_compatible=True is REQUIRED
- **Exp #41**: GroupNorm is 50% of refiner time at 1024
- **Exp #42**: Custom Metal GroupNorm kernel is +41% slower (dead end)
- **Exp #46**: V3 Tile skip blocked by GroupNorm tiling artifact

---

## Handoff Documents
- `research/handoff-2026-03-13-v3-groupnorm-tiling.md` — Problem statement + solution space
- `research/handoff-2026-03-13-post-v1v2.md` — Previous optimization context
- `research/experiments.jsonl` — Full experiment history (46 entries)

