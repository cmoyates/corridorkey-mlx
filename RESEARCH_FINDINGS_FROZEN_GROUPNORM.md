# Repository Analysis: Patterns for FrozenGroupNorm Implementation

Research conducted on 2026-03-13 for implementing FrozenGroupNorm class to solve GroupNorm tiling artifacts in refiner.

## 1. Custom nn.Module Subclass Patterns

All custom modules follow a consistent structure defined by MLX's `nn.Module` base class:

### Structure Template
- **Inheritance**: `class CustomModule(nn.Module)`
- **Constructor**: `__init__` calls `super().__init__()` first, then instantiates child layers
- **Forward method**: `__call__(self, x: mx.array) -> mx.array` (MLX uses `__call__`, not `forward`)
- **Type hints**: All parameters and returns use `mx.array` with shape annotations in docstrings
- **Documentation**: Module-level docstring describes NHWC layout assumptions, input/output shapes

### Real Examples from Codebase

**RefinerBlock** (`src/corridorkey_mlx/model/refiner.py:19-45`)
- Dilated residual block with 2 GroupNorm layers
- Child layers stored as attributes: `self.conv1`, `self.gn1`, `self.conv2`, `self.gn2`
- Forward: `self.gn1(self.conv1(x))` with residual connection `out + residual`
- All GroupNorm use `pytorch_compatible=True` flag

**DecoderHead** (`src/corridorkey_mlx/model/decoder.py:31-175`)
- Complex module with precomputed state (folded BN, 2D weights)
- Uses `_bn_folded`, `_bn_scale`, `_bn_offset` as private cached fields
- Lazy initialization: `if self._fuse_weight_2d is not None` checks before using
- Provides `fold_bn()` method to precompute optimization state (called post-weights, pre-inference)
- Materializes computed tensors with `mx.eval(...)` after reshaping

**GreenFormer** (`src/corridorkey_mlx/model/corridorkey.py:27-300`)
- Top-level composition combining backbone, decoders, refiner
- Parameter passing via `__init__`: stores as `self._param_name` attributes
- Property decorators: `@property` for read-only accessors (`tile_skip_stats`, `compiled`)
- Separate `__call__` (compile-safe) and `forward_eager` (optimizations) methods

---

## 2. Test Structure Patterns

Tests follow pytest conventions with specific organizational patterns:

### Test File Organization
- **Location**: `tests/` directory at repo root
- **Naming**: `test_*.py` files
- **Module imports**: All test files import from `corridorkey_mlx` and `tests.conftest`
- **Protected surface**: All test files read-only (no modifications during experiments)

### Fixture Patterns (`tests/conftest.py`)

**Module-scoped fixtures** (loaded once per test session):
```python
@pytest.fixture(scope="module")
def random_model() -> GreenFormer:
    model = GreenFormer(img_size=IMG_SIZE)
    model.eval()
    mx.eval(model.parameters())
    return model
```

Characteristics:
- Typed return annotations
- Large setup cost uses `scope="module"`
- Parameters materialized with `mx.eval()` to avoid lazy evaluation
- Skip markers for conditional execution:
  `has_golden = pytest.mark.skipif(not GOLDEN_PATH.exists(), ...)`

### Test Categories
1. **test_model_contract.py** — imports, shapes, output ranges, determinism
2. **test_parity.py** — numerical parity vs PyTorch golden reference
3. **test_compilation.py** — mx.compile() correctness
4. **test_tiling.py** — refiner tiling + blending logic
5. **test_engine.py** — end-to-end inference
6. **test_weights.py** — checkpoint loading
7. **test_conversion.py** — PyTorch→MLX conversion

### Tolerance Patterns
```python
PARITY_TOL_TIGHT = 1e-4
PARITY_TOL_BACKBONE = 5e-1
PARITY_TOL_E2E = 1e-3

OUTPUT_TOLERANCES: dict[str, float] = {
    "alpha_logits": 3e-2,
    "fg_logits": 3e-2,
    "alpha_final": 5e-3,
    "fg_final": 1.5e-2,
}
```

---

## 3. Pipeline Parameter Flow Pattern

Parameter passing chain: `load_model()` → `GreenFormer.__init__()` → instance attributes

### load_model Signature (`src/corridorkey_mlx/inference/pipeline.py:35-97`)
```python
def load_model(
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    img_size: int = DEFAULT_IMG_SIZE,
    refiner_skip_confidence: float | None = None,
    refiner_tile_size: int | None = 1024,
) -> GreenFormer:
```

Parameter categories:
1. **Model architecture**: `img_size`, `dtype`, `fused_decode`, `slim`
2. **Optimization flags**: `compile`, `use_sdpa`, `stage_gc`
3. **Precision control**: `refiner_dtype`, `decoder_dtype`, `backbone_bf16_stages123`
4. **Memory tuning**: `wired_limit_bytes`, `cache_limit_bytes`
5. **Refiner control**: `refiner_skip_confidence`, `refiner_tile_size`

### GreenFormer.__init__ Pattern
```python
def __init__(
    self,
    img_size: int = 512,
    refiner_skip_confidence: float | None = None,
    refiner_tile_size: int | None = 1024,
) -> None:
    super().__init__()
    self._refiner_skip_confidence = refiner_skip_confidence
    self._refiner_tile_size = refiner_tile_size
```

Parameters accessed as `self._<param>` throughout class methods.

**Pattern for `refiner_frozen_gn`**:
1. Add to `load_model()` with default `False`
2. Pass to `GreenFormer.__init__()` as `refiner_frozen_gn: bool = False`
3. Store as `self._refiner_frozen_gn`
4. Use in `_refiner_tiled()` or `prepare_refiner()`

---

## 4. Experiment Log Format (`research/experiments.jsonl`)

Each line is a JSON object:
```json
{
  "experiment_name": "frozen-groupnorm-v4",
  "timestamp": "2026-03-13T12:00:00+00:00",
  "resolution": 1024,
  "search_area": "groupnorm-tiling",
  "verdict": "INCONCLUSIVE|KEEP|REVERT",
  "fidelity_passed": true,
  "median_ms": 5552,
  "p95_ms": 5560,
  "peak_memory_mb": 3319.1,
  "notes": "Frozen stats pass enables tile_size=512",
  "files": ["src/corridorkey_mlx/model/refiner.py"],
  "artifact": "/path/to/result.json"
}
```

Optional fields:
- `files`: list of modified files
- `sweep_results`: dict of sub-experiment results
- `improvement_pct`: relative to baseline

---

## 5. Existing Custom Normalization Classes

**Current state**: No custom normalization classes exist.

All normalization uses **MLX built-ins**:
- `nn.GroupNorm(num_groups, num_channels, pytorch_compatible=True)` — 9x in refiner
- `nn.BatchNorm(num_features)` — in decoder

### Critical Finding
From exp #32: `pytorch_compatible=True` is REQUIRED for parity.
Without it: `alpha_final=0.987, fg_final=0.973` error (vs 0.050 threshold) — **catastrophic failure**.

### Utility Pattern
Custom logic lives in `src/corridorkey_mlx/utils/`:
- `quantize.py`: `safe_quantize(module)` wrapper
- Exported via `__init__.py`

**For FrozenGroupNorm**, could go in:
- New file: `src/corridorkey_mlx/model/normalization.py`
- Or inline in `refiner.py` (smallest scope)

---

## 6. Video Pipeline Integration

### VideoProcessor Parameter Flow
Location: `src/corridorkey_mlx/inference/video.py`

```python
class VideoProcessor:
    def __init__(
        self,
        model: GreenFormer,
        img_size: int = 1024,
        async_save: bool = True,
    ) -> None:
        self.model = model  # Pre-configured GreenFormer
```

**Key insight**: VideoProcessor receives pre-configured model. No additional parameter passing needed — `refiner_frozen_gn` is already set in the model.

### Benchmark Integration
Location: `scripts/bench_video.py`

Current flags: `--img-size`, `--async-decode`, `--tile-size`, `--tile-skip-threshold`

For FrozenGroupNorm testing, would add:
- `--refiner-frozen-gn` flag (boolean)
- `--frozen-gn-sweep` for tile_size comparisons

---

## 7. Key Architecture Insights for FrozenGroupNorm

### Refiner Structure (`src/corridorkey_mlx/model/refiner.py`)

- **9 GroupNorm instances**:
  - 1 stem: `RefinerBlock`
  - 4 blocks × 2 GN/block = 8
  - Total = 9

- **All use** `pytorch_compatible=True`
- **All normalize over** (H, W) within each channel group (8 groups for 64 channels)

### Tiling Architecture (`src/corridorkey_mlx/model/corridorkey.py:280+`)

Current approach:
1. Divide image into tiles (default size 1024×1024)
2. 32px overlap for receptive field
3. Blend weights fade at boundaries
4. Each tile runs full refiner independently
5. **Problem**: Each tile's GroupNorm computes stats over that tile only → divergent statistics

### Video Use Case Results

From `handoff-2026-03-13-v3-groupnorm-tiling.md`:
- tile_size=1024 on 2048: 0% skip (boundaries in all quadrants), fidelity PASS
- tile_size=512 on 2048: 33% skip, 8% latency gain, fidelity FAIL (GroupNorm artifact)

Frozen GN enables tile_size=512 without fidelity loss.

---

## 8. Implementation Reference Points

### Preparing Inference Pattern
**DecoderHead.fold_bn()** (`src/corridorkey_mlx/model/decoder.py:77-114`)
- Precomputes fused parameters before inference
- Called after weights loaded
- Computes: `_bn_scale`, `_bn_offset`, `_fuse_weight_chunks`
- Materializes with `mx.eval(*arrays)`
- Sets flag: `self._bn_folded = True`
- Forward checks flag and uses precomputed vs original

**CNNRefinerModule.prepare_inference()** (`src/corridorkey_mlx/model/refiner.py:73-81`)
- Precomputes 2D weight from 1x1 conv
- Reshape, `mx.eval()`, store as `self._final_weight_2d`

**Pattern for FrozenGroupNorm**:
1. Method: `collect_stats(rgb, coarse) -> list[dict]`
   - Full-image forward
   - Capture per-layer (mean, var) from each GroupNorm
   - Return stats list
2. Method: `set_frozen_stats(stats)`
   - Store as `self._frozen_gn_stats`
   - Each GroupNorm checks flag during forward
3. In `_refiner_tiled()`: `collect_stats()` once before tile loop

### Assertion/Debug Pattern
```python
assert set(model_output.keys()) == set(EXPECTED_SHAPES.keys())
assert model_output[key].shape == expected_shape, (
    f"{key}: expected {expected_shape}, got {model_output[key].shape}"
)
```

For FrozenGroupNorm tests:
- Stats list has 9 entries (one per GN layer)
- Each entry has (mean_shape, var_shape) matching feature maps
- Tiled output with frozen stats matches full-image output exactly

---

## 9. Unresolved Questions (from Handoff)

From `research/handoff-2026-03-13-v3-groupnorm-tiling.md:125-132`:

1. **Avoid full-image stats pass?**
   - Option: downscaled stats, running stats from first tile
   - Implication: Full pass required for correctness

2. **Separate FrozenGroupNorm class vs mode flag?**
   - Recommended: Separate class (cleaner, explicit intent)
   - Alternative: Mode flag (less code, less clear)

3. **MLX nn.GroupNorm internals**
   - Weights/bias accessible? (yes)
   - Can inject external (mean, var)? (needs custom impl)

4. **tile_size=512 net benefit after stats overhead?**
   - 27% fewer calls, but 1 full-image stats pass
   - Net: ~6% savings
   - Verdict: Marginal but enables smaller tiles without artifacts

5. **Larger overlap alternative?**
   - 256px overlap on 512 tiles = effective tile ~1024px = same as current
   - Simpler than frozen GN, less memory-efficient

---

## Summary: Implementation Checklist

### Files to Create/Modify
- [ ] `src/corridorkey_mlx/model/refiner.py` — Add FrozenGroupNorm or modify flow
- [ ] `src/corridorkey_mlx/model/corridorkey.py` — Wire frozen stats into `_refiner_tiled()`
- [ ] `src/corridorkey_mlx/inference/pipeline.py` — Add `refiner_frozen_gn` parameter
- [ ] `tests/test_refiner.py` (new) — Test frozen GN stats collection
- [ ] `scripts/bench_video.py` — Add `--refiner-frozen-gn` flag

### Custom Module Structure
- Inherit from `nn.Module`
- Use `__call__` for forward
- Store child layers as instance attributes
- Precompute invariants in `prepare_*()` methods
- Check flags in forward path

### Testing Approach
- Unit: FrozenGroupNorm.collect_stats() returns correct shapes
- Integration: tile_size=512 with frozen GN matches tile_size=1024
- Regression: All 94 existing tests pass
- Benchmark: Latency + memory for tile_size=512 with/without frozen GN

### Parameter Threading
1. Add `refiner_frozen_gn: bool = False` to `load_model()`
2. Pass to `GreenFormer.__init__(refiner_frozen_gn=...)`
3. Store as `self._refiner_frozen_gn`
4. Use in `_refiner_tiled()`: `if self._refiner_frozen_gn: collect_stats_first()`

---

## Key Files for Reference

**Model code**:
- `src/corridorkey_mlx/model/refiner.py` — RefinerBlock, CNNRefinerModule (9x GroupNorm)
- `src/corridorkey_mlx/model/corridorkey.py` — GreenFormer._refiner_tiled() method
- `src/corridorkey_mlx/model/decoder.py` — DecoderHead.fold_bn() pattern (precompute optimization)

**Inference**:
- `src/corridorkey_mlx/inference/pipeline.py` — load_model() signature and param threading
- `src/corridorkey_mlx/inference/video.py` — VideoProcessor (stateful per-frame processing)

**Testing**:
- `tests/conftest.py` — Fixtures, tolerances, skip markers
- `tests/test_model_contract.py` — Shape + output contract tests
- `tests/test_parity.py` — Numerical parity vs PyTorch golden reference

**Research**:
- `research/experiments.jsonl` — Experiment log format
- `research/handoff-2026-03-13-v3-groupnorm-tiling.md` — Detailed problem statement and solution space
- `research/handoff-2026-03-13-post-v1v2.md` — Previous experiment context
