---
title: "feat: MLX VRAM & Performance Optimizations for Apple Silicon Engine"
type: feat
date: 2026-03-07
---

# MLX VRAM & Performance Optimizations

## Overview

Six-phase optimization plan targeting memory reduction and throughput improvement in the corridorkey-mlx inference pipeline. Ports PyTorch branch breakthroughs (FP16, decoupled resolutions, 96px overlap) and adds Apple Silicon-specific memory management (lazy slicing, cache clearing, memory limits).

## Problem Statement

Current pain points at 2048x2048:

1. **FP32 everywhere** -- model weights + activations fully float32, wasting memory bandwidth and ALU throughput on M-series chips
2. **No memory management** -- no cache clearing between tiles, no memory limits, no cleanup of intermediates
3. **CPU-GPU roundtrips** -- preprocessing does numpy to PIL to numpy to mx.array; postprocessing does the reverse
4. **Backbone at full resolution** -- Hiera runs at 2048x2048 (saturates GPU cores) when 1024 suffices
5. **9 output tensors** -- forward pass returns 9 tensors but engine only uses 4 (alpha_coarse/final, fg_coarse/final)
6. **Tiling not integrated** -- tiled_inference() exists in inference/tiling.py but is unreachable from CorridorKeyMLXEngine

## Technical Approach

### API Corrections (from research)

The original spec referenced incorrect API paths. Corrected:

| Original (wrong) | Correct (MLX v0.31+) |
|---|---|
| mx.metal.get_peak_memory() | mx.get_peak_memory() |
| mx.metal.get_active_memory() | mx.get_active_memory() |
| mx.metal.clear_cache() | mx.clear_cache() |
| n/a | mx.get_cache_memory() |
| n/a | mx.reset_peak_memory() |
| n/a | mx.set_cache_limit(limit) |
| n/a | mx.set_memory_limit(limit) |

### Architecture Decision: Two Pipeline Paths

Phase 2 (all-MLX preprocessing) and Phase 4 (numpy outer loop for tiling) are contradictory. Resolution:

- **Non-tiled path**: Phase 2 applies -- all preprocessing stays as mx.array, single np.array() at the end
- **Tiled path**: Phase 4 applies -- full image stays as numpy, only tile slices become mx.array on demand

This means process_frame() branches based on whether tiling is active.

---

## Branch

Feature branch: `feat/misc-optimizations`

---

## Implementation Phases

### Phase 0: Benchmarking Infrastructure & Baseline

Set up measurement tooling before altering anything.

#### Tasks

- [x] **0a. Memory profiling helpers** -- src/corridorkey_mlx/utils/profiling.py
  - Add memory_snapshot() returning {active_mb, peak_mb, cache_mb}
  - Uses mx.get_active_memory(), mx.get_peak_memory(), mx.get_cache_memory()
  - Add reset_peak() wrapper around mx.reset_peak_memory()
- [x] **0b. Update scripts/bench_mlx.py**
  - Add memory columns (peak MB, active MB) to the results table
  - Call mx.reset_peak_memory() before each bench run
  - Report mx.get_peak_memory() after graph materialization
- [x] **0c. Update scripts/smoke_2048.py**
  - Replace the try/except fallback chain with the correct mx.* APIs
  - Add cache memory reporting
- [x] **0d. Capture baseline numbers** at 512, 1024, 2048
  - Document: peak memory, active memory, median latency
  - This determines feasibility of the <6GB target

#### Acceptance Criteria

- [x] scripts/bench_mlx.py prints memory columns for each resolution
- [x] Baseline numbers documented in a docs/benchmarks/ file
- [x] All existing tests pass (uv run pytest)

---

### Phase 1: Selective FP16 Inference

Halve memory footprint and boost M-series ALU throughput.

#### Tasks

- [ ] **1a. FP16 weight casting** -- src/corridorkey_mlx/inference/pipeline.py
  - After load_checkpoint(), cast via: model.update(tree_map(lambda x: x.astype(mx.float16), model.parameters()))
  - Cast BEFORE materializing parameters -- lazy graph means FP32 never materializes, saving ~200MB peak
  - Input tensor must also be cast: x = x.astype(mx.float16) before model(x)
- [ ] **1b. FP16 parity test** -- tests/test_fp16_parity.py
  - Compare FP16 MLX output vs FP32 MLX output (NOT vs PyTorch golden)
  - Tolerance: 1e-3 for final outputs (looser than FP32-vs-PT because FP16 drift + refiner's 10x scale)
  - Test all 4 engine-relevant outputs: alpha_coarse, fg_coarse, alpha_final, fg_final
- [ ] **1c. Mixed precision fallback**
  - If full FP16 exceeds tolerance (especially in Hiera's 24 blocks with cumulative drift, or refiner's REFINER_SCALE=10.0 amplifying rounding):
  - Keep HieraBackbone in FP32, cast only DecoderHead and CNNRefinerModule to FP16
  - Implementation: selective tree_map on model.alpha_decoder, model.fg_decoder, model.refiner
- [ ] **1d. Add fp16: bool parameter** to load_model() in pipeline.py

#### Key Risk

GroupNorm with pytorch_compatible=True in the refiner: variance computation at FP16 may amplify errors. The REFINER_SCALE=10.0 multiplier turns a 5e-5 rounding error into 5e-4. Validate refiner delta specifically.

#### Acceptance Criteria

- [ ] FP16 parity test passes at 1e-3 tolerance
- [ ] Peak memory at 512 drops ~40-50% vs FP32 baseline
- [ ] Existing FP32 parity tests unaffected (FP16 is opt-in)

---

### Phase 2: Eliminate CPU-GPU Roundtrips (Non-Tiled Path)

Applies only to the non-tiled process_frame() path.

#### Tasks

- [ ] **2a. Audit engine.py:process_frame()** -- identify all numpy/PIL conversions
  - Current flow: uint8 ndarray -> float32 ndarray -> PIL resize -> ndarray -> ImageNet norm -> mx.array
  - Target flow: uint8 ndarray -> mx.array -> mx resize -> ImageNet norm (single conversion)
- [ ] **2b. MLX-native resize**
  - Replace PIL resize with nn.Upsample(scale_factor=..., mode="linear") or precomputed scale
  - Note: nn.Upsample takes scale factors, not target sizes. Compute scale factor from input/target dims
- [ ] **2c. MLX-native preprocessing**
  - ImageNet normalization as mx.array ops: (x - mean) / std with broadcast constants
  - Alpha hint concatenation as mx.concatenate
- [ ] **2d. Final sync only**
  - Move np.array(result) to the absolute end of process_frame()
  - All postprocessing (sigmoid, clamp, scale to uint8) stays as mx.array ops
- [ ] **2e. Slim forward mode** (opportunistic)
  - Add slim=True parameter to GreenFormer.__call__() that skips returning the 5 unused intermediate tensors
  - Only compute/return: alpha_coarse, fg_coarse, alpha_final, fg_final
  - Saves ~25% intermediate VRAM at 2048

#### Acceptance Criteria

- [ ] process_frame() has exactly ONE np.array() call (at return)
- [ ] No PIL imports in the hot path
- [ ] Engine integration test passes with same output shapes/dtypes

---

### Phase 3: Decouple Backbone and Refiner Resolutions

Biggest VRAM win -- backbone at 1024 instead of 2048.

#### Tasks

- [ ] **3a. Restructure GreenFormer.__call__()** -- src/corridorkey_mlx/model/corridorkey.py
  - Add backbone_size: int | None = None parameter
  - If backbone_size is set and differs from input spatial dims:
    1. Store original full-res input x_full = x
    2. Downsample x to (1, backbone_size, backbone_size, 4) using bilinear (nn.Upsample mode="linear")
    3. Run backbone + decoder at backbone_size
    4. Upsample decoder outputs (alpha_coarse, fg_coarse) back to full resolution
    5. Extract RGB from x_full[:, :, :, :3] for refiner concatenation
    6. Run refiner at full resolution
  - If backbone_size is None or matches input, current behavior preserved
- [ ] **3b. pos_embed alignment**
  - Verify _interpolate_pos_embed works correctly for backbone_size=1024 (tokens_spatial_shape=[256,256]=65536)
  - This is a 4x downsample from training res (2048). Already handled by cubic interpolation, but validate quality
- [ ] **3c. Compilation strategy**
  - With dual resolutions in one forward pass, mx.compile on the whole __call__ sees varying internal shapes
  - Option A: Don't compile when backbone_size differs from input (simplest)
  - Option B: Split into compiled_backbone(x_down) + compiled_refiner(x_full, coarse) (better perf)
  - Recommend Option A first, Option B as follow-up
- [ ] **3d. Quality validation**
  - No PyTorch reference exists for this modified architecture
  - Validate by visual inspection on real frames + comparing coarse outputs at 1024 vs 2048
  - Document acceptable quality delta (this is a lossy optimization)
- [ ] **3e. Parity test update**
  - Add test_decoupled_resolution.py -- not a parity test vs golden, but a regression test:
  - Assert outputs are valid (no NaN/Inf, alpha in [0,1], reasonable value ranges)
  - Assert shapes match full-resolution output

#### Key Risk: Tiling Interaction

If tiling is active (Phase 4), tiles are 512x512. With backbone_size=1024, each tile is smaller than the backbone expects. Decision: **tiling ignores backbone_size** (tiles always run at tile_size). Simpler and avoids wasteful upsampling.

#### Acceptance Criteria

- [ ] GreenFormer(backbone_size=1024) produces valid outputs at 2048x2048 input
- [ ] Peak memory at 2048 drops significantly vs Phase 1 baseline
- [ ] backbone_size=None preserves exact existing behavior (backward compatible)
- [ ] Compilation still works when backbone_size=None

---

### Phase 4: Advanced Tiled Inference (Apple Silicon Hardened)

Harden inference/tiling.py for memory-constrained environments.

#### Tasks

- [ ] **4a. 96px overlap default** -- src/corridorkey_mlx/inference/tiling.py
  - Change DEFAULT_OVERLAP = 64 to DEFAULT_OVERLAP = 96
  - User-overridable via parameter
- [ ] **4b. Lazy tile slicing**
  - Change tiled_inference() signature: accept x as np.ndarray (NHWC float32)
  - Only convert each tile slice to mx.array right before model(tile)
  - This keeps the full image out of GPU memory
  - Update docstring and type hints
- [ ] **4c. Memory management per tile**
  - After materializing and accumulating each tile, delete mx references and clear cache
  - Pattern: materialize -> accumulate to numpy -> del tile refs -> mx.clear_cache()
- [ ] **4d. Cache limit configuration**
  - At start of tiled_inference(): mx.set_cache_limit(0) (aggressive: no speculative caching)
  - Restore original limit at end
- [ ] **4e. Tiling ignores backbone_size**
  - When tiling is active, each tile runs through the model at tile_size resolution
  - backbone_size parameter is not applied per-tile (Phase 3 interaction, Option A)
  - Document this decision

#### Acceptance Criteria

- [ ] Peak memory during tiled 2048 inference is bounded (no monotonic growth across tiles)
- [ ] mx.get_cache_memory() returns ~0 after each tile
- [ ] 96px overlap produces better edge blending than 64px (visual check)
- [ ] Existing tiling tests updated and passing

---

### Phase 5: Engine API & Compilation Integration

Wire optimizations into CorridorKeyMLXEngine.

#### Tasks

- [ ] **5a. New engine parameters** -- src/corridorkey_mlx/engine.py
  - fp16=True (NEW, default True)
  - backbone_size=1024 (NEW, None = same as img_size)
  - tile_size=512 (NEW, None = no tiling)
  - tile_overlap=96 (NEW, only used when tile_size set)
- [ ] **5b. Parameter validation**
  - backbone_size must be None or divisible by patch_stride (4)
  - backbone_size must be <= img_size
  - tile_overlap must be < tile_size
  - tile_size must be divisible by patch_stride (4)
  - If compile=True and backbone_size != img_size and backbone_size is not None: disable compile with warning (Phase 3 Option A)
- [ ] **5c. Tiling integration in process_frame()**
  - If tile_size is set: route to tiled_inference() instead of direct model(x)
  - Preprocessing still happens once (full image), then tiles are sliced from the preprocessed numpy array
  - Tiling trigger: tile_size is not None (explicit opt-in)
- [ ] **5d. Memory limit safety rail**
  - On init: mx.set_memory_limit(int(0.8 * total_unified_memory)) if detectable
  - Fallback: skip if API unavailable
- [ ] **5e. Benchmark matrix** -- scripts/bench_mlx.py
  - Add configurations: {fp16, fp32} x {backbone_size: None, 1024} x {tiled, non-tiled} at 2048
  - Target: <6GB active Metal memory for fp16=True, backbone_size=1024, tiled
  - Include 512/1024 as regression checks
- [ ] **5f. Update scripts/smoke_2048.py**
  - Exercise all new engine parameters
  - Report memory per configuration

#### Acceptance Criteria

- [ ] Engine accepts all new parameters without breaking existing API (defaults match current behavior)
- [ ] fp16=False, backbone_size=None, tile_size=None produces identical output to current engine
- [ ] Benchmark matrix runs and produces comparison table
- [ ] <6GB active memory target achieved (or documented why not)

---

## Risk Analysis & Mitigation

| Risk | Impact | Mitigation |
|---|---|---|
| FP16 GroupNorm drift in refiner (10x scale amplification) | Visible matte artifacts | Mixed precision fallback: backbone FP32, decoder/refiner FP16 |
| Backbone at 1024 degrades matte edges | Quality regression | Visual validation on real frames; keep backbone_size=None as default until validated |
| mx.compile incompatible with dual-resolution forward | No compilation speedup for Phase 3 | Disable compile when backbone_size differs; split compilation as follow-up |
| Tiling + backbone_size interaction undefined | Implementation deadlock | Decision: tiling ignores backbone_size (Option A) |
| MLX memory APIs version-dependent | Silent failures in benchmarks | Wrap in try/except with clear warnings; document minimum MLX version |
| No PyTorch reference for decoupled architecture | Cannot validate Phase 3 correctness | Use regression tests (valid ranges, no NaN) + visual inspection |

## Dependencies & Prerequisites

- MLX >= 0.31.0 (for mx.get_active_memory(), mx.clear_cache(), mx.set_cache_limit())
- Existing golden fixtures (reference/fixtures/golden.npz) for Phase 0/1 parity
- Real sample images for Phase 3 quality validation
- Apple Silicon hardware for memory benchmarks (M1/M2 16GB target)

## Success Metrics

| Metric | Current (est.) | Target |
|---|---|---|
| Peak memory at 2048 (non-tiled) | ~12GB (FP32) | <6GB (FP16 + backbone 1024) |
| Peak memory at 2048 (tiled) | unbounded growth | bounded to ~2GB per tile |
| Throughput at 2048 | baseline | 1.5-2x via FP16 + compile |
| CPU-GPU syncs per frame | ~6+ | 1 (final output only) |
| Parity vs FP32 MLX | n/a | <1e-3 for FP16, exact for FP32 path |

## Open Questions

1. **Baseline numbers?** Phase 0 must run first -- feasibility of <6GB depends on current actual usage
2. **FP16 tolerance for refiner?** The 10x scale factor may force mixed precision from day one
3. **backbone_size quality at 1024?** Trained at 2048 -- pos_embed interpolation to 1024 is a 4x downsample; needs real-frame validation
4. **Slim forward mode worth it?** Skipping 5 unused outputs saves VRAM but adds conditional logic to __call__
5. **mx.set_wired_limit()?** macOS 15+ only; could guarantee resident memory for weights but adds platform dependency
6. **Split compilation for Phase 3?** Option B (separate compiled_backbone + compiled_refiner) is better perf but more complex; defer?

## References

### Internal

- src/corridorkey_mlx/engine.py -- main engine API
- src/corridorkey_mlx/model/corridorkey.py -- GreenFormer forward pass (9 outputs)
- src/corridorkey_mlx/model/hiera.py -- backbone with shape-dependent reshapes
- src/corridorkey_mlx/model/refiner.py -- GroupNorm pytorch_compatible=True, REFINER_SCALE=10.0
- src/corridorkey_mlx/inference/tiling.py -- current tiling (64px overlap, numpy accumulators)
- src/corridorkey_mlx/inference/pipeline.py -- load_model(), compile_model()
- scripts/bench_mlx.py -- benchmark script (no memory tracking)
- scripts/smoke_2048.py -- smoke test (has memory API exploration)
- tests/conftest.py -- tolerance constants (PARITY_TOL_TIGHT=1e-4, PARITY_TOL_E2E=1e-3)

### External

- MLX Unified Memory docs: https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html
- Writing Fast MLX (Awni Hannun): https://gist.github.com/awni/4beb1f7dfefc6f9426f3a7deee74af50
- MLX Metal module API: https://ml-explore.github.io/mlx/build/html/python/metal.html
