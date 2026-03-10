---
title: "NCHW→NHWC transpose missing in experiment scripts + feature tensor lifecycle leak"
date: 2026-03-10
category: runtime-errors
tags:
  - layout-transpose
  - NCHW-NHWC
  - memory-discipline
  - tensor-lifecycle
  - golden-fixtures
  - tiled-inference
components:
  - scripts/run_research_experiment.py
  - scripts/compare_reference.py
  - src/corridorkey_mlx/model/corridorkey.py
  - src/corridorkey_mlx/inference/tiling.py
severity: medium
resolved: true
---

# NCHW→NHWC Transpose Bug + Tensor Lifecycle Memory Discipline

## Problem

Two issues discovered while setting up the autoresearch experiment loop:

1. **Layout mismatch crash** — `run_research_experiment.py` and `compare_reference.py` loaded golden fixture input as NCHW `(1,4,512,512)` directly into an NHWC model, causing `ValueError: [conv] Expect the input channels in the input and weight array to match`.

2. **Stale backbone features** — `features` list (~13MB at 512×512) stayed alive through the refiner stage in `GreenFormer.__call__` despite the refiner never using them. With `stage_gc=True`, these are materialized Metal buffers occupying GPU memory unnecessarily.

## Solution

### Problem 1: NCHW→NHWC Transpose Bug

**Investigation:**
1. `run_research_experiment.py` crashed on first conv: input `(1,4,512,512)` vs weight `(112,7,7,4)` — NCHW vs NHWC mismatch.
2. Checked `golden.npz` — stored by PyTorch reference harness in NCHW convention.
3. Checked `tests/test_parity.py` — already used `nchw_to_nhwc_np()` correctly. The standalone scripts simply never applied this transform.

**Fix:** Added layout imports and transposes in both scripts:

```diff
+from corridorkey_mlx.utils.layout import nchw_to_nhwc_np, nhwc_to_nchw_np

 # Input: NCHW → NHWC
-x = mx.array(ref["input"])
+x = mx.array(nchw_to_nhwc_np(ref["input"]))

 # Output comparison: NHWC → NCHW (to match golden refs)
-mlx_tensor = np.array(outputs[key])
+mlx_tensor = nhwc_to_nchw_np(np.array(outputs[key]))
```

**Files:** `scripts/run_research_experiment.py`, `scripts/compare_reference.py`

### Problem 2: Tensor Lifecycle Memory Discipline

**Investigation:**
1. In `GreenFormer.__call__`, the `features` list (4 multi-scale tensors from Hiera backbone) was passed to both decoders but never explicitly deleted.
2. With `stage_gc=True`, these are materialized Metal buffers. Python's local variable holds a reference, preventing Metal from reclaiming ~13MB.
3. The refiner uses `rgb` (sliced from input) + `coarse_pred` (from decoder outputs) — never `features`.

**Fix:**

```diff
# corridorkey.py — after both decoders consume features:
  alpha_logits = self.alpha_decoder(features)
  fg_logits = self.fg_decoder(features)
+ del features

# tiling.py — add w, w3d to tile loop cleanup:
- del out, tile, alpha_tile, fg_tile
+ del out, tile, alpha_tile, fg_tile, w, w3d
```

**Result:** -3.9% median latency at 512×512, fidelity PASS (all tensors within 1e-3 of golden), score 1.0289.

## Root Cause

1. **Layout bug:** Convention mismatch between PyTorch (NCHW) and MLX (NHWC) not enforced at script boundaries. The parity tests handled it; the experiment scripts didn't because they were written independently.

2. **Memory leak:** MLX lazy evaluation holds buffer references until Python refs drop. Without explicit `del`, backbone features survive through the entire refiner computation (~20 lines of code) before going out of scope at function return.

## Prevention

### Layout Mismatch Prevention

- **Centralize all layout transforms** in `utils/layout.py`. Never use raw `np.transpose(..., (0,2,3,1))` outside that module.
- **Name variables with layout suffix** at model boundaries: `input_nchw`, `input_nhwc`.
- **Document fixture layout** — golden.npz stores NCHW (PyTorch origin). Every consumer must go through `nchw_to_nhwc_np()`.

**Code review checklist:**
- [ ] Every script loading `.npz` fixtures converts layout before feeding to MLX model
- [ ] No raw `np.transpose` with 4-element tuples outside `utils/layout.py`
- [ ] New fixtures document their storage layout

### Tensor Lifecycle Prevention

- **Treat large MLX tensors like file handles** — explicitly `del` when no longer needed, especially before `gc.collect() + mx.clear_cache()` barriers.
- **Multi-stage pipelines** should release each stage's inputs before the next stage allocates.
- **Never store backbone features** beyond their consumer's scope.

**Code review checklist:**
- [ ] No backbone feature tensors referenced after decoder consumption
- [ ] `stage_gc=True` paths have explicit `del` + `mx.eval` between pipeline stages
- [ ] Large intermediates not captured in closures or class attributes that outlive forward pass
- [ ] Peak memory is part of benchmark output for every experiment

## Cross-References

- **Experiment:** `research/compound/exp001_del_features_after_decoder.md`
- **Plan:** `docs/plans/2026-03-10-feat-tile-lifecycle-memory-discipline-plan.md`
- **Related pattern:** MaskUnitAttention transpose bug (silent corruption) — documented in project memory
- **Layout utilities:** `src/corridorkey_mlx/utils/layout.py`
