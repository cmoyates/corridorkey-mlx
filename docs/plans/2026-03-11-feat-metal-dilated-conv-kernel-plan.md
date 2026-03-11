---
title: "feat: Custom Metal kernel for dilated convolution in refiner"
type: feat
date: 2026-03-11
---

# Custom Metal Kernel for Dilated Convolution in Refiner

## Overview

Replace MLX's im2col-fallback dilated convolutions in the refiner with a custom Metal kernel via `mx.fast.metal_kernel()`. The refiner's 6 dilated convs (dilation 2,4,8) are excluded from MLX's implicit GEMM optimization (PR #3147), inflating activation memory 9x per conv. This is the dominant remaining bottleneck.

**Expected impact**: 15-20% latency reduction + 15-20% peak memory reduction.

## Problem Statement

MLX's Conv2d dispatches to implicit GEMM only when there is **no input dilation**. Dilated convolutions fall back to explicit im2col, which materializes a `(B*H*W, 9*C)` intermediate tensor per conv call. For `C=64, H=W=512`, that's `512*512*576*4 = 603MB` per conv — called 6 times per forward pass.

This explains:
- Peak memory at 2143MB (despite only ~150MB of model weights)
- `cache_limit` tuning regressions (GC thrashing on massive im2col buffers)
- Refiner bf16 fidelity failures (numerical instability under memory pressure)

## Proposed Solution

A custom Metal compute kernel that computes dilated 3x3 convolution **in-place** — reading 9 input neighbors at dilation stride directly, without materializing im2col. One thread per output spatial position, looping over 64 output channels internally.

### Architecture

```
RefinerBlock (current):
  x → nn.Conv2d(dilation=D) → [im2col 9x blowup] → GEMM → GroupNorm → ReLU
  x → nn.Conv2d(dilation=D) → [im2col 9x blowup] → GEMM → GroupNorm → ReLU
  out = ReLU(conv_out + x)

RefinerBlock (proposed, Phase 1):
  x → metal_dilated_conv3x3(dilation=D) → [NO im2col] → GroupNorm → ReLU
  x → metal_dilated_conv3x3(dilation=D) → [NO im2col] → GroupNorm → ReLU
  out = ReLU(conv_out + x)

RefinerBlock (proposed, Phase 2 — optional fusion):
  x → metal_fused_conv_gn_relu(dilation=D) → [single kernel] → output
  x → metal_fused_conv_gn_relu(dilation=D) → [single kernel] → output
  out = ReLU(fused_out + x)
```

## Technical Approach

### Phase 0: Diagnostic (pre-experiment validation)

Before implementing, resolve three blockers:

#### 0a. mx.compile interaction test — RESOLVED

**Result**: COMPATIBLE. `mx.fast.metal_kernel` works inside `mx.compile` with mixed regular MLX ops. Tested with element-wise kernel + `mx.maximum` (relu). Correct results on repeated calls. No changes needed to `compile_forward=True`.

#### 0b. Weight layout confirmation — RESOLVED

**Result**: CONFIRMED. All 8 refiner conv weights are `(64, 3, 3, 64)` = `(O_out, kH, kW, C_in)`. Bias is `(64,)`. Kernel indexing: `weight[o * 9 * 64 + kh * 3 * 64 + kw * 64 + ci]`.

#### 0c. Fidelity threshold reconciliation

Three values exist in the codebase:
- `benchmark_spec.md`: `< 5e-3` (protected surface — ground truth)
- `compare_reference.py`: `< 1e-3` (hardcoded)
- `run_research_experiment.py`: `FIDELITY_THRESHOLD = 1e-1` (relaxed for 1024x1024)

**Decision**: Use `5e-3` from benchmark_spec as the gate. The `1e-1` in the runner was a per-experiment relaxation, not a global change.

### Phase 1: Standalone dilated conv kernel — COMPLETED (REVERT)

**Hypothesis**: Replacing im2col-based dilated conv with a direct-read Metal kernel reduces latency by 10-20% and peak memory by 15-20%.

**Result**: HYPOTHESIS DISPROVEN. Three approaches tested, all regressed:
- Naive Metal kernel: 228ms vs 122ms (1.87x slower)
- SIMD Metal kernel: 342ms vs 122ms (2.8x slower)
- Sub-pixel decomposition: 126ms/2336MB vs 120ms/2143MB (5% latency + 9% memory regression)

**Root cause**: im2col+GEMM leverages Apple AMX matrix hardware. Bypassing im2col = bypassing AMX.

See: `research/compound/dilated_conv_kernel_experiment.md`

#### Metal kernel design

```metal
// header constants (injected per-call):
// constant uint B, H, W, C_in, C_out, D (dilation), KH=3, KW=3;
// constant int PAD; // = dilation (zero-padding amount)

// source (kernel body):
uint idx = thread_position_in_grid.x;
uint total = B * H * W * C_out;
if (idx >= total) return;

// Decompose linear index → (b, h, w, co)
uint co = idx % C_out;
uint rem = idx / C_out;
uint w_pos = rem % W;
rem = rem / W;
uint h_pos = rem % H;
uint b_pos = rem / H;

float sum = bias[co];

for (uint kh = 0; kh < 3; kh++) {
    for (uint kw = 0; kw < 3; kw++) {
        int ih = (int)h_pos + ((int)kh - 1) * (int)D;
        int iw = (int)w_pos + ((int)kw - 1) * (int)D;

        if (ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W) {
            uint inp_base = ((b_pos * H + (uint)ih) * W + (uint)iw) * C_in;
            uint wt_base = (co * 3 * 3 + kh * 3 + kw) * C_in;

            for (uint ci = 0; ci < C_in; ci++) {
                sum += (float)inp[inp_base + ci] * (float)wt[wt_base + ci];
            }
        }
    }
}

out[idx] = (T)sum;
```

**Key design decisions**:
- **1D grid dispatch**: `grid=(B*H*W*C_out, 1, 1)`, `threadgroup=(256, 1, 1)`. Simpler than 2D, avoids padding waste.
- **Accumulate in float32**: The inner loop casts to float to avoid precision loss, then casts back. This matches the behavior that makes FP32 refiner pass fidelity.
- **Zero-padding via bounds check**: `if (ih >= 0 && ih < H && iw >= 0 && iw < W)` — replicates `nn.Conv2d(padding=dilation)`.
- **Weight layout**: `(O_out, kH, kW, C_in)` matching MLX's stored format. No transpose needed.

#### Integration into RefinerBlock

```python
# In RefinerBlock.__init__:
if dilation > 1:
    self._metal_conv1 = _build_dilated_conv3x3_kernel(dilation)
    self._metal_conv2 = _build_dilated_conv3x3_kernel(dilation)
    self._use_metal = True
else:
    self._use_metal = False  # dilation=1 uses implicit GEMM (fast path)

# In RefinerBlock.__call__:
if self._use_metal:
    out = self._metal_conv1(x, self.conv1.weight, self.conv1.bias)
else:
    out = self.conv1(x)
out = nn.relu(self.gn1(out))
# ... same for conv2/gn2 ...
```

**Files modified**: `src/corridorkey_mlx/model/refiner.py` only (1 file).

#### Validation

1. **Parity test**: Compare `metal_dilated_conv3x3(input, weight, bias, dilation=D)` output against `nn.Conv2d(dilation=D)(input)` for D=2,4,8. Must match within `1e-5`.
2. **Fidelity gate**: `uv run python scripts/run_research_experiment.py` — max_abs_error < 5e-3 per tensor vs golden.npz.
3. **Benchmark**: `uv run python scripts/run_research_experiment.py` — median latency + peak memory.
4. **Score**: `uv run python scripts/score_experiment.py`.
5. **Smoke test**: `uv run python scripts/smoke_2048.py` — no crash at 2048x2048.

#### Rollback criteria

- Fidelity failure (any tensor > 5e-3)
- Latency regression (median > 117.18ms baseline)
- Memory regression (peak > 2143.2MB baseline)
- Score < 1.0 (no net improvement)

### Phase 2: Fused conv+GroupNorm+ReLU (conditional, separate experiment)

Only if Phase 1 succeeds and shows meaningful gain. This is a **second experiment** (separate variable).

Fuse the entire `conv → GroupNorm → ReLU` sequence into a single Metal kernel:
- Compute conv output for a threadgroup tile
- Reduce within threadgroup for GroupNorm mean/variance (using `simd_sum`)
- Normalize + scale + bias + ReLU in same kernel
- Write final output — no intermediate materialization

**Complexity**: High. GroupNorm requires cross-thread reduction (8 groups × 8 channels). Requires threadgroup memory and barriers. Defer until Phase 1 validates the approach.

## Alternative Approaches Considered

### A. ASPP restructuring (stride-2 downsample + standard conv + upsample)

Replaces dilated convs with standard convs that qualify for implicit GEMM. Equivalent mathematical receptive field via downsample/upsample.

**Rejected**: High fidelity risk. Changes the computation, not just the dispatch path. Would require retraining or extensive fidelity validation. Out of scope per CLAUDE.md.

### B. Upgrade MLX and hope im2col is fixed

Wait for MLX to extend implicit GEMM to dilated convolutions.

**Rejected**: No upstream PR or issue indicates this is planned. Blocking on external dependency.

### C. Tiled refiner (spatial tiling to reduce working set)

Process refiner in overlapping 512x512 tiles. Doesn't fix im2col but reduces peak memory per tile.

**Deferred**: Complementary to the Metal kernel approach. Could be combined later for high-resolution inference. Doesn't address the root cause (im2col itself).

## Acceptance Criteria

### Functional

- [ ] Custom Metal kernel computes identical output to `nn.Conv2d(dilation=D)` within 1e-5 max_abs_error for D=2,4,8
- [ ] `nn.Conv2d` still used for dilation=1 (unchanged fast path)
- [ ] Weight loading unchanged — kernel reads existing checkpoint weights directly
- [ ] All existing tests pass (`uv run pytest`)
- [ ] Fidelity gate passes: max_abs_error < 5e-3 per tensor vs golden.npz at 512x512
- [ ] No crash at 2048x2048 (`smoke_2048.py`)

### Performance

- [ ] Median latency < 117.18ms at 512x512 (improvement over baseline)
- [ ] Peak memory < 2143.2MB (improvement over baseline)
- [ ] Score > 1.0 from `score_experiment.py`

### Non-Functional

- [ ] `use_metal_dilated_conv` flag on RefinerBlock for clean fallback
- [ ] Kernel built once in `__init__`, not per-call
- [ ] `verbose=True` tested during development to verify generated Metal source
- [ ] Rollback to `nn.Conv2d` path if flag disabled — zero behavioral change

## Dependencies & Risks

### Dependencies

- `mx.fast.metal_kernel()` API stability (available since MLX 0.14.0+, PR #1325)
- Apple Silicon GPU (M1+ required)

### Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| mx.compile incompatibility | Medium | High | Phase 0a diagnostic. Fallback: eager refiner + compiled backbone |
| Fidelity failure from precision differences | Low | High | Accumulate in float32. Test at per-conv granularity before full forward |
| Threadgroup sizing wrong for some hardware | Low | Medium | Use conservative 256 threads. Test with `verbose=True` |
| Latency regression (kernel dispatch overhead > im2col savings) | Low | High | Measure im2col baseline first. 6 custom dispatches vs 6 im2col+GEMM — fewer total dispatches |
| M1 vs M3 numerical differences | Low | Medium | Test on target hardware. Known MLX issue #2205 |

## Scope Boundaries

**In scope**:
- Custom Metal kernel for dilated 3x3 conv
- Integration into RefinerBlock
- Parity + fidelity + benchmark validation
- Experiment log entry

**Out of scope** (separate experiments if pursued):
- GroupNorm fusion (Phase 2)
- bf16 refiner mode (depends on Phase 1 success)
- Tiled refiner inference
- Other kernel optimizations (attention, decoder)

## Open Questions

- Does `mx.compile` handle `mx.fast.metal_kernel` calls in the graph? (Phase 0a resolves)
- Optimal threadgroup size for this workload on M-series? 256 safe default, but 128 or 512 might be better.
- Is the inner C_in loop (64 iterations per output element) too serial? Could tile over C_in with threadgroup cooperation for better throughput.
- Should we profile im2col overhead in isolation first (Metal GPU capture) to validate the 15-20% estimate?
- MLX version pinned in this repo? Need to verify `mx.fast.metal_kernel` is available.

## References

- Brainstorm: `docs/brainstorms/2026-03-11-algorithmica-hpc-insights-brainstorm.md`
- Root cause analysis: `research/compound/deep_dive_findings.md`
- MLX framework findings: `research/compound/mlx_framework_findings.md`
- Upstream research: `research/compound/upstream_research_2_findings.md`
- MLX custom Metal kernels docs: https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html
- MLX PR #3147 (implicit GEMM): https://github.com/ml-explore/mlx/pull/3147
- MLX PR #1325 (metal_kernel API): https://github.com/ml-explore/mlx/pull/1325
- Algorithmica HPC — cache line counting: https://en.algorithmica.org/hpc/cpu-cache/cache-lines/
- Algorithmica HPC — matmul blocking: https://en.algorithmica.org/hpc/algorithms/matmul/
