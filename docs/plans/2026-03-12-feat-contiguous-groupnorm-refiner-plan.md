---
title: "feat: Contiguous GroupNorm for refiner"
type: feat
date: 2026-03-12
---

# feat: Contiguous GroupNorm for Refiner

## Overview

Eliminate GroupNorm transpose-copy overhead (6.94% GPU time @512) in the CNN refiner. Metal trace shows `g3_copyfloat16float16` kernels from `nn.GroupNorm(pytorch_compatible=True)`'s internal transpose producing non-contiguous views. 9 GroupNorm instances in the refiner (stem + 4 blocks x 2).

## Problem Statement

MLX's pytorch-compatible GroupNorm path:
```
reshape(B, HW, G, gs) -> transpose(0,2,1,3) -> layer_norm -> transpose back -> affine
```
Transpose #2 produces non-contiguous output. Subsequent ops force a Metal copy kernel. This happens 9x per refiner forward pass.

**Key constraint**: Dropping `pytorch_compatible=True` causes catastrophic fidelity failure (exp32: alpha=0.987 error). The normalization semantics must be preserved exactly.

## Proposed Solution

Two sequential experiments. Experiment A validates the problem at target resolution and establishes baseline. Experiment B implements the fix only if A confirms the overhead is material.

---

## Experiment A: Validate & Baseline (Quick Win)

**Hypothesis**: The 6.94% GN copy overhead measured at 512 may differ at 1024 (the primary benchmark target). The compiled refiner may already eliminate some copies.

### A.1 -- Metal trace at 1024

Capture a Metal GPU trace at 1024x1024 and measure `g3_copyfloat16float16` kernel time.

**Success criteria**: GN copy kernels account for >=3% of total GPU time at 1024. If <3%, skip Experiment B (ROI too low for custom kernel complexity).

### A.2 -- Compiled vs eager GN copy count

Test whether `mx.compile` on the refiner already eliminates the transpose copies:

- Run model with `compile_refiner=True` (default), capture trace
- Run model with `compile_refiner=False`, capture trace
- Compare `g3_copyfloat16float16` kernel count and time

**Files to modify (temporary, revert after measurement)**:
- `src/corridorkey_mlx/model/refiner.py` -- add materialization points for tracing

### A.3 -- Explicit contiguous hint

Test whether reshaping after GroupNorm changes the compiled graph:

```python
# In RefinerBlock.__call__:
out = nn.relu(self.gn1(self.conv1(x)))
out = mx.reshape(out, out.shape)  # force contiguous view?
```

**Expected**: Likely no effect (mx.compile should handle this), but cheap to test.

### Benchmark & decide

```bash
uv run python scripts/run_research_experiment.py
uv run python scripts/score_experiment.py
```

**Decision gate**: If GN copies <3% at 1024 -> document finding, close. If >=3% -> proceed to Experiment B.

---

## Experiment B: Partial-Fusion GroupNorm Kernel

**Hypothesis**: A fused normalize+affine+relu kernel (without transposes) can eliminate the copy overhead while preserving fidelity.

### Why "partial fusion" not "full fusion"

The SpecFlow analysis identified critical blockers for a single-pass kernel:
1. **8M element reduction** (1024^2 x 8 per group) requires threadgroup shared memory + barriers
2. **`mx.fast.metal_kernel()` reduction capability is unproven** for this scale
3. **Custom kernel ops are NOT fusable** by mx.compile -- 9 fusion barriers could offset gains

**Revised approach**: Use MLX primitives for mean/var computation, custom Metal kernel only for the fused normalize+affine(+optional relu) pass. This eliminates the transpose while keeping reductions in optimized MLX kernels.

### B.1 -- Prototype: reduction feasibility check

Before implementing, verify `mx.fast.metal_kernel()` can do threadgroup reductions:

```python
# Minimal test: compute mean of (1, 1024*1024, 8) tensor via metal_kernel
# If feasible -> single-pass kernel
# If not -> partial-fusion approach (mx.mean/var + fused normalize kernel)
```

**Files**: New `scripts/test_metal_reduction.py` (temporary, not committed)

### B.2 -- Implement ContiguousGroupNorm

Two variants depending on B.1 outcome:

**Variant 1 (full kernel -- if B.1 succeeds)**:
Custom Metal kernel that reads NHWC directly, computes per-group mean/var via two-pass reduction (accumulate in fp32), then normalizes+affine+relu in one pass. No transpose needed.

**Variant 2 (partial fusion -- if B.1 fails)**:
```python
class ContiguousGroupNorm(nn.Module):
    """GroupNorm without transpose -- MLX stats + fused normalize+affine+relu."""

    def __call__(self, x):
        B, *spatial, C = x.shape
        # Reshape to (B, HW, G, gs) -- NO transpose
        x_grouped = x.reshape(B, -1, self.num_groups, self.group_size)
        # Compute stats over spatial+channel dims (axes 1,3) -- MLX handles reduction
        mean = mx.mean(x_grouped, axis=(1, 3), keepdims=True)
        var = mx.var(x_grouped, axis=(1, 3), keepdims=True)
        # Fused normalize+affine+relu via Metal kernel (element-wise only)
        out = self._normalize_kernel(x_grouped, mean, var, self.weight, self.bias)
        return out.reshape(B, *spatial, C)
```

**Critical**: Both variants must accumulate mean/var in fp32 regardless of input dtype (bf16 accumulation over 8M elements = catastrophic precision loss).

**Files to modify**:
- `src/corridorkey_mlx/model/refiner.py` -- add `ContiguousGroupNorm`, wire into `RefinerBlock` and `CNNRefinerModule`

### B.3 -- Fused ReLU handling

Not all GN sites are followed by ReLU:
- **5x GN+ReLU**: stem_gn + gn1 in each of 4 blocks
- **4x GN-only**: gn2 in each of 4 blocks (ReLU comes after residual add)

Use a `relu` flag on `ContiguousGroupNorm`:
```python
self.gn1 = ContiguousGroupNorm(8, 64, relu=True)   # fused
self.gn2 = ContiguousGroupNorm(8, 64, relu=False)  # no relu
```

### B.4 -- Weight loading compatibility

The existing checkpoint has `gn1.weight`, `gn1.bias` etc. The `ContiguousGroupNorm` must use the same weight names so `load_checkpoint()` works without key remapping.

### B.5 -- Fallback path

Make custom GN opt-in via constructor flag:
```python
class CNNRefinerModule(nn.Module):
    def __init__(self, use_contiguous_gn: bool = False):
        GN = ContiguousGroupNorm if use_contiguous_gn else partial(nn.GroupNorm, pytorch_compatible=True)
        ...
```

Default to standard `nn.GroupNorm`. Easy revert = flip flag.

### Benchmark & validate

```bash
uv run python scripts/run_research_experiment.py
uv run python scripts/score_experiment.py
```

**Fidelity gate**: max_abs_error < 5e-3 per tensor vs golden.npz. fg_final has only 8.2% headroom (per exp31) -- monitor closely.

**Rollback**: If fidelity fails or no latency improvement -> revert to standard GN, document finding.

---

## Acceptance Criteria

- [ ] Metal trace at 1024 captured, GN copy overhead quantified
- [ ] Decision gate: proceed/skip custom kernel based on >=3% threshold
- [ ] If proceeding: `ContiguousGroupNorm` implemented with fp32 accumulation
- [ ] Fidelity: all 7 tensors within 5e-3 of golden reference
- [ ] Latency: measurable improvement (>=2% end-to-end)
- [ ] Fallback: `use_contiguous_gn=False` reverts to standard GN
- [ ] Weight loading: compatible with existing checkpoint (no key remapping)
- [ ] Experiment logged to `research/experiments.jsonl`
- [ ] Compound note written if finding is reusable

## Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| GN copies <3% at 1024 | Medium | Kills Exp B | Exp A validates first |
| bf16 accumulation precision | High if naive | Fidelity failure | fp32 accumulation mandatory |
| Custom kernel fusion barriers | Medium | Offset savings | Measure dispatch overhead in isolation |
| metal_kernel can't do reductions | Medium | Forces partial-fusion | Variant 2 fallback uses mx.mean/var |
| Fidelity regression from different reduction order | Low | Blocks merge | fg_final has 8.2% headroom, monitor |

## Open Questions

- `mx.fast.metal_kernel()` threadgroup shared memory API -- supported?
- Does compiled refiner already eliminate any GN copies? (A.2 answers this)
- Partial-fusion variant (mx.mean/var + fused normalize kernel): is the normalize-only kernel fast enough to offset the separate mean/var dispatch?
- At what reduction size does manual mean/var become competitive with transpose+layer_norm?

## References

- Brainstorm: `docs/brainstorms/2026-03-12-contiguous-groupnorm-brainstorm.md`
- Metal trace analysis: `research/compound/metal_trace_512_findings.md`
- Frontier handoff: `research/compound/next_frontiers_handoff.md`
- GroupNorm fidelity failure (exp32): `research/experiments.jsonl`
- Dilated conv kernel experiment: `research/compound/dilated_conv_kernel_experiment.md`
- MLX GroupNorm source: `.venv/lib/python3.12/site-packages/mlx/nn/layers/normalization.py:199-237`
