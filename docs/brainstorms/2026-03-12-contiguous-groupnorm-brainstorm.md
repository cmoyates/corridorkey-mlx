# Contiguous GroupNorm — Brainstorm

**Date:** 2026-03-12
**Status:** Python-level approaches exhausted (exp 41). Custom Metal kernel is only remaining path.
**Context:** Application-level optimization exhausted (40 experiments, 17.7% cumulative gain). Metal trace identified GroupNorm transpose copies as 6.94% of GPU time — the largest single addressable bottleneck remaining.

---

## Research Summary (Round 4 Upstream Mining)

### Current best
- **422.46ms** median @1024, 3319MB peak (1407MB with buffer env vars)
- 40 experiments logged, local optimum confirmed via Metal GPU trace

### What's been exhausted
- Precision: bf16 stages 1-3, bf16 decoder weights, deferred casts
- Layout: precomputed gather perms, einsum fusion, split QKV, pretransposed weights, concat avoidance
- Fusion: folded BN, conv1x1→linear, addmm (regressed)
- Streams: dual-stream (kept), GPU-GPU overlap (disproven on Apple Silicon)
- Memory: materialization barriers, removed stage GC, phased deletion (no effect), buffer limits (58% mem reduction)
- GEMM: stage0 K=112→128 pad (regression), wired limit (regression)
- Quantization: int8 stages 1-3 (kept)
- Custom Metal kernels: AMX-bound, can't beat im2col for conv ops
- Architecture hacks: token routing, skip attention, native GroupNorm — all failed fidelity

### Top remaining GPU time sinks (from Metal trace @512)
| Kernel | % GPU | Notes |
|--------|-------|-------|
| Gather (unroll/reroll perms) | 9.25% | Already optimized, hard to improve |
| GroupNorm transpose copies | 6.94% | **Target of this brainstorm** |
| Softmax in SDPA | 3.95% | Inside mx.fast, not app-controllable |
| Broadcast adds (residuals) | 2.99% | Fundamental compute |

### Upstream: nothing new since last mining
- CorridorKey: no new optimization PRs
- MLX 0.31.1: M5 kernel tuning, 3/5/6-bit QMV kernels. No M3 perf changes
- No distilled checkpoint (Issue #107 closed)
- Watch: PR #3120 (split-K quantized matmul, 25-30% faster) — still open
- Watch: PR #3247 (per-stream locking for concurrent inference) — still open

### Other untried ideas (lower priority)
1. **mxfp8 for backbone stage 0** — dim=112 can't use int4/int8. FP8 has different alignment, might work
2. **5-bit quantization for stages 1-3** — MLX 0.31.1 adds 3/5/6-bit QMV kernels
3. **mx.clear_cache() between stages** — targets buffer cache, different from del backbone (exp34)
4. **MLX 0.31.1 upgrade** — free, minor fixes

---

## What We're Building

Custom `ContiguousGroupNorm` that produces pytorch-compatible normalization results without the internal transpose that causes `g3_copyfloat16float16` copy kernels (6.94% GPU time).

### Problem
MLX `nn.GroupNorm(pytorch_compatible=True)` internally:
1. Reshapes input to group layout
2. Transposes for normalization
3. Normalizes (may use mx.fast.layer_norm)
4. Transposes back
5. Output is **non-contiguous** — next op (conv2d, relu) forces a Metal copy kernel

This happens **10 times** in the refiner (stem GN + 2 per ResBlock × 4 blocks).

### Why This Approach
- Single largest addressable bottleneck in Metal trace
- 6.94% = ~29ms @1024 theoretical savings
- Doesn't require MLX framework changes — can implement in application code
- Doesn't change model semantics (same math, different memory layout)
- exp32 (dropping pytorch_compatible) failed catastrophically (0.987 alpha error) — we MUST keep pytorch-compatible semantics, just avoid the non-contiguous output

### MLX GroupNorm internals (from source analysis)

The `pytorch_compatible=True` path does:
```python
# Input: (B, H, W, C) with C=64, G=8, gs=C//G=8
x = x.reshape(B, -1, G, gs)                          # (B, HW, 8, 8)
x = x.transpose(0, 2, 1, 3).reshape(B, G, -1)       # TRANSPOSE 1 → (B, 8, HW*8)
x = mx.fast.layer_norm(x, eps=eps, weight=None, bias=None)  # fused kernel
x = x.reshape(B, G, -1, gs)                          # (B, 8, HW, 8)
x = x.transpose(0, 2, 1, 3).reshape(B, *rest, C)    # TRANSPOSE 2 → (B, H, W, 64)
return weight * x + bias                              # affine
```

**Why transposes exist**: `mx.fast.layer_norm` normalizes the last dim only. To normalize spatial*channels_per_group jointly (pytorch semantics), data must be rearranged so that contiguous block is in the last dim. NHWC scatters group members across the channel dim.

**Non-pytorch path** reshapes to `(B, HW*gs, G)` and normalizes axis=1 — this is **strided grouping** (group 0 = channels [0, G, 2G, ...]), semantically different from PyTorch (group 0 = channels [0..gs-1]). That's why exp32 failed fidelity.

### Benchmark results (512x512, G=8, C=64)
| Method | Time | Notes |
|--------|------|-------|
| Current (2x transpose + layer_norm) | 2.72ms | 10 calls = ~27ms |
| Manual mean/var (no transpose) | 14.29ms | 5.3x slower — no fused kernel |
| mx.compile'd current | ~10% faster | Fuses some reshape/affine ops |

### Answered questions
- **YES**: GroupNorm(pytorch_compatible=True) already uses mx.fast.layer_norm internally
- **NO**: mx.contiguous() / downstream ops don't help — the copy overhead is intrinsic to the transpose, not a downstream propagation issue
- The 6.94% is the cost of 10x transpose+layer_norm+transpose+affine in the refiner

### Possible approaches (updated)

**ELIMINATED:**
1. ~~Manual GroupNorm~~ — 5.3x slower. mx.fast.layer_norm fused kernel dominates.
2. ~~mx.fast.layer_norm per-group loop~~ — G=8 dispatch calls, worse than 2 transposes
3. ~~Force contiguous output~~ — downstream ops don't care, no measurable benefit

**VIABLE:**
4. **Custom Metal kernel via mx.fast.metal_kernel** — fused GroupNorm+ReLU in one pass. This is element-wise math (mean, var, normalize, scale, bias, relu), NOT matmul/conv, so Metal kernel CAN help here (unlike dilated conv experiments). Would eliminate both transposes AND the separate relu dispatch. One kernel for what's currently: transpose → layer_norm → transpose → affine → relu. **Highest potential, medium effort.**

5. **mx.compile the entire RefinerBlock** — let MLX fuse transpose with surrounding conv/relu ops. Low effort, ~10% GN speedup (not full 6.94%). Worth trying first as a quick win before custom kernel.

### Key Decisions
- Must match pytorch_compatible GroupNorm semantics exactly (fidelity gate)
- Refiner has 10 GroupNorm instances (stem + 2 per ResBlock × 4) — cost multiplied by 10
- GroupNorm params: groups=8, channels=64, so C//G=8 (small group size = good for threadgroup tiling)
- Custom Metal kernel is element-wise — no AMX dependency, viable path

### Open Questions
- mx.fast.metal_kernel: can we do threadgroup-level reductions (mean/var) efficiently?
- mx.compile on RefinerBlock: does it already fuse the transpose+affine? Need to measure
- Fused GN+ReLU kernel: what's the expected speedup over 2x transpose + layer_norm + relu separately?
- At C//G=8, is there enough work per group to saturate GPU threads?

---

## Next: /workflows:plan when approach is selected

Recommended order:
1. Quick win: mx.compile RefinerBlock (low effort, ~10% GN speedup)
2. High potential: custom Metal kernel for fused GroupNorm+ReLU (medium effort, up to 6.94% total GPU savings)
