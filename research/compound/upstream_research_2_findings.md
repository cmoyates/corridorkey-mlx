# Upstream Research Round 2 — Plateau-Breaking Findings

Researched 2026-03-11. Sources: MLX source, CorridorKey-Engine, ZMLX, "Writing Fast MLX" guide.

---

## Critical: LayerNorm already dispatches to mx.fast.layer_norm

**Source**: mlx/python/mlx/nn/layers/normalization.py
**Classification**: already-applied
**Relevance**: Search area #8 (layernorm-fusion) is a NO-OP. nn.LayerNorm.__call__ directly calls mx.fast.layer_norm. No experiment needed.

## Critical: mx.compile Fusion Barriers

**Source**: mlx/mlx/compile.cpp, compile.rst
**Classification**: mlx-portable
**Findings**:
- mx.compile traces through nn.Module.__call__ boundaries — full forward compilation works
- Only element-wise + broadcast ops fuse into kernels
- **Reductions, matmuls, convolutions are NOT fused** — each is a separate Metal dispatch
- Max fusion depth: 11 ops
- Max arrays per fused kernel: 24
- Custom metal_kernel ops are NOT fusable with other ops
- Calling the array materializer inside compile raises an error (our stage_gc bypass when _compiled is correct)

**Relevance**: Explains why our full-forward mx.compile doesn't help much — the expensive ops (matmul, conv, SDPA) are already individual kernels. Compilation only fuses the cheap element-wise glue between them. The real bottleneck is the kernel dispatch sequence itself.

## High: mx.async_eval for Pipelined Inference

**Source**: "Writing Fast MLX" guide (awni)
**Classification**: mlx-portable
**Summary**: mx.async_eval() returns immediately, allowing CPU to build next graph while GPU executes current one. Must run in separate stream via mx.new_stream(mx.gpu).
**Relevance**: For video/batch inference, could overlap frame N+1 preprocessing with frame N GPU inference. Not useful for single-image latency but important for throughput.

## High: Operation Optimization Patterns

**Source**: "Writing Fast MLX" guide
**Classification**: mlx-portable
**Findings**:
- `x @ W.T` faster than `x @ W` for vector-matrix
- `mx.addmm` for fused a @ b + c
- Use Python scalars (not mx.array) for constants to avoid upcasting
- mx.fast functions accumulate in higher precision — don't manually upcast before calling them
- `softmax(precise=True)` instead of manual cast-softmax-cast

**Relevance**: Check if Hiera attention or decoder uses suboptimal matmul ordering. Check if any unnecessary dtype casts around mx.fast calls.

## Medium: CorridorKey-Engine Additional Techniques

**Source**: 99oblivius/CorridorKey-Engine
**Classification**: concept-only (PyTorch/CUDA specific)

New findings beyond previous research:
- **Flash Attention monkey-patching**: Squeezes num_windows dim + contiguous Q/K/V to force SDPA into FlashAttention path. MLX equivalent: we already use mx.fast.scaled_dot_product_attention.
- **torch.compile reduce-overhead mode**: Uses internal CUDA graphs. MLX equivalent: mx.compile (already applied).
- **Manual CUDA Graph capture**: For fixed input sizes. MLX equivalent: mx.compile with fixed shapes (already applied).
- **TF32 matmul precision**: Ampere+ feature. No MLX equivalent.
- **Reflective padding**: Pad to multiples of 32 before inference. Worth checking if our padding strategy is optimal.
- **Linear blend weight ramps**: Already implemented in our tiled inference.

## Medium: ZMLX Fusion Toolkit

**Source**: github.com/Hmbown/ZMLX
**Classification**: concept-only (LLM/MoE focused)
**Summary**: Triton-style kernel toolkit for MLX that fuses MoE gating+combine+SwiGLU into single Metal dispatches. Achieves +7-13% decode throughput on LLMs.
**Relevance**: The pattern (elementwise expression -> compiled Metal kernel) could apply to our refiner's GroupNorm+GELU sequences, but ZMLX doesn't support conv fusion. The elementwise API could fuse activation functions.

## Low: Quantized Conv2d (MLX Issue #2714)

**Source**: github.com/ml-explore/mlx/issues/2714
**Classification**: not-ready
**Summary**: Community PR for quantized Conv2d as drop-in replacement. Current perf: 37ms vs 0.48ms for standard Conv2d — 80x slower. Storage benefit only (33% of original).
**Relevance**: NOT viable for inference speed. Would make refiner dramatically slower.

## Eliminated Search Areas

Based on this research:
- **#8 layernorm-fusion**: Already dispatched. NO-OP.
- **#11 fused-metal-kernels**: Custom kernels break mx.compile fusion. Net effect unclear — replacing composed-but-fused element-wise ops with unfusable custom kernels may not help.
- **#16 operator-fusion**: mx.compile already fuses element-wise ops (up to depth 11). Manual fusion unlikely to beat the compiler for these ops.

## Remaining High-Value Search Areas

1. **Matmul ordering** (from "Writing Fast MLX"): Check if `x @ W.T` pattern is used in attention projections
2. **mx.addmm**: Fused add+matmul in decoder linear projections
3. **Unnecessary dtype casts**: Remove redundant astype() around mx.fast calls
4. **Reflective padding**: Ensure padding is optimal for Metal dispatch
5. **Token routing** (#9): Still the highest theoretical impact (skip 80%+ of attention)
6. **Backbone quantize stages 1-3** (#6): With custom quantize that skips dim=112 layers
