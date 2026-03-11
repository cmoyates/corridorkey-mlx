# Deep Research Prompt — Breaking the 120ms Plateau

Use this prompt with Claude deep research, Perplexity Pro, or similar tools.

---

## Context

I have a **Hiera vision transformer** (hiera_base_plus_224, 24 blocks) ported to **Apple MLX** for inference on Apple Silicon (M3 Max). The model does green-screen matting: RGB + alpha hint input → alpha matte + foreground output.

**Architecture:**
- Backbone: Hiera (24 blocks, 4 stages, channels [112, 224, 448, 896], 144 Linear layers, 48 LayerNorm)
- Attention: MaskUnitAttention with windowed attention, already using `mx.fast.scaled_dot_product_attention`
- Decoder: SegFormer-style dual heads (alpha + foreground), Linear projections + bilinear upsample + BN fusion
- Refiner: 4-block dilated CNN (dilation 1,2,4,8) with GroupNorm+GELU, operates at full resolution
- Compilation: `mx.compile` wrapping the full forward pass (fixed 512x512 shape)
- Precision: backbone fp32, decoders bf16, refiner fp16 weights (fp32 sigmoid)

**Current performance (512x512, M3 Max, MLX v0.31.0):**
- Latency: 119.9ms median, 120.4ms p95
- Peak memory: 2143 MB
- Initial baseline was 156ms — already optimized 23%

**What we've tried (all within noise or failed fidelity):**
- Refiner fp16 activations → fidelity failure (max_abs > 5e-3)
- Cache limit 256MB → +15% latency regression
- Decoder bf16 → no speedup, +5% memory
- Backbone 4-bit quantize → no speedup (dim=112 at stage 0 blocks standard nn.quantize)
- Backbone 8-bit quantize → marginal (<1%)
- Fused decoder pair → no speedup
- Backbone fp16 weights+input → no change
- Wired limit tuning → already applied, current best

## Research Questions

### 1. MLX Compilation Depth
The full forward is wrapped in `mx.compile()`. **How deep does MLX's trace-based compiler actually fuse?** Specifically:
- Does it fuse across Python control flow (if/else, loops)?
- Does it see through `nn.Module.__call__` boundaries?
- Are there known fusion barriers (e.g., `mx.eval`, reshape, transpose, indexing)?
- Is there a way to inspect the compiled graph to identify fusion gaps?
- How does `mx.compile` interact with `mx.fast.scaled_dot_product_attention` — does it fuse pre/post attention ops with the SDPA kernel?

### 2. MLX Metal Kernel Performance on Vision Transformers
- What are the known bottlenecks for ViT inference on MLX vs PyTorch/MPS?
- Are there MLX GitHub issues or discussions about ViT-specific optimizations?
- How does MLX's Conv2d performance compare to its Linear performance? (Refiner is all Conv2d with dilation)
- Does `mx.fast.metal_kernel()` have examples of fusing conv+norm+activation? What's the realistic speedup?

### 3. Memory Bandwidth Analysis
At 512x512, the model likely isn't compute-bound — it's **memory bandwidth bound**.
- What's the theoretical minimum inference time given M3 Max bandwidth (400 GB/s) and model size (~180MB weights)?
- How can I estimate arithmetic intensity for each component (backbone, decoder, refiner)?
- Are there MLX-specific techniques to improve memory access patterns for NHWC convolutions?

### 4. Hiera-Specific Optimizations
- Has anyone published MLX or Metal optimizations for Hiera specifically?
- The MaskUnitAttention uses windowed attention with pooling — are there known tricks for this pattern on Apple GPU?
- Hiera's unrolling (variable spatial dims per stage) — does this interact badly with `mx.compile`?

### 5. Dilated Convolution on Apple GPU
The refiner uses dilated convolutions (dilation 1,2,4,8) which are **excluded from MLX's implicit GEMM** (PR #3147 — requires no input dilation). This means they fall back to explicit im2col.
- Are there alternative implementations of dilated conv that perform better on Metal?
- Could atrous spatial pyramid pooling (ASPP) be restructured to avoid dilation?
- Is there a way to use `mx.fast.metal_kernel()` for dilated conv specifically?

### 6. Beyond Single-Forward Optimization
- **Async/streaming**: Can MLX overlap CPU preprocessing with GPU inference? (`mx.stream()` API)
- **Graph caching**: Does `mx.compile` cache the Metal pipeline state objects? How much does the second call save vs first?
- **Memory pooling**: Are there undocumented MLX APIs for memory arena management beyond `set_cache_limit`?
- **Multi-stream execution**: Can backbone and decoder run on different GPU command queues?

### 7. Alternative Approaches
- **Distillation**: Has anyone distilled Hiera to a smaller backbone for matting tasks?
- **ONNX/CoreML**: Would converting to CoreML and using ANE provide a step change? (M3 Max has 16-core Neural Engine)
- **Structured pruning**: Are there MLX utilities for structured pruning of attention heads or MLP dimensions?
- **Dynamic resolution**: Running backbone at 384 instead of 512 — what's the quality/speed tradeoff for matting?

### 8. MLX Internals
- How does MLX schedule Metal command buffers? Is there a way to reduce dispatch overhead?
- What's the overhead of Python-level tensor operations between compiled regions?
- Does MLX support persistent kernels or occupancy hints?
- Are there Metal Performance Shaders (MPS) that MLX doesn't use but could benefit from?

## Desired Output Format

For each finding:
1. **Source** (URL, paper, GitHub issue/PR)
2. **Applicability** — directly applicable / needs adaptation / concept only
3. **Expected impact** — latency %, memory %, or qualitative
4. **Implementation complexity** — trivial / medium / hard
5. **Fidelity risk** — none / low / medium / high
