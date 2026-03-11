# MLX Framework Optimization Findings

Researched 2026-03-11. MLX v0.31.0 on Darwin 25.3.0 (M3 Max).

---

## Actionable

### nn.quantize() -- 8-bit/4-bit Linear layers
- `nn.quantize(model, bits=8)` quantizes all Linear+Embedding layers
- Supports: affine (int4/int8), nvfp4, mxfp8
- Conv2d is NOT quantizable -- no QuantizedConv2d exists
- Backbone has 144 Linear layers (6/block x 24 blocks) -- these dominate compute
- Decoder has 4 Linear projections -- also quantizable
- Refiner is all Conv2d -- unaffected
- **CRITICAL**: default group_size=64 CRASHES on Hiera — some Linear layers have
  input_dims=112 (not divisible by 64). Shape (336,112) causes ValueError.
  MUST use group_size=16 or manually filter layers by dimension divisibility.
  Alternatively, write a custom quantize function that skips incompatible layers.

### mx.set_wired_limit(bytes) -- pin memory
- Pins Metal allocations as wired/resident, prevents OS paging to swap
- Requires macOS 15+ (we have 25.3.0)
- Default: 0 (no wiring). System limit: `sudo sysctl iogpu.wired_limit_mb=<size>`
- Expected: p95 latency reduction from less variance

### mx.set_cache_limit(bytes) -- buffer reuse pool
- Controls how much freed GPU memory MLX retains for reuse
- Default: same as memory_limit (system RAM)
- Tight limit: lower peak memory, more allocation overhead
- Loose limit: faster buffer reuse, higher peak footprint

### mx.fast.layer_norm -- fused kernel
- Single Metal dispatch for layer normalization
- Question: does nn.LayerNorm already dispatch to this?
- If not, 48 LayerNorm calls in backbone could benefit from manual swap
- Numerically equivalent -- zero fidelity risk

### mx.depends() -- explicit dependency control (v0.30.0+)
- Creates explicit evaluation dependencies without mx.eval() sync points
- Could enable staged eval: force backbone features evaluated before decoder starts
- Unknown if works inside mx.compile-d graphs

## Informational (already benefiting)

### Two-pass SDPA (PR #3023, v0.30.4+)
- Faster scaled_dot_product_attention with improved thread-group locality
- Precision fix in v0.31.0 (PR #3119): scale factor no longer downcast to bf16

### Split-K GEMM tuning (PR #3087, v0.30.4+)
- Up to 26x speedup for sequential matmuls with large K dimensions
- Automatic dispatch -- no code changes needed

### Implicit GEMM for Conv2d (PR #3147)
- Computes im2col on-the-fly without materializing full matrix
- Dispatch condition: channels must be 16-aligned AND no input dilation
- Dilated convs in refiner (dilation 1,2,4,8) disqualified -- explains memory overhead

## Not actionable now

### mx.fast.metal_kernel -- custom fused kernels
- JIT Metal shaders, reported 8-40x over composed ops
- High effort but could fuse conv+GN+GELU in refiner blocks

### Neural Accelerator (M5+ only, v0.30.0+)
- Dispatches matmul to ANE on M5 chips. Not available on M3 Max.

### FFT-based convolution (Issue #811)
- Open enhancement request, still unimplemented
- Would benefit dilated refiner convs
