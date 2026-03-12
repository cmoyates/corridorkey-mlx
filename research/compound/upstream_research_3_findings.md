# Upstream Research Round 3 — MLX APIs, Examples, Community Forks

Researched 2026-03-12. Sources: MLX framework source, MLX examples (SD/FLUX/SAM), CorridorKey upstream, CorridorKey-Engine, CorridorKey-Marcel, EZ-CorridorKey.

Current best: 422ms @1024, 3319MB peak. 28 experiments logged.

---

## High-Signal Findings

### 1. GroupNorm -> mx.fast.layer_norm fusion (MLX framework)
- `nn.GroupNorm` decomposes to ~5 Python ops; `mx.fast.layer_norm` is a single fused Metal kernel
- Rewrite: reshape (B,H,W,C) -> (B*G, H*W, C//G), apply fast.layer_norm, reshape back
- Refiner has 10 GroupNorm calls in hot path
- Classification: mlx-portable
- Risk: reshape overhead may negate fusion gain

### 2. mx.export_function — AOT compiled graph (MLX framework)
- Exports compiled graph to disk, skips Python graph-building on reload
- Fixed 1024 shape makes this viable
- Could save ~20ms Python overhead per inference
- Classification: mlx-portable

### 3. MLX_MAX_MB_PER_BUFFER env var (MLX PR #3192)
- Controls Metal command buffer commit frequency
- Lower = more frequent commits (lower peak memory, more dispatch overhead)
- Higher = larger batches (less overhead, higher peak memory)
- Never swept in any experiment
- Classification: mlx-portable

### 4. FP8 mx.qqmm(mode="mxfp8") — activation quantization (MLX framework)
- Dynamic quantization of both activations AND weights during matmul
- No experiment has tried activation quantization yet
- Less aggressive than 4-bit weight quant but covers activations too
- Unknown: whether M3 Max has hardware FP8 or emulates
- Classification: mlx-portable

### 5. mx.depends — lightweight graph dependencies (MLX framework)
- Inserts dependency edges without materialization
- Could replace mx.async_eval barriers with zero-copy deps
- Classification: mlx-portable

### 6. Phased model deletion (MLX examples — SD/FLUX pattern)
- Both SD and FLUX explicitly `del` submodels between pipeline stages
- del backbone after feature extraction, del decoder before refiner
- Zero-cost memory optimization
- Classification: mlx-portable

### 7. 1x1 Conv2d -> Linear replacement (MLX examples — SD/CLIP pattern)
- Decoder has 1x1 convolutions for channel projection
- Linear may have better kernel dispatch than Conv2d(1,1)
- SD and CLIP examples do this conversion at load time
- Classification: mlx-portable

### 8. mx.metal.start_capture — GPU trace profiling (MLX framework)
- Captures Metal GPU trace for Xcode Instruments analysis
- Shows actual kernel execution times, memory bandwidth, occupancy
- NO experiment has identified the actual bottleneck kernel yet
- Classification: mlx-portable (diagnostic)

### 9. Half-resolution refiner (CorridorKey PR #93)
- Run refiner at 0.5x input res, upsample result
- Distinct from decoupled backbone res (which we already have)
- Refiner is a CNN — 4x fewer pixels = ~4x faster
- Classification: mlx-portable, high impact if fidelity holds

### 10. mx.set_wired_limit — pin GPU memory (MLX framework)
- Prevents OS from paging MLX buffers
- Never tuned; could reduce latency variance
- Classification: mlx-portable

---

## Community Validations
- All forks run full FP16 (validates BF16 direction)
- 512x512 tiles + 128px overlap = community standard
- No fork modifies decoder/refiner architecture
- Decoupled backbone/refiner resolution is novel (not in any fork)

## Gaps Remaining (from program.md)
- S10: Refiner-only tiling — NOT ATTEMPTED
- S12: SDPA in Hiera — needs verification
- S15: Weight format optimization — NOT ATTEMPTED

## Closed/Disproven
- Custom Metal kernels for dilated conv (3 approaches, all slower — im2col IS fast path via AMX)
- LayerNorm fusion (already dispatched internally)
- Token routing (2 fidelity failures)
- shapeless=True compile (broken per PR #3202)

## Priority Queue
1. Metal GPU trace (diagnostic — identify actual bottleneck)
2. GroupNorm -> fast.layer_norm rewrite
3. Half-resolution refiner
4. Phased model deletion
5. 1x1 Conv -> Linear in decoder
6. mx.export_function for AOT compilation
