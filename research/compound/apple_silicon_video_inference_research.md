# Apple Silicon Optimization Patterns for Sustained Video Inference

Research date: 2026-03-12
Context: MLX video matting pipeline, 422ms/frame @ 1024x1024

---

## 1. Sustained GPU Throughput on M-Series Chips

### Measured TFLOPS (FP32, sustained via MPS benchmarks)

| Chip     | Peak FP32 TFLOPS | Memory BW (GB/s) | Notes |
|----------|-------------------|-------------------|-------|
| M1       | 1.36              | 68.25             | CPU = GPU perf |
| M1 Pro   | ~3.6              | 200               | |
| M1 Max   | ~7.2              | 400               | |
| M2       | 2.24              | 100               | GPU >> CPU from M2 onward |
| M2 Pro   | ~4.5              | 204.8             | |
| M2 Max   | ~9.0              | 400               | |
| M3       | 2.47              | 100               | |
| M3 Max   | ~11.0             | 400               | |
| M4       | 2.9               | 120               | |
| M4 Pro   | ~5.8              | 273               | |
| M4 Max   | ~18.4             | 546               | |
| M5 Pro   | 8.29              | 307               | |
| M5 Max   | ~20+              | 614               | chiplet thermal advantage |

Source: Apple vs Oranges HPC paper (arxiv 2502.05317)

### Thermal Throttling Under Sustained Load

- **MacBook Air M1**: loses 21% after 30min sustained (Cinebench R23)
- **MacBook Air M2**: loses 13% after 30min sustained (improved thermals)
- **MacBook Pro M2**: no throttle in same test (has active cooling fan)
- **MacBook Pro M2 Pro**: P-cores drop from 3.2GHz to 2.1GHz under sustained load = 30-40% drop
- **Mac Studio**: best sustained perf — active cooling, desktop thermal envelope
- **M5 Max**: chiplet design allows independent thermal management per die (CPU vs GPU)

**Key takeaway**: For sustained video inference, desktop form factor (Mac Studio/Mac Pro) is strongly preferred. MacBook Pro will throttle 10-20% over extended runs. MacBook Air is unsuitable for sustained workloads.

### Best Practices for Avoiding Throttle
- Use Mac Studio or Mac Pro for production video processing
- If MacBook: use elevated stand, external cooling pad
- Monitor `powermetrics` for thermal state
- Consider frame rate budgeting: accept slightly lower throughput to stay below thermal ceiling
- Batch processing with cool-down gaps (e.g., process 100 frames, pause 5s)

---

## 2. Unified Memory Architecture for Video Pipelines

### Zero-Copy in MLX

MLX's unified memory model means CPU and GPU share the same memory pool — no explicit transfers needed. Arrays created on CPU are directly accessible to GPU kernels without copying. This is a major advantage over CUDA workflows.

However, MLX has a critical memory management nuance:
- **Lazy computation**: operations build a compute graph, only executed on `mx.eval()`
- **Metal buffer cache**: MLX's Metal backend retains allocated buffers for reuse, which can accumulate during long inference sequences
- **`mx.metal.clear_cache()`**: forces release of cached buffers
- **`mx.metal.set_cache_limit(bytes)`**: caps buffer cache size

### Recommended Video Pipeline Pattern

```python
# Double-buffer pattern for frame processing
import mlx.core as mx

def process_video(frames_iterator, model):
    mx.metal.set_cache_limit(512 * 1024 * 1024)  # 512MB cache cap

    for i, frame in enumerate(frames_iterator):
        # frame is already an mx.array (zero-copy from numpy)
        result = model(frame)
        mx.eval(result)  # force computation

        # Periodic cache clearing for long videos
        if i % 100 == 0:
            mx.metal.clear_cache()

        yield result
```

### Memory Pressure Management
- Delete intermediate arrays after `mx.eval()` — once computed, refs can be freed
- **Warning**: deleting arrays before computation loses the computation
- For long videos (1000+ frames), periodic `mx.metal.clear_cache()` prevents cache bloat
- Monitor with `mx.metal.get_active_memory()` and `mx.metal.get_peak_memory()`

### True Double-Buffering (Not Currently Possible in MLX)
MLX does not support true GPU-GPU stream parallelism on Apple Silicon (confirmed in exp38). There is only one GPU command queue. Overlapping frame N inference with frame N+1 preprocessing is limited to CPU-GPU overlap only:
- CPU: load + preprocess frame N+1 (numpy/PIL)
- GPU: inference on frame N (MLX)
- This gives modest overlap but not true pipelining

---

## 3. Alternative Apple APIs Alongside MLX

### Metal Performance Shaders (MPS)
- MPS provides GPU compute kernels for matrix multiply, convolution, etc.
- MLX already uses Metal under the hood — MPS would be a parallel path, not a replacement
- MPS Graph (MPSGraph) is the lower-level graph execution engine
- **Verdict**: No evidence MPS is faster than MLX for equivalent operations. MLX's `mx.compile` fusion likely matches or exceeds manual MPS usage.

### BNNS (Basic Neural Network Subroutines)
- CPU-only framework in Accelerate
- Supports: Conv, BatchNorm, LayerNorm, GroupNorm, Pooling, Activations
- **GroupNorm IS supported** in BNNS
- Faster than GPU for small networks / small batch sizes
- **Verdict**: Not useful for our pipeline — our bottleneck is large tensor operations where GPU dominates. BNNS GroupNorm on CPU would be slower than MLX GPU GroupNorm.

### vImage
- CPU-based image processing using SIMD/vector instructions
- Excellent for: resize, color conversion, format conversion, histogram operations
- Very power-efficient (less battery drain than GPU for simple transforms)
- **Verdict**: Potentially useful for pre/post processing (frame decode, resize, color space conversion) but since MLX already handles these on GPU with zero-copy, benefit is marginal. Would only help if CPU preprocessing could overlap with GPU inference.

### Accelerate Framework
- BLAS, LAPACK, vDSP on CPU
- Already used by numpy under the hood on macOS
- **Verdict**: No direct benefit for MLX pipeline.

---

## 4. CoreML vs MLX for Video Inference

### ANE (Apple Neural Engine) Characteristics
- **Designed for**: fixed-shape convolution, image classification, segmentation
- **Supported ops**: Conv2D, GroupNorm, BatchNorm, LayerNorm, pooling, activations
- **Critical limitation**: requires fixed input shapes (or up to 128 enumerated shapes)
- **Data format**: channels-first 4D tensors (B, C, 1, S) — different from MLX's NHWC
- **M4 ANE**: ~38 TOPS (INT8), significant compute for conv-heavy models

### Could the Refiner Run on ANE?
The refiner is a CNN (stem + 4 dilated ResBlocks + 1x1 conv) with GroupNorm — architecturally well-suited for ANE:
- All ops (Conv2D, GroupNorm, ReLU) are ANE-supported
- Fixed input shape (always img_size x img_size x 7)
- However: requires CoreML conversion, fixed shapes, channels-first layout
- **Orion** project shows direct ANE programming is possible but uses private APIs

### Hybrid CoreML + MLX Pipeline
- No established pattern exists for this
- Would require: CoreML model for refiner, MLX model for backbone
- Data transfer between frameworks: CoreML outputs -> numpy -> MLX arrays (or vice versa)
- The transfer overhead may negate ANE speed gains
- **SqueezeBits' Yetter Engine** demonstrates disaggregated ANE+GPU inference for LLMs, but no vision model examples

### Practical Assessment
- CoreML compilation adds significant startup overhead
- ANE scheduling is opaque — CoreML may silently fall back to GPU/CPU
- For a 422ms/frame pipeline, the refiner is ~50ms — even if ANE halved it, saving 25ms (6% total) is not worth the complexity
- **Verdict**: Not worth pursuing unless backbone+decoder can also move to ANE

---

## 5. Realistic Performance Targets

### Current Baseline
- 422ms/frame @ 1024x1024
- ~2.4 FPS

### Estimated Performance by Hardware (1024x1024 video matting)

| Chip | Est. ms/frame | Est. FPS | Notes |
|------|---------------|----------|-------|
| M1 Pro | ~600-700 | ~1.5 | Memory BW limited |
| M2 Max | ~350-400 | ~2.5-2.9 | Good BW, decent compute |
| M3 Max | ~280-350 | ~2.9-3.6 | Better GPU efficiency |
| M4 Pro | ~350-450 | ~2.2-2.9 | Similar to M2 Max |
| M4 Max | ~200-280 | ~3.6-5.0 | Best current option |
| M5 Max | ~150-220 | ~4.5-6.7 | Chiplet + neural accelerators |

### Is Real-Time (24fps = 41.6ms) Achievable?

**No, not at 1024x1024 on any current Apple Silicon.**

- 24fps requires ~42ms/frame
- Current best case (M4 Max) is ~200-280ms — 5-7x too slow
- Even M5 Max unlikely to close this gap

### Resolution Tradeoffs in Production

| Resolution | Pixels | Relative Cost | Use Case |
|------------|--------|---------------|----------|
| 512x512 | 262K | 1x (baseline) | Preview, real-time on M3+ |
| 768x768 | 590K | ~2.3x | Good quality, near-real-time on M4 Max |
| 1024x1024 | 1049K | ~4x | High quality, offline processing |
| 2048x2048 | 4194K | ~16x | Maximum quality, batch only |

### What Production Pipelines Actually Do
- **Apple's own segmentation**: runs at 512x512, under 10ms on iPhone ANE
- **RobustVideoMatting**: targets 512x288 for real-time on mobile
- **BiRefNet**: 17 FPS at 1024x1024 on RTX 4090 (Apple Silicon ~3-5x slower)
- **Common pattern**: run backbone at lower res, upsample + refine at full res (which is what our decoupled resolution already does)

### Actionable Targets for This Pipeline
1. **Offline processing**: 422ms is acceptable. Optimize to ~300ms = 30% improvement goal
2. **Near-real-time preview**: Run at 512x512 (~100ms target), display upscaled
3. **Production video**: Accept 3-5 FPS at 1024x1024, parallelize with frame I/O
4. **Best hardware path**: M4 Max or wait for M5 Max

---

## Key Takeaways

1. **Thermal**: Use desktop hardware (Mac Studio) for sustained video inference. MacBook Pro loses 10-20% under sustained load.
2. **Memory**: MLX's unified memory is already optimal — focus on cache management (`set_cache_limit`, periodic `clear_cache`) for long videos.
3. **No GPU parallelism**: Apple Silicon has one GPU command queue. True double-buffering is impossible. Only CPU-GPU overlap helps.
4. **ANE is not worth it**: Refiner could theoretically run on ANE, but savings (~25ms) don't justify CoreML conversion complexity.
5. **Real-time at 1024x1024 is impossible** on current Apple Silicon. Production pipelines use 512x512 for real-time.
6. **Best optimization path**: Continue MLX GPU optimizations (mx.compile fusion, quantization, decoupled resolution). Hardware upgrades (M4 Max -> M5 Max) provide the biggest jumps.

---

## Sources

- [Apple vs Oranges: M-Series HPC Evaluation](https://arxiv.org/html/2502.05317v1)
- [MLX Benchmarking on Apple Silicon](https://arxiv.org/html/2510.18921v1)
- [Orion: Programming Apple's Neural Engine](https://arxiv.org/html/2603.06728)
- [Disaggregated Inference on Apple Silicon (SqueezeBits)](https://blog.squeezebits.com/disaggregated-inference-on-apple-silicon-npu-prefill-and-gpu-decode-67176)
- [MLX Unified Memory Docs](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html)
- [MLX Lazy Evaluation Docs](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html)
- [Apple Neural Engine Transformer Deployment](https://machinelearning.apple.com/research/neural-engine-transformers)
- [ANE Supported Operations (hollance)](https://github.com/hollance/neural-engine/blob/master/docs/supported-devices.md)
- [BNNS vs MPSCNN Comparison](https://machinethink.net/blog/apple-deep-learning-bnns-versus-metal-cnn/)
- [Apple Fast Salient Object Segmentation](https://machinelearning.apple.com/research/salient-object-segmentation)
- [vImage Documentation](https://developer.apple.com/documentation/accelerate/vimage)
- [MLX Metal Cache Management (PR #390)](https://github.com/ml-explore/mlx/pull/390)
- [MLX Memory Discussion (#742)](https://github.com/ml-explore/mlx/issues/742)
- [M2 Pro Thermal Throttling](https://smith6612.me/2023/10/26/apple-m2-pro-still-thermal-throttling/)
- [M5 GPU Neural Accelerators (Apple ML Research)](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
- [WWDC25: Get Started with MLX](https://developer.apple.com/videos/play/wwdc2025/315/)
