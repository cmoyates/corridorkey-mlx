# Per-component compile + stage_gc is a local optimum

**Context**: Metal GPU trace at 512x512 (109ms, 1.85GiB) identified top kernel costs. Systematically tested 6 optimization hypotheses against trace evidence.

**Finding**: The model's per-component compile + stage_gc architecture is optimal for M3 Max. Every monolithic-graph approach (compile_forward, export_function) is 15-20% slower because it defeats buffer reuse between backbone/decoder/refiner phases.

**Evidence**:
- compile_forward=True: 96.2ms vs 83.2ms per-component (15.6% slower)
- mx.export_function: 100.1ms vs 83.0ms (20.5% slower)
- Half-res refiner: max_abs_error 0.98 at both 0.5x and 0.75x (fidelity dead)
- GroupNorm fusion: already fused (pytorch_compatible=True uses mx.fast.layer_norm)
- Gather ops: already optimized (precomputed permutation replaces 3x reshape-transpose-reshape)
- Refiner bf16 vs fp16: fp16 1.9% faster on M3 Max

**Implication**: Further gains require framework-level changes (faster GroupNorm transpose, lower-overhead gather kernel) or algorithmic changes orthogonal to fidelity. Application-level optimization is exhausted.
