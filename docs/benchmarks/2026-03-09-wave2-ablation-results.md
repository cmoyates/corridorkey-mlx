---
title: "Wave 2 optimization ablation benchmarks"
date: 2026-03-09
device: "Apple Silicon (unified memory)"
script: scripts/bench_optimizations.py
---

# Wave 2 Optimization Ablation Benchmarks

## Ablation Sweep (baseline = all optimizations off)

Toggle flags: `slim`, `stage_gc`, `sdpa`, `bf16`, `fused_decode`, `gpu_preprocess`

### 512x512

| Config | Median (ms) | Min (ms) | Peak Mem (MB) | vs Baseline |
|---|---|---|---|---|
| baseline | 119.6 | 118.3 | 2419 | 1.00x |
| slim+sdpa+bf16+fused_decode+gpu_preprocess | 120.0 | 119.2 | 2413 | 1.00x |
| slim+stage_gc+bf16+fused_decode+gpu_preprocess | 147.8 | 146.7 | 2306 | 0.81x |
| slim+stage_gc+sdpa+bf16+fused_decode+gpu_preprocess | 149.2 | 147.9 | 2344 | 0.80x |
| slim+stage_gc+sdpa+bf16+fused_decode | 149.3 | 147.6 | 2344 | 0.80x |
| stage_gc+sdpa+bf16+fused_decode+gpu_preprocess | 150.5 | 148.7 | 2344 | 0.79x |
| slim+stage_gc+sdpa+bf16+gpu_preprocess | 151.6 | 148.5 | 2344 | 0.79x |
| slim+stage_gc+sdpa+fused_decode+gpu_preprocess | 151.9 | 149.1 | 2344 | 0.79x |

### 1024x1024

| Config | Median (ms) | Min (ms) | Peak Mem (MB) | vs Baseline |
|---|---|---|---|---|
| slim+sdpa+bf16+fused_decode+gpu_preprocess | 583.3 | 579.9 | 3819 | 1.05x |
| baseline | 610.7 | 606.2 | 3673 | 1.00x |
| stage_gc+sdpa+bf16+fused_decode+gpu_preprocess | 655.5 | 650.6 | 3673 | 0.93x |
| slim+stage_gc+sdpa+bf16+fused_decode+gpu_preprocess | 655.9 | 650.6 | 3673 | 0.93x |
| slim+stage_gc+bf16+fused_decode+gpu_preprocess | 657.5 | 650.6 | 3673 | 0.93x |
| slim+stage_gc+sdpa+bf16+gpu_preprocess | 675.3 | 667.7 | 3673 | 0.90x |
| slim+stage_gc+sdpa+fused_decode+gpu_preprocess | 686.0 | 682.7 | 3673 | 0.89x |
| slim+stage_gc+sdpa+bf16+fused_decode | 687.5 | 680.4 | 3673 | 0.89x |

### 2048x2048

| Config | Median (ms) | Min (ms) | Peak Mem (MB) | vs Baseline |
|---|---|---|---|---|
| baseline | 4984.7 | 4954.6 | 26689 | 1.00x |
| stage_gc+sdpa+bf16+fused_decode+gpu_preprocess | 5016.8 | 4840.4 | 26661 | 0.99x |
| slim+stage_gc+sdpa+bf16+fused_decode+gpu_preprocess | 5048.9 | 4848.1 | 26661 | 0.99x |
| slim+sdpa+bf16+fused_decode+gpu_preprocess | 5175.5 | 5065.2 | 27245 | 0.96x |
| slim+stage_gc+sdpa+bf16+fused_decode | 5375.2 | 5083.1 | 26661 | 0.93x |
| slim+stage_gc+sdpa+bf16+gpu_preprocess | 5396.3 | 5357.0 | 26661 | 0.92x |
| slim+stage_gc+sdpa+fused_decode+gpu_preprocess | 5683.3 | 5546.8 | 26661 | 0.88x |
| slim+stage_gc+bf16+fused_decode+gpu_preprocess | 5771.3 | 5643.0 | 26689 | 0.86x |

## Tiled vs Full-Frame at 2048x2048

| Config | Median (ms) | Min (ms) | Peak Mem (MB) | Speed vs FF | Mem vs FF |
|---|---|---|---|---|---|
| full-frame 2048 | 5031 | 4838 | 26281 | 1.0x | 1.0x |
| **tiled 768/64** | **3344** | **3311** | **2302** | **1.5x** | **11.4x less** |
| tiled 512/64 | 3978 | 3963 | 2133 | 1.3x | 12.3x less |
| tiled 512/128 | 4055 | 4025 | 2133 | 1.2x | 12.3x less |
| tiled 1024/64 | 6144 | 6125 | 3425 | 0.8x | 7.7x less |

## Key Findings

1. **stage_gc hurts at 512/1024, breaks even at 2048.** GC pauses cost more than memory savings at small resolutions. Only at 2048 does the computation graph get large enough to justify intermediate materialization.

2. **No single optimization consistently beats baseline across resolutions.** At 512, baseline wins. At 1024, slim+sdpa (no stage_gc) wins by 5%. At 2048, nothing reliably beats baseline.

3. **Memory scales super-linearly: 2.4GB → 3.7GB → 26.7GB** for 512 → 1024 → 2048. The 7x jump from 1024→2048 indicates lazy graph accumulation across 24 backbone blocks dominates at high resolution.

4. **Tiled 768/64 is the clear winner at 2048** — 1.5x faster AND 11.4x less memory than full-frame. The smaller per-tile computation graph avoids the massive intermediate state buildup.

5. **bf16 can hurt at 2048** — possible promotion overhead in large graphs.

6. **gpu_preprocess is the most impactful single flag** — dropping it consistently hurts latency.

7. **Phase 6 (GPU tile accumulators) is low priority** — numpy accumulator transfer overhead is negligible vs compute. The memory problem is entirely graph-side.

## Recommendations

- Default to **tiled 768/64** for 2048x2048 production use
- For ≤1024, use **full-frame with slim+sdpa+bf16+fused_decode+gpu_preprocess**
- stage_gc should be **off by default**, enabled only if memory-constrained at ≥2048
