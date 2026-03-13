# Video Upstream Research + Next Steps — Brainstorm

**Date:** 2026-03-13
**Context:** Upstream research complete. Skip2 rejected. Single-frame at plateau. Deciding video optimization roadmap.

## What We Learned (Upstream Research)

### Trained temporal modules won't help us
- RVM (ConvGRU), MatAnyone (memory bank), SAM2 (cross-frame attention) — all require retraining
- No retrofit path for CorridorKey's frozen weights

### 4ch input is the fundamental blocker for backbone skip
- DFF/Accel-style caching works for 3ch RGB models
- Our 4ch (RGB + alpha hint) means cached features carry stale spatial guidance
- Confirmed empirically: skip2 had 0.996 max_abs error on fast-motion frames
- **Partial feature reuse** (recompute early layers, cache deep) is an open question worth exploring later

### Error thresholds in the wild
- **W8A8 quantization**: <1% SAD change — safe
- **W4A8**: 77% SAD increase — quality cliff
- **Precision changes** (FP16/BF16): max_abs 5e-3 is appropriate
- **Algorithmic changes** (temporal, interpolation): need perceptual metrics (PSNR, SSIM, dtSSD) — max_abs breaks down when errors are spatially localized
- **Alpha 2-5x more sensitive than foreground** in practice
- **Production VFX**: dtSSD ~1.3 at 1080p considered good; PSNR >30dB acceptable

### Promising training-free techniques
- **Temporal EMA blending**: ~0 compute, eliminates flicker — trivial win
- **Async CPU-GPU pipeline**: 20-30% throughput from I/O overlap
- **Run-Length Tokenization** (NeurIPS 2024): 35% gain, 0.1% accuracy drop — complex with Hiera's windowed attention
- **Partial stem decomposition**: separate RGB/hint encoding paths — speculative, needs investigation

## Key Decisions

### 1. Two-tier fidelity threshold system — ADOPTED
Current max_abs < 5e-3 is correct for precision/numerical changes but wrong for algorithmic/temporal changes where errors concentrate at motion boundaries.

**Tier 1 (precision changes):** max_abs < 5e-3 per tensor vs golden (unchanged)
**Tier 2 (algorithmic/temporal changes):** PSNR, SSIM, dtSSD with separate alpha/fg thresholds

Benchmark spec to be updated.

### 2. Temporal EMA blending — DO NEXT (quick win)
Trivial implementation, ~0 compute cost. Only proceed if zero fidelity degradation on static frames.

### 3. Async CPU-GPU pipeline — DO NEXT
20-30% throughput gain from overlapping I/O with inference. Training-free, no quality impact.

### 4. Run-Length Tokenization — PLAN LATER
High potential (35-40% backbone throughput) but complex integration with Hiera mask units. Worth a deep investigation pass before committing.

### 5. Partial stem decomposition (RGB/hint split) — PLAN LATER
Speculative: if Hiera's stem conv (4ch→112) can be decomposed to separate RGB encoding from hint encoding, partial feature reuse becomes viable. Needs investigation.

## Updated Implementation Order

| Phase | What | Expected Gain | Status |
|-------|------|---------------|--------|
| V0 | Baseline measurement | — | DONE |
| V1 | Temporal EMA blending | Flicker elimination | **NEXT** |
| V2 | Async frame pipeline | 20-30% throughput | **NEXT** |
| ~~V2~~ | ~~Backbone skip~~ | ~~1.5-2x~~ | **REJECTED** (4ch input) |
| V3 | Adaptive refiner tile skip | +10-20% | Planned |
| V4 | Run-Length Tokenization | +35-40% backbone | Plan later |
| V5 | Partial stem decomposition | Enables feature reuse | Plan later |

## Deep Research (Running)

User is running a deep research pass covering:
- Feature warping with guidance channels
- Decomposed temporal strategies (which Hiera layers are temporally stable?)
- Perceptual error metrics for production video matting
- Apple Silicon / MLX async patterns
- RLT implementation details for windowed attention
- Mixed-precision quantization for video

## Open Questions

- EMA blending: output-space or feature-space? Feature-space theoretically better but adds memory
- Async pipeline: does `mx.async_eval` actually overlap with CPU preprocessing on unified memory?
- Two-tier thresholds: what specific PSNR/SSIM/dtSSD values to gate on? (deep research should inform)
- Partial stem decomposition: is the 4ch→112 stem conv separable in practice?
