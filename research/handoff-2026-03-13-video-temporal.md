# Handoff: Video Temporal Optimization — 2026-03-13

## Where We Are

Single-frame optimization exhausted (42 experiments, 422ms→plateau @1024). Video pipeline infrastructure exists (`src/corridorkey_mlx/inference/video.py`). Backbone skip (V2) rejected — 4ch input (RGB + hint) means cached features carry stale spatial guidance.

### Current Baselines

| Metric | Single-frame | Video (37 frames) |
|--------|-------------|-------------------|
| Median latency | 433.8ms | 476.6ms/frame |
| P95 | 435.0ms | 496.3ms |
| Effective FPS | 2.31 | 1.79 |
| Peak memory | 3575MB | 3508MB |

The ~43ms gap = decode overhead (frame read + resize). Async pipeline should reclaim this.

## Decisions Made

1. **Two-tier fidelity thresholds adopted** — see `research/benchmark_spec.md`
   - Tier 1 (precision): max_abs < 5e-3 (unchanged)
   - Tier 2 (algorithmic/temporal): alpha PSNR >35dB, fg PSNR >33dB, SSIM >0.97, dtSSD <1.5

2. **Implementation order:**
   - **V1: Temporal EMA blending** — NEXT, trivial, ~0 compute cost
   - **V2: Async CPU-GPU pipeline** — NEXT, 20-30% throughput from I/O overlap
   - V3: Adaptive refiner tile skip — planned
   - V4+: RLT, partial feature reuse — plan later

## Completed

### Tier 2 metrics — DONE
- [x] PSNR, SSIM (windowed Wang 2004), dtSSD added to `bench_video.py`
- [x] Pure numpy implementation, no new deps
- [x] Reports Tier 1 + Tier 2 pass/fail against benchmark_spec thresholds
- [x] Runs for all modes including V0 baseline (establishes reference values)

### V1: Temporal EMA blending — DONE (fails fidelity gate)
- [x] Output-space EMA in `VideoProcessor._postprocess_frame()`
- [x] CLI: `--ema-alpha 0.7`, `--ema-sweep 0.6 0.7 0.8`
- **Result: ALL α values fail fidelity on motion video.**

| α | alpha PSNR | SSIM | dtSSD | Tier 1 | Tier 2 | Verdict |
|-----|-----------|------|-------|--------|--------|---------|
| 0.6 | 21.7dB | 0.924 | 4.04 | FAIL | FAIL | reject |
| 0.7 | 24.3dB | 0.941 | 3.13 | FAIL | FAIL | reject |
| 0.8 | 27.9dB | 0.953 | 2.15 | FAIL | FAIL | reject |
| 0.9 | 34.0dB | 0.966 | 1.10 | FAIL | FAIL | reject |
| 0.95| 40.2dB | 0.972 | 0.55 | FAIL | PASS | reject |

- α=0.95 passes Tier 2 but fails Tier 1 (max_abs=0.059). Barely useful — 5% blending.
- Output-space EMA is fundamentally limited for motion video — introduces lag
- **Next step if pursuing: feature-space EMA or motion-adaptive α**

### V2: Async CPU-GPU pipeline — DONE (KEEP)
- [x] `mx.async_eval` + ThreadPoolExecutor decode-ahead
- [x] CLI: `--async-decode`
- **Result: 7% wall-clock improvement, zero fidelity impact**

| Mode | Wall-clock | FPS | Fidelity |
|------|-----------|-----|----------|
| V0 sync | 21.31s | 1.74 | PASS (perfect) |
| V2 async | 19.83s | 1.87 | PASS (perfect) |

- Decode overlap partially hidden (~1.5s savings over 37 frames)
- Less than expected 43ms/frame — likely because Apple Silicon unified memory
  means CPU decode and GPU inference compete for memory bandwidth
- Still a clear win with zero quality tradeoff

## What To Do Next

### V3: Adaptive refiner tile skip — planned
### V4+: RLT, partial feature reuse — plan later

## Key Research Finding: Partial Feature Reuse (future)

Deep research confirmed a viable strategy for our 4ch blocker:
- Run Hiera Stages 1-2 every frame (fresh hint re-injection)
- Cache + optically warp Stages 3-4 (semantically stable)
- Gate: cosine similarity on Stage 2 output → skip S3-S4 if high
- `VNGenerateOpticalFlow` on ANE = zero GPU cost for flow
- This is the big win if it works — but needs optical flow infra first

## Key Files

- Brainstorm: `docs/brainstorms/2026-03-13-video-upstream-research-brainstorm.md`
- Deep research: `research/compound/2026-03-13-deep-research-video-matting-optimization.md`
- Full source doc: `docs/Video Matting Inference Optimization Techniques.md`
- Video processor: `src/corridorkey_mlx/inference/video.py`
- Video benchmark: `scripts/bench_video.py`
- Benchmark spec: `research/benchmark_spec.md`
- V0 baseline artifact: `research/artifacts/video_baseline.json`

## Open Questions (answered)

- ~~EMA α parameter: output-space or feature-space first?~~ → Output-space tried, fails fidelity. Feature-space or motion-adaptive α needed.
- ~~`mx.async_eval` — does it actually overlap with CPU on unified memory?~~ → Partially. 7% wall-clock improvement (not full 43ms overlap). Unified memory bandwidth contention likely.
- Hiera S1-S2 vs S3-S4 compute split? (determines max speedup from partial reuse)
- `VNGenerateOpticalFlow` latency @1024 via PyObjC?
