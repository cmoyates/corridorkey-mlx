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

## What To Do Next

### V1: Temporal EMA blending
- Add EMA smoothing to `VideoProcessor._process_loop()`
- Blend on output-space alpha/fg: `out_t = α * current + (1-α) * prev` where α ∈ [0.6, 0.8]
- Gate: ONLY proceed if zero fidelity degradation on static frames (Tier 2 metrics)
- Feature-space EMA is theoretically better but adds memory — try output-space first
- Measure: flicker reduction (dtSSD improvement), verify no lag on fast motion

### V2: Async CPU-GPU pipeline
- Overlap frame N+1 decode with frame N GPU inference
- Use `mx.async_eval` + double-buffered frame loading
- The 43ms decode overhead should be fully hidden
- No quality impact — pure scheduling optimization

### V1 + V2 need Tier 2 metric implementation
- `bench_video.py` needs PSNR, SSIM, dtSSD computation
- PSNR/SSIM: per-frame vs V0 reference
- dtSSD: temporal derivative comparison across frame pairs
- Partition metrics: solid core vs semi-transparent boundary (stretch goal)

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

## Open Questions

- EMA α parameter: output-space or feature-space first?
- `mx.async_eval` — does it actually overlap with CPU on unified memory?
- Hiera S1-S2 vs S3-S4 compute split? (determines max speedup from partial reuse)
- `VNGenerateOpticalFlow` latency @1024 via PyObjC?
