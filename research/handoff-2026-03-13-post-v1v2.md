# Handoff: Post V1/V2 — Video Temporal Optimization — 2026-03-13

## Where We Are

45 experiments total. Single-frame plateau @422ms/1024. Video pipeline has async decode (V2, +7% FPS). EMA blending (V1) rejected — output-space approach fails fidelity on motion video.

### Current Baselines

| Mode | Wall-clock (37fr) | FPS | Peak mem | Fidelity |
|------|-------------------|-----|----------|----------|
| V0 sync | 21.31s | 1.74 | 3508MB | PASS |
| V2 async | 19.83s | 1.87 | 3508MB | PASS |

### Tier 2 Metrics — Now Implemented

`bench_video.py` computes PSNR, SSIM (windowed Wang 2004), dtSSD — all pure numpy, no deps. Reports Tier 1 + Tier 2 pass/fail per benchmark_spec.md thresholds:

| Metric | Threshold |
|--------|-----------|
| alpha PSNR | >35dB |
| fg PSNR | >33dB |
| alpha SSIM | >0.97 |
| dtSSD | <1.5 |

### What Was Tried

| Experiment | Result | Why |
|-----------|--------|-----|
| V1 EMA α=0.6-0.8 | FAIL both tiers | Severe temporal lag, low PSNR/SSIM |
| V1 EMA α=0.9 | FAIL both tiers | PSNR 34.0dB (threshold 35), SSIM 0.966 (threshold 0.97) |
| V1 EMA α=0.95 | FAIL Tier 1 | Passes Tier 2 but max_abs=0.059 (threshold 5e-3) |
| V2 async decode | KEEP | 7% wall-clock improvement, zero quality impact |

### Key Learnings

1. **Output-space EMA is a dead end for motion video** — any blending weight introduces visible lag
2. **Async decode gives less overlap than expected on unified memory** — CPU/GPU share bandwidth
3. **Tier 2 metrics work well** — caught EMA degradation that Tier 1 alone would also catch, but provide richer signal (PSNR/SSIM/dtSSD show exactly how bad)

## What To Do Next

### V3: Adaptive refiner tile skip (INCONCLUSIVE — exp #46)
- Implemented: skip logic works, zero additional error for skipped tiles
- At tile_size=1024 on 2048: 0% skip (subject in all quadrants)
- At tile_size=512 on 2048: 33% skip, 8% faster — but GroupNorm tiling artifact fails fidelity
- Root cause: GroupNorm spatial stats diverge per tile → boundary artifacts
- **Next:** GroupNorm-aware tiling (frozen stats). See `handoff-2026-03-13-v3-groupnorm-tiling.md`

### V4: Partial feature reuse (big win if viable)
From deep research — confirmed viable strategy for our 4ch input blocker:
- Run Hiera Stages 1-2 every frame (fresh hint re-injection at patch embed)
- Cache Stages 3-4 features (semantically stable, high-level)
- Gate: cosine similarity on Stage 2 output → skip S3-S4 when high
- Optionally warp cached features with optical flow
- `VNGenerateOpticalFlow` on ANE = zero GPU cost for flow
- **Prerequisite:** measure S1-S2 vs S3-S4 compute split to know max speedup

### V5: Motion-adaptive temporal smoothing (revisit V1)
- If temporal smoothing is still wanted, needs motion estimation
- Per-pixel or per-region α based on motion magnitude
- Only smooth static regions, pass through moving regions unchanged
- Requires optical flow (same infra as V4)

### Other ideas (lower priority)
- Batch processing: process 2+ frames simultaneously if memory allows
- Half-precision backbone: currently FP32 — could BF16 work for Hiera?
- CoreML/ANE offload: backbone on ANE, decoders on GPU (out of current scope)

## Key Files

- Video processor: `src/corridorkey_mlx/inference/video.py`
- Video benchmark: `scripts/bench_video.py` (now with Tier 2 metrics + EMA/async flags)
- Benchmark spec: `research/benchmark_spec.md`
- Previous handoff: `research/handoff-2026-03-13-video-temporal.md`
- Deep research: `research/compound/2026-03-13-deep-research-video-matting-optimization.md`
- Experiments log: `research/experiments.jsonl` (45 entries)

## CLI Quick Reference

```bash
# V0 baseline (regenerates reference PNGs)
uv run python scripts/bench_video.py

# V2 async (current best)
uv run python scripts/bench_video.py --async-decode --no-save-reference

# EMA sweep (for experimentation)
uv run python scripts/bench_video.py --ema-sweep 0.6 0.7 0.8 0.9 0.95 --no-save-reference

# Combined
uv run python scripts/bench_video.py --async-decode --ema-alpha 0.7 --no-save-reference
```

## Open Questions

- Hiera S1-S2 vs S3-S4 compute split? (determines max speedup from V4 partial reuse)
- `VNGenerateOpticalFlow` latency @1024 via PyObjC? (needed for V4 + V5)
- Refiner tile confidence threshold for V3? (what alpha value = "high confidence"?)
- Pre-existing test_parity failure: alpha_logits max_abs=0.086 > tol=0.03 — investigate or relax?
