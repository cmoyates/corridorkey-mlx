# Backbone Skip Rejected: Hint-in-Backbone Kills Feature Reuse

**Date:** 2026-03-13
**Experiment:** V2 (backbone skip + feature reuse)
**Result:** REJECTED

## Finding

Backbone skip (caching Hiera features across frames, running only decoders+refiner on non-keyframes) does not work for CorridorKey because the backbone encodes a 4-channel input: RGB + alpha hint.

Stale cached features carry stale alpha hint spatial information. When the subject moves between frames, the hint mask shifts but the cached features still encode the old mask position. The refiner (additive residual on fresh RGB) cannot correct this spatial mismatch in the coarse predictions.

## Evidence

| Skip | FPS | Speedup | Non-KF median | Alpha mean err | Visually |
|------|-----|---------|---------------|----------------|----------|
| 2 | 2.42 | 1.36x | 199ms | 4.2% (high-motion frames) | Bad |
| 3 | 2.65 | 1.49x | 202ms | varies | Bad |
| 5 | 2.94 | 1.65x | 206ms | varies | Bad |

- Per-pixel max_abs ~1.0 at motion boundaries (matte flips 0↔1)
- Low-motion frames (e.g., frame 3 at skip=2) had mean error 0.01% — nearly perfect
- High-motion frames (e.g., frame 1) had 6.6% of pixels with >10% error
- Keyframe outputs are bit-exact with single-frame (decomposed forward verified)

## Why This Differs From Prior Art

Deep Feature Flow (CVPR 2017) showed 2-4x speedup with backbone skip on video segmentation. But DFF's backbone only encodes RGB (3ch). CorridorKey's backbone encodes RGB+hint (4ch), making cached features carry stale semantic guidance. The refiner's correction capacity (additive logit residual) is insufficient to overcome stale coarse predictions.

## What Was Kept

- **Decomposed forward API**: `run_backbone()`, `run_decoders()`, `run_refiner()` on GreenFormer — bit-exact with `__call__`, useful for profiling and future experiments
- **VideoProcessor `skip_interval` param** — functional but produces unacceptable quality

## Artifacts

- `research/artifacts/video_v2_skip{2,3,5}.json` — full benchmark + fidelity data
- `output/video_skip2/` — visual output for inspection
