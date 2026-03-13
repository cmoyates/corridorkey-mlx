# Handoff: Post V4 — Next Optimization Targets — 2026-03-13

## Where We Are

47 experiments. Single-frame @1024: 422ms. Video @2048: 5.8s/frame (0.17 FPS). V2 async decode kept (+7%). V1 EMA rejected. V3 tile skip inconclusive (GroupNorm artifact). V4 frozen GroupNorm correct but unprofitable (stats pass overhead > tile skip savings).

### Current Best @2048 (37 frames)

| Config | Median (ms) | FPS | Peak mem | Fidelity |
|--------|-------------|-----|----------|----------|
| V0 unfrozen tile=1024 | 5799 | 0.17 | 27491MB | PASS |
| V2 async (same tile) | 5970 | 0.17 | 27491MB | PASS |

### Exhausted Paths

| Approach | Why Dead |
|----------|----------|
| Custom Metal GroupNorm kernel | mx.fast.metal_kernel lacks shared memory (exp 42) |
| GroupNorm alternatives (contiguous, transposed, fp32) | All slower than mx.fast.layer_norm (exp 41) |
| Output-space EMA | Temporal lag fails fidelity at all α (exp 44) |
| Frozen GroupNorm tiling | Correct but 22% slower — stats pass overhead (exp 47) |
| Tile skip (V3) | 0% skip at tile=1024; fg artifact at tile=512 (exp 46) |
| Addmm fusion in backbone | Dispatch overhead > savings (exp 39) |
| GEMM padding stage 0 | Runtime pad cost > alignment benefit (exp 37) |
| Wired memory limits | Hurts perf on unified memory (exp 30) |
| Stream parallelism | Zero GPU-GPU overlap on Apple Silicon (exp 38) |

## What To Try Next

### Tier 1: Highest potential, most viable

#### V5: Partial backbone feature reuse (cache S3-S4)
**Potential: 40-60% latency reduction on non-keyframes.**

Hiera has 24 blocks across 4 stages. Stages 3-4 (blocks 5-23) = 19/24 blocks = ~80% of backbone compute. These high-level semantic features change slowly across frames.

Approach:
1. Run full backbone on keyframes (every Nth frame)
2. Cache Stage 3-4 features
3. On intermediate frames: run S1-S2 only (5 blocks), reuse cached S3-S4
4. Gate: cosine similarity on S2 output → rerun S3-S4 when scene changes

Why it might work:
- Backbone is ~70% of total frame time
- S3-S4 features are semantic (object identity, not edges) — temporally stable
- 4ch input (RGB + alpha hint) changes hint each frame, but hint influence concentrates in early stages
- No quality loss on keyframes; intermediate quality depends on scene dynamics

Key questions:
- S1-S2 vs S3-S4 compute split? Measure with profiling
- Cosine similarity gate: what threshold detects scene changes?
- Does caching S3-S4 double peak memory? (4 feature maps at strides 16, 32)
- Is reusing stale S3-S4 features with fresh S1-S2 features mathematically valid? (stage transitions have projection layers)

Files to modify:
- `src/corridorkey_mlx/model/backbone.py` — split forward into stages
- `src/corridorkey_mlx/model/corridorkey.py` — caching logic in `forward_eager`
- `src/corridorkey_mlx/inference/video.py` — keyframe interval control

#### V6: Refiner tile skip with per-channel delta
**Potential: 8-15% latency reduction at tile_size=512.**

V3 tile skip failed because zeroing the fg delta in confident-alpha tiles produces fg_max_abs=0.996. Fix: only zero the alpha delta channel, always compute fg delta.

Options:
- **Split-channel skip**: skip alpha delta (ch 0), always compute fg delta (ch 1-3). Requires running refiner on skipped tiles but discarding only alpha channel — no savings.
- **Background-only skip**: only skip tiles where alpha is uniformly near 0 (background). In background regions, fg doesn't matter (gets multiplied by alpha=0 in compositing). Skip rate may drop (from 33% to ~20%) but fidelity should pass.
- **Compositing-aware threshold**: threshold on `alpha * |fg_delta|` instead of just alpha — only skip when the fg error is invisible after compositing.

Doesn't need frozen GN if operating at tile_size=1024 (0% skip there though). Most useful combined with smaller tiles.

### Tier 2: Worth investigating

#### V7: Backbone resolution decoupling (already in codebase)
`backbone_size` param on GreenFormer already exists (opt phase 3). Run backbone at lower resolution, refiner at full resolution. Needs quality validation with real content — was deferred.

#### V8: Optical flow feature warping
From deep research doc. Warp cached backbone features using motion vectors instead of reusing stale features. `VNGenerateOpticalFlow` runs on ANE = zero GPU cost.
- Prerequisite: PyObjC bindings for `VNGenerateOpticalFlow`
- Pairs with V5 (partial feature reuse)

### Tier 3: Long shots

#### Batch frame processing
Process 2+ frames simultaneously. Could amortize backbone overhead. Risk: 2x peak memory, unclear if MLX batched ops are faster than sequential.

#### CoreML/ANE backbone offload
Run Hiera on ANE via CoreML, decoders+refiner on GPU. Out of current scope but potentially 2x+ for backbone.

## Recommendation

**Start with V5 (partial backbone reuse)** — highest potential win, addresses the dominant cost center (backbone = 70% of frame time), and builds infrastructure (stage-level forward, feature caching) useful for V8 later.

First step: profile backbone stages to measure S1-S2 vs S3-S4 compute split. If S3-S4 is >60% of backbone, proceed. If not, V7 (resolution decoupling) becomes more attractive.

## Key Files

- Backbone: `src/corridorkey_mlx/model/backbone.py`
- Hiera blocks: `src/corridorkey_mlx/model/hiera.py`
- GreenFormer: `src/corridorkey_mlx/model/corridorkey.py`
- Video processor: `src/corridorkey_mlx/inference/video.py`
- Video benchmark: `scripts/bench_video.py`
- Deep research: `research/compound/2026-03-13-deep-research-video-matting-optimization.md`
- Experiments: `research/experiments.jsonl` (47 entries)

## CLI

```bash
# V0 baseline @2048 (regenerates reference)
uv run python scripts/bench_video.py --img-size 2048

# V2 async @2048
uv run python scripts/bench_video.py --img-size 2048 --async-decode --no-save-reference

# Frozen GN @512 (correct, slow)
uv run python scripts/bench_video.py --img-size 2048 --tile-size 512 --frozen-gn --no-save-reference

# All tests
uv run pytest
```

## Open Questions

- Backbone stage compute split @2048? (profile S1-S2 vs S3-S4)
- Stage transition projections: can fresh S1-S2 + stale S3-S4 features mix without artifacts?
- VNGenerateOpticalFlow latency @1024 via PyObjC?
- Is the pre-existing test_parity failure (9 tests) worth investigating or should tolerances be relaxed?
