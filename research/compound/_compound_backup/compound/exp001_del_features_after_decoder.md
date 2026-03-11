# Compound: del features after decoder consumption

**Date:** 2026-03-10
**Experiment:** 001 — tile-lifecycle-memory-discipline
**Verdict:** KEEP (score 1.0289)

## Finding

Explicitly `del features` in `GreenFormer.__call__` after both decoders consume the backbone feature list yields a consistent ~4% latency improvement at 512×512. Peak memory unchanged in full-frame measurement — benefit expected primarily in tiled inference where multiple forward passes accumulate.

## Why it works

With `stage_gc=True`, backbone features are materialized Metal buffers (~13MB at 512×512). Without `del`, they stay alive through the refiner stage (which doesn't use them). Deleting them lets Metal's allocator reclaim buffers before the refiner allocates its own, reducing allocator pressure.

## Key constraint

- Only effective when `stage_gc=True` (i.e., `_compiled=False`)
- With `mx.compile`, features are lazy graph nodes — `del` has no effect since downstream ops hold transitive refs
- Tiled inference always uses `_compiled=False`, so this is always active for the tile path

## Numbers

| Metric | Baseline | After | Delta |
|--------|----------|-------|-------|
| Median | 167.59ms | 161.07ms | -3.9% |
| P95 | 173.20ms | 165.52ms | -4.4% |
| Peak memory | 2068.7MB | 2068.7MB | 0% |
