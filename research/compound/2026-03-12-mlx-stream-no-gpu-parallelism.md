# MLX Streams: No GPU-GPU Parallelism on Apple Silicon

**Date**: 2026-03-12
**Category**: disproven
**Tags**: mlx, streams, metal, parallelism, apple-silicon

## Finding

MLX streams do NOT enable GPU-GPU parallelism for compute workloads.
Multiple GPU streams dispatch to separate Metal command queues, but Apple
Silicon GPUs don't meaningfully parallelize across queues for compute.

## Evidence

Micro-benchmark (`scripts/bench_stream_overlap.py`):

| Test | 1 stream | 2 streams | Speedup |
|------|----------|-----------|---------|
| Decoder-scale (16384x256) | 0.82ms | 0.83ms | 0.979x |
| Backbone-scale (16384x896) | 6.42ms | 6.48ms | 0.990x |
| 4 matmuls, 2 streams | 1.29ms | 1.34ms | 0.968x |
| 4 matmuls, 4 streams | 1.29ms | 1.34ms | 0.961x |

Streams consistently add 2-4% overhead from dispatch coordination.

## Why

Apple Silicon's ~2x command queue concurrency is designed for
responsiveness (UI queue + compute queue), not throughput. Per Philip
Turner's Metal microarch research: "Sub-core concurrency only happens
among commands within the same MTLComputeCommandEncoder."

## What Streams ARE Good For

- **CPU-GPU overlap**: Build next graph on CPU while GPU executes current
  one. This is what `mx.async_eval()` exploits (already used in model).
- **Dependency ordering**: Ensure operations on different streams don't
  block each other at graph-build time.

## Implication

Any optimization idea that relies on GPU-GPU parallelism via streams is
dead on arrival. The only productive overlap is CPU graph-building with
GPU execution — already captured by the existing `mx.async_eval` calls
in `corridorkey.py`.
