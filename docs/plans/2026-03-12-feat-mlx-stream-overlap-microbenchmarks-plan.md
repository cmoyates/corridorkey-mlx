---
title: "feat: MLX Stream Overlap Micro-benchmarks"
type: feat
date: 2026-03-12
---

# MLX Stream Overlap Micro-benchmarks

## Overview

Two micro-benchmarks to empirically verify whether MLX streams enable GPU-GPU
parallelism on Apple Silicon. Results determine whether Exp 38 (interleaved
backbone-to-decoder pipeline) is viable.

**Prior research says NO** — Apple Silicon offers only ~2x command queue
concurrency (UI+compute, not throughput). Both streams share GPU cores and
memory bandwidth. But we want data, not just theory.

## Test 1: Do MLX Streams Overlap GPU Work?

### Hypothesis

Two independent matmuls dispatched on separate GPU streams will NOT run
meaningfully faster than dispatched sequentially on one stream, because Apple
Silicon GPUs don't parallelize across command queues for compute workloads.

### Protocol

```
scripts/bench_stream_overlap.py

Setup: two independent matmuls sized like decoder projection
A @ B on stream 1, C @ D on stream 2
Both are (N, 256) @ (256, 256) — match decoder embed_dim
N = 16384 (= 128*128, roughly H/4 * W/4 at 512 input)

Variant A — Sequential (single stream):
  out1 = a @ b
  out2 = c @ d
  mx.eval(out1, out2)   # NOTE: mx.eval is MLX array materialization

Variant B — Parallel (two streams):
  out1 = a @ b
  with mx.stream(stream2):
      out2 = c @ d
  mx.eval(out1, out2)   # NOTE: mx.eval is MLX array materialization

Variant C — Heavy matmuls (backbone-sized, 16384 x 896 @ 896 x 896)
  Same sequential vs parallel comparison at backbone scale

Measure: 5 warmup, 20 bench runs, report median
Speedup = sequential_median / parallel_median
If speedup > 1.1  -> streams overlap (proceed with Exp 38)
If speedup < 1.05 -> no meaningful overlap (abandon Exp 38)
```

### Decision Criteria

| Speedup | Interpretation | Action |
|---------|---------------|--------|
| > 1.10 | Streams overlap meaningfully | Proceed to Test 2 + Exp 38 |
| 1.05-1.10 | Marginal overlap | Run Test 2, re-evaluate |
| < 1.05 | No overlap | Skip Test 2, abandon Exp 38 |

### Files

- `scripts/bench_stream_overlap.py` (new, disposable)

### Time

~15 min including implementation + runs

---

## Test 2: Does async_eval Force a Sync Point?

### Hypothesis

`mx.async_eval()` returns immediately (dispatches to GPU, CPU continues).
It should NOT add latency to a compute chain when inserted between two
independent operations.

### Protocol

```
scripts/bench_async_eval_sync.py

Setup: chain of matmuls simulating backbone -> decoder transition
big_matmul (backbone-like) -> async_eval -> small_matmul (decoder-like)

Variant A — No async_eval (baseline):
  backbone_out = x @ big_w      # "backbone" heavy
  decoder_out = backbone_out @ small_w  # "decoder proj" light
  mx.eval(decoder_out)          # NOTE: mx.eval is MLX materialization

Variant B — async_eval between stages:
  backbone_out = x @ big_w
  mx.async_eval(backbone_out)   # dispatch backbone to GPU
  decoder_out = backbone_out @ small_w
  mx.eval(decoder_out)          # NOTE: mx.eval is MLX materialization

Variant C — async_eval on SEPARATE stream:
  backbone_out = x @ big_w
  mx.async_eval(backbone_out)
  with mx.stream(stream2):
      decoder_out = backbone_out @ small_w
  mx.eval(decoder_out)          # NOTE: mx.eval is MLX materialization

Measure: 5 warmup, 20 bench runs
Compare medians
```

### Decision Criteria

| Result | Interpretation |
|--------|---------------|
| B ~ A (+-2%) | async_eval is non-blocking (expected) |
| B > A by >5% | async_eval forces sync (bad — avoid in pipeline) |
| C < A by >5% | Stream + async_eval enables overlap (bullish for Exp 38) |

### Files

- `scripts/bench_async_eval_sync.py` (new, disposable)

### Time

~15 min

---

## Execution Order

1. Implement + run Test 1
2. If speedup > 1.05 -> implement + run Test 2
3. Record results in `research/experiments.jsonl`
4. Decision: proceed with or abandon Exp 38

## Success Metrics

- Empirical data on stream overlap (median latency, speedup ratio)
- Clear go/no-go decision for Exp 38
- Total time: 15-30 min
