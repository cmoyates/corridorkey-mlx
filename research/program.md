# Research Program — corridorkey-mlx Optimization

## Objective

Reduce steady-state inference latency and peak memory usage on Apple Silicon while preserving baseline fidelity (max abs error < 1e-3 vs golden reference).

## Phase 1 search areas (priority order)

### 1. MLX tile lifecycle and memory discipline
- mx materialization placement — materialize early to release graph nodes
- Delete temporary tensor refs explicitly
- gc.collect() + mx.clear_cache() timing between pipeline stages
- Avoid redundant allocations in tiled inference loop

### 2. Selective precision policy
- Refiner in float16/bfloat16 (most numerically stable component)
- Backbone stays float32 (accumulation-sensitive)
- Decoder: test bf16 with fp32 sigmoid (current approach)

### 3. Tiled inference heuristics
- Tile size sweep (256, 384, 512, 768)
- Overlap sweep (32, 64, 96, 128)
- Blending strategy (linear ramp vs cosine vs flat center)

### 4. Compile-path policy
- mx.compile for fixed-shape paths
- Warmup-aware benchmarking (exclude first N runs)
- Investigate shapeless=True for backbone subgraph only

### 5. Tensor layout / staging / contiguity
- Ensure contiguous tensors before compute-heavy ops
- Minimize NCHW-NHWC transitions
- Staging: avoid CPU-GPU roundtrips in tiled inference

## Out of scope (phase 1)

- Architecture redesign
- Training / finetuning
- CoreML / ANE export
- Temporal coherence / optical flow
- Alpha-hint replacement
- ROI crop-and-paste inference

## Evaluation cadence

Each experiment produces:
1. Structured JSON result in `research/artifacts/`
2. Score via `scripts/score_experiment.py`
3. Summary appended to `research/experiments.jsonl`
4. Compound note if the finding is reusable
