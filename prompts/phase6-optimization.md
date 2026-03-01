Work only on optimization, benchmarking, and Apple-silicon inference ergonomics.

Goal:
Make the MLX CorridorKey port performant and measurable on Apple silicon, with a focus on reliable benchmarking, memory behavior, and practical large-image inference strategy.

Context:

- End-to-end correctness should already exist before this phase starts.
- MLX uses lazy evaluation, so timing code must force evaluation.
- MLX compile() behavior depends on shape stability; changing shapes can trigger recompilation unless shapeless=True is used.
- Shape-dependent logic must be treated carefully if testing shapeless compilation.

Primary deliverables:

- scripts/bench_mlx.py
- src/corridorkey_mlx/inference/tiling.py
- src/corridorkey_mlx/utils/profiling.py
- tests/test_tiling_consistency.py
- tests/test_compiled_vs_eager_consistency.py
- README performance section
- benchmark output examples or artifacts

Requirements:

1. Build a benchmark harness that measures:
   - eager/uncompiled inference
   - compiled inference with fixed shapes
   - optionally shapeless compiled inference if the graph is safe for it
2. Force evaluation in benchmark timing so measurements reflect real execution rather than queued lazy graphs.
3. Separate:
   - first-run / compile cost
   - warm steady-state latency
   - peak practical memory observations
4. Benchmark at several realistic resolutions, for example:
   - small sanity size
   - medium working size
   - high-resolution target or near-target size
5. Add an optional tiled inference path for large images.
6. Validate tiled vs non-tiled consistency where overlap/blending makes sense.
7. Keep the output report concise and decision-oriented:
   - resolution
   - eager latency
   - compiled latency
   - compile warmup cost
   - batch size
   - output parity summary
8. If testing shapeless compile, explicitly verify that no shape-dependent logic breaks correctness.
9. Keep performance changes behind clean flags or config settings so parity remains easy to reproduce.
10. Document recommended settings for an Apple silicon laptop workflow.

Optimization scope to consider:

- fixed-shape compile path
- shapeless compile path only if safe
- reduced intermediate materialization
- memory-aware tiling / chunking
- minimizing unnecessary layout conversions
- avoiding debug prints or implicit materialization in hot paths

Diagnostics to produce:

- resolution
- batch size
- compile mode
- first-run latency
- steady-state latency
- consistency check against eager mode
- notes on memory behavior and failure thresholds

Do not:

- change model behavior to chase speed
- quantize in this phase unless explicitly requested
- add training
- add speculative custom kernels
- benchmark without explicit eval / synchronization semantics
- hide compile warmup inside steady-state numbers

Working style:

- Start by auditing the current forward path for obvious avoidable conversions or materializations.
- Implement the benchmark harness before broad optimization changes.
- Compare eager vs compiled numerics after every meaningful optimization.
- Keep high-resolution strategy explicit and documented.

Definition of done:

- repeatable benchmark script exists
- eager vs compiled comparisons are implemented
- optional tiling path exists with consistency tests
- README includes practical Apple silicon guidance
- the repo can answer “what settings should I use on a real Mac?” with evidence
