# Community CorridorKey Repo Findings

Researched 2026-03-11. Sources: CorridorKey-Engine, MarcelLieb fork, EZ-CorridorKey.

---

## High Signal

### Token routing + LTRM (CorridorKey-Engine)
- Source: 99oblivius/CorridorKey-Engine -- HintBasedTokenRouter, LightweightTokenRefinementModule
- Routes "easy" tokens (alpha hint near 0 or 1) to cheap O(N) module instead of O(N^2) attention
- LTRM: LayerNorm -> Linear -> GELU -> DepthwiseConv5x5 -> GELU -> Linear -> ECA gate
- Applied at Hiera stages 2-3 only (deepest, most expensive)
- Zero-initialized -- acts as identity residual with existing checkpoint (no fine-tuning needed)
- Key question: does skipping attention for solid regions degrade quality?
- At 512x512 typical green screen: 80-90% of tokens may be "easy"

### Refiner-only tiling (CorridorKey-Engine)
- Source: 99oblivius/CorridorKey-Engine -- TiledCNNRefiner(CNNRefinerModule)
- Subclass that tiles only the refiner CNN (512x512, 128px overlap, linear blend)
- Backbone+decoder run once at full res; only memory-heavy refiner (7ch CNN) is tiled
- Cleaner than full-model tiling; lower peak memory since backbone intermediates allocated once
- Most impactful at resolutions > 512

## Medium Signal

### Frozen batch size + zero-pad (CorridorKey-Engine)
- Compute max batch from available memory, pad all batches to fixed size
- Avoids mx.compile retracing on shape changes
- Pattern for video pipeline, not single-image

### GPU-side preprocessing (CorridorKey-Engine)
- Resize + ImageNet normalize on GPU instead of CPU numpy
- Minimal gain on Apple Silicon unified memory, but eliminates numpy dependency in hot path

## Low Signal / Not Applicable

- CUDA graph capture/replay -- covered by mx.compile
- Triple-buffered pinned DMA -- N/A (unified memory)
- Inter-stage cache clearing -- already doing this
- Batched multiprocess post-processing -- pipeline-level, not model
- Linear-space resizing (MarcelLieb) -- quality, not speed
- VRAM-adaptive strategy (EZ-CorridorKey) -- UX pattern
- FlashAttention Hiera patch (EZ-CorridorKey) -- CUDA-only
