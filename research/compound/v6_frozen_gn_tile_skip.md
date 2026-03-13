# V6 Frozen GN + Tile Skip — Correctness Win, No Speedup

## Finding
Frozen GroupNorm with Metal kernel perfectly eliminates tiling artifacts (0.0 error vs non-tiled). But tile skip doesn't produce speedup on real content.

## Evidence
- Frozen GN vs non-tiled: **0.0 max error** (perfect)
- Per-tile stats vs non-tiled: 68.4/255 max error (tiling artifact)
- Skip rate: 0% at tile_size=512, 25% at tile_size=128 on real green screen footage
- Pipeline: 1636ms → 1612ms (~0% improvement)

## Why skip doesn't help
Large subject fills most tiles → alpha boundaries in every tile → nothing to skip. Stats collection (full-image refiner forward) adds overhead that cancels any skip savings.

## Implementation detail
Metal GroupNorm now accepts `frozen_stats=(mean, var)` — converts to (sum, sumsq) format for kernel B, skipping kernel A. Tiny fp32 roundtrip diff (~5e-7).
