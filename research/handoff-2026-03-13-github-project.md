# Handoff: GitHub Project Board Setup — 2026-03-13

## What happened this session

### GitHub Issues Created (#2-#17)
16 issues covering all loose ends, untried ideas, and upstream research findings. Repo: `cmoyates/corridorkey-mlx`.

### Upstream Research (5 agents, 7 BTCA repos)
- **Critical:** `mx.fast.metal_kernel` DOES support threadgroup shared memory + barriers — reopens GroupNorm optimization (#14)
- **Raiden129 fork** added to BTCA — confirms sparse tile skip works with frozen GN stats (PyTorch gets this free via `model.eval()`)
- **EZ-CorridorKey v1.6.0** integrates corridorkey-mlx as Apple Silicon backend
- **Tile overlap** should be 128px not 64px (65px receptive field, <1px margin)
- No new changes in CorridorKey-Engine or Marcel's fork

### Files changed (committed + pushed as `f519b10`)
- `CLAUDE.md` — added experiment tracking via GitHub issues
- `btca.config.jsonc` — added Raiden129/CorridorKey_Test
- `research/compound/2026-03-13-upstream-research-5.md` — compound note

## What needs doing next

### 1. Fix `GITHUB_TOKEN` scope
The `GITHUB_TOKEN` env var overrides `gh` CLI credentials and lacks the `project` scope. Either:
- Update the token source to include `project` scope, OR
- Unset `GITHUB_TOKEN` and use `gh auth refresh -s project` so the CLI stores its own token

### 2. Create GitHub Project board
Once auth works:
```bash
gh project create --owner cmoyates --title "corridorkey-mlx optimization" --format "TABLE"
```
Then add all issues and set up Status field (Todo / In Progress / Done).

### 3. Suggested issue ordering (Todo column, top = highest priority)

**Tier 0 — Unblocks other work**
1. #14 — Re-attempt Metal GroupNorm with shared memory (unblocks #3 tile skip)
2. #11 — Create golden reference at 2048 (unblocks all 2048 re-benchmarks)
3. #8 — Fidelity budget: reverting backbone BF16 (recovers fg headroom)

**Tier 1 — Highest impact**
4. #3 — V6: Refiner tile skip with per-channel delta (depends on #14)
5. #2 — V5: Partial backbone feature reuse (cache S3-S4)
6. #4 — V7: Validate backbone resolution decoupling at 2048

**Tier 2 — Worth investigating**
7. #6 — Re-benchmark env var tuning at 2048
8. #13 — Re-benchmark backbone int8 quantization at 2048
9. #16 — Increase refiner tile overlap 64->128px
10. #15 — Upstream MLX backend alignment check
11. #12 — Investigate 9 pre-existing test parity failures

**Tier 3 — Lower priority**
12. #7 — Feature-space EMA
13. #5 — V8: Optical flow feature warping
14. #9 — Run-Length Tokenization
15. #17 — GELU approx="fast"
16. #10 — Batch frame processing

### 4. Update CLAUDE.md
Add a reference to the project board URL once created, e.g.:
```
- Project board: https://github.com/users/cmoyates/projects/N
- Use the board to track priority ordering — top of Todo = next experiment
```

## Issue Summary

| # | Title | Label | Tier |
|---|-------|-------|------|
| 2 | V5: Partial backbone feature reuse (cache S3-S4) | enhancement | 1 |
| 3 | V6: Refiner tile skip with per-channel delta | enhancement | 1 |
| 4 | V7: Validate backbone resolution decoupling at 2048 | enhancement | 1 |
| 5 | V8: Optical flow feature warping | enhancement | 3 |
| 6 | Re-benchmark env var tuning at 2048 | investigation | 2 |
| 7 | Feature-space EMA | investigation | 3 |
| 8 | Fidelity budget: revert backbone BF16 | investigation | 0 |
| 9 | Run-Length Tokenization | enhancement | 3 |
| 10 | Batch frame processing | enhancement | 3 |
| 11 | Create golden reference at 2048 | infrastructure | 0 |
| 12 | Investigate 9 test parity failures | bug | 2 |
| 13 | Re-benchmark backbone int8 quant at 2048 | investigation | 2 |
| 14 | Re-attempt Metal GroupNorm with shared memory | enhancement | 0 |
| 15 | Upstream MLX backend alignment | investigation | 2 |
| 16 | Increase tile overlap 64->128px | investigation | 2 |
| 17 | GELU approx="fast" | enhancement | 3 |
