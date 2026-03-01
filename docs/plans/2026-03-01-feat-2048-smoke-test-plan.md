---
title: "feat: Add 2048 smoke test"
type: feat
date: 2026-03-01
---

# Add 2048 Smoke Test

## Overview

Validate MLX model runs end-to-end at CorridorKey's native 2048×2048 resolution on Apple Silicon. Smoke/stability check — not a new parity campaign.

## Problem Statement

All existing tests/scripts default to 256–512. No automated path confirms 2048 actually works. The model was *trained* at 2048, so pos_embed interpolation is a no-op at that resolution, but we haven't exercised the full pipeline there.

## Proposed Solution

One new script + one skipable pytest test. Reuse `CorridorKeyMLXEngine` (already defaults to `img_size=2048`). Minimal additions.

## Deliverables

### 1. `scripts/smoke_2048.py`

Mirrors `scripts/smoke_engine.py` pattern but targeted at 2048 with richer diagnostics.

**CLI args:**
- `--checkpoint PATH` (default: `checkpoints/corridorkey_mlx.safetensors`)
- `--img-size INT` (default: **2048** — this script's whole point)
- `--image PATH` (optional — uses synthetic input if omitted)
- `--hint PATH` (optional — uses synthetic hint if omitted)
- `--output-dir DIR` (default: `output/smoke_2048/`)
- `--save-outputs / --no-save-outputs` (default: save)
- `--seed INT` (default: 42)
- `--compile / --no-compile` (default: True — engine default)

**Behavior:**
1. Generate or load 2048×2048 RGB + hint inputs
2. Instantiate `CorridorKeyMLXEngine(checkpoint_path, img_size, compile)`
3. Reset peak memory via `mx.metal.reset_peak_memory()` (if available, try/except)
4. Time `engine.process_frame(rgb, mask)` with wall-clock
5. Read peak memory via `mx.metal.get_peak_memory()` (if available)
6. Report: input shape, output shapes, dtypes, elapsed time, peak memory (MB)
7. Diagnostic: min/max/mean of alpha+fg, NaN/Inf check, flag all-zeros/all-ones
8. Optionally save alpha.png, fg.png, comp.png
9. Catch `RuntimeError`/`MemoryError` with actionable error message

**Synthetic input generation:**
```python
rng = np.random.default_rng(seed)
rgb = rng.integers(0, 256, (2048, 2048, 3), dtype=np.uint8)
# Circular gradient hint — exercises the path better than random noise
mask = <simple radial gradient uint8>
```

### 2. `tests/test_smoke_2048.py`

**Two tests:**

1. `test_smoke_2048_full` — loads checkpoint, runs engine at 2048, asserts shapes + no NaN. Marked `@pytest.mark.skipif(not HAS_CHECKPOINT)` AND `@pytest.mark.slow`. Won't run in normal `uv run pytest`; run via `uv run pytest -m slow`.

2. `test_smoke_2048_wiring` — lightweight, no checkpoint. Verifies `GreenFormer(img_size=2048)` constructs OK and produces correct output shapes with random weights at 2048 (or smaller like 256 if too heavy without checkpoint). Runs in normal suite.

### 3. README update

Add "2048 Smoke Test" section after the existing "Smoke test" subsection:

```markdown
### 2048 smoke test

Validates full end-to-end inference at CorridorKey's native 2048 resolution.
Uses synthetic inputs by default — no test images required.

```bash
uv run python scripts/smoke_2048.py
```

To use real images:
```bash
uv run python scripts/smoke_2048.py --image shot.png --hint hint.png
```

Reports timing, peak memory, output shapes, and value-range diagnostics.
This is an execution check, not a 2048 parity validation.
```

## Technical Considerations

- **Memory**: 2048×2048×4 float32 ≈ 64MB input, but backbone + decoder intermediates will be much larger. Engine already handles this; we just report peak.
- **Compile**: Engine defaults `compile=True`. First call at 2048 will be slow (compile cost). The smoke script is a single-run tool so the compile overhead is included in timing.
- **Pos_embed**: At 2048, no interpolation needed (trained at 2048). This is the easiest resolution to test.
- **`mx.metal`**: May not be available in all environments. Wrap in try/except.

## Acceptance Criteria

- [x] `scripts/smoke_2048.py` runs successfully with checkpoint
- [x] Reports: input size, output shapes, elapsed time, peak memory
- [x] Detects NaN/Inf and flags suspicious outputs
- [x] Supports both synthetic and user-supplied inputs
- [x] `test_smoke_2048_full` passes when checkpoint present + `-m slow`
- [x] `test_smoke_2048_wiring` passes in normal test suite
- [x] README documents the smoke test
- [x] Existing 512 tests unchanged (112 pass, 1 slow deselected)
- [x] No new binary artifacts committed

## Files Changed

| File | Change |
|------|--------|
| `scripts/smoke_2048.py` | New — main smoke script |
| `tests/test_smoke_2048.py` | New — pytest smoke + wiring tests |
| `README.md` | Add "2048 smoke test" subsection |

## Unresolved Questions

- Exact peak memory at 2048 unknown until first run — might be tight on 8GB machines?
- Include `--compile` timing breakdown (warmup vs inference) or just total?
