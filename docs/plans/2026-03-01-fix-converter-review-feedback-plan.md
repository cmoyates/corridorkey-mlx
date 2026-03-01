---
title: Fix converter review feedback (items 4, 1, 2)
type: fix
date: 2026-03-01
---

# Fix converter review feedback

Three fixes from PR #1 review, ordered by simplicity.

## Fix 4: Remove unused `skipped` list

**File:** `src/corridorkey_mlx/convert/converter.py:105,110`

Delete the `skipped` list variable and its `.append()` call. The skip logic already works via `continue` — the list serves no purpose.

## Fix 1: Explicit conv key allowlist

**File:** `src/corridorkey_mlx/convert/converter.py:60-66`

Replace heuristic `_is_conv_weight()` with an explicit `CONV_WEIGHT_KEYS: frozenset[str]` allowlist. This matches the "no regex guessing, no silent fallbacks" philosophy.

The 15 conv keys (from checkpoint analysis + tests):
- `encoder.model.patch_embed.proj.weight` — (112,4,7,7)
- `alpha_decoder.linear_fuse.weight` — (256,1024,1,1)
- `fg_decoder.linear_fuse.weight` — (256,1024,1,1)
- `refiner.stem.0.weight` (pre-remap) — (64,7,3,3)
- `refiner.res{1-4}.conv{1-2}.weight` — 8 keys, all (64,64,3,3)
- `refiner.final.weight` — (4,64,1,1)

**Note:** Use pre-remap key names since `_is_conv_weight` is called before remapping in the loop. Total: 15 keys.

Replace `_is_conv_weight(key, arr)` → simple `key in CONV_WEIGHT_KEYS` check. Remove the `arr` parameter.

## Fix 2: Module-scoped test fixture

**File:** `tests/test_conversion.py`

Add a `@pytest.fixture(scope="module")` that loads + converts once, shared by all tests. This avoids 11 redundant `load_pytorch_checkpoint` + `convert_state_dict` calls.

```python
@pytest.fixture(scope="module")
def converted_checkpoint():
    if not CHECKPOINT_PATH.exists():
        pytest.skip("Checkpoint not found")
    state_dict = load_pytorch_checkpoint(CHECKPOINT_PATH)
    converted, diagnostics = convert_state_dict(state_dict)
    return state_dict, converted, diagnostics
```

Each test method takes `converted_checkpoint` as param, destructures what it needs. Remove `_skip_if_no_checkpoint()` helper.

## Acceptance Criteria

- [ ] `skipped` list removed from `convert_state_dict`
- [ ] `CONV_WEIGHT_KEYS` frozenset with 15 keys replaces `_is_conv_weight` heuristic
- [ ] Module-scoped fixture eliminates repeated checkpoint loading
- [ ] `uv run pytest tests/test_conversion.py -v` — 12/12 pass
- [ ] `uv run ruff check .` — clean
