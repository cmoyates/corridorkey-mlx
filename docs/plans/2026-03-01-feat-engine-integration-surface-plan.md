---
title: "feat: Add CorridorKeyMLXEngine integration surface"
type: feat
date: 2026-03-01
---

# feat: Add CorridorKeyMLXEngine integration surface

## Overview

Expose a stable `CorridorKeyMLXEngine` class so the main CorridorKey repo can consume this package as a drop-in MLX backend. Currently only flat functions exist (`load_model`, `infer`, `infer_and_save`) — none accept in-memory arrays, and there is no engine lifecycle or `process_frame(...)` contract.

## Problem Statement

The main CorridorKey repo expects a backend engine with:
- Constructor: `checkpoint_path`, `device`, `img_size`, `use_refiner`
- Method: `process_frame(image, mask_linear, ...)` returning dict with `alpha`, `fg`, `comp`, `processed`

This repo currently:
- Has no engine class — only loose functions in `pipeline.py`
- `infer()` accepts file paths, not numpy/PIL arrays
- Returns only `alpha` + `foreground`, missing `comp` and `processed`
- Has no `refiner_scale`, despill, despeckle, or linear/sRGB handling
- `DEFAULT_CHECKPOINT` is a relative path (breaks as installed library)
- `requires-python >= 3.12` but main repo targets 3.11

## Proposed Solution

Thin adapter class in `src/corridorkey_mlx/engine.py` wrapping existing model/inference code. No core model changes.

## Technical Approach

### Phase 1: Engine adapter + packaging fixes

#### 1a. Lower Python version to >=3.11

- All source already uses `from __future__ import annotations` — no 3.12-only syntax
- Update `pyproject.toml`: `requires-python = ">=3.11"`
- Update `tool.ruff.target-version` and `tool.mypy.python_version`
- Check dep version floors are 3.11-compatible (numpy >=2.4.2 may need lowering — numpy 2.0+ supports 3.11)
- Run `uv run ruff check .` + `uv run mypy src/` to verify

#### 1b. Create `src/corridorkey_mlx/engine.py`

```python
class CorridorKeyMLXEngine:
    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | None = None,      # ignored on MLX, accepted for compat
        img_size: int = 2048,            # production default
        use_refiner: bool = True,
        compile: bool = True,
    ) -> None: ...

    def process_frame(
        self,
        image: np.ndarray,              # uint8 HWC RGB
        mask_linear: np.ndarray,         # uint8 HW or HW1 grayscale
        refiner_scale: float = 1.0,
        input_is_linear: bool = False,
        fg_is_straight: bool = True,
        despill_strength: float = 1.0,
        auto_despeckle: bool = True,
        despeckle_size: int = 400,
    ) -> dict[str, np.ndarray]: ...
```

Constructor:
- Validates `checkpoint_path` exists (absolute or resolved)
- Calls existing `load_model()` with `img_size` and `compile`
- Stores `use_refiner`, `img_size`
- Logs warning if `device` is not None

`process_frame()` pipeline:
1. **Input validation**: assert uint8 HWC(3) for image, uint8 HW or HW1 for mask
2. **Store original resolution** for output resize
3. **Convert to float32 [0,1]**: `image / 255.0`, `mask / 255.0`
4. **Reshape mask**: ensure `(H, W, 1)`
5. **Resize** to `img_size x img_size` via PIL (bicubic) — reuse existing pattern
6. **Preprocess**: call existing `normalize_rgb()` + `preprocess()` from `io/image.py`
7. **Forward pass**: `self._model(x)` — raw output dict
8. **Materialize** all outputs
9. **Select outputs**: if `use_refiner=True`, use `alpha_final`/`fg_final`; else use `alpha_coarse`/`fg_coarse`
10. **Apply refiner_scale** (output-space lerp): `alpha = lerp(alpha_coarse, alpha_final, refiner_scale)`
11. **Postprocess** to uint8 via existing `postprocess_alpha()`/`postprocess_foreground()`
12. **Resize outputs** back to original input resolution
13. **Despill/despeckle**: no-op stubs for now (documented, warn once)
14. **Composite**: `comp = (fg * alpha_3ch + bg * (1 - alpha_3ch))` with black bg, uint8
15. **Return** `{"alpha": ..., "fg": ..., "comp": ..., "processed": fg}` — `processed` = fg until despill/despeckle implemented

#### 1c. Export from `__init__.py`

```python
from corridorkey_mlx.engine import CorridorKeyMLXEngine
```

So callers can do `from corridorkey_mlx import CorridorKeyMLXEngine`.

#### 1d. Fix DEFAULT_CHECKPOINT

Remove relative-path default from `pipeline.py`. Engine requires explicit `checkpoint_path`.

### Phase 2: Smoke script + tests

#### 2a. `scripts/smoke_engine.py`

- Takes `--image`, `--hint`, `--checkpoint` args
- Instantiates `CorridorKeyMLXEngine`
- Runs `process_frame()`
- Prints output shapes and value ranges
- Optionally saves outputs

#### 2b. Tests in `tests/test_engine.py`

- **test_engine_init_requires_checkpoint**: missing path raises error
- **test_engine_output_keys**: process_frame returns `alpha`, `fg`, `comp`, `processed`
- **test_engine_output_shapes**: all outputs match input spatial dims
- **test_engine_output_dtypes**: all uint8
- **test_engine_mask_shape_normalization**: HW and HW1 both accepted

Use small synthetic inputs (e.g. 64x64 random) with checkpoint. Mark as `@pytest.mark.skipif` when checkpoint not available.

### Phase 3: Documentation

#### 3a. README section: "Using as a backend"

Cover:
- Editable install: `uv pip install -e ../corridorkey-mlx`
- Canonical import: `from corridorkey_mlx import CorridorKeyMLXEngine`
- Constructor params and defaults
- Expected checkpoint format (.safetensors, converted)
- Expected image/hint formats (uint8 HWC RGB, uint8 HW grayscale)
- Output dict keys and shapes
- Smoke command example
- Migration note: standalone script vs backend engine usage

#### 3b. Docstrings

Engine class and `process_frame` get thorough docstrings covering:
- Input formats (uint8 HWC RGB, uint8 HW mask)
- Output formats and semantics
- Preprocessing chain (resize, ImageNet norm, NHWC)
- Which params are stubs (despill, despeckle)
- Linear vs sRGB assumptions
- img_size: 512 for dev, 2048 for production

## Acceptance Criteria

- [ ] `from corridorkey_mlx import CorridorKeyMLXEngine` works
- [ ] Constructor accepts `checkpoint_path`, `device`, `img_size`, `use_refiner`, `compile`
- [ ] `process_frame()` accepts numpy uint8 arrays, returns dict with `alpha`, `fg`, `comp`, `processed`
- [ ] Outputs resized to original input resolution
- [ ] `use_refiner=False` returns coarse predictions
- [ ] `refiner_scale` blends between coarse and refined
- [ ] Despill/despeckle are documented stubs (no-op, warn once)
- [ ] `comp` composites fg over black background
- [ ] `processed` = fg (until despill/despeckle implemented)
- [ ] Python >=3.11 works
- [ ] Smoke script runs one frame successfully
- [ ] Existing 94 tests still pass
- [ ] README documents backend usage + editable install

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| `refiner_scale` semantics | output-space lerp | no model changes needed |
| `use_refiner=False` | use `alpha_coarse`/`fg_coarse` from output dict | model always runs full forward; adapter selects outputs |
| despill/despeckle | no-op stubs, warn once | algorithms unknown; unblock integration now |
| `comp` background | black (0,0,0) | common default for matte compositing |
| `processed` meaning | same as `fg` for now | placeholder until despill/despeckle exist |
| `device` param | accepted, ignored, log warning | compat with Torch engine signature |
| default `img_size` | 2048 (engine), 512 (dev scripts) | model trained at 2048; scripts keep 512 for speed |
| `input_is_linear` | accepted, no-op for now | ImageNet stats assume sRGB; linearization would break normalization |
| checkpoint default | none — required param | relative paths break as library |

## Dependencies & Risks

- **numpy version floor**: `numpy>=2.4.2` may not support 3.11. Need to check and potentially lower to `>=1.26` or `>=2.0`.
- **Despill/despeckle gap**: real implementations need the main repo's algorithm. Document as TODO.
- **refiner_scale at compile time**: output-space lerp works with compiled model since it's post-forward. Logit-space would require model changes and recompilation.

## Unresolved Questions

1. `processed` — what exactly does main repo return here? Despilled fg? Masked fg?
2. `comp` bg — always black or configurable?
3. `use_refiner=False` — skip refiner forward pass (perf) or just ignore outputs (simpler)?
4. `refiner_scale` — logit-space or output-space? (plan assumes output-space)
5. despill/despeckle algorithms — need from main repo for real impl
6. `input_is_linear` — does original model ever receive linear-light inputs?
7. `mask_linear` naming — is it actually linear-light or just naming convention?
