---
title: Deep Modules Refactor
type: refactor
date: 2026-03-03
---

# Deep Modules Refactor

## Overview

Restructure corridorkey-mlx into "deep modules" with explicit public interfaces (`__all__` gateways in `__init__.py`), `_testing` subpackages for parity test access to internals, and `import-linter` enforcing cross-package boundaries.

**Guiding principle:** each subpackage exposes a simple interface via `__init__.py`. Implementation files are internal. Tests access internals through `_testing` re-export modules. Import-linter enforces boundaries between top-level packages only (within a package, free imports).

## Problem Statement

All subpackage `__init__.py` files (except `convert/`) are empty. Consumers must know internal filenames (`from corridorkey_mlx.model.corridorkey import GreenFormer`). No tooling enforces boundaries. An AI agent dropping into this codebase has no way to discover a module's API without reading every file.

## Proposed Module Interfaces

### `model/` — Deep Module

```python
# model/__init__.py
from corridorkey_mlx.model.corridorkey import GreenFormer

__all__ = ["GreenFormer"]
```

Internal: `backbone.py`, `hiera.py`, `decoder.py`, `refiner.py`, `corridorkey.py`

```python
# model/_testing.py
"""Unstable re-exports for parity tests. Not part of public API."""
from corridorkey_mlx.model.backbone import HieraBackbone
from corridorkey_mlx.model.hiera import HieraPatchEmbed
from corridorkey_mlx.model.decoder import DecoderHead, MLP
from corridorkey_mlx.model.refiner import CNNRefinerModule, RefinerBlock
```

### `inference/` — Deep Module

```python
# inference/__init__.py
from corridorkey_mlx.inference.pipeline import (
    load_model,
    compile_model,
    infer,
    infer_and_save,
)
from corridorkey_mlx.inference.tiling import tiled_inference
from corridorkey_mlx.inference.selective_refine import (
    selective_refine,
    SelectiveRefineConfig,
    SelectiveRefineResult,
)

__all__ = [
    "load_model",
    "compile_model",
    "infer",
    "infer_and_save",
    "tiled_inference",
    "SelectiveRefineConfig",
    "SelectiveRefineResult",
    "selective_refine",
]
```

Internal: `pipeline.py`, `tiling.py`, `selective_refine.py`

```python
# inference/_testing.py
"""Unstable re-exports for unit tests. Not part of public API."""
from corridorkey_mlx.inference.tiling import (
    _compute_tile_coords,
    _make_blend_weights_2d,
)
from corridorkey_mlx.inference.selective_refine import (
    _binary_dilation,
    _build_input,
    _gaussian_blur,
    _gaussian_kernel_1d,
    _resize_to_square,
)
```

### `io/` — Thin Module

```python
# io/__init__.py
from corridorkey_mlx.io.image import (
    load_image,
    load_alpha_hint,
    normalize_rgb,
    preprocess,
    postprocess_alpha,
    postprocess_foreground,
    save_alpha,
    save_foreground,
)

__all__ = [
    "load_image",
    "load_alpha_hint",
    "normalize_rgb",
    "preprocess",
    "postprocess_alpha",
    "postprocess_foreground",
    "save_alpha",
    "save_foreground",
]
```

### `utils/` — Shared Base Layer

```python
# utils/__init__.py
from corridorkey_mlx.utils.layout import (
    nchw_to_nhwc,
    nhwc_to_nchw,
    conv_weight_pt_to_mlx,
    nchw_to_nhwc_np,
    nhwc_to_nchw_np,
)
from corridorkey_mlx.utils.profiling import (
    TimingResult,
    time_fn,
    warmup_and_bench,
)

__all__ = [
    "nchw_to_nhwc",
    "nhwc_to_nchw",
    "conv_weight_pt_to_mlx",
    "nchw_to_nhwc_np",
    "nhwc_to_nchw_np",
    "TimingResult",
    "time_fn",
    "warmup_and_bench",
]
```

### `convert/` — Already Done

Keep as-is. Already has proper `__all__`.

### Root-level files — Stay Flat

- `engine.py` — add `__all__ = ["CorridorKeyMLXEngine"]`
- `weights.py` — add `__all__ = ["download_weights"]`
- `weights_cli.py` — add `__all__ = ["main", "build_parser"]`
- `__init__.py` — already has `__all__`
- `__main__.py` — no changes needed

## Technical Approach

### Phase 1: Add Gateways (No Breaking Changes)

1. Populate all `__init__.py` files with re-exports and `__all__`
2. Create `_testing.py` modules in `model/` and `inference/`
3. Add `__all__` to `engine.py`, `weights.py`, `weights_cli.py`
4. Run full test suite — everything must still pass (imports unchanged)

**Success criteria:** `uv run pytest` passes, no import changes yet.

### Phase 2: Migrate Imports

Update all consumers to use the gateway imports:

#### Tests

| File | Before | After |
|------|--------|-------|
| `test_decoder_parity.py` | `from corridorkey_mlx.model.decoder import DecoderHead` | `from corridorkey_mlx.model._testing import DecoderHead` |
| `test_refiner_parity.py` | `from corridorkey_mlx.model.refiner import CNNRefinerModule` | `from corridorkey_mlx.model._testing import CNNRefinerModule` |
| `test_hiera_stage_parity.py` | `from corridorkey_mlx.model.hiera import HieraBackbone` | `from corridorkey_mlx.model._testing import HieraBackbone` |
| `test_hiera_stage_shapes.py` | `from corridorkey_mlx.model.hiera import HieraBackbone, HieraPatchEmbed` | `from corridorkey_mlx.model._testing import HieraBackbone, HieraPatchEmbed` |
| `test_greenformer_forward.py` | `from corridorkey_mlx.model.corridorkey import GreenFormer` | `from corridorkey_mlx.model import GreenFormer` |
| `test_end_to_end_smoke.py` | `from corridorkey_mlx.io.image import ...` | `from corridorkey_mlx.io import ...` |
| `test_end_to_end_parity.py` | `from corridorkey_mlx.model.corridorkey import GreenFormer` | `from corridorkey_mlx.model import GreenFormer` |
| `test_compiled_vs_eager_consistency.py` | `from corridorkey_mlx.model.corridorkey import GreenFormer` | `from corridorkey_mlx.model import GreenFormer` |
| `test_tiling_consistency.py` | `from corridorkey_mlx.inference.tiling import _compute_tile_coords, ...` | `from corridorkey_mlx.inference._testing import _compute_tile_coords, ...` |
| `test_selective_refine.py` | `from corridorkey_mlx.inference.selective_refine import _binary_dilation, ...` | `from corridorkey_mlx.inference._testing import _binary_dilation, ...` |
| `test_engine.py` | `from corridorkey_mlx.engine import _validate_image, _validate_mask` | Keep as-is (engine is flat, linter won't enforce within-file) |
| `test_weights.py` | `from corridorkey_mlx.weights import _parse_hash_line, ...` | Keep as-is (weights is flat) |
| `test_conversion.py` | `from corridorkey_mlx.convert.converter import ...` | `from corridorkey_mlx.convert import ...` |

#### Scripts

| File | Before | After |
|------|--------|-------|
| `infer.py` | `from corridorkey_mlx.inference.pipeline import load_model, infer_and_save` | `from corridorkey_mlx.inference import load_model, infer_and_save` |
| `bench_mlx.py` | `from corridorkey_mlx.model.corridorkey import GreenFormer` | `from corridorkey_mlx.model import GreenFormer` |
| `bench_mlx.py` | `from corridorkey_mlx.utils.profiling import warmup_and_bench` | `from corridorkey_mlx.utils import warmup_and_bench` |
| `compare_reference.py` | `from corridorkey_mlx.model.corridorkey import GreenFormer` | `from corridorkey_mlx.model import GreenFormer` |
| `convert_weights.py` | `from corridorkey_mlx.convert.converter import convert_checkpoint` | `from corridorkey_mlx.convert import convert_checkpoint` |
| `experiment_selective_refine.py` | `from corridorkey_mlx.inference.pipeline import load_model` + `selective_refine.*` | `from corridorkey_mlx.inference import load_model, selective_refine, ...` |

#### Internal cross-package imports

| File | Before | After |
|------|--------|-------|
| `engine.py` | `from corridorkey_mlx.model.corridorkey import GreenFormer` (TYPE_CHECKING) | `from corridorkey_mlx.model import GreenFormer` |
| `engine.py` | `from corridorkey_mlx.inference.pipeline import load_model, compile_model` | `from corridorkey_mlx.inference import load_model, compile_model` |
| `engine.py` | `from corridorkey_mlx.io.image import preprocess, ...` | `from corridorkey_mlx.io import preprocess, ...` |
| `inference/pipeline.py` | `from corridorkey_mlx.model.corridorkey import GreenFormer` | `from corridorkey_mlx.model import GreenFormer` |
| `inference/pipeline.py` | `from corridorkey_mlx.io.image import ...` | `from corridorkey_mlx.io import ...` |
| `inference/selective_refine.py` | `from corridorkey_mlx.io.image import normalize_rgb` | `from corridorkey_mlx.io import normalize_rgb` |

**Success criteria:** `uv run pytest` passes with all new import paths.

### Phase 3: Import Linter

Add `import-linter` to enforce cross-package boundaries.

```toml
# pyproject.toml additions
[tool.importlinter]
root_packages = ["corridorkey_mlx"]
include_external_packages = false

[[tool.importlinter.contracts]]
name = "Package layering"
type = "layers"
layers = [
    "corridorkey_mlx.engine",
    "corridorkey_mlx.inference",
    "corridorkey_mlx.model | corridorkey_mlx.io | corridorkey_mlx.convert",
    "corridorkey_mlx.utils",
]
# engine → inference → model/io/convert → utils

[[tool.importlinter.contracts]]
name = "Convert independence"
type = "independence"
modules = [
    "corridorkey_mlx.convert",
    "corridorkey_mlx.model",
    "corridorkey_mlx.io",
]
# convert, model, io should not import from each other

[[tool.importlinter.contracts]]
name = "Weights independence"
type = "independence"
modules = [
    "corridorkey_mlx.weights",
    "corridorkey_mlx.model",
    "corridorkey_mlx.inference",
    "corridorkey_mlx.io",
]
# weights module is standalone
```

Add to dev dependencies and CLAUDE.md commands:

```bash
uv run lint-imports     # import boundary check
```

**Success criteria:** `uv run lint-imports` passes. All cross-package imports go through gateways.

### Phase 4: Documentation & CLAUDE.md Updates

1. Update CLAUDE.md repo layout section to document public interfaces
2. Add a "Module Boundaries" section explaining the gateway pattern
3. Add `_testing.py` convention to CLAUDE.md
4. Update commands section with `lint-imports`

## Acceptance Criteria

- [ ] Every subpackage `__init__.py` has `__all__` defining its public API
- [ ] `model/_testing.py` and `inference/_testing.py` exist with internal re-exports
- [ ] All tests/scripts use gateway imports (no direct `.corridorkey`, `.decoder`, `.pipeline` imports cross-package)
- [ ] `import-linter` passes with layering + independence contracts
- [ ] Full test suite passes (`uv run pytest`)
- [ ] Lint passes (`uv run ruff check .`)
- [ ] CLAUDE.md documents module boundaries and gateway convention

## Dependencies & Risks

**Risk: circular imports from re-exports.** Mitigated by the existing acyclic dependency graph — no cycles detected.

**Risk: import-linter false positives.** `TYPE_CHECKING` imports should be excluded. `import-linter` supports `--exclude-type-checking-imports`. Verify this works.

**Risk: breaking scripts outside the repo.** Only the root `CorridorKeyMLXEngine` export matters for external consumers. Internal restructuring doesn't affect the public API.

**Dependency:** `import-linter` package added to dev deps.

## Unresolved Questions

- `utils.layout` funcs used inside `convert/converter.py` — does `convert` importing from `utils` violate any intended boundary? (current plan: `utils` is the base layer, everyone can import it)
- Should `_testing.py` modules have their own `__all__`? (leaning yes for explicitness)
- `weights.py` has module-level constants (`DEFAULT_ASSET_NAME`, etc.) used by `test_weights.py` — add to `__all__` or keep test accessing flat file directly?
