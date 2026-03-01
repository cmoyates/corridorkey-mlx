# Phase 4: Hiera Backbone MLX Port

## Context

Phase 4 of the CorridorKey MLX port. Phases 1-3 complete (reference harness, decoder/refiner, converter). Now need the Hiera backbone — the most complex component. Once done, all model pieces exist for end-to-end inference.

## Architecture Summary

Hiera is a hierarchical vision transformer from timm. Key components:

- **PatchEmbed**: Conv2d(4->112, 7x7, stride=4) + reshape to [B, N, C]
- **pos_embed**: Learned (1, N, 112), bicubic interpolated from training res
- **Unroll**: Permutes tokens for mask-unit windowed attention
- **24 HieraBlocks** in 4 stages:
  - Stage 0 (blocks 0-1): dim=112, heads=2, window=64, mask_unit_attn=True
  - Stage 1 (blocks 2-4): dim=224, heads=4, window=16, mask_unit_attn=True
  - Stage 2 (blocks 5-20): dim=448, heads=8, window=4, mask_unit_attn=False (global)
  - Stage 3 (blocks 21-23): dim=896, heads=16, window=1, mask_unit_attn=False (global)
- **Reroll**: Undoes permutation -> spatial [B, H, W, C]
- **Stage transitions** (blocks 2, 5, 21): proj Linear + max-pool reduces tokens 4x
- **Features**: Collected at stage_ends [1, 4, 20, 23], rerolled to NHWC

### MaskUnitAttention
- qkv = Linear(dim_in, 3*dim_out), reshaped per window
- q_stride > 1 at transitions: max-pool over q_stride dim in queries
- Scaled dot-product attention
- proj = Linear(dim_out, dim_out)

### MLP
- fc1(dim->4*dim) -> GELU -> fc2(4*dim->dim)

## Checkpoint Details (from safetensors)

- 297 encoder keys total
- `encoder.model.patch_embed.proj.weight`: (112, 7, 7, 4) — already transposed by converter
- `encoder.model.pos_embed`: (1, 262144, 112) — raw, needs bicubic interpolation
- Transition blocks (2, 5, 21) have extra `proj.weight/bias`
- All block weights are Linear (no transpose needed)
- No LayerScale weights (init_values=None for base_plus)

## Implementation Plan

### Sub-phase 4a: Core backbone module

**File**: `src/corridorkey_mlx/model/hiera.py`

Implement (all in [B, N, C] sequence format, NHWC only at boundaries):

1. **`HieraPatchEmbed`** — Conv2d(4->112, 7x7, stride=4) + flatten to [B, N, C]
2. **`unroll(x, spatial_size, schedule)`** — faithful port of timm's Unroll.forward
3. **`reroll(x, block_idx, schedule_map)`** — faithful port of timm's Reroll.forward (no-mask path)
4. **`undo_windowing(x, shape, mu_shape)`** — helper for reroll
5. **`MaskUnitAttention`** — windowed multi-head attention with optional q max-pool
6. **`HieraMLP`** — fc1 -> GELU -> fc2
7. **`HieraBlock`** — norm1 -> [proj+maxpool if transition] -> attn + residual -> norm2 -> mlp + residual
8. **`HieraBackbone`** — full assembly: patch_embed + pos_embed + unroll + 24 blocks + reroll at stage_ends

### Sub-phase 4b: Converter updates

**File**: `src/corridorkey_mlx/convert/converter.py`

- Encoder keys are already handled (patch_embed.proj in CONV_WEIGHT_KEYS, all else passthrough)
- No changes needed — verified all 297 keys pass through correctly

### Sub-phase 4c: Weight loading + pos_embed interpolation

**In**: `src/corridorkey_mlx/model/hiera.py` (method on HieraBackbone)

- Load safetensors weights into MLX model
- Bicubic interpolation for pos_embed: (1, 262144, 112) -> (1, N', 112)
  - Reshape to (1, H_train, W_train, 112) -> bilinear/bicubic resize -> flatten

### Sub-phase 4d: Tests

**Files**:
- `tests/test_hiera_stage_shapes.py` — shape contract tests (no checkpoint needed)
- `tests/test_hiera_stage_parity.py` — numerical parity vs golden fixtures

Shape tests:
- PatchEmbed output shape
- Each stage feature map shape (128x128x112, 64x64x224, 32x32x448, 16x16x896)
- Correct number of features (4)

Parity tests (require checkpoint + fixtures):
- Load golden `encoder_feature_{0-3}` from `reference/fixtures/golden.npz`
- Load checkpoint weights into MLX backbone
- Compare each stage output (NCHW fixtures -> NHWC comparison)
- Report max_abs and mean_abs error per stage

## Key Files

| File | Action |
|------|--------|
| `src/corridorkey_mlx/model/hiera.py` | create (replace placeholder `backbone.py`) |
| `src/corridorkey_mlx/model/backbone.py` | keep as thin re-export |
| `tests/test_hiera_stage_shapes.py` | create |
| `tests/test_hiera_stage_parity.py` | create |

## Key Decisions

1. **Faithful Unroll/Reroll port** — required for exact parity
2. **File naming**: `hiera.py` — more descriptive; `backbone.py` stays as re-export
3. **No masking support** — inference only, skip masked token paths
4. **pos_embed interpolation at load time** — converter outputs raw checkpoint pos_embed
5. **All backbone ops in [B, N, C]** — NHWC only at boundaries (input image, output features)
6. **MLX key prefix**: `encoder.model.` stripped when loading into HieraBackbone

## Verification

```bash
uv run pytest tests/test_hiera_stage_shapes.py -v
uv run pytest tests/test_hiera_stage_parity.py -v -s
uv run ruff check src/corridorkey_mlx/model/hiera.py
uv run mypy src/corridorkey_mlx/model/hiera.py
```

## Unresolved Questions

- E2e parity tolerance? Expecting ~1e-4 max_abs, Metal float32 may drift more through 24 blocks
- `mx.fast.scaled_dot_product_attention` availability/API — fallback to manual if needed
