Work only on the Hiera backbone port and its stage-by-stage parity.

Goal:
Implement an MLX Hiera backbone that reproduces the feature contract expected by CorridorKey’s downstream heads.

Context:

- The original PyTorch model uses a timm backbone created with features_only=True.
- The downstream model expects a pyramid of 4 multiscale feature maps.
- The first input projection is patched from 3 input channels to 4 input channels for RGB + alpha hint.
- This phase is about reproducing backbone behavior and feature outputs, not full end-to-end inference.

Primary deliverables:

- src/corridorkey_mlx/model/hiera.py
- tests/test_hiera_stage_shapes.py
- tests/test_hiera_stage_parity.py
- updates to converter mapping files as needed
- README notes describing the backbone parity status and any known gaps

Requirements:

1. Inspect the original CorridorKey backbone usage and the underlying Hiera/timm feature contract before editing.
2. Treat the PyTorch reference harness as the source of truth for:
   - number of feature stages
   - output order
   - channel counts
   - spatial reductions
3. Implement the smallest MLX Hiera subset necessary to produce the correct 4 feature maps.
4. Preserve the patched 4-channel patch-embed behavior exactly:
   - do not invent a new initialization rule
   - match the PyTorch conversion semantics for the extra alpha-hint channel
5. Keep tensor layout handling explicit and centralized:
   - use one canonical boundary between PyTorch-style NCHW fixtures and MLX NHWC internals
   - do not scatter transpose logic across the backbone code
6. Add narrow parity checks for:
   - patch embed output
   - each stage output
   - final list of multiscale features
7. If exact numerical parity is not immediately possible, first achieve:
   - correct stage count
   - correct stage order
   - correct shapes
   - correct dtype flow
     then localize the first divergent block
8. Keep the implementation modular:
   - patch embed
   - stage/block definition
   - downsampling / pooling transitions
   - feature collection
9. Update the checkpoint conversion path only as needed to support the backbone weights for completed modules.

Diagnostics to produce:

- stage name
- source tensor shape
- destination tensor shape
- layout assumption
- max abs error
- mean abs error
- whether the mismatch begins before or after a stage transition

Do not:

- wire the full GreenFormer end-to-end yet
- rewrite decoder/refiner code unless required by an interface mismatch
- optimize with compile() yet
- add training code
- silently accept stage reordering or “close enough” feature contracts

Working style:

- Explore first.
- Summarize the exact Hiera feature contract you believe CorridorKey expects before editing.
- After edits, run the narrowest backbone-only tests first.
- When parity fails, stop at the first divergent stage and explain the likely cause before changing more code.

Definition of done:

- MLX backbone returns 4 multiscale feature maps in the correct order
- shape tests pass
- stage-level parity tests exist and run
- the 4-channel patch embed path is implemented and documented
- converter mappings exist for completed backbone weights
