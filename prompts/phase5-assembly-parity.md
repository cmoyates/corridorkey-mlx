Work only on full model assembly and end-to-end inference parity.

Goal:
Assemble the MLX GreenFormer end-to-end by wiring:

- Hiera backbone
- alpha decoder head
- foreground decoder head
- upsampling to full resolution
- refiner input construction
- additive delta-logit refinement
- final sigmoid outputs

Context:

- Decoder and refiner modules already exist in MLX.
- Backbone work from phase 4 should now provide the expected 4 feature maps.
- This phase is about reproducing the original forward pass structure and validating end-to-end outputs against the PyTorch oracle.

Primary deliverables:

- src/corridorkey_mlx/model/greenformer.py
- src/corridorkey_mlx/inference/pipeline.py
- tests/test_greenformer_forward.py
- tests/test_end_to_end_smoke.py
- tests/test_end_to_end_parity.py
- README usage section for single-image inference

Requirements:

1. Recreate the PyTorch forward pass structure exactly:
   - 4-channel input
   - backbone feature extraction
   - dual decoder heads
   - coarse alpha / foreground logits
   - upsample coarse logits to input resolution
   - apply sigmoid to obtain coarse probabilities
   - concatenate RGB + coarse predictions into the 7-channel refiner input
   - predict delta logits
   - add delta logits before final sigmoid
2. Preserve the semantic distinction between:
   - coarse logits
   - coarse probabilities
   - delta logits
   - final probabilities
3. Do not collapse or reorder operations unless the PyTorch oracle proves equivalence.
4. Add parity checks for:
   - coarse alpha logits
   - coarse foreground logits
   - coarse alpha probabilities
   - coarse foreground probabilities
   - refiner input tensor
   - delta logits
   - final alpha
   - final foreground
5. Make inference code explicit about layout conversions and output formats.
6. Support reduced-resolution testing first if needed, but document any deviation from the native target resolution.
7. Provide a simple CLI or script entry point for:
   - loading MLX weights
   - running one image + alpha hint
   - saving or printing a concise summary of output tensors
8. Ensure the model can load completed weights with strict checking wherever practical.
9. Keep preprocessing and postprocessing small and auditable.

Diagnostics to produce:

- stage name
- tensor shape
- tensor dtype
- layout convention
- max abs error
- mean abs error
- whether mismatch appears first in coarse path or refinement path

Do not:

- start performance tuning yet
- introduce tiling unless necessary for a basic smoke path
- expand into video orchestration
- add training/fine-tuning work
- hide parity gaps behind broad tolerances

Working style:

- Before editing, summarize the exact full forward-pass contract in plain English.
- Implement the minimal wiring required for correctness.
- Run narrow forward tests first, then end-to-end smoke, then parity tests.
- If parity fails, identify the earliest divergent artifact and focus there.

Definition of done:

- MLX GreenFormer forward pass is assembled
- end-to-end smoke test passes
- parity tests exist for coarse path, refiner path, and final outputs
- single-image inference entry point works
- README documents current supported inference workflow
