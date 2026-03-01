Work only on checkpoint conversion.

Goal:
Create a robust conversion pipeline from the PyTorch CorridorKey checkpoint to MLX-compatible weights.

Requirements:

- Inspect state_dict keys and map them explicitly.
- Convert conv weights from PyTorch layout to MLX layout.
- Preserve the patched 4-channel first conv behavior exactly.
- Write converter diagnostics:
  - source key
  - destination key
  - source shape
  - destination shape
  - transform applied
- Save output as safetensors or npz.
- Validate load with MLX strict loading wherever possible.

Do not:

- attempt full end-to-end model parity yet if Hiera is incomplete
- hide key mismatches
- use silent fallbacks

Definition of done:

- converter script exists
- mapping file exists
- shape validation passes for completed modules
- conversion report is readable and auditable
