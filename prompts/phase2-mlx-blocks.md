Work only on the MLX implementations of the custom non-backbone blocks.

Goal:
Implement MLX versions of:

- MLP
- DecoderHead
- RefinerBlock
- CNNRefinerModule

Requirements:

- Use MLX idioms and explicit NHWC handling.
- Centralize tensor layout transforms in one utility module.
- Use pytorch-compatible GroupNorm behavior where needed for parity.
- Write parity tests that use saved PyTorch backbone features and saved coarse predictions.
- Report max abs error and mean abs error in test output or helper scripts.

Do not:

- port Hiera yet
- optimize prematurely
- spread layout conversions across many files

Definition of done:

- decoder parity test exists
- refiner parity test exists
- modules are wired into a partial model path for test usage
