"""PyTorch → MLX weight converter (not yet implemented).

Responsibilities:
- Map state_dict keys between PyTorch and MLX naming
- Transpose conv weights from NCHW → NHWC
- Handle patched 4-channel first conv
- Emit diagnostic report (src key, dst key, shapes, transform)
- Save as safetensors
"""
