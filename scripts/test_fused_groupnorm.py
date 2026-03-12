"""Test fused GroupNorm via reshape + layer_norm trick.

Instead of using nn.GroupNorm(pytorch_compatible=True) which transposes internally,
reshape (B,H,W,C) -> (B*H*W, G, C//G), normalize last dim, apply affine, reshape back.
This keeps everything contiguous in NHWC.

Key question: does this match pytorch_compatible=True output?
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np

REFINER_CHANNELS = 64
REFINER_GROUPS = 8
BATCH, HEIGHT, WIDTH = 1, 128, 128

mx.random.seed(42)
x = mx.random.normal((BATCH, HEIGHT, WIDTH, REFINER_CHANNELS))

# Reference: pytorch_compatible GroupNorm
gn_ref = nn.GroupNorm(REFINER_GROUPS, REFINER_CHANNELS, pytorch_compatible=True)
mx.eval(gn_ref.parameters())

out_ref = gn_ref(x)
mx.eval(out_ref)

# Custom fused implementation
C = REFINER_CHANNELS
G = REFINER_GROUPS
CpG = C // G  # 8
eps = 1e-5

weight = gn_ref.weight  # (C,)
bias = gn_ref.bias      # (C,)

# Approach 1: Reshape to (B*H*W, G, C//G), normalize over last dim
# This computes per-group, per-spatial-location stats
x_flat = x.reshape(-1, G, CpG)  # (B*H*W, G, C//G) -- zero-copy reshape
mean = mx.mean(x_flat, axis=-1, keepdims=True)  # (B*H*W, G, 1)
var = mx.var(x_flat, axis=-1, keepdims=True)     # (B*H*W, G, 1)
x_norm = (x_flat - mean) / mx.sqrt(var + eps)     # (B*H*W, G, C//G)

# Apply affine: weight/bias reshaped to (1, G, C//G)
w = weight.reshape(1, G, CpG)
b = bias.reshape(1, G, CpG)
out_fused = (x_norm * w + b).reshape(BATCH, HEIGHT, WIDTH, C)
mx.eval(out_fused)

diff1 = np.abs(np.array(out_ref) - np.array(out_fused))
print("Approach 1: reshape (B*H*W, G, C//G) + normalize last dim")
print(f"  max_abs_diff vs pytorch_compatible: {diff1.max():.6e}")
print(f"  mean_abs_diff: {diff1.mean():.6e}")
print(f"  match (<1e-5): {diff1.max() < 1e-5}")

# Approach 2: Same but pytorch-style -- group over (H*W*C//G) elements
# PyTorch computes mean/var over the spatial AND channel-per-group dims together
# i.e., for group g: stats over all (h, w) and channels [g*CpG : (g+1)*CpG]
x_grouped = x.reshape(BATCH, HEIGHT * WIDTH, G, CpG)  # (B, H*W, G, C//G)
x_grouped = mx.transpose(x_grouped, axes=(0, 2, 1, 3))  # (B, G, H*W, C//G)
x_grouped = x_grouped.reshape(BATCH, G, -1)  # (B, G, H*W*C//G)

mean2 = mx.mean(x_grouped, axis=-1, keepdims=True)  # (B, G, 1)
var2 = mx.var(x_grouped, axis=-1, keepdims=True)     # (B, G, 1)
x_norm2 = (x_grouped - mean2) / mx.sqrt(var2 + eps)   # (B, G, H*W*C//G)

# Reshape back and apply affine
x_norm2 = x_norm2.reshape(BATCH, G, HEIGHT * WIDTH, CpG)
x_norm2 = mx.transpose(x_norm2, axes=(0, 2, 1, 3))  # (B, H*W, G, C//G)
x_norm2 = x_norm2.reshape(BATCH, HEIGHT, WIDTH, C)
out_fused2 = x_norm2 * weight + bias
mx.eval(out_fused2)

diff2 = np.abs(np.array(out_ref) - np.array(out_fused2))
print("\nApproach 2: pytorch-style stats over (H*W, C//G) per group")
print(f"  max_abs_diff vs pytorch_compatible: {diff2.max():.6e}")
print(f"  mean_abs_diff: {diff2.mean():.6e}")
print(f"  match (<1e-5): {diff2.max() < 1e-5}")

# Approach 3: No transpose -- use native NHWC grouping with pytorch-style stats
# Compute stats per group across ALL spatial positions (not per-position)
# Reshape: (B, H, W, C) -> (B, H*W, G, C//G) -> stats over axes (1, 3)
x_g3 = x.reshape(BATCH, HEIGHT * WIDTH, G, CpG)
mean3 = mx.mean(x_g3, axis=(1, 3), keepdims=True)  # (B, 1, G, 1)
var3 = mx.var(x_g3, axis=(1, 3), keepdims=True)     # (B, 1, G, 1)
x_norm3 = (x_g3 - mean3) / mx.sqrt(var3 + eps)
out_fused3 = (x_norm3.reshape(BATCH, HEIGHT, WIDTH, C)) * weight + bias
mx.eval(out_fused3)

diff3 = np.abs(np.array(out_ref) - np.array(out_fused3))
print("\nApproach 3: NHWC native -- stats over axes (spatial, C//G) per group, no transpose")
print(f"  max_abs_diff vs pytorch_compatible: {diff3.max():.6e}")
print(f"  mean_abs_diff: {diff3.mean():.6e}")
print(f"  match (<1e-5): {diff3.max() < 1e-5}")
