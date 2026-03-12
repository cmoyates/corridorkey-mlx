"""Test whether GroupNorm pytorch_compatible=True vs False produces identical output.

For the refiner's specific config (C=64, G=8), the channel grouping
may be identical in both modes since NHWC groups channels along the last dim.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np

REFINER_CHANNELS = 64
REFINER_GROUPS = 8
BATCH, HEIGHT, WIDTH = 1, 128, 128

mx.random.seed(42)
x = mx.random.normal((BATCH, HEIGHT, WIDTH, REFINER_CHANNELS))

# Create two GroupNorm layers with same weights
gn_compat = nn.GroupNorm(REFINER_GROUPS, REFINER_CHANNELS, pytorch_compatible=True)
gn_native = nn.GroupNorm(REFINER_GROUPS, REFINER_CHANNELS, pytorch_compatible=False)

# Copy weights from compat to native (same initialization)
gn_native.weight = gn_compat.weight
gn_native.bias = gn_compat.bias

# Materialize parameters
mx.eval(gn_compat.parameters(), gn_native.parameters())

out_compat = gn_compat(x)
out_native = gn_native(x)

# Materialize outputs
mx.eval(out_compat, out_native)

diff = np.abs(np.array(out_compat) - np.array(out_native))
print(f"pytorch_compatible=True vs False:")
print(f"  max_abs_diff: {diff.max():.6e}")
print(f"  mean_abs_diff: {diff.mean():.6e}")
print(f"  identical: {diff.max() == 0.0}")

if diff.max() > 0:
    # Try weight remapping: permute weights so native mode matches compat output
    # PyTorch compat groups: channels [0..C//G-1] are group 0, etc.
    # Native NHWC groups: same ordering (last dim is channels)
    # So they SHOULD be identical. If not, investigate.
    print("\n  Outputs differ! Investigating channel ordering...")

    # Check a single spatial position
    pos_compat = np.array(out_compat[0, 0, 0, :])
    pos_native = np.array(out_native[0, 0, 0, :])
    print(f"  compat[0,0,0,:8]: {pos_compat[:8]}")
    print(f"  native[0,0,0,:8]: {pos_native[:8]}")

    # Check if it's a permutation issue
    # Try reordering native output channels to match compat
    C = REFINER_CHANNELS
    G = REFINER_GROUPS
    CpG = C // G  # channels per group

    # Test: are the per-group stats different?
    x_np = np.array(x[0, 0, 0, :])
    for g in range(min(G, 3)):
        sl = slice(g * CpG, (g + 1) * CpG)
        print(f"  group {g} input mean: {x_np[sl].mean():.6f}")
        print(f"  group {g} compat out: {pos_compat[sl].mean():.6f}")
        print(f"  group {g} native out: {pos_native[sl].mean():.6f}")
else:
    print("\n  GroupNorm outputs are IDENTICAL — pytorch_compatible=True is unnecessary!")
    print("  Safe to drop the flag for free performance.")
