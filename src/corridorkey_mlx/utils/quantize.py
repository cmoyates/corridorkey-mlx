"""Safe quantization helpers for MLX modules."""

import mlx.nn as nn


def safe_quantize(
    module: nn.Module,
    *,
    group_size: int = 32,
    bits: int = 8,
) -> int:
    """Quantize a module, skipping Linear layers with incompatible dimensions.

    MLX ``mx.quantize`` requires the last dimension of each Linear layer's
    weight to be divisible by ``group_size``.  Early Hiera stages use dim=112,
    which is not divisible by 32, causing a ``ValueError``.  This wrapper
    walks the module tree and only quantizes compatible leaf layers.

    Returns the number of layers successfully quantized.
    """
    quantized_count = 0
    for _name, child in module.named_modules():
        if isinstance(child, nn.Linear):
            last_dim = child.weight.shape[-1]
            if last_dim % group_size != 0:
                continue
            nn.quantize(child, group_size=group_size, bits=bits)
            quantized_count += 1
    return quantized_count
