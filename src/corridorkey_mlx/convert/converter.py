"""PyTorch → MLX weight converter.

Explicit key-by-key mapping from CorridorKey PyTorch checkpoint
to MLX-compatible weights. No regex guessing, no silent fallbacks.

Transforms applied:
- Conv2d weights: (O, I, H, W) → (O, H, W, I)
- Refiner stem: Sequential indices → named attrs (stem.0 → stem_conv, stem.1 → stem_gn)
- BatchNorm num_batches_tracked: dropped (MLX doesn't use it)
- All other weights: passed through unchanged

Output: safetensors file with transformed weights.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COMPILE_PREFIX = "_orig_mod."

# Refiner stem key remapping: PyTorch nn.Sequential indices → MLX named attrs
REFINER_STEM_MAP: dict[str, str] = {
    "refiner.stem.0.weight": "refiner.stem_conv.weight",
    "refiner.stem.0.bias": "refiner.stem_conv.bias",
    "refiner.stem.1.weight": "refiner.stem_gn.weight",
    "refiner.stem.1.bias": "refiner.stem_gn.bias",
}

# Keys to skip entirely (not needed by MLX)
SKIP_SUFFIXES = (".num_batches_tracked",)


# ---------------------------------------------------------------------------
# Diagnostic record
# ---------------------------------------------------------------------------
@dataclass
class ConversionRecord:
    """Single key conversion diagnostic."""

    source_key: str
    dest_key: str
    source_shape: tuple[int, ...]
    dest_shape: tuple[int, ...]
    transform: str


# ---------------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------------
def _is_conv_weight(key: str, arr: np.ndarray) -> bool:
    """Check if a weight is a conv kernel (4D with spatial dims > 1)."""
    if arr.ndim != 4:
        return False
    # 4D with spatial dims — conv weight, not a weird batch param
    # Also check suffix to avoid false positives
    return key.endswith(".weight")


def _transpose_conv_weight(arr: np.ndarray) -> np.ndarray:
    """PyTorch conv (O, I, H, W) → MLX conv (O, H, W, I)."""
    return np.transpose(arr, (0, 2, 3, 1))


def _remap_key(key: str) -> str:
    """Apply key remapping rules. Returns new key."""
    if key in REFINER_STEM_MAP:
        return REFINER_STEM_MAP[key]
    return key


def _should_skip(key: str) -> bool:
    """Check if key should be dropped from output."""
    return any(key.endswith(suffix) for suffix in SKIP_SUFFIXES)


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------
def convert_state_dict(
    state_dict: dict[str, np.ndarray],
) -> tuple[OrderedDict[str, np.ndarray], list[ConversionRecord]]:
    """Convert PyTorch state_dict to MLX-compatible weight dict.

    Args:
        state_dict: PyTorch state_dict as numpy arrays (already stripped of _orig_mod. prefix).

    Returns:
        Tuple of (converted weights dict, list of diagnostic records).

    Raises:
        ValueError: If duplicate destination keys are produced.
    """
    converted: OrderedDict[str, np.ndarray] = OrderedDict()
    diagnostics: list[ConversionRecord] = []
    skipped: list[tuple[str, str]] = []

    for src_key, arr in state_dict.items():
        # Skip keys MLX doesn't need
        if _should_skip(src_key):
            skipped.append((src_key, "num_batches_tracked — not used by MLX"))
            continue

        # Remap key
        dest_key = _remap_key(src_key)

        # Apply transform
        src_shape = arr.shape
        if _is_conv_weight(src_key, arr):
            transformed = _transpose_conv_weight(arr)
            transform_name = "conv_transpose(O,I,H,W→O,H,W,I)"
        else:
            transformed = arr
            transform_name = "passthrough"

        # Check for duplicates
        if dest_key in converted:
            msg = f"Duplicate dest key: {dest_key} (from {src_key})"
            raise ValueError(msg)

        converted[dest_key] = transformed
        diagnostics.append(
            ConversionRecord(
                source_key=src_key,
                dest_key=dest_key,
                source_shape=src_shape,
                dest_shape=transformed.shape,
                transform=transform_name,
            )
        )

    return converted, diagnostics


# ---------------------------------------------------------------------------
# Checkpoint loading (PyTorch → numpy)
# ---------------------------------------------------------------------------
def load_pytorch_checkpoint(checkpoint_path: Path) -> dict[str, np.ndarray]:
    """Load PyTorch checkpoint and return state_dict as numpy arrays.

    Strips _orig_mod. prefix from torch.compile checkpoints.

    Requires torch to be installed (reference dependency group).
    """
    import torch

    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = raw.get("state_dict", raw)

    result: dict[str, np.ndarray] = {}
    for key, tensor in state_dict.items():
        clean_key = key.removeprefix(COMPILE_PREFIX)
        result[clean_key] = tensor.numpy()

    return result


# ---------------------------------------------------------------------------
# Safetensors output
# ---------------------------------------------------------------------------
def save_safetensors(weights: OrderedDict[str, np.ndarray], output_path: Path) -> None:
    """Save converted weights as safetensors."""
    from safetensors.numpy import save_file

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(weights, str(output_path))


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------
def convert_checkpoint(
    checkpoint_path: Path,
    output_path: Path,
) -> list[ConversionRecord]:
    """Full conversion pipeline: load PyTorch checkpoint → convert → save safetensors.

    Args:
        checkpoint_path: Path to PyTorch .pth checkpoint.
        output_path: Path for output .safetensors file.

    Returns:
        List of conversion diagnostic records.
    """
    state_dict = load_pytorch_checkpoint(checkpoint_path)
    converted, diagnostics = convert_state_dict(state_dict)
    save_safetensors(converted, output_path)
    return diagnostics
