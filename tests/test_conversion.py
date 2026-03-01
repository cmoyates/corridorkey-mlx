"""Tests: weight conversion PyTorch → MLX (Phase 3).

Validates key mapping, shape transforms, and round-trip integrity.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from collections import OrderedDict

from corridorkey_mlx.convert.converter import (
    REFINER_STEM_MAP,
    SKIP_SUFFIXES,
    convert_state_dict,
    load_pytorch_checkpoint,
)

CHECKPOINT_PATH = Path("checkpoints/CorridorKey_v1.0.pth")
SAFETENSORS_PATH = Path("checkpoints/corridorkey_mlx.safetensors")


@pytest.fixture(scope="module")
def checkpoint_data() -> (
    tuple[dict[str, np.ndarray], OrderedDict[str, np.ndarray], list]
):
    """Load and convert checkpoint once for all tests in this module."""
    if not CHECKPOINT_PATH.exists():
        pytest.skip("Checkpoint not found — need CorridorKey_v1.0.pth")
    state_dict = load_pytorch_checkpoint(CHECKPOINT_PATH)
    converted, diagnostics = convert_state_dict(state_dict)
    return state_dict, converted, diagnostics


# ---------------------------------------------------------------------------
# Key mapping completeness
# ---------------------------------------------------------------------------
class TestKeyMapping:
    """Every PyTorch key maps to an MLX key with no orphans."""

    def test_no_orphan_source_keys(self, checkpoint_data) -> None:
        """All source keys are either mapped or explicitly skipped."""
        state_dict, _, diagnostics = checkpoint_data

        mapped_src_keys = {r.source_key for r in diagnostics}
        skipped_keys = {k for k in state_dict if any(k.endswith(s) for s in SKIP_SUFFIXES)}
        accounted_for = mapped_src_keys | skipped_keys

        orphans = set(state_dict.keys()) - accounted_for
        assert orphans == set(), f"Orphan source keys: {orphans}"

    def test_no_duplicate_dest_keys(self, checkpoint_data) -> None:
        """No two source keys map to the same destination key."""
        _, _, diagnostics = checkpoint_data

        dest_keys = [r.dest_key for r in diagnostics]
        assert len(dest_keys) == len(set(dest_keys)), "Duplicate destination keys found"

    def test_refiner_stem_remapped(self, checkpoint_data) -> None:
        """Refiner stem Sequential keys correctly remapped to named attrs."""
        _, converted, _ = checkpoint_data

        for pt_key, mlx_key in REFINER_STEM_MAP.items():
            assert mlx_key in converted, f"Expected remapped key {mlx_key} not found"
            assert pt_key not in converted, f"Original key {pt_key} should not be in output"

    def test_num_batches_tracked_dropped(self, checkpoint_data) -> None:
        """BatchNorm num_batches_tracked keys are not in output."""
        _, converted, _ = checkpoint_data

        for key in converted:
            assert "num_batches_tracked" not in key, f"Found num_batches_tracked: {key}"

    def test_expected_key_count(self, checkpoint_data) -> None:
        """Output has expected number of keys (367 source - 2 skipped = 365)."""
        state_dict, converted, _ = checkpoint_data

        source_count = len(state_dict)
        skip_count = sum(1 for k in state_dict if any(k.endswith(s) for s in SKIP_SUFFIXES))
        expected = source_count - skip_count
        assert len(converted) == expected, f"Expected {expected} keys, got {len(converted)}"


# ---------------------------------------------------------------------------
# Conv weight transpose
# ---------------------------------------------------------------------------
class TestConvWeightTranspose:
    """Conv weights correctly transposed from NCHW → NHWC."""

    def test_patch_embed_conv_shape(self, checkpoint_data) -> None:
        """Patch embed conv transposed: (112,4,7,7) → (112,7,7,4)."""
        _, converted, _ = checkpoint_data

        key = "encoder.model.patch_embed.proj.weight"
        assert key in converted
        assert converted[key].shape == (112, 7, 7, 4)

    def test_decoder_conv_shapes(self, checkpoint_data) -> None:
        """Decoder conv weights transposed correctly."""
        _, converted, _ = checkpoint_data

        # linear_fuse: (256, 1024, 1, 1) → (256, 1, 1, 1024)
        for prefix in ("alpha_decoder", "fg_decoder"):
            fuse_key = f"{prefix}.linear_fuse.weight"
            assert converted[fuse_key].shape == (256, 1, 1, 1024)

    def test_refiner_conv_shapes(self, checkpoint_data) -> None:
        """Refiner conv weights transposed correctly."""
        _, converted, _ = checkpoint_data

        # stem_conv: (64, 7, 3, 3) → (64, 3, 3, 7)
        assert converted["refiner.stem_conv.weight"].shape == (64, 3, 3, 7)

        # res block convs: (64, 64, 3, 3) → (64, 3, 3, 64)
        for i in range(1, 5):
            for j in range(1, 3):
                key = f"refiner.res{i}.conv{j}.weight"
                assert converted[key].shape == (64, 3, 3, 64), f"Wrong shape for {key}"

        # final: (4, 64, 1, 1) → (4, 1, 1, 64)
        assert converted["refiner.final.weight"].shape == (4, 1, 1, 64)

    def test_linear_weights_unchanged(self, checkpoint_data) -> None:
        """Linear (2D) weights are not transposed."""
        _, _, diagnostics = checkpoint_data

        passthrough_2d = [
            r for r in diagnostics if r.transform == "passthrough" and len(r.source_shape) == 2
        ]
        for record in passthrough_2d:
            assert record.source_shape == record.dest_shape, (
                f"{record.source_key}: shape changed despite passthrough"
            )


# ---------------------------------------------------------------------------
# 4-channel first conv
# ---------------------------------------------------------------------------
class TestFirstConv4Channel:
    """Patched 4-channel first conv preserved during conversion."""

    def test_input_channels_preserved(self, checkpoint_data) -> None:
        """Patch embed conv has 4 input channels (RGB + alpha hint)."""
        _, converted, _ = checkpoint_data

        weight = converted["encoder.model.patch_embed.proj.weight"]
        # MLX layout: (O, H, W, I) — I is last dim
        input_channels = weight.shape[-1]
        assert input_channels == 4, f"Expected 4 input channels, got {input_channels}"

    def test_values_preserved(self, checkpoint_data) -> None:
        """Conv values are preserved (only transposed, not modified)."""
        state_dict, converted, _ = checkpoint_data

        pt_weight = state_dict["encoder.model.patch_embed.proj.weight"]
        mlx_weight = converted["encoder.model.patch_embed.proj.weight"]

        # Transpose back to PyTorch layout and compare
        roundtrip = np.transpose(mlx_weight, (0, 3, 1, 2))
        np.testing.assert_array_equal(roundtrip, pt_weight)


# ---------------------------------------------------------------------------
# Safetensors output validation
# ---------------------------------------------------------------------------
class TestSafetensorsOutput:
    """Validate saved safetensors file matches conversion output."""

    def test_safetensors_loadable(self, checkpoint_data) -> None:
        """Saved safetensors file can be loaded and has correct keys."""
        if not SAFETENSORS_PATH.exists():
            pytest.skip("Run convert_weights.py first")

        from safetensors.numpy import load_file

        loaded = load_file(str(SAFETENSORS_PATH))

        _, converted, _ = checkpoint_data

        assert set(loaded.keys()) == set(converted.keys())
        for key in converted:
            np.testing.assert_array_equal(loaded[key], converted[key])
