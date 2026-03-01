"""Tests: weight conversion PyTorch → MLX (Phase 3).

Validates key mapping, shape transforms, and round-trip integrity.
"""

import pytest

pytestmark = pytest.mark.skip(reason="Phase 3: converter not yet implemented")


def test_key_mapping_complete() -> None:
    """Every PyTorch key maps to an MLX key with no orphans."""
    ...


def test_conv_weight_transpose() -> None:
    """Conv weights correctly transposed from NCHW → NHWC."""
    ...


def test_first_conv_4ch_preserved() -> None:
    """Patched 4-channel first conv preserved during conversion."""
    ...
