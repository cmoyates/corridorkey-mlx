"""Parity tests: MLX decoder heads vs PyTorch reference (Phase 2).

Uses saved backbone features → runs MLX decoder → compares
against saved PyTorch coarse predictions.
"""

import pytest

pytestmark = pytest.mark.skip(reason="Phase 2: MLX decoder not yet implemented")


def test_alpha_decoder_parity() -> None:
    """MLX alpha decoder matches PyTorch within tolerance."""
    ...


def test_fg_decoder_parity() -> None:
    """MLX foreground decoder matches PyTorch within tolerance."""
    ...
