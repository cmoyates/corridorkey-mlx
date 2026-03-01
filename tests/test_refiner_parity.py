"""Parity tests: MLX refiner vs PyTorch reference (Phase 2).

Uses saved coarse predictions + RGB → runs MLX refiner →
compares against saved PyTorch final outputs.
"""

import pytest

pytestmark = pytest.mark.skip(reason="Phase 2: MLX refiner not yet implemented")


def test_refiner_delta_parity() -> None:
    """MLX refiner delta logits match PyTorch within tolerance."""
    ...


def test_refiner_final_output_parity() -> None:
    """MLX final alpha and fg match PyTorch within tolerance."""
    ...
