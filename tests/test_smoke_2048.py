"""2048 smoke tests — execution check at native resolution.

test_smoke_2048_wiring: lightweight, no checkpoint, runs in normal suite.
test_smoke_2048_full: loads real checkpoint at 2048, marked slow + skipif.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from corridorkey_mlx.model.corridorkey import GreenFormer

CHECKPOINT = Path("checkpoints/corridorkey_mlx.safetensors")
HAS_CHECKPOINT = CHECKPOINT.exists()

WIRING_IMG_SIZE = 256  # small enough to run fast with random weights


def test_smoke_2048_wiring() -> None:
    """GreenFormer constructs at 2048 and produces correct shapes (random weights)."""
    model = GreenFormer(img_size=WIRING_IMG_SIZE)
    x = mx.random.normal((1, WIRING_IMG_SIZE, WIRING_IMG_SIZE, 4))
    # mx.eval materializes lazy MLX arrays (not Python eval)
    mx.eval(x)  # noqa: S307

    out = model(x)
    mx.eval(out)  # noqa: S307

    assert out["alpha_final"].shape == (1, WIRING_IMG_SIZE, WIRING_IMG_SIZE, 1)
    assert out["fg_final"].shape == (1, WIRING_IMG_SIZE, WIRING_IMG_SIZE, 3)

    # No NaN/Inf
    alpha = np.array(out["alpha_final"])
    fg = np.array(out["fg_final"])
    assert not np.isnan(alpha).any(), "alpha_final contains NaN"
    assert not np.isnan(fg).any(), "fg_final contains NaN"
    assert not np.isinf(alpha).any(), "alpha_final contains Inf"
    assert not np.isinf(fg).any(), "fg_final contains Inf"


@pytest.mark.slow
@pytest.mark.skipif(not HAS_CHECKPOINT, reason="Checkpoint not available")
def test_smoke_2048_full() -> None:
    """Full 2048 inference with real checkpoint via engine."""
    from corridorkey_mlx import CorridorKeyMLXEngine

    engine = CorridorKeyMLXEngine(
        checkpoint_path=CHECKPOINT,
        img_size=2048,
        compile=False,  # skip compile overhead in test
    )

    # Use synthetic inputs at 2048
    rng = np.random.default_rng(42)
    rgb = rng.integers(0, 256, (2048, 2048, 3), dtype=np.uint8)
    mask = rng.integers(0, 256, (2048, 2048), dtype=np.uint8)

    result = engine.process_frame(rgb, mask)

    # Shape checks
    assert result["alpha"].shape == (2048, 2048)
    assert result["alpha"].dtype == np.uint8
    assert result["fg"].shape == (2048, 2048, 3)
    assert result["fg"].dtype == np.uint8
    assert result["comp"].shape == (2048, 2048, 3)

    # No NaN/Inf
    for key in ("alpha", "fg", "comp"):
        arr = result[key]
        assert not np.isnan(arr).any(), f"{key} contains NaN"
        assert not np.isinf(arr).any(), f"{key} contains Inf"

    # Alpha shouldn't be degenerate
    assert result["alpha"].min() != result["alpha"].max(), "alpha is constant — suspicious"
