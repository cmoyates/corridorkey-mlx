"""Weight conversion tests through public API.

Reworked from internal convert_state_dict tests to use convert_checkpoint().
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from corridorkey_mlx.convert.converter import convert_checkpoint
from corridorkey_mlx.model.corridorkey import GreenFormer

from .conftest import PT_CHECKPOINT_PATH, has_pt_checkpoint

if TYPE_CHECKING:
    from pathlib import Path

EXPECTED_KEY_COUNT = 365


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


has_torch = pytest.mark.skipif(not _torch_available(), reason="torch not installed")


@pytest.fixture(scope="module")
def converted_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Convert PT checkpoint to safetensors in temp dir."""
    if not PT_CHECKPOINT_PATH.exists():
        pytest.skip("PyTorch checkpoint not found")
    if not _torch_available():
        pytest.skip("torch not installed")
    output = tmp_path_factory.mktemp("weights") / "converted.safetensors"
    convert_checkpoint(PT_CHECKPOINT_PATH, output)
    return output


@has_pt_checkpoint
@has_torch
class TestConvertCheckpoint:
    def test_output_file_exists(self, converted_path: Path) -> None:
        assert converted_path.exists()

    def test_safetensors_loadable(self, converted_path: Path) -> None:
        from safetensors.numpy import load_file

        weights = load_file(str(converted_path))
        assert len(weights) > 0

    def test_key_count(self, converted_path: Path) -> None:
        from safetensors.numpy import load_file

        weights = load_file(str(converted_path))
        assert len(weights) == EXPECTED_KEY_COUNT

    def test_model_loads_and_runs(self, converted_path: Path) -> None:
        """Roundtrip: convert -> load into GreenFormer -> forward succeeds."""
        import mlx.core as mx

        model = GreenFormer(img_size=256)
        model.load_checkpoint(converted_path)

        x = mx.random.normal((1, 256, 256, 4))
        out = model(x)
        # mx.eval is MLX array materialization, not Python eval()
        mx.eval(out)  # noqa: S307

        assert out["alpha_final"].shape == (1, 256, 256, 1)
        assert out["fg_final"].shape == (1, 256, 256, 3)

        # No NaN
        alpha = np.array(out["alpha_final"])
        assert not np.isnan(alpha).any()
