"""Tests for FrozenGroupNorm and frozen-stats tiled refiner."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from corridorkey_mlx.model.refiner import (
    REFINER_CHANNELS,
    REFINER_GROUPS,
    CNNRefinerModule,
    FrozenGroupNorm,
)


class TestFrozenGroupNormParity:
    """FrozenGroupNorm normal mode must match nn.GroupNorm(pytorch_compatible=True)."""

    def test_parity_with_nn_groupnorm(self) -> None:
        """Normal forward vs nn.GroupNorm — should be exact match."""
        frozen_gn = FrozenGroupNorm(REFINER_GROUPS, REFINER_CHANNELS)
        ref_gn = nn.GroupNorm(REFINER_GROUPS, REFINER_CHANNELS, pytorch_compatible=True)

        # Copy weights
        ref_gn.weight = frozen_gn.weight
        ref_gn.bias = frozen_gn.bias

        x = mx.random.normal((1, 32, 32, REFINER_CHANNELS))
        # mx.eval: MLX array materialization
        mx.eval(x)  # noqa: S307

        out_frozen = frozen_gn(x)
        out_ref = ref_gn(x)
        # mx.eval: MLX array materialization
        mx.eval(out_frozen, out_ref)  # noqa: S307

        diff = float(mx.max(mx.abs(out_frozen - out_ref)))
        # Metal kernel uses different reduction order than transpose+layer_norm —
        # tiny fp32 precision difference (~1e-7) is expected and harmless
        assert diff < 1e-5, f"Normal mode diff={diff}, expected < 1e-5"

    def test_parity_with_random_weights(self) -> None:
        """Parity with non-default (random) weight/bias."""
        frozen_gn = FrozenGroupNorm(REFINER_GROUPS, REFINER_CHANNELS)
        ref_gn = nn.GroupNorm(REFINER_GROUPS, REFINER_CHANNELS, pytorch_compatible=True)

        w = mx.random.normal((REFINER_CHANNELS,))
        b = mx.random.normal((REFINER_CHANNELS,))
        frozen_gn.weight = w
        frozen_gn.bias = b
        ref_gn.weight = w
        ref_gn.bias = b

        x = mx.random.normal((2, 16, 16, REFINER_CHANNELS))
        # mx.eval: MLX array materialization
        mx.eval(x)  # noqa: S307

        out_frozen = frozen_gn(x)
        out_ref = ref_gn(x)
        # mx.eval: MLX array materialization
        mx.eval(out_frozen, out_ref)  # noqa: S307

        diff = float(mx.max(mx.abs(out_frozen - out_ref)))
        # Metal kernel uses fp32 throughout vs layer_norm's mixed precision —
        # small difference expected, harmless at pipeline level
        assert diff < 1e-4, f"Random weights diff={diff}, expected < 1e-4"


class TestFrozenStatsIdentity:
    """Collecting stats then freezing must reproduce the same output."""

    def test_frozen_matches_unfrozen(self) -> None:
        """Collect on X, freeze, forward on same X — exact match."""
        gn = FrozenGroupNorm(REFINER_GROUPS, REFINER_CHANNELS)
        x = mx.random.normal((1, 64, 64, REFINER_CHANNELS))
        # mx.eval: MLX array materialization
        mx.eval(x)  # noqa: S307

        # Normal forward
        out_normal = gn(x)
        # mx.eval: MLX array materialization
        mx.eval(out_normal)  # noqa: S307

        # Collecting forward
        gn._collecting = True
        out_collecting = gn(x)
        # mx.eval: MLX array materialization
        mx.eval(out_collecting)  # noqa: S307
        gn._collecting = False

        # Collecting output should match normal (manual vs fused may differ slightly)
        diff_collecting = float(mx.max(mx.abs(out_collecting - out_normal)))
        assert diff_collecting < 1e-5, f"Collecting vs normal diff={diff_collecting}"

        # Freeze and forward again
        assert gn._collected_stats is not None
        gn._frozen_stats = gn._collected_stats
        out_frozen = gn(x)
        # mx.eval: MLX array materialization
        mx.eval(out_frozen)  # noqa: S307

        # Frozen output uses Metal kernel with converted stats — tiny fp32 roundtrip diff
        diff_frozen = float(mx.max(mx.abs(out_frozen - out_collecting)))
        assert diff_frozen < 1e-5, f"Frozen vs collecting diff={diff_frozen}"

        # Cleanup
        gn._frozen_stats = None
        gn._collected_stats = None


class TestFrozenStatsCrossSize:
    """Frozen stats from one spatial size applied to another."""

    def test_cross_size_no_crash(self) -> None:
        """Collect stats on (B,64,64,C), freeze, forward on (B,32,32,C) — no crash."""
        gn = FrozenGroupNorm(REFINER_GROUPS, REFINER_CHANNELS)
        x_large = mx.random.normal((1, 64, 64, REFINER_CHANNELS))
        x_small = mx.random.normal((1, 32, 32, REFINER_CHANNELS))
        # mx.eval: MLX array materialization
        mx.eval(x_large, x_small)  # noqa: S307

        # Collect on large
        gn._collecting = True
        gn(x_large)
        # mx.eval: MLX array materialization
        mx.eval(gn._collected_stats[0], gn._collected_stats[1])  # noqa: S307
        gn._collecting = False

        # Freeze and forward on small
        gn._frozen_stats = gn._collected_stats
        out = gn(x_small)
        # mx.eval: MLX array materialization
        mx.eval(out)  # noqa: S307

        assert out.shape == (1, 32, 32, REFINER_CHANNELS)
        # Output should be finite
        assert not mx.any(mx.isnan(out)).item()
        assert not mx.any(mx.isinf(out)).item()

        gn._frozen_stats = None
        gn._collected_stats = None


class TestCNNRefinerModuleStatsAPI:
    """Test the collect/freeze/unfreeze API on CNNRefinerModule."""

    @pytest.fixture()
    def refiner(self) -> CNNRefinerModule:
        r = CNNRefinerModule()
        r.eval()
        # mx.eval: MLX array materialization
        mx.eval(r.parameters())  # noqa: S307
        return r

    def test_all_groupnorms_count(self, refiner: CNNRefinerModule) -> None:
        """Should return exactly 9 FrozenGroupNorm instances."""
        gns = refiner._all_groupnorms()
        assert len(gns) == 9
        for gn in gns:
            assert isinstance(gn, FrozenGroupNorm)

    def test_collect_freeze_unfreeze_cycle(self, refiner: CNNRefinerModule) -> None:
        """Full cycle: collect → freeze → unfreeze."""
        rgb = mx.random.normal((1, 32, 32, 3))
        coarse = mx.random.normal((1, 32, 32, 4))
        # mx.eval: MLX array materialization
        mx.eval(rgb, coarse)  # noqa: S307

        # Collect
        refiner.collect_groupnorm_stats(rgb, coarse)
        for gn in refiner._all_groupnorms():
            assert gn._collected_stats is not None
            assert not gn._collecting

        # Freeze
        refiner.freeze_groupnorm_stats()
        for gn in refiner._all_groupnorms():
            assert gn._frozen_stats is not None

        # Forward with frozen stats should work
        out = refiner(rgb, coarse)
        # mx.eval: MLX array materialization
        mx.eval(out)  # noqa: S307
        assert out.shape == (1, 32, 32, 4)

        # Unfreeze
        refiner.unfreeze_groupnorm_stats()
        for gn in refiner._all_groupnorms():
            assert gn._frozen_stats is None
            assert gn._collected_stats is None

    def test_frozen_output_matches_normal(self, refiner: CNNRefinerModule) -> None:
        """Frozen forward on same input should closely match normal forward."""
        rgb = mx.random.normal((1, 32, 32, 3))
        coarse = mx.random.normal((1, 32, 32, 4))
        # mx.eval: MLX array materialization
        mx.eval(rgb, coarse)  # noqa: S307

        # Normal forward
        out_normal = refiner(rgb, coarse)
        # mx.eval: MLX array materialization
        mx.eval(out_normal)  # noqa: S307

        # Collect + freeze + forward
        refiner.collect_groupnorm_stats(rgb, coarse)
        refiner.freeze_groupnorm_stats()
        out_frozen = refiner(rgb, coarse)
        # mx.eval: MLX array materialization
        mx.eval(out_frozen)  # noqa: S307
        refiner.unfreeze_groupnorm_stats()

        # Collecting path uses manual mean/var vs fused layer_norm — small numerical diff
        diff = float(mx.max(mx.abs(out_frozen - out_normal)))
        assert diff < 1e-4, f"Frozen vs normal refiner diff={diff}"
