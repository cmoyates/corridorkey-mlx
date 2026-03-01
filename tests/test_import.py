"""Smoke test: verify package imports succeed."""


def test_package_imports() -> None:
    import corridorkey_mlx
    import corridorkey_mlx.convert
    import corridorkey_mlx.inference
    import corridorkey_mlx.io
    import corridorkey_mlx.model
    import corridorkey_mlx.utils

    assert corridorkey_mlx.__version__ == "0.1.0"
