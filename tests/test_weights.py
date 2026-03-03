"""Tests for weights download module: config defaults, checksum, hash parsing."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from corridorkey_mlx.weights import (
    DEFAULT_ASSET_NAME,
    DEFAULT_GITHUB_OWNER,
    DEFAULT_GITHUB_REPO,
    _parse_hash_line,
    cache_dir,
    default_asset_name,
    default_tag,
    github_owner_repo,
    verify_sha256,
)

if TYPE_CHECKING:
    from pathlib import Path


# -- Config defaults ---------------------------------------------------------


class TestConfig:
    def test_default_owner_repo(self) -> None:
        owner, repo = github_owner_repo()
        assert owner == DEFAULT_GITHUB_OWNER
        assert repo == DEFAULT_GITHUB_REPO

    def test_default_asset_name(self) -> None:
        assert default_asset_name() == DEFAULT_ASSET_NAME

    def test_default_tag(self) -> None:
        assert default_tag() == "latest"


# -- Cache directory ---------------------------------------------------------


class TestCacheDir:
    def test_contains_tag(self) -> None:
        p = cache_dir("v1.0.0")
        assert p.parts[-1] == "v1.0.0"
        assert p.parts[-2] == "weights"

    def test_latest_tag(self) -> None:
        p = cache_dir("latest")
        assert p.parts[-1] == "latest"


# -- Checksum ----------------------------------------------------------------


class TestChecksum:
    def test_parse_hash_only(self) -> None:
        h = "a" * 64
        assert _parse_hash_line(h, "file.bin") == h

    def test_parse_hash_with_filename(self) -> None:
        h = "b" * 64
        assert _parse_hash_line(f"{h}  file.bin", "file.bin") == h

    def test_parse_hash_uppercase(self) -> None:
        h = "C" * 64
        assert _parse_hash_line(h, "file.bin") == h.lower()

    def test_parse_hash_garbage(self) -> None:
        assert _parse_hash_line("not a hash", "file.bin") is None

    def test_verify_sha256_valid(self, tmp_path: Path) -> None:
        content = b"hello world"
        expected = hashlib.sha256(content).hexdigest()
        p = tmp_path / "test.bin"
        p.write_bytes(content)
        assert verify_sha256(p, expected) is True

    def test_verify_sha256_invalid(self, tmp_path: Path) -> None:
        p = tmp_path / "test.bin"
        p.write_bytes(b"hello world")
        assert verify_sha256(p, "0" * 64) is False
