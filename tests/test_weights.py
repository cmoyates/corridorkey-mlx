"""Unit tests for weights download module.

Tests config, caching, checksum verification, and CLI arg parsing
without requiring real network access.
"""

from __future__ import annotations

import hashlib
import os
from typing import TYPE_CHECKING
from unittest.mock import patch

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
from corridorkey_mlx.weights_cli import build_parser

if TYPE_CHECKING:
    from pathlib import Path


# -- Config defaults ---------------------------------------------------------


class TestConfig:
    def test_default_owner_repo(self) -> None:
        owner, repo = github_owner_repo()
        assert owner == DEFAULT_GITHUB_OWNER
        assert repo == DEFAULT_GITHUB_REPO

    def test_owner_repo_env_override(self) -> None:
        with patch.dict(os.environ, {"CORRIDORKEY_MLX_WEIGHTS_REPO": "other/repo"}):
            owner, repo = github_owner_repo()
            assert owner == "other"
            assert repo == "repo"

    def test_default_asset_name(self) -> None:
        assert default_asset_name() == DEFAULT_ASSET_NAME

    def test_asset_name_env_override(self) -> None:
        with patch.dict(os.environ, {"CORRIDORKEY_MLX_WEIGHTS_ASSET": "custom.st"}):
            assert default_asset_name() == "custom.st"

    def test_default_tag(self) -> None:
        assert default_tag() == "latest"

    def test_tag_env_override(self) -> None:
        with patch.dict(os.environ, {"CORRIDORKEY_MLX_WEIGHTS_TAG": "v2.0"}):
            assert default_tag() == "v2.0"


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

    def test_verify_sha256(self, tmp_path: Path) -> None:
        content = b"hello world"
        expected = hashlib.sha256(content).hexdigest()
        p = tmp_path / "test.bin"
        p.write_bytes(content)

        assert verify_sha256(p, expected) is True
        assert verify_sha256(p, "0" * 64) is False


# -- CLI arg parsing ---------------------------------------------------------


class TestCLI:
    def test_download_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["download"])
        assert args.command == "download"
        assert args.tag is None
        assert args.asset_name is None
        assert args.out is None
        assert args.force is False
        assert args.no_verify is False
        assert args.print_path is False

    def test_download_all_flags(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "download",
            "--tag", "v1.0",
            "--asset", "custom.st",
            "--out", "/tmp/w",
            "--force",
            "--no-verify",
            "--print-path",
        ])
        assert args.tag == "v1.0"
        assert args.asset_name == "custom.st"
        assert args.out == "/tmp/w"
        assert args.force is True
        assert args.no_verify is True
        assert args.print_path is True
