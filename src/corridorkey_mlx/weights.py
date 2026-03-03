"""Download and cache model weights from GitHub Releases.

Supports streaming download with progress, SHA256 verification,
and platform-appropriate cache directories.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any

import requests
from platformdirs import user_cache_dir
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

# ---------------------------------------------------------------------------
# Config — single source of truth
# ---------------------------------------------------------------------------

PACKAGE_NAME = "corridorkey-mlx"
PACKAGE_SLUG = "corridorkey_mlx"

DEFAULT_GITHUB_OWNER = "cristopheryates"
DEFAULT_GITHUB_REPO = "corridorkey-mlx"

DEFAULT_ASSET_NAME = "corridorkey_mlx.safetensors"
CHECKSUM_EXTENSIONS = (".sha256",)
CHECKSUM_FILENAME = "SHA256SUMS"

DOWNLOAD_CHUNK_SIZE = 1024 * 256  # 256 KiB

ENV_PREFIX = "CORRIDORKEY_MLX"


def _env(name: str, default: str | None = None) -> str | None:
    """Read an env var with the package prefix."""
    return os.environ.get(f"{ENV_PREFIX}_{name}", default)


def github_owner_repo() -> tuple[str, str]:
    repo_override = _env("WEIGHTS_REPO")
    if repo_override and "/" in repo_override:
        owner, repo = repo_override.split("/", 1)
        return owner, repo
    return DEFAULT_GITHUB_OWNER, DEFAULT_GITHUB_REPO


def default_asset_name() -> str:
    return _env("WEIGHTS_ASSET") or DEFAULT_ASSET_NAME


def default_tag() -> str:
    return _env("WEIGHTS_TAG") or "latest"


# ---------------------------------------------------------------------------
# Cache directory
# ---------------------------------------------------------------------------


def cache_dir(tag: str = "latest") -> Path:
    """Platform-appropriate cache directory for a given release tag."""
    base = Path(user_cache_dir(PACKAGE_SLUG, appauthor=False))
    return base / "weights" / tag


# ---------------------------------------------------------------------------
# GitHub Releases API
# ---------------------------------------------------------------------------

_console = Console(stderr=True)


def _session() -> requests.Session:
    s = requests.Session()
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        s.headers["Authorization"] = f"token {token}"
    s.headers["Accept"] = "application/vnd.github+json"
    s.headers["User-Agent"] = f"{PACKAGE_SLUG}-weights-downloader"
    return s


def _api_base() -> str:
    owner, repo = github_owner_repo()
    return f"https://api.github.com/repos/{owner}/{repo}"


def resolve_release(tag: str) -> dict[str, Any]:
    """Fetch release metadata. 'latest' resolves via the GitHub API."""
    sess = _session()
    if tag == "latest":
        url = f"{_api_base()}/releases/latest"
    else:
        url = f"{_api_base()}/releases/tags/{tag}"

    resp = sess.get(url, timeout=30)
    if resp.status_code == 404:
        raise SystemExit(f"Release not found: {tag}")
    resp.raise_for_status()
    return resp.json()


def list_assets(release: dict[str, Any]) -> list[dict[str, Any]]:
    return release.get("assets", [])


def find_asset(
    release: dict[str, Any], asset_name: str | None = None
) -> dict[str, Any]:
    """Find a specific asset in a release, or infer from platform."""
    assets = list_assets(release)
    target = asset_name or default_asset_name()

    for a in assets:
        if a["name"] == target:
            return a

    # list available assets for helpful error
    available = [a["name"] for a in assets]
    msg = f"Asset '{target}' not found in release '{release['tag_name']}'."
    if available:
        msg += f"\nAvailable assets: {', '.join(available)}"
    else:
        msg += "\nThis release has no assets."
    raise SystemExit(msg)


# ---------------------------------------------------------------------------
# Checksum helpers
# ---------------------------------------------------------------------------


def _fetch_checksum_for_asset(
    release: dict[str, Any], asset_name: str, sess: requests.Session
) -> str | None:
    """Try to find a SHA256 checksum for `asset_name` from release assets.

    Looks for:
      1. <asset_name>.sha256  (single-hash file)
      2. SHA256SUMS           (multi-line, grep for asset_name)
    """
    assets = {a["name"]: a for a in list_assets(release)}

    # strategy 1: dedicated .sha256 sidecar
    sidecar = f"{asset_name}.sha256"
    if sidecar in assets:
        url = assets[sidecar]["browser_download_url"]
        resp = sess.get(url, timeout=30)
        resp.raise_for_status()
        return _parse_hash_line(resp.text, asset_name)

    # strategy 2: SHA256SUMS manifest
    if CHECKSUM_FILENAME in assets:
        url = assets[CHECKSUM_FILENAME]["browser_download_url"]
        resp = sess.get(url, timeout=30)
        resp.raise_for_status()
        for line in resp.text.strip().splitlines():
            if asset_name in line:
                return _parse_hash_line(line, asset_name)

    return None


def _parse_hash_line(text: str, _asset_name: str) -> str | None:
    """Extract a hex SHA256 hash from a line like 'abc123  filename' or just 'abc123'."""
    text = text.strip()
    match = re.match(r"([0-9a-fA-F]{64})", text)
    return match.group(1).lower() if match else None


def verify_sha256(path: Path, expected: str) -> bool:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(DOWNLOAD_CHUNK_SIZE):
            h.update(chunk)
    return h.hexdigest() == expected.lower()


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def download_asset(
    release: dict[str, Any],
    asset: dict[str, Any],
    dest_dir: Path,
    *,
    force: bool = False,
    verify: bool = True,
) -> Path:
    """Stream-download a release asset to dest_dir with progress and verification."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    final_path = dest_dir / asset["name"]
    partial_path = dest_dir / f"{asset['name']}.partial"

    # skip if already cached
    if final_path.exists() and not force:
        if verify:
            _console.print(f"[dim]Cached: {final_path}[/dim]")
            sess = _session()
            expected = _fetch_checksum_for_asset(release, asset["name"], sess)
            if expected:
                if not verify_sha256(final_path, expected):
                    _console.print(
                        "[yellow]Checksum mismatch on cached file — re-downloading[/yellow]"
                    )
                else:
                    return final_path
            else:
                _console.print("[dim]No checksum available, using cached file[/dim]")
                return final_path
        else:
            return final_path

    # stream download
    sess = _session()
    # Use browser_download_url for direct download (no auth needed for public repos)
    download_url = asset["browser_download_url"]
    total_size = asset.get("size", 0)

    _console.print(f"Downloading [bold]{asset['name']}[/bold] ({_human_size(total_size)})")

    resp = sess.get(
        download_url,
        stream=True,
        timeout=30,
        headers={"Accept": "application/octet-stream"},
    )
    resp.raise_for_status()

    # use Content-Length if available (more accurate than API size)
    content_length = resp.headers.get("Content-Length")
    if content_length:
        total_size = int(content_length)

    with Progress(
        TextColumn("[bold blue]{task.fields[filename]}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=_console,
    ) as progress:
        task = progress.add_task("download", filename=asset["name"], total=total_size)

        with open(partial_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                f.write(chunk)
                progress.advance(task, len(chunk))

    # verify checksum before finalizing
    if verify:
        expected = _fetch_checksum_for_asset(release, asset["name"], sess)
        if expected:
            _console.print("Verifying SHA256…")
            if not verify_sha256(partial_path, expected):
                actual = _compute_sha256(partial_path)
                partial_path.unlink(missing_ok=True)
                raise SystemExit(
                    f"SHA256 mismatch! Expected {expected}, "
                    f"got {actual}. Download may be corrupted."
                )
            _console.print("[green]Checksum OK[/green]")
        else:
            _console.print(
                "[yellow]No checksum file found — skipping verification[/yellow]"
            )

    # atomic rename
    partial_path.rename(final_path)
    return final_path


def _compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(DOWNLOAD_CHUNK_SIZE):
            h.update(chunk)
    return h.hexdigest()


def _human_size(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024  # type: ignore[assignment]
    return f"{nbytes:.1f} TB"


# ---------------------------------------------------------------------------
# High-level entry point (used by CLI and Python API)
# ---------------------------------------------------------------------------


def download_weights(
    *,
    tag: str | None = None,
    asset_name: str | None = None,
    out: Path | None = None,
    force: bool = False,
    verify: bool = True,
) -> Path:
    """Download model weights, returning the local path.

    Resolves tag, finds asset, downloads with progress, verifies checksum.
    """
    resolved_tag = tag or default_tag()
    release = resolve_release(resolved_tag)
    actual_tag = release["tag_name"]

    asset = find_asset(release, asset_name)

    dest = out or cache_dir(actual_tag)
    return download_asset(release, asset, dest, force=force, verify=verify)
