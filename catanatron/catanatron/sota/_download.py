"""Fetches pinned third-party files into a local cache, verified by SHA-256."""

import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path


def cache_dir() -> Path:
    """Root of the download cache. Override with CATANATRON_CACHE_DIR."""
    root = os.environ.get("CATANATRON_CACHE_DIR")
    if root:
        return Path(root)
    return Path.home() / ".cache" / "catanatron"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, dest: Path, sha256: str) -> Path:
    """Downloads url to dest unless a file with the right hash is already there.

    Safe to call from several processes at once: each downloads to its own
    temp file and atomically renames it into place. On Windows the rename fails
    while another process has dest open; by then dest is already complete.
    """
    if dest.exists() and _sha256(dest) == sha256:
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}\n  -> {dest}")
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, suffix=".part")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url) as response:
            for chunk in iter(lambda: response.read(1 << 20), b""):
                out.write(chunk)
        actual = _sha256(tmp)
        if actual != sha256:
            raise RuntimeError(
                f"SHA-256 mismatch for {url}: expected {sha256}, got {actual}"
            )
        try:
            os.replace(tmp, dest)
        except PermissionError:
            if not (dest.exists() and _sha256(dest) == sha256):
                raise
    finally:
        tmp.unlink(missing_ok=True)
    return dest
