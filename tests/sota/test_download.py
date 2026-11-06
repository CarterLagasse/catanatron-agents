import hashlib
import io

import pytest

from catanatron.sota import _download

DATA = b"print('hello')\n"
SHA = hashlib.sha256(DATA).hexdigest()


@pytest.fixture
def fake_url(monkeypatch):
    monkeypatch.setattr(
        _download.urllib.request, "urlopen", lambda url: io.BytesIO(DATA)
    )


def test_fetch_verifies_and_caches(tmp_path, fake_url):
    dest = tmp_path / "f.py"
    assert _download.fetch("http://x", dest, SHA).read_bytes() == DATA
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        _download.fetch("http://x", tmp_path / "g.py", "0" * 64)
    assert not (tmp_path / "g.py").exists()
    assert [p.name for p in tmp_path.iterdir()] == ["f.py"]  # no temp files left


def test_fetch_tolerates_a_concurrent_download(tmp_path, fake_url, monkeypatch):
    # On Windows, replacing a file another process has open fails; that process
    # already put the complete file in place.
    dest = tmp_path / "f.py"

    def replace_while_open(src, dst):
        dest.write_bytes(DATA)
        raise PermissionError("in use")

    monkeypatch.setattr(_download.os, "replace", replace_while_open)
    assert _download.fetch("http://x", dest, SHA) == dest
    assert [p.name for p in tmp_path.iterdir()] == ["f.py"]
