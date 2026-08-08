"""Offline regression tests for the flaky-data-host handling in core.py.

The ISAAC bulk host (a Globus HTTPS collection) is load-balanced, and a backend
sometimes fails a request with a GridFTP INTERNAL_ERROR that the collection
renders as HTTP 404. fsspec turns that failed size probe into an unknown-length,
non-seekable stream, so pyarrow used to die with "Cannot seek streaming HTTP
file" — nondeterministically, since the failure is picked at connect time and is
sticky for that connection.

These tests fake the host, so they neither need nor touch the network.
"""
import io
import os
import tempfile

import pytest

os.environ.setdefault("ISAAC_DATA_CONFIG", tempfile.mkdtemp(prefix="isaac-cfg-http-"))
os.environ["ISAAC_ACCEPT_AGREEMENT"] = "1"

import isaac_data.core as core  # noqa: E402
from isaac_data import DataHostUnavailable  # noqa: E402

URL = "https://g-05a4b6.2d513.8443.data.globus.org/ability/RC_2007-03.parquet"

_INTERNAL_ERROR = (
    "GlobusError: v=1 c=INTERNAL_ERROR\nGridFTP-Errno: 108\n"
    "GridFTP-Reason: System error in stat\n"
    "GridFTP-Error-String: Cannot send after transport endpoint shutdown\n"
)
_PATH_NOT_FOUND = (
    "GlobusError: v=1 c=PATH_NOT_FOUND\nGridFTP-Errno: 2\n"
    "GridFTP-Reason: System error in stat\n"
    "GridFTP-Error-String: No such file or directory\n"
)


@pytest.fixture
def parquet_file(tmp_path):
    """A small local parquet file standing in for a hosted one."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({"text": [f"row {i}" for i in range(100)], "score": list(range(100))})
    path = tmp_path / "RC_2007-03.parquet"
    pq.write_table(table, path, row_group_size=10)
    return path


class _Handle(io.BufferedReader):
    """A file handle carrying fsspec's ``.size`` (None when the probe failed)."""

    size = None


class _FakeFS:
    """Stands in for fsspec's HTTPFileSystem.

    ``opens`` counts every open across instances so the tests can assert that a
    retry actually happened; each instance is one "connection".
    """

    opens = 0
    bad_opens = 0  # how many of the first opens hand back an unsized stream
    raise_404 = False

    def __init__(self, path):
        self._path = path

    def open(self, url, mode="rb"):
        type(self).opens += 1
        if type(self).raise_404:
            raise FileNotFoundError(url)
        fh = _Handle(io.FileIO(self._path, "rb"))
        # An unsized handle is exactly what breaks pyarrow: it cannot seek to
        # the parquet footer without knowing the length.
        fh.size = None if type(self).opens <= type(self).bad_opens else os.path.getsize(self._path)
        return fh


@pytest.fixture
def fake_fs(monkeypatch, parquet_file):
    import fsspec

    _FakeFS.opens = 0
    _FakeFS.bad_opens = 0
    _FakeFS.raise_404 = False
    monkeypatch.setattr(fsspec, "filesystem", lambda proto, **kw: _FakeFS(parquet_file))
    monkeypatch.setattr(core, "_HTTP_BACKOFF", 0)  # keep the suite fast
    return _FakeFS


def test_unsized_stream_is_retried_on_a_fresh_connection(fake_fs):
    """A non-seekable handle must be retried, not handed to pyarrow."""
    fake_fs.bad_opens = 2
    fh = core._open_remote_parquet(URL)
    assert fh.size is not None       # seekable: pyarrow can find the footer
    assert fake_fs.opens == 3        # two bad connections, then a good one
    fh.close()


def test_read_parquet_survives_a_flaky_host(fake_fs):
    """The end-to-end read used to fail ~half the time on this exact pattern."""
    fake_fs.bad_opens = 2
    df = core.read_parquet(URL, columns=["text"])
    assert len(df) == 100 and list(df.columns) == ["text"]


def test_sampling_survives_a_flaky_host(fake_fs):
    import numpy as np

    fake_fs.bad_opens = 3
    row = type("Row", (), {"format": "parquet", "url": URL, "num_rows": 100,
                           "rel_path": "ability/RC_2007-03.parquet"})()
    df = core._sample_one(row, k=5, columns=["text"], seed_seq=np.random.SeedSequence(1), cache=False)
    assert len(df) == 5


def test_exhausted_retries_fall_back_to_download(fake_fs, monkeypatch, parquet_file, tmp_path):
    """When every connection is bad, fall back to a whole-file download."""
    fake_fs.bad_opens = 999
    monkeypatch.setattr(core, "_HTTP_ATTEMPTS", 3)
    called = {}

    def fake_download(url, dest, chunk=1 << 20):
        called["url"] = url
        return parquet_file

    monkeypatch.setattr(core, "_download_one", fake_download)
    monkeypatch.setattr(core, "cache_dir", lambda: tmp_path)
    with pytest.warns(RuntimeWarning, match="falling back to a whole-file download"):
        df = core.read_parquet(URL, columns=["text"])
    assert len(df) == 100 and called["url"] == URL
    assert fake_fs.opens == 3  # gave the host every attempt before downloading


def test_missing_file_is_not_retried(fake_fs, monkeypatch):
    """A genuine PATH_NOT_FOUND must fail fast with an actionable message."""
    fake_fs.raise_404 = True
    monkeypatch.setattr(core, "_probe_url", lambda url: (False, True, _PATH_NOT_FOUND))
    with pytest.raises(FileNotFoundError, match="does not have it"):
        core._open_remote_parquet(URL)
    assert fake_fs.opens == 1  # no pointless retries against a missing file


def test_backend_404_is_retried_not_reported_as_missing(fake_fs, monkeypatch):
    """A 404 carrying the backend's INTERNAL_ERROR is transient, not a miss."""
    fake_fs.raise_404 = True
    monkeypatch.setattr(core, "_HTTP_ATTEMPTS", 3)
    monkeypatch.setattr(core, "_probe_url", lambda url: (False, False, _INTERNAL_ERROR))
    with pytest.raises(DataHostUnavailable, match="attempts on new connections"):
        core._open_remote_parquet(URL)
    assert fake_fs.opens == 3


CSV_URL = "https://g-05a4b6.2d513.8443.data.globus.org/ability/RC_2007-01.csv"


@pytest.mark.parametrize(
    "error",
    ["empty", "http"],
    ids=["empty-body", "http-error"],
)
def test_csv_read_falls_back_when_the_host_hiccups(monkeypatch, tmp_path, error):
    """The CSV path hits the same flaky backend as parquet.

    An empty body surfaces as pandas' EmptyDataError — a ValueError, not an
    OSError — so catching only network errors let it through untouched.
    """
    import pandas as pd

    local = tmp_path / "RC_2007-01.csv"
    local.write_text("text,score\nhello,3\nworld,4\n")
    calls = {"n": 0}
    real_read_csv = pd.read_csv

    def flaky_read_csv(src, **kw):
        if str(src).startswith("http"):
            calls["n"] += 1
            if error == "empty":
                raise pd.errors.EmptyDataError("No columns to parse from file")
            raise OSError("HTTP Error 404: Not Found")
        return real_read_csv(src, **kw)

    monkeypatch.setattr(pd, "read_csv", flaky_read_csv)
    monkeypatch.setattr(core, "_probe_url", lambda url: (False, False, _INTERNAL_ERROR))
    monkeypatch.setattr(core, "_download_one", lambda url, dest, chunk=1 << 20: local)
    monkeypatch.setattr(core, "cache_dir", lambda: tmp_path)

    with pytest.warns(RuntimeWarning, match="falling back"):
        df = core._read_csv_source(CSV_URL, "ability/RC_2007-01.csv", ["score"], cache=False)
    assert calls["n"] == 1 and list(df["score"]) == [3, 4]


def test_csv_read_reports_a_missing_file_clearly(monkeypatch, tmp_path):
    import pandas as pd

    def flaky_read_csv(src, **kw):
        raise OSError("HTTP Error 404: Not Found")

    monkeypatch.setattr(pd, "read_csv", flaky_read_csv)
    monkeypatch.setattr(core, "_probe_url", lambda url: (False, True, _PATH_NOT_FOUND))
    with pytest.raises(FileNotFoundError, match="does not have it"):
        core._read_csv_source(CSV_URL, "ability/RC_2007-01.csv", None, cache=False)


class _Resp:
    def __init__(self, status, text="", headers=None):
        self.status_code = status
        self.text = text
        self.headers = headers or {}

    @property
    def ok(self):
        return self.status_code < 400


def test_check_response_separates_transient_from_permanent():
    core._check_response(_Resp(206), URL)  # success: no raise
    with pytest.raises(FileNotFoundError):
        core._check_response(_Resp(404, _PATH_NOT_FOUND), URL)
    with pytest.raises(DataHostUnavailable):
        core._check_response(_Resp(404, _INTERNAL_ERROR), URL)
    with pytest.raises(DataHostUnavailable):
        core._check_response(_Resp(503, "Service Unavailable"), URL)


def test_download_retries_transient_404_then_succeeds(monkeypatch, tmp_path, parquet_file):
    """_download_one must not give up on the backend's 404-flavoured error."""
    monkeypatch.setattr(core, "_HTTP_BACKOFF", 0)
    heads = {"n": 0}
    gets = {"n": 0}
    payload = parquet_file.read_bytes()

    class FakeGet:
        def __init__(self, status):
            self.status_code = status
            self.text = _INTERNAL_ERROR if status == 404 else ""
            self.headers = {"Content-Length": str(len(payload))}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def iter_content(self, chunk):
            yield payload

    def fake_head(url, **kw):
        heads["n"] += 1
        # A real HEAD 404 carries no body — the reason is only in the GET.
        return _Resp(404) if heads["n"] == 1 else _Resp(200)

    def fake_get(url, **kw):
        gets["n"] += 1
        return FakeGet(404 if gets["n"] == 1 else 200)

    monkeypatch.setattr(core.requests, "head", fake_head)
    monkeypatch.setattr(core.requests, "get", fake_get)
    dest = tmp_path / "out.parquet"
    assert core._download_one(URL, dest) == dest
    assert (heads["n"], gets["n"]) == (2, 2)  # one failed attempt, then success
    assert dest.read_bytes() == payload


def test_bodyless_head_404_is_not_mistaken_for_a_missing_file(monkeypatch, tmp_path):
    """A HEAD has no body, so it can never prove a file is missing.

    Classifying on the empty HEAD body made a transient backend 404 look
    permanent, which aborted the download fallback with a bogus "file is
    missing" error on a file that was there all along.
    """
    monkeypatch.setattr(core, "_HTTP_BACKOFF", 0)
    monkeypatch.setattr(core, "_HTTP_ATTEMPTS", 3)
    gets = {"n": 0}

    class FakeGet:
        status_code = 404
        text = _INTERNAL_ERROR

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_get(url, **kw):
        gets["n"] += 1
        return FakeGet()

    monkeypatch.setattr(core.requests, "head", lambda url, **kw: _Resp(404))
    monkeypatch.setattr(core.requests, "get", fake_get)
    with pytest.raises(DataHostUnavailable):  # not FileNotFoundError
        core._download_one(URL, tmp_path / "out.parquet")
    assert gets["n"] == 3  # the GET body decided, and it said "retry me"


@pytest.mark.parametrize(
    "body, label",
    [(b"", "empty"), (b"PAR1-truncated", "short")],
)
def test_download_rejects_a_bad_payload_instead_of_caching_it(monkeypatch, tmp_path, body, label):
    """A 200 with an empty or truncated body must retry, not land in the cache.

    The backend answers 200-with-no-bytes often enough that promoting it to the
    cache surfaced as pyarrow's "Parquet file size is 0 bytes" — an error that
    tells the user nothing about the real (transient) cause.
    """
    monkeypatch.setattr(core, "_HTTP_BACKOFF", 0)
    monkeypatch.setattr(core, "_HTTP_ATTEMPTS", 2)

    class FakeGet:
        status_code = 200
        text = ""
        headers = {"Content-Length": "194080"}  # what the host promised

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def iter_content(self, chunk):
            if body:
                yield body

    monkeypatch.setattr(core.requests, "head", lambda url, **kw: _Resp(404))
    monkeypatch.setattr(core.requests, "get", lambda url, **kw: FakeGet())
    dest = tmp_path / "out.parquet"
    with pytest.raises(DataHostUnavailable, match="bytes"):
        core._download_one(URL, dest)
    assert not dest.exists(), f"a {label} download must never be promoted to dest"


def test_download_fails_fast_on_a_missing_file(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "_HTTP_BACKOFF", 0)
    gets = {"n": 0}

    class FakeGet:
        status_code = 404
        text = _PATH_NOT_FOUND

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_get(url, **kw):
        gets["n"] += 1
        return FakeGet()

    monkeypatch.setattr(core.requests, "head", lambda url, **kw: _Resp(404))
    monkeypatch.setattr(core.requests, "get", fake_get)
    with pytest.raises(FileNotFoundError, match="does not have it"):
        core._download_one(URL, tmp_path / "out.parquet")
    assert gets["n"] == 1  # a real miss is not retried
