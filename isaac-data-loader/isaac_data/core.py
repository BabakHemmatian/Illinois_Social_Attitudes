"""Core API for the ISAAC data loader.

Thin client over the ISAAC Direct Download catalog
(https://isaac.psychology.illinois.edu/direct-download/). It uses the published
``manifest.json`` as the catalog; the per-file ``url`` in the manifest points at
the public/anonymous NCSA Taiga Globus collection, so bulk transfers come
straight off the storage DTNs rather than through the web VM. Parquet
reads support column projection over HTTP range requests (via pyarrow + fsspec),
so you transfer only the columns you ask for; full files are cached locally and
downloads are resumable.

Heavy imports (pandas / pyarrow / fsspec) are deferred to call time so that
``import isaac_data`` stays cheap and dependency errors are actionable.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import requests

from .agreement import require_acceptance

__all__ = [
    "BASE_URL", "DATA_BASE", "MANIFEST_URL", "CATEGORIES",
    "cache_dir", "set_cache_dir", "catalog", "files",
    "download", "read_parquet", "load", "DataHostUnavailable",
]

# BASE_URL is the website that serves the small JSON catalog (manifest.json).
# DATA_BASE is the bulk-file host: the public Taiga Globus collection. Actual
# download URLs are taken from each manifest record's ``url`` field, so DATA_BASE
# is only a reference/fallback base for constructing URLs by hand.
BASE_URL = os.environ.get("ISAAC_BASE_URL", "https://isaac.psychology.illinois.edu").rstrip("/")
MANIFEST_URL = f"{BASE_URL}/direct-download/manifest.json"
DATA_BASE = os.environ.get("ISAAC_DATA_BASE", "https://g-05a4b6.2d513.8443.data.globus.org/")

CATEGORIES: Tuple[str, ...] = (
    "ability", "age", "race", "sexuality", "skin_tone", "weight",
)

_MANIFEST_TTL_SECONDS = 24 * 3600
_DEFAULT_MAX_BYTES = 5_000_000_000  # 5 GB guardrail for load() without sampling/columns


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
_cache_override: Optional[Path] = None


def set_cache_dir(path: Union[str, Path]) -> Path:
    """Override the local cache directory used for the manifest and downloads.

    Args:
        path: Directory to use as the cache root (created if it does not exist).

    Returns:
        The resolved cache directory path.
    """
    global _cache_override
    _cache_override = Path(path).expanduser()
    _cache_override.mkdir(parents=True, exist_ok=True)
    return _cache_override


def cache_dir() -> Path:
    """Local cache for the manifest and downloaded files.

    Precedence: set_cache_dir() > $ISAAC_DATA_CACHE > OS-native cache dir
    (e.g. ~/Library/Caches/isaac-data on macOS,
    %LOCALAPPDATA%\\isaac-data\\Cache on Windows, ~/.cache/isaac-data on Linux).
    """
    if _cache_override is not None:
        d = _cache_override
    elif os.environ.get("ISAAC_DATA_CACHE"):
        d = Path(os.environ["ISAAC_DATA_CACHE"]).expanduser()
    else:
        import platformdirs
        d = Path(platformdirs.user_cache_dir("isaac-data"))
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- #
# Catalog
# --------------------------------------------------------------------------- #
def _manifest_file() -> Path:
    return cache_dir() / "manifest.json"


def catalog(refresh: bool = False):
    """Fetch the file catalog (the published ``manifest.json``) as a DataFrame.

    The manifest is the authoritative list of every downloadable file and is the
    basis for `files`, `load`, and `download`. It is cached locally and only
    re-fetched when older than 24h (or when ``refresh=True``). Browsing the
    catalog does not require Data-Use-Agreement acceptance.

    Args:
        refresh: Force a re-download of the manifest instead of using the local
            cache. Defaults to False.

    Returns:
        A pandas DataFrame with one row per file and columns: ``category``,
        ``year``, ``month``, ``format`` ('parquet' or 'csv'), ``filename``,
        ``rel_path``, ``size_bytes``, ``num_rows``, ``url``, ``date``.
    """
    import pandas as pd

    mf = _manifest_file()
    stale = (not mf.exists()) or (time.time() - mf.stat().st_mtime > _MANIFEST_TTL_SECONDS)
    if refresh or stale:
        resp = requests.get(MANIFEST_URL, timeout=60)
        resp.raise_for_status()
        mf.write_text(resp.text)
    records = json.loads(mf.read_text())
    df = pd.DataFrame.from_records(records)
    df["date"] = pd.to_datetime(
        dict(year=df["year"], month=df["month"], day=1)
    )
    return df


def _ym(value: str) -> int:
    """'YYYY-MM' -> integer yyyymm for range comparisons."""
    year, month = str(value).split("-")[:2]
    return int(year) * 100 + int(month)


def files(
    category: Optional[Union[str, Sequence[str]]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    fmt: Optional[str] = "parquet",
    refresh: bool = False,
):
    """Filter the catalog to the files you want (transfers no data).

    Args:
        category: One category or a list of categories; None selects all.
            Valid: ability, age, race, sexuality, skin_tone, weight.
        start: Inclusive lower-bound month as ``'YYYY-MM'`` (e.g. ``'2018-01'``);
            None for no lower bound.
        end: Inclusive upper-bound month as ``'YYYY-MM'``; None for no upper bound.
        fmt: ``'parquet'`` (default), ``'csv'``, or None for both formats.
        refresh: Force a manifest refresh before filtering. Defaults to False.

    Returns:
        A pandas DataFrame (same columns as `catalog`), sorted by category,
        year, month, format.

    Raises:
        ValueError: if an unknown category name is given.
    """
    df = catalog(refresh=refresh)
    if category is not None:
        cats = [category] if isinstance(category, str) else list(category)
        unknown = set(cats) - set(CATEGORIES)
        if unknown:
            raise ValueError(f"Unknown categor(y/ies): {sorted(unknown)}. Valid: {CATEGORIES}")
        df = df[df["category"].isin(cats)]
    if fmt is not None:
        df = df[df["format"] == fmt]
    key = df["year"] * 100 + df["month"]
    mask = key.notna()
    if start is not None:
        mask &= key >= _ym(start)
    if end is not None:
        mask &= key <= _ym(end)
    df = df[mask]
    return df.sort_values(["category", "year", "month", "format"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# HTTP robustness
# --------------------------------------------------------------------------- #
# The bulk files live on a Globus HTTPS collection that is load-balanced across
# several GridFTP backends. A backend occasionally fails a request with
#     GlobusError: v=1 c=INTERNAL_ERROR / GridFTP-Errno: 108
#     "Cannot send after transport endpoint shutdown"
# which the collection renders as HTTP 404 — indistinguishable from a missing
# file by status code alone. Two properties drive the handling below:
#   * The failure is decided at connect time and is sticky for the life of that
#     connection, so retrying on the same pooled connection always re-fails.
#     Retries must use a *fresh* connection (new fsspec filesystem instance).
#   * fsspec turns the failed size probe into an unknown-length, non-seekable
#     stream instead of raising; pyarrow then dies with the opaque
#     "Cannot seek streaming HTTP file" when it reaches for the parquet footer.
# A real 404 (GridFTP-Errno 2, PATH_NOT_FOUND) is permanent and must not be
# retried, so the two are told apart by the error body, not the status code.
_HTTP_ATTEMPTS = int(os.environ.get("ISAAC_HTTP_ATTEMPTS", "5"))
_HTTP_BACKOFF = 0.3  # seconds, multiplied by the attempt number


class DataHostUnavailable(RuntimeError):
    """The ISAAC data host kept failing a request that should have succeeded.

    Raised only after `ISAAC_HTTP_ATTEMPTS` fresh connections have all failed,
    which points at the data host rather than at your query. Retrying later
    normally succeeds.
    """


def _missing_message(url: str, detail: str = "") -> str:
    return (
        f"{url} is listed in the ISAAC manifest but the data host does not have it. "
        "This is a server-side gap rather than a problem with your query — please "
        "report it at https://isaac.psychology.illinois.edu/direct-download/."
        + (f" Host said: {detail}" if detail else "")
    )


def _is_transient_body(body: str) -> bool:
    """True if a 404 body is the Globus backend error rather than a real miss."""
    return "INTERNAL_ERROR" in body


def _check_response(r, url: str) -> None:
    """Raise FileNotFoundError (permanent) or DataHostUnavailable (retryable)."""
    if r.status_code < 400:
        return
    body = " ".join((r.text or "")[:300].split())
    if r.status_code == 404 and not _is_transient_body(body):
        raise FileNotFoundError(_missing_message(url, body))
    raise DataHostUnavailable(f"{url}: HTTP {r.status_code}: {body}")


def _probe_url(url: str) -> Tuple[bool, bool, str]:
    """Ask the host about ``url`` on a fresh connection.

    Returns ``(ok, permanently_missing, detail)``. Used to classify failures that
    reach us without a usable status code (e.g. fsspec's silent size probe).
    """
    try:
        r = requests.get(url, headers={"Range": "bytes=0-0"}, timeout=30, allow_redirects=True)
    except requests.RequestException as e:  # connection reset, timeout, ...
        return False, False, f"{type(e).__name__}: {e}"
    if r.status_code < 400:
        return True, False, ""
    body = " ".join((r.text or "")[:300].split())
    return False, r.status_code == 404 and not _is_transient_body(body), f"HTTP {r.status_code}: {body}"


def _open_remote_parquet(url: str):
    """Open an http(s) parquet URL as a **seekable** file object for pyarrow.

    Retries on a new connection each time, because a flaky backend is pinned to
    the connection it was chosen for. A handle whose size is unknown is treated
    as a failed attempt: pyarrow cannot seek to the footer without it.

    Raises:
        FileNotFoundError: the file is genuinely absent on the host.
        DataHostUnavailable: every attempt failed; the caller may fall back to a
            whole-file download.
    """
    import fsspec

    detail = "the host returned no content length"
    for attempt in range(1, _HTTP_ATTEMPTS + 1):
        # skip_instance_cache: fsspec caches filesystems (and their connection
        # pools) globally, so without this every retry reuses the bad backend.
        fs = fsspec.filesystem("http", skip_instance_cache=True)
        try:
            fh = fs.open(url, "rb")
        except FileNotFoundError:
            ok, missing, why = _probe_url(url)
            if missing:
                raise FileNotFoundError(_missing_message(url, why)) from None
            detail = why or "the host reported the file as missing"
        else:
            if fh.size is not None:
                return fh
            fh.close()
        if attempt < _HTTP_ATTEMPTS:
            time.sleep(_HTTP_BACKOFF * attempt)
    raise DataHostUnavailable(
        f"{url}: {_HTTP_ATTEMPTS} attempts on new connections all failed ({detail}). "
        "This is a transient failure of the ISAAC data host, not of your query; "
        "retry shortly, or pass cache=True to download the file instead."
    )


def _cache_path_for_url(url: str) -> Path:
    """Where a URL's whole-file download lands in the local cache."""
    from urllib.parse import urlparse

    rel = urlparse(url).path.lstrip("/") or "download"
    return cache_dir() / "files" / rel


def _fallback_to_download(url: str, rel_path: Optional[str] = None) -> Path:
    """Last resort when streaming keeps failing: fetch the whole file, cached."""
    import warnings

    warnings.warn(
        f"Streaming {url} failed repeatedly; falling back to a whole-file download "
        "into the local cache. Pass cache=True to do this up front.",
        RuntimeWarning,
        stacklevel=3,
    )
    dest = (cache_dir() / "files" / rel_path) if rel_path else _cache_path_for_url(url)
    return _download_one(url, dest)


# --------------------------------------------------------------------------- #
# Download (resumable, cached)
# --------------------------------------------------------------------------- #
def _download_one(url: str, dest: Path, chunk: int = 1 << 20) -> Path:
    """Download ``url`` to ``dest``, resuming partial files and retrying the host.

    Raises:
        FileNotFoundError: the file is genuinely absent on the host.
        DataHostUnavailable: the host failed every attempt.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    detail = ""
    for attempt in range(1, _HTTP_ATTEMPTS + 1):
        try:
            return _download_attempt(url, dest, chunk)
        except DataHostUnavailable as e:
            detail = str(e)
        except requests.RequestException as e:  # dropped connection mid-transfer
            detail = f"{type(e).__name__}: {e}"
        if attempt < _HTTP_ATTEMPTS:
            time.sleep(_HTTP_BACKOFF * attempt)
    raise DataHostUnavailable(
        f"{url}: download failed on {_HTTP_ATTEMPTS} attempts ({detail}). "
        "This is a transient failure of the ISAAC data host; retry shortly."
    )


def _download_attempt(url: str, dest: Path, chunk: int = 1 << 20) -> Path:
    # The HEAD is only for sizing/resume, and is never used to classify a
    # failure: a HEAD response carries no body, so the Globus error text that
    # separates "missing" from "backend hiccup" is not there to read. A failed
    # HEAD just means "size unknown"; the GET below is the authority.
    head = requests.head(url, allow_redirects=True, timeout=30)
    total = int(head.headers.get("Content-Length", 0)) if head.ok else 0
    if dest.exists() and total and dest.stat().st_size == total:
        return dest  # already fully cached

    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have and total else {}

    with requests.get(url, headers=headers, stream=True, timeout=120) as r:
        _check_response(r, url)
        expected = total or _expected_total(r, have)
        # If we asked to resume but the server ignored Range (200, not 206),
        # restart from the beginning to avoid corrupting the file.
        mode = "ab" if (have and r.status_code == 206) else "wb"
        with open(part, mode) as fh:
            for block in r.iter_content(chunk):
                if block:
                    fh.write(block)

    # A failing backend can answer 200 with an empty or short body. Never
    # promote such a file to ``dest``: pyarrow would report the useless
    # "Parquet file size is 0 bytes" instead of the retryable host problem.
    written = part.stat().st_size if part.exists() else 0
    if written == 0 or (expected and written != expected):
        if written == 0:
            part.unlink(missing_ok=True)  # nothing to resume from
        raise DataHostUnavailable(
            f"{url}: the host sent {written} bytes of "
            f"{expected if expected else 'an unstated number of'} bytes"
        )
    part.replace(dest)
    return dest


def _expected_total(r, have: int) -> int:
    """Full file size implied by a response, or 0 if the host didn't say."""
    content_range = r.headers.get("Content-Range", "")
    if "/" in content_range:
        tail = content_range.rsplit("/", 1)[-1].strip()
        if tail.isdigit():
            return int(tail)
    length = r.headers.get("Content-Length")
    if length and str(length).isdigit():
        return int(length) + (have if r.status_code == 206 else 0)
    return 0


def download(
    category: Optional[Union[str, Sequence[str]]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    fmt: Optional[str] = "parquet",
    dest: Optional[Union[str, Path]] = None,
    refresh: bool = False,
) -> List[Path]:
    """Download selected files to disk (resumable; loads nothing into memory).

    Skips files already fully downloaded and resumes partial ones via HTTP range
    requests. Use this for offline work, very large pulls, or feeding the parquet
    to other tools (DuckDB, Spark). Requires Data-Use-Agreement acceptance.

    Args:
        category: One category or a list; None = all.
        start: Inclusive ``'YYYY-MM'`` lower bound (None = no bound).
        end: Inclusive ``'YYYY-MM'`` upper bound (None = no bound).
        fmt: ``'parquet'`` (default), ``'csv'``, or None for both.
        dest: Destination directory. Defaults to the package cache
            (``cache_dir()/files``). Files are laid out as
            ``<dest>/<category>/<prefix>YYYY-MM.<ext>`` (``ALL_`` for
            sexuality, ``RC_`` for the other categories).
        refresh: Force a manifest refresh first. Defaults to False.

    Returns:
        A list of local ``pathlib.Path`` objects, one per downloaded file.

    Raises:
        ValueError: if no files match the selection.
        AgreementNotAccepted: if the Data Use Agreement has not been accepted.
        FileNotFoundError: if a selected file is missing on the data host.
        DataHostUnavailable: if the data host fails every attempt at a file.
    """
    require_acceptance()
    sel = files(category, start, end, fmt, refresh=refresh)
    if sel.empty:
        raise ValueError("No files match the selection.")
    root = Path(dest).expanduser() if dest else (cache_dir() / "files")
    out: List[Path] = []
    for row in sel.itertuples(index=False):
        out.append(_download_one(row.url, root / row.rel_path))
    return out


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def read_parquet(url: str, columns: Optional[Sequence[str]] = None):
    """Read a single parquet file into a pandas DataFrame.

    Args:
        url: An http(s) URL (e.g. a catalog ``url``) or a local file path.
        columns: Restrict to these columns. For http URLs this is a true
            pushdown — only the file footer and the requested column chunks are
            transferred (HTTP range requests). None reads all columns.

    Returns:
        A pandas DataFrame.

    Raises:
        AgreementNotAccepted: for http URLs, if the Data Use Agreement is not accepted.
        FileNotFoundError: for http URLs, if the host does not have the file.
        DataHostUnavailable: if the host fails every attempt, streamed and downloaded.
    """
    import pyarrow.parquet as pq

    if str(url).startswith(("http://", "https://")):
        require_acceptance()
        try:
            with _open_remote_parquet(url) as fh:
                return pq.read_table(fh, columns=columns).to_pandas()
        except DataHostUnavailable:
            local = _fallback_to_download(url)
        return pq.read_table(str(local), columns=list(columns) if columns else None).to_pandas()
    return pq.read_table(url, columns=list(columns) if columns else None).to_pandas()


def _read_csv_source(url, rel_path, columns, cache):
    """Read a CSV, over HTTP or from the cache, retrying a flaky host."""
    import pandas as pd

    usecols = list(columns) if columns else None
    if cache:
        return pd.read_csv(str(_download_one(url, cache_dir() / "files" / rel_path)), usecols=usecols)
    try:
        return pd.read_csv(url, usecols=usecols)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        # pandas streams CSV over urllib, so a backend hiccup surfaces either as
        # an HTTPError/URLError or as an empty/garbled body. No ISAAC CSV is
        # empty, so re-fetch through the retrying, size-checked download path;
        # if the file really is malformed, reading the local copy raises the
        # same error and the user still learns the truth.
        _, missing, why = _probe_url(url)
        if missing:
            raise FileNotFoundError(_missing_message(url, why)) from None
        return pd.read_csv(str(_fallback_to_download(url, rel_path)), usecols=usecols)


def _read_one(row, columns, cache):
    """Read one whole file (column-projected) into a DataFrame. Used by load()."""
    if row.format == "parquet":
        if cache:
            p = _download_one(row.url, cache_dir() / "files" / row.rel_path)
            return read_parquet(str(p), columns=columns)
        return read_parquet(row.url, columns=columns)
    return _read_csv_source(row.url, row.rel_path, columns, cache)


def _sample_one(row, k, columns, seed_seq, cache):
    """Sample k rows from one file (caller has already done the ToU gate)."""
    import numpy as np
    import pandas as pd
    from .sampling import sample_parquet

    rng = np.random.default_rng(seed_seq)
    num_rows = int(row.num_rows or 0)
    if num_rows <= 0:
        return pd.DataFrame(columns=list(columns) if columns else [])

    if row.format == "parquet":
        if cache:
            p = _download_one(row.url, cache_dir() / "files" / row.rel_path)
            return sample_parquet(str(p), num_rows, k, columns, rng)
        try:
            with _open_remote_parquet(row.url) as fh:
                return sample_parquet(fh, num_rows, k, columns, rng)
        except DataHostUnavailable:
            local = _fallback_to_download(row.url, row.rel_path)
        return sample_parquet(str(local), num_rows, k, columns, rng)

    # CSV fallback: no cheap random access over HTTP, so read then take positions.
    full = _read_csv_source(row.url, row.rel_path, columns, cache)
    if len(full) <= k:
        return full.reset_index(drop=True)
    pos = np.sort(rng.choice(len(full), size=k, replace=False))
    return full.iloc[pos].reset_index(drop=True)


def load(
    category: Optional[Union[str, Sequence[str]]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    fmt: str = "parquet",
    columns: Optional[Sequence[str]] = None,
    n: Optional[int] = None,
    seed: Optional[int] = None,
    combine: bool = True,
    cache: bool = False,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    refresh: bool = False,
):
    """Load selected files into pandas.

    Selects files with `files`, reads each (parquet by default), and returns them
    combined or per file. Requires Data-Use-Agreement acceptance.

    Args:
        category: One category or a list; None = all.
        start: Inclusive ``'YYYY-MM'`` lower bound (None = no bound).
        end: Inclusive ``'YYYY-MM'`` upper bound (None = no bound).
        fmt: ``'parquet'`` (default, recommended) or ``'csv'``.
        columns: Restrict to these columns. For parquet this is pushed down over
            HTTP, so only those columns are transferred. None reads all columns.
        n: If set, draw a total of ``n`` rows, spread **equally across the
            selected months** (capped by each month's size) and sampled uniformly
            at random within each month — matching the web app. For parquet, only
            the **selected columns** of the row groups containing sampled rows are
            transferred (so always pass ``columns=`` when sampling). Row groups
            with no sampled rows are skipped (this helps most when ``n`` is small
            relative to a file's row-group count). ``max_bytes`` is not enforced
            when sampling, since whole files aren't read.
        seed: Random seed making the ``n`` sample reproducible (independent per month).
        combine: True (default) returns one concatenated DataFrame; False returns
            a dict keyed by ``(category, year, month)``.
        cache: If True, download whole files to the cache and read them locally
            instead of streaming (faster when re-reading the same files).
            Defaults to False.
        max_bytes: Safety guardrail (default 5 GB). Refuses selections larger
            than this unless you pass ``columns=``, set ``n=``, raise this value,
            or use `download`.
        refresh: Force a manifest refresh first. Defaults to False.

    Returns:
        A pandas DataFrame (``combine=True``) or a dict mapping
        ``(category, year, month)`` to DataFrames (``combine=False``). Each frame
        gains leading ``_category`` and ``_month`` columns.

    Raises:
        ValueError: if no files match, or the selection exceeds ``max_bytes``
            without ``columns`` or ``n``.
        AgreementNotAccepted: if the Data Use Agreement has not been accepted.
        FileNotFoundError: if a selected file is missing on the data host.
        DataHostUnavailable: if the data host fails every attempt at a file.
    """
    import pandas as pd

    require_acceptance()
    sel = files(category, start, end, fmt, refresh=refresh)
    if sel.empty:
        raise ValueError("No files match the selection.")

    sampling = n is not None
    if sampling:
        import numpy as np
        from .sampling import equal_quotas
        caps = [int(x or 0) for x in sel["num_rows"].tolist()]
        quotas = equal_quotas(caps, int(n))
        seeds = list(np.random.SeedSequence(seed).spawn(len(sel)))
    else:
        total = int(sel["size_bytes"].sum())
        if columns is None and total > max_bytes:
            raise ValueError(
                f"Selection is ~{total/1e9:.1f} GB across {len(sel)} files "
                f"(> max_bytes={max_bytes/1e9:.1f} GB). Narrow the range, pass columns=, "
                f"set n= to sample, raise max_bytes=, or use download()."
            )

    frames: Dict[Tuple[str, int, int], "pd.DataFrame"] = {}
    for i, row in enumerate(sel.itertuples(index=False)):
        if sampling:
            if quotas[i] <= 0:
                continue
            df = _sample_one(row, quotas[i], columns, seeds[i], cache)
        else:
            df = _read_one(row, columns, cache)
        df.insert(0, "_category", row.category)
        df.insert(1, "_month", f"{row.year:04d}-{row.month:02d}")
        frames[(row.category, row.year, row.month)] = df

    if combine:
        return pd.concat(frames.values(), ignore_index=True) if frames else pd.DataFrame()
    return frames
