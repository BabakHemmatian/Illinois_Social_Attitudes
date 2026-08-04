# isaac-data

A thin Python loader for the **ISAAC** Reddit corpus (Illinois Social Attitudes
Aggregate Corpus). It reads the public
[Direct Download catalog](https://isaac.psychology.illinois.edu/direct-download/)
— using the published `manifest.json` as the catalog, with the data files served
directly from the project's public Globus collection on NCSA Taiga — so you don't
have to hand-build URLs or stitch months together.

- **Catalog-driven**: enumerate what exists; never hard-code filenames.
- **Parquet column pushdown**: ask for a few columns and only those bytes are
  transferred over HTTP (via pyarrow + fsspec).
- **Resumable, cached downloads** for bulk/offline work.
- **pandas** out of the box.

## Install

```bash
pip install isaac-data
# or, from source:
pip install git+https://github.com/BabakHemmatian/Illinois_Social_Attitudes.git#subdirectory=isaac-data-loader
```

### Data Use Agreement

The first time you **access data** (`load`, `download`, or a remote `read_parquet`),
the package shows the ISAAC
[Data Use Agreement](https://github.com/BabakHemmatian/Illinois_Social_Attitudes/blob/main/Data_Use_Agreement.md)
and asks you to accept, then asks for your email address. Browsing the catalog
(`catalog`, `files`) needs no acceptance.

**What is recorded.** Acceptance is saved on your machine (in your OS config dir)
and sent to the ISAAC server: your email, the timestamp, and the version
identifiers of the agreement text you were shown. An email address is required
to accept.

> We ask for your email so we can notify you of changes to the Data Use
> Agreement and of corrections or errata affecting the corpus, and to keep a
> record of your acceptance. We do not share it, and we don't use it for
> anything else.

Sending the record is best-effort: if the server is unreachable, acceptance is
still recorded locally and data access proceeds.

**If the agreement changes.** The package identifies the agreement by a SHA-256
of its exact text and re-checks at most once a day. If the text has changed since
you accepted, you are shown the new version and asked to accept it again. If you
are offline, your existing acceptance stands.

For non-interactive use (CI, headless notebooks), accept ahead of time:

```bash
isaac-data accept-agreement                        # interactive review + accept
isaac-data accept-agreement --yes --email you@x.edu # accept non-interactively
isaac-data accept-agreement --status               # show / --withdraw to revoke
```
…or set `ISAAC_ACCEPT_AGREEMENT=1` **together with** `ISAAC_AGREEMENT_EMAIL` —
there is no prompt to fall back on in a non-interactive session, so opting in
without an address raises `AgreementNotAccepted` rather than recording an
anonymous acceptance.

> **Renamed in 0.1.2** — this document was previously the "Terms of Use". The old
> names still work: `isaac-data accept-terms`, `ISAAC_ACCEPT_TERMS`,
> `TermsNotAccepted`, `isaac_data.accept_terms`, and `import isaac_data.terms`.
> Existing local acceptance records remain valid; no need to re-accept.

## Quick start

```python
import isaac_data as isaac

# 1) What's available?
cat = isaac.catalog()                       # full manifest as a DataFrame
race = isaac.files("race", "2018-01", "2018-12")   # filter by category + months

# 2) Load a slice — only the columns you need (pushed down over HTTP)
df = isaac.load("race", "2018-03", "2018-03", columns=["text", "score"])

# 3) Stratified sample: 1000 rows TOTAL, spread equally across the 12 months
#    (uniform within each month, reproducible). Pass columns= when sampling.
sample = isaac.load("age", "2015-01", "2015-12", columns=["text"], n=1000, seed=0)

# 4) Bulk download for offline use (resumable, cached)
paths = isaac.download("weight", "2020-01", "2020-12", dest="./weight2020")
```

Categories: `ability, age, race, sexuality, skin_tone, weight`
(monthly, 2007-01 → 2023-12). Both `parquet` (default, recommended) and `csv`.

## How it works

The package is three layers — **discover → read/fetch → configure**:

1. **Discover.** `catalog()` downloads the published `manifest.json` (the
   authoritative list of every file) and returns it as a DataFrame, cached for
   24h. `files(...)` filters that catalog by category, month range, and format.
   Neither transfers any corpus data, so you can inspect sizes and row-counts
   before pulling anything.
2. **Read or fetch.**
   - `load(...)` is the main entry point. It selects files, then for **parquet**
     streams *only the columns you ask for* over HTTP — it reads the file footer,
     then just those column chunks, so `columns=["text","score"]` from a 285 MB
     file moves a few MB, not 285. `n=` draws a **stratified total** — spread
     equally across the selected months, uniform within each (matching the web
     app) — reading only the selected columns of the row groups that contain
     sampled rows. Each row is tagged with `_category`/`_month`; a `max_bytes`
     guard prevents accidental hundred-GB full loads.
   - `read_parquet(url, columns=...)` is the single-file primitive `load` uses.
   - `download(...)` fetches whole files to disk (resumable, skips complete ones)
     without loading them into memory — for offline work or other tools
     (DuckDB, Spark).
3. **Configure.** Reads and downloads are cached under an OS-native directory
   (`cache_dir()` / `set_cache_dir()` / `$ISAAC_DATA_CACHE`), and the first data
   access prompts for Data-Use-Agreement acceptance (recorded locally, and on
   the ISAAC server when you provide an email).

In short: *catalog tells you what exists → files narrows it → load streams just
the columns you need (or download grabs whole files) → the cache avoids repeat
transfers.*

> Full per-argument reference lives in the function docstrings (`help(isaac.load)`,
> IDE tooltips) and the generated [API docs](#documentation) — the table below is
> a summary.

## API

| Function | Purpose |
|---|---|
| `catalog(refresh=False)` | Full manifest as a DataFrame (cached 24h). |
| `files(category, start, end, fmt="parquet")` | Filtered file list. |
| `load(..., columns=None, n=None, seed=None, combine=True, cache=False)` | Read into pandas; column pushdown for parquet; `n` samples rows per file. |
| `read_parquet(url, columns=None)` | Read one parquet file (local or http) into pandas. |
| `download(..., dest=None)` | Resumable, cached bulk download; returns local paths. |
| `set_cache_dir(path)` / `cache_dir()` | Manage the local cache (default `~/.cache/isaac-data`, or `$ISAAC_DATA_CACHE`). |

`load()` has a safety guardrail (`max_bytes`, default 5 GB): it refuses very
large selections unless you pass `columns=`, set `n=`, raise `max_bytes=`, or use
`download()`.

## CLI

```bash
isaac-data info
isaac-data ls --category race --start 2018-01 --end 2018-12
isaac-data download --category age --start 2015-01 --end 2015-12 --dest ./age2015
```

## Documentation

The full per-argument API reference is generated from the docstrings with
[pdoc](https://pdoc.dev):

```bash
pip install "isaac-data[docs]"
pdoc -d google isaac_data -o docs/api    # static HTML into docs/api/
pdoc -d google isaac_data                # or a live preview server
```

(`docs/api/` is git-ignored; publish it to GitHub Pages if you want a hosted reference.)

## Requirements & platform notes

- **Python 3.9+**, on **Windows, macOS (Intel & Apple Silicon), and Linux**. All
  dependencies ship prebuilt wheels for these platforms.
- **Install footprint**: `pyarrow` is a large dependency (~100 MB installed) and
  `aiohttp` is a compiled extension. Fine on a laptop; size-conscious in slim
  CI/container images.
- **Cache location** is OS-native (via `platformdirs`): `~/Library/Caches/isaac-data`
  (macOS), `%LOCALAPPDATA%\isaac-data\Cache` (Windows), `~/.cache/isaac-data`
  (Linux). Override with `isaac.set_cache_dir(...)` or `$ISAAC_DATA_CACHE`.

## Notes

- **Parquet is recommended** for scripting (column projection, smaller transfers).
  The final labeled ISAAC release will add many per-post fields (moralization,
  sentiment, generalization, emotion, location) — column pushdown makes those
  cheap to query.
- For SQL-style predicate pushdown without Python, query the parquet directly
  with DuckDB (reads only the columns/row groups your query needs):
  ```sql
  SELECT author, score FROM
  read_parquet('https://isaac.psychology.illinois.edu/data/race/RC_2018-03.parquet')
  WHERE score > 100;
  ```
  HTTP has no directory listing, so wildcard globs don't work — for multiple
  months pass an explicit URL list, e.g. from the loader:
  `duckdb.sql("... read_parquet($u) ...", params={"u": isaac.files("race","2018-01","2018-12").url.tolist()})`.
