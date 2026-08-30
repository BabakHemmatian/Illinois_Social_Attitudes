"""
Pre-populate user_map.sqlite3 with every author in a set of curated months, so
that a subsequent parallel organize_anonymize run never needs the write lock.

Why this exists
---------------
organize_anonymize assigns IDs lazily: each previously unseen author gets its
own BEGIN IMMEDIATE / INSERT / COMMIT. That is ~30k commits per month against a
1.2 GB database, and every WAL checkpoint becomes random I/O across it. On the
project's USB drive that dominates runtime -- measured 2026-07-27 at 0.9 MB/s
write, with per-month cost climbing from 12 min (2012-06) to 38 min (2013-05)
while row counts rose only 34%.

Running shards in parallel makes it worse, not better: they all serialize on the
single WAL writer (measured 2026-07-06: 0.1 MB/s across 8 shards vs 2.4 MB/s
single-process).

This script collapses those per-author commits into one transaction per batch.
Afterwards every author is a read hit, so organize_anonymize shards can open the
cache read-only and run without contending for a write lock at all.

Correctness note: this does NOT change which IDs exist, only when they are
created. Author cleaning and ID generation are byte-identical to
organize_anonymize (see clean_author / generate_candidate_id below); if those
ever diverge, warmed IDs would be assigned to keys the anonymizer never looks
up, and it would mint duplicates for the real keys.

Usage:
    python code/warm_author_map.py --group age --years 2013-2023          # warm
    python code/warm_author_map.py --group age --years 2013-2023 --check  # count only
"""

### Imports

import argparse
import csv
import os
import re
import secrets
import sqlite3
import sys
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

csv.field_size_limit(2**31 - 1)

### Configuration

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CURATED = DATA_DIR / "data_reddit_curated"
USER_CACHE = DATA_DIR / "user_map.sqlite3"

MONTH_RE = re.compile(r"(\d{4})-(\d{2})")

# Must stay identical to organize_anonymize.should_preserve_author
PRESERVED_AUTHORS = {"", "[deleted]", "[removed]"}

# Batching bounds peak memory and banks progress: each batch's authors are
# committed before the next batch is read, so an interrupted run only loses the
# batch in flight and can simply be re-run.
#
# Batches are sized by input BYTES, not month count, because months differ by
# four orders of magnitude -- ALL_2007-01 is 2,121 rows while ALL_2023-08 is
# 12 GB. A fixed month count that is safe for 2007 is ruinous for 2023.
#
# Sizing: a Python set costs ~95 bytes per author, and the
# corpus runs ~433k rows/GB with roughly a third of rows carrying a distinct
# author -- so about 12 MB of author set per GB of input. 24 GB per batch keeps
# the parent near 300 MB, which fits comfortably even when free RAM is tight.
DEFAULT_BATCH_GB = 24.0
DEFAULT_BATCH_MONTHS = 12  # hard cap, mainly to bound tiny early-year months


### Author handling -- must mirror organize_anonymize exactly

def clean_author(value: Optional[str]) -> Optional[str]:
    """Return the map key for an author cell, or None if it is a placeholder."""
    if value is None:
        return None
    cleaned = str(value).strip()
    if cleaned in PRESERVED_AUTHORS:
        return None
    return cleaned


def generate_candidate_id(num_digits: int = 12) -> str:
    if num_digits < 6:
        raise ValueError("num_digits should be at least 6")
    lower = 10 ** (num_digits - 1)
    upper = (10 ** num_digits) - 1
    return str(secrets.randbelow(upper - lower + 1) + lower)


### Scanning

# Runs in a worker process: pure read, no database handle. Returns the distinct
# author keys in one month file.
def scan_month(path: str) -> Tuple[str, Set[str], int]:
    authors: Set[str] = set()
    rows = 0
    with open(path, "rb") as f:
        reader = csv.reader(line.decode("utf-8", "replace") for line in f)
        header = next(reader, None)
        if not header:
            return path, authors, 0
        try:
            a_i = header.index("author")
        except ValueError:
            raise ValueError(f"No 'author' column in {Path(path).name}")
        width = len(header)
        for row in reader:
            rows += 1
            if len(row) != width:
                continue
            key = clean_author(row[a_i])
            if key is not None:
                authors.add(key)
    return path, authors, rows


### Database

def open_cache(db_path: Path, synchronous: str = "FULL",
               cache_mb: int = 512) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=120, isolation_level=None)
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    # FULL rather than organize_anonymize's NORMAL: this is the only phase that
    # writes, and NORMAL does not fsync the WAL per commit, so WAL data can be
    # lost if the node drops mid-run. One bulk commit per batch makes the
    # stricter setting essentially free.
    cur.execute(f"PRAGMA synchronous={synchronous};")
    cur.execute("PRAGMA busy_timeout=120000;")
    # SQLite defaults to a 2 MB page cache. Against a 1.3 GB author map that is
    # catastrophic here, because each batch streams 24 GB of CSVs which flushes
    # the OS file cache and evicts every database page before the insert phase
    # begins. With the default cache, insert throughput degrades sharply batch
    # over batch even as the map itself grows only slightly. A large cache is
    # private heap memory, so unlike mmap the CSV streaming cannot evict it.
    cur.execute(f"PRAGMA cache_size=-{cache_mb * 1024};")
    cur.execute("PRAGMA temp_store=MEMORY;")
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS author_map (
            author TEXT PRIMARY KEY,
            anon_id TEXT NOT NULL UNIQUE,
            created_at INTEGER NOT NULL
        );
        """
    )
    cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_author_map_anon_id ON author_map(anon_id);")
    conn.commit()
    return conn


# Probing in SORTED order is the whole trick here. Iterating a set yields hash
# order, so consecutive probes land in unrelated parts of the author index and
# every one costs a physical seek: the database gets re-read many times over
# without finishing a single batch. Sorted probes walk the
# index in ascending order instead, so each page read serves many consecutive
# lookups and the traversal is effectively sequential.
def filter_known(conn: sqlite3.Connection, authors: Set[str]) -> Set[str]:
    """Drop authors that already have an ID, returning only the new ones."""
    unknown = set()
    ids = sorted(authors)
    cur = conn.cursor()
    for i in range(0, len(ids), 900):  # SQLite host-parameter limit
        chunk = ids[i:i + 900]
        q = ",".join("?" * len(chunk))
        cur.execute(f"SELECT author FROM author_map WHERE author IN ({q})", chunk)
        known = {r[0] for r in cur.fetchall()}
        unknown.update(a for a in chunk if a not in known)
    return unknown


# Preloading every existing key is the default because per-author probing is
# orders of magnitude slower than one sequential scan of the key column: random
# index probes cost a seek each and are repeated every batch, while the scan is
# sequential and runs once. Sorting the probes first does not close the gap,
# because a batch's keys spread over a multi-million-row index hit a distinct
# page either way.
#
# Keys are stored as 64-bit hashes in a sorted int64 array, roughly an order of
# magnitude smaller than the set of strings it replaces, which matters when free
# RAM is tight. Python's per-process hash randomisation is fine here -- the
# array is rebuilt from the database on every run and never persisted.
#
# Collision exposure is negligible: ~6e-13 per new author against 11.1M existing
# hashes, so ~2e-6 expected across the whole run. Were one to occur, that author
# would simply never be assigned an ID, and the read-only sharded anonymize run
# fails loudly naming them rather than corrupting anything.
def load_known_hashes(conn: sqlite3.Connection) -> np.ndarray:
    cur = conn.cursor()
    cur.execute("SELECT author FROM author_map")
    parts: List[np.ndarray] = []
    while True:
        chunk = cur.fetchmany(200_000)
        if not chunk:
            break
        parts.append(np.fromiter((hash(r[0]) for r in chunk),
                                 dtype=np.int64, count=len(chunk)))
    if not parts:
        return np.empty(0, dtype=np.int64)
    return np.sort(np.concatenate(parts))


def hash_array(authors: List[str]) -> np.ndarray:
    return np.fromiter((hash(a) for a in authors), dtype=np.int64, count=len(authors))


def split_new(known: np.ndarray, authors: Set[str]) -> Tuple[List[str], np.ndarray]:
    """Return (authors absent from `known`, their hashes)."""
    ordered = list(authors)
    if not ordered:
        return [], np.empty(0, dtype=np.int64)
    h = hash_array(ordered)
    if known.size == 0:
        return ordered, h
    idx = np.minimum(np.searchsorted(known, h), known.size - 1)
    seen = known[idx] == h
    new = [a for a, s in zip(ordered, seen) if not s]
    return new, h[~seen]


# One transaction for the whole batch instead of one per author. A UNIQUE
# violation on anon_id rolls back only the offending statement (SQLite's default
# ON CONFLICT ABORT), so the transaction survives and the row is retried with a
# fresh random ID. At 12 digits against ~10.5M existing IDs a collision runs
# about 1 in 86,000, so retries are rare but not negligible in bulk.
def insert_new_authors(conn: sqlite3.Connection, authors: Set[str],
                       max_retries: int = 8) -> Tuple[int, int]:
    if not authors:
        return 0, 0

    cur = conn.cursor()
    now = int(time.time())
    inserted = 0
    collisions = 0

    # Sorted order was expected to give the primary-key index locality, but
    # shows no measurable benefit. Kept because sorting is negligible next to
    # the inserts and the order is at worst neutral -- but do not expect it to
    # speed anything up. The binding cost is the anon_id unique index, which is
    # random by construction because those IDs are deliberately unpredictable,
    # and no insert ordering can fix that.
    cur.execute("BEGIN IMMEDIATE;")
    try:
        for author in sorted(authors):
            for attempt in range(max_retries):
                try:
                    cur.execute(
                        "INSERT INTO author_map(author, anon_id, created_at) VALUES (?, ?, ?)",
                        (author, generate_candidate_id(), now),
                    )
                    inserted += 1
                    break
                except sqlite3.IntegrityError:
                    # Could be an anon_id collision (retry) or a concurrent
                    # writer having just inserted this author (stop).
                    cur.execute("SELECT 1 FROM author_map WHERE author = ?", (author,))
                    if cur.fetchone() is not None:
                        break
                    collisions += 1
                    if attempt == max_retries - 1:
                        raise RuntimeError(
                            f"Could not find a free anon_id for {author!r} "
                            f"after {max_retries} attempts"
                        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    return inserted, collisions


### Driver

# A month is safe to skip only if it was LOGGED complete. Testing whether the
# output file merely exists is wrong: an interrupted run leaves a partial file
# (e.g. ALL_2013-07.csv, 39 MB of an expected 1.5 GB), and only the authors in
# the portion actually written ever reached the map. Skipping those would leave
# ~97% of that month's authors unwarmed, putting the anonymizer straight back on
# the per-author commit path this script exists to avoid.
#
# OFF BY DEFAULT -- this attribution is NOT reliable, and it is only reachable
# via --trust-anonymize-log. The group/stage below is carried as PARSER STATE
# from the last 'X identified as the most advanced curated dataset for <group>
# entries' line, but every job appends to one shared
# report_organize_anonymize.csv. Concurrent runs interleave, so an 'Anonymized
# ALL_...' line can be attributed to whichever group last wrote a context header
# rather than to the job that emitted it. Real months then get skipped as
# "already anonymized" and are missing from the map, which under a read-only
# (--array) run aborts each month mid-write while the task still exits 0.
#
# warmed_months_from_log() below is the safe skip source: its format is one flat
# 'timestamp,group,stage,months...' record per line, with no cross-line state,
# so interleaved writers cannot mix records up.
def completed_months_from_log(report_path: Path, group: str, stage: str) -> Set[str]:
    if not report_path.exists():
        return set()

    done: Set[str] = set()
    ctx_group = ctx_stage = None
    stage_re = re.compile(r"([A-Za-z]\w*) identified as the most advanced curated "
                          r"dataset for (\w+) entries of type")

    with open(report_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = stage_re.search(line)
            if m:
                ctx_stage, ctx_group = m.group(1), m.group(2)
                continue
            m = re.search(r"Anonymized ALL_(\d{4}-\d{2})\.csv", line)
            if m and ctx_group == group and ctx_stage == stage:
                done.add(m.group(1))
    return done


# Warming is idempotent but not cheap to repeat: re-running re-reads every month
# it already covered (41 GB two batches in, ~600 GB near the end). Recording each
# completed batch keeps a restart proportional to what is actually left.
WARM_LOG = "report_warm_author_map.csv"


def warmed_months_from_log(report_path: Path, group: str, stage: str) -> Set[str]:
    if not report_path.exists():
        return set()
    done: Set[str] = set()
    with open(report_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4 and parts[1] == group and parts[2] == stage:
                done.update(m for m in parts[3:] if MONTH_RE.fullmatch(m))
    return done


def record_warmed(report_path: Path, group: str, stage: str, months: List[Path]) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    names = [MONTH_RE.search(p.name).group(0) for p in months]
    with open(report_path, "a", encoding="utf-8", newline="") as f:
        f.write(",".join([stamp, group, stage, *names]) + "\n")


def resolve_months(merged_dir: Path, years: List[int],
                   completed: Set[str]) -> List[Path]:
    out = []
    for p in sorted(merged_dir.glob("ALL_*.csv")):
        m = MONTH_RE.search(p.name)
        if not m or int(m.group(1)) not in years:
            continue
        if m.group(0) in completed:
            continue
        out.append(p)
    return out


# Greedily fill batches up to max_gb of input, never exceeding max_months. A
# single month larger than max_gb still forms its own batch rather than being
# skipped -- 2023 months run ~12 GB each.
def build_batches(months: List[Path], max_gb: float, max_months: int) -> List[List[Path]]:
    batches: List[List[Path]] = []
    current: List[Path] = []
    current_bytes = 0
    limit = max_gb * 2**30

    for p in months:
        size = p.stat().st_size
        if current and (current_bytes + size > limit or len(current) >= max_months):
            batches.append(current)
            current, current_bytes = [], 0
        current.append(p)
        current_bytes += size

    if current:
        batches.append(current)
    return batches


def parse_years(spec: str) -> List[int]:
    years: Set[int] = set()
    for token in (t.strip() for t in spec.split(",") if t.strip()):
        if "-" in token:
            a, b = token.split("-", 1)
            years.update(range(int(a), int(b) + 1))
        else:
            years.add(int(token))
    return sorted(years)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", required=True)
    ap.add_argument("--years", required=True, help="e.g. 2013-2023 or 2013,2015-2017")
    ap.add_argument("--stage", default="labeled_location")
    ap.add_argument("--db", type=Path, default=USER_CACHE)
    ap.add_argument("--workers", type=int, default=4,
                    help="parallel readers; these do no DB work (default 4)")
    ap.add_argument("--batch-gb", type=float, default=DEFAULT_BATCH_GB,
                    help=f"max input GB per commit; bounds peak RAM "
                         f"(default {DEFAULT_BATCH_GB:g})")
    ap.add_argument("--batch-months", type=int, default=DEFAULT_BATCH_MONTHS,
                    help=f"hard cap on months per commit (default {DEFAULT_BATCH_MONTHS})")
    ap.add_argument("--all-months", action="store_true",
                    help="include months already logged complete (safe but slower; "
                         "warming is idempotent)")
    ap.add_argument("--trust-anonymize-log", action="store_true",
                    help="also skip months that report_organize_anonymize.csv claims "
                         "were already anonymized. OFF by default: concurrent jobs "
                         "share that file and its group attribution is parser state, "
                         "so months get skipped that were never warmed (see "
                         "completed_months_from_log).")
    ap.add_argument("--cache-mb", type=int, default=512,
                    help="SQLite page cache in MB (default 512). The 2 MB default "
                         "collapses on a 1.3 GB map once CSV streaming evicts the "
                         "OS file cache.")
    ap.add_argument("--probe-map", action="store_true",
                    help="query the map per batch instead of preloading it. Only "
                         "sensible for a handful of months: probing runs ~460 "
                         "keys/s versus 80,650 for the one-off preload scan.")
    ap.add_argument("--check", action="store_true",
                    help="report how many authors are new, write nothing")
    args = ap.parse_args()

    years = parse_years(args.years)
    merged = CURATED / args.group / "all" / args.stage
    anon = CURATED / args.group / "all" / f"{args.stage}_anon"
    if not merged.is_dir():
        print(f"No such directory: {merged}", file=sys.stderr)
        return 1

    completed = set()
    if not args.all_months:
        # The warm log is the only trustworthy skip source: flat, one record per
        # line, no cross-line state for interleaved writers to corrupt.
        already_warm = warmed_months_from_log(
            PROJECT_ROOT / WARM_LOG, args.group, args.stage)
        if already_warm:
            print(f"skipping {len(already_warm)} months already warmed in a prior run.")
        completed |= already_warm

        # Opt-in only -- see the warning on completed_months_from_log. Re-warming
        # an already-anonymized month is idempotent and costs one extra scan, so
        # the default trades that scan for not silently leaving months unwarmed.
        if args.trust_anonymize_log:
            anon_done = completed_months_from_log(
                PROJECT_ROOT / "report_organize_anonymize.csv", args.group, args.stage)
            if anon_done:
                print(f"skipping {len(anon_done)} months already anonymized "
                      f"(--trust-anonymize-log; attribution may be wrong).")
            completed |= anon_done
    months = resolve_months(merged, years, completed)
    if completed:
        print(f"{len(completed)} months skipped in total.")
    if not months:
        print("No months to warm.")
        return 0

    total_bytes = sum(p.stat().st_size for p in months)
    print(f"group={args.group} stage={args.stage}  db={args.db}")
    batches = build_batches(months, args.batch_gb, args.batch_months)
    peak = max(sum(p.stat().st_size for p in b) for b in batches) / 2**30
    print(f"{len(months)} months to scan ({total_bytes / 2**30:.1f} GB) with "
          f"{args.workers} readers, in {len(batches)} batches of <= {args.batch_gb:g} GB.")
    print(f"heaviest batch {peak:.1f} GB -> parent author set ~{peak * 12:.0f} MB.", flush=True)
    if args.check:
        print("--check: no writes will be made.\n")

    conn = open_cache(args.db, cache_mb=args.cache_mb)
    try:
        before = conn.execute("SELECT COUNT(*) FROM author_map").fetchone()[0]
        print(f"author_map currently holds {before:,} mappings.\n")

        known: Optional[np.ndarray] = None
        if not args.probe_map:
            t = time.time()
            print("preloading author keys (one sequential scan)...", flush=True)
            known = load_known_hashes(conn)
            print(f"preloaded {known.size:,} keys in {time.time() - t:.0f}s "
                  f"({known.nbytes / 2**20:.0f} MB resident)", flush=True)

        started = time.time()
        grand_new = grand_collisions = grand_rows = 0

        for batch in build_batches(months, args.batch_gb, args.batch_months):
            label = f"{MONTH_RE.search(batch[0].name).group(0)}..{MONTH_RE.search(batch[-1].name).group(0)}"
            batch_gb = sum(p.stat().st_size for p in batch) / 2**30

            authors: Set[str] = set()
            rows = 0
            t0 = time.time()
            done = 0
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(scan_month, str(p)): p for p in batch}
                for fut in as_completed(futs):
                    path, found, n = fut.result()
                    authors |= found
                    rows += n
                    done += 1
                    print(f"    scan {done}/{len(batch)}  {Path(path).name}  "
                          f"{n:,} rows  {len(found):,} authors  "
                          f"[{time.time() - t0:.0f}s]", flush=True)

            t1 = time.time()
            print(f"    looking up {len(authors):,} authors against the map...", flush=True)
            if known is not None:
                new_list, new_hashes = split_new(known, authors)
                new = set(new_list)
            else:
                new = filter_known(conn, authors)
                new_hashes = None
            print(f"    lookup done in {time.time() - t1:.0f}s -> {len(new):,} new", flush=True)
            grand_rows += rows

            if args.check:
                print(f"  {label}  {len(batch):>2} mo  {batch_gb:5.1f} GB  rows={rows:,}  "
                      f"distinct authors={len(authors):,}  new={len(new):,}  "
                      f"[set ~{len(authors) * 95 / 2**20:.0f} MB]", flush=True)
                grand_new += len(new)
                continue

            t2 = time.time()
            inserted, collisions = insert_new_authors(conn, new)
            if known is not None and new_hashes is not None and new_hashes.size:
                known = np.sort(np.concatenate([known, new_hashes]))
            print(f"    inserted {inserted:,} in {time.time() - t2:.0f}s", flush=True)
            # Fold the WAL back into the main DB so it cannot grow unbounded
            # across batches and so a later reader never replays a huge log.
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            # Record only after the commit and checkpoint have both landed, so a
            # crash can never mark a batch warm that is not durably in the map.
            record_warmed(PROJECT_ROOT / WARM_LOG, args.group, args.stage, batch)
            grand_new += inserted
            grand_collisions += collisions
            print(f"  {label}  {len(batch):>2} mo  {batch_gb:5.1f} GB  rows={rows:,}  "
                  f"distinct authors={len(authors):,}  inserted={inserted:,}" + (f"  id-collisions retried={collisions}" if collisions else ""))

        after = conn.execute("SELECT COUNT(*) FROM author_map").fetchone()[0]
        elapsed = (time.time() - started) / 60
        print(f"\n{'=' * 64}")
        print(f"scanned {grand_rows:,} rows across {len(months)} months in {elapsed:.1f} min")
        if args.check:
            print(f"{grand_new:,} authors would be newly assigned "
                  f"(map would go {before:,} -> {before + grand_new:,})")
        else:
            print(f"inserted {grand_new:,} new mappings; author_map {before:,} -> {after:,}")
            if grand_collisions:
                print(f"anon_id collisions retried: {grand_collisions}")
            dupes = conn.execute(
                "SELECT COUNT(*) - COUNT(DISTINCT anon_id) FROM author_map").fetchone()[0]
            print(f"duplicate anon_id check: {dupes} (must be 0)")
            if dupes:
                return 1
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
