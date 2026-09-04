### Imports

import csv
csv.field_size_limit(2**31 - 1) # Increase the field size limit to handle larger fields
import os
import re
import secrets
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# import functions and objects
from cli import get_args, PROJECT_ROOT, DATA_DIR
from utils import (
    AUTHOR_MAP_WARM_LOG,
    parse_range,
    check_reqd_files,
    default_resource,
    log_report,
    find_latest_resource_dir,
    validate_resource_dir,
    get_resume_position,
    month_of_file,
    record_warmed_months,
    reraise_fatal,
    unwarmed_files,
    warmed_months_from_log,
)

### Argument Handling

# Extract and transform CLI arguments
args = get_args()
years = parse_range(args.years)
if isinstance(years, int):
    years = [years]

group = args.group
type_ = args.type

if type_ not in {"comments", "submissions", "all"}:
    raise ValueError(f"Unsupported 'type' argument: {type_}")

# Batch (--slurm) runs need every author pre-assigned an ID before the array
# starts (see "Author-Map Warming" below). cli.py queues that warm-up as a
# single-task job with ORGANIZE_ANONYMIZE_WARM=1 in its environment, and this
# script then warms instead of anonymizing -- the same mechanism label_location
# uses for its post-array merge (LABEL_LOCATION_MERGE=1).
WARM_MODE = os.environ.get("ORGANIZE_ANONYMIZE_WARM", "") == "1"

### Path Handling

# SQLite file storing author -> anonymized numeric ID mapping
USER_CACHE = DATA_DIR / "user_map.sqlite3"
# NOTE: this is where the key connecting user IDs to numbers lives. Keep safe.
# NOTE: label_location should be used before anonymizing, as it requires access to user IDs.

# prepare the report file
# NOTE: Since the dataset should be complete after this integration, the report gets saved to the project root for easier review.
report_file_path = PROJECT_ROOT / "report_organize_anonymize.csv"
# Machine-readable record of which months have been warmed (see utils).
warm_log_path = PROJECT_ROOT / AUTHOR_MAP_WARM_LOG

# Set/survey the input folders
data_base = DATA_DIR / "data_reddit_curated" / group / type_  # default

if not args.input:
    log_report(
        report_file_path,
        f"No custom input path provided. Finding the most advanced curated datasets of type '{type_}' based on default pathing and resource order..."
    )
    input_path = find_latest_resource_dir(data_base, default_resource)

    log_report(
        report_file_path,
        f"{input_path.name} identified as the most advanced curated dataset for {group} entries of type '{type_}'."
    )
else:
    input_path = validate_resource_dir(args.input, default_resource)

# strict=False: select_target_files() below matches on the YYYY-MM parsed from
# each filename, never on position in file_list, so months already anonymized
# (and whose inputs have since been deleted to reclaim storage) can be absent
# without shifting any task -> month mapping.
file_list = check_reqd_files(years=years, type_=type_, check_path=input_path, strict=False)
file_list = sorted(file_list, key=lambda p: Path(p).name)

# parse the output path
if not args.output:
    output_path = DATA_DIR / "data_reddit_curated" / group / type_ / f"{Path(input_path).name}_anon"
else:
    output_path = Path(args.output)

os.makedirs(output_path, exist_ok=True)

if args.output is not None and not output_path.is_dir():
    raise ValueError("The 'output' argument should be a directory path.")

### Slurm/array helpers

def build_requested_months(years_list: List[int]) -> List[Tuple[int, str]]:
    months: List[Tuple[int, str]] = []
    for year in years_list:
        for month in range(1, 13):
            months.append((year, f"{month:02d}"))
    return months

def month_from_filename(file_path: str | Path) -> Tuple[int, str]:
    m = re.search(r"(\d{4})-(\d{2})", Path(file_path).name)
    if not m:
        raise ValueError(f"Could not parse YYYY-MM from filename: {Path(file_path).name}")
    return int(m.group(1)), m.group(2)

def select_target_files(
    all_files: List[Path],
    years_list: List[int],
    array_idx: Optional[int],
    files_per_job: int,
) -> List[Path]:
    requested_months = build_requested_months(years_list)

    if array_idx is None:
        target_months = requested_months
    else:
        start_idx = array_idx * files_per_job
        end_idx = start_idx + files_per_job
        target_months = requested_months[start_idx:end_idx]

    target_month_set = set(target_months)
    if not target_month_set:
        return []

    out: List[Path] = []
    for p in all_files:
        ym = month_from_filename(p)
        if ym in target_month_set:
            out.append(Path(p))

    return sorted(out, key=lambda p: p.name)

### SQLite cache helpers

# read_only=True opens the map with SQLite's ro URI mode, so the connection is
# physically incapable of writing. Used by parallel shards after the warm-up
# pass has pre-assigned every author: with no writer there is no write lock to
# contend for, which is the whole point of warming. It is a stronger guarantee
# than trusting BEGIN IMMEDIATE to serialize correctly, and it turns "an author
# was missed during warming" into an immediate, named error instead of silent
# lock thrash.
#
# synchronous/cache_mb are tuned per caller: the anonymizer keeps SQLite's
# NORMAL (one tiny commit per new author, and a crash only loses IDs that are
# re-minted on resume), while the warm-up uses FULL because it is the one phase
# that bulk-writes and its single commit per batch makes the fsync free. A
# large private page cache matters for both: streaming multi-GB CSVs evicts the
# OS file cache, so without it every author lookup is a physical read.
def open_cache(db_path: str | Path, read_only: bool = False,
               synchronous: str = "NORMAL", cache_mb: int = 512) -> sqlite3.Connection:
    db_path = str(db_path)

    if read_only:
        if not os.path.exists(db_path):
            raise FileNotFoundError(
                f"Author map not found at {db_path}. The warm-up pass "
                f"(ORGANIZE_ANONYMIZE_WARM=1) must run before anonymizing in "
                f"read-only (sharded) mode; cli.py --slurm queues it automatically."
            )
        conn = sqlite3.connect(
            f"file:{Path(db_path).as_posix()}?mode=ro",
            uri=True,
            timeout=120,
            isolation_level=None,
        )
        cur = conn.cursor()
        cur.execute("PRAGMA busy_timeout=120000;")
        cur.execute(f"PRAGMA cache_size=-{cache_mb * 1024};")
        return conn

    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(
        db_path,
        timeout=120,
        isolation_level=None,  # explicit transactions
    )
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute(f"PRAGMA synchronous={synchronous};")
    cur.execute("PRAGMA busy_timeout=120000;")
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

# Generate a random numeric string with a fixed width.
def generate_candidate_id(num_digits: int = 12) -> str:
    if num_digits < 6:
        raise ValueError("num_digits should be at least 6")
    lower = 10 ** (num_digits - 1)
    upper = (10 ** num_digits) - 1
    return str(secrets.randbelow(upper - lower + 1) + lower)

# Read-only single-author lookup. Used by sharded runs, where the map is opened
# with mode=ro and the warm-up pass has already assigned every author, so a
# miss is a warming gap to report rather than an ID to mint.
def lookup_author_id(conn: sqlite3.Connection, author: str) -> str | None:
    if not author:
        return author
    cur = conn.cursor()
    cur.execute("SELECT anon_id FROM author_map WHERE author = ?", (author,))
    row = cur.fetchone()
    return row[0] if row is not None else None

# Concurrency-safe get-or-create: returns existing ID if present, otherwise assigns exactly one new unique random numeric ID.
# Returns (anon_id, created), where created is True only when THIS call performed
# the INSERT. Callers must not infer that from a separate read-then-write pair:
# with parallel array tasks two processes can both observe "absent" before either
# writes, and would then both count the same author as new.
def get_or_create_author_id(conn: sqlite3.Connection, author: str, num_digits: int = 12) -> Tuple[str, bool]:
    if not author:
        return author, False

    cur = conn.cursor()

    # Fast path: read first without taking a write lock
    cur.execute("SELECT anon_id FROM author_map WHERE author = ?", (author,))
    row = cur.fetchone()
    if row is not None:
        return row[0], False

    while True:
        try:
            # Serialize writers so two processes cannot assign two IDs to the same author
            cur.execute("BEGIN IMMEDIATE;")

            # Check again after acquiring the write lock
            cur.execute("SELECT anon_id FROM author_map WHERE author = ?", (author,))
            row = cur.fetchone()
            if row is not None:
                conn.commit()
                return row[0], False

            candidate = generate_candidate_id(num_digits=num_digits)
            now = int(time.time())

            cur.execute(
                "INSERT INTO author_map(author, anon_id, created_at) VALUES (?, ?, ?)",
                (author, candidate, now),
            )
            conn.commit()
            return candidate, True

        except sqlite3.IntegrityError:
            # Either anon_id collided or another process inserted first.
            conn.rollback()

            cur.execute("SELECT anon_id FROM author_map WHERE author = ?", (author,))
            row = cur.fetchone()
            if row is not None:
                return row[0], False

            # Otherwise anon_id collision; retry with a fresh random number.
            continue

        except sqlite3.OperationalError as e:
            conn.rollback()
            msg = str(e).lower()
            if "locked" in msg or "busy" in msg:
                time.sleep(0.1)
                continue
            raise

### Anonymization helpers

# Preserve common non-user Reddit placeholders as-is.
def should_preserve_author(author_value: str) -> bool:
    if author_value is None:
        return True

    author_clean = str(author_value).strip()
    return author_clean in {"", "[deleted]", "[removed]"}

# The map key for an author cell, or None for a preserved placeholder. Warming
# and anonymizing MUST agree on this key byte-for-byte: a warmed ID stored under
# a different key would never be looked up, and the read-only shards would then
# fail on the real key.
def author_map_key(author_value: Optional[str]) -> Optional[str]:
    if should_preserve_author(author_value):
        return None
    return str(author_value).strip()

### Author-Map Warming

# Why warming exists: anonymize_one_file assigns IDs lazily, one BEGIN
# IMMEDIATE / INSERT / COMMIT per previously unseen author. That is ~30k commits
# per month against a multi-GB database, and every WAL checkpoint becomes random
# I/O across it; per-month cost climbed from 12 min (2012-06) to 38 min
# (2013-05) while row counts rose only 34%. Running array tasks in parallel
# makes it WORSE: they all serialize on SQLite's single WAL writer (measured
# 0.1 MB/s across 8 shards vs 2.4 MB/s single-process). Warming collapses the
# per-author commits into one transaction per batch of months, after which
# every author is a read hit and the array tasks can open the map read-only and
# never contend for a write lock.
#
# Warming changes only WHEN IDs are created, never which IDs exist: key cleaning
# (author_map_key) and ID generation (generate_candidate_id) are shared with the
# anonymizer above.

# Batching bounds peak memory and banks progress: each batch's authors are
# committed before the next batch is read, so an interrupted run only loses the
# batch in flight. Batches are sized by input BYTES, not month count, because
# months differ by four orders of magnitude (ALL_2007-01 is 2,121 rows;
# ALL_2023-08 is 12 GB). A Python set costs ~95 bytes per author and the corpus
# runs ~433k rows/GB with roughly a third of rows carrying a distinct author, so
# about 12 MB of author set per GB of input: 24 GB per batch keeps the parent
# near 300 MB.
WARM_BATCH_GB = 24.0
WARM_BATCH_MONTHS = 12   # hard cap, mainly to bound tiny early-year months
WARM_WORKERS = 4         # parallel CSV readers; they do no database work

# Runs in a worker process: pure read, no database handle. Returns the distinct
# author keys in one month file.
def scan_month_authors(path: str) -> Tuple[str, Set[str], int]:
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
            key = author_map_key(row[a_i])
            if key is not None:
                authors.add(key)
    return path, authors, rows

# Preloading every existing key is far cheaper than probing per author: random
# index probes cost a seek each and are repeated every batch, while one
# sequential scan of the key column runs once. Keys are held as 64-bit hashes in
# a sorted int64 array, about an order of magnitude smaller than the strings it
# replaces. Python's per-process hash randomisation is fine here -- the array is
# rebuilt from the database on every run and never persisted. Collision
# exposure is negligible (~2e-6 across the whole corpus); a collided author
# would simply never be assigned an ID, and the read-only array task then fails
# loudly naming them rather than corrupting anything.
def load_known_author_hashes(conn: sqlite3.Connection) -> np.ndarray:
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

def split_new_authors(known: np.ndarray, authors: Set[str]) -> Tuple[List[str], np.ndarray]:
    """Return (authors absent from `known`, their hashes)."""
    ordered = list(authors)
    if not ordered:
        return [], np.empty(0, dtype=np.int64)
    h = np.fromiter((hash(a) for a in ordered), dtype=np.int64, count=len(ordered))
    if known.size == 0:
        return ordered, h
    idx = np.minimum(np.searchsorted(known, h), known.size - 1)
    seen = known[idx] == h
    new = [a for a, s in zip(ordered, seen) if not s]
    return new, h[~seen]

# One transaction for the whole batch instead of one per author. A UNIQUE
# violation on anon_id rolls back only the offending statement (SQLite's default
# ON CONFLICT ABORT), so the transaction survives and the row is retried with a
# fresh random ID. At 12 digits against ~10M existing IDs a collision runs about
# 1 in 86,000, so retries are rare but not negligible in bulk.
def insert_new_authors(conn: sqlite3.Connection, authors: Set[str],
                       max_retries: int = 8) -> Tuple[int, int]:
    if not authors:
        return 0, 0

    cur = conn.cursor()
    now = int(time.time())
    inserted = 0
    collisions = 0

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

# Greedily fill batches up to max_gb of input, never exceeding max_months. A
# single month larger than max_gb still forms its own batch rather than being
# skipped -- 2023 months run ~12 GB each.
def build_warm_batches(months: List[Path], max_gb: float, max_months: int) -> List[List[Path]]:
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

# Assign an ID to every author in the requested months that does not have one
# yet. Idempotent, but not cheap to repeat (a full re-run re-reads hundreds of
# GB), so months already recorded in the warm log are skipped. Returns the
# number of new mappings inserted.
def warm_author_map() -> int:
    start_time = time.time()
    stage = Path(input_path).name

    already_warm = warmed_months_from_log(warm_log_path, group, type_, stage)
    months = unwarmed_files(file_list, already_warm)
    skipped = len(file_list) - len(months)

    log_report(
        report_file_path,
        f"Author-map warm-up for {group} / {type_} ({stage}), {args.years}: {len(months)} month(s) to scan, "
        f"{skipped} already warmed in a prior run. The batch anonymization array opens the author map "
        f"read-only, so every author must hold an ID before it starts."
    )
    if not months:
        return 0

    total_bytes = sum(p.stat().st_size for p in months)
    batches = build_warm_batches(months, WARM_BATCH_GB, WARM_BATCH_MONTHS)
    peak_gb = max(sum(p.stat().st_size for p in b) for b in batches) / 2**30
    log_report(
        report_file_path,
        f"Scanning {total_bytes / 2**30:.1f} GB with {WARM_WORKERS} readers in {len(batches)} batch(es) of "
        f"<= {WARM_BATCH_GB:g} GB (heaviest batch {peak_gb:.1f} GB)."
    )

    conn = open_cache(USER_CACHE, synchronous="FULL")
    try:
        before = conn.execute("SELECT COUNT(*) FROM author_map").fetchone()[0]
        t = time.time()
        known = load_known_author_hashes(conn)
        log_report(
            report_file_path,
            f"Author map holds {before:,} mappings; preloaded their keys in {time.time() - t:.0f}s "
            f"({known.nbytes / 2**20:.0f} MB resident)."
        )

        grand_new = grand_collisions = grand_rows = 0
        for batch in batches:
            label = f"{month_of_file(batch[0])}..{month_of_file(batch[-1])}"
            batch_gb = sum(p.stat().st_size for p in batch) / 2**30

            authors: Set[str] = set()
            rows = 0
            t0 = time.time()
            done = 0
            with ProcessPoolExecutor(max_workers=WARM_WORKERS) as ex:
                futs = {ex.submit(scan_month_authors, str(p)): p for p in batch}
                for fut in as_completed(futs):
                    path, found, n = fut.result()
                    authors |= found
                    rows += n
                    done += 1
                    # Per-file progress to stdout only; the report keeps per-batch lines.
                    print(f"    scan {done}/{len(batch)}  {Path(path).name}  {n:,} rows  "
                          f"{len(found):,} authors  [{time.time() - t0:.0f}s]", flush=True)

            new_list, new_hashes = split_new_authors(known, authors)
            inserted, collisions = insert_new_authors(conn, set(new_list))
            if new_hashes.size:
                known = np.sort(np.concatenate([known, new_hashes]))
            # Fold the WAL back into the main DB so it cannot grow unbounded
            # across batches and so a later reader never replays a huge log.
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            # Record only after the commit and checkpoint have both landed, so a
            # crash can never mark a batch warm that is not durably in the map.
            record_warmed_months(warm_log_path, group, type_, stage, batch)

            grand_rows += rows
            grand_new += inserted
            grand_collisions += collisions
            log_report(
                report_file_path,
                f"Warmed {label} ({len(batch)} month(s), {batch_gb:.1f} GB): rows={rows:,}, "
                f"distinct authors={len(authors):,}, new IDs assigned={inserted:,}"
                + (f", id-collisions retried={collisions}" if collisions else "")
                + f", time={(time.time() - t0) / 60:.2f} minutes"
            )

        after = conn.execute("SELECT COUNT(*) FROM author_map").fetchone()[0]
        dupes = conn.execute(
            "SELECT COUNT(*) - COUNT(DISTINCT anon_id) FROM author_map").fetchone()[0]
        log_report(
            report_file_path,
            f"Finished author-map warm-up. Scanned {grand_rows:,} rows across {len(months)} month(s); "
            f"author map {before:,} -> {after:,} mappings ({grand_new:,} new"
            + (f", {grand_collisions} id-collisions retried" if grand_collisions else "")
            + f"), duplicate anon_id check: {dupes} (must be 0), Time: {(time.time() - start_time) / 60:.2f} minutes"
        )
        if dupes:
            raise RuntimeError(f"author_map holds {dupes} duplicate anon_id value(s)")
    finally:
        conn.close()

    return grand_new

### Main Anonymization Functions

# Stream one CSV to another while anonymizing the 'author' column. Returns (rows_written, new_ids_created_in_this_file).
# Resumes from the last source_row already written to the output if any.
def anonymize_one_file(
    input_file: str | Path,
    output_file: str | Path,
    db_path: str | Path,
    read_only_cache: bool = False,
) -> Tuple[int, int]:
    input_file = Path(input_file)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # determine resume position from any existing output
    #
    # Row ordinal rather than source_row (get_last_source_row, used by the
    # filter_/label_ resources): 'all' inputs are merged files interleaving a
    # comments and a submissions source_row sequence, so a max-based watermark
    # silently drops every row of the trailing stream. This also truncates a row
    # torn by a crash mid-write.
    last_processed = get_resume_position(
        output_file,
        report_file_path=report_file_path,
        file_for_log=input_file,
    )
    mode = "a" if last_processed >= 0 else "w"

    conn = open_cache(db_path, read_only=read_only_cache)
    local_cache: Dict[str, str] = {}
    rows_written = 0
    new_ids_created = 0

    try:
        with (
            open(input_file, "r", encoding="utf-8-sig", errors="ignore", newline="") as in_f,
            open(output_file, mode, encoding="utf-8", newline="") as out_f,
        ):
            reader = csv.DictReader((line.replace("\x00", "") for line in in_f))
            if reader.fieldnames is None:
                raise ValueError(f"Could not read CSV header from {input_file.name}")

            if "author" not in reader.fieldnames:
                raise ValueError(f"Input file {input_file.name} does not contain an 'author' column.")

            writer = csv.DictWriter(out_f, fieldnames=reader.fieldnames)
            if mode == "w":
                writer.writeheader()

            for row_index, row in enumerate(reader):
                # Resume: input row N maps to output row N, so skipping the
                # rows already written is exact regardless of source_row order.
                if row_index < last_processed:
                    continue

                author = author_map_key(row.get("author", ""))

                if author is not None:
                    anon_id = local_cache.get(author)
                    if anon_id is None:
                        if read_only_cache:
                            # Sharded run: the warm-up pass has pre-assigned every
                            # author and this connection physically cannot write, so a
                            # miss is an error rather than a cue to create. Checked
                            # before get_or_create_author_id so the failure names its
                            # cause instead of surfacing as SQLite's opaque
                            # "attempt to write a readonly database".
                            anon_id = lookup_author_id(conn, author)
                            if anon_id is None:
                                raise RuntimeError(
                                    f"Author {author!r} in {input_file.name} has no ID and the "
                                    f"author map is open read-only. Re-run the warm-up "
                                    f"(ORGANIZE_ANONYMIZE_WARM=1) for this month, then retry this shard."
                                )
                        else:
                            # One lookup, not three: get_or_create_author_id already
                            # does its own read-first fast path. It reports
                            # created=True only when THIS process performed the
                            # INSERT -- inferring that from a separate read-then-write
                            # pair let two parallel array tasks both observe "absent"
                            # and both count the same author as new.
                            anon_id, created = get_or_create_author_id(conn, author)
                            if created:
                                new_ids_created += 1

                        local_cache[author] = anon_id

                    row["author"] = anon_id

                writer.writerow(row)
                rows_written += 1

    finally:
        conn.close()

    return rows_written, new_ids_created

### Main execution

def organize_anonymize() -> int:
    start_time = time.time()

    files_per_job = getattr(args, "files_per_job", 1) or 1
    target_files = select_target_files(
        all_files=file_list,
        years_list=years,
        array_idx=args.array,
        files_per_job=files_per_job,
    )

    if not target_files:
        log_report(report_file_path, "No target files assigned to this run.")
        return 0

    # A shard run (--array) means sibling processes are active, so the map must
    # already be warmed and is opened read-only. A plain single-process run keeps
    # the read-write path and can still assign IDs on the fly.
    read_only_cache = args.array is not None
    log_report(
        report_file_path,
        f"Preparing to anonymize {len(target_files)} file(s) for group={group}, type={type_}"
        + (" [author map read-only: sharded run, IDs pre-assigned by the warm-up pass]" if read_only_cache else "") + "."
    )

    processed = 0
    skipped = 0
    failed = 0
    total_rows = 0
    total_new_ids = 0

    for input_file in target_files:
        output_file = output_path / input_file.name

        try:
            rows_written, new_ids_created = anonymize_one_file(
                input_file=input_file,
                output_file=output_file,
                db_path=USER_CACHE,
                read_only_cache=read_only_cache,
            )
            processed += 1
            total_rows += rows_written
            total_new_ids += new_ids_created

            log_report(
                report_file_path,
                f"Anonymized {input_file.name} -> {output_file.name}; rows={rows_written}, new_ids={new_ids_created}"
            )

        except Exception as e:
            failed += 1
            log_report(
                report_file_path,
                f"Error anonymizing {input_file.name}: {e}"
            )

    elapsed_time = (time.time() - start_time) / 60.0
    log_report(
        report_file_path,
        f"Finished anonymization. Successful: {processed}, Skipped: {skipped}, Failed: {failed}, "
        f"Rows written: {total_rows}, New IDs assigned: {total_new_ids}, Time: {elapsed_time:.2f} minutes"
    )
    return failed

if __name__ == "__main__":
    overall_start_time = time.time()

    # Warm-up mode: assign IDs, write no anonymized output, and exit. Under
    # --slurm this runs as the single job the anonymization array is chained
    # to with afterok, so a failure here (non-zero exit) holds the array back
    # instead of letting every task fail on the read-only map.
    if WARM_MODE:
        try:
            warm_author_map()
        except Exception as e:
            reraise_fatal(report_file_path, "organize_anonymize warm-up", e)
        sys.exit(0)

    try:
        n_failed = organize_anonymize()
    except Exception as e:
        reraise_fatal(report_file_path, "organize_anonymize", e)

    total_time = (time.time() - overall_start_time) / 60

    if args.array is None:
        scope_msg = f"{args.years}"
    else:
        files_per_job = getattr(args, "files_per_job", 1) or 1
        requested_months = build_requested_months(years)
        assigned = requested_months[args.array * files_per_job : args.array * files_per_job + files_per_job]
        assigned_str = ", ".join(f"{y}-{m}" for y, m in assigned) if assigned else f"array task {args.array}"
        scope_msg = f"{args.years} (task scope: {assigned_str})"

    log_report(
        report_file_path,
        f"Anonymization for {group} / {type_} for {scope_msg} finished in {total_time:.2f} minutes"
    )

    # Per-file failures are logged but were invisible to Slurm: the process fell
    # off the end at exit 0, so a task that anonymized NOTHING still reported
    # COMPLETED and would satisfy an afterok dependency. A transient "disk I/O
    # error" can leave a task that wrote 0 rows still reporting COMPLETED. The
    # partial (truncated) outputs they leave behind are safe to retry:
    # anonymize_one_file resumes via get_resume_position, which trims the torn
    # trailing row and appends. That only helps if someone knows to retry, hence
    # this exit code.
    if n_failed:
        log_report(
            report_file_path,
            f"Exiting non-zero: {n_failed} file(s) failed for {group} / {type_} ({scope_msg})."
        )
        sys.exit(1)
