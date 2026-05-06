import argparse
import csv
import random
import math
import os
import sys
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from collections import Counter
from pathlib import Path
from typing import List, Dict, Literal, Tuple, Optional, Any
import zstandard
import io
import json
import sqlite3

## Shared constants used across the codebase

# The list of social groups. Marginalized groups always listed first.
groups = {
    "sexuality": ["gay", "straight"],
    "age": ["old", "young"],
    "weight": ["fat", "thin"],
    "ability": ["disabled", "abled"],
    "race": ["black", "white"],
    "skin_tone": ["dark", "light"],
}

# Information stored for each comment in ISAAC output files
headers = ["id", "parent id", "text", "author", "time", "subreddit", "score", "matched patterns"]

# default resource order
default_resource = ["filtered_keywords","filtered_language","filtered_relevance","filtered_keywords_adv","labeled_moralization","labeled_sentiment","labeled_generalization","labeled_emotion","labeled_location"]

# Basic parsing / validation helpers (used by multiple scripts)

# confirm that input year range is valid
def validate_years(years_str: str, parser: argparse.ArgumentParser) -> None:
    """Validate either 'YYYY' or 'YYYY-YYYY' with bounds 2007..2023."""
    match = re.fullmatch(r"(\d{4})(?:-(\d{4}))?", years_str)
    if not match:
        parser.error("--years must be a 4-digit year or a range like 2010-2015.")

    start = int(match.group(1))
    end = int(match.group(2)) if match.group(2) else start

    if not (2007 <= start <= 2023 and 2007 <= end <= 2023):
        parser.error("Years must be between 2007 and 2023.")
    if start > end:
        parser.error("Start year must be less than or equal to end year.")

# process input year range
def parse_range(value: str) -> List[int]:
    """Parse 'YYYY' or 'YYYY-YYYY' into a list of years with bounds 2007..2023."""
    try:
        if "-" in value:
            start, end = map(int, value.split("-", 1))
            if start > end:
                raise argparse.ArgumentTypeError(f"Invalid range '{value}': start must be ≤ end.")
        else:
            start = end = int(value)

        if start < 2007:
            raise argparse.ArgumentTypeError(f"Invalid value '{value}': years must be ≥ 2007.")
        if end > 2023:
            raise argparse.ArgumentTypeError(f"Invalid value '{value}': years must be ≤ 2023.")

        return list(range(start, end + 1))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid value '{value}': must be an integer or a range (e.g., 2007 or 2008-2010)."
        )

# Calculate SLURM array span from year range string.
def array_span_from_years(years_str: str) -> int:
    if "-" in years_str:
        start, end = years_str.split("-", 1)
        start_y, end_y = int(start), int(end)
        if end_y < start_y:
            start_y, end_y = end_y, start_y
        return (end_y - start_y + 1) * 12
    else:
        return 12

# Load newline-delimited keywords, lowercased; skip blanks.
def load_terms(file_path: str) -> List[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.lower().rstrip("\r\n") for line in f if line.strip()]

## Logging helpers (used by multiple scripts)

def log_report(report_file_path: Optional[str] = None, message: Optional[str] = None) -> None:
    if message is None:
        return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if report_file_path:
        os.makedirs(os.path.dirname(report_file_path), exist_ok=True)
        with open(report_file_path, "a", encoding="utf-8", newline="") as report_file:
            writer = csv.writer(report_file)
            writer.writerow([timestamp, message])
    print(f"{timestamp} - {message}")
    sys.stdout.flush()

def log_error(
    function_name: str,
    file: str,
    line_number: int,
    line_content: str,
    error: Exception,
    report_file_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> None:
    try:
        error_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        resource_identifier = os.path.basename(file)
        error_filename = f"error_{resource_identifier}_{line_number}_{error_time}.txt"
        if output_path:
            os.makedirs(output_path, exist_ok=True)
            error_filepath = os.path.join(output_path, error_filename)
            with open(error_filepath, "w", encoding="utf-8") as ef:
                ef.write(f"Error in {function_name} at line {line_number}: {error}\n")
                ef.write(f"Line content: {line_content}\n")
        if report_file_path:
            log_report(report_file_path, f"Logged error in {error_filename}")
    except Exception:
        pass

def f1_calculator(labels,predictions):

    metrics = {i:0 for i in ['tp','tn','fp','fn']}

    for idx,prediction in enumerate(predictions):
        
            if labels[idx] == 0:
                if prediction == 0:
                    metrics['tn'] += 1
                elif prediction == 1:
                    metrics['fp'] += 1
                else:
                    raise Exception
            elif labels[idx] == 1:
                if prediction == 0:
                    metrics['fn'] += 1
                elif prediction == 1:
                    metrics['tp'] += 1
                else:
                    raise Exception

    precision = float(metrics['tp']) / float(metrics['tp'] + metrics['fp'])
    recall = float(metrics['tp']) / float(metrics['tp'] + metrics['fn'])
    F_1 = 2 * float(precision * recall) / float(precision + recall)

    return precision, recall, F_1

## File discovery helpers

FolderType = Literal["comments", "submissions"]
def detect_reddit_folder_type(folder: str | Path) -> FolderType:
    folder = Path(folder)

    if not folder.is_dir():
        raise ValueError(f"Not a directory: {folder}")

    csv_files = [p.name for p in folder.iterdir() if p.is_file() and p.suffix == ".csv"]

    has_rc = any(name.startswith("RC") for name in csv_files)
    has_rs = any(name.startswith("RS") for name in csv_files)

    if has_rc and has_rs:
        raise ValueError(
            f"Folder {folder} contains both RC and RS CSV files. Provide an unambiguous input folder."
        )
    if has_rc:
        return "comments"
    if has_rs:
        return "submissions"

    raise FileNotFoundError(
        f"Folder {folder} contains no Reddit CSV files with RC or RS prefixes."
    )

def check_reqd_files(years: List[int], check_path: str | Path, type_: str) -> List[str]:
    PREFIX_MAP = {
        "comments": "RC",
        "submissions": "RS",
        "all": "ALL",
    }

    prefix = PREFIX_MAP.get(type_)
    if not prefix:
        raise ValueError(f"Invalid type_: {type_}")

    check_path = Path(check_path)
    if not check_path.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {check_path}")

    all_files = sorted(
        p for p in check_path.iterdir()
        if p.is_file() and p.suffix == ".csv" and p.name.startswith(prefix)
    )

    matched_files: List[str] = []
    files_by_year: Dict[str, set] = {str(y): set() for y in years}

    for p in all_files:
        m = re.search(r"(\d{4})-(\d{2})", p.name)
        if not m:
            continue

        year, month = m.groups()

        if year in files_by_year:
            files_by_year[year].add(month)
            matched_files.append(str(p))

    if not matched_files:
        raise FileNotFoundError(
            f"No files found in {check_path} for type_={type_} and years={years}"
        )

    expected_months = {f"{m:02d}" for m in range(1, 13)}
    for y in years:
        missing = expected_months - files_by_year.get(str(y), set())
        if missing:
            print(f"Warning: For {type_} year {y}, missing months: {sorted(missing)}")

    return matched_files

def find_latest_resource_dir(base_dir: str | Path, default_resource: List[str]) -> Path:
    base_dir = Path(base_dir)
    if not base_dir.is_dir():
        raise FileNotFoundError(f"Base curated directory does not exist: {base_dir}")

    curated_folders = {p.name for p in base_dir.iterdir() if p.is_dir()}

    for resource in reversed(default_resource):
        if resource in curated_folders:
            return base_dir / resource

    raise ValueError(
        f"No matching resource found in {base_dir}. "
        f"Expected one of: {', '.join(default_resource)}"
    )

def validate_resource_dir(path: str | Path, default_resource: List[str]) -> Path:
    path = Path(path)
    if not path.is_dir():
        raise ValueError(f"Input path is not a directory: {path}")

    if path.name not in default_resource:
        raise ValueError(
            f"{path.name} does not correspond to a proper curated dataset. "
            f"Choose from: {', '.join(default_resource)}"
        )

    return path

# for location model word features
def resolve_word_feature_src(type_arg: str) -> str:
    src = (type_arg or WORD_FEATURE_SRC).strip().lower()
    if src not in {"comments", "submissions", "all"}:
        raise ValueError("type / WORD_FEATURE_SRC must be one of: comments, submissions, all")
    return src

## Dataset splitting utilities

# splits data into train/test with given proportion
def dataset_split(users: List[Any], labels: List[Any], proportion: float, seed: Optional[int] = None):
    if seed is not None:
        random.seed(seed)

    n = len(users)
    k = math.floor(proportion * n)
    training_id = set(random.sample(range(n), k))
    test_id = [i for i in range(n) if i not in training_id]

    training_users, training_labels, test_users, test_labels = [], [], [], []
    for idx, u in enumerate(users):
        if idx in training_id:
            training_users.append(u)
            training_labels.append(labels[idx])
        else:
            test_users.append(u)
            test_labels.append(labels[idx])

    return training_users, test_users, training_labels, test_labels

# Write data splits to file
def split_dataset_to_file(file: str, items: List[Any]) -> None:
    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, "w", encoding="utf-8", errors="ignore", newline="") as f:
        if ("users" in file) or ("text" in file):
            writer = csv.writer(f)
            for i in items:
                writer.writerow([i])
        elif "label" in file:
            for i in items:
                print(i, file=f)
        else:
            # fallback: csv
            writer = csv.writer(f)
            for i in items:
                writer.writerow([i])

# Read data splits from file
def split_dataset_from_file(file: str, label_cast: Optional[type] = None) -> List[Any]:
    items: List[Any] = []
    with open(file, "r", encoding="utf-8", errors="ignore") as f:
        if ("users" in file) or ("text" in file):
            reader = csv.reader(f)
            for row in reader:
                if row:
                    items.append(row[0])
        elif "label" in file:
            for line in f:
                v = line.strip()
                if label_cast is not None:
                    try:
                        v = label_cast(v)
                    except Exception:
                        pass
                items.append(v)
        else:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    items.append(row[0])
    return items

# Create or load an 80/10/10 split and make sure it persists.
def prepare_splits(users: List[Any], labels: List[Any], split_dir: str, description: str = ""):
    os.makedirs(split_dir, exist_ok=True)

    split_data = ["training", "validation", "test"]
    file_list: List[str] = []
    for cat in split_data:
        file_list.append(os.path.join(split_dir, f"users_{cat}.csv"))
        file_list.append(os.path.join(split_dir, f"label_{cat}.txt"))

    missing_file = any(not os.path.exists(f) for f in file_list)

    if missing_file:
        print(f"Creating {description} training, validation and test sets (80/10/10 split)")
        train_users, valid_users_init, train_labels, valid_labels_init = dataset_split(
            users, labels, proportion=0.8
        )
        valid_users, test_users, valid_labels, test_labels = dataset_split(
            valid_users_init, valid_labels_init, proportion=0.5
        )

        split_dataset_to_file(file_list[0], train_users)
        split_dataset_to_file(file_list[1], train_labels)
        split_dataset_to_file(file_list[2], valid_users)
        split_dataset_to_file(file_list[3], valid_labels)
        split_dataset_to_file(file_list[4], test_users)
        split_dataset_to_file(file_list[5], test_labels)
    else:
        print(f"Loading predetermined {description} training, validation and test sets (80/10/10 split)")
        train_users = split_dataset_from_file(file_list[0])
        train_labels = split_dataset_from_file(file_list[1])
        valid_users = split_dataset_from_file(file_list[2])
        valid_labels = split_dataset_from_file(file_list[3])
        test_users = split_dataset_from_file(file_list[4])
        test_labels = split_dataset_from_file(file_list[5])

    return train_users, train_labels, valid_users, valid_labels, test_users, test_labels

# summarize information about the data split
def summarize_split(name: str, users: List[Any], labels: List[Any]) -> None:
    print(f"Number of {name} documents: {len(users)}")
    print(f"Number of instances for each label in {name} data: {Counter(labels)}")

## Location labeling helpers

_token_re = re.compile(r"[a-z0-9']+")

# Simple tokenizer aligned with the location model's word feature style.
def tokenize(text: str) -> List[str]:
    return _token_re.findall((text or "").lower())

# Parse a timestamp string to hour [0..23]. Supports ISO-like and unix seconds.
def parse_time_to_hour(time_str: str) -> Optional[int]:
    if not time_str:
        return None
    s = str(time_str).strip()
    # unix seconds?
    if s.isdigit():
        try:
            return datetime.fromtimestamp(int(s)).hour
        except Exception:
            return None
    # common formats: 'YYYY-mm-dd HH:MM:SS' or ISO
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).hour
        except Exception:
            pass
    # try fromisoformat
    try:
        return datetime.fromisoformat(s).hour
    except Exception:
        return None

# update a user's sparse counts in-place from a single row
def add_features_for_row(
    counts: Dict[str, int],
    text: str,
    subreddit: str,
    time_value: str,
) -> None:
    for tok in tokenize(text):
        k = f"w:{tok}"
        counts[k] = counts.get(k, 0) + 1

    if subreddit:
        s = subreddit.strip()
        if s:
            k = f"s:{s}"
            counts[k] = counts.get(k, 0) + 1

    hr = parse_time_to_hour(time_value)
    if hr is not None:
        k = f"h:{hr:02d}"
        counts[k] = counts.get(k, 0) + 1

# First pass over an input CSV: aggregate sparse features per author.
def build_author_feature_map_from_csv(
    file_path: str | Path,
    author_col: str = "author",
    text_col: str = "text",
    subreddit_col: str = "subreddit",
    time_col: str = "time",
) -> Dict[str, Dict[str, int]]:
    author_to_counts: Dict[str, Dict[str, int]] = {}
    with open(file_path, "r", encoding="utf-8-sig", errors="ignore") as f:
        reader = csv.DictReader((line.replace("\x00", "") for line in f))
        if reader.fieldnames is None:
            return author_to_counts

        for row in reader:
            author = (row.get(author_col) or "").strip()
            if not author or author == "[deleted]":
                continue
            counts = author_to_counts.get(author)
            if counts is None:
                counts = {}
                author_to_counts[author] = counts
            add_features_for_row(
                counts,
                text=row.get(text_col, ""),
                subreddit=row.get(subreddit_col, ""),
                time_value=row.get(time_col, ""),
            )
    return author_to_counts

# Raw Reddit (.zst) reading helpers for location labeling

# Yield decoded JSON objects from a .zst file (one JSON per line).
def iter_zst_json_lines(file_path: str | Path):

    file_path = str(file_path)
    with open(file_path, "rb") as fh:
        dctx = zstandard.ZstdDecompressor(max_window_size=2 ** 31)
        stream_reader = dctx.stream_reader(fh, read_across_frames=True)
        text_stream = io.TextIOWrapper(stream_reader, encoding="utf-8")
        for line in text_stream:
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

# Stream one or more raw .zst month files and build sparse features per author.
# NOTE: Only collects up to max_items_per_author posts/comments per author to cap work. For submissions, text is title + selftext.
# NOTE: Returns author -> counts dict with keys w:/s:/h:

def build_author_feature_map_from_raw_zst(
    raw_files: List[str | Path],
    target_authors: set[str],
    type_: str,
    max_items_per_author: int = 100,
) -> Dict[str, Dict[str, int]]:
    """Wrapper returning only feature counts (without per-author seen counts)."""
    author_to_counts, _author_seen = build_author_feature_map_from_raw_zst_with_seen(
        raw_files=raw_files,
        target_authors=target_authors,
        type_=type_,
        max_items_per_author=max_items_per_author,
    )
    return author_to_counts

# Process one contiguous chunk of raw .zst files, returning partial (counts, seen) dicts.
# Used by the parallel scan path; no early-exit since chunks run concurrently.
def _scan_raw_file_chunk(
    file_chunk: List[str],
    target_authors: set,
    type_: str,
    max_items_per_author: int,
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, int]]:
    local_counts: Dict[str, Dict[str, int]] = {}
    local_seen: Dict[str, int] = {}
    for rf in file_chunk:
        for obj in iter_zst_json_lines(rf):
            author = (obj.get("author") or "").strip()
            if not author or author not in target_authors:
                continue
            if local_seen.get(author, 0) >= max_items_per_author:
                continue
            if type_ == "comments":
                text = (obj.get("body") or "")
                subreddit = (obj.get("subreddit") or "")
            else:
                title = (obj.get("title") or "")
                body = (obj.get("selftext") or "")
                text = (title + "\n" + body).strip()
                subreddit = (obj.get("subreddit") or "")
            created_utc = obj.get("created_utc", "")
            counts = local_counts.get(author)
            if counts is None:
                counts = {}
                local_counts[author] = counts
            add_features_for_row(counts, text=text, subreddit=subreddit, time_value=str(created_utc))
            local_seen[author] = local_seen.get(author, 0) + 1
    return local_counts, local_seen


# Stream one or more raw .zst files and build sparse features per author.
# NOTE: Collects up to max_items_per_author items per author. Returns (author_to_counts, author_seen).
# NOTE: n_scan_workers > 1 splits files across threads (zstd decompression releases the GIL).
#       Use only in single-process contexts (SLURM array tasks); not for multi-process local runs.
def build_author_feature_map_from_raw_zst_with_seen(
    raw_files: List[str | Path],
    target_authors: set[str],
    type_: str,
    max_items_per_author: int = 100,
    n_scan_workers: int = 1,
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, int]]:
    author_to_counts: Dict[str, Dict[str, int]] = {}
    author_seen: Dict[str, int] = {a: 0 for a in target_authors}

    if not target_authors:
        return author_to_counts, author_seen

    raw_files = [str(f) for f in raw_files]

    if n_scan_workers > 1 and len(raw_files) > 1:
        n_workers = min(n_scan_workers, len(raw_files))
        chunks = [raw_files[i::n_workers] for i in range(n_workers)]

        def scan_chunk(chunk):
            return _scan_raw_file_chunk(chunk, target_authors, type_, max_items_per_author)

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(scan_chunk, chunks))

        for local_counts, local_seen in results:
            for author, counts in local_counts.items():
                merged = author_to_counts.get(author)
                if merged is None:
                    author_to_counts[author] = dict(counts)
                else:
                    for k, v in counts.items():
                        merged[k] = merged.get(k, 0) + v
            for author, seen in local_seen.items():
                author_seen[author] = min(author_seen.get(author, 0) + seen, max_items_per_author)

        return author_to_counts, author_seen

    # Sequential path with early-exit (used locally and in multi-process mode)
    remaining = len(target_authors)
    for rf in raw_files:
        for obj in iter_zst_json_lines(rf):
            author = (obj.get("author") or "").strip()
            if not author or author not in author_seen:
                continue
            if author_seen[author] >= max_items_per_author:
                continue
            if type_ == "comments":
                text = (obj.get("body") or "")
                subreddit = (obj.get("subreddit") or "")
            else:
                title = (obj.get("title") or "")
                body = (obj.get("selftext") or "")
                text = (title + "\n" + body).strip()
                subreddit = (obj.get("subreddit") or "")
            created_utc = obj.get("created_utc", "")
            counts = author_to_counts.get(author)
            if counts is None:
                counts = {}
                author_to_counts[author] = counts
            add_features_for_row(counts, text=text, subreddit=subreddit, time_value=str(created_utc))
            author_seen[author] += 1
            if author_seen[author] == max_items_per_author:
                remaining -= 1
                if remaining <= 0:
                    return author_to_counts, author_seen

    return author_to_counts, author_seen

# Find raw .zst files for a given year-month. Returns list of full paths.
def find_raw_month_files(raw_dir: str | Path, type_: str, year: int, month: str) -> List[str]:
    raw_dir = str(raw_dir)
    prefix = "RC" if type_ == "comments" else "RS"
    ym = f"{year}-{month}"
    out = []
    for fn in os.listdir(raw_dir):
        if not fn.endswith(".zst"):
            continue
        # common patterns: RC_YYYY-MM.zst, RC_YYYY-MM-*.zst, etc.
        if (prefix in fn) and (ym in fn):
            out.append(os.path.join(raw_dir, fn))
    return sorted(out)

## Persistent author -> location cache (SQLite)

# Run a no-arg callable, retrying on sqlite3.OperationalError 'database is
# locked' with exponential backoff and small jitter. SQLite's busy_timeout
# already handles most lock contention, but startup bursts of many array tasks
# briefly hammer the DB during table-touching reads/writes; this gives those
# operations a few extra chances before they surface as a fatal error. Returns
# the callable's result; non-lock OperationalErrors and other exceptions
# propagate immediately.
def _sqlite_retry_on_locked(operation, *, max_attempts: int = 8, base_delay: float = 0.5, max_delay: float = 8.0):
    last_err: Optional[sqlite3.OperationalError] = None
    for attempt in range(max_attempts):
        try:
            return operation()
        except sqlite3.OperationalError as e:
            if "locked" not in str(e).lower():
                raise
            last_err = e
            delay = min(max_delay, base_delay * (2 ** attempt)) + random.uniform(0.0, 0.5)
            time.sleep(delay)
    if last_err is not None:
        raise last_err

# Initialize the SQLite cache for author->location mapping.
# NOTE: Uses WAL journal mode for better concurrent read/write behavior.
# NOTE: location_prob is stored alongside location to support confidence-aware
#       overwrite in cache_put_locations (only replace cached rows when the new
#       prob beats the cached prob).
def init_location_cache(db_path: str) -> None:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=60)
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS author_location (
                author TEXT PRIMARY KEY,
                location TEXT NOT NULL,
                location_prob REAL,
                updated_at INTEGER NOT NULL
            );
            """
        )
        conn.commit()
    finally:
        conn.close()

# Initialize the detail table (per-author top/contender labels with probs and tier).
# NOTE: Pre-create alongside init_location_cache from a single process before launching
# parallel Slurm array tasks to avoid "database is locked" races on first WAL setup.
def init_location_detail_cache(db_path: str) -> None:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=60)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS author_location_detail (
                author TEXT PRIMARY KEY,
                location TEXT NOT NULL,
                location_prob REAL,
                contender_location TEXT,
                contender_location_prob REAL,
                top_location TEXT,
                top_location_prob REAL,
                top_contender_location TEXT,
                top_contender_location_prob REAL,
                tier TEXT,
                seen_count INTEGER,
                updated_at INTEGER NOT NULL
            );
            """
        )
        conn.commit()
    finally:
        conn.close()

# Fetch cached locations for a set of authors. Wrapped in a retry loop because
# concurrent Slurm array tasks can briefly contend on schema-touching writes
# at startup, occasionally surfacing 'database is locked' on reads despite WAL.
def cache_get_locations(db_path: str, authors: set[str]) -> Dict[str, str]:
    if not authors:
        return {}
    def _do() -> Dict[str, str]:
        conn = sqlite3.connect(db_path, timeout=60)
        try:
            cur = conn.cursor()
            out: Dict[str, str] = {}
            # sqlite has a variable limit; chunk in 900s to be safe
            author_list = list(authors)
            for i in range(0, len(author_list), 900):
                chunk = author_list[i:i+900]
                qmarks = ",".join(["?"] * len(chunk))
                cur.execute(f"SELECT author, location FROM author_location WHERE author IN ({qmarks})", chunk)
                for a, loc in cur.fetchall():
                    out[a] = loc
            return out
        finally:
            conn.close()
    return _sqlite_retry_on_locked(_do)

# Upsert many author->location mappings with confidence-aware overwrite.
# Each detail dict must contain at least 'location' (str) and 'location_prob'
# (float | None). Cached rows are atomically overwritten only when the new
# location_prob is strictly greater than the cached one (or the cached one is
# NULL); a new prob of None never overwrites an existing labeled row.
def cache_put_locations(db_path: str, details_by_author: Dict[str, Dict[str, Any]]) -> None:
    if not details_by_author:
        return
    conn = sqlite3.connect(db_path, timeout=60)
    try:
        cur = conn.cursor()
        now = int(time.time())
        rows = []
        for author, d in details_by_author.items():
            if not author:
                continue
            location = d.get("location")
            if not location:
                continue
            rows.append((author, str(location), d.get("location_prob"), now))
        if not rows:
            return
        cur.executemany(
            """
            INSERT INTO author_location(author, location, location_prob, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(author) DO UPDATE SET
                location = excluded.location,
                location_prob = excluded.location_prob,
                updated_at = excluded.updated_at
            WHERE excluded.location_prob IS NOT NULL
              AND (author_location.location_prob IS NULL
                   OR excluded.location_prob > author_location.location_prob)
            """,
            rows,
        )
        conn.commit()
    finally:
        conn.close()