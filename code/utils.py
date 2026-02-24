import argparse
import csv
import random
import math
import os
import sys
import re
import time
from datetime import datetime
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
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

## File discovery helper
def check_reqd_files(years: List[int], check_path: str | Path, type_: str) -> List[str]:
    PREFIX_MAP = {"comments": "RC", "submissions": "RS"}
    prefix = PREFIX_MAP.get(type_)
    if not prefix:
        raise ValueError(f"Invalid type_: {type_}")

    check_path = str(check_path)
    all_files = [f for f in os.listdir(check_path) if f.endswith(".csv") and f.startswith(prefix)]

    matched_files: List[str] = []
    files_by_year: Dict[str, set] = {str(y): set() for y in years}

    for f in sorted(all_files):
        for y in years:
            if str(y) in f:
                matched_files.append(os.path.join(check_path, f))

        m = re.search(r"(\d{4})-(\d{2})", f)
        if m:
            year, month = m.groups()
            if year in files_by_year:
                files_by_year[year].add(month)

    if not matched_files:
        raise FileNotFoundError(
            f"No files found in {check_path} for type_={type_} and years={years}"
        )

    expected_months = set(f"{m:02d}" for m in range(1, 13))
    for y in years:
        missing = expected_months - files_by_year.get(str(y), set())
        if missing:
            print(f"Warning: For {type_} year {y}, missing months: {sorted(missing)}")

    return matched_files

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
    include_hours: bool = True,
) -> None:
    for tok in tokenize(text):
        k = f"w:{tok}"
        counts[k] = counts.get(k, 0) + 1

    if subreddit:
        s = subreddit.strip()
        if s:
            k = f"s:{s}"
            counts[k] = counts.get(k, 0) + 1

    if include_hours:
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
    include_hours: bool = True,
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
                include_hours=include_hours,
            )
    return author_to_counts

# Raw Reddit (.zst) reading helpers for location labeling

# Yield decoded JSON objects from a .zst file (one JSON per line).
def iter_zst_json_lines(file_path: str | Path):

    file_path = str(file_path)
    with open(file_path, "rb") as fh:
        dctx = zstandard.ZstdDecompressor(max_window_size=2 ** 31)  # 2GB window cap
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
    include_hours: bool = True,
) -> Dict[str, Dict[str, int]]:
    """Wrapper returning only feature counts (without per-author seen counts)."""
    author_to_counts, _author_seen = build_author_feature_map_from_raw_zst_with_seen(
        raw_files=raw_files,
        target_authors=target_authors,
        type_=type_,
        max_items_per_author=max_items_per_author,
        include_hours=include_hours,
    )
    return author_to_counts

# Stream one or more raw .zst files and build sparse features per author.
# NOTE: Collects up to max_items_per_author items per author. Returns (author_to_counts, author_seen).
def build_author_feature_map_from_raw_zst_with_seen(
    raw_files: List[str | Path],
    target_authors: set[str],
    type_: str,
    max_items_per_author: int = 100,
    include_hours: bool = True,
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, int]]:
    author_to_counts: Dict[str, Dict[str, int]] = {}
    author_seen: Dict[str, int] = {a: 0 for a in target_authors}

    if not target_authors:
        return author_to_counts, author_seen

    remaining = len(target_authors)

    for rf in raw_files:
        for obj in iter_zst_json_lines(rf):
            author = (obj.get("author") or "").strip()
            if not author or author not in author_seen:
                continue
            if author_seen[author] >= max_items_per_author:
                continue

            # Extract fields by type
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

            add_features_for_row(
                counts,
                text=text,
                subreddit=subreddit,
                time_value=str(created_utc),
                include_hours=include_hours,
            )

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

# Initialize the SQLite cache for author->location mapping. 
# NOTE: Uses WAL journal mode for better concurrent read/write behavior.
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
                updated_at INTEGER NOT NULL
            );
            """
        )
        conn.commit()
    finally:
        conn.close()

# Fetch cached locations for a set of authors.
def cache_get_locations(db_path: str, authors: set[str]) -> Dict[str, str]:
    if not authors:
        return {}
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

# Upsert many author->location mappings.
def cache_put_locations(db_path: str, author_to_loc: Dict[str, str]) -> None:
    if not author_to_loc:
        return
    conn = sqlite3.connect(db_path, timeout=60)
    try:
        cur = conn.cursor()
        now = int(time.time())
        rows = [(a, loc, now) for a, loc in author_to_loc.items() if a and loc]
        cur.executemany(
            "INSERT OR REPLACE INTO author_location(author, location, updated_at) VALUES (?,?,?)",
            rows,
        )
        conn.commit()
    finally:
        conn.close()
