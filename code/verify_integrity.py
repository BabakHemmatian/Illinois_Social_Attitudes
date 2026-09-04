### Imports

import csv
csv.field_size_limit(2**31 - 1)  # match the pipeline's limit; some text fields are huge
import json
import os
import re
import sqlite3
import sys
import time
from calendar import monthrange
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# import functions and objects
from cli import get_args, PROJECT_ROOT, DATA_DIR
from utils import (
    MONTH_RE,
    default_resource,
    headers,
    log_report,
    month_of_file,
    parse_range,
    reraise_fatal,
)

### Argument Handling

# Extract and transform CLI arguments
args = get_args()
group = args.group
type_ = args.type
quick = bool(getattr(args, "quick", False))

if type_ not in {"comments", "submissions", "all"}:
    raise ValueError(f"Unsupported 'type' argument: {type_}")

# --years is optional here: with no years every monthly file present in the
# target directory is verified. Under Slurm, --years is what turns the run into
# a month-keyed array (one task per month, as for the organize_ resources).
years: Optional[List[int]] = None
if args.years:
    years = parse_range(args.years)
    if isinstance(years, int):
        years = [years]

FILE_PREFIX = {"comments": "RC", "submissions": "RS", "all": "ALL"}[type_]

### Path Handling

# prepare the report file. The human-readable report follows the shared
# report_*.csv convention; a JSON-lines sidecar keeps the full per-file
# measurements (row counts, timestamps, anon-ID tallies) for later inspection.
report_file_path = PROJECT_ROOT / "report_verify_integrity.csv"
details_file_path = PROJECT_ROOT / "report_verify_integrity.jsonl"

# Set/survey the input folder: the directory whose monthly CSVs are verified.
# Stage directories may be plain resource outputs or their organize_anonymize
# '_anon' siblings; the checks applied depend on which one this is (see below).
verifiable_dirs = [f"{r}_anon" for r in default_resource] + list(default_resource)

data_base = DATA_DIR / "data_reddit_curated" / group / type_  # default

def find_latest_verifiable_dir(base_dir: Path) -> Path:
    # Most advanced stage first, preferring its anonymized sibling: once a
    # stage has been anonymized its pre-anonymization inputs are often deleted
    # to reclaim storage, so the '_anon' directory is the one that still exists.
    if not base_dir.is_dir():
        raise FileNotFoundError(f"Base curated directory does not exist: {base_dir}")
    present = {p.name for p in base_dir.iterdir() if p.is_dir()}
    for resource in reversed(default_resource):
        for candidate in (f"{resource}_anon", resource):
            if candidate in present:
                return base_dir / candidate
    raise ValueError(
        f"No verifiable curated dataset found in {base_dir}. "
        f"Expected one of: {', '.join(verifiable_dirs)}"
    )

if not args.input:
    log_report(
        report_file_path,
        f"No custom input path provided. Finding the most advanced curated dataset of type '{type_}' for {group} based on default pathing and resource order..."
    )
    input_path = find_latest_verifiable_dir(data_base)
    log_report(
        report_file_path,
        f"{input_path.name} identified as the most advanced curated dataset for {group} entries of type '{type_}'."
    )
else:
    input_path = Path(args.input)
    if not input_path.is_dir():
        raise ValueError(f"Input path is not a directory: {input_path}")
    if input_path.name not in verifiable_dirs:
        raise ValueError(
            f"{input_path.name} does not correspond to a curated dataset. "
            f"Choose from: {', '.join(verifiable_dirs)}"
        )

# Which checks apply follows from the directory being verified:
#   anon  -- an organize_anonymize output: row counts must equal the source
#            stage's, and every anonymized author ID must exist in the map.
#   types -- an organize_types output ('all'): row and per-type counts must
#            equal the comments plus submissions inputs of the same stage.
#   plain -- a filter_/label_ output: structural checks only.
if input_path.name.endswith("_anon"):
    check_kind = "anon"
    stage = input_path.name[: -len("_anon")]
elif type_ == "all":
    check_kind = "types"
    stage = input_path.name
else:
    check_kind = "plain"
    stage = input_path.name

# Reconciliation counterparts are located by the canonical curated layout
# relative to the verified directory, data_reddit_curated/<group>/<type>/<stage>:
# the pre-anonymization source is the sibling <stage> directory, and a merged
# 'all' directory's inputs are ../../comments/<stage> and
# ../../submissions/<stage>. A custom --input therefore works as long as its
# surroundings mirror that layout. A counterpart that is missing (e.g. deleted
# after being consumed) only skips that reconciliation, and the report says so.
anon_source_dir = input_path.parent / stage
types_comments_dir = input_path.parent.parent / "comments" / stage
types_submissions_dir = input_path.parent.parent / "submissions" / stage
# NOTE: opened read-only; the map is never modified by verification.
USER_CACHE = DATA_DIR / "user_map.sqlite3"

# Every monthly file of the requested type present in the directory. Months are
# matched by the YYYY-MM in their names, never by position, so gaps (months
# already consumed and deleted downstream) cannot shift an array task's month.
file_list = sorted(
    (p for p in input_path.iterdir()
     if p.is_file() and p.suffix == ".csv" and p.name.startswith(FILE_PREFIX) and MONTH_RE.search(p.name)),
    key=lambda p: p.name,
)
if years is not None:
    year_set = {str(y) for y in years}
    file_list = [p for p in file_list if month_of_file(p)[:4] in year_set]

### File Checks

# Directory metadata (sizes, mtimes) is unreliable after a storage fault, so no
# verdict here trusts stat() -- every verdict comes from reading bytes. Two
# failure modes this is built to catch: organize_types resumes by checking only
# that ALL_YYYY-MM.csv exists, so a file whose rename committed but whose tail
# was lost is treated as complete forever; and organize_anonymize resumes past a
# torn trailing row, so the damage stays embedded in the file.

# Mirrors organize_types.TIME_FORMATS
TIME_FORMATS = ("%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M")

# Placeholders organize_anonymize leaves untouched (organize_anonymize.should_preserve_author)
PRESERVED_AUTHORS = {"", "[deleted]", "[removed]"}

EXPECTED_HEADER_PREFIX = ",".join(headers[:5]).encode()  # id,parent id,text,author,time

TAIL_BYTES = 8 << 20  # how much of the tail --quick inspects
VERIFY_WORKERS = 2    # parallel readers within one process; keep low, parallel reads thrash a spinning disk

def parse_time(value: str) -> Optional[datetime]:
    value = (value or "").strip()
    for fmt in TIME_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None

# Quick mode: head + tail only. Truncation is the dominant failure mode from a
# lost delayed write, and it is visible without reading the whole file: a
# time-ordered month file that stops early never reaches the end of its month.
def quick_check(path: Path) -> Dict:
    out: Dict = {"file": path.name, "mode": "quick", "size": path.stat().st_size}
    year, month = (int(x) for x in month_of_file(path).split("-"))
    month_end = datetime(year, month, monthrange(year, month)[1], 23, 59, 59)
    month_start = datetime(year, month, 1)

    with open(path, "rb") as f:
        head = f.read(1 << 20)
        size = f.seek(0, os.SEEK_END)
        f.seek(max(0, size - TAIL_BYTES))
        tail = f.read()

    out["nul_in_head"] = b"\x00" in head
    out["nul_in_tail"] = b"\x00" in tail
    out["ends_with_newline"] = tail.endswith(b"\n") if tail else False
    out["header_ok"] = head.startswith(EXPECTED_HEADER_PREFIX)

    # The file is time-ordered, so the newest timestamp anywhere in the tail is
    # effectively the file's last record time.
    stamps = [
        parse_time(s.decode("ascii", "ignore"))
        for s in re.findall(rb"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", tail)
    ]
    stamps = [s for s in stamps if s and month_start <= s <= month_end]

    if not stamps:
        out["last_time"] = None
        out["verdict"] = "SUSPECT"
        out["reason"] = "no in-month timestamp found in tail"
        return out

    last = max(stamps)
    out["last_time"] = last.isoformat()
    # Reddit volume is high enough that a healthy month file ends within hours
    # of the month boundary; a day of slack is generous.
    gap_hours = (month_end - last).total_seconds() / 3600.0
    out["hours_short_of_month_end"] = round(gap_hours, 2)

    problems = []
    if out["nul_in_head"] or out["nul_in_tail"]:
        problems.append("NUL bytes present")
    if not out["ends_with_newline"]:
        problems.append("does not end with newline")
    if not out["header_ok"]:
        problems.append("unexpected header")
    if gap_hours > 24:
        problems.append(f"ends {gap_hours:.1f}h before month end (truncated?)")

    out["verdict"] = "SUSPECT" if problems else "OK"
    out["reason"] = "; ".join(problems)
    return out

# Feeds csv.reader while recording corruption signals the parser would hide.
# Reads in binary and decodes per line rather than using text mode: on Windows,
# TextIOWrapper.readline() has to grow an internal buffer until it finds a
# newline, and on multi-GB files that can fail outright with OSError EINVAL.
# Splitting on b"\n" is safe for UTF-8 because no multi-byte sequence contains
# 0x0A, so a line never cuts a character in half.
def _lines(path: Path, state: Dict):
    last = b""
    with open(path, "rb") as f:
        for raw in f:
            if b"\x00" in raw:
                state["nul_lines"] += 1
            if len(raw) > state["max_line_bytes"]:
                state["max_line_bytes"] = len(raw)
            text = raw.decode("utf-8", "replace")
            if "�" in text:
                state["replacement_chars"] += 1
            last = raw
            yield text
    state["ends_with_newline"] = last.endswith((b"\n", b"\r")) if last else False

def _new_state() -> Dict:
    return {"nul_lines": 0, "replacement_chars": 0, "ends_with_newline": None, "max_line_bytes": 0}

# Full mode: complete parse of every row.
def full_check(path: Path, kind: str, db_path: Optional[Path] = None) -> Dict:
    state = _new_state()
    out: Dict = {"file": path.name, "mode": "full", "size": path.stat().st_size}

    rows = 0
    bad_width = 0
    unparseable_time = 0
    out_of_order = 0
    type_counts: Dict[str, int] = {}
    authors_bad = 0
    anon_ids = set()
    prev_time = None
    parse_error = None
    header: List[str] = []

    try:
        reader = csv.reader(_lines(path, state))
        header = next(reader, []) or []
        idx = {name: i for i, name in enumerate(header)}
        t_i = idx.get("time")
        ty_i = idx.get("type")
        a_i = idx.get("author")

        for row in reader:
            rows += 1
            if len(row) != len(header):
                bad_width += 1
                continue

            if t_i is not None:
                ts = parse_time(row[t_i])
                if ts is None:
                    unparseable_time += 1
                else:
                    if prev_time is not None and ts < prev_time:
                        out_of_order += 1
                    prev_time = ts

            if ty_i is not None:
                type_counts[row[ty_i]] = type_counts.get(row[ty_i], 0) + 1

            # Only anonymized outputs should carry 12-digit IDs; the pre-anon
            # sources legitimately hold real usernames.
            if a_i is not None and kind == "anon":
                a = (row[a_i] or "").strip()
                if a not in PRESERVED_AUTHORS:
                    if len(a) == 12 and a.isdigit():
                        anon_ids.add(int(a))
                    else:
                        authors_bad += 1

    except csv.Error as e:
        parse_error = str(e)
    except Exception as e:  # decode/IO failures mid-file
        parse_error = f"{type(e).__name__}: {e}"

    out.update(
        rows=rows,
        header_len=len(header),
        nul_lines=state["nul_lines"],
        max_line_bytes=state["max_line_bytes"],
        replacement_chars=state["replacement_chars"],
        ends_with_newline=state["ends_with_newline"],
        bad_width_rows=bad_width,
        unparseable_time=unparseable_time,
        out_of_order_times=out_of_order,
        type_counts=type_counts,
        bad_author_values=authors_bad,
        distinct_anon_ids=len(anon_ids),
        parse_error=parse_error,
    )

    problems = []
    if parse_error:
        problems.append(f"CSV parse failed: {parse_error}")
    if state["nul_lines"]:
        problems.append(f"{state['nul_lines']} lines contain NUL bytes")
    if not state["ends_with_newline"]:
        problems.append("does not end with newline")
    if bad_width:
        problems.append(f"{bad_width} rows with wrong column count")
    if unparseable_time:
        problems.append(f"{unparseable_time} unparseable timestamps")
    if authors_bad:
        problems.append(f"{authors_bad} malformed author values")
    # Out-of-order timestamps are baseline noise: the raw dumps are not perfectly
    # sorted and organize_types only preserves whatever order its inputs had.
    # Reported in the details, but not treated as damage on its own.

    # The outputs store anon_id only, never the original author, so a mapping
    # that vanished from the map cannot be reconstructed from the CSVs. Any ID
    # present in a file but absent from author_map marks an orphaned author
    # whose months must be re-anonymized from the pre-anonymization inputs.
    # Checked here, per file inside the worker: hoisting every ID of a
    # 176 GB corpus into the parent process exhausts memory.
    if kind == "anon" and db_path and anon_ids:
        try:
            orphans = orphan_anon_ids(Path(db_path), anon_ids)
            out["orphan_count"] = len(orphans)
            out["orphan_sample"] = [str(o) for o in orphans[:10]]
            if orphans:
                problems.append(f"{len(orphans)} anon IDs absent from author_map")
        except Exception as e:
            out["orphan_count"] = None
            out["xref_error"] = f"{type(e).__name__}: {e}"

    out["verdict"] = "SUSPECT" if problems else "OK"
    out["reason"] = "; ".join(problems)
    return out

### Author-Map Cross-Reference

def orphan_anon_ids(db_path: Path, anon_ids: set) -> List[int]:
    """anon IDs present in a file but missing from author_map."""
    con = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    cur = con.cursor()
    missing: List[int] = []
    ids = sorted(anon_ids)
    for i in range(0, len(ids), 900):  # SQLite host-parameter limit
        chunk = ids[i:i + 900]
        q = ",".join("?" * len(chunk))
        cur.execute(f"SELECT anon_id FROM author_map WHERE anon_id IN ({q})",
                    [str(c) for c in chunk])
        found = {int(r[0]) for r in cur.fetchall()}
        missing.extend(c for c in chunk if c not in found)
    con.close()
    return missing

### Row-Count Reconciliation

# Returns (total rows, distinct id count) of a CSV, or (-1, -1) if it could not
# be read through. Both matter: some anonymized outputs were deduplicated after
# release while their source files still carry the duplicates, so a
# deduplicated output legitimately holds `distinct` rows rather than `total`.
# NOTE: this, not `wc -l`, is the only correct way to count rows in these
# files -- text fields contain embedded newlines.
def count_rows_and_distinct(path: Path) -> Tuple[int, int]:
    state = _new_state()
    try:
        reader = csv.reader(_lines(path, state))
        header = next(reader, None)
        if not header:
            return 0, 0
        try:
            i_id = header.index("id")
        except ValueError:
            return sum(1 for _ in reader), -1
        seen = set()
        total = 0
        for row in reader:
            total += 1
            if len(row) == len(header):
                seen.add(row[i_id])
        return total, len(seen)
    except (csv.Error, OSError, UnicodeError) as e:
        log_report(report_file_path, f"Could not count rows of {path.name}: {type(e).__name__}: {e}")
        return -1, -1

# organize_types is a pure interleave, so the output must contain exactly the
# comment rows plus the submission rows, and its per-type tallies must match the
# two inputs individually.
def reconcile_types(result: Dict, comments_dir: Path, submissions_dir: Path) -> Dict:
    name = result["file"]
    rc = comments_dir / name.replace("ALL", "RC", 1)
    rs = submissions_dir / name.replace("ALL", "RS", 1)

    if not rc.exists() or not rs.exists():
        result["reconcile"] = "skipped: input file missing"
        return result

    n_rc, _ = count_rows_and_distinct(rc)
    n_rs, _ = count_rows_and_distinct(rs)
    if n_rc < 0 or n_rs < 0:
        result["reconcile"] = "skipped: input unreadable"
        return result
    got_c = result["type_counts"].get("comment", 0)
    got_s = result["type_counts"].get("submission", 0)

    result["expected_rows"] = n_rc + n_rs
    result["input_rows"] = {"comments": n_rc, "submissions": n_rs}

    problems = []
    if result["rows"] != n_rc + n_rs:
        problems.append(f"row count {result['rows']} != {n_rc}+{n_rs}={n_rc + n_rs}")
    if got_c != n_rc:
        problems.append(f"comment rows {got_c} != {n_rc}")
    if got_s != n_rs:
        problems.append(f"submission rows {got_s} != {n_rs}")

    if problems:
        result["verdict"] = "SUSPECT"
        result["reason"] = "; ".join(filter(None, [result.get("reason", ""), *problems]))
    result["reconcile"] = "ok" if not problems else "mismatch"
    return result

# Anonymization is row-preserving: the output must have exactly as many rows as
# the file it was built from (or as many distinct ids, if it was deduplicated).
# This is the check that catches a run killed mid-file, since the resume logic
# happily continues past a torn row rather than reporting it.
def reconcile_anon(result: Dict, source_dir: Path) -> Dict:
    src = source_dir / result["file"]
    if not src.exists():
        result["reconcile"] = "skipped: source file missing"
        return result
    expected, distinct = count_rows_and_distinct(src)
    if expected < 0:
        result["reconcile"] = "skipped: source unreadable"
        return result

    result["expected_rows"] = expected
    result["expected_distinct"] = distinct
    got = result["rows"]

    if got == expected:
        result["reconcile"] = "ok"
    elif distinct >= 0 and got == distinct:
        result["reconcile"] = "ok (deduplicated)"
        result["duplicates_in_source"] = expected - distinct
    else:
        msg = (f"row count {got:,} != source total {expected:,} "
               f"and != distinct ids {distinct:,}")
        result["verdict"] = "SUSPECT"
        result["reason"] = "; ".join(filter(None, [result.get("reason", ""), msg]))
        result["reconcile"] = "mismatch"
    return result

### Slurm/array helpers

def build_requested_months(years_list: List[int]) -> List[str]:
    return [f"{y}-{m:02d}" for y in years_list for m in range(1, 13)]

# Months are selected by the YYYY-MM parsed from each filename, so a Slurm
# array slot maps to a fixed month regardless of gaps in the directory.
def select_target_files(all_files: List[Path], array_idx: Optional[int], files_per_job: int) -> List[Path]:
    if array_idx is None or years is None:
        return list(all_files)
    requested = build_requested_months(years)
    target = set(requested[array_idx * files_per_job: (array_idx + 1) * files_per_job])
    return [p for p in all_files if month_of_file(p) in target]

def run_one(path: str, kind: str, quick_mode: bool, db_path: Optional[str]) -> Dict:
    p = Path(path)
    try:
        r = quick_check(p) if quick_mode else full_check(p, kind, Path(db_path) if db_path else None)
    except Exception as e:
        r = {"file": p.name, "verdict": "ERROR", "reason": f"{type(e).__name__}: {e}"}
    r["kind"] = kind
    r["path"] = str(p)
    return r

### Main execution

def verify_integrity() -> int:
    start_time = time.time()

    files_per_job = getattr(args, "files_per_job", 1) or 1
    target_files = select_target_files(file_list, args.array, files_per_job)

    if not target_files:
        log_report(report_file_path, "No target files assigned to this run.")
        return 0

    mode = "quick (head/tail)" if quick else "full (every byte)"
    total_gb = sum(p.stat().st_size for p in target_files) / 2**30
    log_report(
        report_file_path,
        f"Verifying {len(target_files)} file(s) ({total_gb:.1f} GB) in {input_path} for group={group}, "
        f"type={type_} in {mode} mode; checks: {check_kind}."
    )

    db_path = str(USER_CACHE) if (check_kind == "anon" and not quick and USER_CACHE.exists()) else None
    if check_kind == "anon" and not quick and db_path is None:
        log_report(report_file_path, f"Author map not found at {USER_CACHE}; skipping the anon-ID cross-reference.")

    results: List[Dict] = []
    with ProcessPoolExecutor(max_workers=min(VERIFY_WORKERS, len(target_files))) as ex:
        futs = {ex.submit(run_one, str(p), check_kind, quick, db_path): p for p in target_files}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            log_report(report_file_path, f"{r['verdict']:<7} {r['file']}" + (f": {r['reason']}" if r.get("reason") else ""))
    results.sort(key=lambda r: r["file"])

    # Reconciliation reads the counterpart files, so it runs single-threaded
    # afterwards rather than competing with the output reads for disk head time.
    if not quick:
        if check_kind == "types":
            if types_comments_dir.is_dir() and types_submissions_dir.is_dir():
                log_report(report_file_path, f"Reconciling row counts against {types_comments_dir} and {types_submissions_dir}...")
                for r in results:
                    if r["verdict"] != "ERROR":
                        reconcile_types(r, types_comments_dir, types_submissions_dir)
                        if r.get("reconcile") == "mismatch":
                            log_report(report_file_path, f"MISMATCH {r['file']}: {r['reason']}")
            else:
                log_report(report_file_path, "Comments/submissions inputs of this stage not found; skipping row-count reconciliation.")
        elif check_kind == "anon":
            if anon_source_dir.is_dir():
                log_report(report_file_path, f"Reconciling row counts against {anon_source_dir}...")
                for r in results:
                    if r["verdict"] != "ERROR":
                        reconcile_anon(r, anon_source_dir)
                        if r.get("reconcile") == "mismatch":
                            log_report(report_file_path, f"MISMATCH {r['file']}: {r['reason']}")
            else:
                log_report(report_file_path, f"Pre-anonymization source {anon_source_dir} not found; skipping row-count reconciliation.")

        if check_kind == "anon" and db_path:
            total_orphans = sum(r.get("orphan_count") or 0 for r in results)
            with_orphans = [r["file"] for r in results if r.get("orphan_count")]
            failed = [r["file"] for r in results if r.get("xref_error")]
            checked = sum(r.get("distinct_anon_ids") or 0 for r in results)
            log_report(
                report_file_path,
                f"Author-map cross-reference: {total_orphans:,} orphaned anon IDs across {len(with_orphans)} of "
                f"{len(results)} file(s) ({checked:,} distinct IDs checked)"
                + (f"; cross-reference failed for: {', '.join(failed)}" if failed else "") + "."
            )

    with open(details_file_path, "a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps({"group": group, "type": type_, "directory": str(input_path), **r}) + "\n")

    bad = [r for r in results if r["verdict"] != "OK"]
    elapsed = (time.time() - start_time) / 60
    log_report(
        report_file_path,
        f"Finished verification. OK: {len(results) - len(bad)}, Needing attention: {len(bad)}, "
        f"Time: {elapsed:.2f} minutes. Details appended to {details_file_path.name}."
    )
    for r in bad:
        log_report(report_file_path, f"Rebuild {r['file']}: {r['reason']}")
    return len(bad)

if __name__ == "__main__":
    overall_start_time = time.time()
    try:
        n_bad = verify_integrity()
    except Exception as e:
        reraise_fatal(report_file_path, "verify_integrity", e)

    total_time = (time.time() - overall_start_time) / 60
    scope_msg = args.years or "all months present"
    if args.array is not None and years is not None:
        files_per_job = getattr(args, "files_per_job", 1) or 1
        assigned = build_requested_months(years)[args.array * files_per_job: (args.array + 1) * files_per_job]
        scope_msg = f"{args.years} (task scope: {', '.join(assigned) or f'array task {args.array}'})"

    log_report(
        report_file_path,
        f"Integrity verification for {group} / {type_} for {scope_msg} finished in {total_time:.2f} minutes"
    )

    # A file needing attention is the whole point of running this; surface it to
    # Slurm (and any afterok dependent) rather than exiting 0.
    if n_bad:
        sys.exit(1)
