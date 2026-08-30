"""
Post-crash integrity verification for the curated Reddit CSVs.

Directory metadata (including file sizes and mtimes) is unreliable after a
storage fault, so nothing here trusts stat() -- every verdict comes from
reading bytes.

Two failure modes this is built to catch:

  1. organize_types resumes by checking only that ALL_YYYY-MM.csv *exists*
     (organize_types.py, build_merge_jobs). A file whose rename committed but
     whose tail was lost is treated as complete forever.

  2. organize_anonymize resumes via get_last_source_row (utils.py), which
     deliberately tolerates unparseable rows. A torn row stays embedded in the
     file and the next run appends past it.

Usage:
    python code/verify_integrity.py --scope types --quick     # fast triage
    python code/verify_integrity.py --scope all               # full read
    python code/verify_integrity.py --scope anon --db <path>  # + user_map xref
"""

### Imports

import argparse
import csv
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

csv.field_size_limit(2**31 - 1)  # match the pipeline's limit; some text fields are huge

### Configuration

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CURATED = PROJECT_ROOT / "data" / "data_reddit_curated"

# Mirrors organize_types.TIME_FORMATS
TIME_FORMATS = ("%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M")

# Placeholders organize_anonymize leaves untouched
PRESERVED_AUTHORS = {"", "[deleted]", "[removed]"}

TAIL_BYTES = 8 << 20  # how much of the tail --quick inspects
MONTH_RE = re.compile(r"(\d{4})-(\d{2})")


def parse_time(value: str) -> Optional[datetime]:
    value = (value or "").strip()
    for fmt in TIME_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def month_of(path: Path) -> Tuple[int, int]:
    m = MONTH_RE.search(path.name)
    if not m:
        raise ValueError(f"Could not parse YYYY-MM from {path.name}")
    return int(m.group(1)), int(m.group(2))


### Quick mode: head + tail only

# Truncation is the dominant failure mode from a lost delayed write, and it is
# visible without reading the whole file: a time-ordered month file that stops
# early simply never reaches the end of its month.
def quick_check(path: Path) -> Dict:
    out: Dict = {"file": path.name, "mode": "quick", "size": path.stat().st_size}
    year, month = month_of(path)
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
    out["header_ok"] = head.startswith(b"id,parent id,text,author,time")

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


### Full mode: complete parse

# Feeds csv.reader while recording corruption signals the parser would hide.
#
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


def full_check(path: Path, kind: str = "types", db_path: Optional[Path] = None) -> Dict:
    state = {"nul_lines": 0, "replacement_chars": 0, "ends_with_newline": None,
             "max_line_bytes": 0}
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
    # Source and anon files for the same month carry identical counts.
    # Reported, but not treated as damage on its own. A count
    # that DROPS relative to the source file does indicate truncation, which the
    # row-count reconciliation catches directly.
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


### Row-count reconciliation

# organize_types is a pure interleave, so the output must contain exactly the
# comment rows plus the submission rows, and its per-type tallies must match the
# two inputs individually.
def count_rows(path: Path) -> int:
    """Row count for an input file, or -1 if it could not be read through."""
    total, _ = count_rows_and_distinct(path)
    return total


# Returns (total rows, distinct id count). Both matter: some anonymized outputs
# were deduplicated after release while their source files still carry the
# duplicates, so a deduplicated output legitimately holds `distinct` rows rather
# than `total`. Comparing against `total` alone reports every deduplicated month
# as damaged.
def count_rows_and_distinct(path: Path) -> Tuple[int, int]:
    state = {"nul_lines": 0, "replacement_chars": 0, "ends_with_newline": None,
             "max_line_bytes": 0}
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
        print(f"    ! could not count {path.name}: {type(e).__name__}: {e}")
        return -1, -1


def reconcile_types(result: Dict, comments_dir: Path, submissions_dir: Path,
                    cache: Dict[str, int]) -> Dict:
    name = result["file"]
    rc = comments_dir / name.replace("ALL", "RC", 1)
    rs = submissions_dir / name.replace("ALL", "RS", 1)

    if not rc.exists() or not rs.exists():
        result["reconcile"] = "skipped: input file missing"
        return result

    for p in (rc, rs):
        if p.name not in cache:
            cache[p.name] = count_rows_and_distinct(p)

    n_rc = cache[rc.name][0] if isinstance(cache[rc.name], tuple) else cache[rc.name]
    n_rs = cache[rs.name][0] if isinstance(cache[rs.name], tuple) else cache[rs.name]
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


# organize_anonymize logs the row count it wrote for each month. When a month
# was written exactly once, that count IS the source row count, so it can stand
# in for re-reading the source -- roughly halving a full sweep's I/O. Months
# written more than once are omitted: their figures are per-invocation appends,
# not totals.
def expected_rows_from_log(report_path: Path, group: str, stage: str) -> Dict[str, int]:
    if not report_path.exists():
        return {}

    counts: Dict[str, List[int]] = {}
    ctx_group = ctx_stage = None
    # Lines are timestamp-prefixed ("...00:23:20,labeled_location identified..."),
    # so this cannot be anchored to the start of the line.
    stage_re = re.compile(r"([A-Za-z]\w*) identified as the most advanced curated "
                          r"dataset for (\w+) entries of type")

    with open(report_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = stage_re.search(line)
            if m:
                ctx_stage, ctx_group = m.group(1), m.group(2)
                continue
            m = re.search(r"Anonymized ALL_(\d{4}-\d{2})\.csv .*?rows=(\d+)", line)
            if m and ctx_group == group and ctx_stage == stage:
                counts.setdefault(m.group(1), []).append(int(m.group(2)))

    return {mo: v[0] for mo, v in counts.items() if len(v) == 1}


# Anonymization is row-preserving: the output must have exactly as many rows as
# the labeled_location file it was built from. This is the check that catches a
# run killed mid-file, since get_last_source_row will happily resume past a torn
# row rather than reporting it.
def reconcile_anon(result: Dict, source_dir: Path, cache: Dict[str, int],
                   log_counts: Optional[Dict[str, int]] = None) -> Dict:
    src = source_dir / result["file"]
    month = MONTH_RE.search(result["file"]).group(0)

    # log counts predate the dedup pass, so they can only ever be a fast
    # pre-filter -- a shortfall against them is not evidence of damage.
    if log_counts and month in log_counts:
        expected, distinct = log_counts[month], -1
        result["expected_source"] = "log (pre-dedup)"
    else:
        if not src.exists():
            result["reconcile"] = "skipped: source file missing"
            return result
        if src.name not in cache:
            cache[src.name] = count_rows_and_distinct(src)
        expected, distinct = cache[src.name]
        result["expected_source"] = "source file"
    if expected < 0:
        result["reconcile"] = "skipped: source unreadable"
        return result

    result["expected_rows"] = expected
    result["expected_distinct"] = distinct
    got = result["rows"]

    if got == expected:
        result["reconcile"] = "ok"
    elif distinct >= 0 and got == distinct:
        # Matches the source's distinct-id count: this month was deduplicated.
        result["reconcile"] = "ok (deduplicated)"
        result["duplicates_in_source"] = expected - distinct
    elif distinct < 0:
        # Only a pre-dedup log count was available; cannot rule out dedup.
        result["reconcile"] = "unconfirmed: log count is pre-dedup"
        result["reason"] = "; ".join(filter(None, [
            result.get("reason", ""),
            f"row count {got:,} != logged {expected:,}; re-run without "
            f"--use-log-counts to distinguish dedup from data loss"]))
    else:
        msg = (f"row count {got:,} != source total {expected:,} "
               f"and != distinct ids {distinct:,}")
        result["verdict"] = "SUSPECT"
        result["reason"] = "; ".join(filter(None, [result.get("reason", ""), msg]))
        result["reconcile"] = "mismatch"
    return result


### user_map cross-reference

# The outputs store anon_id only, never the original author, so a mapping that
# vanished with the lost WAL cannot be reconstructed from the CSVs. Any anon_id
# present in a file but absent from author_map marks an orphaned author whose
# months must be re-anonymized from the pre-anon inputs.
def orphan_anon_ids(db_path: Path, anon_ids: set) -> List[int]:
    """anon IDs present in a file but missing from author_map.

    Run per file inside the worker: the sexuality corpus is 176 GB across 204
    months, and hoisting every ID into the parent process exhausts memory long
    before the scan finishes.
    """
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cur = con.cursor()
    missing: List[int] = []
    ids = sorted(anon_ids)
    for i in range(0, len(ids), 900):
        chunk = ids[i:i + 900]
        q = ",".join("?" * len(chunk))
        cur.execute(f"SELECT anon_id FROM author_map WHERE anon_id IN ({q})",
                    [str(c) for c in chunk])
        found = {int(r[0]) for r in cur.fetchall()}
        missing.extend(c for c in chunk if c not in found)
    con.close()
    return missing


def crossref_user_map(db_path: Path, anon_ids: set) -> Dict:
    if not anon_ids:
        return {"checked": 0, "orphans": 0, "note": "no anon ids collected"}

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cur = con.cursor()
    known = set()
    ids = sorted(anon_ids)
    for i in range(0, len(ids), 900):
        chunk = ids[i:i + 900]
        q = ",".join("?" * len(chunk))
        cur.execute(f"SELECT anon_id FROM author_map WHERE anon_id IN ({q})", chunk)
        known.update(r[0] for r in cur.fetchall())
    total = cur.execute("SELECT COUNT(*) FROM author_map").fetchone()[0]
    con.close()

    orphans = sorted(anon_ids - known)
    return {
        "checked": len(anon_ids),
        "in_db": len(known),
        "orphans": len(orphans),
        "db_total_rows": total,
        "orphan_sample": orphans[:20],
    }


### Driver

class Layout:
    """Resolves the four directories a group/stage pair spans."""

    def __init__(self, group: str, stage: str):
        self.group, self.stage = group, stage
        self.merged = CURATED / group / "all" / stage
        self.anon = CURATED / group / "all" / f"{stage}_anon"
        self.comments = CURATED / group / "comments" / stage
        self.submissions = CURATED / group / "submissions" / stage


# Groups may sit at different pipeline stages, and older runs can leave an
# earlier *_anon directory behind, so the stage is resolved from disk rather
# than assumed.
def resolve_stage(group: str, stage: Optional[str]) -> str:
    if stage:
        return stage
    base = CURATED / group / "all"
    candidates = [d for d in base.iterdir()
                  if d.is_dir() and not d.name.endswith("_anon")] if base.is_dir() else []
    if not candidates:
        raise SystemExit(f"No stage directories found under {base}")
    # Most-advanced stage == the one with the most month files.
    best = max(candidates, key=lambda d: len(list(d.glob("ALL_*.csv"))))
    return best.name


def gather(scope: str, layout: Layout) -> List[Tuple[str, Path]]:
    jobs: List[Tuple[str, Path]] = []
    if scope in ("types", "all"):
        jobs += [("types", p) for p in sorted(layout.merged.glob("ALL_*.csv"))]
    if scope in ("anon", "all"):
        jobs += [("anon", p) for p in sorted(layout.anon.glob("ALL_*.csv"))]
    return jobs


def run_one(kind: str, path: str, quick: bool, db_path: Optional[str] = None) -> Dict:
    p = Path(path)
    try:
        r = quick_check(p) if quick else full_check(p, kind, db_path)
    except Exception as e:
        r = {"file": p.name, "verdict": "ERROR", "reason": f"{type(e).__name__}: {e}"}
    r["kind"] = kind
    r["path"] = str(p)
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scope", choices=["types", "anon", "all"], default="all")
    ap.add_argument("--group", default="age",
                    help="curated group to verify (age, sexuality, ...)")
    ap.add_argument("--stage", default=None,
                    help="stage directory; defaults to the most advanced one present")
    ap.add_argument("--quick", action="store_true",
                    help="head/tail only; catches truncation without reading everything")
    ap.add_argument("--workers", type=int, default=2,
                    help="keep low: parallel reads thrash a spinning disk (default 2)")
    ap.add_argument("--use-log-counts", action="store_true",
                    help="take expected row counts from report_organize_anonymize.csv "
                         "instead of re-reading source files (much less I/O)")
    ap.add_argument("--no-reconcile", action="store_true",
                    help="skip row-count reconciliation against the RC/RS inputs")
    ap.add_argument("--db", type=Path, default=CURATED.parent / "user_map.sqlite3",
                    help="user_map.sqlite3 to cross-reference (use a copy, not the original)")
    ap.add_argument("--out", type=Path,
                    default=PROJECT_ROOT / "verify_integrity_results.jsonl")
    ap.add_argument("--months", nargs="*", help="limit to specific YYYY-MM values")
    args = ap.parse_args()

    layout = Layout(args.group, resolve_stage(args.group, args.stage))
    print(f"group={layout.group}  stage={layout.stage}")

    jobs = gather(args.scope, layout)
    if args.months:
        want = set(args.months)
        jobs = [(k, p) for k, p in jobs if MONTH_RE.search(p.name).group(0) in want]

    if not jobs:
        print("No files matched.", file=sys.stderr)
        return 1

    total_bytes = sum(p.stat().st_size for _, p in jobs)
    print(f"Verifying {len(jobs)} files ({total_bytes / 2**30:.1f} GB) "
          f"in {'quick' if args.quick else 'full'} mode with {args.workers} workers.\n")

    started = time.time()
    results: List[Dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, k, str(p), args.quick, str(args.db) if args.db.exists() else None): p for k, p in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            flag = {"OK": "  ok  ", "SUSPECT": "SUSPECT", "ERROR": " ERROR"}[r["verdict"]]
            print(f"[{i:>3}/{len(jobs)}] {flag}  {r['file']:<20} {r.get('reason', '')}")

    # Reconciliation reads the RC/RS inputs, so it runs single-threaded afterwards
    # rather than competing with the output reads for disk head time.
    if not args.quick and not args.no_reconcile:
        cache: Dict[str, int] = {}
        merged_dir = layout.merged

        if args.scope in ("types", "all"):
            print("\nReconciling merged files against comment/submission inputs...")
            for r in results:
                if r["kind"] == "types" and r["verdict"] != "ERROR":
                    reconcile_types(r, layout.comments, layout.submissions, cache)
                    if r.get("reconcile") == "mismatch":
                        print(f"  MISMATCH  {r['file']}  {r['reason']}")

        if args.scope in ("anon", "all"):
            print("\nReconciling anonymized files against their sources...")
            for r in results:
                if r["kind"] == "anon" and r["verdict"] != "ERROR":
                    reconcile_anon(r, merged_dir, cache)
                    if r.get("reconcile") == "mismatch":
                        print(f"  MISMATCH  {r['file']}  {r['reason']}")

    # Each worker already checked its own file's IDs against author_map, so this
    # only aggregates -- no corpus-wide ID set is ever built in this process.
    xref = None
    anon_results = [r for r in results if r["kind"] == "anon"]
    if not args.quick and anon_results and args.db.exists():
        total_orphans = sum(r.get("orphan_count") or 0 for r in anon_results)
        files_with = [r for r in anon_results if r.get("orphan_count")]
        failed = [r for r in anon_results if r.get("xref_error")]
        checked = sum(r.get("distinct_anon_ids") or 0 for r in anon_results)
        xref = {"files_checked": len(anon_results), "orphans": total_orphans,
                "files_with_orphans": [r["file"] for r in files_with],
                "xref_failures": [r["file"] for r in failed]}
        print(f"\nuser_map cross-reference: {total_orphans:,} orphaned anon IDs "
              f"across {len(files_with)} of {len(anon_results)} files "
              f"({checked:,} ID occurrences checked)")
        for r in files_with:
            print(f"  {r['file']}  {r['orphan_count']:,} orphans  {r.get('orphan_sample')}")
        for r in failed:
            print(f"  {r['file']}  xref failed: {r['xref_error']}")

    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
        if xref:
            f.write(json.dumps({"kind": "user_map_xref", **xref}) + "\n")

    bad = [r for r in results if r["verdict"] != "OK"]
    print(f"\n{'=' * 62}")
    print(f"{len(results) - len(bad)} OK / {len(bad)} needing attention "
          f"in {(time.time() - started) / 60:.1f} min")
    if bad:
        print("\nRebuild these:")
        for r in sorted(bad, key=lambda r: r["file"]):
            print(f"  {r['file']:<20} {r['reason']}")
    print(f"\nDetails: {args.out}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
