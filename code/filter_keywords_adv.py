# Import functions and objects
from cli import get_args, dir_path
from utils import parse_range, log_report, log_error, load_terms, groups

# Import Python packages
import os, time
import csv
import hyperscan as hs
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import re

# Increase the field size limit to handle larger fields
csv.field_size_limit(2**31 - 1)

# Extract and transform CLI arguments
args = get_args()
years = parse_range(args.years)
group = args.group  

# set path variables
CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

relevance_filtered_path = DATA_DIR / "data_reddit_curated" / group / "filtered_relevance"
output_path = DATA_DIR / "data_reddit_curated" / group / "filtered_keywords_adv"
output_path.mkdir(parents=True, exist_ok=True)

# prepare the report file
output_report_filename = "Report_filter_keywords_adv.csv"
report_file_path = os.path.join(dir_path, output_report_filename)

# Load social group keywords
keyword_path = os.path.join(dir_path.replace("code", "keywords"))
marginalized_words = load_terms(os.path.join(keyword_path, f"{group}_{groups[group][0]}_adv.txt"))
privileged_words   = load_terms(os.path.join(keyword_path, f"{group}_{groups[group][1]}_adv.txt"))
all_words = marginalized_words + privileged_words

# CSV header index assumptions:
MATCHES_COL_INDEX = 7
BODY_COL_INDEX = 2

# Runtime Chimera detection for processing more complex patterns
CH = getattr(hs, "chimera", None)  # None if the installed wheel doesn't expose Chimera
ENGINE = "chimera" if CH is not None else "hyperscan_prefilter"

# Globals populated per process
db = None                  # Hyperscan/Chimera database
id2label = None            # pattern_id -> "Category: term"
compiled_re_by_id = None   # only for fallback: pattern_id -> Python re.Pattern

def _build_id2label(m_words, p_words, group_key):
    mapping = {}
    mid = len(m_words)
    for i, term in enumerate(m_words, start=1):
        mapping[i] = f"{groups[group_key][0]}: {term}"
    for j, term in enumerate(p_words, start=mid + 1):
        mapping[j] = f"{groups[group_key][1]}: {term}"
    return mapping

# Compile either a Chimera DB (if available) or a Hyperscan prefilter DB plus
# Python re confirmers for exact semantics.
def _compile_db(patterns):
    global ENGINE, compiled_re_by_id

    ids = list(range(1, len(patterns) + 1))

    if ENGINE == "chimera":
        db_local = CH.Database()
        db_local.compile(
            expressions=[p.encode("utf-8") for p in patterns],
            ids=ids,
            flags=[hs.HS_FLAG_CASELESS] * len(patterns),
        )
        compiled_re_by_id = None  # not needed
        return db_local

    # Fallback: pure Hyperscan with PREFILTER + Python re confirmation
    # PREFILTER allows unsupported constructs (lookarounds) by approximating them.
    db_local = hs.Database()
    db_local.compile(
        expressions=[p.encode("utf-8") for p in patterns],
        ids=ids,
        flags=[hs.HS_FLAG_CASELESS | hs.HS_FLAG_PREFILTER] * len(patterns),
    )

    # Build Python regexes for exact confirmation (PCRE-like features incl. lookahead)
    compiled = {}
    for i, pat in enumerate(patterns, start=1):
        try:
            compiled[i] = re.compile(pat, re.IGNORECASE)
        except re.error as e:
            # If a pattern can't compile in Python re (rare), fall back to a safe literal
            compiled[i] = re.compile(re.escape(pat), re.IGNORECASE)
            log_report(report_file_path, f"[WARN] Python re compile fallback for pattern ID {i}: {e}")
    compiled_re_by_id = compiled
    return db_local

# intialize the multiprocessing workers: load terms, build id2label, compile DB (Chimera or HS prefilter).
def _worker_init(group_key, keyword_path):

    global db, id2label, marginalized_words, privileged_words, all_words

    mfile = os.path.join(keyword_path, f"{group_key}_{groups[group_key][0]}_adv.txt")
    pfile = os.path.join(keyword_path, f"{group_key}_{groups[group_key][1]}_adv.txt")

    marginalized_words = load_terms(mfile)
    privileged_words   = load_terms(pfile)
    all_words = marginalized_words + privileged_words

    id2label = _build_id2label(marginalized_words, privileged_words, group_key)
    db = _compile_db(all_words)

    # tiny warm-up
    def _noop_cb(*args, **kwargs): return 0
    db.scan(b"warmup", _noop_cb, context=[])

# Read a CSV file from input path, scan BODY_COL_INDEX with multi-regex, write matched rows to output_path with a filled 'matches' column.
def filter_keyword_adv_file(file_name: str):
    input_path = relevance_filtered_path / file_name
    out_stem = Path(file_name).with_suffix('').name
    output_csv_file = output_path / f"{out_stem}.csv"

    total_lines = 0
    matched_lines = 0
    start_time = time.time()

    def on_match(pattern_id, from_, to_, flags_, context):
        context.append(pattern_id)
        return 0

    try:
        with open(input_path, "r", encoding="utf-8-sig", errors="ignore", newline="") as inp, \
             open(output_csv_file, "w", encoding="utf-8", newline="") as outp:

            reader = csv.reader(inp)
            writer = csv.writer(outp)

            header_written = False

            for row_idx, row in enumerate(reader):
                if row_idx == 0:
                    # pass through the header
                    writer.writerow(row)
                    header_written = True
                    continue

                total_lines += 1
                try:
                    if len(row) <= BODY_COL_INDEX:
                        continue

                    text_str = row[BODY_COL_INDEX]
                    text_bytes = text_str.encode("utf-8", "ignore")

                    # Collect candidate IDs
                    match_ids = []
                    db.scan(text_bytes, on_match, context=match_ids)

                    if not match_ids:
                        continue

                    # If using HS prefilter, confirm each candidate ID with Python re
                    if ENGINE != "chimera":
                        uniq = sorted(set(match_ids))
                        confirmed = []
                        for pid in uniq:
                            regex = compiled_re_by_id.get(pid)
                            if regex is not None and regex.search(text_str):
                                confirmed.append(pid)
                        match_ids = confirmed

                    if match_ids:
                        matched_lines += 1
                        writer.writerow(row)

                except Exception as e:
                    log_error("filter_keyword_adv_file", file_name, total_lines, row, e)

            if not header_written:
                # if input was empty, still emit the expected header
                writer.writerow(["id","parent_id","body","author","created_utc","subreddit","score","matches"])

    except Exception as e:
        log_report(report_file_path, f"Error filtering by advanced keywords in file {Path(file_name).name}: {e}")

    elapsed_time = (time.time() - start_time) / 60
    log_report(
        report_file_path,
        f"[{ENGINE}] Filtered {Path(file_name).name} in {elapsed_time:.2f} minutes. "
        f"Total lines: {total_lines}, matched lines: {matched_lines}"
    )
    return total_lines, matched_lines

def filter_keyword_adv_month(year, month, files):
    log_report(report_file_path, f"Started filtering files by advanced keywords for {year}-{month}; Engine: {ENGINE}.")
    start_time = time.time()
    total_lines = 0
    matched_lines = 0

    for file in files:
        try:
            t, m = filter_keyword_adv_file(file)
            total_lines += t
            matched_lines += m
        except Exception as e:
            log_report(report_file_path, f"Error filtering by advanced keywords in file {Path(file).name}: {e}")

    elapsed_time = (time.time() - start_time) / 60
    log_report(report_file_path, f"Completed filtering {year}-{month} in {elapsed_time:.2f} minutes")
    return total_lines, matched_lines

def filter_keyword_adv_parallel():
    total_lines = 0
    matched_lines = 0
    max_workers = min(6, os.cpu_count())
    log_report(report_file_path, f"Using {max_workers} processes for parallel processing. Engine: {ENGINE}")

    initargs = (group, keyword_path)

    with ProcessPoolExecutor(max_workers=max_workers, initializer=_worker_init, initargs=initargs) as executor:
        for year in years:
            log_report(report_file_path, f"Processing year: {year}")
            start_year_time = time.time()
            files_by_month = {}
            futures = []

            # List all CSV files in relevance_filtered_path for this year
            for file in sorted(os.listdir(relevance_filtered_path)):
                if str(year) in file and file.endswith(".csv"):
                    try:
                        # Expect filename format like "YYYY-MM-..."
                        month = file.split('-')[1].split('.')[0]
                    except IndexError:
                        continue
                    files_by_month.setdefault(month, []).append(file)

            # Warn about missing months
            expected_months = [f"{m:02d}" for m in range(1, 13)]
            missing_months = [m for m in expected_months if m not in files_by_month]
            if missing_months:
                log_report(report_file_path, f"Warning: Missing files for months {missing_months} in year {year}")

            # Submit work
            for month, files in sorted(files_by_month.items()):
                futures.append(executor.submit(filter_keyword_adv_month, year, month, files))

            for future in futures:
                try:
                    t, m = future.result()
                    total_lines += t
                    matched_lines += m
                except Exception as e:
                    log_report(report_file_path, f"Error filtering by keywords: {e}")

            year_processing_time = (time.time() - start_year_time) / 60
            log_report(report_file_path, f"Completed filtering year {year} in {year_processing_time:.2f} minutes")

    # Sanity check: output file count
    expected_file_count = len(years) * 12
    actual_file_count = sum(
        1 for f in os.listdir(output_path)
        if f.endswith('.csv') and f != output_report_filename
    )
    if actual_file_count != expected_file_count:
        log_report(report_file_path, f"Warning: Expected {expected_file_count} output files, but generated {actual_file_count}.")

    log_report(report_file_path, f"Total lines processed: {total_lines}")
    log_report(report_file_path, f"Total matched lines: {matched_lines}")

if __name__ == "__main__":
    overall_start_time = time.time()
    try:
        filter_keyword_adv_parallel()
    except Exception as e:
        log_report(report_file_path, f"Fatal error during processing [{ENGINE}]: {e}")
    total_time = (time.time() - overall_start_time) / 60
    log_report(report_file_path, f"Advanced keyword filtering for {group} for {args.years} finished in {total_time:.2f} minutes [{ENGINE}]")