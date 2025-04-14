import os
import csv
import json
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import zstandard
import io
import sys
import ahocorasick

# Import functions and objects from local modules
from cli import get_args, dir_path, raw_data
from utils import load_terms, groups, headers, parse_range, log_report, log_error

# Extract and transform CLI arguments
args = get_args()
years = parse_range(args.years)
if isinstance(years, int):
    years = [years]

# Load social group keywords
keyword_path = os.path.join(dir_path.replace("code", "keywords"))
marginalized_words = load_terms(os.path.join(keyword_path, f"{args.group}_{groups[args.group][0]}.txt"))
privileged_words = load_terms(os.path.join(keyword_path, f"{args.group}_{groups[args.group][1]}.txt"))

# Build an Aho-Corasick automaton for fast pattern matching of the keywords
automaton = ahocorasick.Automaton()
for term in marginalized_words:
    automaton.add_word(term.lower(), (groups[args.group][0], term))
for term in privileged_words:
    automaton.add_word(term.lower(), (groups[args.group][1], term))
automaton.make_automaton()

# Prepare and inspect the output path
output_path = os.path.join(dir_path.replace("code", os.path.join("data", "data_reddit_curated", args.group, "filtered_keywords")))
os.makedirs(output_path, exist_ok=True)
processed_files = set(f for f in os.listdir(output_path) if f.endswith('.csv'))

# Setup the report file (tab-separated format)
output_report_filename = "Report_filter_keywords.csv"
report_file_path = os.path.join(output_path, output_report_filename)
if not os.path.exists(report_file_path):
    mode = 'w'
else:
    mode = 'a'
with open(report_file_path, mode, encoding='utf-8', newline='') as report_file:
    writer = csv.writer(report_file, delimiter='\t')
    if mode == 'w':
        writer.writerow(["Timestamp", "Message"])

def filter_keyword_file(file):
    """
    Process a single raw Reddit file by filtering for keyword matches.
    Writes matching lines to an output CSV file.
    """
    file_path = os.path.join(raw_data, file)
    output_csv_file = os.path.join(output_path, f"{file.split('.zst')[0]}.csv")
    
    buffer = []
    buffer_size = 20
    total_lines = 0
    matched_lines = 0
    start_time = time.time()

    try:
        with open(file_path, 'rb') as fh, \
             open(output_csv_file, 'w', newline='', encoding='utf-8') as csv_file:
            
            writer = csv.writer(csv_file)
            writer.writerow(headers)
            
            dctx = zstandard.ZstdDecompressor(max_window_size=2 ** 31)
            stream_reader = dctx.stream_reader(fh, read_across_frames=True)
            text_stream = io.TextIOWrapper(stream_reader, encoding='utf-8')

            for line in text_stream:
                total_lines += 1
                try:
                    contents = json.loads(line)
                    comment_text = contents.get('body', '').strip().lower()

                    matches = []
                    for end_index, (category, term) in automaton.iter(comment_text):
                        matches.append(f"{category}: {term}")

                    if matches:
                        matched_lines += 1
                        buffer.append([
                            contents.get("id", ""),
                            contents.get("parent_id", ""),
                            comment_text,
                            contents.get("author", ""),
                            datetime.fromtimestamp(int(contents.get("created_utc", 0))).strftime('%Y-%m-%d %H:%M:%S'),
                            contents.get("subreddit", ""),
                            contents.get("score", ""),
                            ', '.join(matches)
                        ])

                        if len(buffer) >= buffer_size:
                            writer.writerows(buffer)
                            buffer.clear()
                except Exception as e:
                    log_error("filter_keyword_file",file, total_lines, line, e)
            
            if buffer:
                writer.writerows(buffer)

    except Exception as e:
        log_report(report_file_path, f"Error filtering by keywords in file {file}: {e}")

    elapsed_time = (time.time() - start_time) / 60
    log_report(report_file_path, f"Filtered {file} by for relevance to the {args.group} social group based on keywords in {elapsed_time:.2f} minutes. Total lines: {total_lines}, matched lines: {matched_lines}")

    return total_lines, matched_lines

def filter_keyword_month(year, month, files):
    """
    Process all files for a specific month.
    """
    log_report(report_file_path, f"Started filtering files for {year}-{month}")
    start_time = time.time()
    total_lines = 0
    matched_lines = 0

    for file in files:
        try:
            file_lines, file_matched = filter_keyword_file(file)
            total_lines += file_lines
            matched_lines += file_matched
        except Exception as e:
            log_report(report_file_path, f"Error filtering by keywords in file {file}: {e}")

    elapsed_time = (time.time() - start_time) / 60
    log_report(report_file_path, f"Completed filtering {year}-{month} in {elapsed_time:.2f} minutes")
    return total_lines, matched_lines

def filter_keyword_parallel():
    """
    Process files in parallel while checking for missing month files.
    Log warnings for missing months before processing.
    """
    total_lines = 0
    matched_lines = 0
    max_workers = min(6, os.cpu_count())
    log_report(report_file_path, f"Using {max_workers} processes for parallel processing.")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        for year in years:
            log_report(report_file_path, f"Processing year: {year}")
            start_year_time = time.time()
            files_by_month = {}

            for file in sorted(os.listdir(raw_data)):
                if str(year) in file and file.endswith(".zst") and file.split('.zst')[0] not in processed_files:
                    try:

                        # Assuming filename format includes month as "YYYY-MM" or "YYYY-MM-..."
                        month = file.split('-')[1].split('.zst')[0]
                    except IndexError:
                        continue
                    files_by_month.setdefault(month, []).append(file)

            # Log a warning if some months are missing for the current year
            expected_months = [f"{m:02d}" for m in range(1, 13)]
            missing_months = [m for m in expected_months if m not in files_by_month]
            if missing_months:
                warning_msg = f"Warning: Missing files for months {missing_months} in year {year}"
                log_report(report_file_path, warning_msg)

            for month, files in sorted(files_by_month.items()):
                futures.append(executor.submit(filter_keyword_month, year, month, files))

            for future in futures:
                try:
                    month_lines, month_matched = future.result()
                    total_lines += month_lines
                    matched_lines += month_matched
                except Exception as e:
                    log_report(report_file_path, f"Error filtering by keywords: {e}")

            year_processing_time = (time.time() - start_year_time) / 60
            log_report(report_file_path, f"Completed filtering year {year} in {year_processing_time:.2f} minutes")

    # TODO: Make the warning message specific to the files in years-months, not counting any .csv output file.
    # Check the number of output CSV files, excluding the report file
    expected_file_count = len(years) * 12
    actual_file_count = sum(1 for f in os.listdir(output_path) if f.endswith('.csv') and f != output_report_filename)
    if actual_file_count != expected_file_count:
        log_report(report_file_path, f"Warning: Expected {expected_file_count} output files, but generated {actual_file_count}.")

    log_report(report_file_path, f"Total lines processed: {total_lines}")
    log_report(report_file_path, f"Total matched lines: {matched_lines}")

if __name__ == "__main__":
    overall_start_time = time.time()
    try:
        filter_keyword_parallel()
    except Exception as e:
        log_report(report_file_path, f"Fatal error during processing: {e}")
    total_time = (time.time() - overall_start_time) / 60
    log_report(report_file_path, f"Keyword filtering for {args.group} for {args.years} finished in {total_time:.2f} minutes")
