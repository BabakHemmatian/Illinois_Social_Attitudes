from cli import get_args, dir_path, raw_data
from utils import load_terms, groups, headers, parse_range
import os
import csv
import json
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import zstandard
import io
import re

# Normalize imported paths to work across OSes
dir_path = os.path.normpath(dir_path)
raw_data = os.path.normpath(raw_data)

# Extract and transform CLI arguments
args = get_args()
years = parse_range(args.years)
if isinstance(years, int):
    years = [years]

# Build the output folder path in an OS-independent way using args.group
output_path = os.path.join(dir_path.replace("code", os.path.join("data", "data_reddit_curated", args.group, "filtered_keywords")))
os.makedirs(output_path, exist_ok=True)

# Set up the report file (tab-separated format)
output_report_filename = "Report_FilterKeywords.csv"
report_file_path = os.path.join(output_path, output_report_filename)
with open(report_file_path, 'w', encoding='utf-8', newline='') as report_file:
    writer = csv.writer(report_file, delimiter='\t')
    writer.writerow(["Timestamp", "Message"])

# Helper function to log report messages (only summary lines are recorded)
def log_report(message):
    # Use M/D/YYYY H:M format
    timestamp = datetime.now().strftime('%-m/%-d/%Y %H:%M')
    with open(report_file_path, 'a', encoding='utf-8', newline='') as report_file:
        writer = csv.writer(report_file, delimiter='\t')
        writer.writerow([timestamp, message])
    print(f"{timestamp} - {message}")

# Extract the social group keywords
keyword_dir = os.path.join(dir_path.replace("code", "keywords"))
marginalized_words = load_terms(os.path.join(keyword_dir, "{}_{}.txt".format(args.group, groups[args.group][0])))
privileged_words = load_terms(os.path.join(keyword_dir, "{}_{}.txt".format(args.group, groups[args.group][1])))

# Build regex patterns for fast pattern matching of the extracted keywords.
marginalized_pattern = re.compile("|".join(re.escape(term) for term in marginalized_words), re.IGNORECASE)
privileged_pattern = re.compile("|".join(re.escape(term) for term in privileged_words), re.IGNORECASE)

# Survey the output directory for already processed files (by filename)
processed_files = set(f for f in os.listdir(output_path) if f.endswith('.csv'))

# Define the error log file (all errors will be appended to this single file)
error_log_filepath = os.path.join(output_path, "Error_FilterKeywords.txt")

# Function for logging errors to a single file
def log_error(file, line_number, line_content, error):
    error_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    resource_identifier = file.split('.zst')[0]
    error_string = f"ErrorType_FilterKeywords_{resource_identifier}_{line_number}_{error_time}: {str(error)} | Line content: {line_content}\n"
    with open(error_log_filepath, 'a', encoding='utf-8') as error_log:
        error_log.write(error_string)
    log_report(f"Logged error: {error_string.strip()}")

# Function for processing a single raw Reddit file
def filter_keyword_file(file):
    file_path = os.path.join(raw_data, file)
    output_csv_file = os.path.join(output_path, f"{file.split('.zst')[0]}.csv")
    
    buffer = []
    buffer_size = 20
    total_lines = 0
    matched_lines = 0
    start_time = time.time()
    
    try:
        with open(file_path, 'rb') as fh, open(output_csv_file, 'w', newline='', encoding='utf-8') as csv_file:
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
                    
                    marg_results = marginalized_pattern.findall(comment_text)
                    priv_results = privileged_pattern.findall(comment_text)
                    
                    if marg_results or priv_results:
                        matched_lines += 1
                        matches = ([f"{groups[args.group][0]}: {term}" for term in set(marg_results)] +
                                   [f"{groups[args.group][1]}: {term}" for term in set(priv_results)])
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
                    log_error(file, total_lines, line, e)
            
            if buffer:
                writer.writerows(buffer)
    
    except Exception as e:
        log_report(f"Error filtering by keywords in file {file}: {e}")
    
    elapsed_time = (time.time() - start_time) / 60
    # Log only the summary line for this file in the report
    log_report(f"Filtered {file} by keywords in {elapsed_time:.2f} minutes. Total lines: {total_lines}, matched lines: {matched_lines}")
    return total_lines, matched_lines

# Wrapper function for processing a month's files (no report logging here)
def filter_keyword_month(year, month, files):
    total_lines = 0
    matched_lines = 0
    
    for file in files:
        try:
            file_lines, file_matched = filter_keyword_file(file)
            total_lines += file_lines
            matched_lines += file_matched
        except Exception as e:
            log_report(f"Error filtering by keywords in file {file}: {e}")
    
    return total_lines, matched_lines

# Process files in parallel and check for missing months/files
def filter_keyword_parallel():
    total_lines = 0
    matched_lines = 0
    max_workers = min(6, os.cpu_count())
    
    # No intermediate "started" logs; only final summaries will be recorded in the report.
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        for year in years:
            files_by_month = {}
            for file in sorted(os.listdir(raw_data)):
                if str(year) in file and file.endswith(".zst") and file.split('.zst')[0] not in processed_files:
                    try:
                        month = file.split('-')[1][:2]
                    except IndexError:
                        continue
                    files_by_month.setdefault(month, []).append(file)
            
            expected_months = [f"{m:02d}" for m in range(1, 13)]
            missing_months = [m for m in expected_months if m not in files_by_month]
            if missing_months:
                error_msg = f"Error: Missing files for months {missing_months} in year {year}"
                log_report(error_msg)
                raise Exception(error_msg)
            
            for month, files in sorted(files_by_month.items()):
                futures.append(executor.submit(filter_keyword_month, year, month, files))
            
            for future in futures:
                try:
                    month_lines, month_matched = future.result()
                    total_lines += month_lines
                    matched_lines += month_matched
                except Exception as e:
                    log_report(f"Error filtering by keywords: {e}")
    
    # Check output CSV file count (subtracting the report file)
    expected_file_count = len(years) * 12
    actual_file_count = sum(1 for f in os.listdir(output_path) if f.endswith('.csv')) - 1
    if actual_file_count != expected_file_count:
        log_report(f"Warning: Expected {expected_file_count} output files, but generated {actual_file_count}.")
    
    log_report(f"Total lines processed across all files: {total_lines}")
    log_report(f"Total matched lines across all files: {matched_lines}")

if __name__ == "__main__":
    overall_start_time = time.time()
    try:
        filter_keyword_parallel()
    except Exception as e:
        log_report(f"Fatal error during processing: {e}")
    total_time = (time.time() - overall_start_time) / 60
    log_report(f"Keyword filtering for the {args.group} social group for {args.years} was finished in {total_time:.2f} minutes")