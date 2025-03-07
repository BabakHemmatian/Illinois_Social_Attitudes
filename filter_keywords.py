# import functions and objects
from cli import get_args,dir_path,raw_data
from utils import load_terms,groups,headers,parse_range

# import python modules
import ahocorasick
import os
import csv
import json
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import zstandard
import io

# Extract and transform CLI arguments 
args = get_args()
years = parse_range(args.years)

# Extract the social group keywords
keyword_path = os.path.join(dir_path.replace("code","keywords"))
marginalized_words = load_terms(keyword_path+"\\{}_{}.txt".format(args.group,groups[args.group][0]))
privileged_words = load_terms(keyword_path+"\\{}_{}.txt".format(args.group,groups[args.group][1]))

# Build an automaton for fast pattern matching of the extracted keywords
automaton = ahocorasick.Automaton()
for term in marginalized_words:
    automaton.add_word(term, (groups[args.group][0], term))
for term in privileged_words:
    automaton.add_word(term, (groups[args.group][1], term))
automaton.make_automaton()

# Prepare and survey the output path
output_path = os.path.join(dir_path.replace("code","data\\data_reddit_curated\\{}\\filtered_keywords".format(args.group)))
os.makedirs(output_path, exist_ok=True)
processed_files = set(f for f in os.listdir(output_path) if f.endswith('.csv'))

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
        with open(file_path, 'rb') as fh, \
             open(output_csv_file, 'w', newline='', encoding='utf-8') as csv_file:
            
            writer = csv.writer(csv_file)
            writer.writerow(headers)
            
            dctx = zstandard.ZstdDecompressor(max_window_size=2 ** 31)
            stream_reader = dctx.stream_reader(fh, read_across_frames=True)
            text_stream = io.TextIOWrapper(stream_reader, encoding='utf-8')

            for line in text_stream:
                total_lines += 1
                contents = json.loads(line)
                comment_text = contents['body'].strip().lower()

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

            if buffer:
                writer.writerows(buffer)

    except Exception as e:
        print(f"Error filtering by keywords in file {file}: {e}")

    elapsed_time = (time.time() - start_time) / 60
    print(f"Filtered {file} by keywords in {elapsed_time:.2f} minutes. Total lines: {total_lines}, matched lines:{matched_lines}")
    return total_lines, matched_lines

# Wrapper function for process_single_file
def filter_keyword_month(year, month, files):
    """Filter a specific month by keywords."""
    print(f"Started filtering the following file by keywords: {year}-{month}")
    start_time = time.time()
    total_lines = 0
    matched_lines = 0

    for file in files:
        try:
            file_lines, file_matched = filter_keyword_file(file)
            total_lines += file_lines
            matched_lines += file_matched
        except Exception as e:
            print(f"Error filtering by keywords in file {file}: {e}")

    elapsed_time = (time.time() - start_time) / 60
    print(f"Completed filtering {year}-{month} by keywords in {elapsed_time:.2f} minutes")
    return total_lines, matched_lines

# Process files in parallel
def filter_keyword_parallel():
    total_lines = 0
    matched_lines = 0

    max_workers = min(6, os.cpu_count())  # Use up to 8 processes or the number of available CPU cores
    print(f"Using {max_workers} processes for parallel processing.")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        for year in years:
            print(f"Processing year: {year}")
            start_year_time = time.time()

            # Group files by month
            files_by_month = {}
            for file in sorted(os.listdir(raw_data)):
                if str(year) in file and file.endswith(".zst") and file.split('.zst')[0] not in processed_files:
                    month = file.split('-')[1]  # Assuming the format includes the month, e.g., "2013-01"
                    files_by_month.setdefault(month, []).append(file)

            # Submit each month as a separate task
            for month, files in sorted(files_by_month.items()):
                futures.append(executor.submit(filter_keyword_month, year, month, files))

            # Collect results from all futures
            for future in futures:
                try:
                    month_lines, month_matched = future.result()
                    total_lines += month_lines
                    matched_lines += month_matched
                except Exception as e:
                    print(f"Error filtering by keywords: {e}")

            year_processing_time = (time.time() - start_year_time) / 60
            print(f"Completed filtering year {year} by keywords in {year_processing_time:.2f} minutes")

    print(f"Total lines processed across all files: {total_lines}")
    print(f"Total matched lines across all files: {matched_lines}")

if __name__ == "__main__":
    start_time = time.time()
    filter_keyword_parallel()
    print(f"Keyword filtering for the {args.group} social group for {args.years} was finished in {(time.time() - start_time) / 60:.2f} minutes")

# TODO: Save errors to a file rather than just posting them to the screen. The filename should contain the function and the timestamp
# TODO: Add a warning if a particular month is missing, either from the raw data, or from the output of the function.
# TODO: Incorporate anonymization