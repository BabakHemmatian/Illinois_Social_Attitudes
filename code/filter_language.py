# Import functions and objects
from cli import get_args, dir_path
from utils import parse_range, headers, check_reqd_files, log_report, log_error

# Import Python packages
import fasttext 
import os
import csv
import time
import re
from datetime import datetime  # For timestamping
import traceback

# Increase the field size limit to handle larger fields
csv.field_size_limit(2**31 - 1)

# Extract and transform CLI arguments 
args = get_args()
years = parse_range(args.years)

# Load the fastText language identification model
model_path = os.path.join(dir_path.replace("code", "models"), "filter_language.bin")
model = fasttext.load_model(model_path)

# Define a function that applies the fastText model to a given text
def detect_language(text):
    predictions = model.predict(text)
    # The language code is returned with a prefix "__label__", which we remove
    return predictions[0][0].replace('__label__', '')

# Survey the keyword-filtered input files and raise an error if an expected file is missing
keyword_filtered_path = os.path.join(
    dir_path.replace("code", "data"),
    "data_reddit_curated", args.group, "filtered_keywords"
)
file_list = check_reqd_files(years=years, check_path=keyword_filtered_path)

# Prepare and survey the output path
output_path = os.path.join(
    dir_path.replace("code", "data"),
    "data_reddit_curated", args.group, "filtered_language"
)
os.makedirs(output_path, exist_ok=True)

# -------------------- Report logging --------------------
# The report file is used to log messages (tab-separated: timestamp and message)
report_file_path = os.path.join(output_path, "Report_filter_language.csv")

# Function for language filtering a single file
def filter_language_file(file):
    function_name = "filter_language_file"
    log_report(report_file_path, f"Started language filtering for {file}")
    try:
        # Get the relative path after "keywords/"
        relative_path = file.split("keywords" + os.sep)[1]
        # Build the full output file path in a platform-safe way
        output_file_path = os.path.join(output_path, relative_path)

        with open(file, "r", encoding='utf-8', errors='ignore') as input_file, \
             open(output_file_path, "w", encoding='utf-8', errors='ignore', newline='') as output_file:
            start_time = time.time()

            error_counter = 0
            filtered_counter = 0
            passed_counter = 0

            # Replace NUL characters with empty strings before passing to csv.reader
            reader = csv.reader((line.replace('\0', '') for line in input_file))
            writer = csv.writer(output_file)

            for id_, line in enumerate(reader):
                if id_ == 0:
                    writer.writerow(headers)
                else:
                    try:
                        line[2] = line[2].strip().replace("\n", " ")
                        if detect_language(line[2]) == 'en':
                            writer.writerow(line)
                            passed_counter += 1
                    except IndexError as e:
                        # Log the error and continue with the next line
                        log_error(function_name, file, id_ + 1, str(line), e)
                        error_counter += 1
                        continue
                    filtered_counter += 1

            elapsed = (time.time() - start_time) / 60
            msg = (f"Finished language filtering {file} in {elapsed:.2f} minutes. "
                   f"# of evaluations: {filtered_counter}, # of English posts: {passed_counter}, # of errors: {error_counter}")
            log_report(report_file_path, msg)
            # Return counters for overall statistics
            return filtered_counter, passed_counter, error_counter
    except Exception as e:
        # Capture full traceback as string
        tb_str = traceback.format_exc()
        log_report(report_file_path, f"Fatal error during processing:\n{tb_str}")

if __name__ == "__main__":
    overall_start_time = time.time()
    total_filtered = 0
    total_passed = 0
    total_errors = 0

    # Process each file and accumulate statistics
    for file in file_list:
        counters = filter_language_file(file)
        if counters:
            filtered_counter, passed_counter, error_counter = counters
            total_filtered += filtered_counter
            total_passed += passed_counter
            total_errors += error_counter

    overall_elapsed = (time.time() - overall_start_time) / 60
    log_report(report_file_path, f"Language filtering for the {args.group} social group for {args.years} finished in {overall_elapsed:.2f} minutes")
    
    # -------------------- Final summary report --------------------
    final_report = [
        ["Timestamp", "Social Group", "Years", "Total Evaluations", "Total Relevant Posts", "Total Missing Lines", "Elapsed Time (minutes)"],
        [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), args.group, args.years, total_filtered, total_passed, total_errors, f"{overall_elapsed:.2f}"]
    ]
    final_report_file = os.path.join(output_path, "Final_report_filter_language.csv")
    with open(final_report_file, "w", encoding="utf-8", newline="") as rf:
        writer = csv.writer(rf)
        writer.writerows(final_report)
    log_report(final_report_file, f"Final report saved to: {final_report_file}")
    # --------------------------------------------------------------

    # -------------------- Warning if a particular month is missing --------------------
    # Assuming that the output filename contains 'YYYY-MM', check if every year has output files for all 12 months.
    processed_months = {}
    for file in os.listdir(output_path):
        if file.endswith('.csv') and file not in [os.path.basename(final_report_file), os.path.basename(report_file_path)]:
            m = re.search(r'(\d{4})-(\d{2})', file)
            if m:
                year, month = m.groups()
                if year not in processed_months:
                    processed_months[year] = set()
                processed_months[year].add(month)
    for year in years:
        year_str = str(year)
        expected_months = set(f"{m:02d}" for m in range(1, 13))
        if year_str in processed_months:
            missing = expected_months - processed_months[year_str]
            if missing:
                log_report(report_file_path, f"Warning: For year {year_str}, missing output files for months: {sorted(list(missing))}")
        else:
            log_report(report_file_path, f"Warning: For year {year_str}, no output files found.")
