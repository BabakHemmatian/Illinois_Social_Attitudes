# import functions and objects
from cli import get_args, dir_path
from utils import parse_range, headers, groups

# import Python packages
import csv
import random
import os
import sys
import time
import datetime
import re
import torch
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification

# Extract and transform CLI arguments 
args = get_args()
years = parse_range(args.years)

# Set relevance filtering hyperparameters
batch_size = 256

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load relevance model
model_path = os.path.join(dir_path.replace("code", "models"),
                          f"filter_relevance_{args.group}_roberta_large")
tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path).to(device)
model.eval()  # Set model to evaluation mode

def get_predictions(texts, max_length=512):
    """
    Tokenize and encode a batch of texts, then return predicted labels.
    """
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(device_type="cuda"):
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            predictions = probs.argmax(dim=1).tolist()
    return predictions

# Survey the language-filtered input files and raise an error if an expected file is missing.
language_filtered_path = os.path.join(
    dir_path.replace("code", "data"),
    "data_reddit_curated", args.group, "filtered_language"
)

# Build file_list organized by year (global mode: each file is processed separately)
file_list = []
for year in years:
    for month in range(1, 13):
        filename = "RC_{}-{:02d}.csv".format(year, month)
        path_ = os.path.join(language_filtered_path, filename)
        if os.path.exists(path_):
            file_list.append(path_)
        else:
            raise Exception("Missing language-filtered file for year {}, month {}".format(year, month))

# Prepare and survey the output path for relevance filtering.
output_path = os.path.join(dir_path.replace("code", "data"),
                           "data_reddit_curated", args.group, "filtered_relevance")
os.makedirs(output_path, exist_ok=True)

##########################################
# Logging Functions
##########################################
# Report logging function: appends a timestamped message to the report file.
report_file_path = os.path.join(dir_path, f"Report_FilterRelevance.csv")
def log_report(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(report_file_path, "a", encoding="utf-8", newline="") as rep_f:
        writer = csv.writer(rep_f, delimiter="\t")
        writer.writerow([timestamp, message])
    print(f"{timestamp} - {message}")

# Error logging function: writes detailed error info to a separate file and logs a summary.
def log_error(function_name, file, line_number, line_content, error):
    error_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    resource_identifier = os.path.basename(file)
    error_filename = f"error_filter_relevance_{resource_identifier}_line{line_number}_{error_time}.txt"
    # Place error log in the same directory as the report file (or choose another path)
    error_filepath = os.path.join(dir_path, error_filename)
    with open(error_filepath, "w", encoding="utf-8") as ef:
        ef.write(f"Error in {function_name} at line {line_number} in file {file}: {error}\n")
        ef.write(f"Line content: {line_content}\n")
    log_report(f"Logged error in {error_filename} (file: {file})")
##########################################

##########################################
# Resume Functionality in Relevance Filtering
##########################################
# For each language-filtered file, add an extra column "source_row" that records the input file's row number.
# If the output file already exists, check the maximum "source_row" value and resume from rows with a greater row number.
def filter_relevance_file(file):
    # Initialize missing lines count
    missing_lines_count = 0
    missing_records_file = os.path.join(output_path, 'missing_records.csv')
    # Create missing records file with header if it does not exist.
    if not os.path.exists(missing_records_file):
        with open(missing_records_file, 'w', newline='') as missing_file:
            missing_writer = csv.writer(missing_file)
            missing_writer.writerow(['Filename', 'MissingLinesCount', 'Timestamp'])

    log_report(f"Started filtering {file} for relevance to the {args.group} social group.")
    print(f"Filtering {file} for relevance to the {args.group} social group.")
    start_time = time.time()
    
    # Build output file path using the relative part from the input file.
    relative_path = file.split("language")[1].lstrip(os.sep)
    output_file_path = os.path.join(output_path, relative_path)
    
    # New header includes extra column "source_row"
    new_headers = headers + ["source_row"]

    # Determine resume position if output file already exists.
    if os.path.exists(output_file_path):
        mode = "a"  # Append mode
        last_processed = 0
        with open(output_file_path, "r", encoding="utf-8-sig", errors="ignore") as existing_file:
            reader_existing = csv.reader(existing_file)
            rows = list(reader_existing)
            if len(rows) > 1:
                try:
                    # Compute the maximum source_row value among existing rows.
                    last_processed = max(int(row[-1]) for row in rows[1:] if row[-1].isdigit())
                except Exception as e:
                    log_report(f"Warning: Could not determine last processed row for {file}: {e}")
                    last_processed = 0
            else:
                last_processed = 0
    else:
        mode = "w"
        last_processed = 0

    with open(file, "r", encoding="utf-8-sig", errors="ignore") as input_file, \
         open(output_file_path, mode, encoding="utf-8-sig", errors="ignore", newline="") as output_file:
        
        reader = csv.reader((line.replace('\x00', '') for line in input_file))
        writer = csv.writer(output_file)
        
        # Write header if starting a new file
        if mode == "w":
            writer.writerow(new_headers)
        
        batch_texts = []
        batch_lines = []
        total_new_rows = 0  # Count only newly processed rows in this run.
        relevant_lines = []  # Accumulate rows to write in bulk

        for id_, line in enumerate(reader):
            if id_ == 0:
                continue  # Skip header row of input
            if id_ <= last_processed:
                continue  # Skip already processed rows

            try:
                if len(line) >= 3:
                    text = line[2].strip().replace("\n", " ")
                    batch_texts.append(text)
                    batch_lines.append(line)
                    total_new_rows += 1

                    # Process in batches when ready.
                    if len(batch_texts) == batch_size:
                        predictions = get_predictions(batch_texts)
                        for idx, pred in enumerate(predictions):
                            if pred == 1:  # Relevant prediction
                                # Append the current input row number (id_) as source_row
                                relevant_lines.append(batch_lines[idx] + [id_])
                        batch_texts.clear()
                        batch_lines.clear()
                        
                        if relevant_lines:
                            writer.writerows(relevant_lines)
                            relevant_lines.clear()
                else:
                    log_report(f"Skipping line {id_} in file {file}: insufficient columns ({len(line)} found)")
                    missing_lines_count += 1
            except Exception as e:
                raise Exception(f"Error filtering {file} for relevance to the {args.group} social group: {e}")

        # Process any remaining batch texts
        try:
            if batch_texts:
                predictions = get_predictions(batch_texts)
                for idx, pred in enumerate(predictions):
                    if pred == 1:
                        relevant_lines.append(batch_lines[idx] + [id_])
                if relevant_lines:
                    writer.writerows(relevant_lines)
                    relevant_lines.clear()
        except Exception as e:
            raise Exception(f"Error filtering {file} for relevance to the {args.group} social group: {e}")

    end_time = time.time()
    elapsed_minutes = (end_time - start_time) / 60
    if total_new_rows == 0:
        message = f"Finished filtering {file}: skipped (all rows previously processed)."
    else:
        message = f"Finished filtering {file} in {elapsed_minutes:.2f} minutes. New rows processed: {total_new_rows}"
    print(message)
    log_report(message)

    # Record missing lines info
    with open(missing_records_file, 'a', newline='') as missing_file:
        missing_writer = csv.writer(missing_file)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        missing_writer.writerow([file, missing_lines_count, timestamp])
    
    return total_new_rows

##########################################
# Main execution: process each file and aggregate stats
##########################################
start_time = time.time()
overall_docs = 0

# Process each file from file_list (global mode)
for file in file_list:
    overall_docs += filter_relevance_file(file)

overall_elapsed = (time.time() - start_time) / 60
print(f"Relevance filtering for the {args.group} social group for {args.years} finished in {overall_elapsed:.2f} minutes. Total new rows processed: {overall_docs}")
log_report(f"Relevance filtering for the {args.group} social group for {args.years} finished in {overall_elapsed:.2f} minutes. Total new rows processed: {overall_docs}")

##########################################
# ----- Check for missing monthly outputs -----
for year in years:
    expected_months = set(f"{m:02d}" for m in range(1, 13))
    processed_months = set()
    for file in os.listdir(output_path):
        m = re.search(r'RC_' + str(year) + r'-(\d{2})\.csv', file)
        if m:
            processed_months.add(m.group(1))
    missing = expected_months - processed_months
    if missing:
        log_report(f"Warning: For year {year}, missing output files for months: {sorted(list(missing))}")
##########################################

##########################################
# ----- Aggregate overall statistics and save final summary report -----
final_report = [
    ["Timestamp", "Social Group", "Years", "Total New Processed Rows", "Total Elapsed Time (min)"],
    [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), args.group, args.years, overall_docs, f"{overall_elapsed:.2f}"]
]
final_report_file = os.path.join(output_path, "Final_Report_FilterRelevance.csv")
with open(final_report_file, "w", encoding="utf-8", newline="") as rf:
    writer = csv.writer(rf)
    writer.writerows(final_report)
log_report(f"Final summary report saved to: {final_report_file}")
##########################################