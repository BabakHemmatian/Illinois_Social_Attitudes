# import functions and objects
from cli import get_args, dir_path
from utils import parse_range, headers, groups

# import Python packages
import os
import csv
import time
import torch
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
import datetime
import sys
import random
import re


# Extract and transform CLI arguments 
args = get_args()
years = parse_range(args.years)

# Set relevance filtering hyperparameters
batch_size = 320

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
        with torch.cuda.amp.autocast():
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

# Prepare and survey the output path for relevance filtering
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
# For each language-filtered file, we add an extra column "source_row" to record the input file row number.
# If the output file already exists, we check the last processed row number and resume from there.
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
        mode = "a"  # Append
        last_processed = 0
        with open(output_file_path, "r", encoding="utf-8-sig", errors="ignore") as existing_file:
            reader_existing = csv.reader(existing_file)
            rows = list(reader_existing)
            if len(rows) > 1:
                try:
                    last_processed = int(rows[-1][-1])
                except:
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
        total_lines = 0
        relevant_lines = []  # Rows to write in bulk

        for id_, line in enumerate(reader):
            if id_ == 0:
                continue  # Skip header row of input
            if id_ <= last_processed:
                continue  # Resume: skip already processed rows

            try:
                if len(line) >= 3:
                    text = line[2].strip().replace("\n", " ")
                    batch_texts.append(text)
                    batch_lines.append(line)
                    total_lines += 1

                    # Process in batches
                    if len(batch_texts) == batch_size:
                        predictions = get_predictions(batch_texts)
                        for idx, pred in enumerate(predictions):
                            if pred == 1:  # Relevant
                                # Append the source row number to the row data
                                relevant_lines.append(batch_lines[idx] + [id_])
                        batch_texts.clear()
                        batch_lines.clear()
                        
                        if relevant_lines:
                            writer.writerows(relevant_lines)
                            relevant_lines.clear()
                else:
                    print(f"Skipping line {id_}: insufficient columns ({len(line)} found)")
                    missing_lines_count += 1
            except Exception as e:
                raise Exception(f"Error filtering {file} for relevance to the {args.group} social group: {e}")

        # Process any remaining texts in the final batch
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
    print(f"Finished filtering {file} in {elapsed_minutes:.2f} minutes. Processed rows: {total_lines}")
    log_report(f"Finished filtering {file} in {elapsed_minutes:.2f} minutes. Processed rows: {total_lines}")

    # Record missing lines info
    with open(missing_records_file, 'a', newline='') as missing_file:
        missing_writer = csv.writer(missing_file)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        missing_writer.writerow([file, missing_lines_count, timestamp])
    
    return total_lines

##########################################
# Main execution: process each file and aggregate stats
##########################################
start_time = time.time()
overall_docs = 0

# Process each file from the file_list (global mode)
for file in file_list:
    overall_docs += filter_relevance_file(file)

overall_elapsed = (time.time() - start_time) / 60
print(f"Relevance filtering for the {args.group} social group for {args.years} finished in {overall_elapsed:.2f} minutes. Total processed rows: {overall_docs}")
log_report(f"Relevance filtering for the {args.group} social group for {args.years} finished in {overall_elapsed:.2f} minutes. Total processed rows: {overall_docs}")

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
    ["Timestamp", "Social Group", "Years", "Total Processed Rows", "Total Elapsed Time (min)"],
    [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), args.group, args.years, overall_docs, f"{overall_elapsed:.2f}"]
]
final_report_file = os.path.join(output_path, "Final_Report_FilterRelevance.csv")
with open(final_report_file, "w", encoding="utf-8", newline="") as rf:
    writer = csv.writer(rf)
    writer.writerows(final_report)
log_report(f"Final summary report saved to: {final_report_file}")
##########################################