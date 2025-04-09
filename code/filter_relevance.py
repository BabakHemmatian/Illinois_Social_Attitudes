# import functions and objects
from cli import get_args, dir_path
from utils import parse_range, headers, check_reqd_files

# import Python packages
import os
import csv
import time
import torch
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
import datetime
import sys
import re

# Extract and transform CLI arguments 
args = get_args()
years = parse_range(args.years)

# Set relevance filtering hyperparameters
batch_size = 320

# Use MPS if available on Mac
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load relevance model
project_root = os.path.dirname(dir_path)
model_path = os.path.join(project_root, "models", f"filter_relevance_{args.group}_roberta_large")
tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path).to(device)
model.eval()  # Set model to evaluation mode

def get_predictions(texts, max_length=512):
    # Tokenize and encode the batch of texts
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        # Use autocast only if the device supports it (e.g., CUDA). Otherwise, use the default mode.
        if device.type == "cuda":
            with torch.amp.autocast("cuda"):
                outputs = model(**inputs)
        else:
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        predictions = probs.argmax(dim=1).tolist()  # List of predictions
    return predictions

# Survey the language-filtered input files and raise an error if an expected file is missing
language_filtered_path = os.path.join(
    project_root,
    "data", "data_reddit_curated", args.group, "filtered_language"
)
file_list = check_reqd_files(years=years, check_path=language_filtered_path)

# Prepare and survey the output path
output_path = os.path.join(
    project_root,
    "data", "data_reddit_curated", args.group, "filtered_relevance"
)
os.makedirs(output_path, exist_ok=True)

# -------------------- Error logging function --------------------
def log_error(function_name, file, line_number, line_content, error):
    """
    Save error details to a file. The filename follows the pattern:
    error_filter_relevance_<resource>_line<line>_<timestamp>.txt
    """
    error_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    resource_identifier = os.path.basename(file)
    error_filename = f"error_filter_relevance_{resource_identifier}_line{line_number}_{error_time}.txt"
    error_filepath = os.path.join(output_path, error_filename)
    with open(error_filepath, "w", encoding="utf-8") as ef:
        ef.write(f"Error in {function_name} at line {line_number}: {error}\n")
        ef.write(f"Line content: {line_content}\n")
    print(f"Logged error in {error_filename}")

# -------------------- Report logging function --------------------
report_file_path = os.path.join(output_path, "Report_FilterRelevance.csv")
def log_report(message):
    """
    Append a log entry with the current timestamp and message to the report file.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(report_file_path, "a", encoding="utf-8", newline="") as rep_f:
        writer = csv.writer(rep_f, delimiter="\t")
        writer.writerow([timestamp, message])
    print(f"{timestamp} - {message}")
# ------------------------------------------------------------------

def filter_relevance_file(file):
    # Initialize missing lines count
    missing_lines_count = 0
    missing_records_file = os.path.join(output_path, 'missing_records.csv')

    # Check if the missing records file exists; if not, create it with a header
    if not os.path.exists(missing_records_file):
        with open(missing_records_file, 'w', newline='') as missing_file:
            missing_writer = csv.writer(missing_file)
            missing_writer.writerow(['Filename', 'MissingLinesCount', 'Timestamp'])

    print(f"Filtering {file} for relevance to the {args.group} social group.")
    log_report(f"Started filtering {file} for relevance to the {args.group} social group.")
    start_time = time.time()
    
    # Use os.path.join to build output file path:
    # Extract the relative path after "language" from the input file path.
    relative_path = file.split("language")[1]
    # Remove any leading separators to avoid duplication
    relative_path = relative_path.lstrip(os.sep)
    output_file_path = os.path.join(output_path, relative_path)

    with open(file, "r", encoding="utf-8-sig", errors="ignore") as input_file, \
         open(output_file_path, "w", encoding="utf-8-sig", errors="ignore", newline="") as output_file:
        
        reader = csv.reader((line.replace('\x00', '') for line in input_file))
        writer = csv.writer(output_file)

        batch_texts = []
        batch_lines = []
        relevant_count = 0
        total_lines = 0
        relevant_lines = []  # Collect relevant lines to write in bulk

        for id_, line in enumerate(reader):
            if id_ == 0:
                writer.writerow(headers)
            else:
                if id_ % 100000 == 0:
                    print(f"{id_} documents filtered in {file}")
                try:
                    if len(line) >= 3:
                        text = line[2].strip().replace("\n", " ")
                        batch_texts.append(text)
                        batch_lines.append(line)
                        total_lines += 1

                        # If batch is full, process it
                        if len(batch_texts) == batch_size:
                            predictions = get_predictions(batch_texts)
                            for idx, pred in enumerate(predictions):
                                if pred == 1:  # Relevant
                                    relevant_count += 1
                                    relevant_lines.append(batch_lines[idx])
                            # Clear the batches
                            batch_texts.clear()
                            batch_lines.clear()
                            
                            # Write relevant lines to output file in bulk
                            if relevant_lines:
                                writer.writerows(relevant_lines)
                                relevant_lines.clear()
                    else:
                        # Log error for insufficient columns and count as missing
                        log_error("filter_relevance_file", file, id_ + 1, str(line), "Insufficient columns")
                        print(f"Skipping line {id_}: insufficient columns ({len(line)} found)")
                        missing_lines_count += 1
                except Exception as e:
                    log_error("filter_relevance_file", file, id_ + 1, str(line), e)
                    missing_lines_count += 1
                    continue

        # Process any remaining texts in the last batch
        try:
            if batch_texts:
                predictions = get_predictions(batch_texts)
                for idx, pred in enumerate(predictions):
                    if pred == 1:
                        relevant_lines.append(batch_lines[idx])
                # Write any remaining relevant lines
                if relevant_lines:
                    writer.writerows(relevant_lines)
                    relevant_lines.clear()
        except Exception as e:
            log_error("filter_relevance_file", file, "final_batch", "Remaining batch", e)

    end_time = time.time()
    elapsed_minutes = (end_time - start_time) / 60
    print(f"Finished filtering {file} for relevance to the {args.group} social group in {elapsed_minutes:.2f} minutes. # lines assessed: {total_lines}, relevant lines: {relevant_count}")
    log_report(f"Finished filtering {file} for relevance in {elapsed_minutes:.2f} minutes. Total lines: {total_lines}, relevant lines: {relevant_count}, missing lines: {missing_lines_count}")

    # Record the missing lines count in the missing records CSV file
    with open(missing_records_file, 'a', newline='') as missing_file:
        missing_writer = csv.writer(missing_file)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        missing_writer.writerow([file, missing_lines_count, timestamp])
    
    return total_lines, relevant_count, missing_lines_count

if __name__ == "__main__":
    overall_start_time = time.time()
    overall_total_lines = 0
    overall_relevant_count = 0
    overall_missing_lines = 0

    for file in file_list:
        total_lines, rel_count, missing_count = filter_relevance_file(file)
        overall_total_lines += total_lines
        overall_relevant_count += rel_count
        overall_missing_lines += missing_count

    overall_elapsed = (time.time() - overall_start_time) / 60
    print(f"Relevance filtering for the {args.group} social group for {args.years} was finished in {overall_elapsed:.2f} minutes.")
    log_report(f"Relevance filtering for the {args.group} social group for {args.years} finished in {overall_elapsed:.2f} minutes.")

    # -------------------- Final summary report --------------------
    final_report = [
        ["Timestamp", "Social Group", "Years", "Total Evaluations", "Total Relevant Posts", "Total Missing Lines", "Elapsed Time (minutes)"],
        [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), args.group, args.years, overall_total_lines, overall_relevant_count, overall_missing_lines, f"{overall_elapsed:.2f}"]
    ]
    final_report_file = os.path.join(output_path, "Final_Report_FilterRelevance.csv")
    with open(final_report_file, "w", encoding="utf-8", newline="") as rf:
        writer = csv.writer(rf)
        writer.writerows(final_report)
    log_report(f"Final report saved to: {final_report_file}")
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
                log_report(f"Warning: For year {year_str}, missing output files for months: {sorted(list(missing))}")
        else:
            log_report(f"Warning: For year {year_str}, no output files found.")