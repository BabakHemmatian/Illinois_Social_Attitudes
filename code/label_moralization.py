# import functions and objects
from cli import get_args, dir_path
from utils import parse_range, headers, log_report

# import Python packages
import os, sys
import csv
import time
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
import datetime
import re
from pathlib import Path

# Extract and transform CLI arguments 
args = get_args()
years = parse_range(args.years)
group = args.group
batch_size = args.batchsize
if args.array is not None:
    array = args.array

# set path variables
CODE_DIR = Path(__file__).resolve().parent         
PROJECT_ROOT = CODE_DIR.parent                     
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

model_path = MODELS_DIR / "label_moralization"
keywords_adv_filtered_path = DATA_DIR / "data_reddit_curated" / group / "filtered_keywords_adv"
output_path = DATA_DIR / "data_reddit_curated" / group / "labeled_moralization"
output_path.mkdir(parents=True, exist_ok=True)

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# prepare the report file
report_file_path = os.path.join(dir_path, f"report_label_moralization.csv")
log_report(report_file_path,f"Using device: {device}")

# Load moralization model
tokenizer = BertTokenizerFast.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(
    model_path,
    device_map=None,   # or "auto" if you want
    use_safetensors=True
).to(device)
if torch.cuda.device_count() > 1: # if more than one GPU is available
    model = torch.nn.DataParallel(model) # parallelize
model.eval() # set model to evaluation mode

# Define function to infer labels for a batch of documents
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
        with torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu"):
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            predictions = probs.argmax(dim=1).tolist()
    return predictions

# Build file_list organized by year and raise an error if an expected file is missing 
file_list = []
for year in years:
    for month in range(1, 13):
        filename = "RC_{}-{:02d}.csv".format(year, month)
        path_ = os.path.join(keywords_adv_filtered_path, filename)
        if os.path.exists(path_):
            file_list.append(path_)
        else:
            raise Exception("Missing relevance-filtered file for the {} social group from year {}, month {}".format(group, year, month))

def label_moralization_file(file):

    missing_lines_count = 0
    log_report(report_file_path, f"Started labeling {Path(file).name} from the {group} social group for moralization.")
    start_time = time.time()

    # Build output file path using the relative part from the input file.
    relative_path = Path(file).relative_to(keywords_adv_filtered_path)
    output_file_path = output_path / relative_path
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    # If the output file already exists, we check the last processed row number and resume from there.
    mode = "w"
    last_processed = -1
    if os.path.exists(output_file_path):
        with open(output_file_path, "r", encoding="utf-8-sig", errors="ignore") as existing_file:
            rows = list(csv.reader(existing_file))
            if len(rows) > 1:
                header_out = rows[0]
                try:
                    src_idx_out = header_out.index("source_row")
                    # find the last non-empty source_row from the bottom
                    for r in reversed(rows[1:]):
                        if len(r) > src_idx_out and r[src_idx_out].strip():
                            last_processed = int(r[src_idx_out])
                            break
                    mode = "a"
                except ValueError:
                    # Output exists but header lacks source_row? Safer to overwrite.
                    mode = "w"
                    last_processed = -1

    with open(file, "r", encoding="utf-8-sig", errors="ignore") as input_file, \
         open(output_file_path, mode, encoding="utf-8-sig", errors="ignore", newline="") as output_file:

        reader = csv.reader((line.replace('\x00', '') for line in input_file))
        writer = csv.writer(output_file)

        # Read input header and locate source_row column by name
        try:
            in_header = next(reader)
        except StopIteration:
            return 0  # empty input

        try:
            src_idx_in = in_header.index("source_row")
        except ValueError:
            raise RuntimeError("Input file is missing required 'source_row' column.")

        # If starting a new output, write header = input header + Moralization
        if mode == "w":
            new_headers = in_header + ["Moralization"]
            writer.writerow(new_headers)

        batch_lines = []
        relevant_lines = []
        total_lines = 0

        for _, line in enumerate(reader):
            # skip obvious bad rows
            if len(line) < 3:
                missing_lines_count += 1
                continue

            # Use the INPUT row's source_row to decide resume/skip
            src_val = line[src_idx_in].strip()
            src_num = int(src_val) if src_val.isdigit() else None
            if src_num is not None and src_num <= last_processed:
                continue

            batch_lines.append(line)
            total_lines += 1

            # get labels for the full batch
            if len(batch_lines) == batch_size:
                texts = [l[2].strip().replace("\n"," ") for l in batch_lines]
                predictions = get_predictions(texts)
                for idx, pred in enumerate(predictions):
                    row_out = batch_lines[idx] + ["Moralized" if pred else "Non-Moralized"]
                    relevant_lines.append(row_out)
                if relevant_lines:
                    writer.writerows(relevant_lines)
                    relevant_lines.clear()
                batch_lines.clear()

        # Flush final batch
        if batch_lines:
            texts = [l[2].strip().replace("\n"," ") for l in batch_lines]
            predictions = get_predictions(texts)
            for idx, pred in enumerate(predictions):
                row_out = batch_lines[idx] + ["Moralized" if pred else "Non-Moralized"]
                relevant_lines.append(row_out)
            if relevant_lines:
                writer.writerows(relevant_lines)
                relevant_lines.clear()

    # generate processing report
    end_time = time.time()
    elapsed_minutes = (end_time - start_time) / 60
    log_report(report_file_path, f"Finished labeling moralization for the {group} social group in {Path(file).name} within {elapsed_minutes:.2f} minutes. Processed rows: {total_lines}")

    if missing_lines_count:
        missing_records_file = os.path.join(output_path, 'missing_records.csv')
        need_header = not os.path.exists(missing_records_file)
        with open(missing_records_file, 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            if need_header:
                w.writerow(['Filename', 'MissingLinesCount', 'Timestamp'])
            w.writerow([str(file), missing_lines_count, datetime.datetime.now().isoformat(timespec="seconds")])

    return total_lines

##########################################
# Main execution: process each file and aggregate stats
##########################################
start_time = time.time()

overall_docs = 0

# Process each file from the file_list (global mode)
if args.array is not None: # for batch processing
    overall_docs += label_moralization_file(file_list[array])

else: # for sequential processing
    for file in file_list:        
        overall_docs += label_moralization_file(file)

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
            log_report(report_file_path, f"Warning: For year {year}, missing output files for months: {sorted(list(missing))}")
    ##########################################

    overall_elapsed = (time.time() - start_time) / 60
    log_report(report_file_path, f"Moralization labeling for the {group} social group for {args.years} finished in {overall_elapsed:.2f} minutes. Total processed rows: {overall_docs}")

    ##########################################
    # ----- Aggregate overall statistics and save final summary report -----
    final_report = [
        ["Timestamp", "Social Group", "Years", "Total Processed Rows", "Total Elapsed Time (min)"],
        [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), group, args.years, overall_docs, f"{overall_elapsed:.2f}"]
    ]
    final_report_file = os.path.join(output_path, "final_report_label_moralization.csv")
    with open(final_report_file, "a+", encoding="utf-8", newline="") as rf:
        writer = csv.writer(rf)
        writer.writerows(final_report)
    log_report(report_file_path, f"Final summary report saved to: {final_report_file}")
    ##########################################
