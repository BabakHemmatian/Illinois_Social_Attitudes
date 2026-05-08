### Imports

# import functions and objects
from cli import get_args, DATA_DIR, MODELS_DIR
from utils import parse_range, headers, log_report, check_reqd_files, log_error, get_last_source_row, detect_source_row

# import Python packages
import os
import csv
csv.field_size_limit(2**31 - 1) # Increase the field size limit to handle larger fields
import time
import torch
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
import datetime
import re
from pathlib import Path
import traceback
import sys
import io

### Argument Handling

# Extract and transform CLI arguments 
args = get_args()
years = parse_range(args.years)
group = args.group
type_ = args.type
batch_size = args.batchsize

# Set relevance filtering hyperparameters
max_length = 512
if group == "skin_tone" or group == "race":
    thresholding = True # If True, the model will use a confidence threshold (set below) to determine the class of a document. If False, it will always return the most probable class.
else:
    thresholding = False # thresholding is only applied to the filter_relevance model for skin_tone
threshold_class = 1 # the class that needs a probability passing the threshold (set below) to be picked as the answer. Only matters if thresholding = True.
threshold = 0.6 # The confidence threshold for the rarest class. If the model's confidence in a class is below this value, it will not return that class. Only matters if thresholding=True and the value is greater than >.50 given the two main labels. 

### Path Handling

# set path variables

# Survey the input files
if args.input:
    input_path = os.path.abspath(args.input)
else:
    input_path = os.path.join(
        DATA_DIR, "data_reddit_curated", group, type_, "filtered_language"
    )

file_list = check_reqd_files(years, input_path, type_)

# Prepare and survey the output path
if args.output:
    output_path = os.path.abspath(args.output)
else:
    output_path = os.path.join(
        DATA_DIR, "data_reddit_curated", group, type_, "filtered_relevance"
    )

os.makedirs(output_path, exist_ok=True)
# prepare the report file
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use CUDA if available
report_file_path = os.path.join(output_path, "Report_filter_relevance.csv")
log_report(report_file_path, f"Using device: {device}")

# the following portion allows each slurm process to process multiple files if files_per_job is set in the command line arguments.
array_index   = getattr(args, "array", None)
files_per_job = getattr(args, "files_per_job", 1)

if array_index is not None:
    total_files = len(file_list)
    start = array_index * files_per_job
    end   = min(start + files_per_job, total_files)

    if start >= total_files:
        msg = (
            f"No files to process for array index {array_index} "
            f"(start={start}, total_files={total_files}). Exiting."
        )
        log_report(report_file_path, msg)
        sys.exit(0)

    file_list = file_list[start:end]
    log_report(
        report_file_path,
        f"Array index {array_index}: processing files {start}..{end-1} (of {total_files})"
    )

# Load relevance model
model_path = os.path.join(MODELS_DIR,
                          f"filter_relevance_{group}")
tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path).to(device)
if torch.cuda.device_count() > 1: # if more than one GPU is available
    model = torch.nn.DataParallel(model) # parallelize
model.eval() # set model to evaluation mode

### Main functions

# Define function to infer labels for a batch of documents
@torch.no_grad()
def get_predictions(texts, threshold_class=threshold_class, threshold=threshold, thresholding=thresholding):
    # Ensure texts is a list
    if isinstance(texts, str):
        texts = [texts]

    # Tokenize batch
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(device)

    # Model inference
    outputs = model(**inputs)
    probs = outputs[0].softmax(1)  # shape: (batch_size, num_classes)

    predictions = []
    for prob in probs:
        if thresholding and prob[threshold_class] > threshold:
            predictions.append(threshold_class)
        else:
            if thresholding:
                masked_probs = prob.clone()
                masked_probs[threshold_class] = -1
                predictions.append(masked_probs.argmax().item())
            else:
                predictions.append(prob.argmax().item())

    return predictions

# Propagate `source_row` from input when present; generate it from the input
# row index when absent (legacy filter_language outputs).
def filter_relevance_file(file):
    function_name = "filter_relevance_file"
    log_report(
        report_file_path,
        f"Started relevance filtering for {Path(file).name} for relevance to the {group} social group."
    )

    try:
        start_time = time.time()
        output_file_path = os.path.join(output_path, Path(file).name)

        error_counter = 0
        evaluated_counter = 0
        passed_counter = 0

        # Determine resume position if output file already exists.
        last_processed = get_last_source_row(
            output_file_path,
            report_file_path=report_file_path,
            file_for_log=file,
        )
        mode = "a" if last_processed >= 0 else "w"

        with open(file, "r", encoding="utf-8-sig", errors="ignore") as input_file, \
             open(output_file_path, mode, encoding="utf-8-sig", errors="ignore", newline="") as output_file:

            reader = csv.reader((line.replace('\x00', '') for line in input_file))
            writer = csv.writer(output_file)

            try:
                in_header = next(reader)
            except StopIteration:
                return 0, 0, 0

            src_idx_in, has_input_source_row = detect_source_row(in_header)
            keywords_idx = in_header.index("matched patterns") if "matched patterns" in in_header else headers.index("matched patterns")

            # Output header: propagate input header verbatim if it already has
            # source_row; otherwise append source_row so downstream resources
            # can resume.
            if mode == "w":
                if has_input_source_row:
                    writer.writerow(in_header)
                else:
                    writer.writerow(list(in_header) + ["source_row"])

            batch_lines = []
            relevant_lines = []

            def flush_batch(batch_lines, relevant_lines):
                nonlocal passed_counter
                if not batch_lines:
                    return

                texts = [l[2].strip().replace("\n", " ") for l in batch_lines]
                predictions = get_predictions(texts)

                for idx, pred in enumerate(predictions):
                    if pred == 1:
                        row = batch_lines[idx]
                        row[keywords_idx] = ",".join(set(row[keywords_idx].split(",")))
                        relevant_lines.append(row)
                        passed_counter += 1

                if relevant_lines:
                    writer.writerows(relevant_lines)
                    relevant_lines.clear()

            for id_, line in enumerate(reader, start=1):
                try:
                    if len(line) < 3:
                        raise IndexError(f"insufficient columns ({len(line)} found)")

                    # Determine source_row for this row (for resume + output).
                    if has_input_source_row:
                        src_val = line[src_idx_in].strip() if len(line) > src_idx_in else ""
                        try:
                            src_num = int(src_val)
                        except ValueError:
                            src_num = None
                    else:
                        src_num = id_

                    # Resume: skip already-processed rows.
                    if src_num is not None and src_num <= last_processed:
                        continue

                    # Append source_row only if input lacked it; otherwise the
                    # input row already carries it through.
                    if has_input_source_row:
                        batch_lines.append(line)
                    else:
                        batch_lines.append(list(line) + [id_])
                    evaluated_counter += 1

                    if len(batch_lines) == batch_size:
                        flush_batch(batch_lines, relevant_lines)
                        batch_lines.clear()

                except Exception as e:
                    log_error(
                        function_name,
                        file,
                        id_ + 1,
                        str(line),
                        e,
                        report_file_path=report_file_path,
                        output_path=output_path,
                    )
                    error_counter += 1
                    continue

            # Process remainder
            if batch_lines:
                try:
                    flush_batch(batch_lines, relevant_lines)
                    batch_lines.clear()
                except Exception as e:
                    log_error(
                        function_name,
                        file,
                        int(batch_lines[0][-1]) if batch_lines else -1,
                        f"final batch ({len(batch_lines)} buffered rows)",
                        e,
                        report_file_path=report_file_path,
                        output_path=output_path,
                    )
                    error_counter += len(batch_lines)

        elapsed_minutes = (time.time() - start_time) / 60
        log_report(
            report_file_path,
            f"Finished relevance filtering {Path(file).name} in {elapsed_minutes:.2f} minutes. "
            f"# of evaluations: {evaluated_counter}, # of relevant posts: {passed_counter}, # of errors: {error_counter}"
        )

        return evaluated_counter, passed_counter, error_counter

    except Exception:
        tb_str = traceback.format_exc()
        log_report(
            report_file_path,
            f"Fatal error during relevance filtering for {Path(file).name}:\n{tb_str}"
        )
        return None

### Main Execution

if __name__ == "__main__":
    target_files = list(file_list)

    overall_start_time = time.time()
    total_evaluated = 0
    total_passed = 0
    total_errors = 0

    for file in file_list:
        counters = filter_relevance_file(file)
        if counters:
            evaluated_counter, passed_counter, error_counter = counters
            total_evaluated += evaluated_counter
            total_passed += passed_counter
            total_errors += error_counter

    overall_elapsed = (time.time() - overall_start_time) / 60

    if array_index is None:
        scope_msg = f"{args.years}"
    else:
        scope_msg = f"{args.years} (array index {array_index})"

    log_report(
        report_file_path,
        f"Relevance filtering for the {group} social group for {scope_msg} finished in {overall_elapsed:.2f} minutes"
    )

    # Check for missing outputs
    processed_months = {}
    for file in os.listdir(output_path):
        if file.endswith(".csv") and file not in [
            os.path.basename(report_file_path),
            "Final_Report_FilterRelevance.csv",
        ]:
            m = re.search(r"(\d{4})-(\d{2})", file)
            if m:
                year, month = m.groups()
                processed_months.setdefault(year, set()).add(month)

    if array_index is None: # on local runs
        for year in years:
            year_str = str(year)
            expected_months = set(f"{m:02d}" for m in range(1, 13))
            actual_months = processed_months.get(year_str, set())
            missing = expected_months - actual_months
            if missing:
                log_report(
                    report_file_path,
                    f"Warning: For year {year_str}, missing output files for months: {sorted(list(missing))}"
                )

        final_report = [
            ["Timestamp", "Social Group", "Years", "Total Evaluations", "Total Relevant Posts", "Total Errors", "Elapsed Time (minutes)"],
            [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), group, args.years, total_evaluated, total_passed, total_errors, f"{overall_elapsed:.2f}"]
        ]
        final_report_file = os.path.join(output_path, "Final_Report_FilterRelevance.csv")
        with open(final_report_file, "w", encoding="utf-8", newline="") as rf:
            writer = csv.writer(rf)
            writer.writerows(final_report)
        log_report(report_file_path, f"Final report saved to: {final_report_file}")

    else: # if running on a cluster
        expected_by_year = {}
        for file in target_files:
            m = re.search(r"(\d{4})-(\d{2})", Path(file).name)
            if m:
                year, month = m.groups()
                expected_by_year.setdefault(year, set()).add(month)

        for year_str, expected_months in expected_by_year.items():
            actual_months = processed_months.get(year_str, set())
            missing = expected_months - actual_months
            if missing:
                log_report(
                    report_file_path,
                    f"Warning: For array index {array_index}, year {year_str}, missing output files for months: {sorted(list(missing))}"
                )
