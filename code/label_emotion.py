# import functions and objects
from cli import get_args, dir_path
from utils import parse_range, headers, log_report

# import Python packages
import os
import csv
import time
import torch
import datetime
import re
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, DistilBertForSequenceClassification, DistilBertTokenizerFast
from pathlib import Path

# Extract and transform CLI arguments 
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

generalization_labeled_path = DATA_DIR / "data_reddit_curated" / group / "labeled_sentiment"
output_path = DATA_DIR / "data_reddit_curated" / group / "labeled_emotion"
output_path.mkdir(parents=True, exist_ok=True)

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# prepare the report file
report_file_path = os.path.join(dir_path, f"report_label_emotion.csv")
log_report(report_file_path,f"Using device: {device}")

# Load emotion models
model1_path = os.path.join(MODELS_DIR,
                          "label_emotion_1")
model2_path = os.path.join(MODELS_DIR,
                          "label_emotion_2")
model3_path = os.path.join(MODELS_DIR,
                          "label_emotion_3")
tokenizer1 = RobertaTokenizerFast.from_pretrained(model1_path)
tokenizer2 = DistilBertTokenizerFast.from_pretrained(model2_path)
# NOTE: Model 3 uses the same tokenizer as model 1.
model1 = RobertaForSequenceClassification.from_pretrained(model1_path,use_safetensors=True).to(device)
model2 = DistilBertForSequenceClassification.from_pretrained(model2_path,use_safetensors=True).to(device)
model3 = RobertaForSequenceClassification.from_pretrained(model3_path,use_safetensors=True).to(device)

if torch.cuda.device_count() > 1: # if more than one GPU is available
    model1 = torch.nn.DataParallel(model1) # parallelize
    model2 = torch.nn.DataParallel(model2) # parallelize
    model3 = torch.nn.DataParallel(model3) # parallelize
model1.eval() # set model to evaluation mode
model2.eval() # set model to evaluation mode
model3.eval() # set model to evaluation mode

def get_predictions(texts, tokenizer=None, model=None, max_length=512):
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
    
    # this part uses mixed-precision classification for faster performance. 
    with torch.no_grad():
        with torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu"):
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            predictions = probs.argmax(dim=1).tolist()

    probs = torch.softmax(probs,dim=1).tolist() # extract the probabilities from output layer activations

    return predictions, probs


# Build file_list organized by year and raise an error if an expected file is missing 
file_list = []
for year in years:
    for month in range(1, 13):
        filename = "RC_{}-{:02d}.csv".format(year, month)
        path_ = os.path.join(generalization_labeled_path, filename)
        if os.path.exists(path_):
            file_list.append(path_)
        else:
            raise Exception("Missing generalization-labeled file for the {} social group from year {}, month {}".format(group, year, month))

# generates labels for a month's worth of documents. Resumes labeling if it finds incomplete output files. 
def label_emotion_file(file):
    
    # setup & logging
    missing_lines_count = 0
    log_report(report_file_path, f"Started labeling {Path(file).name} from the {group} social group for emotion.")
    start_time = time.time()

    # Mirror input directory structure for output path
    relative_path = Path(file).relative_to(generalization_labeled_path)
    output_file_path = output_path / relative_path
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine resume position by reading 'source_row' column by name
    mode = "w"
    last_processed = -1
    if os.path.exists(output_file_path):
        with open(output_file_path, "r", encoding="utf-8-sig", errors="ignore") as existing_file:
            out_rows = list(csv.reader(existing_file))
            if len(out_rows) > 1:
                out_header = out_rows[0]
                try:
                    src_idx_out = out_header.index("source_row")
                    # find the last non-empty source_row from the bottom
                    for r in reversed(out_rows[1:]):
                        if len(r) > src_idx_out and r[src_idx_out].strip():
                            last_processed = int(r[src_idx_out])
                            break
                    mode = "a"
                except ValueError:
                    # Output header missing source_row; safest is to rewrite from scratch.
                    mode = "w"
                    last_processed = -1

    # Open input & output
    with open(file, "r", encoding="utf-8-sig", errors="ignore") as input_file, \
         open(output_file_path, mode, encoding="utf-8", errors="ignore", newline="") as output_file:

        reader = csv.reader((line.replace('\x00', '') for line in input_file))
        writer = csv.writer(output_file)

        # Read input header (this is the output of label_generalization.py)
        try:
            in_header = next(reader)
        except StopIteration:
            return 0  # empty input

        # Locate 'source_row' in the input header
        try:
            src_idx_in = in_header.index("source_row")
        except ValueError:
            raise RuntimeError("Input file is missing required 'source_row' column.")

        # If starting a new output file, write headers
        if mode == "w":
            emotion_headers = [
                "1_anger","1_disgust","1_fear","1_joy","1_neutral","1_sadness","1_surprise",
                "2_sadness","2_joy","2_love","2_anger","2_fear","2_surprise",
                "3_neutral","3_joy","3_surprise","3_anger","3_sadness","3_disgust","3_fear"
            ]
            new_headers = in_header + emotion_headers
            writer.writerow(new_headers)

        batch_lines = []
        relevant_lines = []
        total_lines = 0

        # Iterate rows
        for row_idx, line in enumerate(reader, start=1):  # start=1 since we consumed header
            # Skip bad rows (need at least 3 cols because we read text from line[2])
            if len(line) < 3:
                log_report(report_file_path, f"Skipping line {row_idx}: insufficient columns ({len(line)} found)")
                missing_lines_count += 1
                continue

            # Resume: skip if already processed based on input's source_row
            src_val = line[src_idx_in].strip()
            src_num = int(src_val) if src_val.isdigit() else None
            if src_num is not None and src_num <= last_processed:
                continue

            batch_lines.append(line)
            total_lines += 1

            if len(batch_lines) == batch_size:
                texts = [l[2].strip().replace("\n", " ") for l in batch_lines]

                # Accumulate probabilities from the three models
                for i in range(3):
                    if i == 0:
                        _, probs = get_predictions(texts, tokenizer=tokenizer1, model=model1)
                    elif i == 1:
                        _, probs = get_predictions(texts, tokenizer=tokenizer2, model=model2)
                    else:
                        _, probs = get_predictions(texts, tokenizer=tokenizer1, model=model3)

                    # Append probabilities to each row in-place
                    for idx in range(len(batch_lines)):
                        batch_lines[idx] = batch_lines[idx] + probs[idx]

                relevant_lines.extend(batch_lines)
                if relevant_lines:
                    writer.writerows(relevant_lines)
                    relevant_lines.clear()
                batch_lines.clear()

        # Final flush
        if batch_lines:
            texts = [l[2].strip().replace("\n", " ") for l in batch_lines]

            for i in range(3):
                if i == 0:
                    _, probs = get_predictions(texts, tokenizer=tokenizer1, model=model1)
                elif i == 1:
                    _, probs = get_predictions(texts, tokenizer=tokenizer2, model=model2)
                else:
                    _, probs = get_predictions(texts, tokenizer=tokenizer1, model=model3)

                for idx in range(len(batch_lines)):
                    batch_lines[idx] = batch_lines[idx] + probs[idx]

            relevant_lines.extend(batch_lines)
            if relevant_lines:
                writer.writerows(relevant_lines)

    # generate processing report
    elapsed_minutes = (time.time() - start_time) / 60
    log_report(report_file_path, f"Finished labeling emotion for the {group} social group in {Path(file).name} within {elapsed_minutes:.2f} minutes. Processed rows: {total_lines}")

    # Create missing_records.csv only if there were missing lines
    if missing_lines_count > 0:
        missing_records_file = os.path.join(output_path, 'missing_records.csv')
        need_header = not os.path.exists(missing_records_file)
        with open(missing_records_file, 'a', newline='', encoding='utf-8') as missing_file:
            missing_writer = csv.writer(missing_file)
            if need_header:
                missing_writer.writerow(['Filename', 'MissingLinesCount', 'Timestamp'])
            timestamp = datetime.datetime.now().isoformat(timespec="seconds")
            missing_writer.writerow([str(file), missing_lines_count, timestamp])

    return total_lines

##########################################
# Main execution: process each file and aggregate stats
##########################################
start_time = time.time()

overall_docs = 0

if args.array is not None: # for batch processing
    overall_docs += label_emotion_file(file_list[array])

else: # for sequential processing
    for file in file_list:        
        overall_docs += label_emotion_file(file)

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
    log_report(report_file_path, f"Emotion labeling for the {group} social group for {args.years} finished in {overall_elapsed:.2f} minutes. Total processed rows: {overall_docs}")

    ##########################################
    # ----- Aggregate overall statistics and save final summary report -----
    final_report = [
        ["Timestamp", "Social Group", "Years", "Total Processed Rows", "Total Elapsed Time (min)"],
        [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), group, args.years, overall_docs, f"{overall_elapsed:.2f}"]
    ]
    final_report_file = os.path.join(output_path, "final_report_label_emotion.csv")
    with open(final_report_file, "a+", encoding="utf-8", newline="") as rf:
        writer = csv.writer(rf)
        writer.writerows(final_report)
    log_report(report_file_path, f"Final summary report saved to: {final_report_file}")
    ##########################################
