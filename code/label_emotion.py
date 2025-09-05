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
import sys

# Extract and transform CLI arguments 
args = get_args()
years = parse_range(args.years)
group = args.group

# Set emotion labeling hyperparameters
batch_size = 1000

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# prepare the report file
report_file_path = os.path.join(dir_path, f"report_label_emotion.csv")
log_report(report_file_path,f"Using device: {device}")

# Load emotion models
model1_path = os.path.join(dir_path.replace("code", "models"),
                          f"label_emotion_1")
model2_path = os.path.join(dir_path.replace("code", "models"),
                          f"label_emotion_2")
model3_path = os.path.join(dir_path.replace("code", "models"),
                          f"label_emotion_3")
tokenizer1 = RobertaTokenizerFast.from_pretrained(model1_path)
tokenizer2 = DistilBertTokenizerFast.from_pretrained(model2_path)
# NOTE: Model 3 uses the same tokenizer as model 1.
model1 = RobertaForSequenceClassification.from_pretrained(
    model1_path,
    use_safetensors=True).to(device)
model2 = DistilBertForSequenceClassification.from_pretrained(
    model2_path,
    use_safetensors=True).to(device)
model3 = RobertaForSequenceClassification.from_pretrained(
    model3_path,
    use_safetensors=True).to(device)

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

# Survey the generalization-labeled input files 
generalization_labeled_path = os.path.join(
    dir_path.replace("code", "data"),
    "data_reddit_curated", group, "labeled_generalization"
)

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

# Prepare and survey the output path for emotion labeling
output_path = os.path.join(dir_path.replace("code", "data"),
                           "data_reddit_curated", group, "labeled_emotion")
os.makedirs(output_path, exist_ok=True)

# If the output file already exists, we check the last processed row number and resume from there.
def label_emotion_file(file):
    # Initialize missing lines count
    missing_lines_count = 0
    missing_records_file = os.path.join(output_path, 'missing_records.csv')
    # Create missing records file with header if it does not exist.
    if not os.path.exists(missing_records_file):
        with open(missing_records_file, 'w', newline='') as missing_file:
            missing_writer = csv.writer(missing_file)
            missing_writer.writerow(['Filename', 'MissingLinesCount', 'Timestamp'])

    log_report(report_file_path, f"Started labeling {file} from the {group} social group for emotion.")
    start_time = time.time()
    
    # Build output file path using the relative part from the input file.
    relative_path = file.split("generalization")[1].lstrip(os.sep)
    output_file_path = os.path.join(output_path, relative_path)
    
    # Determine resume position if output file already exists.
    if os.path.exists(output_file_path):
        mode = "a"  # Append
        last_processed = 0
        with open(output_file_path, "r", encoding="utf-8-sig", errors="ignore") as existing_file:
            reader_existing = csv.reader(existing_file)
            rows = list(reader_existing)
            if len(rows) > 1:
                try:
                    last_processed = int(rows[-1][8])
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
            new_headers = headers + ["source_row","Moralization",
                            "Sentiment_Stanza_pos","Sentiment_Stanza_neu","Sentiment_Stanza_neg",
                            "Sentiment_Vader_compound",
                            "Sentiment_TextBlob_Polarity","Sentiment_TextBlob_Subjectivity",
                            "individual labels","genericity: generic count","genericity: specific count",
                            "eventivity: stative count","eventivity: dynamic count",
                            "boundedness: static count","boundedness: episodic count","habitual count","NA count",
                            "genericity: proportion generic","genericity: proportion specific","eventivity: proportion stative","eventivity: proportion dynamic",
                            "boundedness: proportion static","boundedness: proportion episodic","proportion habitual","proportion NA",
                            "1_anger","1_disgust","1_fear","1_joy","1_neutral","1_sadness","1_surprise",
                            "2_sadness","2_joy","2_love","2_anger","2_fear","2_surprise",
                            "3_neutral","3_joy","3_surprise","3_anger","3_sadness","3_disgust","3_fear"]
            writer.writerow(new_headers)

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
                    batch_lines.append(line)
                    total_lines += 1

                    # Process in batches once we have batch_size rows
                    if len(batch_lines) == batch_size:
                        texts = [l[2].strip().replace("\n", " ") for l in batch_lines]

                        for i in range(3):
                            if i == 0:
                                _, probs = get_predictions(texts, tokenizer=tokenizer1, model=model1)
                            elif i == 1:
                                _, probs = get_predictions(texts, tokenizer=tokenizer2, model=model2)
                            else:
                                _, probs = get_predictions(texts, tokenizer=tokenizer1, model=model3)

                            # update in place
                            for idx in range(len(batch_lines)):
                                batch_lines[idx] = batch_lines[idx] + probs[idx]

                        # now add all updated rows to relevant_lines
                        relevant_lines.extend(batch_lines)

                        if relevant_lines:
                            writer.writerows(relevant_lines)
                            relevant_lines.clear()
                        batch_lines.clear()
                else:
                    log_report(output_file_path,
                        f"Skipping line {id_}: insufficient columns ({len(line)} found)")
                    missing_lines_count += 1
            except Exception as e:
                raise Exception(
                    f"Error labeling {file} from the {group} social group for emotion: {e}"
                )

        # Process any remaining texts in the final batch
        try:
            if batch_lines:

                texts = [l[2].strip().replace("\n", " ") for l in batch_lines]

                for i in range(3):
                    if i == 0:
                        _, probs = get_predictions(texts, tokenizer=tokenizer1, model=model1)
                    elif i == 1:
                        _, probs = get_predictions(texts, tokenizer=tokenizer2, model=model2)
                    else:
                        _, probs = get_predictions(texts, tokenizer=tokenizer1, model=model3)

                    # update in place
                    for idx in range(len(batch_lines)):
                        batch_lines[idx] = batch_lines[idx] + probs[idx]

                # now add all updated rows to relevant_lines
                relevant_lines.extend(batch_lines)

                if relevant_lines:
                    writer.writerows(relevant_lines)
                    relevant_lines.clear()
                batch_lines.clear()
        
        except Exception as e:
            raise Exception(
                f"Error labeling {file} from the {group} social group for emotion: {e}"
        )
                
    end_time = time.time()
    elapsed_minutes = (end_time - start_time) / 60
    log_report(report_file_path, f"Finished labeling emotion for the {group} social group in {file} within {elapsed_minutes:.2f} minutes. Processed rows: {total_lines}")

    # Record missing lines info
    if missing_lines_count:
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
        
    overall_docs += label_emotion_file(file)

overall_elapsed = (time.time() - start_time) / 60
log_report(report_file_path, f"Emotion labeling for the {group} social group for {args.years} finished in {overall_elapsed:.2f} minutes. Total processed rows: {overall_docs}")

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

##########################################
# ----- Aggregate overall statistics and save final summary report -----
final_report = [
    ["Timestamp", "Social Group", "Years", "Total Processed Rows", "Total Elapsed Time (min)"],
    [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), group, args.years, overall_docs, f"{overall_elapsed:.2f}"]
]
final_report_file = os.path.join(output_path, "final_report_label_emotion.csv")
with open(final_report_file, "w", encoding="utf-8", newline="") as rf:
    writer = csv.writer(rf)
    writer.writerows(final_report)
log_report(report_file_path, f"Final summary report saved to: {final_report_file}")
##########################################
