# import functions and objects
from cli import get_args,dir_path
from utils import parse_range,headers,check_reqd_files

# import Python packages
import os
import csv
import time
import torch
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
import datetime
import sys

# Extract and transform CLI arguments 
args = get_args()
years = parse_range(args.years)

# set relevance filtering hyperparameters
batch_size = 320

# Use cude if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# load relevance model
model_path = dir_path.replace("code","models\\filter_relevance_{}_roberta_large".format(args.group))
tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path).to(device)
model.eval()  # Set model to evaluation mode

def get_predictions(texts,max_length=512):
    # Tokenize and encode the batch of texts
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            predictions = probs.argmax(dim=1).tolist()  # List of predictions
    return predictions

# survey the language-filtered input files and raise an error if an expected file is missing
language_filtered_path = os.path.join(dir_path.replace("code","data\\data_reddit_curated\\{}\\filtered_language".format(args.group)))
file_list = check_reqd_files(years=years,check_path=language_filtered_path)

# Prepare and survey the output path
output_path = os.path.join(dir_path.replace("code","data\\data_reddit_curated\\{}\\filtered_relevance".format(args.group)))
os.makedirs(output_path, exist_ok=True)

def filter_relevance_file(file):
    # Initialize missing lines count
    missing_lines_count = 0
    missing_records_file = 'missing_records.csv'

    # Check if the missing records file exists; if not, create it with a header
    if not os.path.exists(missing_records_file):
        with open(missing_records_file, 'w', newline='') as missing_file:
            missing_writer = csv.writer(missing_file)
            missing_writer.writerow(['Filename', 'MissingLinesCount', 'Timestamp'])

    print(f"Filtering {file} for relevance to the {args.group} social group.")
    start_time = time.time()
    
    output_file_path = output_path+file.split("language")[1]

    with open(file, "r", encoding='utf-8-sig', errors='ignore') as input_file, \
         open(output_file_path, "w", encoding='utf-8-sig', errors='ignore', newline='') as output_file:
        
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
                    print("{} documents filtered in {}".format(id_,file))
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
                        print(f"Skipping line {id_}: insufficient columns ({len(line)} found)")
                        missing_lines_count += 1  # Increment missing lines count
                except:
                    raise Exception("Error filtering {} for relevance to the {} social group".format(file,args.group))

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
        except:
            raise Exception("Error filtering {} for relevance to the {} social group".format(file,args.group))

    end_time = time.time()
    elapsed_minutes = (end_time - start_time) / 60
    print(f"Finished filtering {file} for relevance to the {args.group} social group in {elapsed_minutes:.2f} minutes. # lines assessed: {total_lines}, relevant lines: {relevant_count}")

    # Record the missing lines count in the missing records CSV file
    with open(missing_records_file, 'a', newline='') as missing_file:
        missing_writer = csv.writer(missing_file)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        missing_writer.writerow([file, missing_lines_count, timestamp])

if __name__ == "__main__":
    start_time = time.time()
    for file in file_list:
        filter_relevance_file(file)
    print(f"Relevance filtering for the {args.group} social group for {args.years} was finished in {(time.time() - start_time) / 60:.2f} minutes.")

# TODO: Save errors, including lines that are skipped to a separate error file, with the named formatted like the one for filter_keywords
# TODO: Add a warning if a particular month is missing from the output of the function.
# TODO: Add overall statistics to the final report
