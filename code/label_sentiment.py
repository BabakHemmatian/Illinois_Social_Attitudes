# import functions and objects
from cli import get_args, dir_path
from utils import parse_range, headers, groups, log_report
import os
import csv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import stanza
from textblob import TextBlob
import numpy as np
import time
import datetime
import re

# note the tool use order in the readme

## Models:
# VADER: to include a rule-based one focused on social media data
# TextBlob: for a neural continuous measure (trained on movie reviews)
# Stanza: for a neural categorical measure (broader training data)

# Extract and transform CLI arguments 
args = get_args()
years = parse_range(args.years)
group = args.group

# prepare the report file
report_file_path = os.path.join(dir_path, f"report_label_sentiment.csv")
log_report(report_file_path)

# Set moralization labeling hyperparameters
batch_size = 2500

# Survey the relevance-filtered input files 
moralization_labeled_path = os.path.join(
    dir_path.replace("code", "data"),
    "data_reddit_curated", group, "labeled_moralization"
)

# Build file_list organized by year and raise an error if an expected file is missing 
file_list = []
for year in years:
    for month in range(1, 13):
        filename = "RC_{}-{:02d}.csv".format(year, month)
        path_ = os.path.join(moralization_labeled_path, filename)
        if os.path.exists(path_):
            file_list.append(path_)
        else:
            raise Exception("Missing moralization-labeled file for the {} social group from year {}, month {}".format(group, year, month))

# Prepare and survey the output path for moralization labeling
output_path = os.path.join(dir_path.replace("code", "data"),
                           "data_reddit_curated", group, "labeled_sentiment")
os.makedirs(output_path, exist_ok=True)

# TODO: Fix last_processed

def label_sentiment_file(file):
    # Initialize missing lines count
    missing_lines_count = 0
    missing_records_file = os.path.join(output_path, 'missing_records.csv')
    # Create missing records file with header if it does not exist.
    if not os.path.exists(missing_records_file):
        with open(missing_records_file, 'w', newline='') as missing_file:
            missing_writer = csv.writer(missing_file)
            missing_writer.writerow(['Filename', 'MissingLinesCount', 'Timestamp'])

    log_report(report_file_path, f"Started labeling {file} from the {group} social group for sentiment.")
    start_time = time.time()
    
    # Build output file path using the relative part from the input file.
    relative_path = file.split("moralization")[1].lstrip(os.sep)
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
                                           "Sentiment_TextBlob_Polarity","Sentiment_TextBlob_Subjectivity"]
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

                        # 1) collect the sentiment labels

                        for line in batch_lines:
                            
                            # extract Stanza sentiment scores                        
                            doc = nlp(line[2].strip().replace("\n"," "))
                            stanza_counts = {"pos":0,"neu":0,"neg":0}
                            stanza_labels = {0:"neg",1:"neu", 2:"pos"}
                            vader_scores = []
                            for sentence in doc.sentences:
                                stanza_counts[stanza_labels[sentence.sentiment]] += 1
                                vader_scores.append(analyzer.polarity_scores(sentence.text)["compound"])

                            # extract the TextBlob sentiment scores
                            doc = TextBlob(line[2].strip().replace("\n"," "))

                            sent_line = line + [stanza_counts["pos"],stanza_counts["neu"],stanza_counts["neg"],np.mean(vader_scores),doc.sentiment.polarity,doc.sentiment.subjectivity]
                            relevant_lines.append(sent_line)

                        # 2) write out and clear buffers
                        if relevant_lines:
                            writer.writerows(relevant_lines)
                            relevant_lines.clear()
                        batch_lines.clear()
                else:
                    log_report(output_file_path,f"Skipping line {id_}: insufficient columns ({len(line)} found)")
                    missing_lines_count += 1
            except Exception as e:
                raise Exception(
                    f"Error labeling {file} from the {group} social group for sentiment: {e}"
                )
            
        # Process any remaining texts in the final batch
        try:
            if batch_lines:
                # 1) collect the sentiment labels

                for line in batch_lines:
                    
                    # extract Stanza sentiment scores                        
                    doc = nlp(line[2].strip().replace("\n"," "))
                    stanza_counts = {"pos":0,"neu":0,"neg":0}
                    stanza_labels = {0:"neg",1:"neu", 2:"pos"}
                    vader_scores = []
                    for sentence in doc.sentences:
                        stanza_counts[stanza_labels[sentence.sentiment]] += 1
                        vader_scores.append(analyzer.polarity_scores(sentence.text)["compound"])

                    # extract the TextBlob sentiment scores
                    doc = TextBlob(line[2].strip().replace("\n"," "))

                    sent_line = line + [stanza_counts["pos"],stanza_counts["neu"],stanza_counts["neg"],np.mean(vader_scores),doc.sentiment.polarity,doc.sentiment.subjectivity]
                    relevant_lines.append(sent_line)

                # 2) write out and clear buffers
                if relevant_lines:
                    writer.writerows(relevant_lines)
                    relevant_lines.clear()
                batch_lines.clear()
        except Exception as e:
            raise Exception(
                f"Error labeling {file} from the {group} social group for sentiment: {e}"
            )

    end_time = time.time()
    elapsed_minutes = (end_time - start_time) / 60
    log_report(report_file_path, f"Finished labeling sentiment for the {group} social group in {file} within {elapsed_minutes:.2f} minutes. Processed rows: {total_lines}")

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

# create the analyzer objects
nlp = stanza.Pipeline(lang='en', processors='tokenize,sentiment')
analyzer = SentimentIntensityAnalyzer()

for file in file_list:
    overall_docs += label_sentiment_file(file)

overall_elapsed = (time.time() - start_time) / 60
log_report(report_file_path, f"Sentiment labeling for the {group} social group for {args.years} finished in {overall_elapsed:.2f} minutes. Total processed rows: {overall_docs}")

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
final_report_file = os.path.join(output_path, "final_report_label_sentiment.csv")
with open(final_report_file, "w", encoding="utf-8", newline="") as rf:
    writer = csv.writer(rf)
    writer.writerows(final_report)
log_report(report_file_path, f"Final summary report saved to: {final_report_file}")
##########################################
