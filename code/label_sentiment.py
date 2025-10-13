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
from pathlib import Path

# note the tool use order in the readme

## Models:
# VADER: to include a rule-based one focused on social media data
# TextBlob: for a neural continuous measure (trained on movie reviews)
# Stanza: for a neural categorical measure (broader training data)

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
DATA_DIR = PROJECT_ROOT / "data"

moralization_labeled_path = DATA_DIR / "data_reddit_curated" / group / "labeled_moralization"
output_path = DATA_DIR / "data_reddit_curated" / group / "labeled_sentiment"
output_path.mkdir(parents=True, exist_ok=True)

# prepare the report file
report_file_path = os.path.join(dir_path, f"report_label_sentiment.csv")
log_report(report_file_path)

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

# generates labels for an entire month's worth of documents. It resumes labeling if it comes across incomplete output.
def label_sentiment_file(file):

    missing_lines_count = 0  
    log_report(report_file_path, f"Started labeling {Path(file).name} from the {group} social group for sentiment.")
    start_time = time.time()

    # Compute output path that mirrors input directory structure
    relative_path = Path(file).relative_to(moralization_labeled_path)
    output_file_path = output_path / relative_path
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    # determine resume position from any existing output
    mode = "w"
    last_processed = -1  # nothing processed yet
    out_header = None
    source_row_out_idx = None

    if os.path.exists(output_file_path):
        # Read header + last data row to find the last processed source_row
        with open(output_file_path, "r", encoding="utf-8-sig", errors="ignore") as f_out:
            out_rows = list(csv.reader(f_out))
            if len(out_rows) > 1:
                out_header = out_rows[0]
                try:
                    source_row_out_idx = out_header.index("source_row")
                    # Walk backwards to find last non-empty source_row
                    for r in reversed(out_rows[1:]):
                        if len(r) > source_row_out_idx and r[source_row_out_idx].strip():
                            last_processed = int(r[source_row_out_idx])
                            break
                    mode = "a"  # append 
                except ValueError:
                    # Output exists but somehow lacks 'source_row' in header; start fresh.
                    mode = "w"
                    last_processed = -1

    # sentiment tools
    try:
        _ = nlp, analyzer  # type: ignore  # these are set later in your script
    except NameError:
        stanza.download('en', processors='tokenize,sentiment', verbose=False)
        globals()["nlp"] = stanza.Pipeline(lang='en', processors='tokenize,sentiment')
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        globals()["analyzer"] = SentimentIntensityAnalyzer()

    # batching
    try:
        BATCH = int(batch_size)  # use your global if present
    except Exception:
        BATCH = 1200

    # process input
    total_lines_written = 0
    with open(file, "r", encoding="utf-8-sig", errors="ignore", newline="") as f_in, \
         open(output_file_path, mode, encoding="utf-8", newline="") as f_out:

        reader = csv.reader(f_in)
        writer = csv.writer(f_out)

        # Read input header and find its 'source_row'
        try:
            in_header = next(reader)
        except StopIteration:
            # Empty input file
            return 0

        try:
            source_row_in_idx = in_header.index("source_row")
        except ValueError:
            raise RuntimeError("Input file is missing required 'source_row' column.")

        # If we're starting a brand-new output file, write headers
        if mode == "w":
            new_headers = in_header + [
                "Sentiment_Stanza_pos", "Sentiment_Stanza_neu", "Sentiment_Stanza_neg",
                "Sentiment_Vader_compound",
                "Sentiment_TextBlob_Polarity", "Sentiment_TextBlob_Subjectivity"
            ]
            writer.writerow(new_headers)

        # Prepare batch buffers
        batch_lines = []
        batch_input_rows = []  # keep the original input rows in parallel

        def flush_batch():
            nonlocal total_lines_written
            if not batch_lines:
                return
            # Collect sentiments for each line in the batch
            out_rows = []
            for line in batch_lines:
                try:
                    # Stanza sentence-level aggregation
                    doc = nlp(line[2].strip().replace("\n", " "))
                    stanza_counts = {"pos": 0, "neu": 0, "neg": 0}
                    stanza_labels = {0: "neg", 1: "neu", 2: "pos"}
                    vader_scores = []
                    for sentence in doc.sentences:
                        stanza_counts[stanza_labels[sentence.sentiment]] += 1
                        vader_scores.append(analyzer.polarity_scores(sentence.text)["compound"])

                    # TextBlob document-level
                    tb = TextBlob(line[2].strip().replace("\n", " "))

                    # Append only new columns; keep the original row intact
                    out_row = line + [
                        stanza_counts["pos"], stanza_counts["neu"], stanza_counts["neg"],
                        (np.mean(vader_scores) if vader_scores else 0.0),
                        tb.sentiment.polarity, tb.sentiment.subjectivity
                    ]
                    out_rows.append(out_row)
                except Exception:
                    # Count and skip any bad line, but keep going
                    nonlocal missing_lines_count
                    missing_lines_count += 1

            if out_rows:
                writer.writerows(out_rows)
                total_lines_written += len(out_rows)

            # clear buffers
            batch_lines.clear()
            batch_input_rows.clear()

        # Iterate the input, skip header 
        for row in reader:
            # Guard against short/blank lines (expect at least 3 columns given usage of row[2] for text)
            if len(row) < 3:
                missing_lines_count += 1
                continue

            # Use the input's source_row to decide skipping/resume
            src_value = row[source_row_in_idx].strip()
            if not src_value.isdigit():
                # If malformed (e.g. empty), treat as missing and process anyway to avoid skipping the whole file accidentally.
                src_num = None
            else:
                src_num = int(src_value)

            if src_num is not None and src_num <= last_processed:
                continue  # already processed in previous run

            batch_lines.append(row)
            batch_input_rows.append(row)

            if len(batch_lines) >= BATCH:
                flush_batch()

        # Flush any remainder
        flush_batch()

    # generate processing report
    end_time = time.time()
    elapsed_minutes = (end_time - start_time) / 60
    log_report(report_file_path, f"Finished labeling sentiment for the {group} social group in {Path(file).name} within {elapsed_minutes:.2f} minutes. Processed rows: {total_lines_written}")

    if missing_lines_count > 0:
        missing_records_file = os.path.join(output_path, 'missing_records.csv')
        need_header = not os.path.exists(missing_records_file)
        with open(missing_records_file, 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            if need_header:
                w.writerow(['Filename', 'MissingLinesCount', 'Timestamp'])
            w.writerow([str(file), missing_lines_count, datetime.datetime.now().isoformat(timespec="seconds")])

    return total_lines_written

##########################################
# Main execution: process each file and aggregate stats
##########################################
start_time = time.time()
overall_docs = 0

# Process each file from the file_list (global mode)

# create the analyzer objects
nlp = stanza.Pipeline(lang='en', processors='tokenize,sentiment')
analyzer = SentimentIntensityAnalyzer()

# Process each file from the file_list (global mode)
if args.array is not None: # for batch processing
    overall_docs += label_sentiment_file(file_list[array])

else: # for sequential processing
    for file in file_list:        
        overall_docs += label_sentiment_file(file)

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
    log_report(report_file_path, f"Sentiment labeling for the {group} social group for {args.years} finished in {overall_elapsed:.2f} minutes. Total processed rows: {overall_docs}")

    ##########################################
    # ----- Aggregate overall statistics and save final summary report -----
    final_report = [
        ["Timestamp", "Social Group", "Years", "Total Processed Rows", "Total Elapsed Time (min)"],
        [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), group, args.years, overall_docs, f"{overall_elapsed:.2f}"]
    ]
    final_report_file = os.path.join(output_path, "final_report_label_sentiment.csv")
    with open(final_report_file, "a+", encoding="utf-8", newline="") as rf:
        writer = csv.writer(rf)
        writer.writerows(final_report)
    log_report(report_file_path, f"Final summary report saved to: {final_report_file}")
    ##########################################

    
