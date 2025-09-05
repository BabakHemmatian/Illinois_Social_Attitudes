# import functions and objects
from cli import get_args, dir_path
from utils import parse_range, headers, log_report

# import Python packages
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # to prevent MKL crash with Torch
import csv
import time
import torch
from transformers import RobertaTokenizerFast, AutoModelForTokenClassification, RobertaForSequenceClassification
import datetime
import re
import spacy
import numpy as np
from copy import deepcopy

# Extract and transform CLI arguments 
args = get_args()
years = parse_range(args.years)
group = args.group

# Set moralization labeling hyperparameters
batch_size = 1000
max_length=512

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# prepare the report file
report_file_path = os.path.join(dir_path, f"report_label_generalization.csv")
log_report(report_file_path,f"Using device: {device}")

# Set the base model path
model_path = os.path.join(dir_path.replace("code", "models"),
                          f"label_generalization")

# load the necessary models and move them to the device being used
nlp = spacy.load("en_core_web_sm")
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base",add_prefix_space=True)  
clause_model = AutoModelForTokenClassification.from_pretrained("{}/label_generalization_segmentation".format(model_path), num_labels=3).to(device)
generalization_model = RobertaForSequenceClassification.from_pretrained("{}/label_generalization".format(model_path), num_labels=18).to(device)

if torch.cuda.device_count() > 1: # if more than one GPU is available
    clause_model = torch.nn.DataParallel(clause_model) # parallelize
    generalization_model = torch.nn.DataParallel(generalization_model)

clause_model.eval() # set model to evaluation mode
generalization_model.eval() # set model to evaluation mode

# Survey the sentiment-labeled input files 
sentiment_labeled_path = os.path.join(
    dir_path.replace("code", "data"),
    "data_reddit_curated", group, "labeled_sentiment"
)

# Build file_list organized by year and raise an error if an expected file is missing 
file_list = []
for year in years:
    for month in range(1, 13):
        filename = "RC_{}-{:02d}.csv".format(year, month)
        path_ = os.path.join(sentiment_labeled_path, filename)
        if os.path.exists(path_):
            file_list.append(path_)
        else:
            raise Exception("Missing sentiment-labeled file for the {} social group from year {}, month {}".format(group, year, month))

# Prepare and survey the output path for moralization labeling
output_path = os.path.join(dir_path.replace("code", "data"),
                           "data_reddit_curated", group, "labeled_generalization")
os.makedirs(output_path, exist_ok=True)

# define the mapping between clause labels and each of the three composing features
labels2attrs = {
    "##BOUNDED EVENT (SPECIFIC)": ("specific", "dynamic", "episodic"),
    "##BOUNDED EVENT (GENERIC)": ("generic", "dynamic", "episodic"),
    "##UNBOUNDED EVENT (SPECIFIC)": ("specific", "dynamic", "static"),  # This should be (static, or habitual)
    "##UNBOUNDED EVENT (GENERIC)": ("generic", "dynamic", "static"),
    "##BASIC STATE": ("specific", "stative", "static"),
    "##COERCED STATE (SPECIFIC)": ("specific", "dynamic", "static"),
    "##COERCED STATE (GENERIC)": ("generic", "dynamic", "static"),
    "##PERFECT COERCED STATE (SPECIFIC)": ("specific", "dynamic", "episodic"),
    "##PERFECT COERCED STATE (GENERIC)": ("generic", "dynamic", "episodic"),
    "##GENERIC SENTENCE (DYNAMIC)": ("generic", "dynamic", "habitual"),   
    "##GENERIC SENTENCE (STATIC)": ("generic", "stative", "static"),  # The car is red now (static)
    "##GENERIC SENTENCE (HABITUAL)": ("generic", "stative", "habitual"),   # I go to the gym regularly (habitual)
    "##GENERALIZING SENTENCE (DYNAMIC)": ("specific", "dynamic", "habitual"),
    "##GENERALIZING SENTENCE (STATIVE)": ("specific", "stative", "habitual"),
    "##QUESTION": ("NA", "NA", "NA"),
    "##IMPERATIVE": ("NA", "NA", "NA"),
    "##NONSENSE": ("NA", "NA", "NA"),
    "##OTHER": ("NA", "NA", "NA"),
}

# create dictionaries for ease of translating between clause labels and attributes
label2index = {l:i for l,i in zip(labels2attrs.keys(), np.arange(len(labels2attrs)))}
index2label = {i:l for l,i in label2index.items()}

# Splits longer text inputs at the end of sentences into parts that the neural networks can label without truncation.
def auto_split(text):
    doc = nlp(text)
    current_len = 0
    snippets = []
    current_snippet = ""
    for sent in doc.sents:
        text = sent.text
        words = text.split()
        if current_len + len(words) > 200:
            snippets.append(current_snippet)
            current_snippet = text
            current_len = len(words)
        else:
            current_snippet += " " + text
            current_len += len(words)
    snippets.append(current_snippet) # the leftover part. 
    return snippets

# Runs the segmentation + generalization pipeline on a batch of texts. Takes str or list of str as input
def run_pipeline(texts, model_batch_size=32, max_length=max_length):

    if isinstance(texts, str):
        texts = [texts]  # wrap single string into list

    # Split each text into snippets
    all_snippets = []
    snippet_map = []  # keep track of which text each snippet came from
    for doc_id, text in enumerate(texts):
        snippets = auto_split(text)
        all_snippets.extend(snippets)
        snippet_map.extend([doc_id] * len(snippets))

    # Tokenize snippets into words for segmentation
    tokenized_snippets = [s.strip().split() for s in all_snippets]

    # Predict segmentation labels in batch
    all_labels = get_pred_clause_labels(
        all_snippets,
        tokenized_snippets,
        model_batch_size=model_batch_size,
        max_length=max_length,
    )

    # Reconstruct clauses
    all_clauses = []
    clause_map = []  # keep track of which text each clause came from
    for snip_id, (words, labels) in enumerate(zip(tokenized_snippets, all_labels)):
        clauses = reconstruct_clauses(words, labels)
        all_clauses.extend(clauses)
        clause_map.extend([snippet_map[snip_id]] * len(clauses))

    # Predict generalization labels in batch
    clause2labels = get_pred_generalization_labels(
        all_clauses, model_batch_size=model_batch_size, max_length=max_length
    )

    # Organize outputs back per input text
    results = [[] for _ in texts]
    results_with_labels = [[] for _ in texts]

    for i, clause in enumerate(all_clauses):
        doc_id = clause_map[i]
        results[doc_id].append((clause, str(len(results[doc_id]) + 1)))

    for s, l in clause2labels:
        doc_id = clause_map[all_clauses.index(s)]
        results_with_labels[doc_id].append((s, l))

    return results, results_with_labels

# extracts generalization labels from clause-segmented input
def get_pred_generalization_labels(clauses, model_batch_size=32, max_length=256):
    clause2labels = []
    for i in range(0, len(clauses), model_batch_size):
        batch_examples = clauses[i : i + model_batch_size]

        model_inputs = tokenizer(
            batch_examples,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = generalization_model(**model_inputs)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits

        pred_labels = logits.argmax(-1).cpu().numpy()
        pred_labels = [index2label[l] for l in pred_labels]

        clause2labels.extend(
            [(s, str(l)) for s, l in zip(batch_examples, pred_labels)]
        )

    return clause2labels

# generates clause segmentations from input
def get_pred_clause_labels(texts, tokenized_texts, model_batch_size=32, max_length=256):
    all_labels = []

    for i in range(0, len(texts), model_batch_size):
        batch_words = tokenized_texts[i : i + model_batch_size]

        encoding = tokenizer(
            batch_words,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            outputs = clause_model(**encoding)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits

        pred_ids = logits.argmax(-1).cpu().numpy()

        # Map predictions back to word-level labels (one label per word)
        for b_idx, words in enumerate(batch_words):
            word_ids = encoding.word_ids(batch_index=b_idx)
            labels = []
            seen = set()
            for j, word_idx in enumerate(word_ids):
                if word_idx is None or word_idx in seen:
                    continue
                labels.append(pred_ids[b_idx][j])  # label from first subword
                seen.add(word_idx)
            all_labels.append(labels)

    return all_labels

# Reconstructs segmented clauses from words + predicted labels.
def reconstruct_clauses(words, labels):

    segmented_clauses = []
    prev_label = 2
    current_clause = None
    for cur_token, cur_label in zip(words, labels):
        if prev_label == 2:
            current_clause = []
        if current_clause is not None:
            current_clause.append(cur_token)

        if cur_label == 2:
            if prev_label in [0, 1]:
                segmented_clauses.append(deepcopy(current_clause))
                current_clause = None
        prev_label = cur_label

    if current_clause is not None and len(current_clause) != 0:
        segmented_clauses.append(deepcopy(current_clause))

    return [" ".join(clause) for clause in segmented_clauses if clause is not None]

# Generates and writes labels for an entire month's worth of documents. If the output file already exists, we check the last processed row number and resume from there.
def label_generalization_file(file):
    # Initialize missing lines count
    missing_lines_count = 0
    missing_records_file = os.path.join(output_path, 'missing_records.csv')
    # Create missing records file with header if it does not exist.
    if not os.path.exists(missing_records_file):
        with open(missing_records_file, 'w', newline='') as missing_file:
            missing_writer = csv.writer(missing_file)
            missing_writer.writerow(['Filename', 'MissingLinesCount', 'Timestamp'])

    log_report(report_file_path, f"Started labeling {file} from the {group} social group for generalization.")
    start_time = time.time()
    
    # Build output file path using the relative part from the input file.
    relative_path = file.split("sentiment")[1].lstrip(os.sep)
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
                    last_processed = int(rows[-1][-1])
                except:
                    last_processed = 0
            else:
                last_processed = 0
    else:
        mode = "w"
        last_processed = 0

    # open the input and output files in the correct modes
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
                                           "genericity:  proportion generic","genericity: proportion specific","eventivity: proportion stative","eventivity: proportion dynamic",
                                           "boundedness: proportion static","boundedness: proportion episodic","proportion habitual","proportion NA"]
            writer.writerow(new_headers)
        
        batch_lines = []
        total_lines = 0
        relevant_lines = []  # Rows to write in bulk
        
        # go line by line through the input file to generate and write labels in batches
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

                        # 1) run predictions on the batch
                        texts = [l[2].strip().replace("\n"," ") for l in batch_lines]
                        _,result = run_pipeline(texts)

                        counts = {}
                        individual_labels = {}
                        props = {}
                        
                        # extracts and counts clause labels for each document in the batch
                        for id_,text in enumerate(result):
                            individual_labels[id_] = []
                            counts[id_] = {"generic":0,"specific":0,"stative":0,"dynamic":0,"static":0,"episodic":0,"habitual":0,"NA genericity":0,"NA eventivity":0,"NA boundedness":0}
                            for clause in text:
                                individual_labels[id_].append(clause[1])
                                label = labels2attrs[clause[1]]
                                for id__,feature in enumerate(label):
                                    if "NA" not in feature:
                                        counts[id_][feature] += 1
                                    elif id__ == 0:
                                        counts[id_]["NA genericity"] += 1
                                    elif id__ == 1:
                                        counts[id_]["NA eventivity"] += 1
                                    else:
                                        counts[id_]["NA boundedness"] += 1

                            # extracts the proportions of different genericity, eventivity, boundedness, habituality and NA labels for a given document
                            props[id_] = []
                            gen_tot = counts[id_]['generic']+counts[id_]['specific']+counts[id_]['NA genericity']
                            if gen_tot:
                                props[id_] += [(counts[id_]['generic'])/gen_tot,
                                counts[id_]['specific']/gen_tot]
                            else:
                                props[id_] += [0,0]
                            eve_tot = counts[id_]['stative']+counts[id_]['dynamic']+counts[id_]['NA eventivity']
                            if eve_tot:
                                props[id_] += [counts[id_]['stative']/eve_tot,
                                counts[id_]['dynamic']/eve_tot]
                            else:
                                props[id_] += [0,0]
                            bou_tot = counts[id_]['static']+counts[id_]['episodic']+counts[id_]["habitual"]+counts[id_]['NA boundedness']
                            if bou_tot:
                                props[id_] += [
                                counts[id_]['static']/bou_tot,
                                counts[id_]['episodic']/bou_tot,
                                counts[id_]['habitual']/bou_tot,
                                counts[id_]['NA boundedness']/bou_tot]
                            else:
                                props[id_] += [0,0,0,0]

                        # 2) collect the generalization labels

                        for id_ in counts.keys():

                            ind_labels = "\n".join(individual_labels[id_])
                            row = batch_lines[id_] + [ind_labels,counts[id_]['generic'],counts[id_]['specific'],counts[id_]['stative'],counts[id_]['dynamic'],counts[id_]['static'],counts[id_]['episodic'],counts[id_]['habitual'],counts[id_]['NA boundedness']]+props[id_]
            
                            relevant_lines.append(row)

                        # 3) write out and clear buffers
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
                    f"Error labeling {file} from the {group} social group for generalization: {e}"
                )
        try: # process any left-over documents as the final batch
            if batch_lines:
                
                 # 1) run predictions on the batch
                
                texts = [l[2].strip().replace("\n"," ") for l in batch_lines]
                _,result = run_pipeline(texts)

                counts = {}
                individual_labels = {}
                props = {}
                
                # extracts and counts clause labels for each document in the batch
                for id_,text in enumerate(result):
                    individual_labels[id_] = []
                    counts[id_] = {"generic":0,"specific":0,"stative":0,"dynamic":0,"static":0,"episodic":0,"habitual":0,"NA genericity":0,"NA eventivity":0,"NA boundedness":0}
                    for clause in text:
                        individual_labels[id_].append(clause[1])
                        label = labels2attrs[clause[1]]
                        for id__,feature in enumerate(label):
                            if "NA" not in feature:
                                counts[id_][feature] += 1
                            elif id__ == 0:
                                counts[id_]["NA genericity"] += 1
                            elif id__ == 1:
                                counts[id_]["NA eventivity"] += 1
                            else:
                                counts[id_]["NA boundedness"] += 1
                    
                    # extracts the proportions of different genericity, eventivity, boundedness, habituality and NA labels for a given document
                    props[id_] = []
                    gen_tot = counts[id_]['generic']+counts[id_]['specific']+counts[id_]['NA genericity']
                    if gen_tot:
                        props[id_] += [(counts[id_]['generic'])/gen_tot,
                        counts[id_]['specific']/gen_tot]
                    else:
                        props[id_] += [0,0]
                    eve_tot = counts[id_]['stative']+counts[id_]['dynamic']+counts[id_]['NA eventivity']
                    if eve_tot:
                        props[id_] += [counts[id_]['stative']/eve_tot,
                        counts[id_]['dynamic']/eve_tot]
                    else:
                        props[id_] += [0,0]
                    bou_tot = counts[id_]['static']+counts[id_]['episodic']+counts[id_]["habitual"]+counts[id_]['NA boundedness']
                    if bou_tot:
                        props[id_] += [
                        counts[id_]['static']/bou_tot,
                        counts[id_]['episodic']/bou_tot,
                        counts[id_]['habitual']/bou_tot,
                        counts[id_]['NA boundedness']/bou_tot]
                    else:
                        props[id_] += [0,0,0,0]


                # 2) collect the generalization labels

                for id_ in counts.keys():

                    ind_labels = "\n".join(individual_labels[id_])
                    row = batch_lines[id_] + [ind_labels,counts[id_]['generic'],counts[id_]['specific'],counts[id_]['stative'],counts[id_]['dynamic'],counts[id_]['static'],counts[id_]['episodic'],counts[id_]['habitual'],counts[id_]['NA boundedness']]+props[id_]
            
                    relevant_lines.append(row)

                # 3) write out and clear buffers
                if relevant_lines:
                    writer.writerows(relevant_lines)
                    relevant_lines.clear()
                batch_lines.clear()
        
        except Exception as e:
            raise Exception(
                f"Error labeling {file} from the {group} social group for generalization: {e}"
            )

    end_time = time.time()
    elapsed_minutes = (end_time - start_time) / 60
    log_report(report_file_path, f"Finished labeling generalization for the {group} social group in {file} within {elapsed_minutes:.2f} minutes. Processed rows: {total_lines}")

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
    overall_docs += label_generalization_file(file)

overall_elapsed = (time.time() - start_time) / 60
log_report(report_file_path, f"Generalization labeling for the {group} social group for {args.years} finished in {overall_elapsed:.2f} minutes. Total processed rows: {overall_docs}")

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
final_report_file = os.path.join(output_path, "final_report_label_generalization.csv")
with open(final_report_file, "w", encoding="utf-8", newline="") as rf:
    writer = csv.writer(rf)
    writer.writerows(final_report)
log_report(report_file_path, f"Final summary report saved to: {final_report_file}")
##########################################
