### Imports

# import functions and objects
from cli import get_args, DATA_DIR
from utils import parse_range, groups, log_report, log_error

# import Python packages
import csv
csv.field_size_limit(2**31 - 1) # Increase the field size limit to handle larger fields
import random
import os
import time
import datetime
from pathlib import Path

### Argument Handling

# Extract and transform CLI arguments 
args = get_args()
years = parse_range(args.years)
type_ = args.type
group = args.group

### sampling hyper-parameters/initializations

num_annot = 2 # number of annotators

sample_size = getattr(args, "sample", 200) # note that because of stratification, the final count might be significantly different from this specific number
if not args.target: # determines the dataset stage to sample from
    target = "filter_keywords_adv" # default to post-filtering dataset
else: 
    target = args.target # otherwise, set from the CLI arguments

# Calculate how many samples to take per year, per category (top/bottom/random)
total_samples_per_year = sample_size // len(years)
samples_per_type_per_year = total_samples_per_year // 3

# Dictionary to store final samples for each annotator
all_samples = {}
for i in range(num_annot):
    all_samples[i] = []

# Global set to track document ids across reservoirs (to prevent duplicates)
seen_ids = set()

### Path Handling

# set path variables

# Survey the input files and raise an error if an expected file is missing

# determine input folder
if not args.input: # assumes the default folder structure and naming conventions for the repository
    sample_path = os.path.join(DATA_DIR,"data_reddit_curated", group, type_, '{}ed_{}'.format(target.split('_')[0],"_".join(target.split('_')[1:])))
else:
    sample_path = args.input

# Organize input files by year
files_by_year = {year: [] for year in years}
if type_ == "comments":
    prefix = "RC"
elif type_ == "submissions":
    prefix = "RS"
else:
    raise Exception("Wrong data type specified. Choose from comments and submissions.")
for year in years:
    for month in range(1, 13):
        path_ = os.path.join(sample_path, f"{prefix}_{year}-{month:02d}.csv")
        if os.path.exists(path_):
            files_by_year[year].append(path_)
        else:
            raise Exception(
                f"Missing {prefix} file for year {year}, month {month}. Expected path: {path_}"
            )

# determine output folder
if not args.output:
    output_dir = os.path.join(DATA_DIR,
        "samples",
        group,
        type_
    )
else:
    output_dir = args.output

os.makedirs(output_dir, exist_ok=True)

# Report file path (placed in the project directory)
report_file_path = os.path.join(output_dir, f"Report_FilterSample.csv")

### Main Functions

# Extract and count unique social group-related keywords from the input string.
def get_unique_keywords(keyword_str, max_keywords=100):
    try:
        # Split by comma and clean each keyword
        keywords = keyword_str.replace('\t', ',').split(',')
        # Use a set for uniqueness
        cleaned_keywords = set()
        for kw in keywords:
            kw = kw.strip()
            # Example: 'fat:' or 'thin:' special logic, if needed
            if '{}:'.format(groups[args.group][0]) in kw or '{}:'.format(groups[args.group][1]) in kw:
                parts = kw.split(':')
                if len(parts) > 1:
                    cleaned_keywords.add(f"{parts[0].strip()}: {parts[1].strip()}")
            elif kw:
                cleaned_keywords.add(kw)
        unique_keywords = list(cleaned_keywords)[:max_keywords]
        return unique_keywords, len(unique_keywords)
    except Exception as e:
        log_report(f"Error processing keywords: {e}")
        return [], 0

# Process each year
def filter_sample_year(year, file_list_for_year):
    log_report(f"Started sampling documents for year {year} in group {args.group}.")
    print(f"\nSampling documents for the {args.group} social group from year {year}...")

    # Reservoirs (lists) for top, bottom, random
    top_reservoir = []
    bottom_reservoir = []
    random_reservoir = []

    total_docs = 0  # How many docs processed for this year

    # Local set to track document ids for this year
    year_seen_ids = set()

    # Iterate through each file for this year
    for file in file_list_for_year:
        print(f"Sampling from {Path(file).name}")
        try:
            with open(file, "r", encoding='utf-8-sig', errors='ignore') as input_file:
                reader = csv.reader(x.replace('\0', '') for x in input_file)
                for id_, line in enumerate(reader):
                    # Skip the header row
                    if id_ == 0:
                        continue
                    try:
                        # Basic row validation: must have at least 3 columns for text
                        if line and len(line) > 2 and line[2].strip():
                            # Extract original_id from first column
                            original_id = line[0].strip()
                            # Skip if this document has already been processed in this year
                            if original_id in year_seen_ids:
                                continue
                            year_seen_ids.add(original_id)
                            seen_ids.add(original_id)
                            
                            text = line[2].strip().replace("\n", " ")
                            
                            # If there's a keywords column (index 7), parse it
                            if len(line) > 7:
                                keywords, unique_count = get_unique_keywords(line[7])
                            else:
                                keywords, unique_count = [], 0
                            
                            total_docs += 1

                            #    TOP SAMPLES: more unique keywords
                            if len(top_reservoir) < samples_per_type_per_year:
                                top_reservoir.append((unique_count, text, keywords, file, original_id))
                                top_reservoir.sort(key=lambda x: x[0], reverse=True)
                            else:
                                if unique_count > top_reservoir[-1][0]:
                                    top_reservoir[-1] = (unique_count, text, keywords, file, original_id)
                                    top_reservoir.sort(key=lambda x: x[0], reverse=True)

                            #    BOTTOM SAMPLES: fewer unique keywords
                            if len(bottom_reservoir) < samples_per_type_per_year:
                                bottom_reservoir.append((unique_count, text, keywords, file, original_id))
                                bottom_reservoir.sort(key=lambda x: x[0])
                            else:
                                if unique_count < bottom_reservoir[-1][0]:
                                    bottom_reservoir[-1] = (unique_count, text, keywords, file, original_id)
                                    bottom_reservoir.sort(key=lambda x: x[0])

                            #    RANDOM SAMPLES
                            if len(random_reservoir) < samples_per_type_per_year:
                                random_reservoir.append((unique_count, text, keywords, file, original_id))
                            else:
                                s = random.randint(0, total_docs - 1)
                                if s < samples_per_type_per_year:
                                    random_reservoir[s] = (unique_count, text, keywords, file, original_id)
                        else:
                            log_report(f"Skipping line {id_} in file {file}: insufficient columns ({len(line)} found)")
                    except Exception as e:
                        log_error("filter_sample_year", file, id_ + 1, str(line), e)
                        continue
        except Exception as e:
            log_error("filter_sample_year", file, 0, "File-level error", e) # Line number for sampling error defaults to 0.
            continue

    log_report(f"{total_docs} documents processed for year {year} in group {args.group}.")

    # Combine the top, bottom, and random samples
    year_samples = top_reservoir + bottom_reservoir + random_reservoir

    # Assign sample type labels in the same order
    sample_types = (
        ["top_sample"] * len(top_reservoir) +
        ["bottom_sample"] * len(bottom_reservoir) +
        ["random_sample"] * len(random_reservoir)
    )

    # Assign random IDs, store results for the year
    random_ids_used = set()
    year_sample_data = []
    
    for i, (unique_count, text, keywords, file, original_id) in enumerate(year_samples):
        rand_int = random.randint(100000, 999999)
        while rand_int in random_ids_used:
            rand_int = random.randint(100000, 999999)
        random_ids_used.add(rand_int)
        
        year_sample_data.append({
            'random_id': rand_int,
            'text': text,
            'keywords': keywords,
            'file': file,
            'sample_type': sample_types[i],
            'original_id': original_id
        })

    # Append these samples to each annotator
    for annot in range(num_annot):
        all_samples[annot].extend(year_sample_data)

# After processing all years, write output
def filter_sample_write(all_samples):
    years_tag = args.years.replace("-", "_to_")
    run_tag = f"{target}_n{sample_size}_{years_tag}"

    for annot in range(num_annot):
        sample_file_path = os.path.join(output_dir, f"filter_sample_{run_tag}_annot{annot}.csv")
        sample_key_file_path = os.path.join(output_dir, f"filter_sample_{run_tag}_annot{annot}_key.csv")
        with open(sample_file_path, "w", encoding='utf-8', newline='') as sample_file, \
             open(sample_key_file_path, "w", encoding='utf-8', newline='') as sample_file_key:
            
            writer = csv.writer(sample_file)
            writer_key = csv.writer(sample_file_key)
            
            # Write headers
            writer.writerow(["random_id", "text"])
            writer_key.writerow(["random_id", "file", "original_id", "keywords", "sample_type"])
            
            # Shuffle samples before writing so we don't group them year by year
            random.shuffle(all_samples[annot])
            
            # Write rows
            for data in all_samples[annot]:
                writer.writerow([data['random_id'], data['text'], "", "", ""])
                writer_key.writerow([
                    data['random_id'],
                    data['file'],
                    data['original_id'],
                    ",".join(data['keywords']),
                    data['sample_type']
                ])

### Main Execution

if __name__ == "__main__":
    start_time = time.time()
    # Process each year with only its corresponding files
    for year in years:
        file_list = files_by_year[year]
        filter_sample_year(year, file_list)
    filter_sample_write(all_samples)
    elapsed = (time.time() - start_time) / 60
    print(f"Reservoir sampling for the {group} social group from {years} was finished in {elapsed:.2f} minutes. Total samples per each of the {num_annot} annotators: {len(all_samples[0])}")
    log_report(f"Reservoir sampling for the {group} social group from {args.years} finished in {elapsed:.2f} minutes. Total samples per annotator: {len(all_samples[0])}")
