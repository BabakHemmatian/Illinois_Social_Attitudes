# import functions and objects
from cli import get_args,dir_path
from utils import parse_range,groups

# import Python packages
import csv
import random
import os
import sys
import time

# Extract and transform CLI arguments 
args = get_args()
years = parse_range(args.years)

# Increase the field size limit to handle larger fields
csv.field_size_limit(sys.maxsize)

### sampling hyper-parameters
num_annot = 2
sample_size = 1500

# survey the language-filtered input files and raise an error if an expected file is missing
language_filtered_path = os.path.join(dir_path.replace("code","data\\data_reddit_curated\\{}\\filtered_language".format(args.group)))
file_list = []
for year in years:
    for month in range(1,13):
        path_ = language_filtered_path+"\\RC_{}-{}.csv".format(year,month)
        if os.path.exists(path_):
            file_list.append(path_)
        else:
            raise Exception("Missing language-filtered file for year {}, month {}".format(year,month))

def get_unique_keywords(keyword_str, max_keywords=100):
    """
    Extract and count unique social group-related keywords from the input string.
    
    Args:
        keyword_str (str): Contains a set of keywords separated by commas
        max_keywords (int): Maximum number of keywords to process
    
    Returns:
        list: List of unique keywords
        int: Number of unique keywords
    """
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
        print(f"Error processing keywords: {e}")
        return [], 0

# Calculate how many samples to take per year, per category (top/bottom/random)
total_samples_per_year = sample_size // len(years)
samples_per_type_per_year = total_samples_per_year // 3

# Dictionary to store final samples for each annotator
all_samples = {}
for i in range(num_annot):
    all_samples[i] = []

# Process each year
def filter_sample_year(year,file_list):
    print(f"\nSampling documents potentially related to the {args.group} social group from year {year}...")

    # Reservoirs (lists) for top, bottom, random
    top_reservoir = []
    bottom_reservoir = []
    random_reservoir = []

    total_docs = 0  # How many docs processed for this year

    # Iterate through each file in the directory
    for file in file_list:
        print(f"Sampling from {file}")
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

                            text = line[2].strip().replace("\n", " ")
                            
                            # If there's a keywords column (index 7), parse it
                            if len(line) > 7:
                                keywords, unique_count = get_unique_keywords(line[7])
                            else:
                                keywords, unique_count = [], 0
                            
                            total_docs += 1

                            # ===========================
                            #    TOP SAMPLES (max)
                            # ===========================
                            if len(top_reservoir) < samples_per_type_per_year:
                                # Not yet filled the top reservoir
                                top_reservoir.append((unique_count, text, keywords, file, original_id))
                                # Keep it sorted in descending order of unique_count
                                top_reservoir.sort(key=lambda x: x[0], reverse=True)
                            else:
                                # If this doc has more unique keywords than the last in top_reservoir
                                if unique_count > top_reservoir[-1][0]:
                                    top_reservoir[-1] = (unique_count, text, keywords, file, original_id)
                                    top_reservoir.sort(key=lambda x: x[0], reverse=True)

                            # ===========================
                            #    BOTTOM SAMPLES (min)
                            # ===========================
                            if len(bottom_reservoir) < samples_per_type_per_year:
                                bottom_reservoir.append((unique_count, text, keywords, file, original_id))
                                bottom_reservoir.sort(key=lambda x: x[0])
                            else:
                                if unique_count < bottom_reservoir[-1][0]:
                                    bottom_reservoir[-1] = (unique_count, text, keywords, file, original_id)
                                    bottom_reservoir.sort(key=lambda x: x[0])

                            # ===========================
                            #    RANDOM SAMPLES
                            # ===========================
                            # Using reservoir sampling
                            if len(random_reservoir) < samples_per_type_per_year:
                                random_reservoir.append((unique_count, text, keywords, file, original_id))
                            else:
                                s = random.randint(0, total_docs - 1)
                                if s < samples_per_type_per_year:
                                    random_reservoir[s] = (unique_count, text, keywords, file, original_id)
                        else:
                            print(f"Skipping line {id_}: insufficient columns ({len(line)} found)")
                    except Exception as e:
                        print(f"Error processing line {id_} in {file}: {e}")
                        continue
        except Exception as e:
            print(f"Error processing file {file}: {e}")
            continue

    print(f"{total_docs} documents processed for year {year}.")

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
        # all_samples[annot] is a list of dict
        all_samples[annot].extend(year_sample_data)


def filter_sample_write(all_samples):
    # ========================================
    # After processing all years, write output
    # ========================================
    for annot in range(num_annot):
        sample_file_path = os.path.join(dir_path, f"filter_sample_{args.group}_{annot}.csv")
        sample_key_file_path = os.path.join(dir_path, f"filter_sample_{args.group}_{annot}.csv")
        
        with open(sample_file_path, "w", encoding='utf-8', newline='') as sample_file, \
            open(sample_key_file_path, "w", encoding='utf-8', newline='') as sample_file_key:
            
            writer = csv.writer(sample_file)
            writer_key = csv.writer(sample_file_key)
            
            # Write headers
            writer.writerow([
                "random_id", "text"
            ])
            # Key file: random_id, file, original_id, keywords, sample_type
            writer_key.writerow(["random_id", "file", "original_id", "keywords", "sample_type"])
            
            # Shuffle samples before writing so we don't group them year by year
            random.shuffle(all_samples[annot])
            
            # Write rows
            for data in all_samples[annot]:
                writer.writerow([
                    data['random_id'],
                    data['text'],
                    "",  # placeholders
                    "",
                    ""
                ])
                writer_key.writerow([
                    data['random_id'],
                    data['file'],
                    data['original_id'],
                    ",".join(data['keywords']),
                    data['sample_type']
                ])

if __name__ == "__main__":
    start_time = time.time()
    for year in years:
        filter_sample_year(year,file_list)
    filter_sample_write(all_samples)
    print(f"Reservoir sampling for the {args.group} social group from {args.years} was finished in {(time.time() - start_time) / 60:.2f} minutes. Total samples per each of the {num_annot} annotators: {len(all_samples[0])}")

# TODO: Save errors to a file rather than just posting them to the screen. The filename should contain the function and the timestamp
# TODO: Use the document id to make sure that none of the documents also appear in more than one reservoir. 