# import functions and objects
from cli import get_args,dir_path
from utils import parse_range,headers

# import Python packages
import fasttext # issue installing it
import os
import csv
import time
import sys

# Increase the field size limit to handle larger fields
csv.field_size_limit(sys.maxsize) 

# Extract and transform CLI arguments 
args = get_args()
years = parse_range(args.years)

# Load the fastText language identification model
model = fasttext.load_model(dir_path.replace("code","models\\filter_language.bin"))

# define a function that applies the fasttext model to a given text
def detect_language(text):
    predictions = model.predict(text)
    # The language code is returned with a prefix "__label__", which we remove
    return predictions[0][0].replace('__label__', '') 

# survey the keyword-filtered input files and raise an error if an expected file is missing
keyword_filtered_path = os.path.join(dir_path.replace("code","data\\data_reddit_curated\\{}\\filtered_keywords".format(args.group)))
file_list = []
for year in years:
    for month in range(1,13):
        path_ = keyword_filtered_path+"\\RC_{}-{}.csv".format(year,month)
        if os.path.exists(path_):
            file_list.append(path_)
        else:
            raise Exception("Missing keyword-filtered file for year {}, month {}".format(year,month))

# Prepare and survey the output path
output_path = os.path.join(dir_path.replace("code","data\\data_reddit_curated\\{}\\filtered_language".format(args.group)))
os.makedirs(output_path, exist_ok=True)

# function for language filtering a single file
def filter_language_file(file):

    print("Started language filtering {}".format(file))

    try:
        with open(file, "r", encoding='utf-8', errors='ignore') as input_file, \
            open("{}\\{}".format(output_path,file.split("keywords\\")[1]),"w",encoding='utf-8',errors='ignore',newline='') as output_file:
            start_time = time.time()

            error_counter = 0
            filtered_counter = 0
            passed_counter = 0

            # Replace NUL characters with empty strings before passing to csv.reader
            reader = csv.reader((line.replace('\0', '') for line in input_file))
            writer = csv.writer(output_file)

            for id_, line in enumerate(reader):
                if id_ == 0:
                    writer.writerow(headers)
                else:
                    try:
                        if detect_language(line[2]) == 'en':
                            writer.writerow(line)
                            passed_counter += 1

                    except IndexError as e:
                        print(f"Skipping line {id_ + 1} due to error: {e}")
                        error_counter += 1
                        continue  # Skip this line and continue with the next one

                    filtered_counter += 1

            end_time = time.time()
            print(f"Finished language filtering {file} in {(end_time - start_time)/60} minutes. # of evaluations: {filtered_counter}, # of English posts: {passed_counter}, # of errors: {error_counter}")
    except:
        raise Exception("Error in the language filtering of {}".format(file))

if __name__ == "__main__":
    start_time = time.time()
    for file in file_list:
        filter_language_file(file)
    print(f"Language filtering for the {args.group} social group for {args.years} was finished in {(time.time() - start_time) / 60:.2f} minutes")

# TODO: Save errors to a file rather than just posting them to the screen. The filename should contain the function and the timestamp
# TODO: Add a warning if a particular month is missing from the output of the function.
# TODO: Add overall statistics to the final report