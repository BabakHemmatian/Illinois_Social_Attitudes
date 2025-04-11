import argparse
import csv
import random
import math
import os

def parse_range(value):
    """Parses a single integer or a range (e.g., '2007' or '2008-2010') into a list of integers,
    ensuring the start is ≥ 2007 and the end is ≤ 2023."""
    try:
        
        if '-' in value:  # Handling a range like "2008-2010"
            start, end = map(int, value.split('-'))
            if start > end:
                raise argparse.ArgumentTypeError(f"Invalid range '{value}': start must be ≤ end.")
        else:  # Handling a single integer
            start = end = int(value)

        # Enforce constraints on the range
        if start < 2007:
            raise argparse.ArgumentTypeError(f"Invalid value '{value}': years must be ≥ 2007.")
        if end > 2023:
            raise argparse.ArgumentTypeError(f"Invalid value '{value}': years must be ≤ 2023.")

        if start == end:
            return [int(value)]
        else:
            return list(range(start, end + 1))
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid value '{value}': must be an integer or a range (e.g., 2007 or 2008-2010).")

# Helper function to load terms from a file
def load_terms(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.lower().strip() for line in f if line.strip()]
    
# The list of social groups. Marginalized groups always listed first.
groups = {"sexuality":['gay','straight'],'age':['old','young'],'weight':['fat','thin'],'ability':['disabled','abled'],'race':['black','white'],'skin_tone':['dark','light']}

# the information we store for each comment in ISAAC
headers = ["id", "parent id", "text", "author", "time", "subreddit", "score", "matched patterns"]

def split_dataset_to_file(file,list):
    with open(file,"w",encoding='utf-8',errors='ignore',newline="") as f:
        if "text" in file:
            writer = csv.writer(f)
            for i in list:
                writer.writerow([i])
        elif "label" in file:
            for i in list:
                print(i,file=f)

def split_dataset_from_file(file):
    list_ = []
    with open(file,"r",encoding='utf-8',errors='ignore') as f:
        if "text" in file:
            reader = csv.reader(f)
            for i in reader:
                list_.append(i[0])
        elif "label" in file:
            for i in f:
                list_.append(int(i.strip()))
    return list_

# custom function for doing a training/validation/test split
def dataset_split(texts,labels,proportion):
    training_id = random.sample(range(len(texts)),math.floor(proportion*len(texts)))
    test_id = [i for i in range(len(texts)) if i not in training_id]
    training_texts = []
    training_labels = []
    test_texts = []
    test_labels = []
    
    for idx,i in enumerate(texts):
        if idx in training_id:
            training_texts.append(i)
            training_labels.append(labels[idx])
        elif idx in test_id:
            test_texts.append(i)
            test_labels.append(labels[idx])
        else:
            raise Exception
    
    return training_texts,test_texts,training_labels,test_labels

def f1_calculator(labels,predictions):
    '''texts (list)
       predictions (list)'''
    metrics = {i:0 for i in ['tp','tn','fp','fn']}

    for idx,prediction in enumerate(predictions):
        
            if labels[idx] == 0:
                if prediction == 0:
                    metrics['tn'] += 1
                elif prediction == 1:
                    metrics['fp'] += 1
                else:
                    raise Exception
            elif labels[idx] == 1:
                if prediction == 0:
                    metrics['fn'] += 1
                elif prediction == 1:
                    metrics['tp'] += 1
                else:
                    raise Exception

    precision = float(metrics['tp']) / float(metrics['tp'] + metrics['fp'])
    recall = float(metrics['tp']) / float(metrics['tp'] + metrics['fn'])
    F_1 = 2 * float(precision * recall) / float(precision + recall)

    return precision, recall, F_1

def check_reqd_files(years=None,check_path=None):
    file_list = []
    for year in years:
        for month in [f"{month_:02}" for month_ in range(1,13)]:
            path_ = check_path+"\\RC_{}-{}.csv".format(year,month)
            if os.path.exists(path_):
                file_list.append(path_)
            else:
                raise Exception("Missing pre-filtered file for year {}, month {}, expected in: {}".format(year,month,path_))
    return file_list

def log_report(message):
    """
    Log a message with a timestamp to the report file and print it.
    Ensures output is flushed immediately.
    """
    # Use unified timestamp format %Y-%m-%d %H:%M:%S
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(report_file_path, 'a', encoding='utf-8', newline='') as report_file:
        writer = csv.writer(report_file, delimiter='\t')
        writer.writerow([timestamp, message])
    print(f"{timestamp} - {message}")
    sys.stdout.flush()
