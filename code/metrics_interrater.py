### Imports

# import functions and objects
from cli import get_args,DATA_DIR

# import python packages
from sklearn.metrics import cohen_kappa_score
from scipy.stats import pearsonr
import csv
csv.field_size_limit(2**31 - 1) # Increase the field size limit to handle larger fields
import os

### Agreement Metric Hyperparameters

num_annot = 2 # number of annotators
# NOTE: This script currently only supports two annotators. And that a '_rated' tag has been added to the name of the files outputted by 'filter_sample' after they were rated.


### Argument Handling

args = get_args()
group = args.group
type_ = args.type
tag = 0 # identifies the 'filter_sample' output version for the rated files to be evaluated

### Path Handling

# where to find the rated relevance samples
if not args.input:    
    ratings_path = DATA_DIR / "samples" / group / type_
else:
    ratings_path = args.input

### Main Evaluation

# start a dictionary for storing each annoator's ratings
ratings = {i:{} for i in range(num_annot)}

# extract and align the annotators' ratings
for rater in range(num_annot):
    with open(os.path.join(ratings_path,f"filter_sample_{rater}_v{tag}_rated.csv"),"r", encoding='utf-8',errors='ignore') as f:
        reader = csv.reader(f)
        for idx,line in enumerate(reader):
            if idx != 0 and len(line) > 0:
                try:
                    ratings[rater][int(line[0].strip())] = int(line[2].strip())
                except:
                    raise Exception(f"Error processing annotator {rater}'s response on line {idx}, with the following contents: {line}")

# confirm that the vectors are of the same length
assert len(ratings[0]) == len(ratings[1])

# see if there is any mismatch in terms of included comments between the two samples
for i in ratings[0]:
    if i not in ratings[1]:
        print(f'Warning! Unmatched entry with ID {i}')

# calculate and print Cohen's kappa interrater agreement score
vector_0 = []
vector_1 = []
for id_ in ratings[0]:
    try:
        vector_0.append(ratings[0][id_])
        vector_1.append(ratings[1][id_])
    except Exception as e:
        print(f"Warning! Skipping rating id={id_!r} due to {type(e).__name__}: {e}")

print(f"Cohen's Kappa for interrater agreement: {cohen_kappa_score(vector_0,vector_1)}")
print(pearsonr(vector_0,vector_1))