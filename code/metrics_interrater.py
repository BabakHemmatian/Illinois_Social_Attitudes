# import functions and objects
from cli import get_args,dir_path

# import python packages
from sklearn.metrics import cohen_kappa_score
from scipy.stats import pearsonr
import csv
from pathlib import Path

num_annot = 2 # number of annotators
# NOTE: This script currently only supports two annotators

args = get_args()
group = args.group
type_ = args.type

# set path variables
CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

# where to find the rated relevance samples
ratings_path = DATA_DIR / "data_relevance_ratings" / type_ 

# start a dictionary for storing each annoator's ratings
ratings = {i:{} for i in range(num_annot)}

# extract and align the annotators' ratings
for rater in range(num_annot):
    with open(ratings_path+"relevance_sample_{}_{}.csv".format(group,rater),"r", encoding='utf-8',errors='ignore') as f:
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
    except:
        print(id_)

print(f"Cohen's Kappa for interrater agreement: {cohen_kappa_score(vector_0,vector_1)}")
print(pearsonr(vector_0,vector_1))
