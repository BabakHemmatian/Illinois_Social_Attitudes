### Imports

# import functions and objects
from cli import get_args, DATA_DIR

# import python packages
from sklearn.metrics import cohen_kappa_score
from scipy.stats import pearsonr
import csv
csv.field_size_limit(2**31 - 1)  # Increase the field size limit to handle larger fields
import os

### Agreement Metric Hyperparameters

num_annot = 2  # number of annotators
# NOTE: This script currently only supports two annotators and the canonical
# double-rated samples under data/data_relevance_ratings/<type>/.

### Argument Handling

args = get_args()
group = args.group
type_ = args.type

### Path Handling

# Where to find the rated relevance samples. Defaults to the canonical location
# data/data_relevance_ratings/<type>/ where the original double-rated samples live.
if not args.input:
    ratings_path = DATA_DIR / "data_relevance_ratings" / type_
else:
    ratings_path = args.input

### Label binarization
# Match the convention used elsewhere in the codebase: only the literal "1" counts
# as relevant; "0", "x", "-1", blanks, etc. all become 0.
def binarize(cell: str) -> int:
    return 1 if str(cell).strip() == "1" else 0

### Main Evaluation

# Per-rater dict: random_id -> 0/1
ratings = {i: {} for i in range(num_annot)}

for rater in range(num_annot):
    fname = os.path.join(
        ratings_path,
        f"relevance_sample_{group}_{rater}_rated.csv",
    )
    with open(fname, "r", encoding="utf-8-sig", errors="ignore") as f:
        reader = csv.reader(f)
        for idx, line in enumerate(reader):
            if idx == 0 or not line:
                continue
            rid_raw = line[0].strip()
            if not rid_raw:
                continue
            try:
                rid = int(rid_raw)
            except ValueError:
                raise Exception(
                    f"Error parsing annotator {rater}'s row {idx}: non-integer id={line[0]!r}"
                )
            v = binarize(line[2] if len(line) >= 3 else "")
            # If the same random_id appears more than once in the same file (a small
            # number of duplicates exist in some samples), OR the labels: any "1"
            # wins over "0".
            ratings[rater][rid] = max(ratings[rater].get(rid, 0), v)

# Restrict to documents both raters rated; warn about any asymmetric IDs
common_ids = sorted(set(ratings[0].keys()) & set(ratings[1].keys()))
only_in_0 = set(ratings[0].keys()) - set(ratings[1].keys())
only_in_1 = set(ratings[1].keys()) - set(ratings[0].keys())
for rid in only_in_0:
    print(f"Warning! Entry with ID {rid} only rated by annotator 0")
for rid in only_in_1:
    print(f"Warning! Entry with ID {rid} only rated by annotator 1")

vector_0 = [ratings[0][rid] for rid in common_ids]
vector_1 = [ratings[1][rid] for rid in common_ids]

n = len(common_ids)
agree = sum(1 for a, b in zip(vector_0, vector_1) if a == b) / n if n else float("nan")
kappa = cohen_kappa_score(vector_0, vector_1)
pear = pearsonr(vector_0, vector_1)

print(f"Group: {group} ({type_})")
print(f"N (both raters): {n}")
print(f"Rater 0 relevant rate: {sum(vector_0)/n:.3f}")
print(f"Rater 1 relevant rate: {sum(vector_1)/n:.3f}")
print(f"Raw agreement: {agree:.3f}")
print(f"Cohen's Kappa for interrater agreement: {kappa:.4f}")
print(f"Pearson r: {pear.statistic:.4f}  (p={pear.pvalue:.2e})")
