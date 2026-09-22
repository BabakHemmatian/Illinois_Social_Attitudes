---
license: other
license_name: isaac-data-use-agreement
license_link: https://github.com/BabakHemmatian/Illinois_Social_Attitudes/blob/main/Data_Use_Agreement.md
pretty_name: "ISAAC: Illinois Social Attitudes Aggregate Corpus"
language:
- en
tags:
- reddit
- social-media
- social-attitudes
- moralization
- sentiment
- emotion
- generalization
size_categories:
- 100M<n<1B
extra_gated_heading: "Request access to ISAAC"
extra_gated_prompt: >-
  ISAAC is released under the project Data Use Agreement:
  https://github.com/BabakHemmatian/Illinois_Social_Attitudes/blob/main/Data_Use_Agreement.md
  By requesting access you agree to those terms and to cite the project in any
  resulting work.
extra_gated_fields:
  I agree to the ISAAC Data Use Agreement: checkbox
configs:
- config_name: ability
  data_files:
  - split: train
    path: ability/ALL_*.parquet
- config_name: age
  data_files:
  - split: train
    path: age/ALL_*.parquet
- config_name: race
  data_files:
  - split: train
    path: race/ALL_*.parquet
- config_name: sexuality
  data_files:
  - split: train
    path: sexuality/ALL_*.parquet
- config_name: skin_tone
  data_files:
  - split: train
    path: skin_tone/ALL_*.parquet
- config_name: weight
  data_files:
  - split: train
    path: weight/ALL_*.parquet
---

# ISAAC: Illinois Social Attitudes Aggregate Corpus

ISAAC is a large, labeled corpus of Reddit discourse about six identity-based
social categories. This is the **gated** Hugging Face mirror; the project also
offers plain HTTP [direct downloads](https://isaac.psychology.illinois.edu/direct-download/)
and a no-code [web app](https://isaac.psychology.illinois.edu/).

> **Access**: this dataset is gated. Request access (you'll be asked to accept the
> Data Use Agreement); approval grants `load_dataset` access with your HF token.

## Configurations (social groups)

One config per social group, one file per month from **2007-01 to 2023-12**
(204 months each, 1224 files in total).

| Config | Rows | Size |
| --- | ---: | ---: |
| `ability` | 22,955,382 | 21.3 GB |
| `age` | 280,203,455 | 198.3 GB |
| `race` | 82,348,611 | 41.9 GB |
| `sexuality` | 79,567,199 | 41.7 GB |
| `skin_tone` | 39,628,662 | 23.7 GB |
| `weight` | 22,357,610 | 14.0 GB |
| **total** | **527,060,919** | **341 GB** |

```python
from datasets import load_dataset

# one group, streaming (no full download):
ds = load_dataset("BabakScrapes/isaac-reddit", "race", split="train", streaming=True)
for row in ds.take(3):
    print(row["text"], row["score"])

# or materialize a group:
race = load_dataset("BabakScrapes/isaac-reddit", "race", split="train")
```

A single month can be pulled without touching the rest of a config:

```python
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

path = hf_hub_download("BabakScrapes/isaac-reddit", "race/ALL_2019-01.parquet",
                       repo_type="dataset")
tbl = pq.read_table(path, columns=["id", "text", "Moralization", "location"])
```

## Schema

Each row is a Reddit submission or comment, in 59 columns:

- **Core**: `id`, `parent id`, `text`, `author` (a pseudonymous numeric id),
  `time` (GMT), `subreddit`, `score`, `type` (`comment` or `submission`), and
  `matched patterns`: the keywords that flagged the post as potentially
  relevant to the group before AI-based pruning.
- **Moralization**: `Moralization`, a binary AI estimate.
- **Sentiment**: sentence counts from Stanza (`Sentiment_Stanza_pos/neu/neg`),
  `Sentiment_Vader_compound`, and `Sentiment_TextBlob_Polarity` /
  `_Subjectivity`.
- **Generalization**: `clauses` and `generalization_clause_labels` (one clause
  per line, same order in both), plus counts and proportions for genericity,
  eventivity, boundedness, habituality and `NA`.
- **Emotion**: `<model_no>_<emotion>` for three models × seven emotions.
  Model 1 is [j-hartmann/emotion-english-distilroberta-base](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base),
  model 2 is [SamLowe/roberta-base-go_emotions](https://huggingface.co/SamLowe/roberta-base-go_emotions),
  model 3 is [tae898/emoberta-base](https://huggingface.co/tae898/emoberta-base).
  Models 1 and 3 are softmax probabilities summing to one across the seven
  categories; model 2 scores are independent per-category probabilities and
  need not sum to one.
- **Location**: `location`, `location_prob`, `contender_location`,
  `contender_location_prob`: a *user-level* estimate, so every post by the same
  account carries the same label.

Two column names are easy to mistype: **`matched patterns` and `parent id`
contain a space**, and **`Moralization` is capitalized**.

The authoritative, column-by-column data dictionary is
[`variable_list.md`](https://github.com/BabakHemmatian/Illinois_Social_Attitudes/blob/main/variable_list.md)
in the project repository.

## File format

Parquet, ZSTD-compressed, with ~100,000 rows per row group. These files decode to
tables identical to the SNAPPY-compressed copies served from the
[direct-download endpoint](https://isaac.psychology.illinois.edu/direct-download/):
same schema, same row groups, same values, but they are about 40% smaller on
the wire, so the bytes themselves are not interchangeable with those copies.

## Confidence recalibration (location)

`location_calibration_maps.json` in this repository recalibrates the `location_prob`
column. The location labeler is consistently under-confident, so the raw score ranks
users well but understates the probability that a label is correct.

Six isotonic fits are provided, one per condition and geographic tier:

| Map | Expected calibration error, before -> after |
| --- | --- |
| `standard/top` | 0.0900 -> 0.0112 |
| `standard/region` | 0.1149 -> 0.0070 |
| `standard/state` | 0.5221 -> 0.0116 |
| `masked/top` | 0.1208 -> 0.0044 |
| `masked/region` | 0.1239 -> 0.0174 |
| `masked/state` | 0.3191 -> 0.0139 |

Isotonic regression is monotone, so recalibration never reorders users or changes any
label. It changes only the interpretation of the score. The maps are fitted on a
validation split of held-out authors and scored on a test split, so the improvement is
out of sample.

Apply a map with linear interpolation; no ISAAC code is needed:

```python
import json, numpy as np

maps = json.load(open('location_calibration_maps.json'))['maps']
m = maps['masked/top']            # condition/tier
calibrated = np.interp(raw_score, m['x'], m['y'])
```

Pick the tier from the label itself: a two-letter code is `state`,
`EUROPE`/`AMERICAS`/`ASIA_OCEANIA`/`AFRICA` is `region`, `US`/`NON_US` is `top`, and
`UNK` has no score. Use the **`masked`** maps unless you know the authors state their
location explicitly; masked is the conservative choice and the closer analogue for
corpus authors who never self-disclose. The file's own `method`, `how_to_apply`,
`choosing_a_condition` and `limitations` fields document this alongside the knots.

Calibration holds in aggregate on the population the maps were fitted on, namely
authors whose location was recoverable from explicit self-disclosure. Per-state and
per-subgroup calibration were not assessed. The location model weights themselves are
not distributed here; see the paper's Code Availability section.

## Provenance & related access

- Web app + direct download: <https://isaac.psychology.illinois.edu/>
- Direct-download docs & manifest: <https://isaac.psychology.illinois.edu/direct-download/>
- Python loader (`isaac-data`) and source: <https://github.com/BabakHemmatian/Illinois_Social_Attitudes>
- Questions, access problems, or corpus errata: <isaac.corpus.support@gmail.com>

## Citation

Please cite the ISAAC paper. **One citation covers the whole project**: the
corpus, the pipeline, and every model. Please do not cite this dataset
repository separately; keeping references in one place is what allows the
project's citations to be found together.

```bibtex
@article{hemmatian2026isaac,
  author = {Hemmatian, Babak and Hadjarab, Sarah and Chen, Jessica and Kurdi, Benedek},
  title  = {The {Illinois} Social Attitudes Aggregate Corpus ({ISAAC}): An Open Tool and Reproducible Pipeline for Analyzing Social Group Discourse at Scale},
  year   = {2026},
  note   = {Manuscript submitted for publication}
}
```

## Data Use Agreement

Use of ISAAC is governed by the
[Data Use Agreement](https://github.com/BabakHemmatian/Illinois_Social_Attitudes/blob/main/Data_Use_Agreement.md),
which you accept when requesting access. Agreeing to cite the project in any
resulting work is one of its terms.
