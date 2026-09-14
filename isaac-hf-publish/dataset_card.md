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
    path: ability/RC_*.parquet
- config_name: age
  data_files:
  - split: train
    path: age/RC_*.parquet
- config_name: race
  data_files:
  - split: train
    path: race/RC_*.parquet
- config_name: sexuality
  data_files:
  - split: train
    path: sexuality/RC_*.parquet
- config_name: skin_tone
  data_files:
  - split: train
    path: skin_tone/RC_*.parquet
- config_name: weight
  data_files:
  - split: train
    path: weight/RC_*.parquet
---

# ISAAC: Illinois Social Attitudes Aggregate Corpus

ISAAC is a large, labeled corpus of Reddit discourse about six identity-based
social categories. This is the **gated** Hugging Face mirror; the project also
offers plain HTTP [direct downloads](https://isaac.psychology.illinois.edu/direct-download/)
and a no-code [web app](https://isaac.psychology.illinois.edu/).

> **Access**: this dataset is gated. Request access (you'll be asked to accept the
> Data Use Agreement); approval grants `load_dataset` access with your HF token.

## Configurations (social groups)

One config per social group, monthly from **2007-01 to 2023-12**:
`ability`, `age`, `race`, `sexuality`, `skin_tone`, `weight`.

```python
from datasets import load_dataset

# one group, streaming (no full download):
ds = load_dataset("ISAAC-corpus/reddit", "race", split="train", streaming=True)
for row in ds.take(3):
    print(row["text"], row["score"])

# or materialize a group:
race = load_dataset("ISAAC-corpus/reddit", "race", split="train")
```

## Schema

Each row is a Reddit post (submission or comment) with:

- **Core**: `id`, `parent id`, `text`, `author`, `time`, `subreddit`, `score`,
  `matched_patterns` (keywords that flagged the post as relevant to the group).
- **Labels**: `moralization`; sentiment from Stanza / VADER / TextBlob;
  generalization features (clause-level genericity, eventivity, boundedness,
  habituality, plus per-clause labels); per-emotion scores from multiple models;
  and estimated user `location` (with confidence and runner-up).

See the [repository](https://github.com/BabakHemmatian/Illinois_Social_Attitudes)
for the full data dictionary.

## Provenance & related access

- Web app + direct download: <https://isaac.psychology.illinois.edu/>
- Direct-download docs & manifest: <https://isaac.psychology.illinois.edu/direct-download/>
- Python loader (`isaac-data`) and source: <https://github.com/BabakHemmatian/Illinois_Social_Attitudes>

## Data Use Agreement & citation

Use of ISAAC is governed by the
[Data Use Agreement](https://github.com/BabakHemmatian/Illinois_Social_Attitudes/blob/main/Data_Use_Agreement.md).
Please cite the project in any resulting work (see the repository README for
citation details).
