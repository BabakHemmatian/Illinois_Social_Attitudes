---
license: cc-by-4.0
language:
- en
base_model:
- FacebookAI/roberta-base
pipeline_tag: token-classification
library_name: transformers
inference: false
tags:
- social-science
- computational-social-science
- clause-segmentation
- discourse-modes
- isaac
- disco
model-index:
- name: isaac-generalization-segmentation
  results:
  - task:
      type: token-classification
      name: Clause segmentation
    dataset:
      type: disco
      name: DiSCo gold corpus (human-verified clauses)
      split: test
    metrics:
    - type: recall
      name: Clause-span coverage
      value: 0.955
---

# ISAAC generalization — clause segmenter

> **Same weights as the public DiSCo release.** This repository is the
> ISAAC-facing copy of [`BabakScrapes/disco-clause-segmenter`](https://huggingface.co/BabakScrapes/disco-clause-segmenter); the checkpoints are
> identical. That card carries the full model description and the DiSCo corpus
> details. This one documents how the model is used inside the ISAAC pipeline and
> restates the performance figures reported in the ISAAC manuscript. Cite
> whichever matches your use; if you are using ISAAC labels, cite both.

Token classifier that splits English text into **clauses**, the unit the
[situation-entity classifier](https://huggingface.co/ISAAC-corpus/isaac-generalization) then labels. Together the two models
produce the clause-level generalization columns of the [Illinois Social Attitudes
Aggregate Corpus (ISAAC)](https://github.com/BabakHemmatian/Illinois_Social_Attitudes).

## Decoding

The model emits one tag per word (majority-voted across sub-word tokens). The
decoder in `code/label_generalization.py` reads them as follows:

* tag `2` marks a **clause-final** word — it closes the clause it appears in;
* tags `0` and `1` mark clause-internal words;
* a new clause opens on the word after a `2`;
* when a word receives no aligned prediction, it defaults to `1`.

Reference implementation, including the sub-word majority vote:

```python
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

REPO = "ISAAC-corpus/isaac-generalization-segmentation"
tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=True,
                                          add_prefix_space=True)
model = AutoModelForTokenClassification.from_pretrained(REPO).eval()

text = "My gay neighbor watered my plants while I was traveling"
words = text.split()
enc = tokenizer(words, is_split_into_words=True, return_tensors="pt",
                truncation=True, max_length=512, padding="max_length")
word_ids = enc.word_ids(batch_index=0)
with torch.no_grad():
    token_preds = model(**enc).logits[0].argmax(dim=-1).tolist()

per_word = [[] for _ in words]
for token_idx, word_id in enumerate(word_ids):
    if word_id is not None:
        per_word[word_id].append(token_preds[token_idx])
tags = [max(set(p), key=p.count) if p else 1 for p in per_word]

clauses, current, prev = [], [], 2
for word, tag in zip(words, tags):
    if prev == 2:
        current = []
    current.append(word)
    if tag == 2 and prev in (0, 1):
        clauses.append(" ".join(current))
        current = []
    prev = tag
if current:
    clauses.append(" ".join(current))

print(clauses)
```

Texts longer than 200 words are split on sentence boundaries before segmentation
in the ISAAC pipeline; see `generalization.py` in the [Space](https://huggingface.co/spaces/BabakScrapes/isaac-classifiers).

## Training data

The DiSCo corpus of opinionated, mixed-register English text (Hemmatian, 2022),
with human-verified clause boundaries. See the
[DiSCo card](https://huggingface.co/BabakScrapes/disco-clause-segmenter).

## Evaluation

Two `FacebookAI/roberta-base` models run in sequence: a clause segmenter, then a
18-way situation-entity classifier. Both are the same weights as
the public DiSCo release — see the
[clause segmenter](https://huggingface.co/BabakScrapes/disco-clause-segmenter) and
[situation-entity classifier](https://huggingface.co/BabakScrapes/disco-se-classifier) cards for the full model
description and the corpus they were trained on.

**Segmentation.** The segmenter covered most of the target clause span for 95.5% of human-verified clauses.

**Classification**, on the held-out 10% of the disco gold corpus (*k* ≈ 2,357):

| Target | Macro F1 | Accuracy |
| --- | --- | --- |
| Full situation entity (18-way) | .514 | .737 |
| Genericity (2-way) | .852 | .860 |
| Eventivity (2-way) | .879 | .894 |
| Boundedness / habituality (4-way) | .804 | .850 |

The 18-way macro F1 of .514 is held down by heavy
label imbalance across the rarer situation-entity types. The three collapsed
features are what ISAAC actually reports, and they are the numbers to rely on.

## Intended use

Clause segmentation as the first stage of the ISAAC generalization pipeline.

## Out-of-scope use

Not a general-purpose syntactic parser, constituency parser, or sentence
splitter. It targets the specific clause unit the situation-entity framework
requires, which does not always coincide with a syntactic clause. English only.

## Limitations and bias

* Coverage is reported as the share of human-verified clauses whose target span
  the model mostly recovered; it is not an exact-match boundary F1, and exact
  boundaries are frequently off by a word.
* Segmentation errors propagate into every downstream generalization label.
* Long inputs are truncated at 512 sub-word tokens, which is why the ISAAC
  pipeline pre-splits at ~200 words.
* Reddit text is noisy — missing punctuation, run-on constructions, markup — and
  segmentation quality degrades accordingly.

## Links

| | |
| --- | --- |
| Try it without code | [ISAAC Text Classifiers Space](https://huggingface.co/spaces/BabakScrapes/isaac-classifiers) |
| Pipeline source, keyword lists, pattern sets | [GitHub](https://github.com/BabakHemmatian/Illinois_Social_Attitudes) |
| Corpus download, samples, SQL playground | [https://isaac.psychology.illinois.edu/](https://isaac.psychology.illinois.edu/) |
| Data Use Agreement | [Data_Use_Agreement.md](https://github.com/BabakHemmatian/Illinois_Social_Attitudes/blob/main/Data_Use_Agreement.md) |

## Citation

Please cite the ISAAC paper. **One citation covers the whole project** — the
corpus, the pipeline, and every model. Please do not cite this model repository
separately; keeping references in one place is what allows the project's
citations to be found together.

```bibtex
@article{hemmatian2026isaac,
  author = {Hemmatian, Babak and Hadjarab, Sarah and Chen, Jessica and Kurdi, Benedek},
  title  = {The {Illinois} Social Attitudes Aggregate Corpus ({ISAAC}): An Open Tool and Reproducible Pipeline for Analyzing Social Group Discourse at Scale},
  year   = {2026},
  note   = {Manuscript submitted for publication}
}
```

## License

Released under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).
You may use, share, and adapt these weights, including commercially, provided
you give appropriate credit — see Citation above.

The ISAAC corpus itself is governed separately by the project
[Data Use Agreement](https://github.com/BabakHemmatian/Illinois_Social_Attitudes/blob/main/Data_Use_Agreement.md).
