---
license: cc-by-4.0
language:
- en
base_model:
- FacebookAI/roberta-base
pipeline_tag: text-classification
library_name: transformers
inference: false
tags:
- social-science
- computational-social-science
- situation-entity
- genericity
- discourse-modes
- isaac
- disco
model-index:
- name: isaac-generalization
  results:
  - task:
      type: text-classification
      name: Situation-entity classification (18-way)
    dataset:
      type: disco
      name: "DiSCo gold corpus: held-out 10%"
      split: test
    metrics:
    - type: accuracy
      name: Accuracy (18-way)
      value: 0.737
    - type: f1
      name: Macro F1 (18-way)
      value: 0.514
  - task:
      type: text-classification
      name: Collapsed generalization features
    dataset:
      type: disco
      name: "DiSCo gold corpus: held-out 10%"
      split: test
    metrics:
    - type: f1
      name: Macro F1 (genericity)
      value: 0.852
    - type: accuracy
      name: Accuracy (genericity)
      value: 0.860
    - type: f1
      name: Macro F1 (eventivity)
      value: 0.879
    - type: accuracy
      name: Accuracy (eventivity)
      value: 0.894
    - type: f1
      name: Macro F1 (boundedness / habituality)
      value: 0.804
    - type: accuracy
      name: Accuracy (boundedness / habituality)
      value: 0.850
---

# ISAAC generalization: situation-entity classifier

> **Same weights as the public DiSCo release.** This repository is the
> ISAAC-facing copy of [`BabakScrapes/disco-se-classifier`](https://huggingface.co/BabakScrapes/disco-se-classifier); the checkpoints are
> identical. That card carries the full model description and the DiSCo corpus
> details. This one documents how the model is used inside the ISAAC pipeline and
> restates the performance figures reported in the ISAAC manuscript. Cite
> whichever matches your use; if you are using ISAAC labels, cite both.

Clause-level classifier that assigns one of **18 situation-entity types** to an
English clause, extending Smith's (2003) discourse-mode framework with
boundedness distinctions. In the [Illinois Social Attitudes Aggregate Corpus
(ISAAC)](https://github.com/BabakHemmatian/Illinois_Social_Attitudes) the 18 types are collapsed into three linguistic features
that together capture **how generalized a statement is**:

* **Genericity**: is the main referent a generic category (*gay people*) or a
  specific individual (*my neighbor*)?
* **Eventivity**: a stable state (*God is benevolent*) or a transient event
  (*I went to Nebraska*)?
* **Boundedness / habituality**: for eventive clauses, temporally bounded
  (*I ate this morning*), unbounded (*God loves us*), or habitually recurring
  (*I went there for years*)?

The most anecdotal content is specific entities in bounded, non-habitual events;
the most generalized is generic categories in stative or habitual clauses. This
distinction matters for research on the persuasiveness of attitude-relevant
communication, and no other corpus of social group discourse carries it.

Run this model **after** the
[clause segmenter](https://huggingface.co/ISAAC-corpus/isaac-generalization-segmentation), which splits text into the clauses this model
scores.

## Labels

The released `config.json` in this repository maps indices to readable names.
The full mapping, and the three-feature decomposition ISAAC uses:

| Index | Situation entity | Genericity | Eventivity | Boundedness |
| --- | --- | --- | --- | --- |
| 0 | `BOUNDED EVENT (SPECIFIC)` | specific | dynamic | episodic |
| 1 | `BOUNDED EVENT (GENERIC)` | generic | dynamic | episodic |
| 2 | `UNBOUNDED EVENT (SPECIFIC)` | specific | dynamic | static |
| 3 | `UNBOUNDED EVENT (GENERIC)` | generic | dynamic | static |
| 4 | `BASIC STATE` | specific | stative | static |
| 5 | `COERCED STATE (SPECIFIC)` | specific | dynamic | static |
| 6 | `COERCED STATE (GENERIC)` | generic | dynamic | static |
| 7 | `PERFECT COERCED STATE (SPECIFIC)` | specific | dynamic | episodic |
| 8 | `PERFECT COERCED STATE (GENERIC)` | generic | dynamic | episodic |
| 9 | `GENERIC SENTENCE (DYNAMIC)` | generic | dynamic | habitual |
| 10 | `GENERIC SENTENCE (STATIC)` | generic | stative | static |
| 11 | `GENERIC SENTENCE (HABITUAL)` | generic | stative | habitual |
| 12 | `GENERALIZING SENTENCE (DYNAMIC)` | specific | dynamic | habitual |
| 13 | `GENERALIZING SENTENCE (STATIVE)` | specific | stative | habitual |
| 14 | `QUESTION` | NA | NA | NA |
| 15 | `IMPERATIVE` | NA | NA | NA |
| 16 | `NONSENSE` | NA | NA | NA |
| 17 | `OTHER` | NA | NA | NA |

`QUESTION`, `IMPERATIVE`, `NONSENSE`, and `OTHER` are non-statements, for which
the three features are undefined. ISAAC counts them separately rather than
folding them into the proportions.

## Usage

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO = "ISAAC-corpus/isaac-generalization"
tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=True,
                                          add_prefix_space=True)
model = AutoModelForSequenceClassification.from_pretrained(REPO).eval()

clauses = ["Gay people are nice", "my gay neighbor watered my plants"]
enc = tokenizer(clauses, padding="max_length", truncation=True, max_length=128,
                return_tensors="pt")
with torch.no_grad():
    pred_ids = model(**enc).logits.argmax(dim=-1).tolist()

for clause, idx in zip(clauses, pred_ids):
    print(model.config.id2label[idx], "|", clause)
```

For the full text-to-features pipeline (segmentation, batching, long-text
splitting, and per-text aggregation), see `code/label_generalization.py` in the
[project repository](https://github.com/BabakHemmatian/Illinois_Social_Attitudes) or `generalization.py` in the
[Space](https://huggingface.co/spaces/BabakScrapes/isaac-classifiers).

## Training data

Pre-trained on the SitEnt corpus, then fine-tuned on the DiSCo corpus of
opinionated, mixed-register English text (Hemmatian, 2022). See the
[DiSCo card](https://huggingface.co/BabakScrapes/disco-se-classifier) for the corpus description.

## Evaluation

Two `FacebookAI/roberta-base` models run in sequence: a clause segmenter, then a
18-way situation-entity classifier. Both are the same weights as
the public DiSCo release; see the
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

Clause-level annotation of English text for genericity, eventivity, and
boundedness/habituality, in aggregate research designs, the role it plays in
ISAAC, where it labels clauses across 527M posts.

## Out-of-scope use

* **Clause-level, not document-level.** Feed it clauses from the segmenter, not
  whole posts. Scoring an unsegmented paragraph will not give a meaningful label.
* **Not a measure of truth, quality, or bias.** A generic statement is not
  thereby a stereotype, and a specific one is not thereby accurate.
* **Not for individual-level decisions.** The 18-way macro F1 of .514
  is far too low for any per-clause consequential use; the collapsed features are
  intended for aggregate proportions over large samples.

## Limitations and bias

* **Heavy label imbalance.** The rarer situation-entity types are poorly
  represented in training, which is why 18-way accuracy (.737) far
  exceeds 18-way macro F1 (.514). Per-class performance on rare types is
  correspondingly weak. Use the collapsed features.
* **Single-topic fine-tuning.** DiSCo is opinionated, mixed-register text on one
  controversial policy topic. Cross-genre transfer is expected given the formal,
  content-agnostic nature of the target features, but it has not been separately
  quantified for Reddit discourse about social groups.
* **English only.**
* **Errors compound.** ISAAC's generalization columns are the product of two
  models in series; segmentation errors propagate into classification.

## Links

| | |
| --- | --- |
| Try it without code | [ISAAC Text Classifiers Space](https://huggingface.co/spaces/BabakScrapes/isaac-classifiers) |
| Pipeline source, keyword lists, pattern sets | [GitHub](https://github.com/BabakHemmatian/Illinois_Social_Attitudes) |
| Corpus download, samples, SQL playground | [https://isaac.psychology.illinois.edu/](https://isaac.psychology.illinois.edu/) |
| Data Use Agreement | [Data_Use_Agreement.md](https://github.com/BabakHemmatian/Illinois_Social_Attitudes/blob/main/Data_Use_Agreement.md) |
| Questions about the models or the corpus | [isaac.corpus.support@gmail.com](mailto:isaac.corpus.support@gmail.com) |

## Citation

Please cite the ISAAC paper. **One citation covers the whole project**: the
corpus, the pipeline, and every model. Please do not cite this model repository
separately; keeping references in one place is what allows the project's
citations to be found together.

```bibtex
@article{hemmatian2026isaac,
  author  = {Hemmatian, Babak and Hadjarab, Sarah and Chen, Jessica and Kurdi, Benedek},
  title   = {The {Illinois} Social Attitudes Aggregate Corpus ({ISAAC}): An Open Tool and Reproducible Pipeline for Analyzing Social Group Discourse at Scale},
  year    = {2026},
  journal = {arXiv},
  eprint  = {2609.27059},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  doi     = {10.48550/arXiv.2609.27059},
  url     = {https://arxiv.org/abs/2609.27059}
}
```

## License

Released under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).
You may use, share, and adapt these weights, including commercially, provided
you give appropriate credit; see Citation above.

The ISAAC corpus itself is governed separately by the project
[Data Use Agreement](https://github.com/BabakHemmatian/Illinois_Social_Attitudes/blob/main/Data_Use_Agreement.md).
