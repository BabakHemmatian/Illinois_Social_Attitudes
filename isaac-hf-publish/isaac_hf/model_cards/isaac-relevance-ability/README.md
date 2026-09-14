---
license: other
license_name: isaac-data-use-agreement
license_link: https://github.com/BabakHemmatian/Illinois_Social_Attitudes/blob/main/Data_Use_Agreement.md
language:
- en
base_model:
- FacebookAI/roberta-large
pipeline_tag: text-classification
library_name: transformers
inference: false
tags:
- social-science
- computational-social-science
- reddit
- content-filtering
- relevance-classification
- isaac
extra_gated_heading: "Request access to the ISAAC relevance classifier (ability)"
extra_gated_prompt: >-
  These weights are released under the ISAAC Data Use Agreement:
  https://github.com/BabakHemmatian/Illinois_Social_Attitudes/blob/main/Data_Use_Agreement.md
  By requesting access you agree to those terms and to cite the ISAAC corpus
  paper in any resulting work.
extra_gated_fields:
  Name: text
  Institution or affiliation: text
  Intended use: text
  I agree to the ISAAC Data Use Agreement: checkbox
model-index:
- name: isaac-relevance-ability
  results:
  - task:
      type: text-classification
      name: Social-group relevance classification
    dataset:
      type: isaac-relevance-annotations
      name: "ISAAC relevance annotations (ability) — Held-out 10% of the rated training sample"
      split: test
    metrics:
    - type: precision
      name: Precision (relevant class)
      value: 0.815
    - type: recall
      name: Recall (relevant class)
      value: 0.880
    - type: f1
      name: F1 (relevant class)
      value: 0.846
---

# ISAAC relevance classifier — Ability (abled vs. disabled)

Binary classifier that decides whether an English text is **relevant to the
ability social group distinction** — that is, whether it actually
discusses people in terms of that attribute, as opposed to merely containing a
keyword that can also mean something else.

This is stage 3 of the four-stage filtering pipeline that built the [Illinois
Social Attitudes Aggregate Corpus (ISAAC)](https://github.com/BabakHemmatian/Illinois_Social_Attitudes), a corpus of 527,060,919
Reddit posts (2007–2023) covering six social group distinctions.

**Labels:** `0` = not relevant, `1` = relevant.

## What problem it solves

Keyword filters for social-group discourse are notoriously imprecise: "black" in
a chess match, "disabled" in a software changelog, "gay" as a Filipino or French
word. Such posts carry no signal about social groups, so they inflate standard
errors, depress reliability, and suppress real effects — and most published work
never reports how many of them survive. Because this model reads each keyword in
the context of the full post, it separates the intended sense from the unintended
one in a way a static list cannot.


## Usage

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO = "ISAAC-corpus/isaac-relevance-ability"
tokenizer = AutoTokenizer.from_pretrained(REPO)
model = AutoModelForSequenceClassification.from_pretrained(REPO).eval()

texts = ["My grandmother is using a wheelchair for the first time this year."]
enc = tokenizer(texts, padding=True, truncation=True, max_length=512,
                return_tensors="pt")
with torch.no_grad():
    probs = torch.softmax(model(**enc).logits, dim=1)

for text, prob in zip(texts, probs):
    label = "relevant" if int(prob.argmax()) == 1 else "not relevant"
    print(f"{label}  (P(relevant)={prob[1]:.3f})  {text}")
```

## Training data

~1,500 posts drawn from the keyword-filtered ISAAC corpus, sampled at random with
stratification by year and by per-post keyword-match count, so the training set
spans the full 2007–2023 range and the full range of keyword density rather than
over-representing heavily matched posts.

Each post was rated for relevance to the ability distinction by **two
independent trained annotators**. Annotator training involved discussing the
rating instructions (Appendix B of the manuscript) and working through examples
together. Labels marked unclear were treated as irrelevant for training, and
residual disagreements were resolved in favor of relevance.

Interrater reliability on this distinction: *k* = 1,476 double-rated posts,
98.2% raw agreement, Cohen's κ = .960.

The annotated validation data are released alongside the corpus.

## Evaluation

Base model `FacebookAI/roberta-large`.

| Evaluation slice | *k* | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| Held-out 10% of the rated training sample | 148 | .815 | .880 | .846 |

**Human agreement on the training data.** Two trained annotators rated
*k* = 1,476 posts: 98.2% raw agreement,
Cohen's κ = .960.

**What the full filter delivers.** Held-out precision describes this model in
isolation. In the released corpus this classifier is one of four filtering
stages, and a double-rated audit of the finished
ability data (*k* = 100)
found 6.0% of comments irrelevant under the stringent
rule and 3.0% under the lenient rule
(4.0% for submissions). That end-to-end figure, not
the F1 above, is the number to quote when describing corpus quality.

## Intended use

Screening English social-media text for relevance to the ability
distinction, as a preprocessing step before substantive analysis — the role it
plays in ISAAC. It is a **topical relevance** filter.

## Out-of-scope use

This model does **not** measure attitudes, sentiment, toxicity, or bias. A post
labeled relevant may be supportive, hostile, or neutral; relevance says only that
the post is *about* the distinction. Do not use its output as a proxy for
prejudice, and do not use it to make decisions about individuals.

It was trained on Reddit text from 2007–2023 and on the ISAAC keyword lists.
Performance on other platforms, other registers, other languages, or on text that
was not first keyword-filtered is unknown and likely lower. It is not a
general-purpose "is this about ability" classifier for arbitrary
input.

## Limitations and bias

* **Reddit-shaped.** Reddit's user base is not representative of any general
  population, and its norms shifted across the 17 years covered. The model
  inherits that distribution.
* **Keyword-conditioned.** It was trained on posts that already matched an ISAAC
  keyword. It has not been evaluated on text that contains no such keyword.
* **Trained on human judgment, which varied.** Cohen's κ of .960 on the
  training sample means trained annotators themselves disagreed on some posts;
  no classifier trained on those labels can be cleaner than they were.
* **Annotation conventions push toward recall at training time.** Unclear posts
  were coded irrelevant and disagreements resolved toward relevance, both
  deliberate choices documented in the manuscript.

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

Released under the project
[Data Use Agreement](https://github.com/BabakHemmatian/Illinois_Social_Attitudes/blob/main/Data_Use_Agreement.md). Using these weights, or any data derived from
them, constitutes acceptance.

