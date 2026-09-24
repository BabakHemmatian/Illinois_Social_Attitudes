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
- reddit
- content-filtering
- relevance-classification
- isaac
model-index:
- name: isaac-relevance-skin_tone
  results:
  - task:
      type: text-classification
      name: Social-group relevance classification
    dataset:
      type: isaac-relevance-annotations
      name: "ISAAC relevance annotations (skin_tone): Held-out 10% of the retraining sample"
      split: test
    metrics:
    - type: precision
      name: Precision (relevant class)
      value: 0.875
    - type: recall
      name: Recall (relevant class)
      value: 0.840
    - type: f1
      name: F1 (relevant class)
      value: 0.857
  - task:
      type: text-classification
      name: Social-group relevance classification
    dataset:
      type: isaac-relevance-annotations
      name: "ISAAC relevance annotations (skin_tone): Held-out 10% of the initial training sample"
      split: test
    metrics:
    - type: precision
      name: Precision (relevant class)
      value: 0.839
    - type: recall
      name: Recall (relevant class)
      value: 0.743
    - type: f1
      name: F1 (relevant class)
      value: 0.788
---

# ISAAC relevance classifier: Skin tone (light vs. dark)

Binary classifier that decides whether an English text is **relevant to the
skin tone social group distinction**, that is, whether it actually
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
errors, depress reliability, and suppress real effects, and most published work
never reports how many of them survive. Because this model reads each keyword in
the context of the full post, it separates the intended sense from the unintended
one in a way a static list cannot.

## Important: this model is thresholded

The released checkpoint was **retrained with an asymmetric penalty on false
positives** and is deployed with a decision rule requiring
`P(relevant) > 0.6`, rather than plain argmax. Both choices trade recall
for precision, because ISAAC's operational target was the residual irrelevance
rate in the finished corpus, not balanced accuracy.

**Taking a plain argmax over the logits will give you a more permissive
classifier than the one that built ISAAC.** The usage snippet below reproduces
the deployed rule; so does the
[Space](https://huggingface.co/spaces/BabakScrapes/isaac-classifiers).


## Usage

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO = "ISAAC-corpus/isaac-relevance-skin_tone"
THRESHOLD = 0.6   # the ISAAC decision rule; do NOT use plain argmax

tokenizer = AutoTokenizer.from_pretrained(REPO)
model = AutoModelForSequenceClassification.from_pretrained(REPO).eval()

texts = ["My grandmother is constantly told she is too dark for the role."]
enc = tokenizer(texts, padding=True, truncation=True, max_length=512,
                return_tensors="pt")
with torch.no_grad():
    probs = torch.softmax(model(**enc).logits, dim=1)

for text, prob in zip(texts, probs):
    relevant = float(prob[1]) > THRESHOLD
    label = "relevant" if relevant else "not relevant"
    print(f"{label}  (P(relevant)={prob[1]:.3f})  {text}")
```

## Training data

~1,500 posts drawn from the keyword-filtered ISAAC corpus, sampled at random with
stratification by year and by per-post keyword-match count, so the training set
spans the full 2007–2023 range and the full range of keyword density rather than
over-representing heavily matched posts.

Each post was rated for relevance to the skin tone distinction by **two
independent trained annotators**. Annotator training involved discussing the
rating instructions (Appendix B of the manuscript) and working through examples
together. Labels marked unclear were treated as irrelevant for training, and
residual disagreements were resolved in favor of relevance.

Interrater reliability on this distinction: *k* = 1,885 double-rated posts,
93.4% raw agreement, Cohen's κ = .725.

The annotated validation data are released alongside the corpus.

## Evaluation

Base model `FacebookAI/roberta-base`.

| Evaluation slice | *k* | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| Held-out 10% of the retraining sample | 40 | .875 | .840 | .857 |
| Held-out 10% of the initial training sample | 189 | .839 | .743 | .788 |

**This classifier applies a P(relevant) > 0.6 decision threshold** and was trained with an asymmetric penalty on false positives, so it is deliberately more conservative than a plain argmax classifier. Scoring the same text with `argmax` will not reproduce the ISAAC corpus.

**Human agreement on the training data.** Two trained annotators rated
*k* = 1,885 posts: 93.4% raw agreement,
Cohen's κ = .725.

**What the full filter delivers.** Held-out precision describes this model in
isolation. In the released corpus this classifier is one of four filtering
stages, and a double-rated audit of the finished
skin tone data (*k* = 150)
found 9.3% of comments irrelevant under the stringent
rule and 4.7% under the lenient rule
(9.0% for submissions). That end-to-end figure, not
the F1 above, is the number to quote when describing corpus quality.

## Intended use

Screening English social-media text for relevance to the skin tone
distinction, as a preprocessing step before substantive analysis, the role it
plays in ISAAC. It is a **topical relevance** filter.

## Out-of-scope use

This model does **not** measure attitudes, sentiment, toxicity, or bias. A post
labeled relevant may be supportive, hostile, or neutral; relevance says only that
the post is *about* the distinction. Do not use its output as a proxy for
prejudice, and do not use it to make decisions about individuals.

It was trained on Reddit text from 2007–2023 and on the ISAAC keyword lists.
Performance on other platforms, other registers, other languages, or on text that
was not first keyword-filtered is unknown and likely lower. It is not a
general-purpose "is this about skin tone" classifier for arbitrary
input.

## Limitations and bias

* **Reddit-shaped.** Reddit's user base is not representative of any general
  population, and its norms shifted across the 17 years covered. The model
  inherits that distribution.
* **Keyword-conditioned.** It was trained on posts that already matched an ISAAC
  keyword. It has not been evaluated on text that contains no such keyword.
* **Trained on human judgment, which varied.** Cohen's κ of .725 on the
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
