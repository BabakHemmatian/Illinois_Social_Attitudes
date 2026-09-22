---
license: cc-by-4.0
language:
- en
base_model:
- google-bert/bert-base-uncased
pipeline_tag: text-classification
library_name: transformers
inference: false
datasets:
- USC-MOLA-Lab/MFRC
tags:
- social-science
- computational-social-science
- reddit
- moralization
- moral-foundations
- isaac
model-index:
- name: isaac-moralization
  results:
  - task:
      type: text-classification
      name: Binary moralization detection
    dataset:
      type: USC-MOLA-Lab/MFRC
      name: "Moral Foundations Reddit Corpus: held-out 10%"
      split: test
    metrics:
    - type: precision
      name: Precision (moralized class)
      value: 0.733
    - type: recall
      name: Recall (moralized class)
      value: 0.788
    - type: f1
      name: F1 (moralized class)
      value: 0.760
    - type: accuracy
      name: Accuracy
      value: 0.755
    - type: f1
      name: Macro F1
      value: 0.755
---

# ISAAC moralization classifier

Binary classifier that decides whether an English text **frames its subject in
moral terms** (judgments of right and wrong, harm, fairness, loyalty, authority,
purity), as opposed to describing it non-morally.

This is the document-level moralization labeler used to annotate all 527,060,919
posts in the [Illinois Social Attitudes Aggregate Corpus (ISAAC)](https://github.com/BabakHemmatian/Illinois_Social_Attitudes).

**Labels:** `0` = non-moralized, `1` = moralized.

## Usage

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO = "ISAAC-corpus/isaac-moralization"
tokenizer = AutoTokenizer.from_pretrained(REPO)
model = AutoModelForSequenceClassification.from_pretrained(REPO).eval()

texts = [
    "People who cut in line are selfish and should be ashamed of themselves.",
    "The bus arrives at the corner of 5th and Main every fifteen minutes.",
]
enc = tokenizer(texts, padding=True, truncation=True, max_length=512,
                return_tensors="pt")
with torch.no_grad():
    probs = torch.softmax(model(**enc).logits, dim=1)

for text, prob in zip(texts, probs):
    label = "moralized" if int(prob.argmax()) == 1 else "non-moralized"
    print(f"{label}  (P(moralized)={prob[1]:.3f})  {text}")
```

No thresholding: the deployed rule is plain argmax.

## Training data

The [Moral Foundations Reddit Corpus](https://huggingface.co/datasets/USC-MOLA-Lab/MFRC)
(MFRC; Trager et al., 2022), with its foundation-level annotations **reduced to a
binary moralized / non-moralized target**. Disagreements between MFRC annotators
were resolved by majority vote; residual ties were broken toward the moralized
label, to maximize sensitivity to moral content.

MFRC was chosen over Twitter-trained alternatives such as MFTC because it samples
the same platform as ISAAC, and platform-specific language style and
conversational context matter for this construct.

### Why binary rather than per-foundation

Two reasons, both deliberate. First, published per-foundation classification
performance is highly uneven; applying unevenly performing labels across ISAAC's
six social group distinctions would produce corresponding unevenness in
downstream construct validity. Second, a binary moralization construct commands
broader theoretical agreement than the individual foundations, around which
debate continues.

If you need foundation-level labels, this is not the model.

## Evaluation

Base model `google-bert/bert-base-uncased`, fine-tuned on the Moral Foundations Reddit
Corpus (MFRC; Trager et al., 2022) reduced to a binary moralized / non-moralized
target.

| Evaluation slice | *k* | Metric | Value |
| --- | --- | --- | --- |
| Held-out 10% of the Moral Foundations Reddit Corpus (MFRC) | 2,682 | Precision (moralized) | .733 |
| | | Recall (moralized) | .788 |
| | | F1 (moralized) | .760 |
| | | Accuracy | .755 |
| | | Macro F1 | .755 |

The evaluation slice is 49.1% moralized, so accuracy is
interpretable against a ~50% baseline.

**Binary by design.** Per-foundation classification performance in the published
literature is highly uneven, and a binary moralization construct commands
broader theoretical agreement than the individual foundations. This model
therefore does not return moral-foundation labels.

### Face validity in the corpus

Applied across ISAAC, the classifier reproduces the expected ordering: highly
contested domains moralize more than less contested ones (race 68.1%, ability
72.2%, sexuality 69.5%, versus body weight 60.2%).

Note that 49–74% moralized is far above the 2–5% reported for unselected everyday
speech and donated personal social media (Atari et al., 2023). That gap is
expected, because ISAAC is pre-filtered for relevance to social distinctions that attract
intense normative scrutiny, and a binary operationalization is more inclusive than
foundation-specific coding, but it means the base rate here should not be read as
a population estimate.

## Intended use

Document-level annotation of English social-media text for the presence of moral
framing, at scale, in aggregate research designs.

## Out-of-scope use

* **Not a moral judgment.** The model detects that moral language is being used,
  not whether the position taken is right, and not whether the author is moral.
* **Not per-foundation.** See above.
* **Not for individual-level decisions.** Accuracy of .755 on a balanced
  held-out set is useful for aggregate estimates over hundreds of thousands of
  posts; it is not adequate for consequential judgments about a single author or
  a single post.
* **Reddit-shaped, English-only.** Trained and evaluated on Reddit comments.
  Performance on other platforms, registers, or languages is unknown.

## Limitations and bias

* Moralization is a contested construct with genuine annotator disagreement in
  the source corpus; the ceiling for any model trained on it is well below 1.0.
* The tie-breaking rule (ties → moralized) means the model is tuned to be
  sensitive rather than conservative, and will over-call ambiguous cases.
* MFRC samples a limited set of subreddits, so topical coverage of moral language
  is narrower than ISAAC's.

## Links

| | |
| --- | --- |
| Try it without code | [ISAAC Text Classifiers Space](https://huggingface.co/spaces/BabakScrapes/isaac-classifiers) |
| Pipeline source, keyword lists, pattern sets | [GitHub](https://github.com/BabakHemmatian/Illinois_Social_Attitudes) |
| Corpus download, samples, SQL playground | [https://isaac.psychology.illinois.edu/](https://isaac.psychology.illinois.edu/) |
| Data Use Agreement | [Data_Use_Agreement.md](https://github.com/BabakHemmatian/Illinois_Social_Attitudes/blob/main/Data_Use_Agreement.md) |

## Citation

Please cite the ISAAC paper. **One citation covers the whole project**: the
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
you give appropriate credit; see Citation above.

The ISAAC corpus itself is governed separately by the project
[Data Use Agreement](https://github.com/BabakHemmatian/Illinois_Social_Attitudes/blob/main/Data_Use_Agreement.md).
