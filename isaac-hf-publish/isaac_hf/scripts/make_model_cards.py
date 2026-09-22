"""Generate the nine ISAAC Hugging Face model cards from performance.py.

Every metric printed into a card is read from `performance.py`, the same module
the Space renders its "Performance & citation" tab from. There is no second
copy of any number, so a card cannot disagree with the app.

Usage (from the repo root):

    python hf_spaces/isaac/../../scripts/make_model_cards.py --out model_cards/

or simply, with performance.py importable:

    python make_model_cards.py --out ./model_cards
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# performance.py lives next to the Space app; allow running from either place.
for candidate in (
    Path(__file__).resolve().parent.parent / "space",
    Path(__file__).resolve().parent.parent / "hf_spaces" / "isaac",
    Path(__file__).resolve().parent,
):
    if (candidate / "performance.py").exists():
        sys.path.insert(0, str(candidate))
        break

import performance as P  # noqa: E402

# Namespace for ISAAC's own model repositories. The Space stays on the personal
# account (ZeroGPU is not available to non-Enterprise orgs) and the DiSCo
# mirrors are third-party, so SPACE_URL and DISCO_*_URL in performance.py keep
# their own namespace and must not be changed to match this.
OWNER = "ISAAC-corpus"
# The DiSCo mirrors are third-party and stayed on the personal account.
DISCO_OWNER = "BabakScrapes"

# ---------------------------------------------------------------------------
# Shared card fragments
# ---------------------------------------------------------------------------

# No ISAAC model card is gated any more: the relevance, moralization and
# generalization models are released under CC-BY-4.0 without access
# restrictions. The location model is gated, but under its own Model Use
# Agreement rather than the Data Use Agreement, and is not generated here.
GATED_BLOCK = ""

LINKS = f"""## Links

| | |
| --- | --- |
| Try it without code | [ISAAC Text Classifiers Space]({P.SPACE_URL}) |
| Pipeline source, keyword lists, pattern sets | [GitHub]({P.REPO_URL}) |
| Corpus download, samples, SQL playground | [{P.SITE_URL}]({P.SITE_URL}) |
| Data Use Agreement | [Data_Use_Agreement.md]({P.DUA_URL}) |
"""

CITATION = f"""## Citation

Please cite the ISAAC paper. **One citation covers the whole project**: the
corpus, the pipeline, and every model. Please do not cite this model repository
separately; keeping references in one place is what allows the project's
citations to be found together.

```bibtex
@article{{hemmatian2026isaac,
  author = {{Hemmatian, Babak and Hadjarab, Sarah and Chen, Jessica and Kurdi, Benedek}},
  title  = {{The {{Illinois}} Social Attitudes Aggregate Corpus ({{ISAAC}}): An Open Tool and Reproducible Pipeline for Analyzing Social Group Discourse at Scale}},
  year   = {{2026}},
  note   = {{Manuscript submitted for publication}}
}}
```

## License

Released under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).
You may use, share, and adapt these weights, including commercially, provided
you give appropriate credit; see Citation above.

The ISAAC corpus itself is governed separately by the project
[Data Use Agreement]({P.DUA_URL}).
"""


def yaml_metrics(entries) -> str:
    """Render a `metrics:` list for a model-index results block."""
    return "\n".join(
        f"    - type: {kind}\n      name: {name}\n      value: {value}"
        for kind, name, value in entries
    )


# ---------------------------------------------------------------------------
# Relevance cards (six)
# ---------------------------------------------------------------------------

RELEVANCE_TEMPLATE = """---
license: cc-by-4.0
language:
- en
base_model:
- {base_model}
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
{gated}model-index:
- name: {repo}
  results:
{results}
---

# ISAAC relevance classifier: {group_label}

Binary classifier that decides whether an English text is **relevant to the
{group_label_lower} social group distinction**, that is, whether it actually
discusses people in terms of that attribute, as opposed to merely containing a
keyword that can also mean something else.

This is stage 3 of the four-stage filtering pipeline that built the [Illinois
Social Attitudes Aggregate Corpus (ISAAC)]({repo_url}), a corpus of 527,060,919
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

{threshold_section}
## Usage

```python
{usage}
```

## Training data

~1,500 posts drawn from the keyword-filtered ISAAC corpus, sampled at random with
stratification by year and by per-post keyword-match count, so the training set
spans the full 2007–2023 range and the full range of keyword density rather than
over-representing heavily matched posts.

Each post was rated for relevance to the {group_label_lower} distinction by **two
independent trained annotators**. Annotator training involved discussing the
rating instructions (Appendix B of the manuscript) and working through examples
together. Labels marked unclear were treated as irrelevant for training, and
residual disagreements were resolved in favor of relevance.

Interrater reliability on this distinction: *k* = {inter_k:,} double-rated posts,
{inter_raw} raw agreement, Cohen's κ = {inter_kappa}.

The annotated validation data are released alongside the corpus.

## Evaluation

{performance}

## Intended use

Screening English social-media text for relevance to the {group_label_lower}
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
general-purpose "is this about {group_label_lower}" classifier for arbitrary
input.

## Limitations and bias

* **Reddit-shaped.** Reddit's user base is not representative of any general
  population, and its norms shifted across the 17 years covered. The model
  inherits that distribution.
* **Keyword-conditioned.** It was trained on posts that already matched an ISAAC
  keyword. It has not been evaluated on text that contains no such keyword.
* **Trained on human judgment, which varied.** Cohen's κ of {inter_kappa} on the
  training sample means trained annotators themselves disagreed on some posts;
  no classifier trained on those labels can be cleaner than they were.
* **Annotation conventions push toward recall at training time.** Unclear posts
  were coded irrelevant and disagreements resolved toward relevance, both
  deliberate choices documented in the manuscript.

{links}
{citation}
"""

RELEVANCE_THRESHOLD_SECTION = """## Important: this model is thresholded

The released checkpoint was **retrained with an asymmetric penalty on false
positives** and is deployed with a decision rule requiring
`P(relevant) > {threshold}`, rather than plain argmax. Both choices trade recall
for precision, because ISAAC's operational target was the residual irrelevance
rate in the finished corpus, not balanced accuracy.

**Taking a plain argmax over the logits will give you a more permissive
classifier than the one that built ISAAC.** The usage snippet below reproduces
the deployed rule; so does the
[Space]({space_url}).

"""

RELEVANCE_USAGE_PLAIN = """import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO = "{owner}/{repo}"
tokenizer = AutoTokenizer.from_pretrained(REPO)
model = AutoModelForSequenceClassification.from_pretrained(REPO).eval()

texts = ["My grandmother is {example_hint}."]
enc = tokenizer(texts, padding=True, truncation=True, max_length=512,
                return_tensors="pt")
with torch.no_grad():
    probs = torch.softmax(model(**enc).logits, dim=1)

for text, prob in zip(texts, probs):
    label = "relevant" if int(prob.argmax()) == 1 else "not relevant"
    print(f"{{label}}  (P(relevant)={{prob[1]:.3f}})  {{text}}")"""

RELEVANCE_USAGE_THRESHOLDED = """import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO = "{owner}/{repo}"
THRESHOLD = {threshold}   # the ISAAC decision rule; do NOT use plain argmax

tokenizer = AutoTokenizer.from_pretrained(REPO)
model = AutoModelForSequenceClassification.from_pretrained(REPO).eval()

texts = ["My grandmother is {example_hint}."]
enc = tokenizer(texts, padding=True, truncation=True, max_length=512,
                return_tensors="pt")
with torch.no_grad():
    probs = torch.softmax(model(**enc).logits, dim=1)

for text, prob in zip(texts, probs):
    relevant = float(prob[1]) > THRESHOLD
    label = "relevant" if relevant else "not relevant"
    print(f"{{label}}  (P(relevant)={{prob[1]:.3f}})  {{text}}")"""

EXAMPLE_HINTS = {
    "ability": "using a wheelchair for the first time this year",
    "age": "in her eighties and still driving",
    "race": "the only Black woman on her block",
    "sexuality": "coming out to her church group",
    "skin_tone": "constantly told she is too dark for the role",
    "weight": "tired of being told to lose weight",
}


def build_relevance_card(group: str) -> tuple[str, str]:
    spec = P.RELEVANCE[group]
    repo = f"isaac-relevance-{group}"
    label = P.GROUP_LABELS[group]
    label_short = label.split(" (")[0]

    results = []
    for row in spec["heldout"]:
        results.append(
            f"  - task:\n"
            f"      type: text-classification\n"
            f"      name: Social-group relevance classification\n"
            f"    dataset:\n"
            f"      type: isaac-relevance-annotations\n"
            f"      name: \"ISAAC relevance annotations ({group}): {row['slice']}\"\n"
            f"      split: test\n"
            f"    metrics:\n"
            + yaml_metrics([
                ("precision", "Precision (relevant class)", f"{row['precision']:.3f}"),
                ("recall", "Recall (relevant class)", f"{row['recall']:.3f}"),
                ("f1", "F1 (relevant class)", f"{row['f1']:.3f}"),
            ])
        )

    if spec["threshold"] is not None:
        threshold_section = RELEVANCE_THRESHOLD_SECTION.format(
            threshold=spec["threshold"], space_url=P.SPACE_URL
        )
        usage = RELEVANCE_USAGE_THRESHOLDED.format(
            owner=OWNER, repo=repo, threshold=spec["threshold"],
            example_hint=EXAMPLE_HINTS[group],
        )
    else:
        threshold_section = ""
        usage = RELEVANCE_USAGE_PLAIN.format(
            owner=OWNER, repo=repo, example_hint=EXAMPLE_HINTS[group]
        )

    inter = spec["interrater"]
    card = RELEVANCE_TEMPLATE.format(
        dua=P.DUA_URL,
        base_model=spec["base_model"],
        gated=GATED_BLOCK.format(
            heading=f"the ISAAC relevance classifier ({label_short.lower()})",
            dua=P.DUA_URL,
        ),
        repo=repo,
        results="\n".join(results),
        group_label=label,
        group_label_lower=label_short.lower(),
        repo_url=P.REPO_URL,
        threshold_section=threshold_section,
        usage=usage,
        inter_k=inter["k"],
        inter_raw=P.pct(inter["raw_agreement"]),
        inter_kappa=P.m(inter["kappa"]),
        performance=P.relevance_summary_md(group, heading=False).strip(),
        links=LINKS,
        citation=CITATION,
    )
    return repo, card


# ---------------------------------------------------------------------------
# Moralization card
# ---------------------------------------------------------------------------

MORALIZATION_CARD = """---
license: cc-by-4.0
language:
- en
base_model:
- {base_model}
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
{gated}model-index:
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
{metrics}
---

# ISAAC moralization classifier

Binary classifier that decides whether an English text **frames its subject in
moral terms** (judgments of right and wrong, harm, fairness, loyalty, authority,
purity), as opposed to describing it non-morally.

This is the document-level moralization labeler used to annotate all 527,060,919
posts in the [Illinois Social Attitudes Aggregate Corpus (ISAAC)]({repo_url}).

**Labels:** `0` = non-moralized, `1` = moralized.

## Usage

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO = "{owner}/isaac-moralization"
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
    print(f"{{label}}  (P(moralized)={{prob[1]:.3f}})  {{text}}")
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

{performance}

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
* **Not for individual-level decisions.** Accuracy of {accuracy} on a balanced
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

{links}
{citation}
"""


def build_moralization_card() -> tuple[str, str]:
    d = P.MORALIZATION
    metrics = yaml_metrics([
        ("precision", "Precision (moralized class)", f"{d['moralized_precision']:.3f}"),
        ("recall", "Recall (moralized class)", f"{d['moralized_recall']:.3f}"),
        ("f1", "F1 (moralized class)", f"{d['moralized_f1']:.3f}"),
        ("accuracy", "Accuracy", f"{d['accuracy']:.3f}"),
        ("f1", "Macro F1", f"{d['macro_f1']:.3f}"),
    ])
    card = MORALIZATION_CARD.format(
        dua=P.DUA_URL,
        base_model=d["base_model"],
        gated=GATED_BLOCK,
        metrics=metrics,
        repo_url=P.REPO_URL,
        owner=OWNER,
        performance=P.moralization_summary_md().strip(),
        accuracy=P.m(d["accuracy"]),
        links=LINKS,
        citation=CITATION,
    )
    return "isaac-moralization", card


# ---------------------------------------------------------------------------
# Generalization cards (two) -- same weights as the public DiSCo release
# ---------------------------------------------------------------------------

SE_LABELS = [
    "BOUNDED EVENT (SPECIFIC)",
    "BOUNDED EVENT (GENERIC)",
    "UNBOUNDED EVENT (SPECIFIC)",
    "UNBOUNDED EVENT (GENERIC)",
    "BASIC STATE",
    "COERCED STATE (SPECIFIC)",
    "COERCED STATE (GENERIC)",
    "PERFECT COERCED STATE (SPECIFIC)",
    "PERFECT COERCED STATE (GENERIC)",
    "GENERIC SENTENCE (DYNAMIC)",
    "GENERIC SENTENCE (STATIC)",
    "GENERIC SENTENCE (HABITUAL)",
    "GENERALIZING SENTENCE (DYNAMIC)",
    "GENERALIZING SENTENCE (STATIVE)",
    "QUESTION",
    "IMPERATIVE",
    "NONSENSE",
    "OTHER",
]

SE_ATTRS = [
    ("specific", "dynamic", "episodic"), ("generic", "dynamic", "episodic"),
    ("specific", "dynamic", "static"), ("generic", "dynamic", "static"),
    ("specific", "stative", "static"), ("specific", "dynamic", "static"),
    ("generic", "dynamic", "static"), ("specific", "dynamic", "episodic"),
    ("generic", "dynamic", "episodic"), ("generic", "dynamic", "habitual"),
    ("generic", "stative", "static"), ("generic", "stative", "habitual"),
    ("specific", "dynamic", "habitual"), ("specific", "stative", "habitual"),
    ("NA", "NA", "NA"), ("NA", "NA", "NA"), ("NA", "NA", "NA"), ("NA", "NA", "NA"),
]

MIRROR_NOTE = """> **Same weights as the public DiSCo release.** This repository is the
> ISAAC-facing copy of [`{owner}/{disco}`]({disco_url}); the checkpoints are
> identical. That card carries the full model description and the DiSCo corpus
> details. This one documents how the model is used inside the ISAAC pipeline and
> restates the performance figures reported in the ISAAC manuscript. Cite
> whichever matches your use; if you are using ISAAC labels, cite both.
"""

GENERALIZATION_CARD = """---
license: cc-by-4.0
language:
- en
base_model:
- {base_model}
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
{metrics_se}
  - task:
      type: text-classification
      name: Collapsed generalization features
    dataset:
      type: disco
      name: "DiSCo gold corpus: held-out 10%"
      split: test
    metrics:
{metrics_features}
---

# ISAAC generalization: situation-entity classifier

{mirror_note}
Clause-level classifier that assigns one of **18 situation-entity types** to an
English clause, extending Smith's (2003) discourse-mode framework with
boundedness distinctions. In the [Illinois Social Attitudes Aggregate Corpus
(ISAAC)]({repo_url}) the 18 types are collapsed into three linguistic features
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
[clause segmenter]({seg_url}), which splits text into the clauses this model
scores.

## Labels

The released `config.json` in this repository maps indices to readable names.
The full mapping, and the three-feature decomposition ISAAC uses:

| Index | Situation entity | Genericity | Eventivity | Boundedness |
| --- | --- | --- | --- | --- |
{label_table}

`QUESTION`, `IMPERATIVE`, `NONSENSE`, and `OTHER` are non-statements, for which
the three features are undefined. ISAAC counts them separately rather than
folding them into the proportions.

## Usage

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO = "{owner}/isaac-generalization"
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
[project repository]({repo_url}) or `generalization.py` in the
[Space]({space_url}).

## Training data

Pre-trained on the SitEnt corpus, then fine-tuned on the DiSCo corpus of
opinionated, mixed-register English text (Hemmatian, 2022). See the
[DiSCo card]({disco_url}) for the corpus description.

## Evaluation

{performance}

## Intended use

Clause-level annotation of English text for genericity, eventivity, and
boundedness/habituality, in aggregate research designs, the role it plays in
ISAAC, where it labels clauses across 527M posts.

## Out-of-scope use

* **Clause-level, not document-level.** Feed it clauses from the segmenter, not
  whole posts. Scoring an unsegmented paragraph will not give a meaningful label.
* **Not a measure of truth, quality, or bias.** A generic statement is not
  thereby a stereotype, and a specific one is not thereby accurate.
* **Not for individual-level decisions.** The 18-way macro F1 of {se_macro_f1}
  is far too low for any per-clause consequential use; the collapsed features are
  intended for aggregate proportions over large samples.

## Limitations and bias

* **Heavy label imbalance.** The rarer situation-entity types are poorly
  represented in training, which is why 18-way accuracy ({se_accuracy}) far
  exceeds 18-way macro F1 ({se_macro_f1}). Per-class performance on rare types is
  correspondingly weak. Use the collapsed features.
* **Single-topic fine-tuning.** DiSCo is opinionated, mixed-register text on one
  controversial policy topic. Cross-genre transfer is expected given the formal,
  content-agnostic nature of the target features, but it has not been separately
  quantified for Reddit discourse about social groups.
* **English only.**
* **Errors compound.** ISAAC's generalization columns are the product of two
  models in series; segmentation errors propagate into classification.

{links}
{citation}
"""

SEGMENTATION_CARD = """---
license: cc-by-4.0
language:
- en
base_model:
- {base_model}
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
      value: {coverage}
---

# ISAAC generalization: clause segmenter

{mirror_note}
Token classifier that splits English text into **clauses**, the unit the
[situation-entity classifier]({clf_url}) then labels. Together the two models
produce the clause-level generalization columns of the [Illinois Social Attitudes
Aggregate Corpus (ISAAC)]({repo_url}).

## Decoding

The model emits one tag per word (majority-voted across sub-word tokens). The
decoder in `code/label_generalization.py` reads them as follows:

* tag `2` marks a **clause-final** word; it closes the clause it appears in;
* tags `0` and `1` mark clause-internal words;
* a new clause opens on the word after a `2`;
* when a word receives no aligned prediction, it defaults to `1`.

Reference implementation, including the sub-word majority vote:

```python
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

REPO = "{owner}/isaac-generalization-segmentation"
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
in the ISAAC pipeline; see `generalization.py` in the [Space]({space_url}).

## Training data

The DiSCo corpus of opinionated, mixed-register English text (Hemmatian, 2022),
with human-verified clause boundaries. See the
[DiSCo card]({disco_url}).

## Evaluation

{performance}

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
* Reddit text is noisy (missing punctuation, run-on constructions, markup), and
  segmentation quality degrades accordingly.

{links}
{citation}
"""


def build_generalization_cards() -> list[tuple[str, str]]:
    d = P.GENERALIZATION
    se = d["situation_entity"]

    metrics_se = yaml_metrics([
        ("accuracy", "Accuracy (18-way)", f"{se['accuracy']:.3f}"),
        ("f1", "Macro F1 (18-way)", f"{se['macro_f1']:.3f}"),
    ])
    metrics_features = yaml_metrics([
        (kind, name, value)
        for f in d["features"]
        for kind, name, value in [
            ("f1", f"Macro F1 ({f['name'].lower()})", f"{f['macro_f1']:.3f}"),
            ("accuracy", f"Accuracy ({f['name'].lower()})", f"{f['accuracy']:.3f}"),
        ]
    ])

    label_table = "\n".join(
        f"| {i} | `{name}` | {attrs[0]} | {attrs[1]} | {attrs[2]} |"
        for i, (name, attrs) in enumerate(zip(SE_LABELS, SE_ATTRS))
    )

    clf = GENERALIZATION_CARD.format(
        base_model=d["base_model"],
        metrics_se=metrics_se,
        metrics_features=metrics_features,
        mirror_note=MIRROR_NOTE.format(
            owner=DISCO_OWNER, disco="disco-se-classifier", disco_url=P.DISCO_CLF_URL
        ),
        repo_url=P.REPO_URL,
        seg_url=f"https://huggingface.co/{OWNER}/isaac-generalization-segmentation",
        label_table=label_table,
        owner=OWNER,
        space_url=P.SPACE_URL,
        disco_url=P.DISCO_CLF_URL,
        performance=P.generalization_summary_md().strip(),
        se_macro_f1=P.m(se["macro_f1"]),
        se_accuracy=P.m(se["accuracy"]),
        links=LINKS,
        citation=CITATION,
    )

    seg = SEGMENTATION_CARD.format(
        base_model=P.SEGMENTATION["base_model"],
        coverage=f"{P.SEGMENTATION['coverage']:.3f}",
        mirror_note=MIRROR_NOTE.format(
            owner=DISCO_OWNER, disco="disco-clause-segmenter", disco_url=P.DISCO_SEG_URL
        ),
        clf_url=f"https://huggingface.co/{OWNER}/isaac-generalization",
        repo_url=P.REPO_URL,
        owner=OWNER,
        space_url=P.SPACE_URL,
        disco_url=P.DISCO_SEG_URL,
        performance=P.generalization_summary_md().strip(),
        links=LINKS,
        citation=CITATION,
    )

    return [("isaac-generalization", clf),
            ("isaac-generalization-segmentation", seg)]


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="model_cards", type=Path)
    args = parser.parse_args()

    cards = [build_relevance_card(g) for g in P.RELEVANCE]
    cards.append(build_moralization_card())
    cards.extend(build_generalization_cards())

    for repo, text in cards:
        target = args.out / repo
        target.mkdir(parents=True, exist_ok=True)
        (target / "README.md").write_text(text, encoding="utf-8")
        print(f"wrote {target / 'README.md'}  ({len(text):,} chars)")

    print(f"\n{len(cards)} cards written to {args.out}/")


if __name__ == "__main__":
    main()
