---
title: ISAAC Text Classifiers
emoji: 🧭
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.9.0
python_version: "3.10"
app_file: app.py
pinned: false
license: other
license_name: isaac-data-use-agreement
license_link: https://github.com/BabakHemmatian/Illinois_Social_Attitudes/blob/main/Data_Use_Agreement.md
models:
- ISAAC-corpus/isaac-relevance-ability
- ISAAC-corpus/isaac-relevance-age
- ISAAC-corpus/isaac-relevance-race
- ISAAC-corpus/isaac-relevance-sexuality
- ISAAC-corpus/isaac-relevance-skin_tone
- ISAAC-corpus/isaac-relevance-weight
- ISAAC-corpus/isaac-moralization
- ISAAC-corpus/isaac-generalization
- ISAAC-corpus/isaac-generalization-segmentation
tags:
- social-science
- reddit
- content-classification
- moralization
- generalization
---

# ISAAC Text Classifiers

A coding-free demo for all three in-house classifier families from [the Illinois
Social Attitudes Aggregate Corpus (ISAAC)](https://github.com/BabakHemmatian/Illinois_Social_Attitudes).
Use is subject to [terms](https://github.com/BabakHemmatian/Illinois_Social_Attitudes/blob/main/Data_Use_Agreement.md).

- **Relevance**: is a text relevant to a given social group distinction (ability, age,
  race, sexuality, skin tone, weight)? Six fine-tuned RoBERTa classifiers behind a
  distinction dropdown (`roberta-large` for five distinctions; `roberta-base` for
  skin tone).
- **Moralization**: does a text frame its subject in moral terms? Fine-tuned binary
  `bert-base-uncased` classifier.
- **Generalization**: clause segmentation plus per-clause labels for linguistic
  features that make for more, or less, generalized statements
  (genericity, eventivity, boundedness/habituality; 18 combinations).
  The released weights are the DiSCo RoBERTa models (see
  [segmenter](https://huggingface.co/BabakScrapes/disco-clause-segmenter) and
  [classifier](https://huggingface.co/BabakScrapes/disco-se-classifier)),
  re-published under ISAAC-facing repo names.

Each task has a **Single text** tab and a **Multiple texts (file)** tab that
accepts a `.txt` (one text per line) or `.csv` (text in the first column) and
returns a labeled results CSV.

For **race** and **skin tone** relevance, a 0.6 confidence threshold is applied to the
"relevant" class, matching the ISAAC pipeline (`filter_relevance.py`).

## Performance

Reported in full in the app's **Performance & citation** tab, which is rendered
from `performance.py`, the single source of truth for every metric this project
publishes, including the nine model cards. Summary:

| Model | Base | Eval *k* | Headline | Residual irrelevance | Decision rule |
| --- | --- | --- | --- | --- | --- |
| Relevance: ability | `roberta-large` | 148 | .846 | 6.0% | argmax |
| Relevance: age | `roberta-large` | 149 | .886 | 3.0% | argmax |
| Relevance: race | `roberta-large` | 150 | .862 | 8.0% | P(rel) > 0.6 |
| Relevance: sexuality | `roberta-large` | 150 | .955 | 5.0% | argmax |
| Relevance: skin tone | `roberta-base` | 189 | .788 | 9.3% | P(rel) > 0.6 |
| Relevance: body weight | `roberta-large` | 148 | .987 | 4.0% | argmax |
| Moralization | `bert-base-uncased` | 2,682 | macro F1 .755 | - | argmax |
| Generalization (18-way) | `roberta-base` | 2,357 | acc. .737, macro F1 .514 | - | argmax |
| Generalization (segmenter) | `roberta-base` | - | 95.5% clause-span coverage | - | argmax |

"Headline" is held-out F1 for the relevance models. "Residual irrelevance" is the
stringent-rule rate in the finished corpus after all four filtering stages, from
the double-rated human audit, the figure to quote for corpus quality. The
collapsed generalization features (genericity .852, eventivity .879,
boundedness/habituality .804 macro F1) are what ISAAC reports and are stronger
than the 18-way figure.

Every published number lives in `performance.py` and is rendered from there into
(a) the collapsed accordion inside each task tab, (b) the **Performance &
citation** tab, and (c) the table above. To regenerate the table after any edit:

```bash
python performance.py --readme
```

## Layout

| File | Contents |
| --- | --- |
| `app.py` | four top-level tabs |
| `common.py` | Device/token resolution, model-path resolution, file I/O, GPU-duration helpers. |
| `performance.py` | Every published metric, as data, plus the markdown renderers for the app, this README, and the model cards. |
| `relevance.py` | Six relevance classifiers, thresholding, UI. |
| `moralization.py` | Moralization classifier and UI. |
| `generalization.py` | Segmenter + clause classifier, aggregation, pie charts, UI. |

## Hardware

The Space runs on **ZeroGPU**, which allocates a GPU only for the duration of a
decorated call. The decorated entry points are:

| Function | Duration |
| --- | --- |
| `relevance.classify_text` | 60 s (default) |
| `relevance.classify_file` | dynamic, from row count |
| `moralization.classify_text` | 60 s (default) |
| `moralization.classify_file` | dynamic, from row count |
| `generalization._clause_labels` | dynamic, from word count |
| `generalization.analyze_file` | dynamic, from row count |

ZeroGPU charges each visitor's daily quota against the duration a call
*requests*, not the time it actually uses, hence the dynamic durations rather
than a flat worst case. Overrunning a request kills the call, so the estimates
assume a cold model load every time.

The **Performance & citation** tab is static markdown and holds no GPU.

## Memory behavior

Every model is loaded lazily and cached, so only the task(s) actually used are
pulled into memory. Relevance keeps at most two of its six distinction models
resident (`functools.lru_cache`, capacity 2), roughly 1.4 GB rather than the
~7.6 GB the full set would need. Moralization and generalization each cache their
own models at capacity 1.

## Models

Each model resolves local-first with a Hugging Face Hub fallback
(`common.resolve_source`):

1. `$ISAAC_MODELS_DIR/<relative path>`: explicit local override.
2. `<repo_root>/models/<relative path>`: in-repo local run.
3. The released Hugging Face model repo.

| Task | Local path under `models/` | Hub default | Env override |
| --- | --- | --- | --- |
| Relevance | `filter_relevance_<group>` | `ISAAC-corpus/isaac-relevance-<group>` | `ISAAC_RELEVANCE_REPO_PREFIX` |
| Moralization | `label_moralization` | `ISAAC-corpus/isaac-moralization` | `ISAAC_MORALIZATION_REPO` |
| Generalization (segmenter) | `label_generalization/label_generalization_segmentation` | `ISAAC-corpus/isaac-generalization-segmentation` | `ISAAC_GENERALIZATION_SEG_REPO` |
| Generalization (classifier) | `label_generalization/label_generalization` | `ISAAC-corpus/isaac-generalization` | `ISAAC_GENERALIZATION_REPO` |

The same code therefore runs locally against the project's `models/` directory
and on Hugging Face against the published model repos.

The nine model repos are public and ungated, under a Creative Commons Attribution
4.0 International License, so the Space loads them without credentials. Only the
corpus itself sits behind the Data Use Agreement.

## Local testing

```bash
cd hf_spaces/isaac
pip install -r requirements.txt
export ISAAC_MODELS_DIR=/path/to/Illinois_Social_Attitudes/models
python app.py
```

## Citation

See the **Performance & citation** tab, or the
[project repository](https://github.com/BabakHemmatian/Illinois_Social_Attitudes).

## Questions

[isaac.corpus.support@gmail.com](mailto:isaac.corpus.support@gmail.com)
