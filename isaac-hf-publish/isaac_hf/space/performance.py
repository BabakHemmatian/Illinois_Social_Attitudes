"""Single source of truth for every performance number this project publishes.

Everything the Space displays, everything the Space README states, and the
"Evaluation" block of every Hugging Face model card is rendered from the data
structures below. Nothing here is duplicated by hand anywhere else, so a metric
cannot drift between the app, the README, and the nine model repos.

All figures are the final, released-checkpoint values reported in:

    Hemmatian, B., Hadjarab, S., Chen, J., & Kurdi, B. (2026).
    The Illinois Social Attitudes Aggregate Corpus (ISAAC).

    Table 3   Residual irrelevance in the final corpus (double-rated audit)
    Table 5   Label-classifier performance
    Table C1  Interrater agreement on relevance training samples
    Table C2  Relevance classifier performance on held-out slices

Run this file directly to regenerate the derived markdown:

    python performance.py --readme                  # Space README section
    python performance.py --card isaac-relevance-race
    python performance.py --card isaac-moralization
    python performance.py --card isaac-generalization
    python performance.py --card isaac-generalization-segmentation
"""

from __future__ import annotations

import sys

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

PAPER_CITE = (
    "Hemmatian, B., Hadjarab, S., Chen, J., & Kurdi, B. (2026). "
    "*The Illinois Social Attitudes Aggregate Corpus (ISAAC): An Open Tool and "
    "Reproducible Pipeline for Analyzing Social Group Discourse at Scale.*"
)

REPO_URL = "https://github.com/BabakHemmatian/Illinois_Social_Attitudes"
DUA_URL = f"{REPO_URL}/blob/main/Data_Use_Agreement.md"
SITE_URL = "https://isaac.psychology.illinois.edu/"
SPACE_URL = "https://huggingface.co/spaces/BabakScrapes/isaac-classifiers"

GROUP_LABELS = {
    "ability": "Ability (abled vs. disabled)",
    "age": "Age (young vs. old)",
    "race": "Race (White vs. Black)",
    "sexuality": "Sexuality (straight vs. gay)",
    "skin_tone": "Skin tone (light vs. dark)",
    "weight": "Body weight (thin vs. fat)",
}

# ---------------------------------------------------------------------------
# Relevance classifiers (six repos)
#
# `heldout` lists the evaluation slices reported for the *released* checkpoint.
# Race and skin tone shipped retrained checkpoints, which the manuscript
# evaluates on two slices; the other four shipped their initial checkpoint,
# evaluated on one.
# ---------------------------------------------------------------------------

RELEVANCE = {
    "ability": {
        "base_model": "FacebookAI/roberta-large",
        "threshold": None,
        "heldout": [
            {"slice": "Held-out 10% of the rated training sample", "k": 148,
             "precision": .815, "recall": .880, "f1": .846},
        ],
        "interrater": {"k": 1476, "raw_agreement": .982, "kappa": .960},
        "residual": {"k": 100, "stringent": .060, "lenient": .030,
                     "kappa": .651, "raw_agreement": .970, "submissions": .040},
    },
    "age": {
        "base_model": "FacebookAI/roberta-large",
        "threshold": None,
        "heldout": [
            {"slice": "Held-out 10% of the rated training sample", "k": 149,
             "precision": .853, "recall": .921, "f1": .886},
        ],
        "interrater": {"k": 1485, "raw_agreement": .906, "kappa": .812},
        "residual": {"k": 100, "stringent": .030, "lenient": .000,
                     "kappa": None, "raw_agreement": .970, "submissions": .020},
    },
    "race": {
        "base_model": "FacebookAI/roberta-large",
        "threshold": .6,
        "heldout": [
            {"slice": "Held-out 10% of the retraining sample", "k": 40,
             "precision": .786, "recall": .917, "f1": .846},
            {"slice": "Held-out 10% of the initial training sample", "k": 150,
             "precision": 1.000, "recall": .757, "f1": .862},
        ],
        "interrater": {"k": 1498, "raw_agreement": .926, "kappa": .784},
        "residual": {"k": 150, "stringent": .080, "lenient": .053,
                     "kappa": .786, "raw_agreement": .973, "submissions": .040},
    },
    "sexuality": {
        "base_model": "FacebookAI/roberta-large",
        "threshold": None,
        "heldout": [
            {"slice": "Held-out 10% of the rated training sample", "k": 150,
             "precision": .962, "recall": .949, "f1": .955},
        ],
        "interrater": {"k": 1493, "raw_agreement": .936, "kappa": .871},
        "residual": {"k": 100, "stringent": .050, "lenient": .020,
                     "kappa": .556, "raw_agreement": .970, "submissions": .050},
    },
    "skin_tone": {
        "base_model": "FacebookAI/roberta-base",
        "threshold": .6,
        "heldout": [
            {"slice": "Held-out 10% of the retraining sample", "k": 40,
             "precision": .875, "recall": .840, "f1": .857},
            {"slice": "Held-out 10% of the initial training sample", "k": 189,
             "precision": .839, "recall": .743, "f1": .788},
        ],
        "interrater": {"k": 1885, "raw_agreement": .934, "kappa": .725},
        "residual": {"k": 150, "stringent": .093, "lenient": .047,
                     "kappa": .643, "raw_agreement": .953, "submissions": .090},
    },
    "weight": {
        "base_model": "FacebookAI/roberta-large",
        "threshold": None,
        "heldout": [
            {"slice": "Held-out 10% of the rated training sample", "k": 148,
             "precision": .975, "recall": 1.000, "f1": .987},
        ],
        "interrater": {"k": 1475, "raw_agreement": .896, "kappa": .690},
        "residual": {"k": 100, "stringent": .040, "lenient": .030,
                     "kappa": .852, "raw_agreement": .990, "submissions": .020},
    },
}

RESIDUAL_POOLED = {"k": 700, "stringent": .063, "lenient": .033, "submissions": .043}

# ---------------------------------------------------------------------------
# Moralization (one repo)
# ---------------------------------------------------------------------------

MORALIZATION = {
    "base_model": "google-bert/bert-base-uncased",
    "eval_set": "Held-out 10% of the Moral Foundations Reddit Corpus (MFRC)",
    "k": 2682,
    "positive_rate": .491,
    "moralized_precision": .733,
    "moralized_recall": .788,
    "moralized_f1": .760,
    "accuracy": .755,
    "macro_f1": .755,
}

# ---------------------------------------------------------------------------
# Generalization (two repos; identical weights to the public DiSCo release)
# ---------------------------------------------------------------------------

GENERALIZATION = {
    "base_model": "FacebookAI/roberta-base",
    "eval_set": "Held-out 10% of the DiSCo gold corpus",
    "k": 2357,
    "situation_entity": {"n_classes": 18, "accuracy": .737, "macro_f1": .514},
    "features": [
        {"name": "Genericity", "n_classes": 2, "macro_f1": .852, "accuracy": .860},
        {"name": "Eventivity", "n_classes": 2, "macro_f1": .879, "accuracy": .894},
        {"name": "Boundedness / habituality", "n_classes": 4,
         "macro_f1": .804, "accuracy": .850},
    ],
}

SEGMENTATION = {
    "base_model": "FacebookAI/roberta-base",
    "coverage": .955,
    "coverage_note": (
        "The segmenter covered most of the target clause span for 95.5% of "
        "human-verified clauses."
    ),
}

DISCO_SEG_URL = "https://huggingface.co/BabakScrapes/disco-clause-segmenter"
DISCO_CLF_URL = "https://huggingface.co/BabakScrapes/disco-se-classifier"

# ---------------------------------------------------------------------------
# Formatting helpers
#
# Metrics are printed in the manuscript's leading-dot style (.846) so the app,
# the README, the cards, and the paper are visually comparable.
# ---------------------------------------------------------------------------


def m(value: float | None) -> str:
    """Format a 0-1 metric as the manuscript does: .846, 1.000, or an em dash."""
    if value is None:
        return "—"
    text = f"{value:.3f}"
    return text[1:] if text.startswith("0.") else text


def pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.1f}%"


# ---------------------------------------------------------------------------
# Per-task summaries -- rendered into the collapsed accordion inside each task
# tab, next to the controls that actually run the model.
# ---------------------------------------------------------------------------


def relevance_summary_md(group: str, heading: bool = True) -> str:
    spec = RELEVANCE[group]
    rows = "\n".join(
        f"| {row['slice']} | {row['k']} | {m(row['precision'])} | "
        f"{m(row['recall'])} | {m(row['f1'])} |"
        for row in spec["heldout"]
    )
    residual = spec["residual"]
    inter = spec["interrater"]

    threshold_note = ""
    if spec["threshold"] is not None:
        threshold_note = (
            f"\n**This classifier applies a P(relevant) > {spec['threshold']} "
            "decision threshold** and was trained with an asymmetric penalty on "
            "false positives, so it is deliberately more conservative than a "
            "plain argmax classifier. Scoring the same text with `argmax` will "
            "not reproduce the ISAAC corpus.\n"
        )

    title = f"**{GROUP_LABELS[group]}** — b" if heading else "B"

    return f"""
{title}ase model `{spec['base_model']}`.

| Evaluation slice | *k* | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
{rows}
{threshold_note}
**Human agreement on the training data.** Two trained annotators rated
*k* = {inter['k']:,} posts: {pct(inter['raw_agreement'])} raw agreement,
Cohen's κ = {m(inter['kappa'])}.

**What the full filter delivers.** Held-out precision describes this model in
isolation. In the released corpus this classifier is one of four filtering
stages, and a double-rated audit of the finished
{GROUP_LABELS[group].split(' (')[0].lower()} data (*k* = {residual['k']})
found {pct(residual['stringent'])} of comments irrelevant under the stringent
rule and {pct(residual['lenient'])} under the lenient rule
({pct(residual['submissions'])} for submissions). That end-to-end figure, not
the F1 above, is the number to quote when describing corpus quality.
"""


def moralization_summary_md() -> str:
    d = MORALIZATION
    return f"""
Base model `{d['base_model']}`, fine-tuned on the Moral Foundations Reddit
Corpus (MFRC; Trager et al., 2022) reduced to a binary moralized / non-moralized
target.

| Evaluation slice | *k* | Metric | Value |
| --- | --- | --- | --- |
| {d['eval_set']} | {d['k']:,} | Precision (moralized) | {m(d['moralized_precision'])} |
| | | Recall (moralized) | {m(d['moralized_recall'])} |
| | | F1 (moralized) | {m(d['moralized_f1'])} |
| | | Accuracy | {m(d['accuracy'])} |
| | | Macro F1 | {m(d['macro_f1'])} |

The evaluation slice is {pct(d['positive_rate'])} moralized, so accuracy is
interpretable against a ~50% baseline.

**Binary by design.** Per-foundation classification performance in the published
literature is highly uneven, and a binary moralization construct commands
broader theoretical agreement than the individual foundations. This model
therefore does not return moral-foundation labels.
"""


def generalization_summary_md() -> str:
    d = GENERALIZATION
    se = d["situation_entity"]
    feature_rows = "\n".join(
        f"| {f['name']} ({f['n_classes']}-way) | {m(f['macro_f1'])} | {m(f['accuracy'])} |"
        for f in d["features"]
    )
    return f"""
Two `{d['base_model']}` models run in sequence: a clause segmenter, then a
{se['n_classes']}-way situation-entity classifier. Both are the same weights as
the public DiSCo release — see the
[clause segmenter]({DISCO_SEG_URL}) and
[situation-entity classifier]({DISCO_CLF_URL}) cards for the full model
description and the corpus they were trained on.

**Segmentation.** {SEGMENTATION['coverage_note']}

**Classification**, on the {d['eval_set'].lower()} (*k* ≈ {d['k']:,}):

| Target | Macro F1 | Accuracy |
| --- | --- | --- |
| Full situation entity ({se['n_classes']}-way) | {m(se['macro_f1'])} | {m(se['accuracy'])} |
{feature_rows}

The {se['n_classes']}-way macro F1 of {m(se['macro_f1'])} is held down by heavy
label imbalance across the rarer situation-entity types. The three collapsed
features are what ISAAC actually reports, and they are the numbers to rely on.
"""


# ---------------------------------------------------------------------------
# Full report -- rendered into the top-level "Performance & citation" tab.
# ---------------------------------------------------------------------------

FRAMING = f"""
# Model performance

Every classifier in this Space was evaluated on data held out from its own
training set. The numbers below are the final, released-checkpoint values
reported in the ISAAC manuscript; nothing here is estimated or approximate.

Two things are worth reading before you use any of these numbers in a write-up:

1. **Held-out performance describes a model, not the corpus.** For the relevance
   classifiers especially, the model is one stage in a four-stage filter. The
   end-to-end residual irrelevance rate from the human audit of the finished
   corpus is reported alongside each classifier, and it is the more meaningful
   figure for most claims about data quality.
2. **Race and skin tone are thresholded.** Both apply a P(relevant) > 0.6
   decision rule rather than plain argmax. This Space reproduces that rule. If
   you load the weights yourself and take an argmax, you will get a more
   permissive classifier than the one that built ISAAC.

Full methodological detail, including the annotation protocol and the
stage-by-stage audit, is in the manuscript and its Appendix C.
"""

CITATION_MD = f"""
## Citation

If you use these classifiers, please cite the ISAAC paper. **One citation covers
the whole project** — the corpus, the pipeline, and every model. Please do not
cite the models or the repository separately.

**APA**

> {PAPER_CITE.replace('*', '')}

**BibTeX**

```bibtex
@article{{hemmatian2026isaac,
  author = {{Hemmatian, Babak and Hadjarab, Sarah and Chen, Jessica and Kurdi, Benedek}},
  title  = {{The {{Illinois}} Social Attitudes Aggregate Corpus ({{ISAAC}}): An Open Tool and Reproducible Pipeline for Analyzing Social Group Discourse at Scale}},
  year   = {{2026}},
  note   = {{Manuscript submitted for publication}}
}}
```

## Terms

The classifiers exposed here are released under a Creative Commons Attribution
4.0 International License. The ISAAC corpus itself is governed separately by the
project [Data Use Agreement]({DUA_URL}).

## Elsewhere

* Corpus, samples, and SQL playground: <{SITE_URL}>
* Pipeline source, keyword lists, and pattern sets: <{REPO_URL}>
"""


def full_report_md() -> str:
    parts = [FRAMING, "\n## Relevance classifiers\n"]
    for group in RELEVANCE:
        parts.append(f"\n### {GROUP_LABELS[group]}\n")
        parts.append(relevance_summary_md(group, heading=False))

    parts.append(f"""
### Residual irrelevance across all six distinctions

The design target was under 10% irrelevant content in the finished corpus, for
every distinction and both post types. A final double-rated audit of stratified
random samples (*k* = {RESIDUAL_POOLED['k']} comments pooled) confirmed it.

| Distinction | *k* | Comments, stringent | Comments, lenient | κ | Raw agreement | Submissions |
| --- | --- | --- | --- | --- | --- | --- |
""")
    for group, spec in RELEVANCE.items():
        r = spec["residual"]
        kappa = "≈ 0" if r["kappa"] is None else m(r["kappa"])
        parts.append(
            f"| {GROUP_LABELS[group].split(' (')[0]} | {r['k']} | "
            f"{pct(r['stringent'])} | {pct(r['lenient'])} | {kappa} | "
            f"{pct(r['raw_agreement'])} | {pct(r['submissions'])} |\n"
        )
    p = RESIDUAL_POOLED
    parts.append(
        f"| **Pooled** | **{p['k']}** | **{pct(p['stringent'])}** | "
        f"**{pct(p['lenient'])}** | — | — | **{pct(p['submissions'])}** |\n"
    )
    parts.append(
        "\nThe stringent rule counts a post as irrelevant if *either* annotator "
        "judged it so; the lenient rule requires both.\n"
    )

    parts.append("\n## Moralization classifier\n")
    parts.append(moralization_summary_md())
    parts.append("\n## Generalization classifiers\n")
    parts.append(generalization_summary_md())

    parts.append(f"""
## A note on the off-the-shelf labels

ISAAC also carries sentiment (VADER, TextBlob, Stanza) and emotion
(emotion-english-distilroberta-base, roberta-base-go_emotions, EmoBERTa) labels.
Those models are not in-house and are not served by this Space; their published
benchmarks and ISAAC's ensemble-agreement analyses are reported in the
manuscript.
""")

    parts.append(CITATION_MD)
    return "".join(parts)


# ---------------------------------------------------------------------------
# Derived-artifact rendering
# ---------------------------------------------------------------------------

_CARD_RENDERERS = {
    "isaac-moralization": moralization_summary_md,
    "isaac-generalization": generalization_summary_md,
    "isaac-generalization-segmentation": generalization_summary_md,
    **{
        f"isaac-relevance-{g}": (lambda g=g: relevance_summary_md(g))
        for g in RELEVANCE
    },
}


def readme_section_md() -> str:
    """The block pasted into the Space README under '## Performance'."""
    rows = []
    for group, spec in RELEVANCE.items():
        # Summarize with the largest evaluation slice available for the released
        # checkpoint; the smaller retraining slices are shown in full in-app.
        best = max(spec["heldout"], key=lambda row: row["k"])
        thr = "P(rel) > 0.6" if spec["threshold"] else "argmax"
        rows.append(
            f"| Relevance — {GROUP_LABELS[group].split(' (')[0].lower()} | "
            f"`{spec['base_model'].split('/')[-1]}` | {best['k']} | "
            f"{m(best['f1'])} | {pct(spec['residual']['stringent'])} | {thr} |"
        )
    d, se = MORALIZATION, GENERALIZATION["situation_entity"]
    return f"""## Performance

Reported in full in the app's **Performance & citation** tab, which is rendered
from `performance.py` — the single source of truth for every metric this project
publishes, including the nine model cards. Summary:

| Model | Base | Eval *k* | Headline | Residual irrelevance | Decision rule |
| --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}
| Moralization | `{d['base_model'].split('/')[-1]}` | {d['k']:,} | macro F1 {m(d['macro_f1'])} | — | argmax |
| Generalization (18-way) | `{GENERALIZATION['base_model'].split('/')[-1]}` | {GENERALIZATION['k']:,} | acc. {m(se['accuracy'])}, macro F1 {m(se['macro_f1'])} | — | argmax |
| Generalization (segmenter) | `{SEGMENTATION['base_model'].split('/')[-1]}` | — | {pct(SEGMENTATION['coverage'])} clause-span coverage | — | argmax |

"Headline" is held-out F1 for the relevance models. "Residual irrelevance" is the
stringent-rule rate in the finished corpus after all four filtering stages, from
the double-rated human audit — the figure to quote for corpus quality. The
collapsed generalization features (genericity {m(GENERALIZATION['features'][0]['macro_f1'])},
eventivity {m(GENERALIZATION['features'][1]['macro_f1'])},
boundedness/habituality {m(GENERALIZATION['features'][2]['macro_f1'])} macro F1)
are what ISAAC reports and are stronger than the 18-way figure.
"""


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["--readme"]:
        print(readme_section_md())
    elif args[:1] == ["--card"] and len(args) == 2:
        renderer = _CARD_RENDERERS.get(args[1])
        if renderer is None:
            sys.exit(f"unknown repo: {args[1]}\nknown: {', '.join(_CARD_RENDERERS)}")
        print(renderer())
    else:
        print(full_report_md())
