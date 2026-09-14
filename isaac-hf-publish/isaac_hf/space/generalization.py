from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import common  # imports `spaces` before torch; see common.py
import gradio as gr
import matplotlib

matplotlib.use("Agg")  # headless backend for Spaces / servers
import matplotlib.pyplot as plt
import numpy as np
import performance
import re
import spaces
import torch
from common import (
    ALLOWED_EXTENSIONS,
    BAD_EXTENSION,
    DEVICE,
    HF_TOKEN,
    MAX_DURATION,
    NO_ROWS,
    UPLOAD_HINT,
    duration_for_rows,
    iter_texts,
    load_error,
    resolve_source,
    write_results_csv,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

# ---------------------------------------------------------------------------
# The generalization pipeline uses TWO fine-tuned models plus the base
# roberta-base tokenizer:
#   * segmenter  -> token classifier that splits text into clauses
#   * classifier -> sequence classifier that labels each clause's discourse type
# ---------------------------------------------------------------------------

SEG_SUBDIR = "label_generalization/label_generalization_segmentation"
CLF_SUBDIR = "label_generalization/label_generalization"

HUB_SEG = os.environ.get(
    "ISAAC_GENERALIZATION_SEG_REPO", "ISAAC-corpus/isaac-generalization-segmentation"
)
HUB_CLF = os.environ.get(
    "ISAAC_GENERALIZATION_REPO", "ISAAC-corpus/isaac-generalization"
)

MAX_WORDS_PER_SNIPPET = 200
SEG_MAX_LENGTH = 512
CLF_MAX_LENGTH = 128
CLF_BATCH_SIZE = 32

# Both models load together, so the cold-start budget is larger than the
# single-model tasks'.
COLD_LOAD_SECONDS = 45
SECONDS_PER_TEXT = 2

labels2attrs = {
    "##BOUNDED EVENT (SPECIFIC)": ("specific", "dynamic", "episodic"),
    "##BOUNDED EVENT (GENERIC)": ("generic", "dynamic", "episodic"),
    "##UNBOUNDED EVENT (SPECIFIC)": ("specific", "dynamic", "static"),
    "##UNBOUNDED EVENT (GENERIC)": ("generic", "dynamic", "static"),
    "##BASIC STATE": ("specific", "stative", "static"),
    "##COERCED STATE (SPECIFIC)": ("specific", "dynamic", "static"),
    "##COERCED STATE (GENERIC)": ("generic", "dynamic", "static"),
    "##PERFECT COERCED STATE (SPECIFIC)": ("specific", "dynamic", "episodic"),
    "##PERFECT COERCED STATE (GENERIC)": ("generic", "dynamic", "episodic"),
    "##GENERIC SENTENCE (DYNAMIC)": ("generic", "dynamic", "habitual"),
    "##GENERIC SENTENCE (STATIC)": ("generic", "stative", "static"),
    "##GENERIC SENTENCE (HABITUAL)": ("generic", "stative", "habitual"),
    "##GENERALIZING SENTENCE (DYNAMIC)": ("specific", "dynamic", "habitual"),
    "##GENERALIZING SENTENCE (STATIVE)": ("specific", "stative", "habitual"),
    "##QUESTION": ("NA", "NA", "NA"),
    "##IMPERATIVE": ("NA", "NA", "NA"),
    "##NONSENSE": ("NA", "NA", "NA"),
    "##OTHER": ("NA", "NA", "NA"),
}

label_names = list(labels2attrs.keys())
index2label = {i: label for i, label in enumerate(label_names)}


@lru_cache(maxsize=1)
def _load_models():
    seg_source = resolve_source(SEG_SUBDIR, HUB_SEG)
    clf_source = resolve_source(CLF_SUBDIR, HUB_CLF)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "roberta-base", use_fast=True, add_prefix_space=True
        )
        clause_model = (
            AutoModelForTokenClassification.from_pretrained(
                seg_source, token=HF_TOKEN
            )
            .to(DEVICE)
            .eval()
        )
        classification_model = (
            AutoModelForSequenceClassification.from_pretrained(
                clf_source, token=HF_TOKEN
            )
            .to(DEVICE)
            .eval()
        )
    except OSError as exc:
        raise load_error("generalization", seg_source, clf_source) from exc
    return tokenizer, clause_model, classification_model


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def auto_split(text: str, max_words: int = MAX_WORDS_PER_SNIPPET) -> List[str]:
    sentences = split_sentences(text)
    if not sentences:
        return []

    snippets: List[str] = []
    current_words: List[str] = []
    for sentence in sentences:
        sent_words = sentence.split()
        if current_words and len(current_words) + len(sent_words) > max_words:
            snippets.append(" ".join(current_words).strip())
            current_words = sent_words[:]
        else:
            current_words.extend(sent_words)
    if current_words:
        snippets.append(" ".join(current_words).strip())
    return snippets


def majority_vote(values: List[int]) -> int:
    if not values:
        return 1
    counts = np.bincount(values)
    return int(np.argmax(counts))


# ---------------------------------------------------------------------------
# Clause segmentation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def get_pred_clause_labels(text: str) -> List[int]:
    tokenizer, clause_model, _ = _load_models()
    words = text.strip().split()
    if not words:
        return []

    enc = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=SEG_MAX_LENGTH,
        padding="max_length",
    )
    word_ids = enc.word_ids(batch_index=0)
    model_inputs = {k: v.to(DEVICE) for k, v in enc.items()}
    logits = clause_model(**model_inputs).logits[0]
    token_preds = logits.argmax(dim=-1).detach().cpu().tolist()

    aligned_preds: List[List[int]] = [[] for _ in words]
    for token_idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        aligned_preds[word_id].append(token_preds[token_idx])

    return [majority_vote(preds) if preds else 1 for preds in aligned_preds]


def seg_clause(text: str) -> List[str]:
    words = text.strip().split()
    if not words:
        return []

    labels = get_pred_clause_labels(text)
    segmented_clauses: List[List[str]] = []
    prev_label = 2
    current_clause: List[str] | None = None

    for word, label in zip(words, labels):
        if prev_label == 2:
            current_clause = []
        if current_clause is not None:
            current_clause.append(word)
        if label == 2 and prev_label in [0, 1]:
            segmented_clauses.append(current_clause[:])
            current_clause = None
        prev_label = label

    if current_clause:
        segmented_clauses.append(current_clause[:])

    return [" ".join(clause) for clause in segmented_clauses if clause]


# ---------------------------------------------------------------------------
# Clause classification
# ---------------------------------------------------------------------------

@torch.inference_mode()
def get_pred_classification_labels(
    clauses: List[str], batch_size: int = CLF_BATCH_SIZE
) -> List[Tuple[str, Tuple[str, str, str]]]:
    if not clauses:
        return []

    tokenizer, _, classification_model = _load_models()
    results: List[Tuple[str, Tuple[str, str, str]]] = []
    for i in range(0, len(clauses), batch_size):
        batch = clauses[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=CLF_MAX_LENGTH,
            padding="max_length",
        )
        model_inputs = {k: v.to(DEVICE) for k, v in enc.items()}
        logits = classification_model(**model_inputs).logits
        pred_ids = logits.argmax(dim=-1).detach().cpu().tolist()
        pred_labels = [index2label[idx] for idx in pred_ids]
        results.extend(
            (clause, labels2attrs[label]) for clause, label in zip(batch, pred_labels)
        )
    return results


def clause_labels_for_text(text: str) -> List[Tuple[str, Tuple[str, str, str]]]:
    """Segment a text into clauses and classify each. Shared by both tabs."""
    text = (text or "").strip()
    if not text:
        return []
    all_clauses: List[str] = []
    for snippet in auto_split(text):
        all_clauses.extend(seg_clause(snippet))
    return get_pred_classification_labels(all_clauses)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def summarize(clause2labels) -> Tuple[int, Dict, Dict, Dict]:
    """Return (n_clauses, genericity_props, eventivity_props, boundedness_props),
    each proportion taken over the total clause count (NA included), matching the
    pie charts."""
    total = len(clause2labels)
    gen = {"generic": 0, "specific": 0, "NA": 0}
    eve = {"dynamic": 0, "stative": 0, "NA": 0}
    bnd = {"static": 0, "episodic": 0, "habitual": 0, "NA": 0}
    for _, (genericity, eventivity, boundedness) in clause2labels:
        gen[genericity] += 1
        eve[eventivity] += 1
        bnd[boundedness] += 1

    def prop(d: Dict[str, int]) -> Dict[str, float]:
        return {k: (v / total if total else 0.0) for k, v in d.items()}

    return total, prop(gen), prop(eve), prop(bnd)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def label_visualization(clause2labels):
    total_clauses = len(clause2labels)
    if total_clauses == 0:
        fig = plt.figure(figsize=(10, 4))
        plt.text(0.5, 0.5, "No clauses detected.", ha="center", va="center")
        plt.axis("off")
        return fig

    genericity_labels, aspect_labels, boundedness_labels = [], [], []
    for _, attrs in clause2labels:
        genericity_label, aspect_label, boundedness_label = attrs
        genericity_labels.append(genericity_label)
        aspect_labels.append(aspect_label)
        boundedness_labels.append(boundedness_label)

    genericity_dict = {
        "Generic": genericity_labels.count("generic"),
        "Specific": genericity_labels.count("specific"),
        "NA": genericity_labels.count("NA"),
    }
    aspect_dict = {
        "Dynamic": aspect_labels.count("dynamic"),
        "Stative": aspect_labels.count("stative"),
        "NA": aspect_labels.count("NA"),
    }
    boundedness_dict = {
        "Static": boundedness_labels.count("static"),
        "Episodic": boundedness_labels.count("episodic"),
        "Habitual": boundedness_labels.count("habitual"),
        "NA": boundedness_labels.count("NA"),
    }

    def proportions(d):
        filtered = {k: v / total_clauses for k, v in d.items() if v > 0}
        return list(filtered.keys()), list(filtered.values())

    fig, axs = plt.subplots(1, 3, figsize=(10, 5))
    fig.tight_layout(pad=5.0)

    labels, values = proportions(genericity_dict)
    axs[0].pie(values, labels=labels, autopct="%.0f%%", normalize=True)
    axs[0].set_title("Genericity")

    labels, values = proportions(aspect_dict)
    axs[1].pie(values, labels=labels, autopct="%.0f%%", normalize=True)
    axs[1].set_title("Eventivity")

    labels, values = proportions(boundedness_dict)
    axs[2].pie(values, labels=labels, autopct="%.0f%%", normalize=True)
    axs[2].set_title("Boundedness / Habituality")

    return fig


def render_pipeline(clause2labels):
    """Turn labeled clauses into the three display payloads: numbered clauses,
    attribute-labeled clauses, and the proportion figure.

    Kept separate from the model pass so the caller can run inference on the GPU
    and build the matplotlib figure afterwards — on ZeroGPU the decorated call
    runs in a separate process, and a Figure is a poor thing to ship across that
    boundary."""
    output_clauses = [
        (clause, str(i + 1)) for i, (clause, _) in enumerate(clause2labels)
    ]
    highlighted_attrs = [(clause, str(attrs)) for clause, attrs in clause2labels]
    figure = label_visualization(clause2labels)
    return output_clauses, highlighted_attrs, figure


# ---------------------------------------------------------------------------
# Entry points
#
# Only the model pass is GPU-decorated; the figure is built afterwards, in this
# process, from the plain (clause, attributes) data the GPU call returns.
# ---------------------------------------------------------------------------

def _text_duration(text) -> int:
    n_words = len((text or "").split())
    n_snippets = max(1, -(-n_words // MAX_WORDS_PER_SNIPPET))
    return min(MAX_DURATION, COLD_LOAD_SECONDS + n_snippets * SECONDS_PER_TEXT)


@spaces.GPU(duration=_text_duration)
def _clause_labels(text):
    return clause_labels_for_text(text)


def analyze_text(text):
    return render_pipeline(_clause_labels(text))


def _file_duration(file) -> int:
    return duration_for_rows(file, COLD_LOAD_SECONDS, SECONDS_PER_TEXT)


@spaces.GPU(duration=_file_duration)
def analyze_file(file):
    if file is None:
        return None, UPLOAD_HINT

    path = Path(file)
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        return None, BAD_EXTENSION

    texts = list(iter_texts(path))
    if not texts:
        return None, NO_ROWS

    def rows():
        for text in texts:
            clause2labels = clause_labels_for_text(text)
            n, gen, eve, bnd = summarize(clause2labels)
            clauses_joined = "\n".join(
                f"{i + 1}: {clause}" for i, (clause, _) in enumerate(clause2labels)
            )
            attrs_joined = "\n".join("/".join(attrs) for _, attrs in clause2labels)
            yield [
                text, n, clauses_joined, attrs_joined,
                f"{gen['generic']:.4f}", f"{gen['specific']:.4f}", f"{gen['NA']:.4f}",
                f"{eve['dynamic']:.4f}", f"{eve['stative']:.4f}", f"{eve['NA']:.4f}",
                f"{bnd['static']:.4f}", f"{bnd['episodic']:.4f}",
                f"{bnd['habitual']:.4f}", f"{bnd['NA']:.4f}",
            ]

    out_path = write_results_csv(
        [
            "text",
            "n_clauses",
            "clauses",
            "clause_attributes",
            "genericity_generic", "genericity_specific", "genericity_NA",
            "eventivity_dynamic", "eventivity_stative", "eventivity_NA",
            "boundedness_static", "boundedness_episodic",
            "boundedness_habitual", "boundedness_NA",
        ],
        rows(),
    )
    return out_path, f"Processed {len(texts)} text(s). Download the full results below."


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

color_panel_1 = [
    "red", "green", "yellow", "DodgerBlue", "orange", "DarkSalmon",
    "pink", "cyan", "gold", "aqua", "violet",
]
# Clause-segmentation panel: each clause gets a rotating color keyed by its
# 1-based index.
index_colormap = {
    str(i): color_panel_1[i % len(color_panel_1)] for i in range(1, 100000)
}

color_panel_2 = [
    "Gray", "DodgerBlue", "Wheat", "OliveDrab", "DarkKhaki", "DarkSalmon",
    "Orange", "Gold", "Aqua", "Tomato", "Violet",
]
# Attribute panel: one color per distinct (genericity, eventivity, boundedness)
# triple.
str_attrs = sorted({str(v) for v in labels2attrs.values()})
attr_colormap = {attr: color for attr, color in zip(str_attrs, color_panel_2)}

DESCRIPTION = """
Segments text into **clauses** and labels each clause along three discourse
dimensions, revealing how generalized vs. anecdotal the language is:

* **Genericity** — generic category (e.g., *humanity*) vs. specific instance (e.g., *my cousin*).
* **Eventivity** — a state (*God is benevolent*) vs. an event (*I went to Nebraska*).
* **Boundedness / Habituality** — bounded (*I ate this morning*) vs. unbounded (*God loves us*) vs. repeated (*I went there for years*).

The most anecdotal content involves specific entities in bounded, non-habitual events.
"""

SINGLE_EXAMPLES = [
    ["My cousin was arrested for marijuana possession, and it ruined his life for years."],
    ["Legalizing marijuana would reduce incarceration costs and improve tax revenue."],
]


def build_ui():
    gr.Markdown(DESCRIPTION)

    # Held-out performance for both models in the chain, collapsed by default;
    # text comes from performance.py, the single source of truth for every
    # published metric.
    with gr.Accordion("Performance for these classifiers", open=False):
        gr.Markdown(performance.generalization_summary_md())

    # Explicit gr.Tabs: these sit inside a top-level task tab in app.py, so the
    # implicit grouping Gradio applies to bare sibling gr.Tab blocks is not
    # something to rely on here.
    with gr.Tabs():
        with gr.Tab("Single text"):
            text_in = gr.Textbox(
                label="Input Text", lines=6, placeholder="Paste text here..."
            )
            text_btn = gr.Button("Analyze", variant="primary")

            # The two text panels show the SAME text twice: once split into
            # numbered clauses, once colored by discourse attributes.
            # Side-by-side so they can be read in register.
            with gr.Row(equal_height=True):
                seg_out = gr.HighlightedText(
                    label="Clause Segmentation",
                    color_map=index_colormap,
                    combine_adjacent=False,
                    show_legend=False,
                )
                attr_out = gr.HighlightedText(
                    label="Attribute Classification (genericity, eventivity, boundedness)",
                    color_map=attr_colormap,
                    combine_adjacent=False,
                    show_legend=True,
                )

            # The proportion charts summarize the whole text; full width below.
            plot_out = gr.Plot(label="Proportion of Attributes")

            gr.Examples(examples=SINGLE_EXAMPLES, inputs=text_in)

            text_btn.click(
                analyze_text, inputs=text_in, outputs=[seg_out, attr_out, plot_out]
            )
            text_in.submit(
                analyze_text, inputs=text_in, outputs=[seg_out, attr_out, plot_out]
            )

        with gr.Tab("Multiple texts (file)"):
            gr.Markdown(
                "Upload a `.txt` file (one text per line) or a `.csv` file "
                "(text in the first column). Each text is segmented and "
                "classified; the results CSV reports the clauses, their "
                "attributes, and the attribute proportions per text."
            )
            file_in = gr.File(label="Input File", file_types=[".txt", ".csv"])
            file_btn = gr.Button("Analyze file", variant="primary")
            file_status = gr.Markdown()
            file_out = gr.File(label="Results CSV")
            file_btn.click(
                analyze_file, inputs=file_in, outputs=[file_out, file_status]
            )
