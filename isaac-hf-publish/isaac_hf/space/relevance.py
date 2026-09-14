from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import common  # imports `spaces` before torch; see common.py
import gradio as gr
import performance
import spaces
import torch
from common import (
    ALLOWED_EXTENSIONS,
    BAD_EXTENSION,
    DEVICE,
    HF_TOKEN,
    NO_ROWS,
    UPLOAD_HINT,
    duration_for_rows,
    iter_texts,
    load_error,
    resolve_source,
    write_results_csv,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ---------------------------------------------------------------------------
# Six fine-tuned relevance classifiers, one per social distinction.
#
# Models are loaded lazily and cached: only the distinction(s) actually used are
# pulled into memory, and at most CACHE_SIZE stay resident at once.
# ---------------------------------------------------------------------------

GROUPS = ["ability", "age", "race", "sexuality", "skin_tone", "weight"]

# Released model-repo naming: prefix + group, e.g. "ISAAC-corpus/isaac-relevance-race".
HUB_PREFIX = os.environ.get(
    "ISAAC_RELEVANCE_REPO_PREFIX", "ISAAC-corpus/isaac-relevance-"
)

# Number of distinction models kept resident simultaneously.
CACHE_SIZE = 2

MAX_LENGTH = 512
BATCH_SIZE = 16
COLD_LOAD_SECONDS = 40
SECONDS_PER_BATCH = 2

# Class index -> human label. The pipeline keeps argmax==1 rows as relevant.
LABELS = {0: "Not relevant", 1: "Relevant"}

# Confidence thresholding (matches filter_relevance.py): for race and skin_tone
# the model only assigns the rare "relevant" class when its probability clears
# THRESHOLD; otherwise it falls back to the next-most-probable class.
THRESHOLDED_GROUPS = {"race", "skin_tone"}
THRESHOLD_CLASS = 1
THRESHOLD = 0.6


@lru_cache(maxsize=CACHE_SIZE)
def _load(group: str):
    source = resolve_source(f"filter_relevance_{group}", f"{HUB_PREFIX}{group}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(source, token=HF_TOKEN)
        model = AutoModelForSequenceClassification.from_pretrained(
            source, token=HF_TOKEN
        ).to(DEVICE)
    except OSError as exc:
        raise load_error(f"'{group}' relevance", source) from exc
    model.eval()
    return tokenizer, model


def _decide(prob, group: str) -> int:
    """Map a class-probability vector to a predicted class index, replicating
    the thresholding behavior of filter_relevance.py.predict_tokenized."""
    if group in THRESHOLDED_GROUPS:
        if prob[THRESHOLD_CLASS] > THRESHOLD:
            return THRESHOLD_CLASS
        masked = prob.clone()
        masked[THRESHOLD_CLASS] = -1.0
        return int(masked.argmax())
    return int(prob.argmax())


@torch.no_grad()
def _predict(texts, group):
    tokenizer, model = _load(group)
    results = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(DEVICE)
        probs = torch.softmax(model(**enc).logits, dim=1).cpu()
        for prob in probs:
            cls = _decide(prob, group)
            results.append((LABELS[cls], float(prob[cls]), float(prob[1])))
    return results


@spaces.GPU(duration=60)
def classify_text(text, group):
    if not text or not text.strip():
        return {}
    tokenizer, model = _load(group)
    with torch.no_grad():
        enc = tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(DEVICE)
        prob = torch.softmax(model(**enc).logits, dim=1)[0].cpu()
    cls = _decide(prob, group)
    # Surface the model's raw class probabilities; the decided label is marked
    # so thresholding effects stay visible.
    return {
        f"{LABELS[1]} (decided)" if cls == 1 else LABELS[1]: float(prob[1]),
        f"{LABELS[0]} (decided)" if cls == 0 else LABELS[0]: float(prob[0]),
    }


def _file_duration(file, group=None) -> int:
    return duration_for_rows(file, COLD_LOAD_SECONDS, SECONDS_PER_BATCH, BATCH_SIZE)


@spaces.GPU(duration=_file_duration)
def classify_file(file, group):
    if file is None:
        return None, UPLOAD_HINT

    path = Path(file)
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        return None, BAD_EXTENSION

    texts = list(iter_texts(path))
    if not texts:
        return None, NO_ROWS

    preds = _predict(texts, group)
    out_path = write_results_csv(
        ["text", "social_distinction", "relevance", "confidence"],
        (
            [text, group, label, f"{conf:.4f}"]
            for text, (label, conf, _) in zip(texts, preds)
        ),
    )

    n_relevant = sum(1 for label, _, _ in preds if label == "Relevant")
    status = (
        f"Processed {len(texts)} text(s) against the '{group}' distinction: "
        f"{n_relevant} Relevant, {len(texts) - n_relevant} Not relevant. "
        f"Download the full results below."
    )
    return out_path, status


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

DESCRIPTION = """
Judges whether a text is **relevant to a given social distinction**, that is,
whether it actually discusses people in terms of that attribute. Pick a
distinction below; the app loads the matching fine-tuned classifier on demand.
For **race** and **skin tone**, a 0.6 confidence threshold is applied to the
"relevant" class (matching the ISAAC pipeline), so a text is only marked relevant
when the model is sufficiently confident.
"""

SINGLE_EXAMPLES = [
    ["My grandmother uses a wheelchair and the new ramp finally lets her visit on her own.", "ability"],
    ["The recipe calls for two cups of flour and a pinch of salt.", "ability"],
    ["People kept assuming things about him because he was Black.", "race"],
]


def build_ui():
    gr.Markdown(DESCRIPTION)

    group_dd = gr.Dropdown(
        choices=GROUPS,
        value="ability",
        label="Social Distinction",
        info="Which fine-tuned relevance classifier to apply.",
    )

    # Held-out performance for the selected distinction, collapsed by default.
    # Kept in this tab rather than only under "Performance & citation" so the
    # thresholding caveat sits next to the control that triggers it. Text comes
    # from performance.py, the single source of truth for every published metric.
    with gr.Accordion("Performance for this classifier", open=False):
        perf_md = gr.Markdown(performance.relevance_summary_md(GROUPS[0]))
    group_dd.change(performance.relevance_summary_md, inputs=group_dd, outputs=perf_md)

    # Explicit gr.Tabs: these sit inside a top-level task tab in app.py, so the
    # implicit grouping Gradio applies to bare sibling gr.Tab blocks is not
    # something to rely on here.
    with gr.Tabs():
        with gr.Tab("Single text"):
            text_in = gr.Textbox(
                label="Input Text", lines=8, placeholder="Paste text here..."
            )
            text_btn = gr.Button("Classify", variant="primary")
            text_out = gr.Label(label="Relevance", num_top_classes=2)
            gr.Examples(examples=SINGLE_EXAMPLES, inputs=[text_in, group_dd])
            text_btn.click(classify_text, inputs=[text_in, group_dd], outputs=text_out)
            text_in.submit(classify_text, inputs=[text_in, group_dd], outputs=text_out)

        with gr.Tab("Multiple texts (file)"):
            gr.Markdown(
                "Upload a `.txt` file (one text per line) or a `.csv` file "
                "(text in the first column). All rows are scored against the "
                "selected distinction."
            )
            file_in = gr.File(label="Input File", file_types=[".txt", ".csv"])
            file_btn = gr.Button("Classify file", variant="primary")
            file_status = gr.Markdown()
            file_out = gr.File(label="Results CSV")
            file_btn.click(
                classify_file,
                inputs=[file_in, group_dd],
                outputs=[file_out, file_status],
            )
