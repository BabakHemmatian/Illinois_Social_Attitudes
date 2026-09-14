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

LOCAL_DIRNAME = "label_moralization"
HUB_ID = os.environ.get("ISAAC_MORALIZATION_REPO", "ISAAC-corpus/isaac-moralization")

MAX_LENGTH = 512
BATCH_SIZE = 16
COLD_LOAD_SECONDS = 40
SECONDS_PER_BATCH = 2

# Class index -> human label. The training pipeline (label_moralization.py)
# treats argmax==1 as the moralized class.
LABELS = {0: "Non-Moralized", 1: "Moralized"}


@lru_cache(maxsize=1)
def _load():
    source = resolve_source(LOCAL_DIRNAME, HUB_ID)
    try:
        tokenizer = AutoTokenizer.from_pretrained(source, token=HF_TOKEN)
        model = AutoModelForSequenceClassification.from_pretrained(
            source, token=HF_TOKEN
        ).to(DEVICE)
    except OSError as exc:
        raise load_error("moralization", source) from exc
    model.eval()
    return tokenizer, model


@torch.no_grad()
def _predict(texts):
    tokenizer, model = _load()
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
            cls = int(prob.argmax())
            results.append((LABELS[cls], float(prob[cls]), float(prob[1])))
    return results


@spaces.GPU(duration=60)
def classify_text(text):
    if not text or not text.strip():
        return {}
    tokenizer, model = _load()
    with torch.no_grad():
        enc = tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(DEVICE)
        prob = torch.softmax(model(**enc).logits, dim=1)[0].cpu()
    # gr.Label renders this dict as a ranked bar chart of class confidences.
    return {LABELS[0]: float(prob[0]), LABELS[1]: float(prob[1])}


def _file_duration(file) -> int:
    return duration_for_rows(file, COLD_LOAD_SECONDS, SECONDS_PER_BATCH, BATCH_SIZE)


@spaces.GPU(duration=_file_duration)
def classify_file(file):
    if file is None:
        return None, UPLOAD_HINT

    path = Path(file)
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        return None, BAD_EXTENSION

    texts = list(iter_texts(path))
    if not texts:
        return None, NO_ROWS

    preds = _predict(texts)
    out_path = write_results_csv(
        ["text", "moralization", "confidence"],
        (
            [text, label, f"{conf:.4f}"]
            for text, (label, conf, _) in zip(texts, preds)
        ),
    )

    n_moralized = sum(1 for label, _, _ in preds if label == "Moralized")
    status = (
        f"Processed {len(texts)} text(s): "
        f"{n_moralized} Moralized, {len(texts) - n_moralized} Non-Moralized. "
        f"Download the full results below."
    )
    return out_path, status


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

DESCRIPTION = """
Detects whether a text frames its subject in **moral** terms (right/wrong, virtue,
harm, fairness, purity, etc.) versus non-moral terms. This is the document-level
moralization model from the ISAAC project.
"""

SINGLE_EXAMPLES = [
    ["People who cut in line are selfish and should be ashamed of themselves."],
    ["The bus arrives at the corner of 5th and Main every fifteen minutes."],
]


def build_ui():
    gr.Markdown(DESCRIPTION)

    # Held-out performance, collapsed by default; text comes from
    # performance.py, the single source of truth for every published metric.
    with gr.Accordion("Performance for this classifier", open=False):
        gr.Markdown(performance.moralization_summary_md())

    # Explicit gr.Tabs: these sit inside a top-level task tab in app.py, so the
    # implicit grouping Gradio applies to bare sibling gr.Tab blocks is not
    # something to rely on here.
    with gr.Tabs():
        with gr.Tab("Single text"):
            text_in = gr.Textbox(
                label="Input Text", lines=8, placeholder="Paste text here..."
            )
            text_btn = gr.Button("Classify", variant="primary")
            text_out = gr.Label(label="Moralization", num_top_classes=2)
            gr.Examples(examples=SINGLE_EXAMPLES, inputs=text_in)
            text_btn.click(classify_text, inputs=text_in, outputs=text_out)
            text_in.submit(classify_text, inputs=text_in, outputs=text_out)

        with gr.Tab("Multiple texts (file)"):
            gr.Markdown(
                "Upload a `.txt` file (one text per line) or a `.csv` file "
                "(text in the first column)."
            )
            file_in = gr.File(label="Input File", file_types=[".txt", ".csv"])
            file_btn = gr.Button("Classify file", variant="primary")
            file_status = gr.Markdown()
            file_out = gr.File(label="Results CSV")
            file_btn.click(
                classify_file, inputs=file_in, outputs=[file_out, file_status]
            )
