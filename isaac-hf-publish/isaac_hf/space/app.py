from __future__ import annotations

# `common` must be imported first: it imports `spaces` before torch, which is
# what ZeroGPU requires. The task modules below all import torch transitively.
import common

import gradio as gr

import generalization
import moralization
import performance
import relevance

DESCRIPTION = """
# ISAAC Text Classifiers

Coding-free demos for the classifiers built for the **Illinois Social Attitudes
(ISAAC)** project. Pick a task below; each one loads its fine-tuned model on
demand and offers a single-text tab and a bulk file tab.

* **Relevance**: is a text about a given social distinction (ability, age, race,
  sexuality, skin tone, weight)?
* **Moralization**: does a text frame its subject in moral terms?
* **Generalization**: how generalized vs. anecdotal is the language, clause by clause?

Held-out performance for every model is reported under
**Performance & citation**, and each task tab repeats the figures for the model
it runs.
"""

with gr.Blocks(title="ISAAC Text Classifiers") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Tabs():
        with gr.Tab("Relevance"):
            relevance.build_ui()
        with gr.Tab("Moralization"):
            moralization.build_ui()
        with gr.Tab("Generalization"):
            generalization.build_ui()
        with gr.Tab("Performance & citation"):
            gr.Markdown(performance.full_report_md())

    gr.Markdown(common.FOOTER)


if __name__ == "__main__":
    demo.launch()
