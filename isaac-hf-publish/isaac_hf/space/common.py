from __future__ import annotations

# Must be imported before torch: on ZeroGPU hardware the `spaces` package
# patches torch's CUDA layer at import time. Every other module in this Space
# imports `common` before touching torch, so this one import fixes the ordering
# for all of them. ZeroGPU also refuses to start a Space that registers no
# @spaces.GPU function; the decorated entry points live in the task modules.
import spaces  # noqa: F401  -- imported for its import-time side effect

import csv
import os
import tempfile
from pathlib import Path

import gradio as gr
import torch

# hf_spaces/isaac/common.py -> repo root, for the in-repo models/ fallback.
# On a Space this file is deployed flat at /app/common.py, which has no third
# parent, so parents[2] raises IndexError at import and the app never starts.
# REPO_ROOT is only ever used to look for a local models/ directory, so falling
# back to this file's own directory is correct: the lookup simply misses and
# resolve_model() falls through to the released Hub repo, which is what a Space
# should use anyway.
_HERE = Path(__file__).resolve()
REPO_ROOT = _HERE.parents[2] if len(_HERE.parents) > 2 else _HERE.parent

# On ZeroGPU a real GPU only exists inside @spaces.GPU functions; outside them
# torch runs in CUDA emulation mode, so "cuda" is the correct target either way.
# Off ZeroGPU (local runs, plain CPU/GPU Spaces) fall back to what's available.
ON_ZERO_GPU = bool(os.environ.get("SPACES_ZERO_GPU"))
DEVICE = torch.device(
    "cuda" if ON_ZERO_GPU or torch.cuda.is_available() else "cpu"
)

# Private (or gated) model repos need a read token. On Spaces, add it as a
# secret named HF_TOKEN under Settings -> Variables and secrets. huggingface_hub
# picks the variable up on its own, but passing it explicitly keeps the failure
# legible when it is missing: without it the Hub answers 401 and transformers
# raises a bare OSError at first use, long after the build has gone green.
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

ALLOWED_EXTENSIONS = {".txt", ".csv"}

# ZeroGPU charges quota against the duration a call *requests*, not the time it
# actually uses, so the file tabs estimate their own budget from the input size
# instead of always reserving a worst-case slot. Overrunning the request kills
# the call, so the estimates assume a cold model load every time.
MAX_DURATION = 180
EMPTY_INPUT_DURATION = 15

UPLOAD_HINT = (
    "Upload a .txt (one text per line) or .csv (text in the first column) file."
)
BAD_EXTENSION = "Unsupported file type. Please upload a .txt or .csv file."
NO_ROWS = "No text rows found in the uploaded file."


def resolve_source(relative: str, hub_id: str) -> str:
    """Local-first, Hub-fallback model resolution, shared by all three tasks.

    1. $ISAAC_MODELS_DIR/<relative>   (explicit local override)
    2. <repo_root>/models/<relative>  (in-repo local run)
    3. <hub_id>                       (the released HF model repo)
    """
    models_dir = os.environ.get("ISAAC_MODELS_DIR")
    if models_dir:
        candidate = Path(models_dir) / relative
        if candidate.exists():
            return str(candidate)

    sibling = REPO_ROOT / "models" / relative
    if sibling.exists():
        return str(sibling)

    return hub_id


def load_error(what: str, *sources: str) -> gr.Error:
    """Uniform, actionable message for a model that would not load."""
    return gr.Error(
        f"Could not load the {what} model(s) from {', '.join(sources)}. "
        "If those model repos are private, add a read token as a Space secret "
        "named HF_TOKEN (Settings -> Variables and secrets), then restart the "
        "Space."
    )


def iter_texts(path: Path):
    """Yield the input texts from a .txt (one per line) or .csv (first column)."""
    suffix = path.suffix.lower()
    if suffix == ".txt":
        with path.open("r", encoding="utf-8-sig", errors="ignore") as f:
            for line in f:
                text = line.strip()
                if text:
                    yield text
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if row and row[0].strip():
                    yield row[0].strip()


def count_rows(file) -> int | None:
    """Row count for duration estimation. None if the file cannot be read, so
    the caller can fall back to the maximum request."""
    try:
        return sum(1 for _ in iter_texts(Path(file)))
    except OSError:
        return None


def duration_for_rows(file, cold_load: int, seconds_per_unit: float, unit: int = 1) -> int:
    """Dynamic @spaces.GPU duration: cold model load plus per-unit inference,
    where a unit is `unit` input rows (e.g. one batch)."""
    if file is None:
        return EMPTY_INPUT_DURATION
    n_rows = count_rows(file)
    if n_rows is None:
        return MAX_DURATION
    n_units = max(1, -(-n_rows // unit))
    return min(MAX_DURATION, int(cold_load + n_units * seconds_per_unit))


def write_results_csv(header, rows) -> str:
    """Write a results CSV to a temp file and return its path for gr.File."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".csv", newline="", encoding="utf-8"
    )
    with tmp as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)
    return tmp.name


FOOTER = """
---
By using this tool you agree to ISAAC's
[Use Agreement](https://github.com/BabakHemmatian/Illinois_Social_Attitudes/blob/main/Data_Use_Agreement.md).
For citation information and more details, see the
[Illinois Social Attitudes (ISAAC) repository](https://github.com/BabakHemmatian/Illinois_Social_Attitudes).
"""
