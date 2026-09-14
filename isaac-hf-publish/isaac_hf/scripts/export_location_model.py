"""Export the ISAAC location model as a pickle-free release bundle.

The trained artifacts on disk are scikit-learn pickles. Publishing pickles means
(a) HuggingFace flags the repo with a security banner and (b) anyone loading the
weights executes arbitrary code. Neither is acceptable for a released asset, and
neither is necessary: these are logistic regressions, so the entire model is a
coefficient matrix, an intercept, and a feature vocabulary.

This script reads the pickles *without* importing scikit-learn — every non-numpy
class is replaced by a stub that captures its state — so it also works when the
installed scikit-learn version no longer matches the one used for training.

What it writes (see EXCLUDE below for what it deliberately does not):

    location_model.npz   float32 coefficients + intercepts for all six models,
                         plus the tf-idf idf_ vector
    vocab.json           the 50,000 word features and 20,024 structured features
    classes.json         class labels per model, in coefficient row order
    config.json          feature-extraction settings and deployment thresholds
    metrics/*.json       held-out evaluation metrics, copied verbatim

Usage:
    python export_location_model.py --models-dir ./models --out ./location_release
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path

import numpy as np

# Deployment thresholds, mirrored from code/label_location.py. Changing these
# changes the labels the pipeline emits, so they travel with the weights.
THRESHOLDS = {
    "TOP_CONF_THRESHOLD": 0.60,
    "REG_CONF_MARGIN": 0.10,
    "STA_CONF_MARGIN": 0.05,
}

# Files under preprocessed_streaming/ that must NEVER be published: these are the
# training data whose release Appendix A rules out, because the labels derive
# from explicit self-disclosures and would permit re-identification.
EXCLUDE = {
    "X_words__src-all.npz",
    "X_words_masked__src-all.npz",
    "X_struct__src-all.npz",
    "labels_and_splits__src-all.npz",
    "users__src-all.npy",
}

MODELS = [
    ("words", "top"), ("words", "region"), ("words", "state"),
    ("struct", "top"), ("struct", "region"), ("struct", "state"),
]


class _Stub:
    """Stands in for any non-numpy class so its state can be read, not executed."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.__dict__["_state"] = state


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy") or module == "_codecs":
            return super().find_class(module, name)
        return type(name, (_Stub,), {})


def _load(path: Path):
    with open(path, "rb") as fh:
        return _SafeUnpickler(fh).load()


def _state(obj):
    if isinstance(obj, dict):
        return obj
    return getattr(obj, "_state", None) or getattr(obj, "__dict__", {}) or {}


def _find(obj, key, depth=0):
    """Depth-first search for `key` in a nested state tree."""
    if depth > 4:
        return None
    st = _state(obj)
    if isinstance(st, dict):
        if key in st:
            return st[key]
        for value in st.values():
            if isinstance(value, (dict, _Stub)) or hasattr(value, "_state"):
                found = _find(value, key, depth + 1)
                if found is not None:
                    return found
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", default="./models", type=Path)
    ap.add_argument("--out", default="./location_release", type=Path)
    args = ap.parse_args()

    root = args.models_dir / "label_location"
    trained, preproc_dir = root / "trained_lr", root / "preprocessed_streaming"
    out = args.out
    (out / "metrics").mkdir(parents=True, exist_ok=True)

    # ---- coefficients -----------------------------------------------------
    arrays, classes = {}, {}
    for feature_set, level in MODELS:
        name = f"{feature_set}_{level}"
        src = trained / f"lr__{feature_set}__{level}__src-all.pkl"
        if not src.exists():
            raise SystemExit(f"missing expected model: {src}")
        coef = _find(_load(src), "coef_")
        intercept = _find(_load(src), "intercept_")
        cls = _find(_load(src), "classes_")
        arrays[f"{name}_coef"] = np.asarray(coef, dtype=np.float32)
        arrays[f"{name}_intercept"] = np.asarray(intercept, dtype=np.float32)
        classes[name] = [str(c) for c in np.asarray(cls).tolist()]
        print(f"  {name:<14} coef {arrays[name + '_coef'].shape}  "
              f"{len(classes[name])} classes")

    # ---- vocabulary and tf-idf -------------------------------------------
    pre = _load(preproc_dir / "preprocessor__src-all.pkl")
    words = list(pre["selected_words"])
    idf = _find(pre["word_tfidf"], "idf_")
    struct_names = _find(pre["struct_vectorizer"], "feature_names_")
    if idf is not None:
        arrays["word_idf"] = np.asarray(idf, dtype=np.float32)

    np.savez_compressed(out / "location_model.npz", **arrays)
    (out / "vocab.json").write_text(
        json.dumps({"words": words, "struct_features": [str(s) for s in struct_names]}),
        encoding="utf8")
    (out / "classes.json").write_text(json.dumps(classes, indent=2), encoding="utf8")

    meta_path = preproc_dir / "metadata__src-all.json"
    meta = json.loads(meta_path.read_text(encoding="utf8")) if meta_path.exists() else {}
    word_sel = pre.get("word_selection", {})
    (out / "config.json").write_text(json.dumps({
        "feature_extraction": {k: v for k, v in meta.items()
                               if not k.startswith("n_users")},
        "word_selection": {k: (v if isinstance(v, (int, float, str, bool, type(None)))
                               else str(v)) for k, v in word_sel.items()},
        "tier1_word_masking": bool(pre.get("mask_tier1_words", False)),
        "masking_notes": pre.get("masking_notes", ""),
        "deployment_thresholds": THRESHOLDS,
        "note": ("Deployed labels come from a weighted blend of the words and "
                 "struct models followed by these thresholds; see "
                 "code/label_location.py in the pipeline repository."),
    }, indent=2), encoding="utf8")

    for metric in sorted(trained.glob("*__metrics.json")):
        shutil.copy2(metric, out / "metrics" / metric.name)

    present = {p.name for p in preproc_dir.iterdir()} & EXCLUDE
    print(f"\n  excluded from release (training data): {sorted(present)}")
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"  wrote {out}  ({total / 2**20:.1f} MB)")


if __name__ == "__main__":
    main()
