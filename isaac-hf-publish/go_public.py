#!/usr/bin/env python3
"""Take the nine ISAAC model repos and the Space public.

Deliberately excludes the location model: it is gated behind a Model Use
Agreement still with Legal Counsel, and nothing location-related may be
published until that is signed off. The script refuses to touch any repo whose
id mentions "location", so a later edit to REPOS cannot publish it by accident.

Fails closed: every model repo is re-checked for Reddit text and stray training
artifacts immediately before it is flipped. A repo that fails is skipped, not
published.
"""
import sys
from huggingface_hub import HfApi

MODELS = [
    "ISAAC-corpus/isaac-relevance-ability",
    "ISAAC-corpus/isaac-relevance-age",
    "ISAAC-corpus/isaac-relevance-race",
    "ISAAC-corpus/isaac-relevance-sexuality",
    "ISAAC-corpus/isaac-relevance-skin_tone",
    "ISAAC-corpus/isaac-relevance-weight",
    "ISAAC-corpus/isaac-moralization",
    "ISAAC-corpus/isaac-generalization",
    "ISAAC-corpus/isaac-generalization-segmentation",
]
SPACE = "BabakScrapes/isaac-classifiers"

# Never publish: verbatim Reddit text (DUA-governed, outside the CC-BY regime),
# training inputs, or the duplicate pickled weights.
BANNED = ("results", ".csv", ".tsv", "users__", "X_words",
          "labels_and_splits", "pytorch_model.bin", ".pkl", ".npy", ".npz")


def main():
    api = HfApi()
    ok = fail = 0

    for rid in MODELS:
        if "location" in rid.lower():
            print(f"REFUSING {rid}: location model is MUA-gated, pending Counsel")
            fail += 1
            continue
        files = api.list_repo_files(rid)
        bad = [f for f in files if any(b in f for b in BANNED)]
        if bad:
            print(f"SKIP  {rid}: would publish {bad}")
            fail += 1
            continue
        if "README.md" not in files:
            print(f"SKIP  {rid}: no model card")
            fail += 1
            continue
        api.update_repo_settings(rid, repo_type="model", private=False)
        print(f"PUBLIC {rid}  ({len(files)} files)")
        ok += 1

    api.update_repo_settings(SPACE, repo_type="space", private=False)
    print(f"PUBLIC {SPACE}")

    print(f"\n{ok}/{len(MODELS)} model repos public, Space public, {fail} skipped")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
