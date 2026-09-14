"""Fetch every model the ISAAC pipeline needs into ./models.

The weights are too large to keep in this repository, so they live on
HuggingFace (ISAAC's own models) and at their upstream sources (the third-party
emotion models and fastText's language identifier). This script puts all of them
in the directory layout the pipeline resources expect.

    pip install huggingface_hub
    python get_models.py

About 9 GB in total. Downloads resume, so it is safe to re-run after an
interruption; files already present are not fetched again.

The location model is gated. Before running, accept the ISAAC Model Use
Agreement at https://huggingface.co/ISAAC-corpus/isaac-location, then
authenticate once with a token from https://huggingface.co/settings/tokens:

    huggingface-cli login

Pass --skip-location to fetch everything else if you do not need it.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

# Destination is relative to this file, so the script works from any cwd.
MODELS_DIR = Path(__file__).resolve().parent / "models"

RELEVANCE_GROUPS = ["ability", "age", "race", "sexuality", "skin_tone", "weight"]

# HuggingFace repo id -> path under models/
ISAAC_MODELS = {
    **{f"ISAAC-corpus/isaac-relevance-{g}": f"filter_relevance_{g}"
       for g in RELEVANCE_GROUPS},
    "ISAAC-corpus/isaac-moralization": "label_moralization",
    # label_generalization.py loads these two from nested subdirectories.
    "ISAAC-corpus/isaac-generalization":
        "label_generalization/label_generalization",
    "ISAAC-corpus/isaac-generalization-segmentation":
        "label_generalization/label_generalization_segmentation",
}

LOCATION_MODEL = {"ISAAC-corpus/isaac-location": "label_location"}

# Off-the-shelf emotion models, fetched from their authors' own repositories.
THIRD_PARTY = {
    "j-hartmann/emotion-english-distilroberta-base": "label_emotion_1",
    "SamLowe/roberta-base-go_emotions": "label_emotion_2",
    "tae898/emoberta-base": "label_emotion_3",
}

# fastText language identification; not distributed through HuggingFace.
FASTTEXT_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
FASTTEXT_DEST = "filter_language.bin"


def fetch_fasttext() -> None:
    dest = MODELS_DIR / FASTTEXT_DEST
    if dest.exists():
        print(f"  [skip] models/{FASTTEXT_DEST} already present")
        return
    print(f"  fastText lid.176.bin  ->  models/{FASTTEXT_DEST}  (~125 MB)")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".bin.part")
    urllib.request.urlretrieve(FASTTEXT_URL, tmp)
    tmp.replace(dest)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-location", action="store_true",
                    help="skip the gated location model")
    ap.add_argument("--only", choices=["isaac", "third-party", "fasttext"],
                    help="fetch only one group")
    args = ap.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub is required:  pip install huggingface_hub",
              file=sys.stderr)
        return 1

    repos: dict[str, str] = {}
    if args.only in (None, "isaac"):
        repos.update(ISAAC_MODELS)
        if not args.skip_location:
            repos.update(LOCATION_MODEL)
    if args.only in (None, "third-party"):
        repos.update(THIRD_PARTY)

    failed = []
    for repo, dest in repos.items():
        target = MODELS_DIR / dest
        print(f"  {repo}  ->  models/{dest}")
        try:
            snapshot_download(repo_id=repo, local_dir=target)
        except Exception as exc:  # noqa: BLE001 - report and continue
            failed.append((repo, exc))
            print(f"    FAILED: {exc}", file=sys.stderr)

    if args.only in (None, "fasttext"):
        fetch_fasttext()

    if failed:
        print("\nSome downloads failed:", file=sys.stderr)
        for repo, exc in failed:
            print(f"  {repo}: {exc}", file=sys.stderr)
        if any("gated" in str(e).lower() or "401" in str(e) or "403" in str(e)
               for _, e in failed):
            print("\nGated repositories need the Model Use Agreement accepted on "
                  "the model page, then `huggingface-cli login`.", file=sys.stderr)
        return 1

    print(f"\nDone. Models are in {MODELS_DIR}")
    print("Override the location with the ISAAC_MODELS_DIR environment variable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
