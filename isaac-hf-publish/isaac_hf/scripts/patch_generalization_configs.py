"""Give the two generalization models readable label names.

Both checkpoints currently ship `id2label` as `LABEL_0 ... LABEL_17` (classifier)
and `LABEL_0 ... LABEL_2` (segmenter). The real names exist only in the ordering
of `labels2attrs` inside `code/label_generalization.py`, so anyone who loads the
weights straight from the Hub gets meaningless integers back.

This script rewrites `id2label` / `label2id` in place (or into an output
directory) so the models are self-describing.

Notes
-----
* The classifier mapping is exact: it is the key order of `labels2attrs`, which
  is what `index2label` is built from at inference time.
* Names are written WITHOUT the leading `##` of the original annotation format.
  Nothing in the ISAAC pipeline reads `config.id2label` -- `label_generalization.py`
  and the Space both use their own `labels2attrs` dict -- so this change is
  cosmetic for the pipeline and purely a benefit for external users.
* The segmenter mapping is NOT documented anywhere in the codebase. The decoder
  in `label_generalization.py` establishes only that tag `2` closes a clause and
  that `0`/`1` are clause-internal; what distinguishes `0` from `1` is unknown.
  It is therefore behind an explicit `--include-segmenter` flag. Confirm
  SEGMENTER_LABELS below before using it.

Usage
-----
    # dry run, prints the diff
    python patch_generalization_configs.py --models-dir /path/to/models

    # write patched copies next to the originals for upload
    python patch_generalization_configs.py --models-dir /path/to/models \\
        --out ./patched

    # rewrite the checkpoints in place (a .bak is kept)
    python patch_generalization_configs.py --models-dir /path/to/models \\
        --in-place --include-segmenter
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

CLASSIFIER_SUBDIR = Path("label_generalization") / "label_generalization"
SEGMENTER_SUBDIR = Path("label_generalization") / "label_generalization_segmentation"

# Exact key order of `labels2attrs` in code/label_generalization.py.
CLASSIFIER_LABELS = [
    "BOUNDED EVENT (SPECIFIC)",
    "BOUNDED EVENT (GENERIC)",
    "UNBOUNDED EVENT (SPECIFIC)",
    "UNBOUNDED EVENT (GENERIC)",
    "BASIC STATE",
    "COERCED STATE (SPECIFIC)",
    "COERCED STATE (GENERIC)",
    "PERFECT COERCED STATE (SPECIFIC)",
    "PERFECT COERCED STATE (GENERIC)",
    "GENERIC SENTENCE (DYNAMIC)",
    "GENERIC SENTENCE (STATIC)",
    "GENERIC SENTENCE (HABITUAL)",
    "GENERALIZING SENTENCE (DYNAMIC)",
    "GENERALIZING SENTENCE (STATIVE)",
    "QUESTION",
    "IMPERATIVE",
    "NONSENSE",
    "OTHER",
]

# CONFIRM BEFORE USE. Only tag 2 is established by the decoder; the distinction
# between 0 and 1 is not documented in the pipeline.
SEGMENTER_LABELS = [
    "CLAUSE_INTERNAL_0",
    "CLAUSE_INTERNAL_1",
    "CLAUSE_END",
]


def patch(config_path: Path, labels: list[str]) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))

    n_existing = len(config.get("id2label", {})) or config.get("num_labels")
    if n_existing and n_existing != len(labels):
        raise SystemExit(
            f"{config_path}: checkpoint has {n_existing} labels, "
            f"mapping has {len(labels)}. Refusing to patch."
        )

    config["id2label"] = {str(i): name for i, name in enumerate(labels)}
    config["label2id"] = {name: i for i, name in enumerate(labels)}
    return config


def report(path: Path, config: dict) -> None:
    print(f"\n{path}")
    for index, name in sorted(config["id2label"].items(), key=lambda kv: int(kv[0])):
        print(f"  {index:>2}  {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models-dir", required=True, type=Path,
        help="the project's models/ directory",
    )
    parser.add_argument(
        "--out", type=Path,
        help="write patched config.json files under this directory instead of "
             "printing a dry run",
    )
    parser.add_argument(
        "--in-place", action="store_true",
        help="rewrite the checkpoints' config.json (keeps a .bak)",
    )
    parser.add_argument(
        "--include-segmenter", action="store_true",
        help="also patch the segmenter -- confirm SEGMENTER_LABELS first",
    )
    args = parser.parse_args()

    targets = [(CLASSIFIER_SUBDIR, CLASSIFIER_LABELS)]
    if args.include_segmenter:
        targets.append((SEGMENTER_SUBDIR, SEGMENTER_LABELS))
    else:
        print(
            "note: segmenter skipped. Its 0/1 tag meanings are undocumented; "
            "confirm SEGMENTER_LABELS, then pass --include-segmenter."
        )

    for subdir, labels in targets:
        source = args.models_dir / subdir / "config.json"
        if not source.exists():
            raise SystemExit(f"missing: {source}")

        config = patch(source, labels)
        report(source, config)

        payload = json.dumps(config, indent=2, ensure_ascii=False) + "\n"

        if args.in_place:
            shutil.copy2(source, source.with_suffix(".json.bak"))
            source.write_text(payload, encoding="utf-8")
            print(f"  -> rewritten in place (backup at {source.name}.bak)")
        elif args.out:
            destination = args.out / subdir.name / "config.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(payload, encoding="utf-8")
            print(f"  -> wrote {destination}")
        else:
            print("  -> dry run, nothing written")


if __name__ == "__main__":
    main()
