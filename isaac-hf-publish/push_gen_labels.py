#!/usr/bin/env python3
"""Upload readable id2label/label2id to ISAAC-corpus/isaac-generalization.

The public repo serves LABEL_0 ... LABEL_17, so anyone loading the weights
directly gets meaningless class names. The replacement mapping is the key order
of `labels2attrs` in code/label_generalization.py, minus the `##` prefix.

DELIBERATELY EXCLUDES the segmenter. Its patched config uses the placeholder
names CLAUSE_INTERNAL_0 / CLAUSE_INTERNAL_1: only tag 2 (clause end) is
established by the decoder in label_generalization.py, and what distinguishes
tag 0 from tag 1 is undocumented. A guessed label name is worse than LABEL_n,
because it looks authoritative.

Re-verifies the mapping against the pipeline source and refuses to upload if
anything but id2label/label2id would change.
"""
import ast, json, pathlib, re, sys
from huggingface_hub import HfApi, hf_hub_download

REPO = "ISAAC-corpus/isaac-generalization"
HERE = pathlib.Path(__file__).parent
LOCAL = HERE / "isaac_hf" / "patched_configs" / "label_generalization" / "config.json"
SOURCE = HERE.parent / "code" / "label_generalization.py"


def main():
    new = json.loads(LOCAL.read_text(encoding="utf-8"))

    # 1. the mapping must match the pipeline's own label order
    src = SOURCE.read_text(encoding="utf-8")
    m = re.search(r"labels2attrs = (\{.*?\n\})", src, re.S)
    keys = [k.lstrip("#") for k in ast.literal_eval(m.group(1))]
    got = [new["id2label"][str(i)] for i in range(len(keys))]
    if keys != got:
        sys.exit(f"mapping does not match {SOURCE.name}: {keys} != {got}")
    if not all(new["label2id"][v] == int(k) for k, v in new["id2label"].items()):
        sys.exit("label2id is not the inverse of id2label")
    print(f"mapping verified against {SOURCE.name}: {len(keys)} labels")

    # 2. nothing but the labels may change
    live = json.loads(pathlib.Path(hf_hub_download(REPO, "config.json")).read_text(encoding="utf-8"))
    changed = {k for k in set(live) | set(new) if live.get(k) != new.get(k)}
    if changed - {"id2label", "label2id"}:
        sys.exit(f"refusing: would also change {sorted(changed - {'id2label', 'label2id'})}")
    print(f"diff vs live is exactly: {sorted(changed)}")

    HfApi().upload_file(
        path_or_fileobj=str(LOCAL), path_in_repo="config.json",
        repo_id=REPO, repo_type="model",
        commit_message="Readable class labels (id2label/label2id) from labels2attrs",
    )
    print(f"uploaded -> https://huggingface.co/{REPO}")
    print("segmenter NOT touched (placeholder labels, undocumented tags 0/1)")


if __name__ == "__main__":
    main()
