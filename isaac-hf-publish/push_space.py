#!/usr/bin/env python3
"""Push the corrected ISAAC Space to BabakScrapes/isaac-classifiers.

One atomic commit: app.py must never be live without the performance.py it
imports. requirements.txt is deliberately NOT sent — the deployed copy is
current and every dependency the new code needs is already listed there.
"""
import pathlib
from huggingface_hub import HfApi, CommitOperationAdd

SPACE = "BabakScrapes/isaac-classifiers"
LOCAL = pathlib.Path(__file__).parent / "isaac_hf" / "space"
FILES = ["performance.py",      # new: metrics + citation, all stdlib
         "app.py",              # adds the 4th tab
         "generalization.py",   # per-model performance accordion
         "moralization.py",
         "relevance.py",
         "common.py",           # REPO_ROOT fix: Spaces now deploy flat at /app
         "README.md"]           # Performance section, frontmatter, CC-BY terms

def main():
    api = HfApi()
    ops = []
    for f in FILES:
        p = LOCAL / f
        if not p.exists():
            raise SystemExit(f"missing local file: {p}")
        print(f"  staging {f:22s} {p.stat().st_size:>7d} bytes")
        ops.append(CommitOperationAdd(path_in_repo=f, path_or_fileobj=str(p)))
    info = api.create_commit(
        repo_id=SPACE, repo_type="space", operations=ops,
        commit_message="Add Performance & citation tab; single ISAAC paper citation",
    )
    print("\ncommitted:", getattr(info, "commit_url", info))
    print("Watch the rebuild at https://huggingface.co/spaces/" + SPACE)

if __name__ == "__main__":
    main()
