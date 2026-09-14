#!/usr/bin/env python3
"""Publish the ISAAC parquet files to a (gated) Hugging Face dataset repo.

DRY-RUN by default — it only reports what would be uploaded. Pass --execute
(with an HF token) to actually create the repo and push. Gating is enabled by
the `extra_gated_*` fields in dataset_card.md (uploaded as the repo README.md);
set auto- vs manual-approval in the repo's Settings afterward.

Requires: pip install huggingface_hub
Token: --token or $HF_TOKEN (a write token from https://huggingface.co/settings/tokens)
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

CATEGORIES = ["ability", "age", "race", "sexuality", "skin_tone", "weight"]


def enumerate_files(data_dir: str, sample: int = 0):
    out = []
    for c in CATEGORIES:
        pq = sorted(glob.glob(str(Path(data_dir) / c / "RC_*.parquet")))
        out.append((c, pq[:sample] if sample else pq))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="HF dataset repo id, e.g. ISAAC-corpus/ISAAC")
    ap.add_argument("--data-dir", default="/home/ubuntu/bulk/data", help="local dir with <category>/RC_*.parquet")
    ap.add_argument("--card", default=str(Path(__file__).parent / "dataset_card.md"),
                    help="dataset card uploaded as README.md")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF write token (or $HF_TOKEN)")
    ap.add_argument("--sample", type=int, default=0,
                    help="upload only the first N months per category (for a validation run)")
    ap.add_argument("--private", action="store_true", help="create repo private (default: public + gated)")
    ap.add_argument("--execute", action="store_true", help="actually upload (omit for a dry run)")
    args = ap.parse_args()

    groups = enumerate_files(args.data_dir, args.sample)
    files = [f for _, fs in groups for f in fs]
    if not files:
        sys.exit(f"No parquet found under {args.data_dir}/<category>/RC_*.parquet")
    total = sum(os.path.getsize(f) for f in files)

    print(f"repo        : {args.repo}  ({'private' if args.private else 'public + gated'})")
    print(f"data-dir    : {args.data_dir}")
    print(f"card        : {args.card}  ({'exists' if Path(args.card).exists() else 'MISSING'})")
    print(f"selection   : {len(files)} parquet, {total/1e9:.1f} GB"
          + (f"  (sample={args.sample}/category)" if args.sample else ""))
    for c, fs in groups:
        print(f"   {c:10s} {len(fs):4d} files")

    if not args.execute:
        print("\nDRY RUN — nothing uploaded. Re-run with --execute and a token to publish.")
        print("Example: HF_TOKEN=hf_xxx python upload_to_hf.py --repo ISAAC-corpus/reddit --execute")
        return 0

    if not Path(args.card).exists():
        sys.exit(f"Dataset card not found: {args.card}")
    if not args.token:
        sys.exit("No HF token. Pass --token or set $HF_TOKEN.")

    from huggingface_hub import HfApi
    api = HfApi(token=args.token)
    print(f"\nCreating dataset repo {args.repo} (if needed)...")
    api.create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)

    print("Uploading dataset card (README.md)...")
    api.upload_file(path_or_fileobj=args.card, path_in_repo="README.md",
                    repo_id=args.repo, repo_type="dataset")

    if args.sample:
        print(f"Uploading {len(files)} sampled parquet files...")
        for c, fs in groups:
            for f in fs:
                api.upload_file(path_or_fileobj=f, path_in_repo=f"{c}/{Path(f).name}",
                                repo_id=args.repo, repo_type="dataset")
    else:
        print("Uploading all parquet (resumable upload_large_folder)...")
        api.upload_large_folder(
            repo_id=args.repo, repo_type="dataset", folder_path=args.data_dir,
            allow_patterns=[f"{c}/RC_*.parquet" for c in CATEGORIES],
        )

    print(f"\nDone: https://huggingface.co/datasets/{args.repo}")
    print("Next: in the repo Settings, confirm 'Gated' is on and pick auto/manual approval.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
