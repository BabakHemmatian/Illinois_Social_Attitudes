# What to put where

Everything below is generated from `space/performance.py`, the single source of
truth for every metric this project publishes. No number is written by hand twice.

## 1. The Space — `BabakScrapes/isaac-classifiers`

Copy into `hf_spaces/isaac/` in the GitHub repo, then push the same files to the
Space repo:

| File | Change |
| --- | --- |
| `space/performance.py` | **new** — all metrics as data, plus renderers |
| `space/app.py` | adds a 4th tab, "Performance & citation" |
| `space/relevance.py` | adds a collapsed per-distinction performance accordion that updates with the dropdown |
| `space/moralization.py` | adds a collapsed performance accordion |
| `space/generalization.py` | adds a collapsed performance accordion |
| `space/README.md` | adds a Performance section, `license`/`models`/`tags` frontmatter, corrects the base-model descriptions |

`common.py` and `requirements.txt` are unchanged — keep the ones you have.

Regenerate the README table after any edit to `performance.py`:

```bash
cd hf_spaces/isaac && python performance.py --readme
```

## 2. The nine model repos

Each `model_cards/<repo>/README.md` goes to the root of
`BabakScrapes/<repo>` on the Hub. Either drag it into the web UI, or:

```bash
python - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
for repo in [
    "isaac-relevance-ability", "isaac-relevance-age", "isaac-relevance-race",
    "isaac-relevance-sexuality", "isaac-relevance-skin_tone",
    "isaac-relevance-weight", "isaac-moralization",
    "isaac-generalization", "isaac-generalization-segmentation",
]:
    api.upload_file(
        path_or_fileobj=f"model_cards/{repo}/README.md",
        path_in_repo="README.md",
        repo_id=f"BabakScrapes/{repo}",
        repo_type="model",
        commit_message="Add model card",
    )
PY
```

Then, per repo, **Settings → Gated → automatic (or manual)**. The
`extra_gated_*` frontmatter defines what requesters see; the toggle is what
actually turns gating on.

The two generalization cards are `cc-by-4.0` and ungated, matching the public
DiSCo repos they mirror — see the note in the summary.

To regenerate the cards:

```bash
python scripts/make_model_cards.py --out model_cards
```

## 3. Readable labels for the generalization models

`patched_configs/` holds `config.json` files with real `id2label` /`label2id`
mappings instead of `LABEL_0 … LABEL_17`. Upload them to
`ISAAC-corpus/isaac-generalization` and
`ISAAC-corpus/isaac-generalization-segmentation` (and the DiSCo repos, if you
want them to match).

Nothing in the ISAAC pipeline reads `config.id2label` — `label_generalization.py`
and the Space both use their own `labels2attrs` dict — so this is cosmetic for
you and a real improvement for anyone loading the weights directly.

To regenerate, or to apply in place:

```bash
python scripts/patch_generalization_configs.py --models-dir ./models --out ./patched_configs
python scripts/patch_generalization_configs.py --models-dir ./models --in-place --include-segmenter
```

**Confirm the segmenter mapping first.** Only tag `2` is established by the
decoder in `label_generalization.py` (it closes a clause). What distinguishes
tag `0` from tag `1` is not documented anywhere in the codebase, so the script
skips the segmenter unless you pass `--include-segmenter`.
