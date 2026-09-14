# Tier-2 runbook — publish ISAAC as a gated Hugging Face dataset

**Posture A** (soft gate): the HF dataset is gated for consent + an audit trail;
the UIUC `/data/` endpoint stays open. **Publish when the labeled data is ready.**

## 0. One-time setup
1. **Create an org** (recommended over the personal account) and decide the name
   with your UIUC co-author, e.g. `ISAAC-corpus`. Add the co-author as an admin.
   Personal accounts also work: just use `BabakScrapes/ISAAC` as the repo id.
2. Create a **write token**: https://huggingface.co/settings/tokens
3. Install the client (e.g. in the loader venv):
   ```bash
   pip install huggingface_hub
   ```

## 1. Edit the dataset card
- The `extra_gated_prompt` / `extra_gated_fields` block is what **enables gating**
  and what requesters see. Adjust fields/wording as desired. (The Data Use Agreement
  text itself lives at the linked GitHub file.)

## 2. Dry run (no upload)
```bash
python upload_to_hf.py --repo ISAAC-corpus/reddit          # lists files + sizes, uploads nothing
```

## 3. Optional validation push (tiny sample)
Validate the card renders, configs load, and gating works, with a few months:
```bash
HF_TOKEN=hf_xxx python upload_to_hf.py --repo ISAAC-corpus/reddit-sample --sample 2 --execute
```
Check: dataset viewer renders; `load_dataset("ISAAC-corpus/reddit-sample","race")` works after
you accept the gate. Delete the sample repo when satisfied.

## 4. Full publish (when labeled data is final)
Point `--data-dir` at the final parquet, then:
```bash
HF_TOKEN=hf_xxx python upload_to_hf.py --repo ISAAC-corpus/reddit --execute
```
`upload_large_folder` is resumable — safe to re-run if the connection drops.
Note: this is a large transfer (current parquet ≈ 272 GB). HF hosts large
datasets; if you hit a storage limit, request a quota increase.

## 5. Confirm gating
In the repo **Settings → Gated**: ensure it's on and choose **automatic** or
**manual** approval. Verify a logged-out view shows the access request, and that
`load_dataset(...)` works only after acceptance + token.

## 6. Cross-link
- Add the HF dataset link to the Direct Download page and the GitHub README.
- The `isaac-data` loader and HF are complementary: HF for `datasets`/streaming +
  the gate; the loader/`/data/` for plain HTTP and column pushdown.

## Notes
- Publish **parquet only** to HF (native format, powers the viewer + streaming);
  CSV stays on the UIUC endpoint.
- Re-run step 4 to update the dataset when the corpus changes (e.g. labeled release).
