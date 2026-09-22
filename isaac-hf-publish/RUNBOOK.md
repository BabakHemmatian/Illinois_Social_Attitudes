# Runbook — publish ISAAC as a gated Hugging Face dataset

**Posture A** (soft gate): the HF dataset is gated for consent + an audit trail;
the UIUC `/data/` endpoint stays open.

## 0. One-time setup

1. **Pick the namespace.** The dataset lives at **`BabakScrapes/isaac-reddit`**, on
   the personal PRO account, *not* under the `ISAAC-corpus` org. This is a storage
   constraint, not a preference: HF gives a free organization only *best-effort*
   public storage and a hard 100 GB private quota, and the corpus is far larger
   than that. PRO covers 10 TB of public storage, and PRO is a personal plan that
   does not extend to org-owned repos. The **models stay under `ISAAC-corpus`**
   (they are small) and the Space stays at `BabakScrapes/isaac-classifiers` for
   ZeroGPU.
2. Create a **write token**: https://huggingface.co/settings/tokens
3. Install the client and the Xet transfer backend:
   ```bash
   pip install "huggingface_hub[hf_xet]"
   ```

## 1. Recompress the parquet (SNAPPY -> ZSTD)

The corpus as served from Globus is SNAPPY-compressed. The HF mirror ships
**ZSTD-9**, which is ~41% smaller for identical data — the same schema, the same
row-group boundaries, the same values. Local read speed is unchanged (measured:
within noise across full reads, column projections and row-group reads), but
every download and every `streaming=True` session moves ~40% fewer bytes.

```bash
python recompress_zstd.py --src /path/to/parquet-snappy --dst /path/to/parquet-zstd --workers 8
```

Write the output to a **different physical drive** from the source if you can —
the source lives on a USB HDD here, and reading and writing the same spindle
halves throughput. Every output file is verified row group by row group against
its source (`pyarrow.Table.equals`) before it is marked done, and progress is
journalled to `recompress_journal.jsonl`, so the run is resumable and re-running
it skips completed files.

## 2. Edit the dataset card

- The `extra_gated_prompt` / `extra_gated_fields` block is what **enables gating**
  and what requesters see. Adjust fields/wording as desired. (The Data Use Agreement
  text itself lives at the linked GitHub file.)
- The `configs:` blocks must match the on-disk filenames, which are
  `<category>/ALL_<YYYY>-<MM>.parquet` — the same convention the
  direct-download manifest uses.

## 3. Dry run (no upload)

```bash
python upload_to_hf.py --repo BabakScrapes/isaac-reddit --data-dir /path/to/parquet-zstd
```

Lists files and sizes, uploads nothing. Expect **1224 files across 6 categories**
(204 months each, 2007-01 to 2023-12).

## 4. Optional validation push (tiny sample)

Validate the card renders, configs load, and gating works, with a few months:
```bash
HF_TOKEN=hf_xxx python upload_to_hf.py --repo BabakScrapes/isaac-reddit-sample \
    --data-dir /path/to/parquet-zstd --sample 2 --execute
```
Check: dataset viewer renders; `load_dataset("BabakScrapes/isaac-reddit-sample","race")`
works after you accept the gate. Delete the sample repo when satisfied.

## 5. Full publish

```bash
HF_TOKEN=hf_xxx python upload_to_hf.py --repo BabakScrapes/isaac-reddit \
    --data-dir /path/to/parquet-zstd --execute
```
`upload_large_folder` is resumable — safe to re-run if the connection drops; it
re-scans the folder and skips what is already on the Hub.

Note this is a large transfer. The dataset **cannot be uploaded private**: 100 GB
is the private ceiling on every plan below Team, so the repo goes up public and
**gated** from the first byte. Gating is what protects the data; "public" here
means the card and the file list are visible, not the rows.

## 6. Confirm gating

In the repo **Settings → Gated**: ensure it's on and choose **automatic** or
**manual** approval. Verify a logged-out view shows the access request, and that
`load_dataset(...)` works only after acceptance + token.

## 7. Cross-link

- Add the HF dataset link to the Direct Download page and the GitHub README.
- The `isaac-data` loader and HF are complementary: HF for `datasets`/streaming +
  the gate; the loader/`/data/` for plain HTTP and column pushdown.

## Notes

- Publish **parquet only** to HF (native format, powers the viewer + streaming);
  CSV stays on the UIUC endpoint.
- The HF parquet are byte-for-byte *different* from the Globus copies because of
  the codec change, but decode to identical tables. Do not checksum one against
  the other.
- Re-run step 5 to update the dataset when the corpus changes (e.g. labeled release).
