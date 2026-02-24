# import necessary functions and packages
import os
import csv
import re
import time
import math
import pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional
from cli import get_args, dir_path
from utils import (
    check_reqd_files,
    parse_range,
    log_report,
    find_raw_month_files,
    build_author_feature_map_from_raw_zst_with_seen,
    init_location_cache,
    cache_get_locations,
    cache_put_locations,
)

## Confidence variables
CONF_MARGIN = 1.5            # log-score gap needed between top1 and top2
MIN_SAMPLES_FOR_CACHE = 10   # require at least this many raw items collected
UNKNOWN_LABEL = "__UNKNOWN__" # For users that cannot be geolocated with decent confidence:

## get run arguments from CLI
args = get_args()
type_ = args.type
years = parse_range(args.years)
if isinstance(years, int):
    years = [years]
group = args.group
batch_size = getattr(args, "batchsize", 512)
max_items_per_author = getattr(args, "maxitems", 25)   # sample target per author. Has default.
max_files_to_scan = getattr(args, "maxfiles", 60)         # hard cap month-positions to scan. Has default.
max_radius = getattr(args, "maxradius", 30)               # ±30 months around target month. Has default.
include_hours = True

## path variables
CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
DATA_DIR = PROJECT_ROOT

MODELS_DIR = PROJECT_ROOT / "models"
RAW_DIR = DATA_DIR / "data_reddit_raw" / type_
emotion_labeled_path = DATA_DIR / "data_reddit_curated" / group / type_ / "labeled_emotion"
file_list = check_reqd_files(years, emotion_labeled_path, type_)

output_path = DATA_DIR / "data_reddit_curated" / group / type_ / "labeled_location"
output_path.mkdir(parents=True, exist_ok=True)

# cache inside output folder (shared across months and runs)
CACHE_DB_PATH = str(output_path / "author_location_cache.sqlite")
init_location_cache(CACHE_DB_PATH)

report_file_path = os.path.join(dir_path, "report_label_location.csv")

MODEL_PATH = os.path.join(MODELS_DIR, "location_model.pkl")
# NOTE: model is loaded inside each worker process to avoid large inter-process pickling.

processed_stems = {Path(f).stem for f in os.listdir(output_path) if f.endswith(".csv")}

## Model loading + inference

# Load the pickled location model
def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

# Extract (vocab, priors, loc_params, locations) from a saved model object.
def _safe_model_fields(model):
    if isinstance(model, dict):
        vocab = model.get("vocab") or model.get("Vocab") or model.get("features")
        priors = model.get("priors") or model.get("train_priors") or {}
        loc_params = model.get("loc_params") or model.get("params") or {}
        locations = model.get("locations") or list(priors.keys())
    else:
        vocab = getattr(model, "vocab", None)
        priors = getattr(model, "priors", {}) or {}
        loc_params = getattr(model, "loc_params", {}) or {}
        locations = getattr(model, "locations", None) or list(priors.keys())

    if vocab is None or loc_params is None:
        raise ValueError("Loaded model missing required fields: vocab/loc_params.")
    if not priors:
        # allow priors missing; treat as uniform-ish
        priors = {k: 1.0 for k in (locations or loc_params.keys())}
        s = sum(priors.values()) or 1.0
        priors = {k: v / s for k, v in priors.items()}
    if not locations:
        locations = list(priors.keys()) if priors else list(loc_params.keys())
    return vocab, priors, loc_params, locations

# Dirichlet-multinomial log-likelihood for one user given one location's params.
def dm_loglik(counts: Dict[str, int], loc_param, vocab_set: set) -> float:
    n = 0 # total in-vocab tokens
    for w, c in counts.items():
        if w in vocab_set:
            n += int(c)
    if n == 0:
        return 0.0

    ll = math.lgamma(loc_param.alpha_sum) - math.lgamma(loc_param.alpha_sum + n)
    alpha_dict = loc_param.alpha
    for w, c in counts.items():
        if w not in vocab_set:
            continue
        a = alpha_dict.get(w, loc_param.alpha0)
        ll += math.lgamma(a + int(c)) - math.lgamma(a)
    return ll

# Batched prediction. Returns top-k [(label, log_score)] per sample.
def predict_batch(batch_counts: List[Dict[str, int]], model, topk: int = 1) -> List[List[Tuple[str, float]]]:
    vocab, priors, loc_params, locations = _safe_model_fields(model)
    vocab_set = set(vocab)

    out: List[List[Tuple[str, float]]] = []
    for counts in batch_counts:
        scored: List[Tuple[str, float]] = []
        for gh in locations:
            lp = loc_params.get(gh)
            if lp is None:
                continue
            ll = dm_loglik(counts, lp, vocab_set)
            prior = priors.get(gh, 1e-30)
            scored.append((gh, math.log(prior) + ll))
        scored.sort(key=lambda x: x[1], reverse=True)
        out.append(scored[:topk] if scored else [(UNKNOWN_LABEL, float("-inf"))])
    return out

# User activity scanning
# NOTE: Return [(year, 'MM'), ...] in an expanding spiral around (year, month).
def month_spiral(year: int, month: int, max_files_to_scan: int = 60, max_radius: int = 30) -> List[Tuple[int, str]]:
    center = year * 12 + (month - 1)
    offsets = [0]
    for r in range(1, max_radius + 1):
        offsets.append(-r)
        offsets.append(r)

    out: List[Tuple[int, str]] = []
    for off in offsets:
        m = center + off
        y = m // 12
        mo = m % 12 + 1
        out.append((y, f"{mo:02d}"))
        if len(out) >= max_files_to_scan:
            break
    return out

def _extract_year_month_from_name(path: str) -> Optional[Tuple[int, int]]:
    m = re.search(r"(\d{4})-(\d{2})", path)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))

## Location labeling pipeline

# Monthly processing function
# 1. Reads a curated file to set authors, uses a persistent cache to quickly label authors
# 2. If author not found in cache, scans raw monthly files in a spiral window around the current month.
# 3. Predicts location labels in batches and updates cache
# 4. Writes output CSV with added location column.
# NOTE: Returns: (stem, rows_written, n_authors_total, n_authors_scanned_raw)
# NOTE: Does NOT follow where the previous attempt left off. Delete incomplete output files before running this again. 
def label_location_month(curated_csv_path: str) -> Tuple[str, int, int, int]:
    curated_csv_path = str(curated_csv_path)
    stem = Path(curated_csv_path).stem

    # If output file already exists, skip (but keep cache file for later months).
    if stem in processed_stems:
        log_report(report_file_path, f"[skip] output already exists for {stem}")
        return (stem, 0, 0, 0)

    ym = _extract_year_month_from_name(curated_csv_path)
    if ym is None:
        log_report(report_file_path, f"[warn] could not parse year-month from {curated_csv_path}; skipping")
        return (stem, 0, 0, 0)
    year, month_int = ym

    start = time.time()

    # Pass 1: read curated file, gather authors, store rows
    rows: List[List[str]] = []
    authors: set[str] = set()

    with open(curated_csv_path, "r", encoding="utf-8-sig", errors="ignore") as f:
        reader = csv.reader((line.replace("\x00", "") for line in f))
        try:
            header = next(reader)
        except StopIteration:
            return (stem, 0, 0, 0)

        author_idx = header.index("author") if "author" in header else 3

        for r in reader:
            if not r:
                continue
            rows.append(r)
            if len(r) > author_idx:
                a = r[author_idx].strip()
                if a and a != "[deleted]":
                    authors.add(a)

    if not authors:
        log_report(report_file_path, f"[warn] {stem}: no authors found")
        return (stem, 0, 0, 0)

    # Pass 2: use cache for already-known authors
    cached = cache_get_locations(CACHE_DB_PATH, authors)
    preds: Dict[str, str] = {a: loc for a, loc in cached.items() if loc}

    remaining_authors = set(a for a in authors if a not in preds)
    n_cached = len(preds)
    n_total = len(authors)

    # If everyone is already cached, we can write immediately
    if not remaining_authors:
        out_file = output_path / f"{stem}.csv"
        with open(out_file, "w", encoding="utf-8", newline="", errors="ignore") as fo:
            writer = csv.writer(fo)
            writer.writerow(header + ["location"])
            author_idx = header.index("author") if "author" in header else 3
            for r in rows:
                a = r[author_idx].strip() if len(r) > author_idx else ""
                writer.writerow(r + [preds.get(a, UNKNOWN_LABEL)])
        elapsed = (time.time() - start) / 60
        log_report(report_file_path, f"[done-cache] {stem}: rows={len(rows):,} authors={n_total:,} cached={n_cached:,} minutes={elapsed:.2f}")
        return (stem, len(rows), n_total, 0)

    # Pass 3: scan raw files in spiral window to collect up to N samples/author
    scan_months = month_spiral(year, month_int, max_files_to_scan=max_files_to_scan, max_radius=max_radius)

    raw_files: List[str] = []
    months_with_files = 0
    for y, mstr in scan_months:
        files = find_raw_month_files(RAW_DIR, type_, y, mstr)
        if files:
            raw_files.extend(files)
            months_with_files += 1

    if not raw_files:
        # No raw files found anywhere in the scan window -> label remaining as UNKNOWN and write
        for a in remaining_authors:
            preds[a] = UNKNOWN_LABEL

        out_file = output_path / f"{stem}.csv"
        with open(out_file, "w", encoding="utf-8", newline="", errors="ignore") as fo:
            writer = csv.writer(fo)
            writer.writerow(header + ["location"])
            author_idx = header.index("author") if "author" in header else 3
            for r in rows:
                a = r[author_idx].strip() if len(r) > author_idx else ""
                writer.writerow(r + [preds.get(a, UNKNOWN_LABEL)])
        elapsed = (time.time() - start) / 60
        log_report(report_file_path, f"[warn] {stem}: no raw files in scan window; wrote UNKNOWN for {len(remaining_authors):,}. minutes={elapsed:.2f}")
        return (stem, len(rows), n_total, len(remaining_authors))

    log_report(
        report_file_path,
        f"[start] {stem}: authors={n_total:,} cached={n_cached:,} need_raw={len(remaining_authors):,} "
        f"scan_months={len(scan_months)} months_with_files={months_with_files} raw_files={len(raw_files)} "
        f"samples_per_author={max_items_per_author} max_files_to_scan={max_files_to_scan} max_radius={max_radius}",
    )

    # Load model inside worker
    model = load_model(MODEL_PATH)

    author_to_counts, author_seen = build_author_feature_map_from_raw_zst_with_seen(
        raw_files=raw_files,
        target_authors=remaining_authors,
        type_=type_,
        max_items_per_author=max_items_per_author,
        include_hours=include_hours,
    )

    # Pass 4: predict remaining authors in batches
    remaining_list = sorted(remaining_authors)
    to_cache: Dict[str, str] = {}
    n_cache_confident = 0
    n_cache_skipped_lowconf = 0
    n_cache_skipped_lowsamples = 0

    for i in range(0, len(remaining_list), batch_size):
        chunk = remaining_list[i : i + batch_size]
        batch_counts = [author_to_counts.get(a, {}) for a in chunk]

        # Get top-2 to compute a confidence margin (log-score gap)
        top2 = predict_batch(batch_counts, model, topk=2)

        for a, scores in zip(chunk, top2):
            # predicted label
            if scores and scores[0] and scores[0][0]:
                preds[a] = scores[0][0]
            else:
                preds[a] = UNKNOWN_LABEL

            # confidence + sample gating for caching (Option A)
            seen = author_seen.get(a, 0)
            if seen < MIN_SAMPLES_FOR_CACHE:
                n_cache_skipped_lowsamples += 1
                continue

            # margin: top1 - top2 (if no top2, treat as infinite margin)
            if scores and len(scores) >= 2 and scores[1] and scores[1][0]:
                margin = float(scores[0][1]) - float(scores[1][1])
            else:
                margin = float("inf")

            if margin >= CONF_MARGIN:
                to_cache[a] = preds[a]
                n_cache_confident += 1
            else:
                n_cache_skipped_lowconf += 1

    # Update cache ONLY for confident predictions
    if to_cache:
        cache_put_locations(CACHE_DB_PATH, to_cache)

    log_report(
        report_file_path,
        f"[cache] {stem}: newly_labeled={len(remaining_authors):,} cached_confident={n_cache_confident:,} "
        f"skipped_lowconf={n_cache_skipped_lowconf:,} skipped_lowsamples={n_cache_skipped_lowsamples:,} "
        f"conf_margin>={CONF_MARGIN} min_samples={MIN_SAMPLES_FOR_CACHE}",
    )

    # Pass 5: write output file
    out_file = output_path / f"{stem}.csv"
    with open(out_file, "w", encoding="utf-8", newline="", errors="ignore") as fo:
        writer = csv.writer(fo)
        writer.writerow(header + ["location"])
        author_idx = header.index("author") if "author" in header else 3
        for r in rows:
            a = r[author_idx].strip() if len(r) > author_idx else ""
            writer.writerow(r + [preds.get(a, UNKNOWN_LABEL)])

    elapsed = (time.time() - start) / 60
    log_report(
        report_file_path,
        f"[done] {stem}: rows={len(rows):,} authors={n_total:,} cached={n_cached:,} scanned_raw={len(remaining_authors):,} "
        f"covered={len(author_to_counts):,} minutes={elapsed:.2f}",
    )
    return (stem, len(rows), n_total, len(remaining_authors))

def label_location_parallel():
    array_idx = getattr(args, "array", None)
    if array_idx is not None:
        try:
            idx = int(array_idx)
            label_location_month(file_list[idx])
            return
        except Exception:
            log_report(report_file_path, f"[warn] invalid --array '{array_idx}', running full set")

    max_workers = min(4, os.cpu_count() or 1)
    log_report(report_file_path, f"Using {max_workers} processes for parallel month processing.")
    total_rows = 0
    total_authors = 0
    total_raw = 0

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(label_location_month, f) for f in file_list]
        for fut in as_completed(futs):
            try:
                _, rows, n_auth, n_raw = fut.result()
                total_rows += rows
                total_authors += n_auth
                total_raw += n_raw
            except Exception as e:
                log_report(report_file_path, f"[error] month failed: {e}")

    log_report(report_file_path, f"[summary] total rows written: {total_rows:,} total authors: {total_authors:,} raw-scanned authors: {total_raw:,}")

if __name__ == "__main__":
    overall = time.time()
    try:
        label_location_parallel()
    except Exception as e:
        log_report(report_file_path, f"Fatal error during location labeling: {e}")
    mins = (time.time() - overall) / 60
    log_report(report_file_path, f"Location labeling finished in {mins:.2f} minutes")
