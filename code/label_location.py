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
import torch
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

### High-level parameters

# Confidence variables
CONF_MARGIN = 1.5            # log-score gap needed between top1 and top2
MIN_SAMPLES_FOR_CACHE = 10   # require at least this many raw items collected
UNKNOWN_LABEL = "__UNKNOWN__" # For users that cannot be geolocated with decent confidence:

# get run arguments from CLI
args = get_args()
type_ = args.type
years = parse_range(args.years)
if isinstance(years, int):
    years = [years]
group = args.group
batch_size = getattr(args, "batchsize", 4096)
max_items_per_author = getattr(args, "maxitems", 25)      # sample target per author. Has default.
max_files_to_scan = getattr(args, "maxfiles", 60)         # hard cap month-positions to scan. Has default.
max_radius = getattr(args, "maxradius", 30)               # ±30 months around target month. Has default.
include_hours = True

# path variables
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

### Model loading + inference

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

### Torch GPU batched scorer

def _torch_available() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
        return False

# Vectorized (batch x locations) scorer on GPU (CUDA) with a sparse correction path.
# NOTE: Score(user, loc) = log(prior(loc)) + const(loc) - lgamma(alpha_sum(loc) + n_user) + sum_{w in user} [lgamma(alpha_loc_w + c_w) - lgamma(alpha_loc_w)], 
# where alpha_loc_w = loc_param.alpha.get(w, alpha0)  (only if w in vocab_set)
class TorchBatchedScorer:
    def __init__(self, model):
        vocab, priors, loc_params, locations = _safe_model_fields(model)
        self.vocab_set = set(vocab)
        self.locations = list(locations)

        # Device
        self.device = torch.device("cuda") if _torch_available() else torch.device("cpu")

        # Build dense per-location tensors
        L = len(self.locations)
        const = torch.empty(L, dtype=torch.float32)
        alpha_sum = torch.empty(L, dtype=torch.float32)
        log_priors = torch.empty(L, dtype=torch.float32)

        # We assume a single alpha0 shared across locations. If missing, default to 0.01.
        alpha0_val = None

        # Feature -> list of (loc_index, alpha_value) overrides. We store only features in vocab_set.
        feat_locs: Dict[str, list] = {}
        feat_alphas: Dict[str, list] = {}

        for li, gh in enumerate(self.locations):
            lp = loc_params.get(gh)
            if lp is None:
                # Make it extremely unlikely
                const[li] = float("-inf")
                alpha_sum[li] = 1.0
                log_priors[li] = -100.0
                continue

            c = getattr(lp, "const", None)
            if c is None:
                # Older models may not have const; fall back (slower) by using dm_loglik CPU path
                raise ValueError("Model lacks loc_param.const; GPU batched scorer requires precomputed const.")
            const[li] = float(c)

            asum = getattr(lp, "alpha_sum", None)
            if asum is None:
                raise ValueError("Model lacks loc_param.alpha_sum; GPU batched scorer requires alpha_sum.")
            alpha_sum[li] = float(asum)

            p = float(priors.get(gh, 1e-30))
            log_priors[li] = math.log(p) if p > 0 else -1000.0

            a0 = getattr(lp, "alpha0", None)
            if alpha0_val is None and a0 is not None:
                alpha0_val = float(a0)

            alpha_dict = getattr(lp, "alpha", None)
            if not alpha_dict:
                continue

            # Build inverted index for overrides
            for feat, aval in alpha_dict.items():
                if feat not in self.vocab_set:
                    continue
                # If alpha dict redundantly stores alpha0, skip (saves memory + work)
                try:
                    aval_f = float(aval)
                except Exception:
                    continue
                if alpha0_val is not None and abs(aval_f - alpha0_val) < 1e-12:
                    continue
                feat_locs.setdefault(feat, []).append(li)
                feat_alphas.setdefault(feat, []).append(aval_f)

        if alpha0_val is None:
            alpha0_val = 0.01
        self.alpha0 = torch.tensor(alpha0_val, dtype=torch.float32, device=self.device)

        self.const = const.to(self.device)
        self.alpha_sum = alpha_sum.to(self.device)
        self.log_priors = log_priors.to(self.device)

        # Convert feature override lists to tensors on device
        self.feat_override = {}
        for feat, loc_list in feat_locs.items():
            loc_t = torch.tensor(loc_list, dtype=torch.long, device=self.device)
            a_t = torch.tensor(feat_alphas[feat], dtype=torch.float32, device=self.device)
            self.feat_override[feat] = (loc_t, a_t)

    # Convert list of count dicts into:
    # - per user: feature list and counts
    # - n_user tensor (sum counts over in-vocab features)
    # - base_term tensor (sum [lgamma(alpha0+c)-lgamma(alpha0)] over in-vocab features)
    # - feat -> (user_indices tensor, counts tensor) for features present in batch
    def _batch_prepare(self, batch_counts: List[Dict[str, int]]):
        B = len(batch_counts)
        n_user = torch.zeros(B, dtype=torch.float32, device=self.device)
        base_term = torch.zeros(B, dtype=torch.float32, device=self.device)

        feat_users: Dict[str, list] = {}
        feat_counts: Dict[str, list] = {}

        # Keep CPU loop to build sparse structures; GPU does the heavy math.
        for ui, counts in enumerate(batch_counts):
            if not counts:
                continue
            for feat, c in counts.items():
                if feat not in self.vocab_set:
                    continue
                ic = int(c)
                if ic <= 0:
                    continue
                n_user[ui] += ic
                # base DM contribution for this feature if alpha == alpha0 everywhere
                base_term[ui] += torch.lgamma(self.alpha0 + ic) - torch.lgamma(self.alpha0)

                if feat in self.feat_override:
                    feat_users.setdefault(feat, []).append(ui)
                    feat_counts.setdefault(feat, []).append(ic)

        feat_batch = {}
        for feat, ulist in feat_users.items():
            u_t = torch.tensor(ulist, dtype=torch.long, device=self.device)
            c_t = torch.tensor(feat_counts[feat], dtype=torch.float32, device=self.device)
            feat_batch[feat] = (u_t, c_t)
        return n_user, base_term, feat_batch

    @torch.no_grad()
    def predict_topk(self, batch_counts: List[Dict[str, int]], topk: int = 2):
        """
        Returns list of length B, each item: [(label, score), ...] (up to topk)
        """
        if not batch_counts:
            return []

        B = len(batch_counts)
        L = self.const.shape[0]

        n_user, base_term, feat_batch = self._batch_prepare(batch_counts)

        # Base scores: shape [B, L]
        # scores = log_priors + const - lgamma(alpha_sum + n_user) + base_term
        
        # Broadcasting
        scores = self.log_priors.unsqueeze(0) + self.const.unsqueeze(0)
        scores = scores - torch.lgamma(self.alpha_sum.unsqueeze(0) + n_user.unsqueeze(1))
        scores = scores + base_term.unsqueeze(1)

        # Apply sparse corrections for per-location feature overrides
        # NOTE: For each feature: for affected users, adjust scores at override locations by:(lgamma(alpha_loc + c) - lgamma(alpha_loc)) - (lgamma(alpha0 + c) - lgamma(alpha0))
        # where second term is the base already included in base_term. We compute correction per user and add for all override locations.
        THRESH_OUTER = 2_000_000  # outer product elements threshold for vectorized update
        for feat, (u_idx, c_vec) in feat_batch.items():
            loc_idx, a_loc = self.feat_override[feat]
            # base for these users
            base = torch.lgamma(self.alpha0 + c_vec) - torch.lgamma(self.alpha0)  # [U]
            U = u_idx.numel()
            K = loc_idx.numel()
            if U == 0 or K == 0:
                continue

            if U * K <= THRESH_OUTER:
                # Vectorized: compute [U,K] correction and scatter-add
                # corr = f(a_loc, c) - base
                corr = (torch.lgamma(a_loc.unsqueeze(0) + c_vec.unsqueeze(1)) - torch.lgamma(a_loc).unsqueeze(0)) - base.unsqueeze(1)
                # Build index tensors
                u_exp = u_idx.unsqueeze(1).expand(U, K).reshape(-1)
                l_exp = loc_idx.unsqueeze(0).expand(U, K).reshape(-1)
                v_exp = corr.reshape(-1)
                scores.index_put_((u_exp, l_exp), v_exp, accumulate=True)
            else:
                # Fallback: loop users for this feature (still vectorized over locations)
                for j in range(U):
                    ui = int(u_idx[j].item())
                    c = c_vec[j]
                    base_j = base[j]
                    corr_loc = (torch.lgamma(a_loc + c) - torch.lgamma(a_loc)) - base_j  # [K]
                    scores.index_put_((torch.full((K,), ui, device=self.device, dtype=torch.long), loc_idx), corr_loc, accumulate=True)

        # Top-k over locations
        k = min(topk, L)
        vals, idxs = torch.topk(scores, k=k, dim=1)

        # Convert to python output
        out = []
        vals_cpu = vals.detach().cpu().tolist()
        idxs_cpu = idxs.detach().cpu().tolist()
        for b in range(B):
            pairs = []
            for j in range(k):
                li = idxs_cpu[b][j]
                pairs.append((self.locations[li], float(vals_cpu[b][j])))
            out.append(pairs if pairs else [(UNKNOWN_LABEL, float("-inf"))])
        return out


# Predict top-k locations for a batch of users.
# NOTE: If `scorer` is provided and is a TorchBatchedScorer, scoring runs on GPU (CUDA) in a true batch x locations tensor. Otherwise falls back to the original CPU implementation.
def predict_batch(batch_counts: List[Dict[str, int]], model, topk: int = 1, scorer: Optional[object] = None) -> List[List[Tuple[str, float]]]:
    if scorer is not None:
        try:
            return scorer.predict_topk(batch_counts, topk=max(topk, 1))
        except Exception:
            # fallback to CPU if something goes wrong (e.g., model missing const)
            pass

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

### Location labeling pipeline

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

    # Optional GPU scorer (true batch x locations). Falls back to CPU if CUDA not available or model incompatible.
    scorer = None
    try:
        scorer = TorchBatchedScorer(model)
        log_report(report_file_path, f"[gpu] {stem}: using torch device={scorer.device} batch_scoring=yes")
    except Exception as e:
        scorer = None
        log_report(report_file_path, f"[gpu] {stem}: torch batch scorer unavailable; using CPU. reason={type(e).__name__}")

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
        top2 = predict_batch(batch_counts, model, topk=2, scorer=scorer)

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
    # If CUDA is available, avoid multiple processes competing for one GPU.
    if _torch_available():
        max_workers = 1
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
    log_report(report_file_path, f"Location labeling finished in {mins:.2f} minutes.")
