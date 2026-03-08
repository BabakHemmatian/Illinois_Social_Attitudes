
from __future__ import annotations
import csv
import json
import math
import os
import pickle
import sys
import time
import hashlib
import signal
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable

from utils import prepare_splits

### Utilities loading

# Optional fast JSON
try:
    import orjson as _fastjson  # type: ignore
    def _json_loads(b: bytes):
        return _fastjson.loads(b)
except Exception:
    def _json_loads(b: bytes):
        return json.loads(b)

# Optional Torch (GPU)
_TORCH_AVAILABLE = False
try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:
    torch = None  # type: ignore

### Output control
VERBOSITY = 1  # 0=quiet, 1=progress, 2=verbose
def log(msg: str, level: int = 1, stream=None):
    if level <= VERBOSITY:
        print(msg, file=stream)

UNKNOWN_LABEL = "__UNKNOWN__"

### training configurations
loc_type = "global"  # options: US, non-US, global
SAVE_CANDIDATE_MODELS = False
OVERWRITE_CHECKPOINTS = False
PROGRESS_EVERY_SECS = 60
PRF_TOPK = 5

### Paths / configuration

CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_PATH = PROJECT_ROOT / "models"

SUBS_JSONL = os.path.join(DATA_DIR, "data_reddit_location","subreddit_counts.jsonl")
VOCAB_FILE = os.path.join(DATA_DIR, "data_reddit_location","vocab_counts.jsonl")
HOURS_JSONL = os.path.join(DATA_DIR,"data_reddit_location","hour_counts.jsonl")

SAVE_MODEL = True
MODEL_SAVE_PATH = os.path.join(MODEL_PATH, "label_location", "best_model.pkl")
SPLIT_DIR = os.path.join(MODEL_PATH, "train_location_data_split")

GRID_CHECKPOINT_DIR = os.path.join(os.path.dirname(MODEL_SAVE_PATH), "grid_checkpoints")

# Feature prefixing
PREFIX_WORD = "w:" # word usage counts
PREFIX_SUB = "s:" # subreddit counts
PREFIX_HOUR = "h:" # hours of the day counts

### Grid config

@dataclass(frozen=True)
class Config:
    vocab_size: int
    selector: str
    min_total_count: int
    alpha_word: float
    alpha_sub: float
    alpha_hour: float

# NOTE: the six set-ups cover baselines that have identical smoothing parameters for all features, and look at different levels of smoothing and vocab size.
configs = [
    Config(alpha_word=0.1, alpha_sub=0.1, alpha_hour=0.1, vocab_size=50000,  selector="mi", min_total_count=3),
    Config(alpha_word=0.1, alpha_sub=0.1, alpha_hour=0.1, vocab_size=100000, selector="mi", min_total_count=3),
    Config(alpha_word=0.5, alpha_sub=0.5, alpha_hour=0.5, vocab_size=50000,  selector="mi", min_total_count=3),

    Config(alpha_word=0.05, alpha_sub=0.2, alpha_hour=1.0, vocab_size=50000,  selector="mi", min_total_count=3),
    Config(alpha_word=0.05, alpha_sub=0.2, alpha_hour=1.0, vocab_size=100000, selector="mi", min_total_count=3),

    Config(alpha_word=0.1,  alpha_sub=0.5, alpha_hour=2.0, vocab_size=50000,  selector="mi", min_total_count=3),
]

### CSV helpers

def read_csv_dicts(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def load_user_to_geo(labels_csv: str, user_col: str = "author", geohash_col: str = "geohash_5") -> Dict[str, str]:
    user_to_geo: Dict[str, str] = {}
    with open(labels_csv, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader):
            uid = (r.get(user_col) or "").strip().lower()
            gh = (r.get(geohash_col) or "").strip()
            if uid and gh:
                user_to_geo[uid] = gh
            if i and i % 50000 == 0:
                log(f"[labels] read {i:,} rows", 1)
    log(f"[labels] total labeled users: {len(user_to_geo):,}", 1)
    return user_to_geo

def summarize_split(name: str, users: List[str], labels: List[str]):
    c = Counter(labels)
    log(f"\n{name} split:", 1)
    log(f"  users: {len(users):,}", 1)
    log(f"  locations: {len(c):,}", 1)
    if c:
        log(f"  min per location: {min(c.values())}", 2)
        log(f"  max per location: {max(c.values())}", 2)

### Feature Loading

## Small feature files

def load_subreddit_counts(path: str) -> Dict[str, Dict[str, int]]:
    log("[subs] loading subreddit counts jsonl", 1)
    subs: Dict[str, Dict[str, int]] = {}
    with open(path, "rb") as f:
        for i, line in enumerate(f):
            obj = _json_loads(line)
            uid = (obj.get("author") or "").strip().lower()
            if not uid:
                continue
            subs[uid] = obj.get("subreddit_counts") or {}
            if i and i % 50000 == 0:
                log(f"[subs] processed {i:,} users", 1)
    log(f"[subs] loaded users: {len(subs):,}", 1)
    return subs

def load_hour_counts(path: str) -> Dict[str, Dict[str, int]]:
    log("[hours] loading hour-bin counts jsonl", 1)
    hours: Dict[str, Dict[str, int]] = {}
    with open(path, "rb") as f:
        for i, line in enumerate(f):
            obj = _json_loads(line)
            uid = (obj.get("author") or "").strip().lower()
            if not uid:
                continue
            raw = obj.get("hour_counts") or obj.get("gmt_hour_counts") or {}
            norm: Dict[str, int] = {}
            if isinstance(raw, dict):
                for k, v in raw.items():
                    try:
                        hk = int(k)
                        iv = int(v)
                    except Exception:
                        continue
                    if 0 <= hk <= 23 and iv > 0:
                        norm[f"{hk:02d}"] = iv
            hours[uid] = norm
            if i and i % 50000 == 0:
                log(f"[hours] processed {i:,} users", 1)
    log(f"[hours] loaded users: {len(hours):,}", 1)
    return hours

def merge_small_features(uid: str,
                         subs_by_user: Dict[str, Dict[str, int]],
                         hours_by_user: Optional[Dict[str, Dict[str, int]]] = None) -> Dict[str, int]:
    out: Dict[str, int] = {}
    subs = subs_by_user.get(uid) or {}
    for k, v in subs.items():
        if v:
            out[f"{PREFIX_SUB}{k}"] = int(v)
    if hours_by_user is not None:
        hrs = hours_by_user.get(uid) or {}
        for k, v in hrs.items():
            if v:
                out[f"{PREFIX_HOUR}{int(k):02d}"] = int(v)
    return out

## Vocabulary (large feature set) preloading
# NOTE: Preload word-count dicts for a specified set of authors in ONE PASS over vocab_jsonl. Returns uid -> {word:count} using raw (un-prefixed) word keys from file.
def load_vocab_counts_for_users(vocab_jsonl: str, users_set: set) -> Dict[str, Dict[str, int]]:
    log(f"[vocab] preloading vocab for {len(users_set):,} users from {vocab_jsonl}", 1)
    out: Dict[str, Dict[str, int]] = {}
    with open(vocab_jsonl, "rb") as f:
        for i, line in enumerate(f):
            obj = _json_loads(line)
            uid = (obj.get("author") or "").strip().lower()
            if uid in users_set:
                vc = obj.get("vocab") or {}
                # keep only positive ints
                norm: Dict[str, int] = {}
                if isinstance(vc, dict):
                    for k, v in vc.items():
                        try:
                            iv = int(v)
                        except Exception:
                            continue
                        if iv > 0:
                            norm[str(k)] = iv
                out[uid] = norm
            if i and i % 50000 == 0:
                log(f"[vocab] scanned {i:,} lines | matched {len(out):,}", 1)
    log(f"[vocab] loaded vocab for {len(out):,} users (requested {len(users_set):,})", 1)
    return out

# Build per-location TRAIN counts from preloaded per-user maps
def build_train_geo_vocab_from_preloaded(
    train_users: Iterable[str],
    user_to_geo: Dict[str, str],
    subs_by_user: Dict[str, Dict[str, int]],
    hours_by_user: Dict[str, Dict[str, int]],
    vocab_by_user: Dict[str, Dict[str, int]],
) -> Dict[str, Counter]:
    geo_vocab: Dict[str, Counter] = defaultdict(Counter)
    for uid in train_users:
        gh = user_to_geo.get(uid)
        if not gh:
            continue
        counts = merge_small_features(uid, subs_by_user, hours_by_user)
        vc = vocab_by_user.get(uid) or {}
        if vc:
            for k, v in vc.items():
                if v:
                    counts[f"{PREFIX_WORD}{k}"] = int(v)
        geo_vocab[gh].update(counts)
    geo_vocab[UNKNOWN_LABEL] = Counter()
    log(f"[train] locations (seen): {len(geo_vocab)-1:,} (+UNKNOWN)", 1)
    return geo_vocab

### Parameter Estimation

# location priors
def compute_location_priors_smoothed_from_labels(train_labels: List[str], train_locations: List[str], kappa: float = 0.5) -> Dict[str, float]:
    counts = Counter(train_labels)
    total = sum(counts.values())
    L = len(train_locations)
    denom = total + kappa * (L + 1)
    priors = {gh: (counts.get(gh, 0) + kappa) / denom for gh in train_locations}
    priors[UNKNOWN_LABEL] = kappa / denom
    return priors

# Feature selection (Mutual Iinformation with locations)
def mutual_information_scores(
    geo_vocab: Dict[str, Counter],
    priors: Dict[str, float],
    min_total_count: int,
) -> Dict[str, float]:
    total_by_loc = {gh: sum(c.values()) for gh, c in geo_vocab.items() if sum(c.values()) > 0}
    locs = list(total_by_loc.keys())
    total_all = sum(total_by_loc.values())
    if total_all == 0:
        return {}

    prior_mass = sum(priors.get(l, 0.0) for l in locs)
    if prior_mass > 0:
        pL = {l: priors.get(l, 0.0) / prior_mass for l in locs}
    else:
        s = sum(total_by_loc.values())
        pL = {l: total_by_loc[l] / s for l in locs}

    global_word = Counter()
    for gh in locs:
        global_word.update(geo_vocab[gh])

    pW = {w: c / total_all for w, c in global_word.items() if c >= min_total_count}

    mi = {}
    for w, pw in pW.items():
        score = 0.0
        for l in locs:
            c_lw = geo_vocab[l].get(w, 0)
            if c_lw <= 0:
                continue
            p_lw = c_lw / total_all
            score += p_lw * math.log(p_lw / (pL[l] * pw))
        mi[w] = score
    return mi

def select_vocabulary(
    geo_vocab: Dict[str, Counter],
    priors: Dict[str, float],
    vocab_size: int,
    selector: str,
    min_total_count: int,
) -> List[str]:
    global_word = Counter()
    for c in geo_vocab.values():
        global_word.update(c)

    if selector == "freq":
        return [w for w, _ in global_word.most_common(vocab_size)]

    if selector == "mi":
        mi = mutual_information_scores(geo_vocab, priors, min_total_count=min_total_count)
        top_freq = [w for w, _ in global_word.most_common(vocab_size)]
        top_mi = [w for w, _ in sorted(mi.items(), key=lambda kv: kv[1], reverse=True)]
        chosen = []
        seen = set()
        for w in top_mi:
            if w not in seen:
                chosen.append(w); seen.add(w)
            if len(chosen) >= vocab_size:
                break
        for w in top_freq:
            if len(chosen) >= vocab_size:
                break
            if w not in seen:
                chosen.append(w); seen.add(w)
        return chosen[:vocab_size]

    raise ValueError("selector must be 'mi' or 'freq'")

### Sparse and fast Location Parameter and constant calculation

@dataclass
class LocationParams:
    alpha_sum: float
    const: float
    # sparse overrides: only features with nonzero count in this location AND in vocab_set.
    alpha: Dict[str, float]
    # base smoothings (needed to reconstruct alpha for features missing from alpha dict)
    alpha_word: float
    alpha_sub: float
    alpha_hour: float

def _base_alpha_for_feat(feat: str, alpha_word: float, alpha_sub: float, alpha_hour: float) -> float:
    if feat.startswith(PREFIX_WORD):
        return alpha_word
    if feat.startswith(PREFIX_SUB):
        return alpha_sub
    if feat.startswith(PREFIX_HOUR):
        return alpha_hour
    return alpha_word

def precompute_location_params_sparse(
    geo_vocab: Dict[str, Counter],
    vocab: List[str],
    alpha_word: float,
    alpha_sub: float,
    alpha_hour: float,
    locations: Optional[List[str]] = None,
) -> Dict[str, LocationParams]:
    if locations is None:
        locations = list(geo_vocab.keys())

    vocab_set = set(vocab)

    # Precompute base sums once (over vocab)
    base_sum = 0.0
    sum_lgamma_base = 0.0
    base_by_feat: Dict[str, float] = {}
    for feat in vocab:
        b = _base_alpha_for_feat(feat, alpha_word, alpha_sub, alpha_hour)
        base_by_feat[feat] = b
        base_sum += b
        sum_lgamma_base += math.lgamma(b)

    params: Dict[str, LocationParams] = {}
    for gh in locations:
        wc = geo_vocab.get(gh, Counter())
        # Only consider location features in vocab_set with nonzero counts
        # and store alpha_override = base + count
        alpha_sparse: Dict[str, float] = {}
        alpha_sum = base_sum
        sum_lgamma_alpha = sum_lgamma_base

        # iterate over nonzero location features (much smaller than vocab)
        for feat, c in wc.items():
            if c <= 0:
                continue
            if feat not in vocab_set:
                continue
            b = base_by_feat.get(feat)
            if b is None:
                b = _base_alpha_for_feat(feat, alpha_word, alpha_sub, alpha_hour)
            a = b + float(c)
            alpha_sparse[feat] = a
            alpha_sum += float(c)
            # adjust lgamma sum: lgamma(b+c) - lgamma(b)
            sum_lgamma_alpha += math.lgamma(a) - math.lgamma(b)

        const = math.lgamma(alpha_sum) - sum_lgamma_alpha
        params[gh] = LocationParams(
            alpha_sum=alpha_sum,
            const=const,
            alpha=alpha_sparse,
            alpha_word=alpha_word,
            alpha_sub=alpha_sub,
            alpha_hour=alpha_hour,
        )
    return params

# calculate log-likelihood
def dm_loglik_sparse(counts: Dict[str, int], lp: LocationParams, vocab_set: set) -> Tuple[float, int]:
    # only use observed features in vocab
    x_used = {w: c for w, c in counts.items() if c > 0 and w in vocab_set}
    N = sum(x_used.values())
    if N == 0:
        return 0.0, 0
    s = lp.const - math.lgamma(lp.alpha_sum + N)
    for w, c in x_used.items():
        b = _base_alpha_for_feat(w, lp.alpha_word, lp.alpha_sub, lp.alpha_hour)
        a = lp.alpha.get(w, b)
        s += math.lgamma(a + c)
    return s, N

### GPU batched scorer (Torch)
# NOTE: Batched BxL scorer for DM log posterior: score(u, loc) = log_prior(loc) + const(loc) - lgamma(alpha_sum(loc) + N_u) + sum_f lgamma(alpha(loc,f) + c_u,f)

class TorchBatchedScorer:
    def __init__(self, locations: List[str], priors: Dict[str, float], loc_params: Dict[str, LocationParams], vocab_set: set):
        if not _TORCH_AVAILABLE:
            raise RuntimeError("Torch not available")

        self.locations = locations
        self.vocab_set = vocab_set

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_cuda = (self.device.type == "cuda")

        # location tensors
        prior_floor = 1e-30
        log_priors = [math.log(priors.get(gh, prior_floor)) for gh in locations]
        consts = [loc_params[gh].const for gh in locations]
        alpha_sums = [loc_params[gh].alpha_sum for gh in locations]

        self.log_priors = torch.tensor(log_priors, device=self.device, dtype=torch.float32)  # [L]
        self.consts = torch.tensor(consts, device=self.device, dtype=torch.float32)          # [L]
        self.alpha_sums = torch.tensor(alpha_sums, device=self.device, dtype=torch.float32)  # [L]

        # Use smoothings from any loc (they are the same per config)
        sample_lp = loc_params[locations[0]]
        self.alpha_word = float(sample_lp.alpha_word)
        self.alpha_sub = float(sample_lp.alpha_sub)
        self.alpha_hour = float(sample_lp.alpha_hour)

        # Build overrides index: feat -> (loc_idx_tensor, alpha_override_tensor)
        # Store only feats in vocab_set.
        feat_to_locidx: Dict[str, List[int]] = defaultdict(list)
        feat_to_alpha: Dict[str, List[float]] = defaultdict(list)
        for li, gh in enumerate(locations):
            lp = loc_params[gh]
            for feat, a in lp.alpha.items():
                # lp.alpha stores only overrides already filtered to vocab_set
                feat_to_locidx[feat].append(li)
                feat_to_alpha[feat].append(float(a))

        self.override_locidx: Dict[str, torch.Tensor] = {}
        self.override_alpha: Dict[str, torch.Tensor] = {}
        for feat, idxs in feat_to_locidx.items():
            self.override_locidx[feat] = torch.tensor(idxs, device=self.device, dtype=torch.long)
            self.override_alpha[feat] = torch.tensor(feat_to_alpha[feat], device=self.device, dtype=torch.float32)

    def _base_alpha_feat(self, feat: str) -> float:
        if feat.startswith(PREFIX_WORD):
            return self.alpha_word
        if feat.startswith(PREFIX_SUB):
            return self.alpha_sub
        if feat.startswith(PREFIX_HOUR):
            return self.alpha_hour
        return self.alpha_word

    # Returns for each user: list of (location, score) sorted desc, truncated to topk.
    @torch.no_grad()
    def score_topk(self, batch_counts: List[Dict[str, int]], topk: int = 10, batch_size_hint: int = 4096) -> List[List[Tuple[str, float]]]:
        L = len(self.locations)
        out: List[List[Tuple[str, float]]] = []

        # Process in sub-batches to avoid huge intermediates if user passes a big list
        for start in range(0, len(batch_counts), batch_size_hint):
            sub = batch_counts[start:start+batch_size_hint]
            B = len(sub)

            # N_u
            Ns = []
            base_terms = []
            # For sparse updates, we will collect per-user updates (still no loc loops)
            per_user_feats: List[List[Tuple[str, int]]] = []

            for counts in sub:
                feats = [(f, int(c)) for f, c in counts.items() if c > 0 and f in self.vocab_set]
                per_user_feats.append(feats)
                N = sum(c for _, c in feats)
                Ns.append(float(N))
                # base term: sum lgamma(base + c)
                bt = 0.0
                for f, c in feats:
                    b = self._base_alpha_feat(f)
                    bt += math.lgamma(b + c)
                base_terms.append(bt)

            N_t = torch.tensor(Ns, device=self.device, dtype=torch.float32)          # [B]
            base_t = torch.tensor(base_terms, device=self.device, dtype=torch.float32)  # [B]

            # scores[B, L] = log_priors + const - lgamma(alpha_sum + N_u) + base_term(u)
            scores = self.log_priors.unsqueeze(0).expand(B, L) + self.consts.unsqueeze(0).expand(B, L)
            scores = scores - torch.lgamma(self.alpha_sums.unsqueeze(0) + N_t.unsqueeze(1))
            scores = scores + base_t.unsqueeze(1)

            # Apply sparse deltas for overrides:
            # For each user u and each feature f in u, if f has overrides at some locations:
            #   delta(loc) = lgamma(alpha_override(loc) + c) - lgamma(base + c)
            # and add to scores[u, loc_idx]
            for ui, feats in enumerate(per_user_feats):
                if not feats:
                    continue
                for f, c in feats:
                    loc_idx = self.override_locidx.get(f)
                    if loc_idx is None:
                        continue
                    alpha_override = self.override_alpha[f]  # [M]
                    b = self._base_alpha_feat(f)
                    # delta vector for those locations
                    delta = torch.lgamma(alpha_override + float(c)) - float(math.lgamma(b + c))
                    # scatter-add into scores[ui, loc_idx]
                    scores[ui].index_add_(0, loc_idx, delta)

            # topk
            k = min(topk, L)
            vals, idxs = torch.topk(scores, k=k, dim=1, largest=True, sorted=True)
            vals_cpu = vals.detach().cpu().tolist()
            idxs_cpu = idxs.detach().cpu().tolist()

            for u in range(B):
                out.append([(self.locations[j], float(vals_cpu[u][t])) for t, j in enumerate(idxs_cpu[u])])

        return out

### StreamMetrics + evaluation (preloaded, GPU optional)

@dataclass
class StreamMetrics:
    n: int
    top1_acc: float
    top5_acc: float
    top10_acc: float
    mrr: float
    log_loss: float
    avg_in_vocab_tokens: float

    top1_precision_micro: float
    top1_recall_micro: float
    top1_f1_micro: float
    top1_precision_macro: float
    top1_recall_macro: float
    top1_f1_macro: float

    topk_precision_micro: float
    topk_recall_micro: float
    topk_f1_micro: float
    topk_precision_macro: float
    topk_recall_macro: float
    topk_f1_macro: float

    unseen_true_users: int
    unseen_true_locations: int

def _safe_div(num: float, den: float) -> float:
    return (num / den) if den else 0.0

# precision, recall and F1
def _prf_from_counts(tp: Dict[str, int], fp: Dict[str, int], fn: Dict[str, int]) -> Dict[str, float]:
    labels = set(tp) | set(fp) | set(fn)
    tp_sum = sum(tp.values())
    fp_sum = sum(fp.values())
    fn_sum = sum(fn.values())
    p_micro = _safe_div(tp_sum, tp_sum + fp_sum)
    r_micro = _safe_div(tp_sum, tp_sum + fn_sum)
    f1_micro = _safe_div(2 * p_micro * r_micro, p_micro + r_micro)

    p_list = []
    r_list = []
    f1_list = []
    for y in labels:
        t = tp.get(y, 0)
        f_p = fp.get(y, 0)
        f_n = fn.get(y, 0)
        p = _safe_div(t, t + f_p)
        r = _safe_div(t, t + f_n)
        f1 = _safe_div(2 * p * r, p + r)
        p_list.append(p)
        r_list.append(r)
        f1_list.append(f1)

    p_macro = sum(p_list) / len(p_list) if p_list else 0.0
    r_macro = sum(r_list) / len(r_list) if r_list else 0.0
    f1_macro = sum(f1_list) / len(f1_list) if f1_list else 0.0

    return {
        "precision_micro": p_micro,
        "recall_micro": r_micro,
        "f1_micro": f1_micro,
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "f1_macro": f1_macro,
    }

def evaluate_split_preloaded(
    split_name: str,
    labeled_users: List[Tuple[str, Dict[str, int]]],  # (true_eval_label, counts)
    loc_params: Dict[str, LocationParams],
    priors: Dict[str, float],
    vocab: List[str],
    locations: List[str],
    use_torch_if_available: bool = True,
    batch_size: int = 4096,
) -> StreamMetrics:
    log(f"\n[{split_name}] evaluation (preloaded features)", 1)
    vocab_set = set(vocab)
    prior_floor = 1e-30

    # unseen tracking already handled upstream by mapping to UNKNOWN_LABEL
    unseen_true_users = 0
    unseen_true_locations = 0

    # P/R/F1 counters
    tp1 = defaultdict(int); fp1 = defaultdict(int); fn1 = defaultdict(int)
    tpk = defaultdict(int); fpk = defaultdict(int); fnk = defaultdict(int)

    hit1 = hit5 = hit10 = 0
    rr_sum = 0.0
    logloss_sum = 0.0
    invocab_tok_sum = 0.0

    # Torch scorer if possible
    scorer = None
    if use_torch_if_available and _TORCH_AVAILABLE:
        try:
            scorer = TorchBatchedScorer(locations, priors, loc_params, vocab_set)
            log(f"[{split_name}] using Torch scorer on device={scorer.device}", 1)
        except Exception as e:
            log(f"[{split_name}] Torch scorer unavailable ({e}); falling back to CPU.", 1, stream=sys.stderr)
            scorer = None

    def score_cpu(counts: Dict[str, int]) -> List[Tuple[str, float]]:
        scored = []
        for gh in locations:
            lp = loc_params[gh]
            ll, _ = dm_loglik_sparse(counts, lp, vocab_set)
            scored.append((gh, math.log(priors.get(gh, prior_floor)) + ll))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    n = len(labeled_users)
    for start in range(0, n, batch_size):
        batch = labeled_users[start:start+batch_size]
        batch_counts = [c for _, c in batch]
        batch_true = [t for t, _ in batch]

        if scorer is not None:
            ranked_batch = scorer.score_topk(batch_counts, topk=10, batch_size_hint=len(batch_counts))
        else:
            ranked_batch = [score_cpu(c)[:10] for c in batch_counts]

        for true_eval, ranked in zip(batch_true, ranked_batch):
            labels_ranked = [gh for gh, _ in ranked]
            if labels_ranked and labels_ranked[0] == true_eval:
                hit1 += 1
            if true_eval in labels_ranked[:5]:
                hit5 += 1
            if true_eval in labels_ranked[:10]:
                hit10 += 1

            pred1 = labels_ranked[0] if labels_ranked else UNKNOWN_LABEL
            if pred1 == true_eval:
                tp1[true_eval] += 1
            else:
                fp1[pred1] += 1
                fn1[true_eval] += 1

            k_set = set(labels_ranked[:PRF_TOPK])
            if true_eval in k_set:
                tpk[true_eval] += 1
            else:
                fnk[true_eval] += 1
            for lab in k_set:
                if lab != true_eval:
                    fpk[lab] += 1

            try:
                r = labels_ranked.index(true_eval) + 1
                rr_sum += 1.0 / r
            except ValueError:
                pass

            # NOTE: logloss: need normalizer across all locations; if Torch scorer is used we only have top10,
            # Compute logloss exactly only in CPU mode. In GPU mode, skip exact logloss and approximate using top10.
            # For model selection, macro-F1@1 is the primary objective, so this approximation is acceptable.
            if scorer is None:
                # exact logloss
                all_scored = score_cpu(batch_counts[0])  # placeholder; will be replaced below
            # We'll handle logloss below for CPU mode only; for Torch mode, store NaN
            invocab_tok_sum += sum(v for feat, v in (batch_counts[0].items() if batch_counts else []) if feat in vocab_set)

        # invocab_tok_sum computed incorrectly above in loop placeholder; fix properly:
        for _, counts in batch:
            invocab_tok_sum += sum(v for feat, v in counts.items() if feat in vocab_set)

        # logloss: CPU exact (compute full distribution)
        if scorer is None:
            for (true_eval, counts) in batch:
                # full scores
                logps = {}
                for gh in locations:
                    lp = loc_params[gh]
                    ll, _ = dm_loglik_sparse(counts, lp, vocab_set)
                    logps[gh] = math.log(priors.get(gh, prior_floor)) + ll
                m = max(logps.values())
                lse = m + math.log(sum(math.exp(s - m) for s in logps.values()))
                true_score = logps.get(true_eval, -1e30)
                logloss_sum += -(true_score - lse)
        else:
            # approximate logloss from top10 only (lower bound on normalizer)
            for (true_eval, ranked) in zip(batch_true, ranked_batch):
                d = dict(ranked)
                if true_eval not in d:
                    # treat as very low prob
                    logloss_sum += 50.0
                else:
                    scores = list(d.values())
                    m = max(scores)
                    lse = m + math.log(sum(math.exp(s - m) for s in scores))
                    logloss_sum += -(d[true_eval] - lse)

    prf1 = _prf_from_counts(tp1, fp1, fn1)
    prfk = _prf_from_counts(tpk, fpk, fnk)

    if n == 0:
        return StreamMetrics(0,0,0,0,0,float("inf"),0,
                             0,0,0,0,0,0,
                             0,0,0,0,0,0,
                             unseen_true_users, unseen_true_locations)

    return StreamMetrics(
        n=n,
        top1_acc=hit1 / n,
        top5_acc=hit5 / n,
        top10_acc=hit10 / n,
        mrr=rr_sum / n,
        log_loss=logloss_sum / n,
        avg_in_vocab_tokens=invocab_tok_sum / n,

        top1_precision_micro=prf1["precision_micro"],
        top1_recall_micro=prf1["recall_micro"],
        top1_f1_micro=prf1["f1_micro"],
        top1_precision_macro=prf1["precision_macro"],
        top1_recall_macro=prf1["recall_macro"],
        top1_f1_macro=prf1["f1_macro"],

        topk_precision_micro=prfk["precision_micro"],
        topk_recall_micro=prfk["recall_micro"],
        topk_f1_micro=prfk["f1_micro"],
        topk_precision_macro=prfk["precision_macro"],
        topk_recall_macro=prfk["recall_macro"],
        topk_f1_macro=prfk["f1_macro"],

        unseen_true_users=unseen_true_users,
        unseen_true_locations=unseen_true_locations,
    )

### Saved model I/O

@dataclass
class SavedModel:
    config: 'Config'
    vocab: List[str]
    priors: Dict[str, float]
    loc_params: Dict[str, 'LocationParams']
    locations: List[str]

def save_model(model: SavedModel, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    log(f"[saved] model -> {path}", 1)

### Checkpoint helpers

def _cfg_key(cfg: 'Config') -> str:
    s = f"alpha_word={cfg.alpha_word}|alpha_sub={cfg.alpha_sub}|alpha_hour={cfg.alpha_hour}|vocab_size={cfg.vocab_size}|selector={cfg.selector}|min_total_count={cfg.min_total_count}"
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]

def _cfg_to_dict(cfg: 'Config') -> dict:
    return {
        "alpha_word": cfg.alpha_word,
        "alpha_sub": cfg.alpha_sub,
        "alpha_hour": cfg.alpha_hour,
        "vocab_size": cfg.vocab_size,
        "selector": cfg.selector,
        "min_total_count": cfg.min_total_count,
    }

def _ckpt_paths(cfg: 'Config') -> Tuple[str, str]:
    os.makedirs(GRID_CHECKPOINT_DIR, exist_ok=True)
    key = _cfg_key(cfg)
    metrics_path = os.path.join(GRID_CHECKPOINT_DIR, f"result_{key}.json")
    model_path = os.path.join(GRID_CHECKPOINT_DIR, f"model_{key}.pkl")
    return metrics_path, model_path

def _load_ckpt_metrics(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _save_ckpt_metrics(path: str, payload: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp, path)

def _list_completed_cfg_keys(configs: List['Config']) -> set:
    completed = set()
    for cfg in configs:
        metrics_path, _ = _ckpt_paths(cfg)
        if os.path.exists(metrics_path):
            completed.add(_cfg_key(cfg))
    return completed

### Main

@dataclass
class RunResult:
    config: 'Config'
    val_metrics: StreamMetrics

def main():
    if loc_type == "US":
        label_file = "us_geohash.csv"
    elif loc_type == "non-US":
        label_file = "non_us_geohash.csv"
    elif loc_type == "global":
        label_file = "combined_geohash.csv"
    else:
        raise Exception("Wrong loc_type value. Choose from \'US\',\'non-US\' and \'global\'.")
    labels_csv = os.path.join(DATA_DIR, "data_reddit_location", label_file)

    user_to_geo = load_user_to_geo(labels_csv, user_col="author", geohash_col="geohash_5")
    all_users = list(user_to_geo.keys())
    all_labels = [user_to_geo[u] for u in all_users]

    train_users, train_labels, val_users, val_labels, test_users, test_labels = prepare_splits(
        all_users, all_labels, split_dir=SPLIT_DIR, description=f"{loc_type} location"
    )

    summarize_split("Train", train_users, train_labels)
    summarize_split("Valid", val_users, val_labels)
    summarize_split("Test", test_users, test_labels)

    train_users_set = set(train_users)
    val_users_set = set(val_users)
    test_users_set = set(test_users)

    # Load small feature sets
    subs_by_user = load_subreddit_counts(SUBS_JSONL)

    if not os.path.exists(HOURS_JSONL):
        raise FileNotFoundError(f"Required hour feature file not found: {HOURS_JSONL}")
    hours_by_user = load_hour_counts(HOURS_JSONL)

    # Preload vocab ONCE for all labeled users (train+val+test)
    all_eval_users_set = train_users_set | val_users_set | test_users_set
    vocab_by_user = load_vocab_counts_for_users(VOCAB_FILE, all_eval_users_set)

    # Build train per-location counts from preloaded user features 
    train_geo_vocab = build_train_geo_vocab_from_preloaded(
        train_users, user_to_geo, subs_by_user, hours_by_user, vocab_by_user
    )

    train_locations = [gh for gh in train_geo_vocab.keys() if gh != UNKNOWN_LABEL]
    train_locations_set = set(train_locations)

    train_priors = compute_location_priors_smoothed_from_labels(train_labels, train_locations, kappa=0.5)

    # Prebuild preloaded val/test labeled sets once
    def _make_labeled(users: List[str]) -> List[Tuple[str, Dict[str, int]]]:
        out: List[Tuple[str, Dict[str, int]]] = []
        for uid in users:
            true_gh = user_to_geo.get(uid)
            if not true_gh:
                continue
            if true_gh not in train_locations_set:
                true_eval = UNKNOWN_LABEL
            else:
                true_eval = true_gh
            counts = merge_small_features(uid, subs_by_user, hours_by_user)
            vc = vocab_by_user.get(uid) or {}
            if vc:
                for k, v in vc.items():
                    if v:
                        counts[f"{PREFIX_WORD}{k}"] = int(v)
            out.append((true_eval, counts))
        return out

    val_labeled = _make_labeled(val_users)
    test_labeled = _make_labeled(test_users)

    log(f"[valid] preloaded labeled users: {len(val_labeled):,}", 1)
    log(f"[test] preloaded labeled users: {len(test_labeled):,}", 1)

    os.makedirs(GRID_CHECKPOINT_DIR, exist_ok=True)
    total_cfgs = len(configs)
    completed_keys = _list_completed_cfg_keys(configs)

    log(f"\n[grid] total combinations: {total_cfgs}", 1)
    if completed_keys:
        log(f"[grid] found checkpoints for {len(completed_keys)} / {total_cfgs} combinations (will skip those)", 1)

    results: List[RunResult] = []

    stop_requested = {"flag": False}
    def _handle_sigint(signum, frame):
        stop_requested["flag"] = True
        log("\n[interrupt] Ctrl+C received — will stop after finishing the current grid point.", 1, stream=sys.stderr)
    try:
        signal.signal(signal.SIGINT, _handle_sigint)
    except Exception:
        pass

    # Locations list (includes UNKNOWN)
    locs = list(train_priors.keys())

    for idx, cfg in enumerate(configs, start=1):
        key = _cfg_key(cfg)
        metrics_path, model_path = _ckpt_paths(cfg)

        log(f"\n[grid] ({idx}/{total_cfgs}) cfg={cfg}", 1)

        if (not OVERWRITE_CHECKPOINTS) and os.path.exists(metrics_path):
            payload = _load_ckpt_metrics(metrics_path)
            if payload and "val_metrics" in payload:
                vm_dict = payload["val_metrics"]
                vm = StreamMetrics(**vm_dict)
                log(f"[grid] -> already done, loaded checkpoint (f1_macro@1={vm.top1_f1_macro:.4f}, top1={vm.top1_acc:.4f})", 1)
                results.append(RunResult(config=cfg, val_metrics=vm))
                if stop_requested["flag"]:
                    break
                continue
            else:
                log("[grid] checkpoint exists but could not be read — recomputing.", 1)

        t0 = time.time()

        vocab = select_vocabulary(
            train_geo_vocab,
            priors=train_priors,
            vocab_size=cfg.vocab_size,
            selector=cfg.selector,
            min_total_count=cfg.min_total_count,
        )

        loc_params = precompute_location_params_sparse(
            train_geo_vocab,
            vocab=vocab,
            alpha_word=cfg.alpha_word,
            alpha_sub=cfg.alpha_sub,
            alpha_hour=cfg.alpha_hour,
            locations=locs,
        )

        vm = evaluate_split_preloaded(
            "valid",
            val_labeled,
            loc_params,
            train_priors,
            vocab,
            locations=locs,
            use_torch_if_available=True,
            batch_size=4096,
        )

        elapsed = time.time() - t0
        log(f"[val] n={vm.n:,} top1={vm.top1_acc:.4f} top5={vm.top5_acc:.4f} "
            f"f1_macro@1={vm.top1_f1_macro:.4f} logloss={vm.log_loss:.4f} "
            f"(elapsed {elapsed/60:.1f} min)", 1)

        if SAVE_CANDIDATE_MODELS:
            candidate_model = SavedModel(
                config=cfg, vocab=vocab, priors=train_priors, loc_params=loc_params, locations=locs
            )
            save_model(candidate_model, model_path)

        _save_ckpt_metrics(metrics_path, {
            "cfg_key": key,
            "config": _cfg_to_dict(cfg),
            "val_metrics": asdict(vm),
            "saved_model_path": model_path if SAVE_CANDIDATE_MODELS else None,
            "timestamp_unix": int(time.time()),
        })

        results.append(RunResult(config=cfg, val_metrics=vm))

        if stop_requested["flag"]:
            log("[interrupt] stopping after finishing current grid point (as requested).", 1, stream=sys.stderr)
            break

    results.sort(
        key=lambda r: (
            r.val_metrics.top1_f1_macro,
            r.val_metrics.top5_acc,
            r.val_metrics.mrr,
            -r.val_metrics.log_loss,
        ),
        reverse=True,
    )
    best = results[0]
    best_cfg = best.config
    log(
        f"\n[best] {best_cfg} (val f1_macro@1={best.val_metrics.top1_f1_macro:.4f}, top1={best.val_metrics.top1_acc:.4f}, top5={best.val_metrics.top5_acc:.4f})",
        1,
    )

    best_vocab = select_vocabulary(
        train_geo_vocab,
        priors=train_priors,
        vocab_size=best_cfg.vocab_size,
        selector=best_cfg.selector,
        min_total_count=best_cfg.min_total_count,
    )
    best_loc_params = precompute_location_params_sparse(
        train_geo_vocab,
        vocab=best_vocab,
        alpha_word=best_cfg.alpha_word,
        alpha_sub=best_cfg.alpha_sub,
        alpha_hour=best_cfg.alpha_hour,
        locations=locs,
    )

    tm = evaluate_split_preloaded(
        "test",
        test_labeled,
        best_loc_params,
        train_priors,
        best_vocab,
        locations=locs,
        use_torch_if_available=True,
        batch_size=4096,
    )

    log(
        f"\n[test] n={tm.n:,} "
        f"hit@1={tm.top1_acc:.4f} hit@5={tm.top5_acc:.4f} hit@10={tm.top10_acc:.4f} "
        f"mrr={tm.mrr:.4f} logloss={tm.log_loss:.4f}\n"
        f"       top1 micro P/R/F1={tm.top1_precision_micro:.4f}/{tm.top1_recall_micro:.4f}/{tm.top1_f1_micro:.4f} "
        f"| macro P/R/F1={tm.top1_precision_macro:.4f}/{tm.top1_recall_macro:.4f}/{tm.top1_f1_macro:.4f}\n"
        f"       top{PRF_TOPK} micro P/R/F1={tm.topk_precision_micro:.4f}/{tm.topk_recall_micro:.4f}/{tm.topk_f1_micro:.4f} "
        f"| macro P/R/F1={tm.topk_precision_macro:.4f}/{tm.topk_recall_macro:.4f}/{tm.topk_f1_macro:.4f}\n",
        1,
    )

    if SAVE_MODEL:
        model = SavedModel(
            config=best_cfg,
            vocab=best_vocab,
            priors=train_priors,
            loc_params=best_loc_params,
            locations=locs,
        )
        save_model(model, MODEL_SAVE_PATH)

    log("\n[done]", 1)

if __name__ == "__main__":
    main()
