## imports
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
from typing import Dict, List, Tuple, Optional
from utils import prepare_splits

### Parameters and Training Configuration

# Determine grid search parameter specifications for the Dirichlet-multinomial model
@dataclass(frozen=True)
class Config:
    vocab_size: int
    selector: str
    min_total_count: int
    alpha_word: float

    alpha_sub: float

    alpha_hour: float

configs = [
    # Baselines
    # Same alpha applied to all feature types (words/subs/hours), as before.
    Config(alpha_word=0.1, alpha_sub=0.1, alpha_hour=0.1, vocab_size=50000,  selector="mi", min_total_count=3),
    Config(alpha_word=0.1, alpha_sub=0.1, alpha_hour=0.1, vocab_size=100000, selector="mi", min_total_count=3),
    Config(alpha_word=0.5, alpha_sub=0.5, alpha_hour=0.5, vocab_size=50000,  selector="mi", min_total_count=3),

    # Type-specific smoothing
    # Words are sparse (need less smoothing), subreddits are denser (moderate smoothing),
    # and hour-of-day is tiny (stronger smoothing to avoid overfitting).
    Config(alpha_word=0.05, alpha_sub=0.2, alpha_hour=1.0, vocab_size=50000,  selector="mi", min_total_count=3),
    Config(alpha_word=0.05, alpha_sub=0.2, alpha_hour=1.0, vocab_size=100000, selector="mi", min_total_count=3),

    # A stronger-smoothing variant to test whether macro-F1 improves under heavy class imbalance
    # by reducing overconfident “head-class” predictions.
    Config(alpha_word=0.1,  alpha_sub=0.5, alpha_hour=2.0, vocab_size=50000,  selector="mi", min_total_count=3),
]

# Prefer faster JSON when available
try:
    import orjson as _fastjson  # type: ignore
    def _json_loads(b: bytes):
        return _fastjson.loads(b)
except Exception:
    def _json_loads(b: bytes):
        return json.loads(b)

# Output control
VERBOSITY = 1  # 0=quiet, 1=progress, 2=verbose
def log(msg: str, level: int = 1, stream=None):
    if level <= VERBOSITY:
        print(msg, file=stream)

UNKNOWN_LABEL = "__UNKNOWN__"

# Paths / configuration
loc_type = "global"  # options: US, non-US, global

# set path variables
CODE_DIR = Path(__file__).resolve().parent              # absolute /code
PROJECT_ROOT = CODE_DIR.parent                          # absolute project root
DATA_DIR = PROJECT_ROOT / "data"
MODEL_PATH = PROJECT_ROOT / "models"

# Feature files (confidential; write to the repsoitory owner for possible research access)
SUBS_JSONL = os.path.join(DATA_DIR, "data_reddit_location","subreddit_counts.jsonl")
VOCAB_FILE = os.path.join(DATA_DIR, "data_reddit_location","vocab_counts.jsonl")

USE_HOURS = False # TODO: add time features
HOURS_JSONL = os.path.join(DATA_DIR,"data_reddit_location","hour_counts.jsonl")

# Model saving
SAVE_MODEL = True
MODEL_SAVE_PATH = os.path.join(MODEL_PATH, "label_location", "best_model.pkl")
SPLIT_DIR = os.path.join(MODEL_PATH, "train_location_data_split")

# Grid-search checkpointing (resume support)
GRID_CHECKPOINT_DIR = os.path.join(os.path.dirname(MODEL_SAVE_PATH), "grid_checkpoints")
SAVE_CANDIDATE_MODELS = False  # saves a model for each evaluated grid point (storage-heavy but enables resume/inspection)
OVERWRITE_CHECKPOINTS = False  # set True to recompute even if checkpoint exists
PROGRESS_EVERY_SECS = 60  # for long streaming eval, print a heartbeat at least this often

PRF_TOPK = 5  # for reporting precision/recall/F1 with top-k sets

### Feature reading functions

# Namespacing to avoid collisions between feature types
PREFIX_WORD = "w:"
PREFIX_SUB = "s:"
PREFIX_HOUR = "h:"
def try_parse_json_counts(s: str) -> Optional[Dict[str, int]]:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            try:
                iv = int(v)
            except Exception:
                continue
            if iv > 0:
                out[str(k)] = iv
        return out
    return None

def row_to_counts(
    row: dict,
    user_id_col: str,
    counts_col_candidates: List[str],
) -> Tuple[str, Dict[str, int]]:
    uid = (row.get(user_id_col) or "").strip()
    if not uid:
        raise ValueError(f"Missing user id in row: expected column '{user_id_col}'")

    # Try JSON counts columns first
    for c in counts_col_candidates:
        if c in row and row[c] is not None:
            parsed = try_parse_json_counts(row[c])
            if parsed is not None:
                return uid, parsed

    # Otherwise interpret as wide BoW columns
    counts: Dict[str, int] = {}
    for k, v in row.items():
        if k == user_id_col:
            continue
        if v is None:
            continue
        v = str(v).strip()
        if not v:
            continue
        try:
            iv = int(float(v))
        except Exception:
            continue
        if iv > 0:
            counts[str(k)] = iv

    return uid, counts

## Labeled data loading/parsing

def load_split_users(
    split_csv: str,
    user_id_col: str,
    counts_col_candidates: List[str],
) -> Dict[str, Dict[str, int]]:
    rows = read_csv_dicts(split_csv)
    out = {}
    for r in rows:
        uid, counts = row_to_counts(r, user_id_col, counts_col_candidates)
        out[uid] = counts
    return out

# username to geohash and human-readable version of it
def load_labels(
    labels_csv: str,
    user_id_col_in_labels: str,
    geohash_col: str,
    city_col: str = "city",
    state_col: str = "state",
) -> Tuple[Dict[str, str], Dict[str, str]]:
    rows = read_csv_dicts(labels_csv)
    user_to_geo: Dict[str, str] = {}
    name_counts = defaultdict(Counter)

    for r in rows:
        uid = (r.get(user_id_col_in_labels) or "").strip()
        gh = (r.get(geohash_col) or "").strip()
        if not uid or not gh:
            continue
        user_to_geo[uid] = gh

        city = (r.get(city_col) or "").strip()
        state = (r.get(state_col) or "").strip()
        if city and state:
            name = f"{city}, {state}"
        elif city:
            name = city
        else:
            name = gh
        name_counts[gh][name] += 1

    geo_to_name = {}
    for gh, ctr in name_counts.items():
        geo_to_name[gh] = ctr.most_common(1)[0][0]

    return user_to_geo, geo_to_name

def join_users_with_labels(
    users: Dict[str, Dict[str, int]],
    user_to_geo: Dict[str, str],
) -> List[Tuple[str, Dict[str, int]]]:
    out = []
    missing = 0
    for uid, counts in users.items():
        gh = user_to_geo.get(uid)
        if not gh:
            missing += 1
            continue
        out.append((gh, counts))
    if missing:
        print(f"[warn] {missing} users had no label and were dropped.", file=sys.stderr)
    return out

### Vocab building and feature selection
# NOTE: Based on confidential data. Contact the repository owner for potential research access.

# Aggregate per-location word counts from training users.
def build_geohash_vocab(train_labeled: List[Tuple[str, Dict[str, int]]]) -> Dict[str, Counter]:
    geo_vocab: Dict[str, Counter] = defaultdict(Counter)
    for gh, counts in train_labeled:
        geo_vocab[gh].update(counts)
    return geo_vocab

def compute_location_priors(train_labeled: List[Tuple[str, Dict[str, int]]]) -> Dict[str, float]:
    counts = Counter(gh for gh, _ in train_labeled)
    total = sum(counts.values())
    return {gh: c / total for gh, c in counts.items()} if total else {}

# Smoothed class prior P(geohash) based on training-label frequencies.
# NOTE: kappa is symmetric Dirichlet pseudo-count per class. kappa=0 recovers MLE.
def compute_location_priors_smoothed(
    train_labeled: List[Tuple[str, Dict[str, int]]],
    kappa: float = 0.5,
) -> Dict[str, float]:
    counts = Counter(gh for gh, _ in train_labeled)
    total = sum(counts.values())
    if not total:
        return {}
    L = len(counts)
    denom = total + kappa * L
    return {gh: (c + kappa) / denom for gh, c in counts.items()}

# MI(word,location), treating each word as an event
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

    # P(L)
    prior_mass = sum(priors.get(l, 0.0) for l in locs)
    if prior_mass > 0:
        pL = {l: priors.get(l, 0.0) / prior_mass for l in locs}
    else:
        s = sum(total_by_loc.values())
        pL = {l: total_by_loc[l] / s for l in locs}

    global_word = Counter()
    for gh in locs:
        global_word.update(geo_vocab[gh])

    # P(W)
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
        # If MI yields too few, backfill with frequency
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

# Dirichlet-Multinomial scoring

@dataclass
class LocationParams:
    alpha_sum: float
    const: float
    alpha: Dict[str, float]  # word -> alpha_w

# Precompute per-location constants for fast Dirichlet-multinomial log likelihood.
def precompute_location_params(
    geo_vocab: Dict[str, Counter],
    vocab: List[str],
    alpha_word: float,
    alpha_sub: float,
    alpha_hour: float,
    locations: Optional[List[str]] = None,
) -> Dict[str, LocationParams]:
    """Precompute per-location Dirichlet-multinomial constants.

    We use different symmetric Dirichlet pseudo-counts for different feature families:
      - w:* tokens  -> alpha_word
      - s:* subs    -> alpha_sub
      - h:* hours   -> alpha_hour

    This lets you smooth high-dimensional sparse word features lightly while smoothing
    low-dimensional features (e.g., hours) more strongly.
    """
    if locations is None:
        locations = list(geo_vocab.keys())

    params: Dict[str, LocationParams] = {}
    for gh in locations:
        wc = geo_vocab.get(gh, Counter())

        alpha: Dict[str, float] = {}
        alpha_sum = 0.0
        for feat in vocab:
            # Determine which alpha base applies
            if feat.startswith(PREFIX_WORD):
                base = alpha_word
            elif feat.startswith(PREFIX_SUB):
                base = alpha_sub
            elif feat.startswith(PREFIX_HOUR):
                base = alpha_hour
            else:
                # If somehow un-prefixed, treat like a word feature.
                base = alpha_word

            c = wc.get(feat, 0)
            a = base + c
            alpha[feat] = a
            alpha_sum += a

        # const = log Γ(alpha_sum) - Σ log Γ(alpha_i)
        const = math.lgamma(alpha_sum) - sum(math.lgamma(a) for a in alpha.values())
        params[gh] = LocationParams(alpha_sum=alpha_sum, const=const, alpha=alpha)

    return params

# log p(x | location) under Dirichlet-multinomial, restricted to vocab_set. Returns (loglik, N_used).
def dm_loglik(counts: Dict[str, int], lp: LocationParams, vocab_set: set) -> Tuple[float, int]:    
    x_used = {w: c for w, c in counts.items() if c > 0 and w in vocab_set}
    N = sum(x_used.values())
    if N == 0:
        return 0.0, 0
    s = lp.const - math.lgamma(lp.alpha_sum + N)
    for w, c in x_used.items():
        s += math.lgamma(lp.alpha[w] + c)
    return s, N

def softmax_from_logps(logps: Dict[str, float]) -> Dict[str, float]:
    m = max(logps.values())
    exps = {k: math.exp(v - m) for k, v in logps.items()}
    Z = sum(exps.values())
    return {k: v / Z for k, v in exps.items()}

### Evaluation

def _safe_div(num: float, den: float) -> float:
    return (num / den) if den else 0.0

# Return micro and macro precision/recall/F1 from per-class tp/fp/fn.
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

@dataclass
class Metrics:
    n: int
    top1_acc: float
    top5_acc: float
    top10_acc: float
    mrr: float
    log_loss: float
    avg_in_vocab_tokens: float

def evaluate_split(
    labeled_users: List[Tuple[str, Dict[str, int]]],  # (true_geo, counts)
    loc_params: Dict[str, LocationParams],
    priors: Dict[str, float],
    vocab: List[str],
    topk_list: Tuple[int, int] = (5, 10),
) -> Metrics:
    vocab_set = set(vocab)
    locs = list(loc_params.keys())

    n = 0
    top1 = 0
    top5 = 0
    top10 = 0
    rr_sum = 0.0
    ll_sum = 0.0
    invocab_tokens_sum = 0.0

    k5, k10 = topk_list

    # tiny prior floor for unseen
    prior_floor = 1e-12

    for true_geo, counts in labeled_users:
        n += 1
        logps = {}
        used_tokens_total = sum(c for w, c in counts.items() if w in vocab_set)
        invocab_tokens_sum += used_tokens_total

        for gh in locs:
            loglik, _ = dm_loglik(counts, loc_params[gh], vocab_set)
            logprior = math.log(priors.get(gh, prior_floor))
            logps[gh] = loglik + logprior

        # rank
        ranked = sorted(logps.items(), key=lambda kv: kv[1], reverse=True)
        rank_list = [gh for gh, _ in ranked]

        if rank_list and rank_list[0] == true_geo:
            top1 += 1
        if true_geo in rank_list[:k5]:
            top5 += 1
        if true_geo in rank_list[:k10]:
            top10 += 1

        # reciprocal rank
        try:
            r = rank_list.index(true_geo) + 1
            rr_sum += 1.0 / r
        except ValueError:
            rr_sum += 0.0

        # log loss uses posterior prob of true class
        post = softmax_from_logps(logps)
        p_true = max(post.get(true_geo, 0.0), 1e-15)
        ll_sum += -math.log(p_true)

    if n == 0:
        return Metrics(0, 0, 0, 0, 0, float("inf"), 0)

    return Metrics(
        n=n,
        top1_acc=top1 / n,
        top5_acc=top5 / n,
        top10_acc=top10 / n,
        mrr=rr_sum / n,
        log_loss=ll_sum / n,
        avg_in_vocab_tokens=invocab_tokens_sum / n,
    )

# Grid search

@dataclass
class CandidateResult:
    config: Config
    val_metrics: Metrics

def grid_search(
    train_geo_vocab: Dict[str, Counter],
    train_priors: Dict[str, float],
    val_labeled: List[Tuple[str, Dict[str, int]]],
    configs: List[Config],
    max_locations: Optional[int] = None,
) -> List[CandidateResult]:
    locs = list(train_priors.keys())
    if max_locations is not None and max_locations > 0 and len(locs) > max_locations:
        locs = [gh for gh, _ in sorted(train_priors.items(), key=lambda kv: kv[1], reverse=True)[:max_locations]]

    results: List[CandidateResult] = []
    for cfg in configs:
        vocab = select_vocabulary(
            train_geo_vocab,
            priors=train_priors,
            vocab_size=cfg.vocab_size,
            selector=cfg.selector,
            min_total_count=cfg.min_total_count,
        )
        loc_params = precompute_location_params(
            train_geo_vocab,
            vocab=vocab,
            alpha_word=cfg.alpha_word,
            alpha_sub=cfg.alpha_sub,
            alpha_hour=cfg.alpha_hour,
            locations=locs,
        )
        m = evaluate_split(val_labeled, loc_params, train_priors, vocab)
        results.append(CandidateResult(cfg, m))

        print(
            f"[val] cfg={cfg}  "
            f"logloss={m.log_loss:.4f}  top1={m.top1_acc:.4f}  top5={m.top5_acc:.4f}  "
            f"mrr={m.mrr:.4f}  avg_in_vocab_tokens={m.avg_in_vocab_tokens:.1f}",
            file=sys.stderr
        )
    return results

def read_csv_dicts(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def load_user_to_geo(labels_csv: str, user_col: str = "author", geohash_col: str = "geohash_6") -> Dict[str, str]:
    """Return uid -> geohash (string)."""
    user_to_geo: Dict[str, str] = {}
    with open(labels_csv, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader):
            uid = (r.get(user_col) or "").strip()
            gh = (r.get(geohash_col) or "").strip()
            if uid and gh:
                user_to_geo[uid] = gh
            if i and i % 50000 == 0:
                log(f"[labels] read {i:,} rows", 1)
    log(f"[labels] total labeled users: {len(user_to_geo):,}", 1)
    return user_to_geo

# NOTE: Provides somewhat different info than the generic summarize_split() function in utils.py
def summarize_split(name: str, users: List[str], labels: List[str]):
    c = Counter(labels)
    log(f"\n{name} split:", 1)
    log(f"  users: {len(users):,}", 1)
    log(f"  locations: {len(c):,}", 1)
    if c:
        log(f"  min per location: {min(c.values())}", 2)
        log(f"  max per location: {max(c.values())}", 2)

# Load subreddit-count dicts for all users. Much smaller than vocab.
def load_subreddit_counts(path: str) -> Dict[str, Dict[str, int]]:
    log("[subs] loading subreddit counts jsonl", 1)
    subs: Dict[str, Dict[str, int]] = {}
    with open(path, "rb") as f:
        for i, line in enumerate(f):
            obj = _json_loads(line)
            uid = (obj.get("author") or "").strip()
            if not uid:
                continue
            subs[uid] = obj.get("subreddit_counts") or {}
            if i and i % 50000 == 0:
                log(f"[subs] processed {i:,} users", 1)
    log(f"[subs] loaded users: {len(subs):,}", 1)
    return subs

# TODO: load hour-bin counts. Expected schema: {'author':..., 'hour_counts':{...}}
def load_hour_counts(path: str) -> Dict[str, Dict[str, int]]:
    log("[hours] loading hour-bin counts jsonl", 1)
    hours: Dict[str, Dict[str, int]] = {}
    with open(path, "rb") as f:
        for i, line in enumerate(f):
            obj = _json_loads(line)
            uid = (obj.get("author") or "").strip()
            if not uid:
                continue
            hours[uid] = obj.get("hour_counts") or obj.get("gmt_hour_counts") or {}
            if i and i % 50000 == 0:
                log(f"[hours] processed {i:,} users", 1)
    log(f"[hours] loaded users: {len(hours):,}", 1)
    return hours

# Merge subreddit + hour features (small feature sources) for one user.
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

# Smoothed prior over training locations plus UNKNOWN.
def compute_location_priors_smoothed_from_labels(train_labels: List[str], train_locations: List[str], kappa: float = 0.5) -> Dict[str, float]:
    counts = Counter(train_labels)
    total = sum(counts.values())
    L = len(train_locations)
    denom = total + kappa * (L + 1)
    priors = {gh: (counts.get(gh, 0) + kappa) / denom for gh in train_locations}
    priors[UNKNOWN_LABEL] = kappa / denom
    return priors

@dataclass
class SavedModel:
    config: 'Config'
    vocab: List[str]
    priors: Dict[str, float]
    loc_params: Dict[str, 'LocationParams']
    locations: List[str]

def save_model(model: SavedModel, path: str):
    with open(path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    log(f"[saved] model -> {path}", 1)

def load_model(path: str) -> SavedModel:
    with open(path, "rb") as f:
        return pickle.load(f)

### Checkpoint helpers (allow resuming grid search)

# Stable key for a grid config (used for checkpoint filenames).
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

# Return (metrics_path, model_path) for cfg.
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

# Main inference function. Returns per-sample top-k (label, log_score).
def predict_batch(batch_counts: List[Dict[str, int]], model: SavedModel, topk: int = 5) -> List[List[Tuple[str, float]]]:
    vocab_set = set(model.vocab)
    out: List[List[Tuple[str, float]]] = []
    for counts in batch_counts:
        scored: List[Tuple[str, float]] = []
        for gh in model.locations:
            lp = model.loc_params[gh]
            ll, _ = dm_loglik(counts, lp, vocab_set)
            prior = model.priors.get(gh, 1e-30)
            scored.append((gh, math.log(prior) + ll))
        scored.sort(key=lambda x: x[1], reverse=True)
        out.append(scored[:topk])
    return out

# Build geohash -> Counter(feature->count) for TRAIN ONLY.
# NOTE: Does not store per-user vocab to save memory. Supports both jsonl and json, although json increases memory use.
def build_train_geo_vocab_from_vocab_file(
    vocab_file: str,
    train_users_set: set,
    user_to_geo: Dict[str, str],
    subs_by_user: Dict[str, Dict[str, int]],
    hours_by_user: Optional[Dict[str, Dict[str, int]]] = None,
) -> Dict[str, Counter]:
    geo_vocab: Dict[str, Counter] = defaultdict(Counter)

    # small features first (subreddits / hours)
    for uid in train_users_set:
        gh = user_to_geo.get(uid)
        if not gh:
            continue
        geo_vocab[gh].update(merge_small_features(uid, subs_by_user, hours_by_user))

    def _update_for_user(uid: str, vc: Dict[str, int]) -> None:
        if uid not in train_users_set:
            return
        gh = user_to_geo.get(uid)
        if not gh:
            return
        if vc:
            geo_vocab[gh].update({f"{PREFIX_WORD}{k}": int(v) for k, v in vc.items() if v})

    log(f"[train] building per-location counts from vocab file: {vocab_file}", 1)

    if vocab_file.lower().endswith(".jsonl"):
        with open(vocab_file, "rb") as f:
            for i, line in enumerate(f):
                obj = _json_loads(line)
                uid = (obj.get("author") or "").strip()
                vc = obj.get("vocab") or {}
                _update_for_user(uid, vc)
                if i and i % 50000 == 0:
                    log(f"[vocab] scanned {i:,} lines", 1)
    else:
        # JSON fallback (loads whole file) — expected formats:
        # 1) list[ {author:..., vocab:{...}}, ... ]
        # 2) dict[author] -> vocab_dict
        with open(vocab_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            for i, obj in enumerate(data):
                if not isinstance(obj, dict):
                    continue
                uid = (obj.get("author") or "").strip()
                vc = obj.get("vocab") or {}
                _update_for_user(uid, vc)
                if i and i % 50000 == 0:
                    log(f"[vocab] processed {i:,} records", 1)
        elif isinstance(data, dict):
            # try dict[author] -> vocab
            for i, (uid, vc) in enumerate(data.items()):
                if not isinstance(vc, dict):
                    continue
                _update_for_user(str(uid).strip(), vc)
                if i and i % 50000 == 0:
                    log(f"[vocab] processed {i:,} authors", 1)
        else:
            raise ValueError(f"Unsupported JSON format in {vocab_file}: {type(data)}")

    geo_vocab[UNKNOWN_LABEL] = Counter()
    log(f"[train] locations (seen): {len(geo_vocab)-1:,} (+UNKNOWN)", 1)
    return geo_vocab

@dataclass
class StreamMetrics:
    n: int
    top1_acc: float
    top5_acc: float
    top10_acc: float
    mrr: float
    log_loss: float
    avg_in_vocab_tokens: float

    # Top-1 classification-style P/R/F1 (single predicted label)
    top1_precision_micro: float
    top1_recall_micro: float
    top1_f1_micro: float
    top1_precision_macro: float
    top1_recall_macro: float
    top1_f1_macro: float

    # Top-k set-style P/R/F1 where predicted set = top PRF_TOPK labels
    topk_precision_micro: float
    topk_recall_micro: float
    topk_f1_micro: float
    topk_precision_macro: float
    topk_recall_macro: float
    topk_f1_macro: float

    unseen_true_users: int
    unseen_true_locations: int

@dataclass
class RunResult:
    config: 'Config'
    val_metrics: StreamMetrics

# Stream through vocab jsonl and evaluate only users in users_set.
def stream_evaluate_split(
    split_name: str,
    users_set: set,
    user_to_geo: Dict[str, str],
    vocab_jsonl: str,
    subs_by_user: Dict[str, Dict[str, int]],
    hours_by_user: Optional[Dict[str, Dict[str, int]]],
    loc_params: Dict[str, 'LocationParams'],
    priors: Dict[str, float],
    vocab: List[str],
    train_locations_set: set,
    topk_list: Tuple[int, int, int] = (1, 5, 10),
    progress_every_lines: int = 50000,
    progress_every_secs: int = PROGRESS_EVERY_SECS,
) -> StreamMetrics:
    log(f"\n[{split_name}] streaming evaluation", 1)
    vocab_set = set(vocab)
    locations = list(priors.keys())

    n = 0
    hit = {k: 0 for k in topk_list}

    # P/R/F1 counters
    tp1 = defaultdict(int)
    fp1 = defaultdict(int)
    fn1 = defaultdict(int)
    tpk = defaultdict(int)
    fpk = defaultdict(int)
    fnk = defaultdict(int)
    rr_sum = 0.0
    logloss_sum = 0.0
    invocab_tok_sum = 0.0
    unseen_true_users = 0
    unseen_true_locations_set = set()

    def score_user(counts: Dict[str, int]) -> List[Tuple[str, float]]:
        scored = []
        for gh in locations:
            lp = loc_params[gh]
            ll, _ = dm_loglik(counts, lp, vocab_set)
            prior = priors.get(gh, 1e-30)
            scored.append((gh, math.log(prior) + ll))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    with open(vocab_jsonl, "rb") as f:
        for i, line in enumerate(f):
            obj = _json_loads(line)
            uid = (obj.get("author") or "").strip()
            if uid not in users_set:
                continue

            true_gh = user_to_geo.get(uid)
            if not true_gh:
                continue

            if true_gh not in train_locations_set:
                unseen_true_users += 1
                unseen_true_locations_set.add(true_gh)
                true_eval = UNKNOWN_LABEL
            else:
                true_eval = true_gh

            counts = merge_small_features(uid, subs_by_user, hours_by_user)
            vc = obj.get("vocab") or {}
            if vc:
                for k, v in vc.items():
                    if v:
                        counts[f"{PREFIX_WORD}{k}"] = int(v)

            ranked = score_user(counts)
            labels_ranked = [gh for gh, _ in ranked]
            n += 1

            # --- Top-1 confusion updates ---
            pred1 = labels_ranked[0] if labels_ranked else UNKNOWN_LABEL
            if pred1 == true_eval:
                tp1[true_eval] += 1
            else:
                fp1[pred1] += 1
                fn1[true_eval] += 1

            # --- Top-k (set) confusion updates ---
            k_set = set(labels_ranked[:PRF_TOPK])
            if true_eval in k_set:
                tpk[true_eval] += 1
            else:
                fnk[true_eval] += 1
            for lab in k_set:
                if lab != true_eval:
                    fpk[lab] += 1

            for k in topk_list:
                if true_eval in labels_ranked[:k]:
                    hit[k] += 1

            try:
                rank = labels_ranked.index(true_eval) + 1
                rr_sum += 1.0 / rank
            except ValueError:
                pass

            scores = [s for _, s in ranked]
            m = max(scores)
            lse = m + math.log(sum(math.exp(s - m) for s in scores))
            true_score = dict(ranked).get(true_eval, -1e30)
            logloss_sum += -(true_score - lse)

            invocab_tok_sum += sum(v for feat, v in counts.items() if feat in vocab_set)

            now = time.time()
            if (i and progress_every_lines > 0 and i % progress_every_lines == 0) or (progress_every_secs > 0 and (now - last_print) >= progress_every_secs):
                last_print = now
                log(f"[{split_name}] scanned {i:,} lines | evaluated {n:,} users", 1)

    if n == 0:
        return StreamMetrics(0, 0, 0, 0, 0, float("inf"), 0,
                            0,0,0,0,0,0,
                            0,0,0,0,0,0,
                            unseen_true_users, len(unseen_true_locations_set))

    # Compute P/R/F1 summaries
    prf1 = _prf_from_counts(tp1, fp1, fn1)
    prfk = _prf_from_counts(tpk, fpk, fnk)

    return StreamMetrics(
        n=n,
        top1_acc=hit[topk_list[0]] / n,
        top5_acc=hit[topk_list[1]] / n,
        top10_acc=hit[topk_list[2]] / n,
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
        unseen_true_locations=len(unseen_true_locations_set),
    )

# main function
def main():

    # identify the relevant geohash label file
    if loc_type == "US":
        label_file = "us_geohash.csv"
    elif loc_type == "non-US":
        label_file = "non_us_geohash.csv"
    elif loc_type == "global":
        label_file = "combined_geohash.csv"
    else:
        raise Exception("Wrong loc_type value. Choose from US,non-US and global.")
    labels_csv = os.path.join(DATA_DIR, "data_reddit_location", label_file)

    # load user-geo mapping
    user_to_geo = load_user_to_geo(labels_csv, user_col="author", geohash_col="geohash_6")

    all_users = list(user_to_geo.keys())
    all_labels = [user_to_geo[u] for u in all_users]

    # do 80/10/10 training/validation/test split of the labeled data
    train_users, train_labels, val_users, val_labels, test_users, test_labels = prepare_splits(
        all_users, all_labels, split_dir=SPLIT_DIR, description=f"{loc_type} location"
    )

    summarize_split("Train", train_users, train_labels)
    summarize_split("Valid", val_users, val_labels)
    summarize_split("Test", test_users, test_labels)

    # remove any repetitions in the sets
    train_users_set = set(train_users)
    val_users_set = set(val_users)
    test_users_set = set(test_users)

    # load the small feature sets
    subs_by_user = load_subreddit_counts(SUBS_JSONL)
    hours_by_user = load_hour_counts(HOURS_JSONL) if USE_HOURS and os.path.exists(HOURS_JSONL) else None

    # combine with vocab (large feature set) and build the set of training locations
    train_geo_vocab = build_train_geo_vocab_from_vocab_file(
        VOCAB_FILE, train_users_set, user_to_geo, subs_by_user, hours_by_user
    )

    train_locations = [gh for gh in train_geo_vocab.keys() if gh != UNKNOWN_LABEL]
    train_locations_set = set(train_locations)

    # create smoothed location priors based on the training data
    train_priors = compute_location_priors_smoothed_from_labels(train_labels, train_locations, kappa=0.5)

    # Grid search with progress update and resumption support
    
    os.makedirs(GRID_CHECKPOINT_DIR, exist_ok=True)

    total_cfgs = len(configs)
    completed_keys = _list_completed_cfg_keys(configs)

    log(f"\n[grid] total combinations: {total_cfgs}", 1)
    if completed_keys:
        log(f"[grid] found checkpoints for {len(completed_keys)} / {total_cfgs} combinations (will skip those)", 1)

    results: List[RunResult] = []

    # Graceful Ctrl+C: stop after current config and keep checkpoints already written.
    stop_requested = {"flag": False}
    def _handle_sigint(signum, frame):
        stop_requested["flag"] = True
        log("\n[interrupt] Ctrl+C received — will stop after finishing the current grid point.", 1, stream=sys.stderr)
    try:
        signal.signal(signal.SIGINT, _handle_sigint)
    except Exception:
        pass

    for idx, cfg in enumerate(configs, start=1):
        key = _cfg_key(cfg)
        metrics_path, model_path = _ckpt_paths(cfg)

        log(f"\n[grid] ({idx}/{total_cfgs}) cfg={cfg}", 1)

        if (not OVERWRITE_CHECKPOINTS) and os.path.exists(metrics_path):
            payload = _load_ckpt_metrics(metrics_path)
            if payload and "val_metrics" in payload:
                vm_dict = payload["val_metrics"]
                vm = StreamMetrics(**vm_dict)
                log(f"[grid] -> already done, loaded checkpoint (top1={vm.top1_acc:.4f}, logloss={vm.log_loss:.4f})", 1)
                results.append(RunResult(config=cfg, val_metrics=vm))
                if stop_requested["flag"]:
                    break
                continue
            else:
                log("[grid] checkpoint file exists but could not be read — recomputing this point.", 1)

        t0 = time.time()

        # Build vocab + location params for this config
        vocab = select_vocabulary(
            train_geo_vocab,
            priors=train_priors,
            vocab_size=cfg.vocab_size,
            selector=cfg.selector,
            min_total_count=cfg.min_total_count,
        )

        locs = list(train_priors.keys())  # includes UNKNOWN
        loc_params = precompute_location_params(
            train_geo_vocab,
            vocab=vocab,
            alpha_word=cfg.alpha_word,
            alpha_sub=cfg.alpha_sub,
            alpha_hour=cfg.alpha_hour,
            locations=locs,
        )

        # Validate (streaming)
        try:
            vm = stream_evaluate_split(
                "valid", val_users_set, user_to_geo, VOCAB_FILE, subs_by_user, hours_by_user,
                loc_params, train_priors, vocab, train_locations_set
            )
        except KeyboardInterrupt:
            # If we were interrupted mid-eval, do not write a "completed" checkpoint for this cfg.
            log("\n[interrupt] stopped during evaluation; previous checkpoints are preserved.", 1, stream=sys.stderr)
            break

        elapsed = time.time() - t0
        log(f"[val] n={vm.n:,} top1={vm.top1_acc:.4f} top5={vm.top5_acc:.4f} logloss={vm.log_loss:.4f} "
            f"mrr={vm.mrr:.4f} unseen_users={vm.unseen_true_users:,} unseen_locs={vm.unseen_true_locations:,} "
            f"(elapsed {elapsed/60:.1f} min)", 1)

        # Save candidate model (optional) + metrics checkpoint.
        if SAVE_CANDIDATE_MODELS:
            candidate_model = SavedModel(
                config=cfg,
                vocab=vocab,
                priors=train_priors,
                loc_params=loc_params,
                locations=locs,
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

    # Select best by top1 accuracy, tie-break by lower logloss
    # Selection criterion for imbalanced labels: Use macro-F1 on top-1 predictions to give each location equal weight, mitigating class imbalance.
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
        f"\n[best] {best_cfg} (val f1_macro@1={best.val_metrics.top1_f1_macro:.4f}, top1={best.val_metrics.top1_acc:.4f}, top5={best.val_metrics.top5_acc:.4f}, logloss={best.val_metrics.log_loss:.4f})",
        1,
    )

    best_vocab = select_vocabulary(
        train_geo_vocab,
        priors=train_priors,
        vocab_size=best_cfg.vocab_size,
        selector=best_cfg.selector,
        min_total_count=best_cfg.min_total_count,
    )
    best_locs = list(train_priors.keys())  # includes UNKNOWN
    best_loc_params = precompute_location_params(
        train_geo_vocab,
        vocab=best_vocab,
        alpha_word=best_cfg.alpha_word,
        alpha_sub=best_cfg.alpha_sub,
        alpha_hour=best_cfg.alpha_hour,
        locations=best_locs,
    )

    tm = stream_evaluate_split(
        "test", test_users_set, user_to_geo, VOCAB_FILE, subs_by_user, hours_by_user,
        best_loc_params, train_priors, best_vocab, train_locations_set
    )
    log(
        f"\n[test] n={tm.n:,} "
        f"hit@1={tm.top1_acc:.4f} hit@5={tm.top5_acc:.4f} hit@10={tm.top10_acc:.4f} "
        f"mrr={tm.mrr:.4f} logloss={tm.log_loss:.4f}\n"
        f"       top1 micro P/R/F1={tm.top1_precision_micro:.4f}/{tm.top1_recall_micro:.4f}/{tm.top1_f1_micro:.4f} "
        f"| macro P/R/F1={tm.top1_precision_macro:.4f}/{tm.top1_recall_macro:.4f}/{tm.top1_f1_macro:.4f}\n"
        f"       top{PRF_TOPK} micro P/R/F1={tm.topk_precision_micro:.4f}/{tm.topk_recall_micro:.4f}/{tm.topk_f1_micro:.4f} "
        f"| macro P/R/F1={tm.topk_precision_macro:.4f}/{tm.topk_recall_macro:.4f}/{tm.topk_f1_macro:.4f}\n"
        f"       unseen true users={tm.unseen_true_users:,} (locations unseen in train={tm.unseen_true_locations:,}; mapped to {UNKNOWN_LABEL})",
        1,
    )

    if SAVE_MODEL:
        model = SavedModel(
            config=best_cfg,
            vocab=best_vocab,
            priors=train_priors,
            loc_params=best_loc_params,
            locations=best_locs,
        )
        save_model(model, MODEL_SAVE_PATH)

    log("\n[done]", 1)

if __name__ == "__main__":
    main()
