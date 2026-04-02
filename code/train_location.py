from __future__ import annotations
import csv
import json
import math
import os
import pickle
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable

from utils import prepare_splits

# -------------------------
# Optional fast JSON
# -------------------------
try:
    import orjson as _fastjson  # type: ignore

    def _json_loads(b: bytes):
        return _fastjson.loads(b)
except Exception:
    def _json_loads(b: bytes):
        return json.loads(b)

# -------------------------
# Optional Torch (GPU)
# -------------------------
_TORCH_AVAILABLE = False
try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:
    torch = None  # type: ignore

# -------------------------
# Output control
# -------------------------
VERBOSITY = 1  # 0=quiet, 1=progress, 2=verbose

def log(msg: str, level: int = 1, stream=None):
    if level <= VERBOSITY:
        print(msg, file=stream)

UNKNOWN_LABEL = "__UNKNOWN__"
NON_US_LABEL = "NON_US"
PRF_TOPK = 5

# -------------------------
# Paths / configuration
# -------------------------
# For this task, global labels file is usually the right choice, because we want
# US-state labels + NON_US + UNKNOWN.
loc_type = "global"  # options: US, non-US, global

CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_PATH = PROJECT_ROOT / "models"

SUBS_JSONL = os.path.join(DATA_DIR, "data_reddit_location", "subreddit_counts.jsonl")
HOURS_JSONL = os.path.join(DATA_DIR, "data_reddit_location", "hour_counts.jsonl")

# -------------------------
# Word feature source control
# -------------------------
WORD_FEATURE_SRC = os.environ.get("WORD_FEATURE_SRC", "comments").strip().lower()  # comments, submissions, all

if WORD_FEATURE_SRC not in {"comments", "submissions", "all"}:
    raise ValueError("WORD_FEATURE_SRC must be one of: comments, submissions, all")

VOCAB_FILE_COMMENTS = os.path.join(DATA_DIR, "data_reddit_location", "vocab_counts_comments.jsonl")
VOCAB_FILE_SUBMISSIONS = os.path.join(DATA_DIR, "data_reddit_location", "vocab_counts_submissions.jsonl")

SAVE_MODEL = False
MODEL_SAVE_PATH = os.path.join(MODEL_PATH, "label_location", "best_model_us_state_nonus.pkl")

# -------------------------
# Runtime knobs
# -------------------------
GPU_BATCH_SIZE = int(os.environ.get("GPU_BATCH_SIZE", "512"))
CONFIG_INDEX = int(os.environ.get("CONFIG_INDEX", "0"))
RUN_GPU_CPU_SANITY = os.environ.get("RUN_GPU_CPU_SANITY", "1") == "1"
GPU_CPU_SANITY_N = int(os.environ.get("GPU_CPU_SANITY_N", "100"))
SCORER_DTYPE = os.environ.get("SCORER_DTYPE", "float64").lower()
TRAIN_EVAL_LIMIT = int(os.environ.get("TRAIN_EVAL_LIMIT", "0"))  # 0 => full train eval
PRIOR_TEMPERATURE = float(os.environ.get("PRIOR_TEMPERATURE","1.0"))
FEATURE_DROPOUT = float(os.environ.get("FEATURE_DROPOUT", "0.0"))
DROPOUT_WORD_ONLY = os.environ.get("DROPOUT_WORD_ONLY", "1") == "1"
RNG_SEED = int(os.environ.get("RNG_SEED", "1337"))
_rng = random.Random(RNG_SEED)

if SCORER_DTYPE not in {"float64", "float32"}:
    raise ValueError("SCORER_DTYPE must be 'float64' or 'float32'")

# -------------------------
# Per-user feature normalization
# -------------------------
# Each user contributes fixed mass per feature family, regardless of verbosity.
WORD_MASS = float(os.environ.get("WORD_MASS", "0.3")) # noisier and more numerous, lower weight
SUB_MASS = float(os.environ.get("SUB_MASS", "1.5")) # highly informative for location
HOUR_MASS = float(os.environ.get("HOUR_MASS", "2.0")) # informative, yet fewer bins

# Feature normalization; Choices: "log1p_l1", "l1", "binary_l1"
USER_FEATURE_NORMALIZATION = os.environ.get("USER_FEATURE_NORMALIZATION", "binary_l1").lower()
if USER_FEATURE_NORMALIZATION not in {"log1p_l1", "l1", "binary_l1"}:
    raise ValueError("USER_FEATURE_NORMALIZATION must be one of: log1p_l1, l1, binary_l1")

# -------------------------
# Feature prefixing
# -------------------------
PREFIX_WORD = "w:"
PREFIX_SUB = "s:"
PREFIX_HOUR = "h:"

PRF_TOPK = 5


# -------------------------
# Configs
# -------------------------
@dataclass(frozen=True)
class Config:
    vocab_size: int
    selector: str
    min_total_count: float
    alpha_word: float
    alpha_sub: float
    alpha_hour: float

configs = [
    # 0. Safer default after DE/WY sink behavior: weaker smoothing, empirical priors, broad vocab
    Config(
        alpha_word=0.5,
        alpha_sub=1.0,
        alpha_hour=3.0,
        vocab_size=20000,
        selector="freq",
        min_total_count=20.0,
    ),

    # 1. Same smoothing, MI with controlled backfill
    Config(
        alpha_word=0.5,
        alpha_sub=1.0,
        alpha_hour=3.0,
        vocab_size=20000,
        selector="mi",
        min_total_count=20.0,
    ),

    # 2. Stronger smoothing frequency baseline
    Config(
        alpha_word=1.0,
        alpha_sub=2.0,
        alpha_hour=5.0,
        vocab_size=15000,
        selector="freq",
        min_total_count=20.0,
    ),

    # 3. Stronger smoothing MI alternative
    Config(
        alpha_word=1.0,
        alpha_sub=2.0,
        alpha_hour=5.0,
        vocab_size=15000,
        selector="mi",
        min_total_count=20.0,
    ),
]

# -------------------------
# CSV / label helpers
# -------------------------

US_OTHER_LABEL = "US_OTHER"
UNKNOWN_LABEL = "__UNKNOWN__"
NON_US_LABEL = "NON_US"

US_STATE_TO_CODE = {
    "alabama": "AL",
    "alaska": "AK",
    "arizona": "AZ",
    "arkansas": "AR",
    "california": "CA",
    "colorado": "CO",
    "connecticut": "CT",
    "delaware": "DE",
    "district of columbia": "DC",
    "florida": "FL",
    "georgia": "GA",
    "hawaii": "HI",
    "idaho": "ID",
    "illinois": "IL",
    "indiana": "IN",
    "iowa": "IA",
    "kansas": "KS",
    "kentucky": "KY",
    "louisiana": "LA",
    "maine": "ME",
    "maryland": "MD",
    "massachusetts": "MA",
    "michigan": "MI",
    "minnesota": "MN",
    "mississippi": "MS",
    "missouri": "MO",
    "montana": "MT",
    "nebraska": "NE",
    "nevada": "NV",
    "new hampshire": "NH",
    "new jersey": "NJ",
    "new mexico": "NM",
    "new york": "NY",
    "north carolina": "NC",
    "north dakota": "ND",
    "ohio": "OH",
    "oklahoma": "OK",
    "oregon": "OR",
    "pennsylvania": "PA",
    "rhode island": "RI",
    "south carolina": "SC",
    "south dakota": "SD",
    "tennessee": "TN",
    "texas": "TX",
    "utah": "UT",
    "vermont": "VT",
    "virginia": "VA",
    "washington": "WA",
    "west virginia": "WV",
    "wisconsin": "WI",
    "wyoming": "WY",
}
US_CODE_TO_LABEL = {v: f"US_{v}" for v in US_STATE_TO_CODE.values()}

def _merge_vocab_dicts(
    base: Dict[str, Dict[str, int]],
    other: Dict[str, Dict[str, int]],
) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {uid: dict(vc) for uid, vc in base.items()}

    for uid, vc in other.items():
        if uid not in out:
            out[uid] = dict(vc)
            continue
        tgt = out[uid]
        for w, c in vc.items():
            tgt[w] = tgt.get(w, 0) + c

    return out


def load_vocab_counts_by_source(
    users_set: set,
    word_feature_src: str,
) -> Dict[str, Dict[str, int]]:
    if word_feature_src == "comments":
        log("[vocab] source=comments", 1)
        return load_vocab_counts_for_users(VOCAB_FILE_COMMENTS, users_set)

    if word_feature_src == "submissions":
        log("[vocab] source=submissions", 1)
        return load_vocab_counts_for_users(VOCAB_FILE_SUBMISSIONS, users_set)

    if word_feature_src == "all":
        log("[vocab] source=all (comments + submissions)", 1)
        vocab_comments = load_vocab_counts_for_users(VOCAB_FILE_COMMENTS, users_set)
        vocab_submissions = load_vocab_counts_for_users(VOCAB_FILE_SUBMISSIONS, users_set)
        return _merge_vocab_dicts(vocab_comments, vocab_submissions)

    raise ValueError("word_feature_src must be one of: comments, submissions, all")

def load_user_to_label(labels_csv=None):
    """
    Uses the same simple logic as the verified counting script,
    but reads the file as tab-delimited.
    """
    user_to_label: Dict[str, str] = {}
    state_counts = {}
    us_other = 0
    non_us = 0
    skipped = 0

    with open(labels_csv, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        for idx, line in enumerate(reader):
            if idx:
                if idx % 50000 == 0:
                     log(f"[labels] read {idx:,} rows", 1)
                uid = line[1].strip()
                state = line[3].strip().lower()
                if state in US_STATE_TO_CODE:
                    user_to_label[uid] = US_STATE_TO_CODE[state]
                    if state in state_counts:
                        state_counts[US_STATE_TO_CODE[state]] += 1
                    else:
                        state_counts[US_STATE_TO_CODE[state]] = 1
                elif "United States" in line[4].strip():
                    user_to_label[uid] = US_OTHER_LABEL
                    us_other += 1
                else:
                    user_to_label[uid] = NON_US_LABEL
                    non_us += 1

    log(f"[labels] total labeled users: {len(user_to_label):,}", 1)
    log(
        f"[labels] mapped counts: "
        f"US_state={sum(state_counts.values()):,} "
        f"US_other={us_other:,} "
        f"NON_US={non_us:,} "
        f"skipped={skipped:,}",
        1,
    )

    return user_to_label

def summarize_split(name: str, users: List[str], labels: List[str]):
    c = Counter(labels)
    log(f"\n{name} split:", 1)
    log(f"  users: {len(users):,}", 1)
    log(f"  locations: {len(c):,}", 1)

def print_label_count_diagnostics(train_labels: List[str], topn: int = 15):
    c = Counter(train_labels)
    counts = sorted(c.values())
    if not counts:
        log("[diag] no train labels found", 1)
        return

    def q(p: float) -> int:
        idx = int(p * (len(counts) - 1))
        return counts[idx]

    log("\n[diag] train label-count diagnostics:", 1)
    log(f"  total locations: {len(c):,}", 1)
    log(f"  min users/location: {counts[0]:,}", 1)
    log(f"  median users/location: {q(0.50):,}", 1)
    log(f"  p90 users/location: {q(0.90):,}", 1)
    log(f"  p95 users/location: {q(0.95):,}", 1)
    log(f"  p99 users/location: {q(0.99):,}", 1)
    log(f"  max users/location: {counts[-1]:,}", 1)
    log(f"  locations with 1 user: {sum(v == 1 for v in counts):,}", 1)
    log(f"  locations with <=2 users: {sum(v <= 2 for v in counts):,}", 1)
    log(f"  locations with <=5 users: {sum(v <= 5 for v in counts):,}", 1)

    most_common = c.most_common(topn)
    log(f"  top {min(topn, len(most_common))} locations by train users:", 1)
    for lab, n in most_common:
        log(f"    {lab}: {n:,}", 1)

def print_majority_baseline(train_labels: List[str], eval_labels: List[str], split_name: str):
    c = Counter(train_labels)
    if not c or not eval_labels:
        return
    majority_label, majority_n = c.most_common(1)[0]
    acc = sum(1 for y in eval_labels if y == majority_label) / len(eval_labels)
    log(
        f"[baseline:{split_name}] majority_label={majority_label} "
        f"train_count={majority_n:,} acc={acc:.4f}",
        1,
    )

# -------------------------
# Load feature files
# -------------------------
def load_subreddit_counts(path: str) -> Dict[str, Dict[str, int]]:
    log("[subs] loading subreddit counts jsonl", 1)
    subs: Dict[str, Dict[str, int]] = {}
    with open(path, "rb") as f:
        for i, line in enumerate(f):
            obj = _json_loads(line)
            uid = (obj.get("author") or "").strip().lower()
            if not uid:
                continue
            raw = obj.get("subreddit_counts") or {}
            norm: Dict[str, int] = {}
            if isinstance(raw, dict):
                for k, v in raw.items():
                    try:
                        iv = int(v)
                    except Exception:
                        continue
                    if iv > 0:
                        norm[str(k)] = iv
            subs[uid] = norm
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

def load_vocab_counts_for_users(vocab_jsonl: str, users_set: set) -> Dict[str, Dict[str, int]]:
    log(f"[vocab] preloading vocab for {len(users_set):,} users from {vocab_jsonl}", 1)
    out: Dict[str, Dict[str, int]] = {}
    with open(vocab_jsonl, "rb") as f:
        for i, line in enumerate(f):
            obj = _json_loads(line)
            uid = (obj.get("author") or "").strip().lower()
            if uid in users_set:
                vc = obj.get("vocab") or {}
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

# -------------------------
# Per-user normalized feature vector
# -------------------------
def _normalize_family_counts(raw: Dict[str, int], prefix: str, family_mass: float) -> Dict[str, float]:
    if family_mass <= 0 or not raw:
        return {}

    transformed: Dict[str, float] = {}
    total = 0.0

    for k, v in raw.items():
        fv = float(v)
        if fv <= 0:
            continue

        if USER_FEATURE_NORMALIZATION == "log1p_l1":
            tv = math.log1p(fv)
        elif USER_FEATURE_NORMALIZATION == "l1":
            tv = fv
        elif USER_FEATURE_NORMALIZATION == "binary_l1":
            tv = 1.0
        else:
            raise ValueError(f"Unknown USER_FEATURE_NORMALIZATION={USER_FEATURE_NORMALIZATION}")

        if tv <= 0:
            continue

        transformed[f"{prefix}{k}"] = tv
        total += tv

    if total <= 0:
        return {}

    scale = family_mass / total
    return {feat: val * scale for feat, val in transformed.items()}

def _apply_feature_dropout(vec: Dict[str, float], p: float) -> Dict[str, float]:
    if p <= 0.0 or not vec:
        return vec

    kept = {feat: val for feat, val in vec.items() if _rng.random() >= p}

    # Keep at least one feature if the original vector was non-empty.
    if not kept:
        feat = _rng.choice(list(vec.keys()))
        kept = {feat: vec[feat]}

    # Renormalize so the family keeps the same total mass.
    orig_total = sum(vec.values())
    kept_total = sum(kept.values())
    if orig_total > 0.0 and kept_total > 0.0:
        scale = orig_total / kept_total
        kept = {feat: val * scale for feat, val in kept.items()}

    return kept

def build_user_feature_vector(
    uid: str,
    subs_by_user: Dict[str, Dict[str, int]],
    hours_by_user: Dict[str, Dict[str, int]],
    vocab_by_user: Dict[str, Dict[str, int]],
    apply_dropout: bool = False,
) -> Dict[str, float]:
    out: Dict[str, float] = {}

    # Words
    raw_words = vocab_by_user.get(uid) or {}
    word_vec = _normalize_family_counts(raw_words, PREFIX_WORD, WORD_MASS)
    if apply_dropout and FEATURE_DROPOUT > 0.0:
        word_vec = _apply_feature_dropout(word_vec, FEATURE_DROPOUT)
    out.update(word_vec)

    # Subreddits
    raw_subs = subs_by_user.get(uid) or {}
    sub_vec = _normalize_family_counts(raw_subs, PREFIX_SUB, SUB_MASS)
    if apply_dropout and FEATURE_DROPOUT > 0.0 and not DROPOUT_WORD_ONLY:
        sub_vec = _apply_feature_dropout(sub_vec, FEATURE_DROPOUT)
    out.update(sub_vec)

    # Hours
    raw_hours = hours_by_user.get(uid) or {}
    norm_hours_raw: Dict[str, int] = {f"{int(k):02d}": int(v) for k, v in raw_hours.items() if int(v) > 0}
    hour_vec = _normalize_family_counts(norm_hours_raw, PREFIX_HOUR, HOUR_MASS)
    if apply_dropout and FEATURE_DROPOUT > 0.0 and not DROPOUT_WORD_ONLY:
        hour_vec = _apply_feature_dropout(hour_vec, FEATURE_DROPOUT)
    out.update(hour_vec)

    return out

# -------------------------
# Build per-label TRAIN counts
# -------------------------
def build_train_label_vocab_from_preloaded(
    train_users: Iterable[str],
    user_to_label: Dict[str, str],
    subs_by_user: Dict[str, Dict[str, int]],
    hours_by_user: Dict[str, Dict[str, int]],
    vocab_by_user: Dict[str, Dict[str, int]],
) -> Dict[str, Counter]:
    label_vocab: Dict[str, Counter] = defaultdict(Counter)
    for uid in train_users:
        lab = user_to_label.get(uid)
        if not lab:
            continue
        feats = build_user_feature_vector(
        uid,
        subs_by_user,
        hours_by_user,
        vocab_by_user,
        apply_dropout=True,
        )
        label_vocab[lab].update(feats)
    label_vocab[UNKNOWN_LABEL] = Counter(label_vocab.get(UNKNOWN_LABEL, Counter()))
    log(f"[train] locations (seen): {len(label_vocab)-1:,} (+UNKNOWN)", 1)
    return label_vocab

def compute_location_priors_smoothed_from_labels(
    train_labels: List[str],
    train_locations: List[str],
    kappa: float = 0.5,
    temperature: float = 1.0,
) -> Dict[str, float]:
    """
    temperature=1.0  -> empirical-count prior
    temperature<1.0  -> flatter prior
    temperature=0.0  -> uniform prior over seen labels (approximately)
    """
    counts = Counter(train_labels)

    raw = {}
    for gh in train_locations:
        raw_count = float(counts.get(gh, 0) + kappa)
        raw[gh] = raw_count ** temperature

    raw_unknown = float(counts.get(UNKNOWN_LABEL, 0) + kappa) ** temperature
    raw[UNKNOWN_LABEL] = raw_unknown

    z = sum(raw.values())
    if z <= 0:
        uniform = 1.0 / max(len(raw), 1)
        return {gh: uniform for gh in raw}

    return {gh: val / z for gh, val in raw.items()}

# -------------------------
# Feature selection
# -------------------------
def mutual_information_scores(
    geo_vocab: Dict[str, Counter],
    priors: Dict[str, float],
    min_total_count: float,
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

    global_word: Counter = Counter()
    postings: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

    n_locs = len(locs)
    progress_every = max(1, n_locs // 20)

    for i, gh in enumerate(locs, start=1):
        wc = geo_vocab[gh]
        if not wc:
            continue
        global_word.update(wc)
        for w, c in wc.items():
            if c > 0:
                postings[w].append((gh, float(c)))
        if VERBOSITY >= 1 and (i % progress_every == 0 or i == n_locs):
            log(f"[mi] indexed {i:,}/{n_locs:,} locations | unique features so far: {len(global_word):,}", 1)

    mi: Dict[str, float] = {}
    eligible = sum(1 for _, c in global_word.items() if c >= min_total_count)
    done = 0
    progress_every_feat = max(1, eligible // 20) if eligible > 0 else 1

    for w, global_c in global_word.items():
        if global_c < min_total_count:
            continue
        pw = global_c / total_all
        score = 0.0
        for l, c_lw in postings[w]:
            p_lw = c_lw / total_all
            score += p_lw * math.log(p_lw / (pL[l] * pw))
        mi[w] = score
        done += 1
        if VERBOSITY >= 1 and (done % progress_every_feat == 0 or done == eligible):
            log(f"[mi] scored {done:,}/{eligible:,} eligible features", 1)

    return mi

def select_vocabulary(
    geo_vocab: Dict[str, Counter],
    priors: Dict[str, float],
    vocab_size: int,
    selector: str,
    min_total_count: float,
) -> List[str]:
    global_word = Counter()
    for c in geo_vocab.values():
        global_word.update(c)

    if selector == "freq":
        return [w for w, _ in global_word.most_common(vocab_size)]

    if selector == "mi":
        log("[train] computing sparse MI scores", 1)
        mi = mutual_information_scores(geo_vocab, priors, min_total_count=min_total_count)
        top_mi = [w for w, _ in sorted(mi.items(), key=lambda kv: kv[1], reverse=True)]

        if len(top_mi) >= vocab_size:
            return top_mi[:vocab_size]

        log(f"[train] MI produced only {len(top_mi):,} features; backfilling with freq", 1)
        chosen = list(top_mi)
        seen = set(chosen)
        for w, _ in global_word.most_common():
            if w not in seen:
                chosen.append(w)
                seen.add(w)
            if len(chosen) >= vocab_size:
                break
        return chosen[:vocab_size]

    raise ValueError("selector must be 'mi' or 'freq'")

# -------------------------
# Sparse location params + exact CPU scorer
# -------------------------
@dataclass
class LocationParams:
    alpha_sum: float
    const: float
    alpha: Dict[str, float]
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
        alpha_sparse: Dict[str, float] = {}
        alpha_sum = base_sum
        sum_lgamma_alpha = sum_lgamma_base

        for feat, c in wc.items():
            fc = float(c)
            if fc <= 0 or feat not in vocab_set:
                continue
            b = base_by_feat.get(feat)
            if b is None:
                b = _base_alpha_for_feat(feat, alpha_word, alpha_sub, alpha_hour)
            a = b + fc
            alpha_sparse[feat] = a
            alpha_sum += fc
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

def dm_loglik_sparse(counts: Dict[str, float], lp: LocationParams, vocab_set: set) -> Tuple[float, float]:
    x_used = {w: c for w, c in counts.items() if c > 0 and w in vocab_set}
    N = sum(x_used.values())
    if N <= 0:
        return 0.0, 0.0
    s = lp.const - math.lgamma(lp.alpha_sum + N)
    for w, c in x_used.items():
        b = _base_alpha_for_feat(w, lp.alpha_word, lp.alpha_sub, lp.alpha_hour)
        a = lp.alpha.get(w, b)
        s += math.lgamma(a + c)
    return s, N

# -------------------------
# Exact GPU scorer
# -------------------------
class TorchBatchedScorer:
    """
    Exact batched Dirichlet-Multinomial scorer with real-valued normalized counts.

    score(u,l) = log_prior(l) + const(l) - lgamma(alpha_sum(l)+N_u)
                 + sum_f lgamma(alpha(l,f)+c_u,f)

    with:
      const(l) = lgamma(alpha_sum(l)) - sum_f lgamma(alpha(l,f))
    """
    def __init__(self, locations: List[str], priors: Dict[str, float], loc_params: Dict[str, LocationParams], vocab_set: set):
        if not _TORCH_AVAILABLE:
            raise RuntimeError("Torch not available")

        self.locations = list(locations)
        self.vocab_set = set(vocab_set)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.L = len(self.locations)

        self.dtype = torch.float64 if SCORER_DTYPE == "float64" else torch.float32
        prior_floor = 1e-300

        self.log_priors = torch.tensor(
            [math.log(priors.get(gh, prior_floor)) for gh in self.locations],
            device=self.device, dtype=self.dtype
        )
        self.consts = torch.tensor(
            [loc_params[gh].const for gh in self.locations],
            device=self.device, dtype=self.dtype
        )
        self.alpha_sums = torch.tensor(
            [loc_params[gh].alpha_sum for gh in self.locations],
            device=self.device, dtype=self.dtype
        )

        sample_lp = loc_params[self.locations[0]]
        self.alpha_word = float(sample_lp.alpha_word)
        self.alpha_sub = float(sample_lp.alpha_sub)
        self.alpha_hour = float(sample_lp.alpha_hour)

        feat_to_locidx: Dict[str, List[int]] = defaultdict(list)
        feat_to_alpha: Dict[str, List[float]] = defaultdict(list)
        for li, gh in enumerate(self.locations):
            lp = loc_params[gh]
            for feat, a in lp.alpha.items():
                feat_to_locidx[feat].append(li)
                feat_to_alpha[feat].append(float(a))

        self.override_locidx: Dict[str, torch.Tensor] = {}
        self.override_alpha: Dict[str, torch.Tensor] = {}
        for feat, idxs in feat_to_locidx.items():
            self.override_locidx[feat] = torch.tensor(idxs, device=self.device, dtype=torch.long)
            self.override_alpha[feat] = torch.tensor(feat_to_alpha[feat], device=self.device, dtype=self.dtype)

        log(
            f"[gpu] device={self.device} dtype={SCORER_DTYPE} "
            f"locations={self.L:,} vocab={len(self.vocab_set):,} override_feats={len(self.override_locidx):,}",
            1,
        )

    def _base_alpha_feat(self, feat: str) -> float:
        if feat.startswith(PREFIX_WORD):
            return self.alpha_word
        if feat.startswith(PREFIX_SUB):
            return self.alpha_sub
        if feat.startswith(PREFIX_HOUR):
            return self.alpha_hour
        return self.alpha_word

    @torch.no_grad()
    def score_full_matrix(self, batch_counts: List[Dict[str, float]]) -> torch.Tensor:
        B = len(batch_counts)
        L = self.L
        device = self.device
        dtype = self.dtype

        Ns = torch.zeros(B, device=device, dtype=dtype)
        global_add = torch.zeros(B, device=device, dtype=dtype)

        feat_users: Dict[str, List[int]] = defaultdict(list)
        feat_counts: Dict[str, List[float]] = defaultdict(list)

        for ui, counts in enumerate(batch_counts):
            n_ui = 0.0
            add_ui = 0.0
            for feat, c in counts.items():
                if feat not in self.vocab_set:
                    continue
                fc = float(c)
                if fc <= 0:
                    continue

                n_ui += fc
                base = self._base_alpha_feat(feat)
                add_ui += math.lgamma(base + fc)

                if feat in self.override_locidx:
                    feat_users[feat].append(ui)
                    feat_counts[feat].append(fc)

            Ns[ui] = float(n_ui)
            global_add[ui] = float(add_ui)

        scores = self.log_priors.unsqueeze(0).expand(B, L).clone()
        scores += self.consts.unsqueeze(0)
        scores -= torch.lgamma(self.alpha_sums.unsqueeze(0) + Ns.unsqueeze(1))
        scores += global_add.unsqueeze(1)

        flat_scores = scores.reshape(-1)
        THRESH_OUTER = 2_500_000

        for feat, users in feat_users.items():
            u_idx = torch.tensor(users, device=device, dtype=torch.long)
            c_vec = torch.tensor(feat_counts[feat], device=device, dtype=dtype)
            loc_idx = self.override_locidx[feat]
            a_loc = self.override_alpha[feat]

            U = int(u_idx.numel())
            K = int(loc_idx.numel())
            if U == 0 or K == 0:
                continue

            base = self._base_alpha_feat(feat)
            base_scalar = torch.tensor(base, dtype=dtype, device=device)
            base_term = torch.lgamma(base_scalar + c_vec)

            if U * K <= THRESH_OUTER:
                corr = torch.lgamma(a_loc.unsqueeze(0) + c_vec.unsqueeze(1)) - base_term.unsqueeze(1)
                rows = u_idx.repeat_interleave(K)
                cols = loc_idx.repeat(U)
                flat_idx = rows * L + cols
                flat_scores.index_add_(0, flat_idx, corr.reshape(-1))
            else:
                chunk = max(1, THRESH_OUTER // max(K, 1))
                for cs in range(0, U, chunk):
                    uu = u_idx[cs:cs + chunk]
                    cc = c_vec[cs:cs + chunk]
                    bt = base_term[cs:cs + chunk]
                    corr = torch.lgamma(a_loc.unsqueeze(0) + cc.unsqueeze(1)) - bt.unsqueeze(1)
                    rows = uu.repeat_interleave(K)
                    cols = loc_idx.repeat(int(uu.numel()))
                    flat_idx = rows * L + cols
                    flat_scores.index_add_(0, flat_idx, corr.reshape(-1))

        return flat_scores.view(B, L)

# -------------------------
# Metrics
# -------------------------
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

# -------------------------
# Build labeled examples
# -------------------------
def make_labeled_examples(
    users: List[str],
    user_to_label: Dict[str, str],
    train_locations_set: set,
    subs_by_user: Dict[str, Dict[str, int]],
    hours_by_user: Dict[str, Dict[str, int]],
    vocab_by_user: Dict[str, Dict[str, int]],
) -> List[Tuple[str, Dict[str, float]]]:
    out: List[Tuple[str, Dict[str, float]]] = []
    for uid in users:
        true_lab = user_to_label.get(uid)
        if not true_lab:
            continue
        true_eval = true_lab if true_lab in train_locations_set else UNKNOWN_LABEL
        counts = build_user_feature_vector(
        uid,
        subs_by_user,
        hours_by_user,
        vocab_by_user,
        apply_dropout=False,
        )
        out.append((true_eval, counts))
    return out

# -------------------------
# Exact CPU scoring for sanity check
# -------------------------
def score_cpu_full(
    counts: Dict[str, float],
    loc_params: Dict[str, LocationParams],
    priors: Dict[str, float],
    vocab_set: set,
    locations: List[str],
) -> List[Tuple[str, float]]:
    prior_floor = 1e-300
    scored = []
    for gh in locations:
        lp = loc_params[gh]
        ll, _ = dm_loglik_sparse(counts, lp, vocab_set)
        scored.append((gh, math.log(priors.get(gh, prior_floor)) + ll))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored

def run_gpu_cpu_sanity_check(
    labeled_users: List[Tuple[str, Dict[str, float]]],
    scorer: TorchBatchedScorer,
    loc_params: Dict[str, LocationParams],
    priors: Dict[str, float],
    vocab: List[str],
    locations: List[str],
    n_check: int,
):
    if not labeled_users:
        log("[sanity] no users available", 1)
        return

    subset = labeled_users[:min(len(labeled_users), n_check)]
    vocab_set = set(vocab)
    batch_counts = [c for _, c in subset]

    gpu_scores = scorer.score_full_matrix(batch_counts).detach().cpu()
    top1_agree = 0
    max_abs_diff = 0.0
    mean_abs_diff_sum = 0.0
    mean_abs_diff_n = 0

    for i, (_, counts) in enumerate(subset):
        cpu_ranked = score_cpu_full(counts, loc_params, priors, vocab_set, locations)
        cpu_map = dict(cpu_ranked)
        cpu_vec = torch.tensor([cpu_map[gh] for gh in locations], dtype=torch.float64)
        gpu_vec = gpu_scores[i].to(torch.float64)

        diff = (gpu_vec - cpu_vec).abs()
        max_abs_diff = max(max_abs_diff, float(diff.max().item()))
        mean_abs_diff_sum += float(diff.sum().item())
        mean_abs_diff_n += int(diff.numel())

        gpu_top1 = locations[int(torch.argmax(gpu_vec).item())]
        cpu_top1 = locations[int(torch.argmax(cpu_vec).item())]
        if gpu_top1 == cpu_top1:
            top1_agree += 1

    mean_abs_diff = mean_abs_diff_sum / max(mean_abs_diff_n, 1)
    log(
        f"[sanity] users={len(subset):,} top1_agreement={top1_agree}/{len(subset)}={top1_agree/max(len(subset),1):.6f} "
        f"mean_abs_diff={mean_abs_diff:.12f} max_abs_diff={max_abs_diff:.12f}",
        1,
    )

# -------------------------
# Exact evaluation using full GPU score matrices
# -------------------------
def evaluate_split_exact_gpu(
    split_name: str,
    labeled_users: List[Tuple[str, Dict[str, float]]],
    scorer: TorchBatchedScorer,
    vocab: List[str],
    locations: List[str],
    batch_size: int = 512,
) -> StreamMetrics:
    log(f"\n[{split_name}] exact evaluation on GPU", 1)
    vocab_set = set(vocab)

    label_to_idx = {gh: i for i, gh in enumerate(locations)}
    unknown_idx = label_to_idx.get(UNKNOWN_LABEL, None)

    unseen_true_users = sum(1 for t, _ in labeled_users if t == UNKNOWN_LABEL)
    unseen_true_locations = 1 if unseen_true_users > 0 else 0

    tp1 = defaultdict(int); fp1 = defaultdict(int); fn1 = defaultdict(int)
    tpk = defaultdict(int); fpk = defaultdict(int); fnk = defaultdict(int)

    hit1 = hit5 = hit10 = 0
    rr_sum = 0.0
    logloss_sum = 0.0
    invocab_tok_sum = 0.0
    zero_N = 0

    n = len(labeled_users)
    if n == 0:
        return StreamMetrics(0,0,0,0,0,float("inf"),0,0,0,0,0,0,0,0,0,0,0,0,unseen_true_users,unseen_true_locations)

    pred1_counter = Counter()

    for start in range(0, n, batch_size):
        batch = labeled_users[start:start + batch_size]
        batch_counts = [c for _, c in batch]
        batch_true = [t for t, _ in batch]

        for counts in batch_counts:
            inv = sum(v for feat, v in counts.items() if feat in vocab_set and v > 0)
            invocab_tok_sum += inv
            if inv <= 0:
                zero_N += 1

        scores = scorer.score_full_matrix(batch_counts)
        k = min(10, len(locations))
        _, idxs = torch.topk(scores, k=k, dim=1, largest=True, sorted=True)

        idxs_cpu = idxs.detach().cpu().tolist()
        lse = torch.logsumexp(scores, dim=1).detach().cpu().tolist()
        scores_cpu = scores.detach().cpu()

        for i, true_eval in enumerate(batch_true):
            pred_idxs = idxs_cpu[i]
            pred_labels = [locations[j] for j in pred_idxs]
            if pred_labels:
                pred1_counter[pred_labels[0]] += 1

            if pred_labels and pred_labels[0] == true_eval:
                hit1 += 1
            if true_eval in pred_labels[:5]:
                hit5 += 1
            if true_eval in pred_labels[:10]:
                hit10 += 1

            pred1 = pred_labels[0] if pred_labels else UNKNOWN_LABEL
            if pred1 == true_eval:
                tp1[true_eval] += 1
            else:
                fp1[pred1] += 1
                fn1[true_eval] += 1

            k_set = set(pred_labels[:PRF_TOPK])
            if true_eval in k_set:
                tpk[true_eval] += 1
            else:
                fnk[true_eval] += 1
            for lab in k_set:
                if lab != true_eval:
                    fpk[lab] += 1

            true_idx = label_to_idx.get(true_eval, unknown_idx)
            if true_idx is None:
                true_score = -1e30
            else:
                true_score = float(scores_cpu[i, true_idx].item())

            logloss_sum += -(true_score - lse[i])

            try:
                r = pred_labels.index(true_eval) + 1
                rr_sum += 1.0 / r
            except ValueError:
                pass

    log(f"[{split_name}] debug: zero_N={zero_N:,}/{n:,} ({zero_N/n:.3%}) mapped_unknown={unseen_true_users:,}/{n:,} ({unseen_true_users/n:.3%})", 1)
    log(f"[{split_name}] top predicted labels:", 1)
    for lab, cnt in pred1_counter.most_common(10):
        log(f"  {lab}: {cnt:,}", 1)

    prf1 = _prf_from_counts(tp1, fp1, fn1)
    prfk = _prf_from_counts(tpk, fpk, fnk)

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


def log_selected_vocab_diagnostics(vocab: List[str], requested_vocab_size: int):
    log(
        f"[debug] selected vocab size: {len(vocab):,} "
        f"(requested {requested_vocab_size:,})",
        1,
    )


def log_prior_examples(priors: Dict[str, float], labels: Optional[List[str]] = None):
    if labels is None:
        labels = ["DE", "WY", "CA", "NON_US", "US_OTHER"]
    log("[debug] example priors:", 1)
    for gh in labels:
        if gh in priors:
            log(f"  {gh}: {priors[gh]:.8f}", 1)


def log_smallest_alpha_sum_classes(loc_params: Dict[str, LocationParams], topn: int = 10):
    log(f"[debug] smallest alpha_sum classes (bottom {topn}):", 1)
    for gh, lp in sorted(loc_params.items(), key=lambda kv: kv[1].alpha_sum)[:topn]:
        log(f"  {gh}: alpha_sum={lp.alpha_sum:.6f}", 1)


# -------------------------
# Saved model I/O
# -------------------------
@dataclass
class SavedModel:
    config: Config
    vocab: List[str]
    priors: Dict[str, float]
    loc_params: Dict[str, LocationParams]
    locations: List[str]

def save_model(model: SavedModel, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    log(f"[saved] model -> {path}", 1)

# -------------------------
# Main
# -------------------------
def main():
    if not _TORCH_AVAILABLE:
        raise RuntimeError("Torch is required for this exact GPU evaluation script.")

    if loc_type == "US":
        label_file = "us_geohash.csv"
    elif loc_type == "non-US":
        label_file = "non_us_geohash.csv"
    elif loc_type == "global":
        label_file = "combined_geohash.csv"
    else:
        raise Exception("Wrong loc_type value. Choose from US, non-US and global.")

    labels_csv = os.path.join(DATA_DIR, "data_reddit_location", label_file)

    if not (0 <= CONFIG_INDEX < len(configs)):
        raise ValueError(f"CONFIG_INDEX must be in [0, {len(configs)-1}]")
    cfg = configs[CONFIG_INDEX]

    split_dir = os.path.join(MODEL_PATH, "train_location_data_split")
    log(f"[config] using CONFIG_INDEX={CONFIG_INDEX}: {cfg}", 1)
    log(f"[config] normalization={USER_FEATURE_NORMALIZATION} WORD_MASS={WORD_MASS} SUB_MASS={SUB_MASS} HOUR_MASS={HOUR_MASS}", 1)
    log(f"[config] WORD_FEATURE_SRC={WORD_FEATURE_SRC}", 1)
    log(f"[config] PRIOR_TEMPERATURE={PRIOR_TEMPERATURE}", 1)
    log(f"[config] split_dir={split_dir}", 1)
    log(
    f"[config] FEATURE_DROPOUT={FEATURE_DROPOUT} "
    f"DROPOUT_WORD_ONLY={DROPOUT_WORD_ONLY} RNG_SEED={RNG_SEED}",
    1,
    )

    t0_all = time.time()

    user_to_label = load_user_to_label(labels_csv)

    all_users = list(user_to_label.keys())
    all_labels = [user_to_label[u] for u in all_users]

    subs_by_user = load_subreddit_counts(SUBS_JSONL)

    if not os.path.exists(HOURS_JSONL):
        raise FileNotFoundError(f"Required hour feature file not found: {HOURS_JSONL}")
    hours_by_user = load_hour_counts(HOURS_JSONL)

    all_labeled_users_set = set(all_users)
    vocab_by_user = load_vocab_counts_by_source(all_labeled_users_set, WORD_FEATURE_SRC)

    vocab_users_set = set(vocab_by_user.keys())
    subs_users_set = set(subs_by_user.keys())
    hours_users_set = set(hours_by_user.keys())
    featured_users_set = all_labeled_users_set & vocab_users_set & subs_users_set & hours_users_set

    log(f"[debug] labeled users total: {len(all_labeled_users_set):,}", 1)
    log(
        f"[debug] feature coverage among labeled users: "
        f"vocab={len(all_labeled_users_set & vocab_users_set):,} "
        f"subs={len(all_labeled_users_set & subs_users_set):,} "
        f"hours={len(all_labeled_users_set & hours_users_set):,}",
        1,
    )
    log(f"[debug] users with labels + all three feature sets: {len(featured_users_set):,}", 1)
    log(f"[debug] dropped labeled users missing >=1 feature set: {len(all_labeled_users_set - featured_users_set):,}", 1)

    user_to_label = {u: lab for u, lab in user_to_label.items() if u in featured_users_set}
    all_users = list(user_to_label.keys())
    all_labels = [user_to_label[u] for u in all_users]

    train_users, train_labels, val_users, val_labels, test_users, test_labels = prepare_splits(
        all_users,
        all_labels,
        split_dir=split_dir,
        description=f"{loc_type} us_state_nonus_feature_aligned",
    )

    current_user_set = set(all_users)
    if not set(train_users).issubset(current_user_set):
        raise RuntimeError("Loaded train split contains users not present in current filtered dataset.")
    if not set(val_users).issubset(current_user_set):
        raise RuntimeError("Loaded validation split contains users not present in current filtered dataset.")
    if not set(test_users).issubset(current_user_set):
        raise RuntimeError("Loaded test split contains users not present in current filtered dataset.")
    split_total = len(train_users) + len(val_users) + len(test_users)
    if split_total != len(all_users):
        raise RuntimeError(f"Split size mismatch: split_total={split_total:,} current_users={len(all_users):,}")

    summarize_split("Train (aligned)", train_users, train_labels)
    summarize_split("Valid (aligned)", val_users, val_labels)
    summarize_split("Test (aligned)", test_users, test_labels)
    print_label_count_diagnostics(train_labels)
    print_majority_baseline(train_labels, train_labels, "train")
    print_majority_baseline(train_labels, val_labels, "valid")
    print_majority_baseline(train_labels, test_labels, "test")

    t0 = time.time()
    train_label_vocab = build_train_label_vocab_from_preloaded(
        train_users, user_to_label, subs_by_user, hours_by_user, vocab_by_user
    )
    train_locations = [gh for gh in train_label_vocab.keys() if gh != UNKNOWN_LABEL]
    train_locations_set = set(train_locations)
    train_priors = compute_location_priors_smoothed_from_labels(
    train_labels,
    train_locations,
    kappa=0.5,
    temperature=PRIOR_TEMPERATURE,
    )
    locs = list(train_locations)
    log_prior_examples(train_priors)
    log(f"[train] build_train_label_vocab elapsed {(time.time() - t0)/60:.2f} min", 1)

    t0 = time.time()
    log("[train] starting vocabulary selection", 1)
    vocab = select_vocabulary(
        train_label_vocab,
        priors=train_priors,
        vocab_size=cfg.vocab_size,
        selector=cfg.selector,
        min_total_count=cfg.min_total_count,
    )
    log_selected_vocab_diagnostics(vocab, cfg.vocab_size)
    log(f"[train] vocabulary selection elapsed {(time.time() - t0)/60:.2f} min", 1)

    t0 = time.time()
    log("[train] starting location param precompute", 1)
    loc_params = precompute_location_params_sparse(
        train_label_vocab,
        vocab=vocab,
        alpha_word=cfg.alpha_word,
        alpha_sub=cfg.alpha_sub,
        alpha_hour=cfg.alpha_hour,
        locations=locs,
    )
    log_smallest_alpha_sum_classes(loc_params)
    log(f"[train] location param precompute elapsed {(time.time() - t0)/60:.2f} min", 1)

    t0 = time.time()
    train_labeled = make_labeled_examples(
        train_users, user_to_label, train_locations_set, subs_by_user, hours_by_user, vocab_by_user
    )
    val_labeled = make_labeled_examples(
        val_users, user_to_label, train_locations_set, subs_by_user, hours_by_user, vocab_by_user
    )
    test_labeled = make_labeled_examples(
        test_users, user_to_label, train_locations_set, subs_by_user, hours_by_user, vocab_by_user
    )
    log(f"[train] labeled examples: {len(train_labeled):,}", 1)
    log(f"[valid] labeled examples: {len(val_labeled):,}", 1)
    log(f"[test] labeled examples: {len(test_labeled):,}", 1)
    log(f"[eval] build labeled examples elapsed {(time.time() - t0)/60:.2f} min", 1)

    scorer = TorchBatchedScorer(locs, train_priors, loc_params, set(vocab))

    if RUN_GPU_CPU_SANITY:
        run_gpu_cpu_sanity_check(
            labeled_users=val_labeled,
            scorer=scorer,
            loc_params=loc_params,
            priors=train_priors,
            vocab=vocab,
            locations=locs,
            n_check=GPU_CPU_SANITY_N,
        )

    train_eval_data = train_labeled if TRAIN_EVAL_LIMIT <= 0 else train_labeled[:TRAIN_EVAL_LIMIT]

    t0 = time.time()
    trm = evaluate_split_exact_gpu(
        "train",
        train_eval_data,
        scorer,
        vocab,
        locations=locs,
        batch_size=GPU_BATCH_SIZE,
    )
    log(
        f"[train] n={trm.n:,} top1={trm.top1_acc:.4f} top5={trm.top5_acc:.4f} top10={trm.top10_acc:.4f} "
        f"mrr={trm.mrr:.4f} logloss={trm.log_loss:.4f} "
        f"top1_f1_macro={trm.top1_f1_macro:.4f} "
        f"(elapsed {(time.time() - t0)/60:.2f} min)",
        1,
    )

    t0 = time.time()
    vm = evaluate_split_exact_gpu(
        "valid",
        val_labeled,
        scorer,
        vocab,
        locations=locs,
        batch_size=GPU_BATCH_SIZE,
    )
    log(
        f"[valid] n={vm.n:,} top1={vm.top1_acc:.4f} top5={vm.top5_acc:.4f} top10={vm.top10_acc:.4f} "
        f"mrr={vm.mrr:.4f} logloss={vm.log_loss:.4f} "
        f"top1_f1_macro={vm.top1_f1_macro:.4f} "
        f"(elapsed {(time.time() - t0)/60:.2f} min)",
        1,
    )

    t0 = time.time()
    tm = evaluate_split_exact_gpu(
        "test",
        test_labeled,
        scorer,
        vocab,
        locations=locs,
        batch_size=GPU_BATCH_SIZE,
    )
    log(
        f"\n[test] n={tm.n:,} "
        f"hit@1={tm.top1_acc:.4f} hit@5={tm.top5_acc:.4f} hit@10={tm.top10_acc:.4f} "
        f"mrr={tm.mrr:.4f} logloss={tm.log_loss:.4f}\n"
        f"       top1 micro P/R/F1={tm.top1_precision_micro:.4f}/{tm.top1_recall_micro:.4f}/{tm.top1_f1_micro:.4f} "
        f"| macro P/R/F1={tm.top1_precision_macro:.4f}/{tm.top1_recall_macro:.4f}/{tm.top1_f1_macro:.4f}\n"
        f"       top{PRF_TOPK} micro P/R/F1={tm.topk_precision_micro:.4f}/{tm.topk_recall_micro:.4f}/{tm.topk_f1_micro:.4f} "
        f"| macro P/R/F1={tm.topk_precision_macro:.4f}/{tm.topk_recall_macro:.4f}/{tm.topk_f1_macro:.4f}\n"
        f"       elapsed {(time.time() - t0)/60:.2f} min",
        1,
    )

    if SAVE_MODEL:
        model = SavedModel(
            config=cfg,
            vocab=vocab,
            priors=train_priors,
            loc_params=loc_params,
            locations=locs,
        )
        save_model(model, MODEL_SAVE_PATH)

    log(f"\n[done] total elapsed {(time.time() - t0_all)/60:.2f} min", 1)

if __name__ == "__main__":
    main()
