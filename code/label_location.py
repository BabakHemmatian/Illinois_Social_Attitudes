from __future__ import annotations

import csv
import json
import math
import os
import pickle
import re
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse

from cli import get_args, MODELS_DIR
from utils import (
    build_author_feature_map_from_raw_zst_with_seen,
    cache_get_locations,
    cache_put_locations,
    check_reqd_files,
    find_raw_month_files,
    init_location_cache,
    log_report,
    parse_range,
)


### Argument Handling

TOP_CONF_THRESHOLD = 0.60
REG_CONF_MARGIN = 0.10
STA_CONF_MARGIN = 0.05
UNKNOWN_LABEL = "UNK"

MIN_SAMPLES_FOR_INFERENCE = 5
MIN_SAMPLES_FOR_CACHE = 50

regional_weights = {"words": 0.7, "struct": 0.3}
top_weights = {"words": 0.55, "struct": 0.45}
state_weights = {"words": 0.5, "struct": 0.5}

RAW_START_YM = (2007, 1)
RAW_END_YM = (2023, 12)
DEFAULT_BATCH_SIZE = 256
PROGRESS_HEARTBEAT_SECONDS = 30 * 60

args = get_args()
type_ = args.type
years = parse_range(args.years)
if isinstance(years, int):
    years = [years]
group = args.group
max_items_per_author = getattr(args, "maxitems", 25)
max_files_to_scan = getattr(args, "maxfiles", 60)
max_radius = getattr(args, "maxradius", 30)
batch_size = max(1, int(getattr(args, "batchsize", DEFAULT_BATCH_SIZE) or DEFAULT_BATCH_SIZE))


### Path Handling

MODEL_DIR = MODELS_DIR / "label_location" / "trained_lr"
PREPROC_DIR = MODELS_DIR / "label_location" / "preprocessed_streaming"
PREPROC_TAG = "src-all"
PREPROCESSOR_PATH = PREPROC_DIR / f"preprocessor__{PREPROC_TAG}.pkl"
PREPROC_METADATA_PATH = PREPROC_DIR / f"metadata__{PREPROC_TAG}.json"

TOP_WORDS_MODEL = str(MODEL_DIR / f"lr__words__top__{PREPROC_TAG}.pkl")
REG_WORDS_MODEL = str(MODEL_DIR / f"lr__words__region__{PREPROC_TAG}.pkl")
STA_WORDS_MODEL = str(MODEL_DIR / f"lr__words__state__{PREPROC_TAG}.pkl")
TOP_STRUCT_MODEL = str(MODEL_DIR / f"lr__struct__top__{PREPROC_TAG}.pkl")
REG_STRUCT_MODEL = str(MODEL_DIR / f"lr__struct__region__{PREPROC_TAG}.pkl")
STA_STRUCT_MODEL = str(MODEL_DIR / f"lr__struct__state__{PREPROC_TAG}.pkl")

RAW_DIR = DATA_DIR / "data_reddit_raw" / type_

if args.input:
    input_path = Path(args.input)
else:
    input_path = DATA_DIR / "data_reddit_curated" / group / type_ / "labeled_emotion"

file_list = check_reqd_files(years, input_path, type_)

if args.output:
    output_path = Path(args.output)
else:
    output_path = DATA_DIR / "data_reddit_curated" / group / type_ / "labeled_location"
output_path.mkdir(parents=True, exist_ok=True)

processed_stems = {Path(f).stem for f in os.listdir(output_path) if f.endswith(".csv")}

CACHE_DB_PATH = str(output_path / "author_location_cache.sqlite")
init_location_cache(CACHE_DB_PATH)
report_file_path = os.path.join(output_path, "report_label_location.csv")


### Local helpers aligned with train_location_preprocess.py

_token_re = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> List[str]:
    return _token_re.findall((text or "").lower())


def parse_time_to_hour(time_str: str) -> Optional[int]:
    if not time_str:
        return None
    s = str(time_str).strip()
    if s.isdigit():
        try:
            return time.localtime(int(s)).tm_hour
        except Exception:
            return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            import datetime as _dt
            return _dt.datetime.strptime(s, fmt).hour
        except Exception:
            pass
    try:
        import datetime as _dt
        return _dt.datetime.fromisoformat(s).hour
    except Exception:
        return None


def add_features_for_row(counts: Dict[str, int], text: str, subreddit: str, time_value: str) -> None:
    for tok in tokenize(text):
        key = f"w:{tok}"
        counts[key] = counts.get(key, 0) + 1

    subreddit = (subreddit or "").strip()
    if subreddit:
        key = f"s:{subreddit}"
        counts[key] = counts.get(key, 0) + 1

    hr = parse_time_to_hour(time_value)
    if hr is not None:
        key = f"h:{hr:02d}"
        counts[key] = counts.get(key, 0) + 1


def _normalize_struct_counts(raw: Dict[str, int], mode: str, weight: float, prefix: str) -> Dict[str, float]:
    if not raw or weight <= 0.0:
        return {}
    out: Dict[str, float] = {}
    total = 0.0
    for k, v in raw.items():
        fv = float(v)
        if fv <= 0:
            continue
        if mode == "log1p_l1":
            tv = math.log1p(fv)
        elif mode in {"l1", "tfidf"}:
            tv = fv
        elif mode == "binary_l1":
            tv = 1.0
        else:
            raise ValueError(f"unsupported struct mode: {mode}")
        out[f"{prefix}{k}"] = tv
        total += tv
    if mode in {"log1p_l1", "l1", "binary_l1"} and total > 0.0:
        scale = weight / total
        out = {k: v * scale for k, v in out.items()}
    elif mode == "tfidf" and weight != 1.0:
        out = {k: v * weight for k, v in out.items()}
    return out


def _extract_year_month_from_name(path: str) -> Optional[Tuple[int, int]]:
    m = re.search(r"(\d{4})-(\d{2})", path)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _ym_to_index(year: int, month: int) -> int:
    return year * 12 + (month - 1)


def month_spiral(year: int, month: int, max_files_to_scan: int = 60, max_radius: int = 30) -> List[Tuple[int, str]]:
    center = _ym_to_index(year, month)
    min_idx = _ym_to_index(*RAW_START_YM)
    max_idx = _ym_to_index(*RAW_END_YM)

    offsets = [0]
    for r in range(1, max_radius + 1):
        offsets.append(-r)
        offsets.append(r)

    out: List[Tuple[int, str]] = []
    seen: set[Tuple[int, str]] = set()
    for off in offsets:
        idx = center + off
        if idx < min_idx or idx > max_idx:
            continue
        y = idx // 12
        mo = idx % 12 + 1
        pair = (y, f"{mo:02d}")
        if pair in seen:
            continue
        seen.add(pair)
        out.append(pair)
        if len(out) >= max_files_to_scan:
            break
    return out


### Cache detail helpers


def _init_location_detail_cache(db_path: str) -> None:
    conn = sqlite3.connect(db_path, timeout=60)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS author_location_detail (
                author TEXT PRIMARY KEY,
                location TEXT NOT NULL,
                location_prob REAL,
                contender_location TEXT,
                contender_location_prob REAL,
                top_location TEXT,
                top_location_prob REAL,
                top_contender_location TEXT,
                top_contender_location_prob REAL,
                tier TEXT,
                seen_count INTEGER,
                updated_at INTEGER NOT NULL
            );
            """
        )
        conn.commit()
    finally:
        conn.close()


_init_location_detail_cache(CACHE_DB_PATH)


def cache_get_location_details(db_path: str, authors: Sequence[str]) -> Dict[str, Dict[str, object]]:
    if not authors:
        return {}
    conn = sqlite3.connect(db_path, timeout=60)
    try:
        cur = conn.cursor()
        out: Dict[str, Dict[str, object]] = {}
        author_list = list(authors)
        for i in range(0, len(author_list), 900):
            chunk = author_list[i:i + 900]
            qmarks = ",".join(["?"] * len(chunk))
            cur.execute(
                f"""
                SELECT author, location, location_prob, contender_location, contender_location_prob,
                       top_location, top_location_prob, top_contender_location, top_contender_location_prob,
                       tier, seen_count
                FROM author_location_detail
                WHERE author IN ({qmarks})
                """,
                chunk,
            )
            for row in cur.fetchall():
                out[row[0]] = {
                    "location": row[1],
                    "location_prob": row[2],
                    "contender_location": row[3],
                    "contender_location_prob": row[4],
                    "top_location": row[5],
                    "top_location_prob": row[6],
                    "top_contender_location": row[7],
                    "top_contender_location_prob": row[8],
                    "tier": row[9],
                    "seen_count": row[10],
                }
        return out
    finally:
        conn.close()


def cache_put_location_details(db_path: str, details_by_author: Dict[str, Dict[str, object]]) -> None:
    if not details_by_author:
        return
    conn = sqlite3.connect(db_path, timeout=60)
    try:
        cur = conn.cursor()
        now = int(time.time())
        rows = []
        for author, d in details_by_author.items():
            if not author or not d.get("location"):
                continue
            rows.append(
                (
                    author,
                    d.get("location"),
                    d.get("location_prob"),
                    d.get("contender_location"),
                    d.get("contender_location_prob"),
                    d.get("top_location"),
                    d.get("top_location_prob"),
                    d.get("top_contender_location"),
                    d.get("top_contender_location_prob"),
                    d.get("tier"),
                    d.get("seen_count"),
                    now,
                )
            )
        cur.executemany(
            """
            INSERT OR REPLACE INTO author_location_detail(
                author, location, location_prob, contender_location, contender_location_prob,
                top_location, top_location_prob, top_contender_location, top_contender_location_prob,
                tier, seen_count, updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            rows,
        )
        conn.commit()
    finally:
        conn.close()


### Model loading + inference


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _unwrap_saved_model(obj) -> Tuple[object, List[str], Dict[str, object]]:
    if hasattr(obj, "model") and hasattr(obj, "classes_"):
        classes = list(getattr(obj, "classes_"))
        metadata = getattr(obj, "metadata", {}) or {}
        return getattr(obj, "model"), classes, metadata
    if hasattr(obj, "predict_proba") and hasattr(obj, "classes_"):
        classes = list(getattr(obj, "classes_"))
        return obj, classes, {}
    raise TypeError(f"Unsupported model payload loaded from pickle: {type(obj)!r}")


class LRBatchScorer:
    def __init__(
        self,
        model_payload,
        feature_set: str,
        selected_words: Sequence[str],
        word_tfidf,
        struct_vectorizer,
        struct_tfidf,
        struct_sub_mode: str,
        struct_hour_mode: str,
    ) -> None:
        self.model, self.classes_, self.metadata = _unwrap_saved_model(model_payload)
        self.feature_set = feature_set
        self.selected_words = list(selected_words)
        self.word_tfidf = word_tfidf
        self.struct_vectorizer = struct_vectorizer
        self.struct_tfidf = struct_tfidf
        self.struct_sub_mode = struct_sub_mode
        self.struct_hour_mode = struct_hour_mode
        self.word_index = {str(w): i for i, w in enumerate(self.selected_words)}

        self.struct_feature_names: List[str] = []
        if struct_vectorizer is not None:
            names = getattr(struct_vectorizer, "feature_names_", None)
            if names is None:
                try:
                    names = list(struct_vectorizer.get_feature_names_out())
                except Exception:
                    names = []
            self.struct_feature_names = list(names or [])
        self.struct_sub_cols = [i for i, name in enumerate(self.struct_feature_names) if str(name).startswith("s:")]

    def _build_word_matrix(self, batch_counts: List[Dict[str, int]]) -> sparse.csr_matrix:
        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []
        for rix, counts in enumerate(batch_counts):
            for feat, value in counts.items():
                if not feat.startswith("w:"):
                    continue
                col = self.word_index.get(feat[2:])
                if col is None:
                    continue
                fv = float(value)
                if fv <= 0:
                    continue
                rows.append(rix)
                cols.append(col)
                data.append(fv)
        X_counts = sparse.csr_matrix(
            (np.asarray(data, dtype=np.float32), (rows, cols)),
            shape=(len(batch_counts), len(self.selected_words)),
            dtype=np.float32,
        )
        X_counts.sum_duplicates()
        X_counts.sort_indices()
        return self.word_tfidf.transform(X_counts).astype(np.float32).tocsr()

    def _build_struct_matrix(self, batch_counts: List[Dict[str, int]]) -> sparse.csr_matrix:
        dict_rows: List[Dict[str, float]] = []
        for counts in batch_counts:
            raw_subs = {k[2:]: int(v) for k, v in counts.items() if k.startswith("s:") and int(v) > 0}
            raw_hours = {k[2:]: int(v) for k, v in counts.items() if k.startswith("h:") and int(v) > 0}
            feats: Dict[str, float] = {}
            feats.update(_normalize_struct_counts(raw_subs, self.struct_sub_mode, 1.0, "s:"))
            feats.update(_normalize_struct_counts(raw_hours, self.struct_hour_mode, 1.0, "h:"))
            dict_rows.append(feats)
        X = self.struct_vectorizer.transform(dict_rows).astype(np.float32).tocsr()
        if self.struct_sub_mode == "tfidf" and self.struct_tfidf is not None and self.struct_sub_cols:
            X_sub = self.struct_tfidf.transform(X[:, self.struct_sub_cols])
            X = X.tolil(copy=True)
            X[:, self.struct_sub_cols] = X_sub
            X = X.tocsr().astype(np.float32)
        return X

    def transform(self, batch_counts: List[Dict[str, int]]) -> sparse.csr_matrix:
        if self.feature_set == "words":
            return self._build_word_matrix(batch_counts)
        if self.feature_set == "struct":
            return self._build_struct_matrix(batch_counts)
        raise ValueError(f"Unsupported feature_set for inference scorer: {self.feature_set}")

    def predict_proba(self, batch_counts: List[Dict[str, int]]) -> np.ndarray:
        X = self.transform(batch_counts)
        if X.shape[0] == 0:
            return np.zeros((0, len(self.classes_)), dtype=np.float64)
        return np.asarray(self.model.predict_proba(X), dtype=np.float64)

    def predict_topk(self, batch_counts: List[Dict[str, int]], topk: int = 1) -> List[List[Tuple[str, float]]]:
        proba = self.predict_proba(batch_counts)
        if proba.shape[0] == 0:
            return []
        k = min(max(1, topk), proba.shape[1])
        idx = np.argpartition(-proba, kth=k - 1, axis=1)[:, :k]
        row_vals = np.take_along_axis(proba, idx, axis=1)
        order = np.argsort(-row_vals, axis=1)
        sorted_idx = np.take_along_axis(idx, order, axis=1)
        out: List[List[Tuple[str, float]]] = []
        for row, probs in zip(sorted_idx, np.take_along_axis(proba, sorted_idx, axis=1)):
            out.append([(self.classes_[int(i)], float(p)) for i, p in zip(row, probs)])
        return out


def predict_batch(batch_counts: List[Dict[str, int]], model=None, topk: int = 1, scorer: Optional[LRBatchScorer] = None) -> List[List[Tuple[str, float]]]:
    if scorer is not None:
        return scorer.predict_topk(batch_counts, topk=max(topk, 1))
    if model is None:
        raise ValueError("predict_batch requires either a model/scorer or an explicit scorer")
    raise ValueError("Direct model-only prediction is not supported in this resource; pass scorer instead")


_WORKER_BUNDLE = None


def _verify_required_artifacts() -> None:
    required = [
        PREPROCESSOR_PATH,
        PREPROC_METADATA_PATH,
        Path(TOP_WORDS_MODEL),
        Path(REG_WORDS_MODEL),
        Path(STA_WORDS_MODEL),
        Path(TOP_STRUCT_MODEL),
        Path(REG_STRUCT_MODEL),
        Path(STA_STRUCT_MODEL),
    ]
    missing = [str(p) for p in required if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("Missing required preprocessing/model artifacts: " + ", ".join(missing))


def get_worker_bundle():
    global _WORKER_BUNDLE
    if _WORKER_BUNDLE is not None:
        return _WORKER_BUNDLE

    _verify_required_artifacts()

    with open(PREPROCESSOR_PATH, "rb") as f:
        preprocessor = pickle.load(f)
    with open(PREPROC_METADATA_PATH, "r", encoding="utf-8") as f:
        preproc_meta = json.load(f)

    selected_words = preprocessor.get("selected_words")
    word_tfidf = preprocessor.get("word_tfidf")
    struct_vectorizer = preprocessor.get("struct_vectorizer")
    struct_tfidf = preprocessor.get("struct_tfidf")
    if selected_words is None or word_tfidf is None or struct_vectorizer is None:
        raise RuntimeError("Preprocessor artifact is missing selected_words, word_tfidf, or struct_vectorizer")

    struct_sub_mode = str(preproc_meta.get("struct_sub_mode", "log1p_l1"))
    struct_hour_mode = str(preproc_meta.get("struct_hour_mode", "l1"))

    _WORKER_BUNDLE = {
        "top_words": LRBatchScorer(load_pickle(TOP_WORDS_MODEL), "words", selected_words, word_tfidf, struct_vectorizer, struct_tfidf, struct_sub_mode, struct_hour_mode),
        "top_struct": LRBatchScorer(load_pickle(TOP_STRUCT_MODEL), "struct", selected_words, word_tfidf, struct_vectorizer, struct_tfidf, struct_sub_mode, struct_hour_mode),
        "reg_words": LRBatchScorer(load_pickle(REG_WORDS_MODEL), "words", selected_words, word_tfidf, struct_vectorizer, struct_tfidf, struct_sub_mode, struct_hour_mode),
        "reg_struct": LRBatchScorer(load_pickle(REG_STRUCT_MODEL), "struct", selected_words, word_tfidf, struct_vectorizer, struct_tfidf, struct_sub_mode, struct_hour_mode),
        "sta_words": LRBatchScorer(load_pickle(STA_WORDS_MODEL), "words", selected_words, word_tfidf, struct_vectorizer, struct_tfidf, struct_sub_mode, struct_hour_mode),
        "sta_struct": LRBatchScorer(load_pickle(STA_STRUCT_MODEL), "struct", selected_words, word_tfidf, struct_vectorizer, struct_tfidf, struct_sub_mode, struct_hour_mode),
    }
    return _WORKER_BUNDLE


### Probability aggregation / decision logic


def _rankings_to_prob_map(rankings: List[Tuple[str, float]]) -> Dict[str, float]:
    return {label: float(prob) for label, prob in rankings if label}


def blend_rankings(rankings_by_name: Dict[str, List[Tuple[str, float]]], weights: Dict[str, float], topk: int = 2) -> List[Tuple[str, float]]:
    totals: Dict[str, float] = {}
    weight_sum = 0.0
    for name, rankings in rankings_by_name.items():
        weight = float(weights.get(name, 0.0))
        if weight <= 0.0:
            continue
        prob_map = _rankings_to_prob_map(rankings)
        if not prob_map:
            continue
        weight_sum += weight
        for label, prob in prob_map.items():
            totals[label] = totals.get(label, 0.0) + (weight * prob)
    if weight_sum <= 0.0:
        return []
    blended = [(label, score / weight_sum) for label, score in totals.items()]
    blended.sort(key=lambda x: (-x[1], x[0]))
    return blended[:max(1, topk)]


def _margin(scores: List[Tuple[str, float]]) -> float:
    if len(scores) < 2:
        return float("inf")
    return float(scores[0][1]) - float(scores[1][1])


def _fmt_prob(value: Optional[float]) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.6f}"
    except Exception:
        return ""


def unknown_result(top_scores: Optional[List[Tuple[str, float]]] = None, seen_count: int = 0, reason: str = "unknown") -> Dict[str, object]:
    contender_label = ""
    contender_prob = None
    if top_scores and len(top_scores) >= 2:
        contender_label = top_scores[1][0]
        contender_prob = float(top_scores[1][1])
    return {
        "location": UNKNOWN_LABEL,
        "location_prob": None,
        "contender_location": contender_label,
        "contender_location_prob": contender_prob,
        "top_location": top_scores[0][0] if top_scores else "",
        "top_location_prob": float(top_scores[0][1]) if top_scores else None,
        "top_contender_location": contender_label,
        "top_contender_location_prob": contender_prob,
        "tier": reason,
        "seen_count": seen_count,
    }


def top_level_fallback(top_scores: List[Tuple[str, float]], seen_count: int, reason: str) -> Dict[str, object]:
    top1 = top_scores[0] if top_scores else (UNKNOWN_LABEL, None)
    top2 = top_scores[1] if len(top_scores) >= 2 else ("", None)
    return {
        "location": top1[0] if top1[0] else UNKNOWN_LABEL,
        "location_prob": float(top1[1]) if top1[1] is not None else None,
        "contender_location": top2[0] if top2[0] else "",
        "contender_location_prob": float(top2[1]) if top2[1] is not None else None,
        "top_location": top1[0] if top1[0] else "",
        "top_location_prob": float(top1[1]) if top1[1] is not None else None,
        "top_contender_location": top2[0] if top2[0] else "",
        "top_contender_location_prob": float(top2[1]) if top2[1] is not None else None,
        "tier": reason,
        "seen_count": seen_count,
    }


def final_tier_result(label_scores: List[Tuple[str, float]], top_scores: List[Tuple[str, float]], tier: str, seen_count: int) -> Dict[str, object]:
    top1 = label_scores[0] if label_scores else (UNKNOWN_LABEL, None)
    top2 = label_scores[1] if len(label_scores) >= 2 else ("", None)
    top_top1 = top_scores[0] if top_scores else ("", None)
    top_top2 = top_scores[1] if len(top_scores) >= 2 else ("", None)
    return {
        "location": top1[0] if top1[0] else UNKNOWN_LABEL,
        "location_prob": float(top1[1]) if top1[1] is not None else None,
        "contender_location": top2[0] if top2[0] else "",
        "contender_location_prob": float(top2[1]) if top2[1] is not None else None,
        "top_location": top_top1[0] if top_top1[0] else "",
        "top_location_prob": float(top_top1[1]) if top_top1[1] is not None else None,
        "top_contender_location": top_top2[0] if top_top2[0] else "",
        "top_contender_location_prob": float(top_top2[1]) if top_top2[1] is not None else None,
        "tier": tier,
        "seen_count": seen_count,
    }


def infer_locations_for_batch(batch_counts: List[Dict[str, int]], batch_seen: List[int], bundle) -> List[Dict[str, object]]:
    top_words = predict_batch(batch_counts, scorer=bundle["top_words"], topk=2)
    top_struct = predict_batch(batch_counts, scorer=bundle["top_struct"], topk=2)
    reg_words = predict_batch(batch_counts, scorer=bundle["reg_words"], topk=2)
    reg_struct = predict_batch(batch_counts, scorer=bundle["reg_struct"], topk=2)
    sta_words = predict_batch(batch_counts, scorer=bundle["sta_words"], topk=2)
    sta_struct = predict_batch(batch_counts, scorer=bundle["sta_struct"], topk=2)

    out: List[Dict[str, object]] = []
    for idx, seen in enumerate(batch_seen):
        top_scores = blend_rankings(
            {"words": top_words[idx], "struct": top_struct[idx]},
            top_weights,
            topk=2,
        )
        if seen < MIN_SAMPLES_FOR_INFERENCE:
            out.append(unknown_result(top_scores=top_scores, seen_count=seen, reason="low_samples"))
            continue
        if not top_scores or not top_scores[0][0]:
            out.append(unknown_result(top_scores=top_scores, seen_count=seen, reason="no_top_prediction"))
            continue
        if float(top_scores[0][1]) < TOP_CONF_THRESHOLD:
            out.append(unknown_result(top_scores=top_scores, seen_count=seen, reason="low_top_conf"))
            continue

        top_label = top_scores[0][0]
        if top_label == "NON_US":
            region_scores = blend_rankings(
                {"words": reg_words[idx], "struct": reg_struct[idx]},
                regional_weights,
                topk=2,
            )
            if region_scores and _margin(region_scores) >= REG_CONF_MARGIN:
                out.append(final_tier_result(region_scores, top_scores, tier="region", seen_count=seen))
            else:
                out.append(top_level_fallback(top_scores, seen_count=seen, reason="top_non_us_fallback"))
        elif top_label == "US":
            state_scores = blend_rankings(
                {"words": sta_words[idx], "struct": sta_struct[idx]},
                state_weights,
                topk=2,
            )
            if state_scores and _margin(state_scores) >= STA_CONF_MARGIN:
                out.append(final_tier_result(state_scores, top_scores, tier="state", seen_count=seen))
            else:
                out.append(top_level_fallback(top_scores, seen_count=seen, reason="top_us_fallback"))
        else:
            out.append(top_level_fallback(top_scores, seen_count=seen, reason="top_only"))
    return out


### Location labeling pipeline


def _find_header_index(header: Sequence[str], name: str, fallback: int) -> int:
    try:
        return header.index(name)
    except ValueError:
        return fallback


def _read_month_rows_and_seed_features(curated_csv_path: str) -> Tuple[List[str], List[List[str]], set[str], Dict[str, Dict[str, int]], Dict[str, int], Dict[str, int]]:
    rows: List[List[str]] = []
    authors: set[str] = set()
    local_counts: Dict[str, Dict[str, int]] = {}
    local_seen: Dict[str, int] = {}

    with open(curated_csv_path, "r", encoding="utf-8-sig", errors="ignore") as f:
        reader = csv.reader((line.replace("\x00", "") for line in f))
        header = next(reader)

        author_idx = _find_header_index(header, "author", 3)
        text_idx = _find_header_index(header, "text", 2)
        time_idx = _find_header_index(header, "time", 4)
        subreddit_idx = _find_header_index(header, "subreddit", 5)

        for r in reader:
            if not r:
                continue
            rows.append(r)
            if len(r) <= author_idx:
                continue
            author = r[author_idx].strip()
            if not author or author == "[deleted]":
                continue
            authors.add(author)
            counts = local_counts.setdefault(author, {})
            add_features_for_row(
                counts,
                text=r[text_idx] if len(r) > text_idx else "",
                subreddit=r[subreddit_idx] if len(r) > subreddit_idx else "",
                time_value=r[time_idx] if len(r) > time_idx else "",
            )
            local_seen[author] = local_seen.get(author, 0) + 1

    return header, rows, authors, local_counts, local_seen, {
        "author": author_idx,
        "text": text_idx,
        "time": time_idx,
        "subreddit": subreddit_idx,
    }


def _merge_feature_maps(
    local_counts: Dict[str, Dict[str, int]],
    local_seen: Dict[str, int],
    raw_counts: Dict[str, Dict[str, int]],
    raw_seen: Dict[str, int],
    target_authors: Sequence[str],
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, int]]:
    merged_counts: Dict[str, Dict[str, int]] = {}
    merged_seen: Dict[str, int] = {}
    for author in target_authors:
        counts: Dict[str, int] = {}
        for src in (local_counts.get(author, {}), raw_counts.get(author, {})):
            for k, v in src.items():
                counts[k] = counts.get(k, 0) + int(v)
        merged_counts[author] = counts
        merged_seen[author] = int(local_seen.get(author, 0)) + int(raw_seen.get(author, 0))
    return merged_counts, merged_seen


def _details_from_cache_or_label(author: str, cached_details: Dict[str, Dict[str, object]], cached_locations: Dict[str, str]) -> Dict[str, object]:
    if author in cached_details:
        return cached_details[author]
    loc = cached_locations.get(author, "")
    return {
        "location": loc or UNKNOWN_LABEL,
        "location_prob": None,
        "contender_location": "",
        "contender_location_prob": None,
        "top_location": loc or "",
        "top_location_prob": None,
        "top_contender_location": "",
        "top_contender_location_prob": None,
        "tier": "cache_legacy",
        "seen_count": None,
    }


def _write_output_csv(header: Sequence[str], rows: Sequence[Sequence[str]], author_idx: int, detail_by_author: Dict[str, Dict[str, object]], out_file: Path) -> None:
    with open(out_file, "w", encoding="utf-8", newline="", errors="ignore") as fo:
        writer = csv.writer(fo)
        writer.writerow(list(header) + ["location", "location_prob", "contender_location", "contender_location_prob"])
        for r in rows:
            author = r[author_idx].strip() if len(r) > author_idx else ""
            detail = detail_by_author.get(author, {"location": UNKNOWN_LABEL})
            writer.writerow(
                list(r)
                + [
                    detail.get("location", UNKNOWN_LABEL),
                    _fmt_prob(detail.get("location_prob")),
                    detail.get("contender_location", "") or "",
                    _fmt_prob(detail.get("contender_location_prob")),
                ]
            )


def label_location_month(curated_csv_path: str) -> Tuple[str, int, int, int]:
    curated_csv_path = str(curated_csv_path)
    stem = Path(curated_csv_path).stem

    if stem in processed_stems:
        log_report(report_file_path, f"[skip] output already exists for {stem}")
        return (stem, 0, 0, 0)

    ym = _extract_year_month_from_name(curated_csv_path)
    if ym is None:
        log_report(report_file_path, f"[warn] could not parse year-month from {curated_csv_path}; skipping")
        return (stem, 0, 0, 0)
    year, month_int = ym

    start = time.time()

    try:
        header, rows, authors, local_counts, local_seen, idx_map = _read_month_rows_and_seed_features(curated_csv_path)
    except StopIteration:
        return (stem, 0, 0, 0)

    if not authors:
        log_report(report_file_path, f"[warn] {stem}: no authors found")
        return (stem, 0, 0, 0)

    cached_locations = cache_get_locations(CACHE_DB_PATH, authors)
    cached_details = cache_get_location_details(CACHE_DB_PATH, list(authors))

    detail_by_author: Dict[str, Dict[str, object]] = {}
    for author in authors:
        if author in cached_locations or author in cached_details:
            detail_by_author[author] = _details_from_cache_or_label(author, cached_details, cached_locations)

    remaining_authors = sorted(a for a in authors if a not in detail_by_author)
    n_cached = len(detail_by_author)
    n_total = len(authors)

    if not remaining_authors:
        out_file = output_path / f"{stem}.csv"
        _write_output_csv(header, rows, idx_map["author"], detail_by_author, out_file)
        elapsed = (time.time() - start) / 60
        log_report(report_file_path, f"[done-cache] {stem}: rows={len(rows):,} authors={n_total:,} cached={n_cached:,} minutes={elapsed:.2f}")
        return (stem, len(rows), n_total, 0)

    scan_months = month_spiral(year, month_int, max_files_to_scan=max_files_to_scan, max_radius=max_radius)
    raw_files: List[str] = []
    months_with_files = 0
    for y, mstr in scan_months:
        files = find_raw_month_files(RAW_DIR, type_, y, mstr)
        if files:
            raw_files.extend(files)
            months_with_files += 1

    if not raw_files:
        for author in remaining_authors:
            detail_by_author[author] = unknown_result(seen_count=local_seen.get(author, 0), reason="no_raw_files")
        out_file = output_path / f"{stem}.csv"
        _write_output_csv(header, rows, idx_map["author"], detail_by_author, out_file)
        elapsed = (time.time() - start) / 60
        log_report(report_file_path, f"[warn] {stem}: no raw files in scan window; wrote UNKNOWN for {len(remaining_authors):,}. minutes={elapsed:.2f}")
        return (stem, len(rows), n_total, len(remaining_authors))

    log_report(
        report_file_path,
        f"[start] {stem}: authors={n_total:,} cached={n_cached:,} need_raw={len(remaining_authors):,} "
        f"scan_months={len(scan_months)} months_with_files={months_with_files} raw_files={len(raw_files)} "
        f"samples_per_author={max_items_per_author} batch_size={batch_size} max_files_to_scan={max_files_to_scan} max_radius={max_radius}",
    )

    bundle = get_worker_bundle()

    raw_scan_start = time.time()
    raw_counts, raw_seen = build_author_feature_map_from_raw_zst_with_seen(
        raw_files=raw_files,
        target_authors=set(remaining_authors),
        type_=type_,
        max_items_per_author=max_items_per_author,
    )
    log_report(
        report_file_path,
        f"[scan] {stem}: collected_raw_for={len(raw_counts):,} authors in {(time.time() - raw_scan_start)/60:.2f} minutes",
    )

    author_to_counts, author_seen = _merge_feature_maps(local_counts, local_seen, raw_counts, raw_seen, remaining_authors)

    to_cache_labels: Dict[str, str] = {}
    to_cache_details: Dict[str, Dict[str, object]] = {}
    n_cache_confident = 0
    n_cache_skipped_lowconf = 0
    n_cache_skipped_lowsamples = 0

    for i in range(0, len(remaining_authors), batch_size):
        chunk = remaining_authors[i:i + batch_size]
        batch_counts = [author_to_counts.get(a, {}) for a in chunk]
        batch_seen = [int(author_seen.get(a, 0)) for a in chunk]
        batch_results = infer_locations_for_batch(batch_counts, batch_seen, bundle)

        for author, detail in zip(chunk, batch_results):
            detail_by_author[author] = detail
            seen = int(detail.get("seen_count") or 0)
            if seen < MIN_SAMPLES_FOR_CACHE:
                n_cache_skipped_lowsamples += 1
                continue
            if detail.get("location") == UNKNOWN_LABEL:
                n_cache_skipped_lowconf += 1
                continue
            to_cache_labels[author] = str(detail["location"])
            to_cache_details[author] = detail
            n_cache_confident += 1

    if to_cache_labels:
        cache_put_locations(CACHE_DB_PATH, to_cache_labels)
        cache_put_location_details(CACHE_DB_PATH, to_cache_details)

    log_report(
        report_file_path,
        f"[cache] {stem}: newly_labeled={len(remaining_authors):,} cached_confident={n_cache_confident:,} "
        f"skipped_lowconf_or_unknown={n_cache_skipped_lowconf:,} skipped_lowsamples={n_cache_skipped_lowsamples:,} "
        f"top_conf>={TOP_CONF_THRESHOLD} reg_margin>={REG_CONF_MARGIN} state_margin>={STA_CONF_MARGIN} min_samples_cache={MIN_SAMPLES_FOR_CACHE}",
    )

    out_file = output_path / f"{stem}.csv"
    _write_output_csv(header, rows, idx_map["author"], detail_by_author, out_file)

    elapsed = (time.time() - start) / 60
    covered = sum(1 for a in remaining_authors if author_seen.get(a, 0) > 0)
    log_report(
        report_file_path,
        f"[done] {stem}: rows={len(rows):,} authors={n_total:,} cached={n_cached:,} scanned_raw={len(remaining_authors):,} "
        f"covered={covered:,} minutes={elapsed:.2f}",
    )
    return (stem, len(rows), n_total, len(remaining_authors))


def label_location_parallel() -> None:
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
    pending = {}
    started = time.time()
    last_heartbeat = started

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for f in file_list:
            fut = ex.submit(label_location_month, f)
            pending[fut] = f

        while pending:
            done, not_done = wait(list(pending.keys()), timeout=60, return_when=FIRST_COMPLETED)
            if not done:
                now = time.time()
                if now - last_heartbeat >= PROGRESS_HEARTBEAT_SECONDS:
                    completed = len(file_list) - len(pending)
                    elapsed_min = (now - started) / 60.0
                    log_report(
                        report_file_path,
                        f"[progress] completed_months={completed:,}/{len(file_list):,} pending={len(pending):,} elapsed_minutes={elapsed_min:.2f}",
                    )
                    last_heartbeat = now
                continue

            for fut in done:
                src = pending.pop(fut)
                try:
                    _, rows, n_auth, n_raw = fut.result()
                    total_rows += rows
                    total_authors += n_auth
                    total_raw += n_raw
                except Exception as e:
                    log_report(report_file_path, f"[error] month failed for {src}: {e}")

    log_report(report_file_path, f"[summary] total rows written: {total_rows:,} total authors: {total_authors:,} raw-scanned authors: {total_raw:,}")


if __name__ == "__main__":
    overall = time.time()
    try:
        label_location_parallel()
    except Exception as e:
        log_report(report_file_path, f"Fatal error during location labeling: {e}")
        raise
    finally:
        mins = (time.time() - overall) / 60
        log_report(report_file_path, f"Location labeling finished in {mins:.2f} minutes.")
