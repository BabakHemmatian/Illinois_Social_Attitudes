"""Reproduce the residual-irrelevance and inter-annotator agreement figures
reported for ISAAC's relevance-filtering pipeline from the rating files in
this directory.

    python qa_fpr_interrater.py            # print Table 3 and Table C3
    python qa_fpr_interrater.py --check    # also assert every cell equals the published value

Tables reproduced (see README.md for the file -> table mapping):
    Table 3  (main text)     final double-rated comment audit + single-rater submission audit
    Table C3 (Appendix C.3)  residual irrelevance by development stage, single-rater

Every file has the same schema: random_id, text, relevance[, ...].
relevance is "1" (relevant), "0" (irrelevant) or "X" (annotator marked the
document as unclear). Binarization: "1" -> relevant, anything else -> not
relevant, matching the rule stated in the Method section ("unclear labels were
marked as irrelevant"). Two annotators' files for the same sample share
random_id, so pairs are joined on that column.

Requires only the standard library; Cohen's kappa is computed directly.
"""
import csv
import os
import sys

csv.field_size_limit(2**31 - 1)
HERE = os.path.dirname(os.path.abspath(__file__))

GROUPS = ["ability", "age", "weight", "race", "sexuality", "skin_tone"]
LABEL = {"ability": "Ability", "age": "Age", "weight": "Body weight",
         "race": "Race", "sexuality": "Sexuality", "skin_tone": "Skin tone"}

# --- Table 3: last comment stage each distinction required, double-rated ------
FINAL_COMMENTS = {
    "ability":   ("qa_d_finalregex_ability_n100_rated_r0.csv",  "qa_d_finalregex_ability_n100_rated_r2.csv"),
    "age":       ("qa_b1_postregex_age_n100_rated_r0.csv",      "qa_b1_postregex_age_n100_rated_r2.csv"),
    "weight":    ("qa_b1_postregex_weight_n100_rated_r0.csv",   "qa_b1_postregex_weight_n100_rated_r2.csv"),
    "race":      ("qa_c_postretrain_race_n150_rated_r1.csv",    "qa_c_postretrain_race_n150_rated_r2.csv"),
    "sexuality": ("qa_b1_postregex_sexuality_n100_rated_r0.csv","qa_b1_postregex_sexuality_n100_rated_r2.csv"),
    "skin_tone": ("qa_c_postretrain_skin_tone_n150_rated_r1.csv","qa_c_postretrain_skin_tone_n150_rated_r2.csv"),
}
# --- Table 3, last column: submission transfer audit, single-rated (annotator 2)
FINAL_SUBMISSIONS = {
    "ability":   "qa_d_finalregex_ability_subm_n100_rated_r2.csv",
    "age":       "qa_b1_postregex_age_subm_n100_rated_r2.csv",
    "weight":    "qa_b1_postregex_weight_subm_n100_rated_r2.csv",
    "race":      "qa_d_finalregex_race_subm_n100_rated_r2.csv",
    "sexuality": "qa_b1_postregex_sexuality_subm_n100_rated_r2.csv",
    "skin_tone": "qa_d_finalregex_skin_tone_subm_n100_rated_r2.csv",
}
# --- Table C3: development stages, single-rated ------------------------------
STAGES = [
    ("A: post-classifier (k = 200)", {g: f"qa_a_postinit_{g}_n200_rated.csv" for g in GROUPS}),
    ("B1: patterns v1 (k = 100)",    {g: f"qa_b1_postregex_{g}_n100_rated_r0.csv" for g in GROUPS}),
    ("B2: patterns v2 (k = 100)",    {"race": "qa_b2_postregex_race_n100_rated.csv",
                                      "skin_tone": "qa_b2_postregex_skin_tone_n100_rated.csv"}),
    ("C: post-retraining (k = 153)", {"race": "qa_c_postretrain_race_n153_rated_r1.csv",
                                      "skin_tone": "qa_c_postretrain_skin_tone_n153_rated_r1.csv"}),
    ("D: submissions patterns (k = 100)", {"ability": FINAL_SUBMISSIONS["ability"],
                                           "race": FINAL_SUBMISSIONS["race"],
                                           "skin_tone": FINAL_SUBMISSIONS["skin_tone"]}),
]
# Comment-level Stage D checks (not a Table C3 column; the checks behind the
# Figure C1 note that ability/race/skin tone received a final pattern pass).
STAGE_D_COMMENTS = {"ability": "qa_d_finalregex_ability_n100_rated_r2.csv",
                    "race": "qa_d_finalregex_race_n100_rated_r2.csv",
                    "skin_tone": "qa_d_finalregex_skin_tone_n100_rated_r2.csv"}

# Published values (percent, kappa, percent) for --check.
PUBLISHED_T3 = {  # stringent %, lenient %, kappa, raw agreement %, submissions %
    "ability":   (6.0, 3.0, .651, 97.0, 4.0),
    "age":       (2.0, 0.0, .000, 98.0, 2.0),
    "weight":    (4.0, 3.0, .852, 99.0, 2.0),
    "race":      (8.0, 5.3, .786, 97.3, 4.0),
    "sexuality": (5.0, 2.0, .556, 97.0, 5.0),
    "skin_tone": (9.3, 4.7, .643, 95.3, 9.0),
}
PUBLISHED_T3_POOLED = (43, 700, 23, 700, 26, 600)  # stringent, lenient, submissions
PUBLISHED_C3 = {
    "A: post-classifier (k = 200)":       {"ability": 20.5, "age": 5.5, "weight": 16.5, "race": 51.0, "sexuality": 14.5, "skin_tone": 44.0},
    "B1: patterns v1 (k = 100)":          {"ability": 7.0, "age": 0.0, "weight": 3.0, "race": 31.0, "sexuality": 3.0, "skin_tone": 34.0},
    "B2: patterns v2 (k = 100)":          {"race": 41.0, "skin_tone": 29.0},
    "C: post-retraining (k = 153)":       {"race": 6.5, "skin_tone": 5.2},
    "D: submissions patterns (k = 100)":  {"ability": 4.0, "race": 4.0, "skin_tone": 9.0},
}


def load(fn):
    """random_id -> 1 if rated relevant else 0."""
    with open(os.path.join(HERE, fn), encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        assert r.fieldnames[:3] == ["random_id", "text", "relevance"], (fn, r.fieldnames)
        d = {}
        for row in r:
            rid = row["random_id"]
            assert rid not in d, (fn, "duplicate random_id", rid)
            d[rid] = 1 if row["relevance"] == "1" else 0
    return d


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa1, pb1 = sum(a) / n, sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return 0.0 if pe == 1 else (po - pe) / (1 - pe)


def two_rater(fa, fb):
    a, b = load(fa), load(fb)
    ids = sorted(set(a) & set(b))
    assert len(ids) == len(a) == len(b), (fa, fb, "rater files do not cover the same documents")
    va, vb = [a[i] for i in ids], [b[i] for i in ids]
    n = len(ids)
    stringent = sum(1 for x, y in zip(va, vb) if not (x and y))
    lenient = sum(1 for x, y in zip(va, vb) if not (x or y))
    raw = 100 * sum(x == y for x, y in zip(va, vb)) / n
    return n, stringent, lenient, kappa(va, vb), raw


def single_rater(fn):
    d = load(fn)
    return len(d), sum(1 for v in d.values() if v == 0)


def main(check=False):
    bad = []

    def cmp(label, got, want, tol):
        if want is not None and abs(got - want) > tol:
            bad.append(f"{label}: computed {got:.3f}, published {want:.3f}")

    print("Table 3. Final residual irrelevance audit on comments and submissions")
    print(f"{'Distinction':12} {'k':>4} {'stringent':>12} {'lenient':>12} {'kappa':>6} {'raw':>6} {'subm.':>12}")
    ts = tl = tn = si = sn = 0
    for g in GROUPS:
        n, s, l, k, raw = two_rater(*FINAL_COMMENTS[g])
        m, irr = single_rater(FINAL_SUBMISSIONS[g])
        print(f"{LABEL[g]:12} {n:4} {s:3}/{n} ={100*s/n:5.1f}% {l:3}/{n} ={100*l/n:5.1f}% {k:6.3f} {raw:5.1f}% {irr:3}/{m} ={100*irr/m:5.1f}%")
        ts += s; tl += l; tn += n; si += irr; sn += m
        if check:
            p = PUBLISHED_T3[g]
            cmp(f"T3 {g} stringent", 100 * s / n, p[0], .05); cmp(f"T3 {g} lenient", 100 * l / n, p[1], .05)
            cmp(f"T3 {g} kappa", k, p[2], .0005); cmp(f"T3 {g} raw", raw, p[3], .05); cmp(f"T3 {g} subm", 100 * irr / m, p[4], .05)
    print(f"{'Pooled':12} {tn:4} {ts:3}/{tn} ={100*ts/tn:5.1f}% {tl:3}/{tn} ={100*tl/tn:5.1f}% {'':6} {'':6} {si:3}/{sn} ={100*si/sn:5.1f}%")
    if check:
        if (ts, tn, tl, tn, si, sn) != PUBLISHED_T3_POOLED:
            bad.append(f"T3 pooled: computed {(ts, tn, tl, tn, si, sn)}, published {PUBLISHED_T3_POOLED}")

    print("\nTable C3. Residual irrelevance rate by development stage (single-rater)")
    for stage, files in STAGES:
        print(f"  {stage}")
        for g, fn in files.items():
            n, irr = single_rater(fn)
            print(f"    {LABEL[g]:12} {irr:3}/{n} = {100*irr/n:5.1f}%")
            if check:
                cmp(f"C3 {stage} {g}", 100 * irr / n, PUBLISHED_C3[stage].get(g), .05)
    print("  D: final patterns, comments (annotator 2; behind the Figure C1 note, not a Table C3 column)")
    for g, fn in STAGE_D_COMMENTS.items():
        n, irr = single_rater(fn)
        print(f"    {LABEL[g]:12} {irr:3}/{n} = {100*irr/n:5.1f}%")

    if check:
        if bad:
            print("\nMISMATCH against published values:\n  " + "\n  ".join(bad))
            sys.exit(1)
        print("\nAll cells match the published values.")


if __name__ == "__main__":
    main(check="--check" in sys.argv)
