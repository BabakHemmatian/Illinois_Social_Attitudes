# Relevance-filtering QA ratings

Human relevance ratings of stratified random samples drawn from the ISAAC
filtering pipeline at successive development stages. These files are the
evidence behind the residual-irrelevance and inter-annotator agreement figures
in the paper: **Table 3** (final double-rated comment audit and single-rater
submission transfer audit) and **Table C3** (residual irrelevance by
development stage). `qa_fpr_interrater.py` regenerates both tables from this
directory alone:

```
python qa_fpr_interrater.py            # print Table 3 and Table C3
python qa_fpr_interrater.py --check    # exit non-zero if any cell differs from the published value
```

The script needs only the Python standard library.

## File naming

```
qa_<stage>_<stage name>_<distinction>[_subm]_n<k>_rated_r<annotator>.csv
```

| Token | Values | Meaning |
|---|---|---|
| `stage` / `stage name` | `a_postinit`, `b1_postregex`, `b2_postregex`, `c_postretrain`, `d_finalregex`, `r_retraining_input` | Pipeline checkpoint the sample was drawn after (below) |
| `distinction` | `ability`, `age`, `race`, `sexuality`, `skin_tone`, `weight` | Social group distinction |
| `subm` | present / absent | Sample of submissions; absent = comments |
| `n<k>` | e.g. `n100`, `n150` | Number of documents in the file |
| `r<annotator>` | `r0`, `r1`, `r2` | Which of the three trained annotators produced the ratings. Annotators 0 and 1 are the pair who rated the classifier training samples in `data/data_relevance_ratings` (`_0` / `_1` there). Annotator 0 rated the single-rater development samples at Stages A, B1 and B2 and the retraining inputs; annotator 1 rated the Stage C development samples; annotator 2 rated every submission audit and provided the second rating in every final audit |

Stages, in pipeline order:

| Stage | Drawn after | Distinctions |
|---|---|---|
| A `postinit` | keyword filter + initial relevance classifier | all six |
| B1 `postregex` | first complex-pattern (regular-expression) pass | all six |
| B2 `postregex` | second complex-pattern pass | race, skin tone |
| C `postretrain` | relevance-classifier retraining with a conservative threshold | race, skin tone |
| D `finalregex` | final complex-pattern pass, applied so that filtering quality transferred to submissions | ability, race, skin tone |
| R `retraining_input` | not an audit: the 400-document rated samples used to retrain the Stage C classifiers (held-out slice reported in Table C2) | race, skin tone |

## Schema

Every file is UTF-8, LF-terminated, RFC 4180 CSV with a header row and one
row per document.

| Column | Present in | Values |
|---|---|---|
| `random_id` | all | Blinded document identifier. Files that hold different annotators' ratings of the same sample share `random_id`, so they join on this column. Identifiers are random and carry no information about the source post. |
| `text` | all | Post text as rated (lower-cased; may contain newlines, which are quoted). |
| `relevance` | all | `1` relevant to the distinction; `0` irrelevant; `X` annotator marked the document as unclear. For every reported rate, `1` counts as relevant and `0` or `X` as not relevant ("unclear labels were marked as irrelevant", Method). |
| `false_positive_words`, `false_positive_category` | some Stage A / B1 files | Free-text annotator notes on why a document was irrelevant (the matched word, and the kind of false positive). Filled only for irrelevant documents; not analyzed in the paper. |
| `<pole>_attitude` (`white_`/`black_`, `light_`/`dark_`) | Stage C comment files | Pilot attitude coding on a −2 (very negative) to +2 (very positive) scale toward the named pole, per the attitude annotation guide; `0` when neutral or the pole is not mentioned. Not analyzed in the paper. |

## Which file backs which published number

**Table 3.** Double-rated comment sample at the last stage each distinction
required, and the single-rater (annotator 2) submission audit.

| Distinction | Comments, annotator pair | Submissions |
|---|---|---|
| Ability | `qa_d_finalregex_ability_n100_rated_r0` + `_r2` | `qa_d_finalregex_ability_subm_n100_rated_r2` |
| Age | `qa_b1_postregex_age_n100_rated_r0` + `_r2` | `qa_b1_postregex_age_subm_n100_rated_r2` |
| Body weight | `qa_b1_postregex_weight_n100_rated_r0` + `_r2` | `qa_b1_postregex_weight_subm_n100_rated_r2` |
| Race | `qa_c_postretrain_race_n150_rated_r1` + `_r2` | `qa_d_finalregex_race_subm_n100_rated_r2` |
| Sexuality | `qa_b1_postregex_sexuality_n100_rated_r0` + `_r2` | `qa_b1_postregex_sexuality_subm_n100_rated_r2` |
| Skin tone | `qa_c_postretrain_skin_tone_n150_rated_r1` + `_r2` | `qa_d_finalregex_skin_tone_subm_n100_rated_r2` |

Stringent rule: a document is relevant only if both annotators rated it `1`.
Lenient rule: either annotator's `1` suffices. Cohen's κ and raw agreement
are computed on the binarized ratings.

**Table C3.** Single-rater development samples on comments, one column per stage.

| Column | Files |
|---|---|
| A (k = 200) | `qa_a_postinit_<distinction>_n200_rated_r0` |
| B1 (k = 100) | `qa_b1_postregex_<distinction>_n100_rated_r0` |
| B2 (k = 100) | `qa_b2_postregex_{race,skin_tone}_n100_rated_r0` |
| C (k = 150) | `qa_c_postretrain_{race,skin_tone}_n150_rated_r1` (annotator 1's ratings of the sample that Table 3 double-rates) |
| D (k = 100) | the three Stage D submission files above (this column is the submission audit; the comment-level Stage D files `qa_d_finalregex_{ability,race,skin_tone}_n100_rated_r2` are the checks behind the Figure C1 note and are printed by the script as well) |