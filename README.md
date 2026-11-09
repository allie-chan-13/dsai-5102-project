# Drug Review Sentiment Classification

Course project for **DSAI 5102**. The notebook turns patient-written drug reviews into a **3-class sentiment** label and compares classical and neural classifiers.

The dataset can be reused on its own. The course models are context; the files under `drug-review/dataset/` are the thing to build on.

## Repository

| Path | What it is |
| --- | --- |
| `drug-review/` | Dataset, later methods, and tracked scores in `drug-review/results/` |
| `project-code.ipynb` | Course notebook (EDA, baseline, and the five original models) |
| `project-report.docx`, `project-slides.pdf` | Course writeup. Kept locally and gitignored |

Model checkpoints (`*.pt`, `*.bin`, `*.ckpt`), `.venv/`, and Python caches are gitignored. The metrics CSV and the comparison chart under `drug-review/results/` are kept.

## What the project does

Each row is one patient review of one drug, written on Drugs.com, with a 1–10 star rating. The project does not predict the raw 1–10 score. It maps the rating to a label and trains text classifiers to recover that label from the review:

| Rating | Label | How the report reads it |
| --- | --- | --- |
| 1–4 | Negative | Poor outcome or serious side effects |
| 5–6 | Neutral | Some benefit, with side effects |
| 7–10 | Positive | Satisfactory outcome |

The course notebook is `project-code.ipynb`. Its pipeline is:

1. Load **only** `drug-review/dataset/drugsComTrain_raw.csv` (161,297 rows).
2. Split that file again with PySpark `randomSplit([0.8, 0.2], seed=42)` into 128,885 train and 32,412 test rows. The official test file is not used by the notebook.
3. Drop rows with a missing or scraped-garbage `condition`, decode HTML in the review text, and lemmatize.
4. Fit a PySpark baseline: TF-IDF + logistic regression (about **79.6% accuracy**). Neutral is only ~9% of rows, so raw accuracy mostly reflects the Positive class. Neutral F1 on that baseline is about **0.44**.
5. Compare five later models (LinearSVC + Word2Vec, XGBoost + Word2Vec, HistGradientBoosting + LSA, Bi-LSTM + GloVe, MLP + frozen DistilBERT). The report’s preferred advanced model is **HistGradientBoosting on 300-d LSA**, at about **78.5% accuracy**, chosen for balanced per-class F1 rather than peak accuracy.

## Dataset

**Name:** Drug Review Dataset (Drugs.com)  
**Publisher:** UCI Machine Learning Repository  
**Download:** https://archive.ics.uci.edu/ml/machine-learning-databases/00462/drugsCom_raw.zip  
**DOI:** 10.24432/C5SK5S  
**Paper:** Felix Gräßer, Surya Kallumadi, Hagen Malberg, and Sebastian Zaunseder. 2018. *Aspect-Based Sentiment Analysis of Drug Reviews Applying Cross-Domain and Cross-Data Learning.* In *Proceedings of the 2018 International Conference on Digital Health (DH ’18)*, 121–125. DOI 10.1145/3194658.3194677

The catalog page (`archive.ics.uci.edu/dataset/462`) and the DOI resolver both return **403** right now. The zip above is the same UCI release and still returns 200. It contains `drugsComTrain_raw.tsv` and `drugsComTest_raw.tsv` (tab-separated). The CSVs in this project are that same split, exported to comma-separated files.

Patient reviews crawled from [Drugs.com](https://www.drugs.com/). The UCI release is **CC BY 4.0**. Cite Kallumadi & Gräßer (2018) and the paper above when you publish something trained on it. Reviews describe real people’s health; treat the text as sensitive even though it is public.

`uniqueID` does not overlap between the two files.

| File | Rows | Role |
| --- | --- | --- |
| `drug-review/dataset/drugsComTrain_raw.csv` | 161,297 | Official training split. This is the only file the course notebook loads. |
| `drug-review/dataset/drugsComTest_raw.csv` | 53,766 | Official held-out split. Unused by the notebook. Use this if you want a split you did not tune on. |
| **Both** | **215,063** | Full release. |

Dates run from **24 Feb 2008** through **12 Dec 2017**. Date strings are `DD-Mon-YY` (`20-May-12`), parsed with `%d-%b-%y`.

### Columns

There is no sentiment column. The 3-class label above is derived from `rating`.

| Column | Type | Meaning |
| --- | --- | --- |
| `uniqueID` | int | Review id. Unique inside each file and across the two files. |
| `drugName` | string | Drug the review is about. 3,436 distinct names in train, 2,637 in test. |
| `condition` | string | Condition the patient says the drug was for. Nullable. |
| `review` | string | Free-text patient comment. |
| `rating` | int | Overall satisfaction, 1 through 10. No missing values. |
| `date` | string | Review date, `DD-Mon-YY`. |
| `usefulCount` | int | How many readers marked the review useful. Long-tailed (train median 16, max 1,291). |

One row is one review. The same drug appears under many conditions, and the same condition covers many drugs. Birth Control alone is 28,788 of the 161,297 training rows. The next conditions are Depression, Pain, Anxiety, and Acne. The most-reviewed drugs in train are Levonorgestrel, Etonogestrel, and Ethinyl estradiol / norethindrone.

### Rating balance (official files, before any cleaning)

| Split | 1–4 Negative | 5–6 Neutral | 7–10 Positive |
| --- | --- | --- | --- |
| Train | 40,075 (24.8%) | 14,356 (8.9%) | 106,866 (66.2%) |
| Test | 13,497 (25.1%) | 4,829 (9.0%) | 35,440 (65.9%) |

Stars themselves are bimodal: 10 and 1 dominate, 4–6 are the thin middle. Train review length is about 459 characters on average (median 455, max 10,787).

### Quirks to handle before you train

These are properties of the files, not of the course split.

- **Missing condition.** 899 train rows and 295 test rows have an empty `condition`.
- **Scrape leftovers in `condition`.** About 900 train rows and 271 test rows contain HTML such as `0</span> users found this comment helpful.` instead of a condition. The course project drops missing and HTML-artifact conditions. After that filter the report counts **811** real conditions.
- **Truncated condition names.** The crawl cut some labels, for example `Bipolar Disorde`. They are real categories in this file, not typos you should silently merge unless you build a map.
- **HTML entities inside `review`.** Apostrophes are stored as `&#039;`. Decode with `html.unescape` before tokenizing.
- **Extra quote characters.** Many review strings still include a leading and trailing `"` because that quote was part of the original field, not CSV syntax.
- **A few near-empty reviews.** A handful of train and test reviews are under 5 characters.
- **Imbalance is the modelling problem.** If you keep the 5–6 band as Neutral, that class is ~9%. If you drop 5–6 and do binary sentiment, say so — the course project kept all three classes.
- **Do not mix the two splits by accident.** The notebook’s “test” set is a re-split of the official train file. `drugsComTest_raw.csv` is a different 53,766 rows. For a new project, either use the official pair, or resplit and document the seed. Do not report the notebook’s 32,412-row test as the UCI test set.

### Load it

From `drug-review/`:

```python
import html
import pandas as pd

train = pd.read_csv("dataset/drugsComTrain_raw.csv")
test = pd.read_csv("dataset/drugsComTest_raw.csv")

def clean_review(text: str) -> str:
    text = html.unescape(text or "")
    return text.strip().strip('"').strip()

train["review_clean"] = train["review"].map(clean_review)

# Same 3-class map as the course project. Drop the middle band for a binary task.
train["sentiment"] = pd.cut(
    train["rating"],
    bins=[0, 4, 6, 10],
    labels=["Negative", "Neutral", "Positive"],
)
```

## Later methods

The course notebook stops at TF-IDF logistic regression, LinearSVC, XGBoost, histogram gradient boosting, a Bi-LSTM, and a frozen DistilBERT embedding. `drug-review/drug_review/` adds methods that use the official test file and the columns the notebook left out:

| Method | What is different |
| --- | --- |
| Complement Naive Bayes, word TF-IDF | Imbalance-aware linear baseline |
| Complement Naive Bayes, char 3–5 grams | Drug-name fragments and typos |
| Logistic regression + metadata | `drugName`, `condition`, `usefulCount`, review length |
| Frank–Hall ordinal logistic | Respects Negative < Neutral < Positive |
| LightGBM on LSA + metadata | Boosting with the extra columns |
| Star regression | Predicts the 1–10 rating, then applies the same bins |
| LSA + aspect features | Effectiveness vs side-effect sentences, plus VADER |
| TextCNN | Local phrases, kernels of width 3, 4, and 5 |
| Fine-tuned DistilBERT | Updates the transformer weights. The course run froze them |

There are no aspect labels in the file. The aspect model tags sentences with a small lexicon (benefit vs side effect) and scores those sentences. It is an extra feature set, not a claim that the aspects were annotated.

DistilBERT is fine-tuned on a stratified 6,000-row draw and scored on a stratified 2,999-row draw of the official test file. The other rows in `drug-review/results/metrics.csv` use the full cleaned test file (111,070 train / 47,696 test). Macro-F1 is the column to read. Accuracy mostly tracks the Positive class.

On that test file, word TF-IDF logistic regression leads at accuracy 0.857 and macro-F1 0.688 (Neutral F1 0.332). LightGBM on LSA plus drug, condition, usefulCount, and review length has the best Neutral F1, 0.453, at accuracy 0.757. One epoch of DistilBERT on 6,000 reviews does not beat TF-IDF.

## Environment

Python 3.14, managed with uv, from `drug-review/`. Java is not required for these later methods. The course notebook’s PySpark cells still need a JDK if you re-run those sections.

```bash
cd drug-review
uv sync
uv run python -m drug_review.pipeline
```

`DR_FAST=1` runs a small slice to check the code. `DR_SKIP_BERT=1` skips the transformer fine-tune. The notebook for the later methods is `drug-review/analysis.ipynb`. After `uv sync`, select the `drug-review` kernel.

## Layout

```
project-code.ipynb                  course EDA, baseline, and the five original models
project-report.docx                 course report (local, gitignored)
project-slides.pdf                  course slides (local, gitignored)
drug-review/
  dataset/drugsComTrain_raw.csv     official train split
  dataset/drugsComTest_raw.csv      official test split
  drug_review/                      later methods (official split)
  analysis.ipynb                    notebook for those methods
  results/                          accuracy, macro-F1, per-condition table, chart
  pyproject.toml                    uv environment
```

`.venv/` is local and is gitignored. `drug-review/results/` holds the scores from the last full run (`metrics.csv`, `per_condition.csv`, `comparison.png`). Checkpoint files (`*.pt`, `*.bin`) are gitignored.

## Citation

```bibtex
@misc{kallumadi2018drugreview,
  author       = {Kallumadi, Surya and Gr{\"a}{\ss}er, Felix},
  title        = {Drug Review Dataset (Drugs.com)},
  year         = {2018},
  howpublished = {UCI Machine Learning Repository},
  doi          = {10.24432/C5SK5S},
  url          = {https://archive.ics.uci.edu/ml/machine-learning-databases/00462/drugsCom_raw.zip}
}

@inproceedings{grasser2018aspect,
  author    = {Gr{\"a}{\ss}er, Felix and Kallumadi, Surya and Malberg, Hagen and Zaunseder, Sebastian},
  title     = {Aspect-Based Sentiment Analysis of Drug Reviews Applying Cross-Domain and Cross-Data Learning},
  booktitle = {Proceedings of the 2018 International Conference on Digital Health},
  year      = {2018},
  pages     = {121--125},
  doi       = {10.1145/3194658.3194677}
}
```
