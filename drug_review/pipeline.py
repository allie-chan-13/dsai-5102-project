"""Train the later methods and write a comparison table."""

from __future__ import annotations

import os
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from drug_review.aspects import aspect_features
from drug_review.data import LABELS, RESULTS, load_official
from drug_review.models import (
    char_vectorizer,
    complement_nb,
    frank_hall,
    lightgbm_classifier,
    logistic,
    metadata_matrix,
    ordinal_rating_regression,
    per_condition,
    reduce_svd,
    score,
    train_distilbert,
    train_textcnn,
    word_vectorizer,
)


def _maybe_shrink(train: pd.DataFrame, test: pd.DataFrame, fast: bool):
    if not fast:
        return train, test, 6000, 3000, 3
    train = train.groupby("label", group_keys=False).head(400)
    test = test.groupby("label", group_keys=False).head(150)
    return train.reset_index(drop=True), test.reset_index(drop=True), 120, 80, 1


def run(fast: bool = False) -> pd.DataFrame:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    started = perf_counter()
    train, test, bert_train_n, bert_test_n, cnn_epochs = _maybe_shrink(*load_official(), fast)
    print(
        f"official split after cleaning: train {len(train):,}  test {len(test):,}",
        flush=True,
    )
    print("class shares (train):", train["sentiment"].value_counts(normalize=True).round(3).to_dict(), flush=True)

    y_train = train["label"].to_numpy()
    y_test = test["label"].to_numpy()
    rows: list[dict] = []
    full_predictions: dict[str, np.ndarray] = {}

    print("fitting word TF-IDF...", flush=True)
    word = word_vectorizer()
    word_train = word.fit_transform(train["review_norm"])
    word_test = word.transform(test["review_norm"])

    print("ComplementNB word...", flush=True)
    pred = complement_nb(word_train, y_train, word_test)
    rows.append(score("ComplementNB word TF-IDF", y_test, pred, len(train), "word and bigram TF-IDF, imbalance-aware NB"))
    full_predictions[rows[-1]["model"]] = pred

    print("logistic regression word TF-IDF...", flush=True)
    pred = logistic(word_train, y_train, word_test, balanced=False)
    rows.append(score("LogReg word TF-IDF", y_test, pred, len(train), "same family as the course baseline, official test file, no class weights"))
    full_predictions[rows[-1]["model"]] = pred

    print("logistic regression + drug/condition/usefulCount...", flush=True)
    meta_train, meta_test = metadata_matrix(train, test, word_train, word_test)
    pred = logistic(meta_train, y_train, meta_test, balanced=True)
    rows.append(score("LogReg TF-IDF + metadata", y_test, pred, len(train), "word TF-IDF plus drug, condition, log usefulCount, review length; balanced"))
    full_predictions[rows[-1]["model"]] = pred
    del meta_train, meta_test

    print("Frank-Hall ordinal logistic...", flush=True)
    pred = frank_hall(word_train, y_train, word_test)
    rows.append(score("Frank-Hall ordinal LogReg", y_test, pred, len(train), "two binary models that respect Negative < Neutral < Positive"))
    full_predictions[rows[-1]["model"]] = pred

    print("fitting character TF-IDF...", flush=True)
    chars = char_vectorizer()
    char_train = chars.fit_transform(train["review_norm"])
    char_test = chars.transform(test["review_norm"])
    pred = complement_nb(char_train, y_train, char_test)
    rows.append(score("ComplementNB char n-grams", y_test, pred, len(train), "character 3-5 grams, keeps drug-name fragments and typos"))
    full_predictions[rows[-1]["model"]] = pred
    del char_train, char_test

    print("SVD of word TF-IDF...", flush=True)
    svd_train, svd_test = reduce_svd(word_train, word_test, components=120 if not fast else 30)
    del word_train, word_test

    print("LightGBM on SVD + metadata...", flush=True)
    drug_enc_frame_train = train[["drugName", "condition"]].copy()
    # Reuse the dense metadata columns beside the SVD. Categories are frequency-capped.
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    drug_enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=40)
    cond_enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=40)
    scaler = StandardScaler()
    numeric_train = scaler.fit_transform(
        np.column_stack([np.log1p(train["usefulCount"]), train["review_len"]])
    )
    numeric_test = scaler.transform(
        np.column_stack([np.log1p(test["usefulCount"]), test["review_len"]])
    )
    dense_train = np.hstack(
        [
            svd_train,
            drug_enc.fit_transform(drug_enc_frame_train[["drugName"]]),
            cond_enc.fit_transform(train[["condition"]]),
            numeric_train,
        ]
    ).astype(np.float32)
    dense_test = np.hstack(
        [
            svd_test,
            drug_enc.transform(test[["drugName"]]),
            cond_enc.transform(test[["condition"]]),
            numeric_test,
        ]
    ).astype(np.float32)
    pred, booster_name = lightgbm_classifier(dense_train, y_train, dense_test)
    rows.append(score(f"{booster_name} SVD + metadata", y_test, pred, len(train), "120-d LSA plus drug, condition, usefulCount, length; balanced"))
    full_predictions[rows[-1]["model"]] = pred
    del dense_train, dense_test

    print("3-class logistic on the same SVD (control for the ordinal and aspect models)...", flush=True)
    control = LogisticRegression(max_iter=500, class_weight="balanced")
    control.fit(svd_train, y_train)
    pred = control.predict(svd_test)
    rows.append(score("LogReg SVD 3-class", y_test, pred, len(train), "control: same 120-d LSA vectors, ordinary 3-class labels"))
    full_predictions[rows[-1]["model"]] = pred

    print("ordinal star regression...", flush=True)
    pred = ordinal_rating_regression(svd_train, train["rating"].to_numpy(), svd_test)
    rows.append(score("Ordinal star regression", y_test, pred, len(train), "predicts the 1-10 rating on the same SVD, then bins 1-4 / 5-6 / 7-10"))
    full_predictions[rows[-1]["model"]] = pred

    print("aspect and lexicon features...", flush=True)
    aspect_train = aspect_features(train["review"])
    aspect_test = aspect_features(test["review"])
    aspect_model = LogisticRegression(max_iter=500, class_weight="balanced")
    aspect_model.fit(np.hstack([svd_train, aspect_train.to_numpy()]), y_train)
    pred = aspect_model.predict(np.hstack([svd_test, aspect_test.to_numpy()]))
    coverage = float(((aspect_train["n_effect"] + aspect_train["n_side"]) > 0).mean())
    rows.append(
        score(
            "LogReg SVD + aspects",
            y_test,
            pred,
            len(train),
            f"same SVD plus VADER and effectiveness/side-effect sentence cues; {coverage:.0%} of train reviews have an aspect sentence",
        )
    )
    full_predictions[rows[-1]["model"]] = pred
    del svd_train, svd_test

    print("TextCNN...", flush=True)
    pred = train_textcnn(train["review_norm"], y_train, test["review_norm"], y_test, epochs=cnn_epochs)
    rows.append(score("TextCNN", y_test, pred, len(train), f"word-level CNN, kernels 3/4/5, {cnn_epochs} epoch(s), class-weighted loss"))
    full_predictions[rows[-1]["model"]] = pred

    if os.environ.get("DR_SKIP_BERT") == "1":
        print("skipping DistilBERT (DR_SKIP_BERT=1)", flush=True)
    else:
        print("fine-tuning DistilBERT...", flush=True)
        pred, y_bert, n_tr, n_te = train_distilbert(train, test, bert_train_n, bert_test_n, epochs=1)
        rows.append(
            score(
                "DistilBERT fine-tuned",
                y_bert,
                pred,
                n_tr,
                f"end-to-end fine-tune, stratified {n_tr} train / {n_te} test, 1 epoch, max length 96. Not the full test file.",
            )
        )

    results = pd.DataFrame(rows).sort_values(["macro_f1", "accuracy"], ascending=False)
    RESULTS.mkdir(exist_ok=True)
    results.to_csv(RESULTS / "metrics.csv", index=False)

    full = results[results["n_test"] == len(test)]
    best_name = full.iloc[0]["model"]
    condition = per_condition(test, y_test, full_predictions[best_name])
    condition.to_csv(RESULTS / "per_condition.csv", index=False)

    plot = results.sort_values("macro_f1")
    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(plot))
    ax.barh(y - 0.15, plot["accuracy"], height=0.3, label="Accuracy", color="#4C78A8")
    ax.barh(y + 0.15, plot["macro_f1"], height=0.3, label="Macro F1", color="#F58518")
    ax.set_yticks(y)
    ax.set_yticklabels(plot["model"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Score on the official test split")
    ax.set_title("Later methods")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS / "comparison.png", dpi=140)
    plt.close(fig)

    print()
    print(results.drop(columns=["notes"]).round(4).to_string(index=False), flush=True)
    print()
    print(f"best full-test model by macro-F1: {best_name}", flush=True)
    print(condition.round(4).to_string(index=False), flush=True)
    print(f"finished in {(perf_counter() - started) / 60:.1f} min", flush=True)
    return results


def main() -> None:
    run(fast=os.environ.get("DR_FAST") == "1")


if __name__ == "__main__":
    main()
