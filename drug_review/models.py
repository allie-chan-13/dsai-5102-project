"""Classifiers that the course notebook did not train.

The course notebook already ran TF-IDF logistic regression, LinearSVC, XGBoost,
histogram gradient boosting, a Bi-LSTM, and a frozen DistilBERT embedding.
Everything here is a different feature set, label, or training setup, scored on
the official UCI test file.
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from drug_review.data import LABELS

PAD, UNK = 0, 1


def score(name: str, y_true, y_pred, n_train: int, notes: str) -> dict:
    per_class = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2])
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "f1_negative": per_class[0],
        "f1_neutral": per_class[1],
        "f1_positive": per_class[2],
        "n_train": n_train,
        "n_test": int(len(y_true)),
        "notes": notes,
    }


def word_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=5,
        max_features=30000,
        sublinear_tf=True,
        dtype=np.float32,
    )


def char_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=5,
        max_features=30000,
        sublinear_tf=True,
        dtype=np.float32,
    )


def complement_nb(x_train, y_train, x_test):
    model = ComplementNB(alpha=0.5)
    model.fit(x_train, y_train)
    return model.predict(x_test)


def logistic(x_train, y_train, x_test, balanced: bool):
    model = LogisticRegression(
        max_iter=300,
        solver="saga",
        tol=1e-3,
        C=2.0,
        class_weight="balanced" if balanced else None,
    )
    model.fit(x_train, y_train)
    return model.predict(x_test)


def metadata_matrix(train: pd.DataFrame, test: pd.DataFrame, word_train, word_test):
    """Word TF-IDF plus drug, condition, usefulCount, and review length."""
    drug_enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True, max_categories=40)
    cond_enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True, max_categories=40)
    drug_train = drug_enc.fit_transform(train[["drugName"]])
    drug_test = drug_enc.transform(test[["drugName"]])
    cond_train = cond_enc.fit_transform(train[["condition"]])
    cond_test = cond_enc.transform(test[["condition"]])

    scaler = StandardScaler()
    numeric_train = np.column_stack(
        [np.log1p(train["usefulCount"].to_numpy()), train["review_len"].to_numpy()]
    )
    numeric_test = np.column_stack(
        [np.log1p(test["usefulCount"].to_numpy()), test["review_len"].to_numpy()]
    )
    numeric_train = sparse.csr_matrix(scaler.fit_transform(numeric_train).astype(np.float32))
    numeric_test = sparse.csr_matrix(scaler.transform(numeric_test).astype(np.float32))
    x_train = sparse.hstack([word_train, drug_train, cond_train, numeric_train], format="csr")
    x_test = sparse.hstack([word_test, drug_test, cond_test, numeric_test], format="csr")
    return x_train, x_test


def frank_hall(x_train, y_train, x_test) -> np.ndarray:
    """Ordinal model for Negative < Neutral < Positive (Frank & Hall, 2001)."""
    greater = []
    for threshold in (0, 1):
        binary = (y_train > threshold).astype(int)
        model = LogisticRegression(
            max_iter=200,
            solver="liblinear",
            class_weight="balanced",
            tol=1e-3,
        )
        model.fit(x_train, binary)
        greater.append(model.predict_proba(x_test)[:, 1])
    above_neg, above_neu = greater
    probs = np.vstack([1.0 - above_neg, above_neg - above_neu, above_neu]).T
    probs = np.clip(probs, 1e-6, None)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs.argmax(axis=1)


def reduce_svd(x_train, x_test, components: int = 150):
    svd = TruncatedSVD(n_components=components, random_state=42)
    return svd.fit_transform(x_train), svd.transform(x_test)


def lightgbm_classifier(x_train, y_train, x_test):
    try:
        import lightgbm as lgb
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier

        model = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.08,
            class_weight="balanced",
            random_state=42,
        )
        model.fit(x_train, y_train)
        return model.predict(x_test), "HistGradientBoosting fallback"
    model = lgb.LGBMClassifier(
        n_estimators=250,
        learning_rate=0.08,
        num_leaves=31,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(x_train, y_train)
    return model.predict(x_test), "LightGBM"


def rating_to_class(pred_rating: np.ndarray) -> np.ndarray:
    pred = np.clip(pred_rating, 1.0, 10.0)
    return np.where(pred <= 4, 0, np.where(pred <= 6, 1, 2)).astype(int)


def ordinal_rating_regression(x_train, rating_train, x_test) -> np.ndarray:
    """Predict the 1-10 star, then apply the same bins as the 3-class label."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    model = HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.08,
        random_state=42,
    )
    model.fit(x_train, rating_train)
    return rating_to_class(model.predict(x_test))


def per_condition(test: pd.DataFrame, y_true, y_pred, top: int = 5) -> pd.DataFrame:
    frame = test[["condition"]].copy()
    frame["y_true"] = np.asarray(y_true)
    frame["y_pred"] = np.asarray(y_pred)
    counts = frame["condition"].value_counts().head(top)
    rows = []
    for condition in counts.index:
        part = frame[frame["condition"] == condition]
        rows.append(
            {
                "condition": condition,
                "n": int(len(part)),
                "accuracy": accuracy_score(part["y_true"], part["y_pred"]),
                "macro_f1": f1_score(part["y_true"], part["y_pred"], average="macro", labels=[0, 1, 2], zero_division=0),
            }
        )
    return pd.DataFrame(rows)


def _tokenize(texts: pd.Series) -> list[list[str]]:
    return [text.split() for text in texts.fillna("")]


def _vocab(tokenized: list[list[str]], limit: int = 20000) -> dict[str, int]:
    counts: dict[str, int] = {}
    for tokens in tokenized:
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
    ranked = sorted((count, token) for token, count in counts.items() if count >= 2)
    ranked.sort(reverse=True)
    table = {token: index + 2 for index, (_, token) in enumerate(ranked[: limit - 2])}
    return table


def _encode(tokenized: list[list[str]], table: dict[str, int], length: int) -> np.ndarray:
    encoded = np.zeros((len(tokenized), length), dtype=np.int64)
    for row, tokens in enumerate(tokenized):
        ids = [table.get(token, UNK) for token in tokens[:length]]
        if ids:
            encoded[row, : len(ids)] = ids
    return encoded


class TextCNN(nn.Module):
    def __init__(self, vocab_size: int, embed: int = 64, channels: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed, padding_idx=PAD)
        self.convs = nn.ModuleList([nn.Conv1d(embed, channels, kernel) for kernel in (3, 4, 5)])
        self.dropout = nn.Dropout(0.3)
        self.output = nn.Linear(channels * 3, 3)

    def forward(self, tokens):
        hidden = self.embedding(tokens).transpose(1, 2)
        pooled = [torch.amax(torch.relu(conv(hidden)), dim=2) for conv in self.convs]
        return self.output(self.dropout(torch.cat(pooled, dim=1)))


def train_textcnn(train_text, y_train, test_text, y_test, epochs: int = 3, length: int = 120):
    random.seed(42)
    torch.manual_seed(42)
    tokenized_train = _tokenize(train_text)
    table = _vocab(tokenized_train)
    x_train = np.array(_encode(tokenized_train, table, length), dtype=np.int64, copy=True)
    x_test = np.array(_encode(_tokenize(test_text), table, length), dtype=np.int64, copy=True)
    y_train_t = np.array(y_train, dtype=np.int64, copy=True)
    weights = torch.tensor(
        [len(y_train_t) / max((y_train_t == label).sum(), 1) for label in (0, 1, 2)],
        dtype=torch.float32,
    )
    model = TextCNN(len(table) + 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(weight=weights)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train_t)),
        batch_size=256,
        shuffle=True,
    )
    model.train()
    for epoch in range(epochs):
        total = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total += float(loss.item())
        print(f"  textcnn epoch {epoch + 1}/{epochs} loss {total / max(len(loader), 1):.4f}", flush=True)
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(x_test), 512):
            logits = model(torch.from_numpy(x_test[start : start + 512]))
            preds.append(logits.argmax(dim=1).numpy())
    return np.concatenate(preds)


def _stratified_frame(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n >= len(frame):
        return frame
    parts = []
    for _, group in frame.groupby("label"):
        take = max(1, int(round(len(group) * n / len(frame))))
        parts.append(group.sample(n=min(take, len(group)), random_state=seed))
    sampled = pd.concat(parts, ignore_index=True)
    if len(sampled) > n:
        sampled = sampled.sample(n=n, random_state=seed)
    return sampled.reset_index(drop=True)


def train_distilbert(train: pd.DataFrame, test: pd.DataFrame, train_n: int, test_n: int, epochs: int = 1):
    """Fine-tune DistilBERT end to end. The course notebook froze it and fit an MLP."""
    from torch.optim import AdamW
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    train_s = _stratified_frame(train, train_n, 42)
    test_s = _stratified_frame(test, test_n, 42)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=3,
    )
    counts = np.array([(train_s["label"] == label).sum() for label in (0, 1, 2)], dtype=np.float32)
    weights = torch.tensor(len(train_s) / np.maximum(counts, 1.0))
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    def batches(frame, batch_size, shuffle, seed):
        order = np.arange(len(frame))
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(order)
        texts = frame["review"].tolist()
        labels = frame["label"].to_numpy()
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            encoded = tokenizer(
                [texts[i] for i in index],
                truncation=True,
                padding=True,
                max_length=96,
                return_tensors="pt",
            )
            yield encoded, torch.tensor(labels[index], dtype=torch.long)

    model.train()
    for epoch in range(epochs):
        total = 0.0
        steps = 0
        for encoded, batch_y in batches(train_s, 16, shuffle=True, seed=42 + epoch):
            optimizer.zero_grad()
            logits = model(**encoded).logits
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            steps += 1
            if steps % 50 == 0:
                print(f"  distilbert epoch {epoch + 1} step {steps} loss {total / steps:.4f}", flush=True)
        print(f"  distilbert epoch {epoch + 1}/{epochs} loss {total / max(steps, 1):.4f}", flush=True)

    model.eval()
    preds = []
    with torch.no_grad():
        for encoded, _ in batches(test_s, 32, shuffle=False, seed=0):
            preds.append(model(**encoded).logits.argmax(dim=1).numpy())
    return np.concatenate(preds), test_s["label"].to_numpy(), len(train_s), len(test_s)
