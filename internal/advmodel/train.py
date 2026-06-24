#!/usr/bin/env python3
"""train.py — the reproducible training run for the advisory adjudication model.

This is the consumer of the harvest LabelRow corpus (issue #580): it reads the
frozen, floor-labeled content-bearing corpus (testdata/corpus.jsonl — every label
re-witnessed against the REAL adjudicator floor by corpus_test.go), trains a small
logistic-regression classifier over a bag of call tokens, writes the model
artifact (testdata/adjudicator.json), and prints held-out precision/recall/F1 vs
the stock reference.

THE MODEL. It is deliberately SMALL: a logistic regression over binary bag-of-
tokens features (the "small syscall/adjudication model" the issue's acceptance
permits), not a fine-tune of the fused SmolLM2 forward pass (that needs GPU +
weights + hours and is out of scope here). The stock reference it is compared
against is the inert/untrained artifact — the stock SmolLM2 emits NO adjudication
signal, so its deny-class F1 is 0 by definition; this run's contribution is going
from "no signal" to "a real learned signal".

FEATURIZER PARITY. The token extractor below MUST match Go's advmodel.Tokens()
byte-for-byte (lower-case(tool + "\\x00" + args), regex [a-z0-9_]+, unique). A
drift would make the loaded weights score the wrong features; advmodel_test pins
the Go side. Deterministic: zero init, fixed iterations, no RNG — same input
gives the same artifact and the same numbers on every platform.

USAGE
    python internal/advmodel/train.py
        # reads testdata/corpus.jsonl, writes testdata/adjudicator.json,
        # prints the held-out eval.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "testdata", "corpus.jsonl")
ARTIFACT = os.path.join(HERE, "testdata", "adjudicator.json")

SCHEMA = "fak-advmodel/v1"
TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokens(tool: str, args: str) -> set[str]:
    """Mirror advmodel.Tokens() exactly: lower-case(tool + \\x00 + args), then
    unique alphanumeric/underscore runs."""
    s = (tool + "\x00" + args).lower()
    return set(TOKEN_RE.findall(s))


def load_corpus(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def split(rows):
    """Deterministic held-out split: every 5th row (indices 4,9,14,...) is
    held out, the rest train. Same rows land in each split on every run, so the
    eval is reproducible. Holds back ~20%, leaving the rest to learn from."""
    train, held = [], []
    for i, r in enumerate(rows):
        (held if i % 5 == 4 else train).append(r)
    return train, held


def featurize(rows, vocab):
    """Binary bag-of-tokens feature matrix over `vocab` (sorted token list)."""
    X = np.zeros((len(rows), len(vocab)), dtype=np.float64)
    y = np.zeros(len(rows), dtype=np.float64)
    idx = {t: j for j, t in enumerate(vocab)}
    for i, r in enumerate(rows):
        y[i] = 1.0 if r["deny"] else 0.0
        for t in tokens(r["tool"], r["args"]):
            j = idx.get(t)
            if j is not None:
                X[i, j] = 1.0
    return X, y


def train_logreg(X, y, lr=0.5, iters=4000, l2=1e-3):
    """Binary logistic regression by full-batch gradient descent with L2.
    Zero init, deterministic. Returns weight vector + bias."""
    n, d = X.shape
    w = np.zeros(d, dtype=np.float64)
    b = 0.0
    for _ in range(iters):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        err = p - y
        gw = (X.T @ err) / n + l2 * w
        gb = err.mean()
        w -= lr * gw
        b -= lr * gb
    return w, b


def prf(y_true, y_pred):
    """Precision/recall/F1 for the positive (deny) class."""
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def main():
    rows = load_corpus(CORPUS)
    train_rows, held_rows = split(rows)

    # Vocabulary from TRAIN only — held-out tokens unseen in train contribute
    # nothing at score time, exactly as they would on a live unseen call.
    vocab = sorted({t for r in train_rows for t in tokens(r["tool"], r["args"])})

    Xtr, ytr = featurize(train_rows, vocab)
    w, b = train_logreg(Xtr, ytr)

    # Decision boundary: logit >= 0 (sigmoid >= 0.5). The Go scorer uses
    # Threshold=0.0, so this is the exact boundary it applies.
    Xh, yh = featurize(held_rows, vocab)
    held_logit = Xh @ w + b
    held_pred = (held_logit >= 0.0).astype(float)
    prec, rec, f1 = prf(yh, held_pred)

    # Stock reference: the inert (untrained) artifact — the stock SmolLM2 emits
    # no adjudication signal, so it predicts no denies (all allow). Deny-class
    # F1 is 0. Also report the majority-class baseline for context.
    stock_pred = np.zeros_like(yh)
    _, _, stock_f1 = prf(yh, stock_pred)
    maj_pred = np.ones_like(yh)  # deny is the majority class in this corpus
    maj_prec, maj_rec, maj_f1 = prf(yh, maj_pred)

    # Train-split fit (sanity: the model separates its training data).
    train_pred = ((Xtr @ w + b) >= 0.0).astype(float)
    tr_prec, tr_rec, tr_f1 = prf(ytr, train_pred)

    features = {t: float(w[j]) for j, t in enumerate(vocab)}
    art = {
        "schema": SCHEMA,
        "bias": float(b),
        "threshold": 0.0,
        "features": features,
        "meta": {
            "train_rows": len(train_rows),
            "held_rows": len(held_rows),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "stock_ref_f1": round(stock_f1, 4),
            "majority_f1": round(maj_f1, 4),
            "train_f1": round(tr_f1, 4),
            "trained": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    }
    with open(ARTIFACT, "w", encoding="utf-8") as fh:
        json.dump(art, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(f"advisory adjudication model — training run over harvest corpus")
    print(f"corpus: {len(rows)} rows  (train {len(train_rows)} / held-out {len(held_rows)})")
    print(f"vocab:  {len(vocab)} tokens (from train split only)")
    print()
    print(f"held-out (deny class): precision={prec:.3f} recall={rec:.3f} f1={f1:.3f}")
    print(f"  stock reference (inert, no signal): f1={stock_f1:.3f}")
    print(f"  majority-class baseline (deny-all):  f1={maj_f1:.3f}  (p={maj_prec:.3f} r={maj_rec:.3f})")
    print(f"train-split fit (sanity):              f1={tr_f1:.3f}  (p={tr_prec:.3f} r={tr_rec:.3f})")
    print()
    print(f"wrote {ARTIFACT}")
    print(f"FAIL-CLOSED: the Go Adjudicator loads this and can only corroborate a deny")


if __name__ == "__main__":
    main()
