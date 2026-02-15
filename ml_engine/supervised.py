"""
Supervised Prediction (Explainable, No Black Box)

We implement logistic regression from scratch (gradient descent) for:
- Probability of achieving grade A (>=75)
- Probability of achieving grade A+ (>=80)

This is intended for academic demonstration:
- clear feature list
- transparent loss function
- simple evaluation metrics
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import math
import random


def sigmoid(z: float) -> float:
    # numerically stable sigmoid
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def standardize(X: List[List[float]]) -> Tuple[List[List[float]], List[float], List[float]]:
    if not X:
        return X, [], []
    d = len(X[0])
    mean = [0.0] * d
    std = [0.0] * d

    for j in range(d):
        col = [row[j] for row in X]
        m = sum(col) / len(col)
        v = sum((x - m) ** 2 for x in col) / len(col)
        s = math.sqrt(v) or 1.0
        mean[j] = m
        std[j] = s

    Xs = [[(row[j] - mean[j]) / std[j] for j in range(d)] for row in X]
    return Xs, mean, std


def train_test_split(X, y, test_ratio=0.2, seed=42):
    idx = list(range(len(X)))
    rnd = random.Random(seed)
    rnd.shuffle(idx)
    cut = int(len(idx) * (1 - test_ratio))
    train_idx = idx[:cut]
    test_idx = idx[cut:]
    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]
    return X_train, X_test, y_train, y_test


@dataclass(frozen=True)
class LogRegModel:
    feature_names: List[str]
    w: List[float]
    b: float
    mean: List[float]
    std: List[float]


def fit_logreg(
    X: List[List[float]],
    y: List[int],
    feature_names: List[str],
    lr: float = 0.1,
    l2: float = 0.001,
    epochs: int = 300,
    seed: int = 42,
) -> LogRegModel:
    """
    Logistic regression with L2 regularization.
    Loss: -log-likelihood + l2*||w||^2
    """
    Xs, mean, std = standardize(X)
    rnd = random.Random(seed)
    d = len(Xs[0]) if Xs else 0
    w = [rnd.uniform(-0.01, 0.01) for _ in range(d)]
    b = 0.0

    n = len(Xs)
    for _ in range(epochs):
        # batch gradient descent
        dw = [0.0] * d
        db = 0.0
        for xi, yi in zip(Xs, y):
            p = sigmoid(dot(w, xi) + b)
            err = p - yi
            for j in range(d):
                dw[j] += err * xi[j]
            db += err

        # average + regularize
        for j in range(d):
            dw[j] = (dw[j] / n) + (2.0 * l2 * w[j])
        db = db / n

        # update
        for j in range(d):
            w[j] -= lr * dw[j]
        b -= lr * db

    return LogRegModel(feature_names=feature_names, w=w, b=b, mean=mean, std=std)


def predict_proba(model: LogRegModel, x: List[float]) -> float:
    xs = [(x[j] - model.mean[j]) / model.std[j] for j in range(len(x))]
    return sigmoid(dot(model.w, xs) + model.b)


def evaluate_binary(model: LogRegModel, X: List[List[float]], y: List[int], threshold: float = 0.5) -> Dict[str, float]:
    tp = fp = tn = fn = 0
    for xi, yi in zip(X, y):
        p = predict_proba(model, xi)
        pred = 1 if p >= threshold else 0
        if pred == 1 and yi == 1:
            tp += 1
        elif pred == 1 and yi == 0:
            fp += 1
        elif pred == 0 and yi == 0:
            tn += 1
        else:
            fn += 1

    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }

