"""
scikit-learn Pipelines for Academic Prediction (Explainable)

Students do NOT input final exam marks.
We predict final marks (0..50) using pre-final signals:
- assignment/hand (0..20)
- attendance (0..5)
- quiz/CT (0..15)
- midterm (0..30)
- previous CGPA (0..4)

Models:
1) Regression: Ridge (predict final marks)
2) Classification: LogisticRegression (P(A), P(A+))

Explainability:
- Linear models expose coefficients per feature (direction + strength).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import os
import pickle

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression


FEATURE_NAMES = [
    # Naming matches the academic dataset spec
    "assignment_marks",        # 0..5
    "attendance_percentage",   # 0..100
    "quiz_marks",              # 0..10
    "midterm_marks",           # 0..30
    "previous_cgpa",           # 0..4
]


def build_X(rows: List[Dict]) -> np.ndarray:
    return np.array([[float(r[n]) for n in FEATURE_NAMES] for r in rows], dtype=float)


def train_regressor() -> Pipeline:
    # Ridge is stable, explainable, and works well for small-noise synthetic academic data.
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0, random_state=42)),
    ])


def train_classifier() -> Pipeline:
    # Logistic regression is explainable and produces calibrated-ish probabilities for academic use.
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, solver="lbfgs")),
    ])


def save_model(model: Pipeline, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str) -> Pipeline:
    with open(path, "rb") as f:
        return pickle.load(f)


def explain_linear_model(pipeline: Pipeline, top_k: int = 5) -> List[Tuple[str, float]]:
    """
    Returns top_k features by absolute coefficient magnitude.
    Works for Ridge and LogisticRegression.
    """
    model = pipeline.named_steps["model"]
    coef = getattr(model, "coef_", None)
    if coef is None:
        return []
    coef = np.array(coef).reshape(-1)
    pairs = list(zip(FEATURE_NAMES, coef.tolist()))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    return pairs[:top_k]
