"""
Explainable Academic "ML" for Study Recommendation (Django-friendly)

Outputs (explainable):
- performance_cluster: Weak / Average / Strong
- study_level: Beginner / Intermediate / Advanced
- explanation: why the student was assigned (rule-based or nearest centroid)

We use transparent feature engineering:
- avg_midterm, avg_total, weak_ratio, strong_ratio, dispersion

Clustering is optional. If cohort data is not provided (or too small),
the system falls back to a rule-based decision (viva-friendly).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math


@dataclass(frozen=True)
class SubjectRecord:
    """
    One subject record for a student.
    midterm: 0..30
    total:   0..100
    """
    semester: int
    subject: str
    midterm: float
    total: float


@dataclass(frozen=True)
class StudentRecommendation:
    performance_cluster: str   # Weak / Average / Strong
    study_level: str           # Beginner / Intermediate / Advanced
    features: Dict[str, float]
    explanation: str


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: List[float]) -> float:
    if not xs:
        return 0.0
    mu = _mean(xs)
    var = _mean([(x - mu) ** 2 for x in xs])
    return math.sqrt(var)


def build_features(records: List[SubjectRecord]) -> Dict[str, float]:
    totals = [r.total for r in records]
    mids = [r.midterm for r in records]

    n = len(records)
    weak_ratio = (sum(1 for t in totals if t < 50) / n) if n else 0.0
    strong_ratio = (sum(1 for t in totals if t >= 80) / n) if n else 0.0

    return {
        "avg_midterm": _mean(mids),
        "avg_total": _mean(totals),
        "weak_ratio": weak_ratio,
        "strong_ratio": strong_ratio,
        "dispersion": _std(totals),
        "n_subjects": float(n),
    }


def rule_based_cluster(features: Dict[str, float]) -> Tuple[str, str]:
    """
    Explainable decision (good for viva):
    adjusted = avg_total - 10*weak_ratio - 0.15*dispersion
    """
    avg_total = features["avg_total"]
    weak_ratio = features["weak_ratio"]
    dispersion = features["dispersion"]

    adjusted = avg_total - (weak_ratio * 10.0) - (dispersion * 0.15)

    if adjusted < 50:
        return "Weak", "Beginner"
    if adjusted < 80:
        return "Average", "Intermediate"
    return "Strong", "Advanced"


def _euclid(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def kmeans_fit_predict(X: List[List[float]], k: int = 3, max_iter: int = 30) -> Tuple[List[int], List[List[float]]]:
    if len(X) < k:
        raise ValueError("Not enough samples for KMeans.")

    centroids = [X[i][:] for i in range(k)]  # deterministic init
    labels = [0] * len(X)

    for _ in range(max_iter):
        changed = False

        for i, x in enumerate(X):
            best_j = min(range(k), key=lambda j: _euclid(x, centroids[j]))
            if best_j != labels[i]:
                labels[i] = best_j
                changed = True

        new_centroids = [[0.0] * len(X[0]) for _ in range(k)]
        counts = [0] * k
        for lab, x in zip(labels, X):
            counts[lab] += 1
            for d in range(len(x)):
                new_centroids[lab][d] += x[d]

        for j in range(k):
            if counts[j] == 0:
                new_centroids[j] = centroids[j]
            else:
                new_centroids[j] = [v / counts[j] for v in new_centroids[j]]

        centroids = new_centroids
        if not changed:
            break

    return labels, centroids


def _cluster_name_by_centroid(centroid: List[float]) -> str:
    avg_total = centroid[1]  # vector ordering below
    if avg_total < 50:
        return "Weak"
    if avg_total < 80:
        return "Average"
    return "Strong"


def _study_level_from_cluster(cluster: str) -> str:
    return {"Weak": "Beginner", "Average": "Intermediate", "Strong": "Advanced"}[cluster]


def recommend_student_level(
    student_records: List[SubjectRecord],
    cohort_records: Optional[Dict[int, List[SubjectRecord]]] = None,
) -> StudentRecommendation:
    feats = build_features(student_records)

    if feats["n_subjects"] == 0:
        return StudentRecommendation(
            performance_cluster="Weak",
            study_level="Beginner",
            features=feats,
            explanation="No subject records found. Defaulting to Beginner support level.",
        )

    use_kmeans = False
    if cohort_records:
        valid = []
        for _, recs in cohort_records.items():
            f = build_features(recs)
            if f["n_subjects"] >= 3:
                valid.append(f)
        if len(valid) >= 6:
            use_kmeans = True

    if not use_kmeans:
        cluster, level = rule_based_cluster(feats)
        exp = (
            "Rule-based classification on engineered features.\n"
            f"avg_total={feats['avg_total']:.1f}, weak_ratio={feats['weak_ratio']:.2f}, "
            f"dispersion={feats['dispersion']:.1f}.\n"
            f"Result: {cluster} -> {level}."
        )
        return StudentRecommendation(cluster, level, feats, exp)

    def vec(f: Dict[str, float]) -> List[float]:
        return [f["avg_midterm"], f["avg_total"], f["weak_ratio"], f["strong_ratio"], f["dispersion"]]

    cohort_X = []
    cohort_ids = []
    for sid, recs in cohort_records.items():
        f = build_features(recs)
        if f["n_subjects"] >= 3:
            cohort_ids.append(sid)
            cohort_X.append(vec(f))

    labels, centroids = kmeans_fit_predict(cohort_X, k=3)
    student_vec = vec(feats)
    nearest = min(range(3), key=lambda j: _euclid(student_vec, centroids[j]))

    cluster = _cluster_name_by_centroid(centroids[nearest])
    level = _study_level_from_cluster(cluster)
    dist = _euclid(student_vec, centroids[nearest])

    exp = (
        "KMeans-style clustering on engineered features (pure Python).\n"
        f"Assigned to '{cluster}' by nearest centroid (distance={dist:.2f}).\n"
        "Study level mapping: Weak->Beginner, Average->Intermediate, Strong->Advanced."
    )

    return StudentRecommendation(cluster, level, feats, exp)

