"""
Feature Engineering (Academic, Explainable)

We compute features from per-subject components (total=100 scheme):
  Hand (20), Attendance (5), CT (15), Midterm (30), Final (50)

Features requested:
- performance_index: a transparent summary score
- improvement_gap: how far the student is from A/A+ (required final / gap)
- consistency_score: stability across subjects (inverse dispersion)
"""

from __future__ import annotations

from typing import Dict, List
import math


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: List[float]) -> float:
    if not xs:
        return 0.0
    mu = _mean(xs)
    var = _mean([(x - mu) ** 2 for x in xs])
    return math.sqrt(var)


def required_final(prefinal: float, cutoff: float, final_max: float = 50.0) -> float:
    return max(0.0, cutoff - prefinal)


def build_student_features_from_totals(
    totals: List[float],
    prefinals: List[float],
    midterms: List[float],
) -> Dict[str, float]:
    """
    Student-level features (semester or overall):
    - performance_index: avg_total - 10*weak_ratio - 0.15*dispersion
    - improvement_gap: mean required final for A+ given current prefinals
    - consistency_score: 1/(1+dispersion)
    """
    n = len(totals)
    avg_total = _mean(totals)
    dispersion = _std(totals)
    weak_ratio = (sum(1 for t in totals if t < 50) / n) if n else 0.0
    strong_ratio = (sum(1 for t in totals if t >= 80) / n) if n else 0.0

    perf_index = avg_total - (10.0 * weak_ratio) - (0.15 * dispersion)
    gap_aplus = _mean([required_final(b, 80) for b in prefinals]) if prefinals else 0.0
    consistency = 1.0 / (1.0 + dispersion) if dispersion >= 0 else 0.0

    return {
        "avg_total": avg_total,
        "avg_midterm": _mean(midterms),
        "weak_ratio": weak_ratio,
        "strong_ratio": strong_ratio,
        "dispersion": dispersion,
        "performance_index": perf_index,
        "improvement_gap": gap_aplus,  # required final average for A+
        "consistency_score": consistency,
        "n_subjects": float(n),
    }


def build_subject_features(prefinal: float, hand: float, att: float, ct: float, mid: float) -> Dict[str, float]:
    """
    Subject-level features for supervised prediction (before final).
    """
    return {
        "hand": hand,
        "attendance": att,
        "ct": ct,
        "midterm": mid,
        "prefinal_total": prefinal,
        "gap_a": required_final(prefinal, 75),
        "gap_aplus": required_final(prefinal, 80),
    }

