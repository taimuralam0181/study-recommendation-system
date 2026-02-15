"""
Inference helper for prediction views.

Primary path:
- Uses active TrainedModel artifacts via ml_engine.ml_model.predict_student_outcome.

Fallback path:
- Uses legacy pickled models if new registry-based models are unavailable.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple

from django.conf import settings
from ml_engine.sklearn_pipeline import load_model, FEATURE_NAMES
from ml_engine.ml_model import predict_student_outcome


def _models_dir() -> str:
    return os.path.join(settings.BASE_DIR, "models")


def _path(name: str) -> str:
    return os.path.join(_models_dir(), name)


def predict_final_and_probs(features: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """
    features keys:
      semester (1..12) [optional, default=1]
      subject (string) [optional, default='General CSE']
      assignment_marks (0..5)
      attendance_percentage (0..100)
      quiz_marks (0..10)
      midterm_marks (0..30)
      previous_cgpa (0..4)
    """
    try:
        prediction = predict_student_outcome(
            {
                "semester": int(features.get("semester", 1)),
                "subject": str(features.get("subject", "General CSE")),
                "assignment_marks": float(features["assignment_marks"]),
                "attendance_percentage": float(features["attendance_percentage"]),
                "quiz_marks": float(features["quiz_marks"]),
                "midterm_marks": float(features["midterm_marks"]),
                "previous_cgpa": float(features["previous_cgpa"]),
            }
        )
        probs = prediction.get("grade_probabilities", {})
        final_pred = float(prediction["predicted_final_marks"])
        prob_aplus = float(probs.get("A+", 0.0))
        prob_a = prob_aplus + float(probs.get("A", 0.0))
        prob_fail = float(probs.get("F", 0.0))
    except Exception:
        # Backward-compatible fallback if registry-based models are not available.
        x = [[float(features[n]) for n in FEATURE_NAMES]]

        reg = load_model(_path("final_regressor.pkl"))
        clf_a = load_model(_path("sk_logreg_A.pkl"))
        clf_ap = load_model(_path("sk_logreg_Aplus.pkl"))
        clf_fail = load_model(_path("sk_logreg_FailRisk.pkl"))

        final_pred = float(reg.predict(x)[0])
        prob_a = float(clf_a.predict_proba(x)[0, 1])
        prob_aplus = float(clf_ap.predict_proba(x)[0, 1])
        prob_fail = float(clf_fail.predict_proba(x)[0, 1])

    # Clamp to valid bounds
    final_pred = max(0.0, min(50.0, final_pred))
    prob_a = max(0.0, min(1.0, prob_a))
    prob_aplus = max(0.0, min(1.0, prob_aplus))
    prob_fail = max(0.0, min(1.0, prob_fail))

    return final_pred, prob_a, prob_aplus, prob_fail
