from __future__ import annotations

import logging
from typing import Any, Dict

from ml_engine.models import PredictionLog


logger = logging.getLogger(__name__)


def log_academic_prediction(
    *,
    user,
    input_features: Dict[str, Any],
    prediction_result: Dict[str, Any],
    response_time_ms: float,
) -> None:
    """
    Best-effort audit log. Failures should never break user flow.
    """
    try:
        PredictionLog.objects.create(
            user=user,
            model_type="academic",
            input_features=input_features,
            prediction_result=prediction_result,
            response_time_ms=float(response_time_ms),
            model_version=None,
        )
    except Exception as exc:
        logger.exception("Failed to persist PredictionLog: %s", exc)
