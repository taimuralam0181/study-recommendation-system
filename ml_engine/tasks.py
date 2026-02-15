from __future__ import annotations

import logging
from typing import Dict

from celery import shared_task
from django.contrib.auth import get_user_model

from ml_engine.ml_model import predict_student_outcome, train_models
from ml_engine.models import UploadedDataset


logger = logging.getLogger(__name__)
User = get_user_model()


@shared_task(bind=True, name="ml_engine.train_uploaded_dataset_task")
def train_uploaded_dataset_task(
    self,
    *,
    uploaded_dataset_id: int,
    trained_by_id: int | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    max_training_rows: int = 25000,
) -> Dict:
    dataset = UploadedDataset.objects.filter(id=uploaded_dataset_id).first()
    if not dataset:
        raise ValueError(f"UploadedDataset not found: id={uploaded_dataset_id}")

    trained_by = None
    if trained_by_id:
        trained_by = User.objects.filter(id=trained_by_id).first()

    logger.info(
        "Async training started. dataset_id=%s task_id=%s",
        uploaded_dataset_id,
        self.request.id,
    )
    result = train_models(
        uploaded_dataset=dataset,
        trained_by=trained_by,
        test_size=float(test_size),
        random_state=int(random_state),
        max_training_rows=int(max_training_rows),
    )
    logger.info(
        "Async training completed. dataset_id=%s task_id=%s",
        uploaded_dataset_id,
        self.request.id,
    )
    return result


@shared_task(bind=True, name="ml_engine.predict_student_outcome_task")
def predict_student_outcome_task(self, payload: Dict) -> Dict:
    logger.info("Async prediction started. task_id=%s", self.request.id)
    result = predict_student_outcome(payload)
    logger.info("Async prediction completed. task_id=%s", self.request.id)
    return result
