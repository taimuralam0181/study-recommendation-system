"""
University-grade ML pipeline for dataset validation, training, and inference.

This module keeps the existing lightweight explainable helpers (recommend_level,
explain_recommendation) and adds a robust sklearn-based workflow for:
1) Final marks regression
2) Final grade multi-class classification
"""

from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from django.conf import settings
from django.db import transaction
from django.utils import timezone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from ml_engine.models import (
    CSETrainingExample,
    DatasetStudentPerformance,
    MLModelRun,
    TrainedModel,
    UploadedDataset,
)


logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "student_id",
    "semester",
    "subject",
    "assignment_marks",
    "quiz_marks",
    "attendance_percentage",
    "midterm_marks",
    "previous_cgpa",
    "final_marks",
    "final_grade",
]

NUMERIC_COLUMNS = [
    "semester",
    "assignment_marks",
    "quiz_marks",
    "attendance_percentage",
    "midterm_marks",
    "previous_cgpa",
    "final_marks",
]

FEATURE_COLUMNS = [
    "semester",
    "subject",
    "assignment_marks",
    "quiz_marks",
    "attendance_percentage",
    "midterm_marks",
    "previous_cgpa",
]

RANGE_CHECKS = {
    "semester": (1, 12),
    "assignment_marks": (0, 5),
    "quiz_marks": (0, 10),
    "attendance_percentage": (0, 100),
    "midterm_marks": (0, 30),
    "previous_cgpa": (0.0, 4.0),
    "final_marks": (0, 50),
}

ALLOWED_GRADES = {"A+", "A", "A-", "B+", "B", "C+", "C", "D", "F"}
GRADE_ORDER = ["A+", "A", "A-", "B+", "B", "C+", "C", "D", "F"]


@dataclass
class DatasetValidationSummary:
    total_rows: int
    rows_after_cleanup: int
    dropped_missing_rows: int
    dropped_invalid_numeric_rows: int


def recommend_level(math: float, physics: float, cs: float) -> str:
    """
    Rule-based, explainable mapping used by legacy UI modules.
    """
    avg = (math + physics + cs) / 3
    if avg >= 80:
        return "Advanced"
    if avg < 50:
        return "Beginner"
    return "Intermediate"


def explain_recommendation(math: float, physics: float, cs: float) -> Tuple[str, float, str]:
    """
    Returns (level, avg_score, explanation_text) for old dashboards.
    """
    avg = (math + physics + cs) / 3
    level = recommend_level(math, physics, cs)

    if level == "Advanced":
        explanation = "Average score is 80 or above, so the student is classified as Advanced."
    elif level == "Beginner":
        explanation = "Average score is below 50, so the student is classified as Beginner."
    else:
        explanation = "Average score is between 50 and 79, so the student is classified as Intermediate."
    return level, avg, explanation


def _models_dir() -> Path:
    base = Path(settings.BASE_DIR) / "models" / "university_ml"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _clean_grade(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().upper()


def _calculate_total(row: pd.Series) -> float:
    attendance_marks = 5.0 * (float(row["attendance_percentage"]) / 100.0)
    return (
        float(row["assignment_marks"])
        + attendance_marks
        + float(row["quiz_marks"])
        + float(row["midterm_marks"])
        + float(row["final_marks"])
    )


def validate_dataset_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, DatasetValidationSummary]:
    """
    Validate teacher-uploaded dataset and return a clean DataFrame ready for DB write.

    Rules:
    - required columns must exist
    - NaN/blank rows in required fields are dropped
    - numeric conversion and ranges are enforced
    - final_grade must be from allowed set
    - duplicate (student_id, semester, subject) rows are rejected
    """
    if df is None or df.empty:
        raise ValueError("Uploaded file is empty.")

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError("Missing required columns: " + ", ".join(missing_cols))

    total_rows = len(df)
    df = df[REQUIRED_COLUMNS].copy()
    df = df.replace(r"^\s*$", pd.NA, regex=True)
    df = df.dropna(how="all")
    rows_after_blank_cleanup = len(df)

    before_missing_drop = len(df)
    df = df.dropna(subset=REQUIRED_COLUMNS)
    dropped_missing_rows = before_missing_drop - len(df)

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before_numeric_drop = len(df)
    df = df.dropna(subset=NUMERIC_COLUMNS)
    dropped_invalid_numeric_rows = before_numeric_drop - len(df)

    if df.empty:
        raise ValueError("No usable rows remain after cleaning missing/invalid values.")

    for col, (min_val, max_val) in RANGE_CHECKS.items():
        invalid_mask = (df[col] < min_val) | (df[col] > max_val)
        if invalid_mask.any():
            bad_count = int(invalid_mask.sum())
            raise ValueError(
                f"{col} must be between {min_val} and {max_val}. Found {bad_count} invalid rows."
            )

    df["student_id"] = df["student_id"].astype(str).str.strip()
    if (df["student_id"] == "").any():
        raise ValueError("student_id cannot be empty.")

    df["semester"] = df["semester"].astype(int)
    df["subject"] = df["subject"].astype(str).str.strip()
    if (df["subject"] == "").any():
        raise ValueError("subject cannot be empty.")

    df["final_grade"] = df["final_grade"].map(_clean_grade)
    invalid_grades = sorted(set(df.loc[~df["final_grade"].isin(ALLOWED_GRADES), "final_grade"].tolist()))
    if invalid_grades:
        raise ValueError(
            "Invalid final_grade values: "
            + ", ".join(invalid_grades[:8])
            + ". Allowed: "
            + ", ".join(GRADE_ORDER)
        )

    duplicates = df.duplicated(subset=["student_id", "semester", "subject"], keep=False)
    if duplicates.any():
        dup_rows = df.loc[duplicates, ["student_id", "semester", "subject"]].head(5).to_dict(orient="records")
        raise ValueError(
            "Duplicate rows detected for (student_id, semester, subject). "
            f"Examples: {dup_rows}"
        )

    summary = DatasetValidationSummary(
        total_rows=total_rows,
        rows_after_cleanup=len(df),
        dropped_missing_rows=dropped_missing_rows,
        dropped_invalid_numeric_rows=dropped_invalid_numeric_rows,
    )
    logger.info(
        "Dataset validated. total=%s cleaned=%s dropped_missing=%s dropped_invalid_numeric=%s",
        total_rows,
        summary.rows_after_cleanup,
        dropped_missing_rows,
        dropped_invalid_numeric_rows,
    )
    return df.reset_index(drop=True), summary


def persist_dataset_rows(
    df: pd.DataFrame,
    uploaded_dataset: UploadedDataset,
    clear_existing: bool = False,
    batch_size: int = 1000,
) -> Dict[str, int]:
    """
    Persist validated rows into DatasetStudentPerformance and CSETrainingExample.
    """
    if clear_existing:
        DatasetStudentPerformance.objects.all().delete()
        CSETrainingExample.objects.all().delete()

    dataset_rows: List[DatasetStudentPerformance] = []
    training_rows: List[CSETrainingExample] = []
    source = f"teacher_upload:{uploaded_dataset.id}"

    for row in df.to_dict(orient="records"):
        assignment_scaled = float(row["assignment_marks"]) * 20.0
        quiz_scaled = float(row["quiz_marks"]) * 10.0
        midterm_scaled = float(row["midterm_marks"]) * (100.0 / 30.0)
        avg_score = (assignment_scaled + quiz_scaled + midterm_scaled) / 3.0
        level = recommend_level(assignment_scaled, quiz_scaled, midterm_scaled)
        total = _calculate_total(pd.Series(row))

        dataset_rows.append(
            DatasetStudentPerformance(
                source=source,
                uploaded_dataset=uploaded_dataset,
                student_id=str(row["student_id"]),
                semester=int(row["semester"]),
                subject=str(row["subject"]),
                assignment_marks=float(row["assignment_marks"]),
                quiz_marks=float(row["quiz_marks"]),
                attendance_percentage=float(row["attendance_percentage"]),
                midterm_marks=float(row["midterm_marks"]),
                previous_cgpa=float(row["previous_cgpa"]),
                final_marks=float(row["final_marks"]),
                final_grade=str(row["final_grade"]),
                final_marks_norm=float(row["final_marks"]) / 50.0,
                previous_cgpa_norm=float(row["previous_cgpa"]) / 4.0,
                math_marks=assignment_scaled,
                physics_marks=quiz_scaled,
                cs_marks=midterm_scaled,
                avg_score=avg_score,
                level=level,
                math_norm=assignment_scaled / 100.0,
                physics_norm=quiz_scaled / 100.0,
                cs_norm=midterm_scaled / 100.0,
                avg_norm=avg_score / 100.0,
            )
        )

        training_rows.append(
            CSETrainingExample(
                uploaded_dataset=uploaded_dataset,
                student_id=str(row["student_id"]),
                semester=int(row["semester"]),
                subject=str(row["subject"]),
                assignment_marks=float(row["assignment_marks"]),
                attendance_percentage=float(row["attendance_percentage"]),
                quiz_marks=float(row["quiz_marks"]),
                midterm_marks=float(row["midterm_marks"]),
                previous_cgpa=float(row["previous_cgpa"]),
                final_marks=float(row["final_marks"]),
                total=total,
                final_grade=str(row["final_grade"]),
            )
        )

    DatasetStudentPerformance.objects.bulk_create(dataset_rows, batch_size=batch_size)
    CSETrainingExample.objects.bulk_create(training_rows, batch_size=batch_size)

    return {
        "dataset_rows": len(dataset_rows),
        "training_rows": len(training_rows),
    }


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("scaler", StandardScaler())]),
                [c for c in FEATURE_COLUMNS if c != "subject"],
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["subject"]),
        ]
    )


def _safe_json_dict(data: Dict) -> Dict:
    out: Dict = {}
    for key, value in data.items():
        if isinstance(value, (np.floating, float)):
            out[key] = float(value)
        elif isinstance(value, (np.integer, int)):
            out[key] = int(value)
        elif isinstance(value, dict):
            out[key] = _safe_json_dict(value)
        elif isinstance(value, list):
            out[key] = [
                _safe_json_dict(v) if isinstance(v, dict) else float(v) if isinstance(v, (np.floating, float)) else v
                for v in value
            ]
        else:
            out[key] = value
    return out


def _feature_stats(frame: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    numeric_cols = [c for c in FEATURE_COLUMNS if c != "subject"]
    for col in numeric_cols:
        series = pd.to_numeric(frame[col], errors="coerce").dropna()
        if series.empty:
            continue
        stats[col] = {
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0)),
            "min": float(series.min()),
            "max": float(series.max()),
        }
    return stats


def _class_distribution(values: np.ndarray, labels: List[str]) -> Dict[str, float]:
    counts = {label: 0 for label in labels}
    unique, freq = np.unique(values, return_counts=True)
    for idx, count in zip(unique.tolist(), freq.tolist()):
        label = labels[int(idx)]
        counts[label] = int(count)
    total = max(1, int(sum(counts.values())))
    return {k: float(v) / float(total) for k, v in counts.items()}


def train_models(
    uploaded_dataset: UploadedDataset | None = None,
    trained_by=None,
    test_size: float = 0.2,
    random_state: int = 42,
    max_training_rows: int = 25000,
) -> Dict[str, Dict]:
    """
    Train and persist best regression/classification models from CSETrainingExample.

    Regression target: final_marks
    Classification target: final_grade (multi-class)
    """
    if uploaded_dataset:
        uploaded_dataset.train_status = UploadedDataset.STATUS_RUNNING
        uploaded_dataset.error_message = ""
        uploaded_dataset.save(update_fields=["train_status", "error_message"])

    try:
        training_qs = CSETrainingExample.objects.all()
        if uploaded_dataset is not None:
            training_qs = training_qs.filter(uploaded_dataset=uploaded_dataset)

        qs = training_qs.values(
            "semester",
            "subject",
            "assignment_marks",
            "quiz_marks",
            "attendance_percentage",
            "midterm_marks",
            "previous_cgpa",
            "final_marks",
            "final_grade",
        )
        frame = pd.DataFrame(list(qs))
        if frame.empty or len(frame) < 200:
            if uploaded_dataset is not None:
                raise ValueError(
                    f"Uploaded dataset #{uploaded_dataset.id} does not have enough rows for training. "
                    "Training requires at least 200 examples."
                )
            raise ValueError("Training requires at least 200 examples.")

        frame["subject"] = frame["subject"].astype(str).str.strip()
        frame["final_grade"] = frame["final_grade"].map(_clean_grade)
        frame = frame.dropna(subset=FEATURE_COLUMNS + ["final_marks", "final_grade"])
        frame = frame[frame["final_grade"].isin(ALLOWED_GRADES)].copy()

        for col in [c for c in FEATURE_COLUMNS if c != "subject"] + ["final_marks"]:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = frame.dropna(subset=FEATURE_COLUMNS + ["final_marks"])

        if frame.empty:
            if uploaded_dataset is not None:
                raise ValueError(
                    f"No valid training rows remain after cleaning for uploaded dataset #{uploaded_dataset.id}."
                )
            raise ValueError("No valid records after cleaning training examples.")

        if len(frame) > max_training_rows:
            frame = frame.sample(n=max_training_rows, random_state=random_state).reset_index(drop=True)

        X = frame[FEATURE_COLUMNS].copy()
        y_reg = frame["final_marks"].astype(float).values
        y_grade = frame["final_grade"].astype(str).values

        label_encoder = LabelEncoder()
        y_cls = label_encoder.fit_transform(y_grade)
        class_names = [str(x) for x in label_encoder.classes_.tolist()]
        class_distribution = _class_distribution(y_cls, class_names)
        smallest_class = min(class_distribution.values()) if class_distribution else 0.0
        severe_imbalance = bool(smallest_class < 0.05)
        if severe_imbalance:
            logger.warning(
                "Detected severe class imbalance during training: smallest class ratio=%.4f",
                smallest_class,
            )

        training_diagnostics = {
            "feature_stats": _feature_stats(frame),
            "subject_distribution": {
                str(k): float(v)
                for k, v in frame["subject"].value_counts(normalize=True).to_dict().items()
            },
            "class_distribution": class_distribution,
            "imbalance": {
                "smallest_class_ratio": float(smallest_class),
                "severe_imbalance": severe_imbalance,
            },
        }

        unique_labels, label_counts = np.unique(y_cls, return_counts=True)
        if len(unique_labels) < 2:
            raise ValueError("Classification training requires at least 2 distinct final grades.")
        stratify_vector = None
        if len(unique_labels) > 1 and int(label_counts.min()) >= 2:
            stratify_vector = y_cls
        (
            X_train,
            X_test,
            y_train_reg,
            y_test_reg,
            y_train_cls,
            y_test_cls,
        ) = train_test_split(
            X,
            y_reg,
            y_cls,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_vector,
        )

        regressors = {
            "RandomForestRegressor": RandomForestRegressor(
                n_estimators=80,
                random_state=random_state,
                n_jobs=1,
                max_depth=16,
            ),
            "GradientBoostingRegressor": GradientBoostingRegressor(
                random_state=random_state,
            ),
        }
        classifiers = {
            "RandomForestClassifier": RandomForestClassifier(
                n_estimators=80,
                random_state=random_state,
                class_weight="balanced_subsample",
                n_jobs=1,
                max_depth=16,
            ),
            "GradientBoostingClassifier": GradientBoostingClassifier(
                random_state=random_state,
            ),
        }

        best_regression = None
        best_reg_metrics = None
        best_reg_name = None
        for model_name, estimator in regressors.items():
            pipeline = Pipeline(
                steps=[
                    ("preprocess", _build_preprocessor()),
                    ("model", estimator),
                ]
            )
            pipeline.fit(X_train, y_train_reg)
            pred = pipeline.predict(X_test)
            rmse = float(mean_squared_error(y_test_reg, pred) ** 0.5)
            mae = float(mean_absolute_error(y_test_reg, pred))
            metrics = {"rmse": rmse, "mae": mae}

            if best_reg_metrics is None or rmse < best_reg_metrics["rmse"]:
                best_regression = pipeline
                best_reg_metrics = metrics
                best_reg_name = model_name

        if best_regression is None or best_reg_metrics is None:
            raise ValueError("Regression training failed.")

        reg_cv = KFold(
            n_splits=5,
            shuffle=True,
            random_state=random_state,
        )
        reg_cv_scores = cross_validate(
            best_regression,
            X,
            y_reg,
            cv=reg_cv,
            scoring=("neg_root_mean_squared_error", "neg_mean_absolute_error"),
            n_jobs=1,
        )
        best_reg_metrics["cv_rmse_mean"] = float(-np.mean(reg_cv_scores["test_neg_root_mean_squared_error"]))
        best_reg_metrics["cv_rmse_std"] = float(np.std(-reg_cv_scores["test_neg_root_mean_squared_error"]))
        best_reg_metrics["cv_mae_mean"] = float(-np.mean(reg_cv_scores["test_neg_mean_absolute_error"]))
        best_reg_metrics["cv_mae_std"] = float(np.std(-reg_cv_scores["test_neg_mean_absolute_error"]))

        best_classification = None
        best_cls_metrics = None
        best_cls_name = None
        best_confusion = None
        best_report = None

        cls_classes = np.unique(y_train_cls)
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=cls_classes,
            y=y_train_cls,
        )
        class_weight_lookup = dict(zip(cls_classes.tolist(), class_weights.tolist()))
        sample_weight = np.array([class_weight_lookup[x] for x in y_train_cls], dtype=float)

        for model_name, estimator in classifiers.items():
            pipeline = Pipeline(
                steps=[
                    ("preprocess", _build_preprocessor()),
                    ("model", estimator),
                ]
            )
            try:
                pipeline.fit(X_train, y_train_cls, model__sample_weight=sample_weight)
            except TypeError:
                pipeline.fit(X_train, y_train_cls)

            pred = pipeline.predict(X_test)
            acc = float(accuracy_score(y_test_cls, pred))
            f1 = float(f1_score(y_test_cls, pred, average="weighted", zero_division=0))
            report = classification_report(
                y_test_cls,
                pred,
                labels=list(range(len(class_names))),
                target_names=class_names,
                output_dict=True,
                zero_division=0,
            )
            conf = confusion_matrix(
                y_test_cls,
                pred,
                labels=list(range(len(class_names))),
            )
            metrics = {"accuracy": acc, "f1_weighted": f1}

            if best_cls_metrics is None or f1 > best_cls_metrics["f1_weighted"]:
                best_classification = pipeline
                best_cls_metrics = metrics
                best_cls_name = model_name
                best_confusion = conf
                best_report = report

        if best_classification is None or best_cls_metrics is None:
            raise ValueError("Classification training failed.")

        cls_proba = best_classification.predict_proba(X_test)
        brier_per_grade: Dict[str, float] = {}
        for idx, grade in enumerate(class_names):
            target = (y_test_cls == idx).astype(int)
            brier_per_grade[str(grade)] = float(brier_score_loss(target, cls_proba[:, idx]))
        best_cls_metrics["brier_macro"] = float(np.mean(list(brier_per_grade.values())))
        best_cls_metrics["brier_per_grade"] = brier_per_grade

        min_class_count = int(label_counts.min()) if len(label_counts) else 0
        if min_class_count >= 2:
            cls_cv_splits = min(5, min_class_count)
            cls_cv = StratifiedKFold(
                n_splits=cls_cv_splits,
                shuffle=True,
                random_state=random_state,
            )
        else:
            cls_cv_splits = 3
            cls_cv = KFold(
                n_splits=cls_cv_splits,
                shuffle=True,
                random_state=random_state,
            )
        cls_cv_scores = cross_validate(
            best_classification,
            X,
            y_cls,
            cv=cls_cv,
            scoring=("accuracy", "f1_weighted"),
            n_jobs=1,
        )
        best_cls_metrics["cv_accuracy_mean"] = float(np.mean(cls_cv_scores["test_accuracy"]))
        best_cls_metrics["cv_accuracy_std"] = float(np.std(cls_cv_scores["test_accuracy"]))
        best_cls_metrics["cv_f1_weighted_mean"] = float(np.mean(cls_cv_scores["test_f1_weighted"]))
        best_cls_metrics["cv_f1_weighted_std"] = float(np.std(cls_cv_scores["test_f1_weighted"]))

        version = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        models_dir = _models_dir()
        reg_path = models_dir / f"final_marks_{best_reg_name}_{version}.pkl"
        cls_path = models_dir / f"final_grade_{best_cls_name}_{version}.pkl"
        le_path = models_dir / f"final_grade_label_encoder_{version}.pkl"

        with reg_path.open("wb") as f:
            pickle.dump(best_regression, f)
        with cls_path.open("wb") as f:
            pickle.dump(best_classification, f)
        with le_path.open("wb") as f:
            pickle.dump(label_encoder, f)

        regression_metrics_payload = _safe_json_dict(
            {
                **best_reg_metrics,
                "rows": int(len(frame)),
                "test_size": float(test_size),
                "diagnostics": training_diagnostics,
            }
        )
        classification_metrics_payload = _safe_json_dict(
            {
                **best_cls_metrics,
                "rows": int(len(frame)),
                "test_size": float(test_size),
                "diagnostics": training_diagnostics,
            }
        )

        with transaction.atomic():
            TrainedModel.objects.filter(
                task_type=TrainedModel.TASK_REGRESSION,
                is_active=True,
            ).update(is_active=False)
            TrainedModel.objects.filter(
                task_type=TrainedModel.TASK_CLASSIFICATION,
                is_active=True,
            ).update(is_active=False)

            regression_model = TrainedModel.objects.create(
                task_type=TrainedModel.TASK_REGRESSION,
                algorithm=best_reg_name,
                version=version,
                model_path=str(reg_path),
                feature_columns=FEATURE_COLUMNS,
                target_column="final_marks",
                metrics=regression_metrics_payload,
                trained_by=trained_by,
                trained_on=uploaded_dataset,
                is_active=True,
            )

            classification_model = TrainedModel.objects.create(
                task_type=TrainedModel.TASK_CLASSIFICATION,
                algorithm=best_cls_name,
                version=version,
                model_path=str(cls_path),
                label_encoder_path=str(le_path),
                feature_columns=FEATURE_COLUMNS,
                target_column="final_grade",
                metrics=classification_metrics_payload,
                confusion_matrix=best_confusion.tolist() if best_confusion is not None else [],
                classification_report=_safe_json_dict(best_report or {}),
                trained_by=trained_by,
                trained_on=uploaded_dataset,
                is_active=True,
            )

            regression_run = MLModelRun.objects.create(
                kind="university_regression",
                params=_safe_json_dict(
                    {
                        "algorithm": best_reg_name,
                        "version": version,
                        "features": FEATURE_COLUMNS,
                        "uploaded_dataset_id": int(uploaded_dataset.id) if uploaded_dataset else None,
                    }
                ),
                metrics=regression_metrics_payload,
            )
            classification_run = MLModelRun.objects.create(
                kind="university_classification",
                params=_safe_json_dict(
                    {
                        "algorithm": best_cls_name,
                        "version": version,
                        "features": FEATURE_COLUMNS,
                        "classes": class_names,
                        "uploaded_dataset_id": int(uploaded_dataset.id) if uploaded_dataset else None,
                    }
                ),
                metrics=classification_metrics_payload,
            )

        if uploaded_dataset:
            uploaded_dataset.train_status = UploadedDataset.STATUS_SUCCESS
            uploaded_dataset.trained_at = timezone.now()
            uploaded_dataset.error_message = ""
            uploaded_dataset.save(update_fields=["train_status", "trained_at", "error_message"])

        return {
            "regression": {
                "id": regression_model.id,
                "algorithm": best_reg_name,
                "version": version,
                "run_id": regression_run.id,
                **best_reg_metrics,
            },
            "classification": {
                "id": classification_model.id,
                "algorithm": best_cls_name,
                "version": version,
                "run_id": classification_run.id,
                **best_cls_metrics,
            },
        }
    except Exception as exc:
        logger.exception("ML training failed: %s", exc)
        if uploaded_dataset:
            uploaded_dataset.train_status = UploadedDataset.STATUS_FAILED
            uploaded_dataset.trained_at = timezone.now()
            uploaded_dataset.error_message = str(exc)
            uploaded_dataset.save(update_fields=["train_status", "trained_at", "error_message"])
        raise


def _load_active_model(task_type: str) -> TrainedModel:
    model = (
        TrainedModel.objects.filter(task_type=task_type, is_active=True)
        .order_by("-created_at")
        .first()
    )
    if not model:
        raise ValueError(f"No active {task_type} model available. Train models first.")
    return model


def _prepare_feature_row(payload: Dict) -> pd.DataFrame:
    try:
        row = {
            "semester": int(payload["semester"]),
            "subject": str(payload["subject"]).strip(),
            "assignment_marks": float(payload["assignment_marks"]),
            "quiz_marks": float(payload["quiz_marks"]),
            "attendance_percentage": float(payload["attendance_percentage"]),
            "midterm_marks": float(payload["midterm_marks"]),
            "previous_cgpa": float(payload["previous_cgpa"]),
        }
    except KeyError as exc:
        raise ValueError(f"Missing required prediction field: {exc}") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid value type in prediction payload: {exc}") from exc

    for col, (min_val, max_val) in RANGE_CHECKS.items():
        if col == "final_marks":
            continue
        val = row[col]
        if val < min_val or val > max_val:
            raise ValueError(f"{col} must be between {min_val} and {max_val}.")
    if not row["subject"]:
        raise ValueError("subject cannot be empty.")

    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def predict_student_outcome(payload: Dict) -> Dict:
    """
    Predict final marks and final grade probabilities for one student row.
    """
    reg_meta = _load_active_model(TrainedModel.TASK_REGRESSION)
    cls_meta = _load_active_model(TrainedModel.TASK_CLASSIFICATION)

    with open(reg_meta.model_path, "rb") as f:
        reg_model = pickle.load(f)
    with open(cls_meta.model_path, "rb") as f:
        cls_model = pickle.load(f)
    with open(cls_meta.label_encoder_path, "rb") as f:
        label_encoder: LabelEncoder = pickle.load(f)

    X = _prepare_feature_row(payload)
    final_marks_pred = float(reg_model.predict(X)[0])
    final_marks_pred = max(0.0, min(50.0, final_marks_pred))

    proba = cls_model.predict_proba(X)[0]
    labels = label_encoder.inverse_transform(np.arange(len(proba)))
    probabilities = {
        str(label): float(prob)
        for label, prob in sorted(zip(labels, proba), key=lambda x: GRADE_ORDER.index(str(x[0])))
    }
    predicted_grade = max(probabilities, key=probabilities.get)

    attendance_marks = 5.0 * (float(X.iloc[0]["attendance_percentage"]) / 100.0)
    prefinal_total = (
        float(X.iloc[0]["assignment_marks"])
        + float(X.iloc[0]["quiz_marks"])
        + float(X.iloc[0]["midterm_marks"])
        + attendance_marks
    )
    total_pred = prefinal_total + final_marks_pred

    weak_signals: List[str] = []
    if float(X.iloc[0]["assignment_marks"]) < 2.5:
        weak_signals.append("Low assignment performance")
    if float(X.iloc[0]["quiz_marks"]) < 5.0:
        weak_signals.append("Low quiz performance")
    if float(X.iloc[0]["midterm_marks"]) < 15.0:
        weak_signals.append("Midterm below expected range")
    if float(X.iloc[0]["attendance_percentage"]) < 70.0:
        weak_signals.append("Attendance below 70%")

    return {
        "predicted_final_marks": round(final_marks_pred, 2),
        "predicted_grade": predicted_grade,
        "grade_probabilities": {k: round(v, 4) for k, v in probabilities.items()},
        "predicted_total": round(total_pred, 2),
        "weak_signals": weak_signals,
        "model_versions": {
            "regression": reg_meta.version,
            "classification": cls_meta.version,
        },
    }
