import json
from datetime import timedelta
from typing import Dict, List

import numpy as np
from django.core.management.base import BaseCommand
from django.utils import timezone

from ml_engine.alerts import send_ml_alert
from ml_engine.ml_model import train_models
from ml_engine.models import MLModelRun, TrainedModel
from userpanel.models import Student, StudentSubjectPerformance


class Command(BaseCommand):
    help = "Monitor data quality and model drift; optionally trigger retraining."

    def add_arguments(self, parser):
        parser.add_argument("--lookback-days", type=int, default=30)
        parser.add_argument("--min-samples", type=int, default=50)
        parser.add_argument("--drift-threshold", type=float, default=0.35)
        parser.add_argument("--auto-retrain", action="store_true")
        parser.add_argument("--test-size", type=float, default=0.2)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--max-rows", type=int, default=25000)

    def handle(self, *args, **options):
        lookback_days = int(options["lookback_days"])
        min_samples = int(options["min_samples"])
        drift_threshold = float(options["drift_threshold"])
        auto_retrain = bool(options["auto_retrain"])

        reg_model = (
            TrainedModel.objects.filter(
                task_type=TrainedModel.TASK_REGRESSION,
                is_active=True,
            )
            .order_by("-created_at")
            .first()
        )
        cls_model = (
            TrainedModel.objects.filter(
                task_type=TrainedModel.TASK_CLASSIFICATION,
                is_active=True,
            )
            .order_by("-created_at")
            .first()
        )
        if not reg_model or not cls_model:
            self.stdout.write(self.style.ERROR("Active regression/classification model not found."))
            return

        diagnostics = dict((reg_model.metrics or {}).get("diagnostics") or {})
        baseline_feature_stats = dict(diagnostics.get("feature_stats") or {})
        baseline_subject_dist = dict(diagnostics.get("subject_distribution") or {})

        cutoff = timezone.now() - timedelta(days=lookback_days)
        recent_rows = list(
            StudentSubjectPerformance.objects
            .select_related("subject")
            .filter(created_at__gte=cutoff)
        )

        if len(recent_rows) < min_samples:
            self.stdout.write(
                self.style.WARNING(
                    f"Not enough recent samples for robust monitoring ({len(recent_rows)} < {min_samples})."
                )
            )

        user_ids = list({r.user_id for r in recent_rows})
        cgpa_by_user = {
            row.user_id: float(row.cgpa)
            for row in Student.objects.filter(user_id__in=user_ids).only("user_id", "cgpa")
        }

        features: Dict[str, List[float]] = {
            "semester": [],
            "assignment_marks": [],
            "quiz_marks": [],
            "attendance_percentage": [],
            "midterm_marks": [],
            "previous_cgpa": [],
        }
        subject_counts: Dict[str, int] = {}
        invalid_rows = 0

        for row in recent_rows:
            vals = {
                "semester": float(row.semester.number),
                "assignment_marks": float(row.hand_marks),
                "quiz_marks": float(row.ct_marks),
                "attendance_percentage": float(row.attendance_percentage),
                "midterm_marks": float(row.midterm_marks),
                "previous_cgpa": float(cgpa_by_user.get(row.user_id, 0.0)),
            }
            if not (
                1 <= vals["semester"] <= 12
                and 0 <= vals["assignment_marks"] <= 5
                and 0 <= vals["quiz_marks"] <= 10
                and 0 <= vals["attendance_percentage"] <= 100
                and 0 <= vals["midterm_marks"] <= 30
                and 0 <= vals["previous_cgpa"] <= 4.0
            ):
                invalid_rows += 1
                continue

            for key, value in vals.items():
                features[key].append(float(value))

            subject_name = str(row.subject.name).strip()
            subject_counts[subject_name] = int(subject_counts.get(subject_name, 0)) + 1

        sample_count = len(features["semester"])
        current_subject_dist: Dict[str, float] = {}
        if sample_count > 0:
            current_subject_dist = {k: float(v) / float(sample_count) for k, v in subject_counts.items()}

        drift_by_feature: Dict[str, float] = {}
        for feat, baseline in baseline_feature_stats.items():
            current_vals = features.get(feat) or []
            if not current_vals:
                drift_by_feature[feat] = 0.0
                continue
            base_mean = float(baseline.get("mean", 0.0))
            base_std = abs(float(baseline.get("std", 1.0))) or 1.0
            current_mean = float(np.mean(current_vals))
            drift_by_feature[feat] = float(abs(current_mean - base_mean) / base_std)

        feature_drift_score = float(np.mean(list(drift_by_feature.values()))) if drift_by_feature else 0.0

        all_subjects = set(baseline_subject_dist.keys()) | set(current_subject_dist.keys())
        if all_subjects:
            subject_shift = sum(
                abs(float(current_subject_dist.get(s, 0.0)) - float(baseline_subject_dist.get(s, 0.0)))
                for s in all_subjects
            ) / 2.0
        else:
            subject_shift = 0.0

        invalid_ratio = (float(invalid_rows) / float(max(1, len(recent_rows))))
        overall_drift = float((feature_drift_score + subject_shift + invalid_ratio) / 3.0)

        should_retrain = bool(
            sample_count >= min_samples
            and (
                overall_drift >= drift_threshold
                or invalid_ratio >= 0.10
            )
        )

        report = {
            "lookback_days": lookback_days,
            "recent_rows": len(recent_rows),
            "valid_rows": sample_count,
            "invalid_rows": invalid_rows,
            "invalid_ratio": round(invalid_ratio, 4),
            "feature_drift_score": round(feature_drift_score, 4),
            "subject_shift": round(float(subject_shift), 4),
            "overall_drift": round(overall_drift, 4),
            "drift_by_feature": {k: round(v, 4) for k, v in drift_by_feature.items()},
            "threshold": drift_threshold,
            "should_retrain": should_retrain,
            "auto_retrain": auto_retrain,
            "retrain_triggered": False,
        }

        retrain_result = None
        if should_retrain and auto_retrain:
            retrain_result = train_models(
                uploaded_dataset=None,
                trained_by=None,
                test_size=float(options["test_size"]),
                random_state=int(options["seed"]),
                max_training_rows=int(options["max_rows"]),
            )
            report["retrain_triggered"] = True
            report["retrain_result"] = retrain_result

        MLModelRun.objects.create(
            kind="monitor_drift",
            params={
                "lookback_days": lookback_days,
                "drift_threshold": drift_threshold,
                "min_samples": min_samples,
                "auto_retrain": auto_retrain,
            },
            metrics=report,
        )

        self.stdout.write(json.dumps(report, indent=2, default=str))
        if report["retrain_triggered"]:
            send_ml_alert(
                "ML retraining triggered",
                json.dumps(report, indent=2, default=str),
            )
            self.stdout.write(self.style.SUCCESS("Retraining triggered due to drift threshold breach."))
        elif report["should_retrain"]:
            send_ml_alert(
                "ML retraining recommended",
                json.dumps(report, indent=2, default=str),
            )
            self.stdout.write(self.style.WARNING("Retraining recommended. Re-run with --auto-retrain to execute."))
        else:
            self.stdout.write(self.style.SUCCESS("No retraining required."))
