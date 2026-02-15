import json
from datetime import timedelta

import numpy as np
from django.core.management.base import BaseCommand
from django.utils import timezone

from ml_engine.alerts import send_ml_alert
from ml_engine.models import MLModelRun
from userpanel.academic_logic import grade_from_total
from userpanel.models import StudentSubjectPerformance


class Command(BaseCommand):
    help = "Evaluate real outcomes vs predictions and log production feedback metrics."

    def add_arguments(self, parser):
        parser.add_argument("--lookback-days", type=int, default=180)
        parser.add_argument("--min-samples", type=int, default=30)
        parser.add_argument("--mae-alert-threshold", type=float, default=8.0)
        parser.add_argument("--grade-acc-alert-threshold", type=float, default=0.55)

    def handle(self, *args, **options):
        lookback_days = int(options["lookback_days"])
        min_samples = int(options["min_samples"])
        mae_alert_threshold = float(options["mae_alert_threshold"])
        grade_acc_alert_threshold = float(options["grade_acc_alert_threshold"])

        cutoff = timezone.now() - timedelta(days=lookback_days)
        rows = list(
            StudentSubjectPerformance.objects
            .filter(
                created_at__gte=cutoff,
                final_marks__gt=0,
            )
            .exclude(predicted_grade="")
            .select_related("semester", "subject", "user")
        )

        if len(rows) < min_samples:
            report = {
                "lookback_days": lookback_days,
                "sample_count": len(rows),
                "status": "insufficient_samples",
                "min_samples": min_samples,
            }
            MLModelRun.objects.create(
                kind="production_feedback",
                params={"lookback_days": lookback_days, "min_samples": min_samples},
                metrics=report,
            )
            self.stdout.write(json.dumps(report, indent=2))
            self.stdout.write(self.style.WARNING("Not enough samples for reliable feedback evaluation."))
            return

        final_actual = []
        final_pred = []
        grade_hits = []
        fail_hits = []
        fail_actual = []

        for row in rows:
            actual_final = float(row.final_marks)
            predicted_final = float(row.final_pred)
            actual_total = (
                float(row.hand_marks)
                + float(row.attendance_marks)
                + float(row.ct_marks)
                + float(row.midterm_marks)
                + actual_final
            )
            actual_grade = grade_from_total(actual_total)
            predicted_grade = str(row.predicted_grade or "").strip()

            final_actual.append(actual_final)
            final_pred.append(predicted_final)
            grade_hits.append(1.0 if predicted_grade == actual_grade else 0.0)

            actual_fail = 1 if actual_total < 40 else 0
            predicted_fail = 1 if float(row.prob_fail) >= 0.40 else 0
            fail_actual.append(actual_fail)
            fail_hits.append(1.0 if predicted_fail == actual_fail else 0.0)

        actual_arr = np.array(final_actual, dtype=float)
        pred_arr = np.array(final_pred, dtype=float)
        abs_err = np.abs(actual_arr - pred_arr)

        mae_final = float(np.mean(abs_err))
        rmse_final = float(np.sqrt(np.mean((actual_arr - pred_arr) ** 2)))
        grade_accuracy = float(np.mean(np.array(grade_hits, dtype=float)))
        fail_signal_accuracy = float(np.mean(np.array(fail_hits, dtype=float)))
        fail_rate = float(np.mean(np.array(fail_actual, dtype=float)))

        status = "ok"
        if mae_final >= mae_alert_threshold or grade_accuracy <= grade_acc_alert_threshold:
            status = "warning"

        report = {
            "lookback_days": lookback_days,
            "sample_count": len(rows),
            "mae_final": round(mae_final, 4),
            "rmse_final": round(rmse_final, 4),
            "grade_accuracy": round(grade_accuracy, 4),
            "fail_signal_accuracy": round(fail_signal_accuracy, 4),
            "fail_rate": round(fail_rate, 4),
            "status": status,
            "thresholds": {
                "mae_alert_threshold": mae_alert_threshold,
                "grade_acc_alert_threshold": grade_acc_alert_threshold,
            },
        }

        MLModelRun.objects.create(
            kind="production_feedback",
            params={"lookback_days": lookback_days, "min_samples": min_samples},
            metrics=report,
        )

        self.stdout.write(json.dumps(report, indent=2))
        if status == "warning":
            send_ml_alert("Production feedback warning", json.dumps(report, indent=2))
            self.stdout.write(self.style.WARNING("Feedback quality degraded. Consider retraining."))
        else:
            self.stdout.write(self.style.SUCCESS("Feedback metrics within expected range."))
