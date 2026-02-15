from django.core.management.base import BaseCommand

from ml_engine.models import MLModelRun
from userpanel.models import StudentSubjectPerformance

from ml_engine.feature_engineering import (
    build_student_features_from_totals,
    build_subject_features,
)
from ml_engine.academic_ml import kmeans_fit_predict
from ml_engine.supervised import fit_logreg, evaluate_binary, train_test_split


class Command(BaseCommand):
    """
    Trains and evaluates academic ML models from the database.

    Models:
    1) Cohort KMeans-style clustering on student features (Weak/Average/Strong)
    2) Logistic regression (from scratch) for:
       - P(A) where A means total >= 75
       - P(A+) where A+ means total >= 80

    This produces an MLModelRun row per model kind with params + metrics.
    """

    def add_arguments(self, parser):
        parser.add_argument('--test-ratio', type=float, default=0.2)
        parser.add_argument('--seed', type=int, default=42)

    def handle(self, *args, **opts):
        test_ratio = opts['test_ratio']
        seed = opts['seed']

        self._train_kmeans()
        self._train_logreg(kind='logreg_a', cutoff=75, test_ratio=test_ratio, seed=seed)
        self._train_logreg(kind='logreg_aplus', cutoff=80, test_ratio=test_ratio, seed=seed)

        self.stdout.write(self.style.SUCCESS("Training complete."))

    def _train_kmeans(self):
        # Build student-level feature vectors across all semesters
        qs = StudentSubjectPerformance.objects.select_related('user', 'semester', 'subject')
        by_user = {}
        for p in qs:
            by_user.setdefault(p.user_id, []).append(p)

        feature_names = [
            "avg_midterm",
            "avg_total",
            "weak_ratio",
            "strong_ratio",
            "dispersion",
            "performance_index",
            "improvement_gap",
            "consistency_score",
        ]

        X = []
        user_ids = []
        for uid, perfs in by_user.items():
            totals = [float(p.total) for p in perfs]
            midterms = [float(p.midterm_marks) for p in perfs]
            prefinals = [float(p.hand_marks + p.attendance_marks + p.ct_marks + p.midterm_marks) for p in perfs]
            feats = build_student_features_from_totals(totals, prefinals, midterms)
            if feats["n_subjects"] < 3:
                continue
            X.append([feats[n] for n in feature_names])
            user_ids.append(uid)

        if len(X) < 6:
            self.stdout.write(self.style.WARNING("Not enough cohort data for KMeans training; skipping."))
            return

        labels, centroids = kmeans_fit_predict(X, k=3, max_iter=40)

        MLModelRun.objects.create(
            kind="kmeans_cohort",
            params={
                "feature_names": feature_names,
                "centroids": centroids,
            },
            metrics={
                "n_students": len(X),
            }
        )
        self.stdout.write(self.style.SUCCESS(f"KMeans cohort trained on {len(X)} students."))

    def _train_logreg(self, kind: str, cutoff: int, test_ratio: float, seed: int):
        # Subject-level dataset: features from prefinal; label from final total
        qs = StudentSubjectPerformance.objects.all()

        feature_names = ["hand", "attendance", "ct", "midterm", "prefinal_total", "gap_a", "gap_aplus"]
        X = []
        y = []
        for p in qs:
            prefinal = float(p.hand_marks + p.attendance_marks + p.ct_marks + p.midterm_marks)
            feats = build_subject_features(
                prefinal=prefinal,
                hand=float(p.hand_marks),
                att=float(p.attendance_marks),
                ct=float(p.ct_marks),
                mid=float(p.midterm_marks),
            )
            X.append([feats[n] for n in feature_names])
            y.append(1 if p.total >= cutoff else 0)

        if len(X) < 200:
            self.stdout.write(self.style.WARNING(f"Not enough rows for {kind}; skipping."))
            return

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=test_ratio, seed=seed)
        model = fit_logreg(X_train, y_train, feature_names=feature_names, lr=0.15, l2=0.001, epochs=400, seed=seed)
        metrics = evaluate_binary(model, X_test, y_test, threshold=0.5)

        MLModelRun.objects.create(
            kind=kind,
            params={
                "feature_names": model.feature_names,
                "w": model.w,
                "b": model.b,
                "mean": model.mean,
                "std": model.std,
                "cutoff": cutoff,
            },
            metrics={
                "n_rows": len(X),
                "test_ratio": test_ratio,
                **metrics,
            }
        )

        self.stdout.write(self.style.SUCCESS(f"{kind} trained (cutoff={cutoff}) acc={metrics['accuracy']} f1={metrics['f1']}"))

