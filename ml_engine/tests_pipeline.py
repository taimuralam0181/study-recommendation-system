import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse

from accounts.models import UserProfile
from ml_engine.models import PredictionLog, StudyMaterial, UploadedDataset
from userpanel.models import Semester, Student, StudentSubjectPerformance, Subject

try:
    import celery  # noqa: F401
    HAS_CELERY = True
except Exception:
    HAS_CELERY = False


class PredictionPlanMaterialsFlowTests(TestCase):
    def setUp(self):
        self.student_user = User.objects.create_user(username="student1", password="pass12345")
        UserProfile.objects.create(user=self.student_user, role="student")
        Student.objects.create(user=self.student_user, cgpa=3.2)

        self.teacher_user = User.objects.create_user(username="teacher1", password="pass12345")
        UserProfile.objects.create(user=self.teacher_user, role="teacher")

        self.semester = Semester.objects.create(number=1, title="S1")
        self.subject = Subject.objects.create(semester=self.semester, name="Data Structures", department="CSE")
        StudyMaterial.objects.create(
            subject=self.subject,
            title="DS Quick Revision",
            description="Important topics for final exam",
            material_type="Link",
            link="https://example.com/ds-revision",
            added_by=self.teacher_user,
        )

    @patch("userpanel.views.predict_final_and_probs", return_value=(28.5, 0.62, 0.31, 0.09))
    def test_enter_marks_redirects_to_recovery_plan_and_shows_materials(self, _mock_predict):
        self.client.force_login(self.student_user)

        response = self.client.post(
            reverse(
                "enter_subject_marks",
                kwargs={"semester_number": self.semester.number, "subject_id": self.subject.id},
            ),
            {
                "hand_marks": "4",
                "attendance_percentage": "88",
                "ct_marks": "8",
                "midterm_marks": "22",
            },
            follow=True,
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Recovery Plan")
        self.assertContains(response, "Recommended Study Materials")
        self.assertContains(response, "DS Quick Revision")

        perf = StudentSubjectPerformance.objects.get(
            user=self.student_user,
            semester=self.semester,
            subject=self.subject,
        )
        self.assertGreater(perf.total, 0)
        self.assertGreater(perf.final_pred, 0)

    @patch(
        "ml_engine.predict_api.predict_student_outcome",
        return_value={
            "predicted_final_marks": 29.1,
            "predicted_grade": "A-",
            "grade_probabilities": {"A+": 0.1, "A": 0.3, "A-": 0.2, "B+": 0.2, "F": 0.05},
            "predicted_total": 72.4,
            "weak_signals": [],
            "model_versions": {"regression": "v1", "classification": "v1"},
        },
    )
    def test_predict_api_returns_links_and_persists_audit_log(self, _mock_predict):
        self.client.force_login(self.student_user)

        StudentSubjectPerformance.objects.create(
            user=self.student_user,
            semester=self.semester,
            subject=self.subject,
            hand_marks=2,
            attendance_marks=3,
            attendance_percentage=62.0,
            ct_marks=4,
            midterm_marks=12,
            total=45,
            final_pred=20.0,
            total_pred=45.0,
            predicted_grade="C",
            prob_fail=0.42,
        )

        payload = {
            "semester": 1,
            "subject": "Data Structures",
            "assignment_marks": 3.5,
            "quiz_marks": 7.0,
            "attendance_percentage": 85,
            "midterm_marks": 21,
            "previous_cgpa": 3.2,
        }
        response = self.client.post(
            reverse("api_predict"),
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertTrue(data["success"])
        self.assertIn("study_plan_links", data)
        self.assertIn("response_time_ms", data)
        links = data["study_plan_links"]
        self.assertTrue(any("Recovery Plan" in item["title"] for item in links))
        self.assertTrue(any("DS Quick Revision" in item["title"] for item in links))

        self.assertEqual(
            PredictionLog.objects.filter(user=self.student_user, model_type="academic").count(),
            1,
        )

    @override_settings(ML_PREDICT_ASYNC=True)
    @unittest.skipUnless(HAS_CELERY, "Celery not installed in test environment")
    @patch("ml_engine.tasks.predict_student_outcome_task.delay")
    def test_predict_api_async_mode_returns_task_id(self, mock_delay):
        self.client.force_login(self.student_user)
        mock_delay.return_value = SimpleNamespace(id="task-123")

        payload = {
            "semester": 1,
            "subject": "Data Structures",
            "assignment_marks": 3.5,
            "quiz_marks": 7.0,
            "attendance_percentage": 85,
            "midterm_marks": 21,
            "previous_cgpa": 3.2,
            "async_mode": True,
        }
        response = self.client.post(
            reverse("api_predict"),
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 202)
        body = response.json()
        self.assertTrue(body["success"])
        self.assertTrue(body["queued"])
        self.assertEqual(body["task_id"], "task-123")
        self.assertIn("/api/predict/tasks/task-123/", body["status_url"])


class SecurityRateLimitTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="rate_user", password="pass12345")
        UserProfile.objects.create(user=self.user, role="student")
        Student.objects.create(user=self.user, cgpa=3.0)
        semester = Semester.objects.create(number=2, title="S2")
        Subject.objects.create(semester=semester, name="Algorithms", department="CSE")

    @override_settings(DASHBOARD_PREDICT_RATE_LIMIT=1, DASHBOARD_PREDICT_RATE_WINDOW_SECONDS=120)
    @patch("templates.dashboard.api.predict_final_and_probs", return_value=(30.0, 0.6, 0.25, 0.08))
    def test_dashboard_predict_rate_limit(self, _mock_predict):
        self.client.force_login(self.user)
        payload = {
            "semester": 2,
            "subject": "Algorithms",
            "hand_marks": 4,
            "attendance_percentage": 90,
            "ct_marks": 8,
            "midterm_marks": 24,
            "previous_cgpa": 3.0,
        }

        first = self.client.post(
            reverse("api_predict_grade"),
            data=json.dumps(payload),
            content_type="application/json",
        )
        second = self.client.post(
            reverse("api_predict_grade"),
            data=json.dumps(payload),
            content_type="application/json",
        )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 429)
        self.assertIn("retry_after_seconds", second.json())


class UploadTrainingOrchestrationTests(TestCase):
    def setUp(self):
        self.teacher = User.objects.create_user(username="teacher_upload", password="pass12345")
        UserProfile.objects.create(user=self.teacher, role="teacher")

    @patch("adminpanel.views.train_models")
    @patch("adminpanel.views.persist_dataset_rows")
    @patch("adminpanel.views.validate_dataset_frame")
    def test_upload_dataset_invokes_training_with_uploaded_dataset(
        self,
        mock_validate,
        mock_persist,
        mock_train,
    ):
        self.client.force_login(self.teacher)

        cleaned = pd.DataFrame(
            [
                {
                    "student_id": "S1",
                    "semester": 1,
                    "subject": "Data Structures",
                    "assignment_marks": 4.0,
                    "quiz_marks": 8.0,
                    "attendance_percentage": 90.0,
                    "midterm_marks": 23.0,
                    "previous_cgpa": 3.1,
                    "final_marks": 34.0,
                    "final_grade": "A-",
                }
            ]
        )
        mock_validate.return_value = (
            cleaned,
            SimpleNamespace(
                total_rows=1000,
                rows_after_cleanup=1000,
                dropped_missing_rows=0,
                dropped_invalid_numeric_rows=0,
            ),
        )
        mock_persist.return_value = {"dataset_rows": 1, "training_rows": 1}
        mock_train.return_value = {
            "regression": {"rmse": 1.2, "mae": 0.8, "algorithm": "RF", "id": 1, "version": "v1"},
            "classification": {
                "accuracy": 0.81,
                "f1_weighted": 0.79,
                "algorithm": "RF",
                "id": 2,
                "version": "v1",
            },
        }

        csv = (
            "student_id,semester,subject,assignment_marks,quiz_marks,attendance_percentage,"
            "midterm_marks,previous_cgpa,final_marks,final_grade\n"
            "S1,1,Data Structures,4,8,90,23,3.1,34,A-\n"
        )
        upload = SimpleUploadedFile("sample.csv", csv.encode("utf-8"), content_type="text/csv")

        response = self.client.post(
            reverse("upload_dataset"),
            data={"file": upload, "clear_existing": "on"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(UploadedDataset.objects.count(), 1)
        self.assertEqual(mock_train.call_count, 1)

        called_dataset = mock_train.call_args.kwargs["uploaded_dataset"]
        self.assertEqual(called_dataset.uploaded_by_id, self.teacher.id)

    @override_settings(ML_TRAIN_ASYNC=True)
    @unittest.skipUnless(HAS_CELERY, "Celery not installed in test environment")
    @patch("ml_engine.tasks.train_uploaded_dataset_task.delay")
    @patch("adminpanel.views.persist_dataset_rows")
    @patch("adminpanel.views.validate_dataset_frame")
    def test_upload_dataset_async_training_queue(
        self,
        mock_validate,
        mock_persist,
        mock_delay,
    ):
        self.client.force_login(self.teacher)
        mock_delay.return_value = SimpleNamespace(id="train-task-001")

        cleaned = pd.DataFrame(
            [
                {
                    "student_id": "S1",
                    "semester": 1,
                    "subject": "Data Structures",
                    "assignment_marks": 4.0,
                    "quiz_marks": 8.0,
                    "attendance_percentage": 90.0,
                    "midterm_marks": 23.0,
                    "previous_cgpa": 3.1,
                    "final_marks": 34.0,
                    "final_grade": "A-",
                }
            ]
        )
        mock_validate.return_value = (
            cleaned,
            SimpleNamespace(
                total_rows=1200,
                rows_after_cleanup=1200,
                dropped_missing_rows=0,
                dropped_invalid_numeric_rows=0,
            ),
        )
        mock_persist.return_value = {"dataset_rows": 1, "training_rows": 1}

        csv = (
            "student_id,semester,subject,assignment_marks,quiz_marks,attendance_percentage,"
            "midterm_marks,previous_cgpa,final_marks,final_grade\n"
            "S1,1,Data Structures,4,8,90,23,3.1,34,A-\n"
        )
        upload = SimpleUploadedFile("sample.csv", csv.encode("utf-8"), content_type="text/csv")

        response = self.client.post(
            reverse("upload_dataset"),
            data={"file": upload, "clear_existing": "on"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Training queued")
        self.assertEqual(mock_delay.call_count, 1)
