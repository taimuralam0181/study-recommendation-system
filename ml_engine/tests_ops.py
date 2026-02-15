from django.contrib.auth.models import User
from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import TestCase, override_settings

from ml_engine.models import MLModelRun, TrainedModel


class ModelRollbackCommandTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="ops_teacher", password="pass12345")

        TrainedModel.objects.create(
            task_type=TrainedModel.TASK_REGRESSION,
            algorithm="RF",
            version="v_old",
            model_path="models/old_reg.pkl",
            feature_columns=["semester", "subject"],
            target_column="final_marks",
            metrics={},
            trained_by=self.user,
            is_active=True,
        )
        TrainedModel.objects.create(
            task_type=TrainedModel.TASK_REGRESSION,
            algorithm="RF",
            version="v_new",
            model_path="models/new_reg.pkl",
            feature_columns=["semester", "subject"],
            target_column="final_marks",
            metrics={},
            trained_by=self.user,
            is_active=False,
        )

    def test_activate_model_version_command_sets_requested_version_active(self):
        call_command("activate_model_version", "--model-version", "v_new", "--task", "regression")

        active_versions = list(
            TrainedModel.objects
            .filter(task_type=TrainedModel.TASK_REGRESSION, is_active=True)
            .values_list("version", flat=True)
        )
        self.assertEqual(active_versions, ["v_new"])
        self.assertTrue(MLModelRun.objects.filter(kind="model_rollback").exists())


class SecuritySmokeCommandTests(TestCase):
    def test_security_smoke_test_runs(self):
        call_command("security_smoke_test")

    @override_settings(
        DEBUG=True,
        SECRET_KEY="django-insecure-test-key",
        ALLOWED_HOSTS=[],
        SESSION_COOKIE_SECURE=False,
        CSRF_COOKIE_SECURE=False,
        SECURE_HSTS_SECONDS=0,
    )
    def test_security_smoke_test_strict_fails_on_dev_defaults(self):
        with self.assertRaises(CommandError):
            call_command("security_smoke_test", "--strict")
