from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from ml_engine.models import MLModelRun, TrainedModel


class Command(BaseCommand):
    help = "Activate a previously trained model version (rollback support)."

    def add_arguments(self, parser):
        parser.add_argument("--model-version", required=True, help="Model version string to activate.")
        parser.add_argument(
            "--task",
            default="both",
            choices=["regression", "classification", "both"],
            help="Which task to activate for the provided version.",
        )

    def handle(self, *args, **options):
        version = str(options["model_version"]).strip()
        task = str(options["task"]).strip()
        if not version:
            raise CommandError("version is required.")

        task_types = [TrainedModel.TASK_REGRESSION, TrainedModel.TASK_CLASSIFICATION]
        if task == "regression":
            task_types = [TrainedModel.TASK_REGRESSION]
        elif task == "classification":
            task_types = [TrainedModel.TASK_CLASSIFICATION]

        activated_ids = []
        with transaction.atomic():
            for task_type in task_types:
                target = (
                    TrainedModel.objects
                    .filter(task_type=task_type, version=version)
                    .order_by("-created_at")
                    .first()
                )
                if not target:
                    raise CommandError(f"No model found for task={task_type} with version={version}.")

                TrainedModel.objects.filter(task_type=task_type, is_active=True).update(is_active=False)
                target.is_active = True
                target.save(update_fields=["is_active"])
                activated_ids.append(target.id)

            MLModelRun.objects.create(
                kind="model_rollback",
                params={"version": version, "task": task},
                metrics={"activated_model_ids": activated_ids},
            )

        self.stdout.write(
            self.style.SUCCESS(
                f"Activated version={version} for task={task}. model_ids={activated_ids}"
            )
        )
