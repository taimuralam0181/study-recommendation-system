from django.core.management.base import BaseCommand

from ml_engine.ml_model import train_models
from ml_engine.models import UploadedDataset


class Command(BaseCommand):
    help = "Train university-grade sklearn models (regression + multi-class classification)."

    def add_arguments(self, parser):
        parser.add_argument("--dataset-id", type=int, default=None)
        parser.add_argument("--test-size", type=float, default=0.2)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--max-rows", type=int, default=25000)

    def handle(self, *args, **options):
        dataset = None
        dataset_id = options.get("dataset_id")
        if dataset_id:
            dataset = UploadedDataset.objects.filter(id=dataset_id).first()

        result = train_models(
            uploaded_dataset=dataset,
            trained_by=None,
            test_size=float(options["test_size"]),
            random_state=int(options["seed"]),
            max_training_rows=int(options["max_rows"]),
        )

        self.stdout.write(
            self.style.SUCCESS(
                "Training complete. "
                f"Regression[{result['regression']['algorithm']}] "
                f"RMSE={result['regression']['rmse']:.4f}, MAE={result['regression']['mae']:.4f}; "
                f"Classification[{result['classification']['algorithm']}] "
                f"Acc={result['classification']['accuracy']:.4f}, "
                f"F1={result['classification']['f1_weighted']:.4f}"
            )
        )
