import csv
import os
import zipfile
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
from django.conf import settings

from ml_engine.models import DatasetStudentPerformance
from ml_engine.ml_model import recommend_level


class Command(BaseCommand):
    """
    Academic dataset loader (Kaggle -> preprocess -> DB).

    IMPORTANT:
    - Kaggle requires an API token in ~/.kaggle/kaggle.json (or %USERPROFILE%\\.kaggle\\kaggle.json on Windows).
    - This command downloads a public dataset, extracts a CSV, maps 3 exam scores into
      (math, physics, cs) "equivalent subjects", normalizes them, labels levels, and saves to DB.

    Example (dataset with math/reading/writing scores):
      python manage.py load_kaggle_dataset --dataset spscientist/students-performance-in-exams

    Mapping (explainable for viva):
      math     -> math score
      physics  -> reading score (equivalent analytical subject)
      cs       -> writing score (equivalent technical/logic subject)
    """

    def add_arguments(self, parser):
        parser.add_argument(
            '--dataset',
            required=True,
            help='Kaggle dataset slug, e.g. "spscientist/students-performance-in-exams".'
        )
        parser.add_argument(
            '--clear',
            action='store_true',
            help='Delete existing imported rows for this dataset source before importing.'
        )
        parser.add_argument(
            '--limit',
            type=int,
            default=0,
            help='Optional row limit for quicker imports (0 = no limit).'
        )

    def handle(self, *args, **options):
        dataset_slug = options['dataset']
        limit = options['limit']
        clear = options['clear']

        data_dir = Path(settings.BASE_DIR) / 'data' / 'kaggle' / dataset_slug.replace('/', '__')
        data_dir.mkdir(parents=True, exist_ok=True)
        zip_path = data_dir / 'dataset.zip'

        # Download using Kaggle API (python package). Network and token must be available on the user's machine.
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except Exception as e:
            raise CommandError(
                "Kaggle API package not available. Install with `pip install kaggle` "
                "and configure your Kaggle API token. Original error: %s" % e
            )

        api = KaggleApi()
        try:
            api.authenticate()
        except Exception as e:
            raise CommandError(
                "Kaggle authentication failed. Ensure kaggle.json token is configured. Error: %s" % e
            )

        self.stdout.write(self.style.WARNING(f"Downloading Kaggle dataset: {dataset_slug}"))
        api.dataset_download_files(dataset_slug, path=str(data_dir), quiet=False, unzip=False)

        # Kaggle often saves as <dataset>.zip; find the newest zip in the folder.
        zips = sorted(data_dir.glob('*.zip'), key=lambda p: p.stat().st_mtime, reverse=True)
        if not zips:
            raise CommandError("No zip file found after download in %s" % data_dir)
        zip_path = zips[0]

        extract_dir = data_dir / 'extracted'
        extract_dir.mkdir(parents=True, exist_ok=True)

        self.stdout.write(self.style.WARNING(f"Extracting: {zip_path.name}"))
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)

        csv_files = list(extract_dir.glob('*.csv'))
        if not csv_files:
            # Some datasets contain nested folders
            csv_files = list(extract_dir.rglob('*.csv'))
        if not csv_files:
            raise CommandError("No CSV file found in extracted dataset.")

        csv_path = csv_files[0]
        self.stdout.write(self.style.WARNING(f"Using CSV: {csv_path}"))

        if clear:
            deleted, _ = DatasetStudentPerformance.objects.filter(source=dataset_slug).delete()
            self.stdout.write(self.style.WARNING(f"Cleared {deleted} existing rows for source={dataset_slug}"))

        rows = []
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            # Try common column names used by popular "student performance" datasets
            # math candidates
            math_keys = ['math score', 'Math score', 'math', 'Math']
            physics_keys = ['physics score', 'Physics score', 'reading score', 'Reading score', 'physics', 'Physics', 'reading', 'Reading']
            cs_keys = ['computer science score', 'Computer Science score', 'writing score', 'Writing score', 'cs', 'CS', 'computer_science', 'writing', 'Writing']

            def pick_key(fieldnames, candidates):
                for c in candidates:
                    if c in fieldnames:
                        return c
                return None

            fieldnames = reader.fieldnames or []
            math_k = pick_key(fieldnames, math_keys)
            phy_k = pick_key(fieldnames, physics_keys)
            cs_k = pick_key(fieldnames, cs_keys)

            if not (math_k and phy_k and cs_k):
                raise CommandError(
                    "Could not auto-detect 3 score columns. Found fieldnames: %s" % fieldnames
                )

            for i, r in enumerate(reader, start=1):
                if limit and len(rows) >= limit:
                    break

                try:
                    math = float(r.get(math_k, '').strip())
                    physics = float(r.get(phy_k, '').strip())
                    cs = float(r.get(cs_k, '').strip())
                except Exception:
                    continue

                # Basic cleaning: enforce 0..100 range
                if not (0 <= math <= 100 and 0 <= physics <= 100 and 0 <= cs <= 100):
                    continue

                avg = (math + physics + cs) / 3
                level = recommend_level(math, physics, cs)

                # Simple normalization: score/100 (explainable min-max scaling for exam scores)
                mn = math / 100.0
                pn = physics / 100.0
                cn = cs / 100.0
                an = avg / 100.0

                rows.append(DatasetStudentPerformance(
                    source=dataset_slug,
                    math_marks=math,
                    physics_marks=physics,
                    cs_marks=cs,
                    avg_score=avg,
                    level=level,
                    math_norm=mn,
                    physics_norm=pn,
                    cs_norm=cn,
                    avg_norm=an,
                ))

        if not rows:
            raise CommandError("No valid rows parsed from dataset.")

        DatasetStudentPerformance.objects.bulk_create(rows, batch_size=1000)
        self.stdout.write(self.style.SUCCESS(f"Imported {len(rows)} rows into DatasetStudentPerformance."))
