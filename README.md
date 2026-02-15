# University-Grade Personalized Study Recommendation System

This Django project implements a production-style ML lifecycle for university academic prediction.

## What It Does

1. Teachers upload CSV/Excel datasets containing:
   - `student_id`
   - `semester`
   - `subject`
   - `assignment_marks`
   - `quiz_marks`
   - `attendance_percentage`
   - `midterm_marks`
   - `previous_cgpa`
   - `final_marks`
   - `final_grade`
2. System validates and ingests rows into database tables.
3. Regression + classification models are trained and versioned automatically.
4. Students get predictions, grade probabilities, weak-subject signals, and study-plan links.
5. API endpoint provides JSON predictions: `POST /api/predict/`.

## Core Data Models

- `UploadedDataset`: upload metadata + training status lifecycle.
- `DatasetStudentPerformance`: persisted validated dataset rows.
- `CSETrainingExample`: ML-ready training table.
- `TrainedModel`: artifact registry (version, metrics, confusion matrix, report, paths).

## Training Pipeline

Implemented in `ml_engine/ml_model.py`:

- Feature engineering:
  - Numeric scaling with `StandardScaler`
  - Categorical encoding for `subject` via `OneHotEncoder`
- Models:
  - Regression: `RandomForestRegressor`, `GradientBoostingRegressor`
  - Classification: `RandomForestClassifier`, `GradientBoostingClassifier`
- Model selection:
  - Best regression by lowest RMSE
  - Best classification by highest weighted F1
- Evaluation:
  - Regression: RMSE, MAE
  - Classification: Accuracy, weighted F1, confusion matrix, classification report
  - Cross-validation: CV RMSE/MAE and CV Accuracy/F1
  - Calibration: Brier score (macro + per-grade)
  - Class imbalance diagnostics
- Artifacts:
  - Saved under `models/university_ml/*.pkl`
  - Metadata saved to `TrainedModel`
  - Experiment runs saved to `MLModelRun`

## Teacher Workflow

1. Open `Teacher -> Upload Dataset`.
2. Upload CSV/Excel (1000+ valid rows recommended/required by view).
3. System validates dataset and imports rows.
4. Training starts automatically.
5. View upload history, status, metrics, at-risk students, and download model files.

## Student Workflow

1. Open `ML -> Recommendation`.
2. Enter semester + subject + current marks.
3. View:
   - Predicted final marks
   - Predicted grade
   - Probabilities for each grade
   - Weak subjects
   - Personalized study plan links

## REST API

Endpoint: `POST /api/predict/`

Payload:

```json
{
  "semester": 3,
  "subject": "Data Structures",
  "assignment_marks": 4.2,
  "quiz_marks": 8.0,
  "attendance_percentage": 90,
  "midterm_marks": 23,
  "previous_cgpa": 3.4
}
```

Response includes:
- predicted final marks
- predicted grade
- grade probabilities
- weak subjects (for authenticated user)
- study plan links
- audit logging (`PredictionLog`) with response time

Security and reliability:
- strict payload validation for prediction endpoints
- per-user rate limiting for prediction APIs

## Setup

```bash
pip install -r requirements.txt
python manage.py makemigrations
python manage.py migrate
python manage.py runserver
```

For async training/prediction queue:

```bash
celery -A study_recommendation worker -l info
```

Set env values (see `.env.example`):
- `ML_TRAIN_ASYNC=true`
- `ML_PREDICT_ASYNC=true`
- `CELERY_BROKER_URL=redis://localhost:6379/0`

## Generate Example Dataset

```bash
python scripts/generate_example_dataset.py --students 120 --output media/datasets/university_example_dataset.csv
```

This generates a realistic 1000+ row dataset suitable for teacher upload.

## Drift Monitoring and Retraining

Run monitoring:

```bash
python manage.py monitor_ml_models --lookback-days 30 --min-samples 50
```

Auto-retrain if drift threshold is breached:

```bash
python manage.py monitor_ml_models --lookback-days 30 --drift-threshold 0.35 --auto-retrain
```

Feedback loop (real outcomes vs predictions):

```bash
python manage.py evaluate_prediction_feedback --lookback-days 180 --min-samples 30
```

## Rollback Model Version

Activate a previous model version:

```bash
python manage.py activate_model_version --model-version 20260214101530 --task both
```

`--task` supports: `regression`, `classification`, `both`

## Security and Load Testing

Security smoke test:

```bash
python manage.py security_smoke_test
python manage.py security_smoke_test --strict
```

Load test prediction endpoint:

```bash
python scripts/load_test_predict.py --url http://127.0.0.1:8000/api/predict/ --requests 100 --concurrency 10 --cookie "sessionid=..."
```
