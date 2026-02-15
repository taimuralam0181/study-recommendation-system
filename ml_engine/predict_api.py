import json
import logging
import time

from django.contrib.auth.decorators import login_required
from django.conf import settings
from django.core.cache import cache
from django.http import JsonResponse
from django.urls import reverse
from django.views.decorators.http import require_GET, require_POST
try:
    from celery.result import AsyncResult
except Exception:  # Optional until celery is installed
    AsyncResult = None

from ml_engine.audit import log_academic_prediction
from ml_engine.ml_model import predict_student_outcome
from ml_engine.models import StudyMaterial
from ml_engine.security import check_rate_limit
from userpanel.models import StudentSubjectPerformance


logger = logging.getLogger(__name__)


def _validate_payload(payload):
    cleaned_payload = {
        "semester": int(payload["semester"]),
        "subject": str(payload["subject"]).strip(),
        "assignment_marks": float(payload["assignment_marks"]),
        "quiz_marks": float(payload["quiz_marks"]),
        "attendance_percentage": float(payload["attendance_percentage"]),
        "midterm_marks": float(payload["midterm_marks"]),
        "previous_cgpa": float(payload["previous_cgpa"]),
    }
    if not cleaned_payload["subject"]:
        raise ValueError("subject cannot be empty.")
    if len(cleaned_payload["subject"]) > 120:
        raise ValueError("subject is too long.")
    if not (1 <= cleaned_payload["semester"] <= 12):
        raise ValueError("semester must be between 1 and 12.")
    if not (0 <= cleaned_payload["assignment_marks"] <= 5):
        raise ValueError("assignment_marks must be between 0 and 5.")
    if not (0 <= cleaned_payload["quiz_marks"] <= 10):
        raise ValueError("quiz_marks must be between 0 and 10.")
    if not (0 <= cleaned_payload["attendance_percentage"] <= 100):
        raise ValueError("attendance_percentage must be between 0 and 100.")
    if not (0 <= cleaned_payload["midterm_marks"] <= 30):
        raise ValueError("midterm_marks must be between 0 and 30.")
    if not (0 <= cleaned_payload["previous_cgpa"] <= 4.0):
        raise ValueError("previous_cgpa must be between 0 and 4.0.")
    return cleaned_payload


def _build_support_context(user, subject_name):
    weak_rows = (
        StudentSubjectPerformance.objects
        .select_related("semester", "subject")
        .filter(user=user, total__lt=50)
        .order_by("total")[:5]
    )
    weak_subjects = []
    study_plan_links = []
    for row in weak_rows:
        plan_url = reverse(
            "recovery_plan",
            kwargs={
                "semester_number": row.semester.number,
                "subject_id": row.subject_id,
            },
        )
        weak_subjects.append(
            {
                "semester": row.semester.number,
                "subject": row.subject.name,
                "total": row.total,
                "predicted_grade": row.predicted_grade,
                "plan_url": plan_url,
            }
        )
        study_plan_links.append(
            {
                "title": f"Recovery Plan - S{row.semester.number} {row.subject.name}",
                "url": plan_url,
            }
        )

    if subject_name:
        materials = (
            StudyMaterial.objects
            .select_related("subject", "subject__semester")
            .filter(subject__name__iexact=subject_name, subject__is_active=True)
            .order_by("-created_at")[:5]
        )
        for material in materials:
            target_url = material.link or (material.file.url if material.file else "")
            if target_url:
                study_plan_links.append(
                    {
                        "title": f"{material.subject.name}: {material.title}",
                        "url": target_url,
                    }
                )

    return weak_subjects, study_plan_links


@login_required
@require_POST
def api_predict(request):
    """
    POST /api/predict/
    JSON body:
    {
      "semester": 3,
      "subject": "Data Structures",
      "assignment_marks": 4.1,
      "quiz_marks": 7.8,
      "attendance_percentage": 88,
      "midterm_marks": 22,
      "previous_cgpa": 3.2
    }
    """
    rate_limit = int(getattr(settings, "ML_API_RATE_LIMIT", 30))
    rate_window = int(getattr(settings, "ML_API_RATE_WINDOW_SECONDS", 60))
    allowed, retry_after = check_rate_limit(
        request=request,
        scope="ml_api_predict",
        limit=rate_limit,
        window_seconds=rate_window,
    )
    if not allowed:
        return JsonResponse(
            {
                "success": False,
                "message": "Too many prediction requests. Please retry shortly.",
                "retry_after_seconds": retry_after,
            },
            status=429,
        )

    started = time.perf_counter()
    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        return JsonResponse(
            {
                "success": False,
                "message": "Invalid JSON payload.",
            },
            status=400,
        )

    required_fields = [
        "semester",
        "subject",
        "assignment_marks",
        "quiz_marks",
        "attendance_percentage",
        "midterm_marks",
        "previous_cgpa",
    ]
    missing = [field for field in required_fields if field not in payload]
    if missing:
        return JsonResponse(
            {
                "success": False,
                "message": "Missing required fields.",
                "missing_fields": missing,
            },
            status=400,
        )

    try:
        cleaned_payload = _validate_payload(payload)
        async_mode = bool(payload.get("async_mode")) and bool(getattr(settings, "ML_PREDICT_ASYNC", False))
        if async_mode:
            if AsyncResult is None:
                raise ValueError("Async queue is not available. Install and configure Celery first.")
            from ml_engine.tasks import predict_student_outcome_task

            task = predict_student_outcome_task.delay(cleaned_payload)
            cache.set(
                f"predict_task_meta:{task.id}",
                {
                    "user_id": int(request.user.id),
                    "payload": cleaned_payload,
                    "started_at": time.time(),
                    "logged": False,
                },
                timeout=3600,
            )
            return JsonResponse(
                {
                    "success": True,
                    "queued": True,
                    "task_id": task.id,
                    "status_url": reverse("api_predict_status", kwargs={"task_id": task.id}),
                },
                status=202,
            )

        prediction = predict_student_outcome(cleaned_payload)
        weak_subjects, study_plan_links = _build_support_context(request.user, cleaned_payload["subject"])
        response_payload = {
            "success": True,
            "prediction": prediction,
            "weak_subjects": weak_subjects,
            "study_plan_links": study_plan_links,
        }
        response_time_ms = (time.perf_counter() - started) * 1000.0
        log_academic_prediction(
            user=request.user,
            input_features=cleaned_payload,
            prediction_result=response_payload,
            response_time_ms=response_time_ms,
        )

        return JsonResponse(
            {
                **response_payload,
                "response_time_ms": round(response_time_ms, 2),
            }
        )
    except ValueError as exc:
        return JsonResponse(
            {
                "success": False,
                "message": str(exc),
            },
            status=400,
        )
    except Exception as exc:
        logger.exception("Predict API failed: %s", exc)
        return JsonResponse(
            {
                "success": False,
                "message": "Server error while generating prediction.",
            },
            status=500,
        )


@login_required
@require_GET
def api_predict_status(request, task_id):
    if AsyncResult is None:
        return JsonResponse(
            {
                "success": False,
                "message": "Async queue is not available.",
            },
            status=503,
        )

    result = AsyncResult(task_id)
    meta_key = f"predict_task_meta:{task_id}"
    meta = cache.get(meta_key) or {}
    if meta and meta.get("user_id") not in (None, request.user.id):
        return JsonResponse(
            {
                "success": False,
                "message": "You are not allowed to access this prediction task.",
            },
            status=403,
        )

    if result.state in {"PENDING", "RECEIVED", "STARTED", "RETRY"}:
        return JsonResponse(
            {
                "success": True,
                "queued": True,
                "task_id": task_id,
                "state": result.state,
                "ready": False,
            },
            status=202,
        )

    if result.state == "FAILURE":
        message = str(result.result) if result.result else "Async prediction failed."
        return JsonResponse(
            {
                "success": False,
                "task_id": task_id,
                "state": result.state,
                "message": message,
            },
            status=500,
        )

    prediction = result.result or {}
    payload = dict(meta.get("payload") or {})
    subject_name = str(payload.get("subject", "")).strip()
    weak_subjects, study_plan_links = _build_support_context(request.user, subject_name)

    response_payload = {
        "success": True,
        "queued": True,
        "ready": True,
        "task_id": task_id,
        "state": result.state,
        "prediction": prediction,
        "weak_subjects": weak_subjects,
        "study_plan_links": study_plan_links,
    }

    if meta.get("user_id") == request.user.id and not bool(meta.get("logged")) and payload:
        response_time_ms = max(0.0, (time.time() - float(meta.get("started_at", time.time()))) * 1000.0)
        log_academic_prediction(
            user=request.user,
            input_features=payload,
            prediction_result=response_payload,
            response_time_ms=response_time_ms,
        )
        meta["logged"] = True
        cache.set(meta_key, meta, timeout=300)
        response_payload["response_time_ms"] = round(response_time_ms, 2)

    return JsonResponse(response_payload)
