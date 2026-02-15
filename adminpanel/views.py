from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.conf import settings
from accounts.models import UserProfile
from django.db.models import Count, Avg
from django.db import DatabaseError, transaction
from django.http import FileResponse, HttpResponse, HttpResponseForbidden, Http404
import csv
import pandas as pd
import hashlib
import logging
from userpanel.models import Student
from userpanel.models import StudentSubjectPerformance
from userpanel.models import Semester, Subject
from django.contrib.auth.models import User
from ml_engine.models import StudyMaterial
from ml_engine.models import DatasetStudentPerformance, MLModelRun, UploadedDataset, TrainedModel
from ml_engine.ml_model import persist_dataset_rows, train_models, validate_dataset_frame
from .forms import DatasetUploadForm
from userpanel.academic_logic import (
    prefinal_total,
    required_final_for_cutoff,
    best_feasible_grade,
    generate_recovery_plan,
)


logger = logging.getLogger(__name__)


def _require_teacher(request):
    profile, _ = UserProfile.objects.get_or_create(
        user=request.user,
        defaults={'role': 'student'}
    )
    if profile.role != 'teacher':
        return redirect('dashboard')
    return None


def _dataset_page_context(form=None, **extra):
    uploads = (
        UploadedDataset.objects
        .select_related('uploaded_by')
        .order_by('-uploaded_at')[:10]
    )
    trained_models = (
        TrainedModel.objects
        .select_related('trained_by', 'trained_on')
        .order_by('-created_at')[:10]
    )
    at_risk_students = (
        StudentSubjectPerformance.objects
        .select_related('user', 'semester', 'subject')
        .filter(prob_fail__gte=0.40)
        .order_by('-prob_fail')[:20]
    )
    context = {
        'form': form or DatasetUploadForm(),
        'uploads': uploads,
        'trained_models': trained_models,
        'at_risk_students': at_risk_students,
    }
    context.update(extra)
    return context


def _dataset_error(request, message, form):
    return render(request, 'adminpanel/upload_dataset.html', _dataset_page_context(form=form, error=message))


def _mark_upload_failed(uploaded, message):
    if not uploaded or not uploaded.pk:
        return
    try:
        updated = UploadedDataset.objects.filter(pk=uploaded.pk).update(
            train_status=UploadedDataset.STATUS_FAILED,
            error_message=str(message),
        )
        if not updated:
            logger.warning("Could not mark dataset upload %s as failed; row not found.", uploaded.pk)
    except DatabaseError as exc:
        logger.exception("Could not mark dataset upload %s as failed: %s", uploaded.pk, exc)


@login_required
def upload_dataset(request):
    profile, _ = UserProfile.objects.get_or_create(
        user=request.user,
        defaults={'role': 'student'}
    )
    if profile.role != 'teacher':
        return HttpResponseForbidden("Teacher access required.")

    if request.method == 'POST':
        form = DatasetUploadForm(request.POST, request.FILES)
        if not form.is_valid():
            return _dataset_error(request, "Please upload a valid CSV or Excel file.", form)

        file_obj = form.cleaned_data['file']
        checksum = hashlib.sha256()
        for chunk in file_obj.chunks():
            checksum.update(chunk)
        file_obj.seek(0)

        filename = (file_obj.name or "").lower()
        try:
            if filename.endswith((".xlsx", ".xls")):
                raw_df = pd.read_excel(file_obj, engine="openpyxl" if filename.endswith(".xlsx") else None)
            else:
                raw_df = pd.read_csv(file_obj)
        except Exception as exc:
            logger.exception("Failed to parse uploaded dataset: %s", exc)
            return _dataset_error(request, "Unable to parse the uploaded file. Please verify the file format.", form)

        clear_existing = request.POST.get('clear_existing') == 'on'
        uploaded = None
        try:
            file_obj.seek(0)
            uploaded = UploadedDataset.objects.create(
                file=file_obj,
                uploaded_by=request.user,
                checksum=checksum.hexdigest(),
                file_size=file_obj.size or 0,
                train_status=UploadedDataset.STATUS_PENDING,
            )

            cleaned_df, validation = validate_dataset_frame(raw_df)
            min_rows = int(getattr(settings, "ML_UPLOAD_MIN_ROWS", 1000))
            if validation.rows_after_cleanup < min_rows:
                raise ValueError(
                    f"Dataset must contain at least {min_rows} valid rows after cleaning."
                )

            with transaction.atomic():
                persisted = persist_dataset_rows(
                    cleaned_df,
                    uploaded_dataset=uploaded,
                    clear_existing=clear_existing,
                    batch_size=1000,
                )

            train_result = None
            queued_task_id = None
            if bool(getattr(settings, "ML_TRAIN_ASYNC", False)):
                try:
                    from ml_engine.tasks import train_uploaded_dataset_task
                except Exception as exc:
                    raise RuntimeError(
                        "Async training is enabled but Celery is not available. Install/configure Celery first."
                    ) from exc

                async_task = train_uploaded_dataset_task.delay(
                    uploaded_dataset_id=int(uploaded.id),
                    trained_by_id=int(request.user.id),
                    test_size=0.2,
                    random_state=42,
                    max_training_rows=int(getattr(settings, "ML_TRAIN_MAX_ROWS", 25000)),
                )
                queued_task_id = str(async_task.id)
                UploadedDataset.objects.filter(pk=uploaded.pk).update(
                    train_status=UploadedDataset.STATUS_RUNNING,
                    error_message="",
                )
            else:
                train_result = train_models(
                    uploaded_dataset=uploaded,
                    trained_by=request.user,
                    test_size=0.2,
                    random_state=42,
                )
        except ValueError as exc:
            _mark_upload_failed(uploaded, exc)
            return _dataset_error(request, str(exc), form)
        except Exception as exc:
            logger.exception("Dataset upload/training failed: %s", exc)
            _mark_upload_failed(uploaded, exc)
            return _dataset_error(
                request,
                "Dataset upload failed due to a server error. Please review file quality and try again.",
                form,
            )

        return render(
            request,
            'adminpanel/upload_dataset.html',
            _dataset_page_context(
                form=DatasetUploadForm(),
                success=(
                    (
                        f"Upload complete. Rows imported: {persisted['training_rows']}. "
                        f"Training queued (task id: {queued_task_id})."
                    )
                    if queued_task_id
                    else (
                        f"Upload complete. Rows imported: {persisted['training_rows']}. "
                        f"Regression RMSE: {train_result['regression']['rmse']:.4f}, "
                        f"MAE: {train_result['regression']['mae']:.4f}. "
                        f"Classification Accuracy: {train_result['classification']['accuracy']:.4f}, "
                        f"F1: {train_result['classification']['f1_weighted']:.4f}."
                    )
                ),
                validation=validation,
                train_result=train_result,
                queued_task_id=queued_task_id,
            )
        )

    return render(request, 'adminpanel/upload_dataset.html', _dataset_page_context(form=DatasetUploadForm()))


@login_required
def download_model_version(request, model_id):
    redirect_response = _require_teacher(request)
    if redirect_response:
        return redirect_response

    model = TrainedModel.objects.filter(id=model_id).first()
    if not model:
        raise Http404("Model version not found.")
    if not model.model_path:
        raise Http404("Model artifact path is missing.")

    try:
        file_handle = open(model.model_path, "rb")
    except OSError as exc:
        logger.exception("Failed to open model artifact %s: %s", model.model_path, exc)
        raise Http404("Model artifact file not available.")

    filename = model.model_path.split("\\")[-1].split("/")[-1]
    return FileResponse(file_handle, as_attachment=True, filename=filename)


@login_required
def teacher_dashboard(request):
    redirect_response = _require_teacher(request)
    if redirect_response:
        return redirect_response

    total_students = Student.objects.count()
    total_materials = StudyMaterial.objects.count()
    avg_math = Student.objects.aggregate(val=Avg('math_marks'))['val'] or 0
    avg_physics = Student.objects.aggregate(val=Avg('physics_marks'))['val'] or 0
    avg_cs = Student.objects.aggregate(val=Avg('cs_marks'))['val'] or 0
    avg_overall = Student.objects.aggregate(val=Avg('avg_score'))['val'] or 0

    subject_avgs = {
        'Math': avg_math,
        'Physics': avg_physics,
        'Computer Science': avg_cs,
    }
    weakest_subject = min(subject_avgs, key=subject_avgs.get) if total_students else 'N/A'
    level_counts = (
        Student.objects.values('level')
        .annotate(count=Count('id'))
        .order_by('level')
    )

    dataset_rows = DatasetStudentPerformance.objects.count()
    dataset_sources = list(
        DatasetStudentPerformance.objects.values_list('source', flat=True).distinct()[:5]
    )
    uploads = (
        UploadedDataset.objects
        .select_related('uploaded_by')
        .order_by('-uploaded_at')[:8]
    )
    trained_models = (
        TrainedModel.objects
        .select_related('trained_by', 'trained_on')
        .order_by('-created_at')[:8]
    )
    at_risk_students = (
        StudentSubjectPerformance.objects
        .select_related('user', 'semester', 'subject')
        .filter(prob_fail__gte=0.40)
        .order_by('-prob_fail')[:12]
    )
    upload_status_counts = {
        row['train_status']: row['count']
        for row in UploadedDataset.objects.values('train_status').annotate(count=Count('id'))
    }
    latest_monitor_run = (
        MLModelRun.objects
        .filter(kind='monitor_drift')
        .order_by('-created_at')
        .first()
    )
    latest_feedback_run = (
        MLModelRun.objects
        .filter(kind='production_feedback')
        .order_by('-created_at')
        .first()
    )
    ml_alerts = []
    if latest_monitor_run and bool((latest_monitor_run.metrics or {}).get('should_retrain')):
        ml_alerts.append("Drift threshold exceeded. Retraining recommended.")
    if latest_feedback_run:
        metrics = latest_feedback_run.metrics or {}
        if metrics.get('sample_count', 0) and (
            float(metrics.get('mae_final', 0.0)) >= float(getattr(settings, 'ML_FEEDBACK_ALERT_MAE_THRESHOLD', 8.0))
            or float(metrics.get('grade_accuracy', 1.0)) <= float(getattr(settings, 'ML_FEEDBACK_ALERT_GRADE_ACC_THRESHOLD', 0.55))
        ):
            ml_alerts.append("Prediction feedback metrics degraded. Review latest model quality.")

    return render(request, 'adminpanel/dashboard.html', {
        'total_students': total_students,
        'total_materials': total_materials,
        'level_counts': level_counts,
        'avg_math': avg_math,
        'avg_physics': avg_physics,
        'avg_cs': avg_cs,
        'avg_overall': avg_overall,
        'weakest_subject': weakest_subject,
        'dataset_rows': dataset_rows,
        'dataset_sources': dataset_sources,
        'uploads': uploads,
        'trained_models': trained_models,
        'at_risk_students': at_risk_students,
        'upload_status_counts': upload_status_counts,
        'latest_monitor_run': latest_monitor_run,
        'latest_feedback_run': latest_feedback_run,
        'ml_alerts': ml_alerts,
    })


@login_required
def student_list(request):
    redirect_response = _require_teacher(request)
    if redirect_response:
        return redirect_response

    students = Student.objects.select_related('user').all()

    return render(request, 'adminpanel/student_list.html', {
        'students': students
    })


@login_required
def material_list(request):
    redirect_response = _require_teacher(request)
    if redirect_response:
        return redirect_response

    materials = (
        StudyMaterial.objects
        .select_related('subject', 'subject__semester', 'added_by')
        .order_by('-created_at')
    )

    return render(request, 'adminpanel/material_list.html', {
        'materials': materials
    })


@login_required
def add_material(request):
    redirect_response = _require_teacher(request)
    if redirect_response:
        return redirect_response

    subjects = Subject.objects.filter(is_active=True).select_related('semester').order_by('semester__number', 'name')

    allowed_types = {'PDF', 'Image', 'Link'}
    if request.method == 'POST':
        subject_id = request.POST.get('subject_id', '').strip()
        title = request.POST.get('title', '').strip()
        description = request.POST.get('description', '').strip()
        material_type = request.POST.get('material_type', '').strip()
        link = request.POST.get('link', '').strip()
        upload = request.FILES.get('file')

        subject = Subject.objects.filter(id=subject_id, is_active=True).select_related('semester').first()
        if not subject or not all([title, description, material_type]):
            return render(request, 'adminpanel/material_form.html', {
                'error': 'Subject, title, description, and material type are required.',
                'subjects': subjects,
                'form': request.POST,
                'mode': 'add',
            })
        if material_type not in allowed_types:
            return render(request, 'adminpanel/material_form.html', {
                'error': 'Invalid material type.',
                'subjects': subjects,
                'form': request.POST,
                'mode': 'add',
            })

        if material_type == 'Link' and not link:
            return render(request, 'adminpanel/material_form.html', {
                'error': 'A link is required when material type is Link.',
                'subjects': subjects,
                'form': request.POST,
                'mode': 'add',
            })

        if material_type in ['PDF', 'Image'] and not upload:
            return render(request, 'adminpanel/material_form.html', {
                'error': 'Please upload a file for PDF or Image materials.',
                'subjects': subjects,
                'form': request.POST,
                'mode': 'add',
            })

        StudyMaterial.objects.create(
            subject=subject,
            title=title,
            description=description,
            material_type=material_type,
            file=upload,
            link=link,
            added_by=request.user
        )

        return redirect('material_list')

    return render(request, 'adminpanel/material_form.html', {
        'subjects': subjects,
        'mode': 'add',
    })


@login_required
def edit_material(request, material_id):
    redirect_response = _require_teacher(request)
    if redirect_response:
        return redirect_response

    material = StudyMaterial.objects.select_related('subject', 'subject__semester').filter(id=material_id).first()
    if not material:
        return redirect('material_list')

    subjects = Subject.objects.filter(is_active=True).select_related('semester').order_by('semester__number', 'name')

    allowed_types = {'PDF', 'Image', 'Link'}
    if request.method == 'POST':
        subject_id = request.POST.get('subject_id', '').strip()
        title = request.POST.get('title', '').strip()
        description = request.POST.get('description', '').strip()
        material_type = request.POST.get('material_type', '').strip()
        link = request.POST.get('link', '').strip()
        upload = request.FILES.get('file')

        subject = Subject.objects.filter(id=subject_id, is_active=True).select_related('semester').first()
        if not subject or not all([title, description, material_type]):
            return render(request, 'adminpanel/material_form.html', {
                'error': 'Subject, title, description, and material type are required.',
                'subjects': subjects,
                'material': material,
                'form': request.POST,
                'mode': 'edit',
            })
        if material_type not in allowed_types:
            return render(request, 'adminpanel/material_form.html', {
                'error': 'Invalid material type.',
                'subjects': subjects,
                'material': material,
                'form': request.POST,
                'mode': 'edit',
            })

        if material_type == 'Link' and not link:
            return render(request, 'adminpanel/material_form.html', {
                'error': 'A link is required when material type is Link.',
                'subjects': subjects,
                'material': material,
                'form': request.POST,
                'mode': 'edit',
            })

        if material_type in ['PDF', 'Image'] and not (upload or material.file):
            return render(request, 'adminpanel/material_form.html', {
                'error': 'Please upload a file for PDF or Image materials.',
                'subjects': subjects,
                'material': material,
                'form': request.POST,
                'mode': 'edit',
            })

        material.subject = subject
        material.title = title
        material.description = description
        material.material_type = material_type
        material.link = link
        if upload:
            material.file = upload
        material.save()

        return redirect('material_list')

    return render(request, 'adminpanel/material_form.html', {
        'subjects': subjects,
        'material': material,
        'mode': 'edit',
    })


@login_required
def delete_material(request, material_id):
    redirect_response = _require_teacher(request)
    if redirect_response:
        return redirect_response

    material = StudyMaterial.objects.filter(id=material_id).first()
    if material and request.method == 'POST':
        material.delete()
    return redirect('material_list')


@login_required
def subject_list(request):
    redirect_response = _require_teacher(request)
    if redirect_response:
        return redirect_response

    subjects = Subject.objects.select_related('semester').order_by('semester__number', 'name')
    semesters = Semester.objects.order_by('number').all()

    return render(request, 'adminpanel/subject_list.html', {
        'subjects': subjects,
        'semesters': semesters,
    })


@login_required
def add_subject(request):
    redirect_response = _require_teacher(request)
    if redirect_response:
        return redirect_response

    semesters = Semester.objects.order_by('number').all()
    if request.method == 'POST':
        name = request.POST.get('name', '').strip()
        semester_id = request.POST.get('semester_id', '').strip()
        department = request.POST.get('department', '').strip() or 'CSE'

        semester = Semester.objects.filter(id=semester_id).first()
        if not semester or not name:
            return render(request, 'adminpanel/subject_form.html', {
                'error': 'Semester and subject name are required.',
                'semesters': semesters,
                'form': request.POST,
                'mode': 'add',
            })

        Subject.objects.get_or_create(
            semester=semester,
            name=name,
            department=department,
            defaults={'is_active': True},
        )
        return redirect('teacher_subjects')

    return render(request, 'adminpanel/subject_form.html', {
        'semesters': semesters,
        'mode': 'add',
    })


@login_required
def edit_subject(request, subject_id):
    redirect_response = _require_teacher(request)
    if redirect_response:
        return redirect_response

    subject = Subject.objects.select_related('semester').filter(id=subject_id).first()
    if not subject:
        return redirect('teacher_subjects')

    semesters = Semester.objects.order_by('number').all()
    if request.method == 'POST':
        name = request.POST.get('name', '').strip()
        semester_id = request.POST.get('semester_id', '').strip()
        department = request.POST.get('department', '').strip() or 'CSE'
        is_active = request.POST.get('is_active') == 'on'

        semester = Semester.objects.filter(id=semester_id).first()
        if not semester or not name:
            return render(request, 'adminpanel/subject_form.html', {
                'error': 'Semester and subject name are required.',
                'semesters': semesters,
                'subject': subject,
                'form': request.POST,
                'mode': 'edit',
            })

        subject.name = name
        subject.semester = semester
        subject.department = department
        subject.is_active = is_active
        subject.save()
        return redirect('teacher_subjects')

    return render(request, 'adminpanel/subject_form.html', {
        'semesters': semesters,
        'subject': subject,
        'mode': 'edit',
    })


@login_required
def deactivate_subject(request, subject_id):
    redirect_response = _require_teacher(request)
    if redirect_response:
        return redirect_response

    subject = Subject.objects.filter(id=subject_id).first()
    if subject and request.method == 'POST':
        subject.is_active = False
        subject.save()
    return redirect('teacher_subjects')


@login_required
def delete_subject(request, subject_id):
    redirect_response = _require_teacher(request)
    if redirect_response:
        return redirect_response

    subject = Subject.objects.filter(id=subject_id).first()
    if not subject:
        return redirect('teacher_subjects')

    if request.method == 'POST':
        subject.delete()
    return redirect('teacher_subjects')


@login_required
def analytics(request):
    """
    Academic analytics for teachers:
    - Semester-wise average totals
    - Weakest subject per semester (by average total)
    - Weak/Average/Strong distribution per semester
    """
    redirect_response = _require_teacher(request)
    if redirect_response:
        return redirect_response

    # Overall semester averages
    sem_avgs_qs = (
        StudentSubjectPerformance.objects
        .values('semester__number')
        .annotate(avg_total=Avg('total'), n=Count('id'))
        .order_by('semester__number')
    )
    sem_avgs = {row['semester__number']: row for row in sem_avgs_qs}

    # Per-subject averages per semester (for weakest/strongest subject)
    subj_avgs_qs = (
        StudentSubjectPerformance.objects
        .values('semester__number', 'subject__name')
        .annotate(avg_total=Avg('total'), n=Count('id'))
        .order_by('semester__number', 'avg_total')
    )

    # Overall weakest subjects (across all semesters) by average total
    overall_weakest = (
        StudentSubjectPerformance.objects
        .values('subject__name')
        .annotate(avg_total=Avg('total'), n=Count('id'))
        .order_by('avg_total')[:10]
    )

    # Weak/Average/Strong distribution per semester
    perf_qs = StudentSubjectPerformance.objects.values('semester__number', 'total')
    dist = {}
    for row in perf_qs:
        sem = row['semester__number']
        total = row['total']
        dist.setdefault(sem, {'weak': 0, 'average': 0, 'strong': 0})
        if total < 50:
            dist[sem]['weak'] += 1
        elif total >= 80:
            dist[sem]['strong'] += 1
        else:
            dist[sem]['average'] += 1

    # Build per-semester analytics objects
    analytics_rows = []
    current_sem = None
    current_subjects = []
    for row in subj_avgs_qs:
        sem = row['semester__number']
        if current_sem is None:
            current_sem = sem
        if sem != current_sem:
            analytics_rows.append(_build_sem_analytics(current_sem, current_subjects, sem_avgs, dist))
            current_sem = sem
            current_subjects = []
        current_subjects.append(row)
    if current_sem is not None:
        analytics_rows.append(_build_sem_analytics(current_sem, current_subjects, sem_avgs, dist))

    # Ensure semesters with no subject data still appear (optional)
    # We show only semesters that have at least one performance row.

    return render(request, 'adminpanel/analytics.html', {
        'analytics_rows': analytics_rows,
        'overall_weakest': overall_weakest,
    })


def _build_sem_analytics(sem_number, subjects, sem_avgs, dist):
    subjects_sorted = sorted(subjects, key=lambda r: r['avg_total'] if r['avg_total'] is not None else 0)
    weakest = subjects_sorted[0] if subjects_sorted else None
    strongest = subjects_sorted[-1] if subjects_sorted else None

    return {
        'semester': sem_number,
        'sem_avg': (sem_avgs.get(sem_number) or {}).get('avg_total') or 0,
        'sem_n': (sem_avgs.get(sem_number) or {}).get('n') or 0,
        'weakest_subject': weakest,
        'strongest_subject': strongest,
        'distribution': dist.get(sem_number, {'weak': 0, 'average': 0, 'strong': 0}),
        'subjects': subjects_sorted,
    }


@login_required
def analytics_export_csv(request):
    """
    Exports per-subject averages per semester as CSV for academic reports.
    """
    redirect_response = _require_teacher(request)
    if redirect_response:
        return redirect_response

    rows = (
        StudentSubjectPerformance.objects
        .values('semester__number', 'subject__name')
        .annotate(avg_total=Avg('total'), n=Count('id'))
        .order_by('semester__number', 'subject__name')
    )

    resp = HttpResponse(content_type='text/csv')
    resp['Content-Disposition'] = 'attachment; filename="semester_subject_analytics.csv"'

    writer = csv.writer(resp)
    writer.writerow(['semester', 'subject', 'avg_total', 'samples'])
    for r in rows:
        writer.writerow([
            r['semester__number'],
            r['subject__name'],
            round(r['avg_total'] or 0, 2),
            r['n'],
        ])
    return resp


@login_required
def recovery_overview(request):
    """
    Teacher: choose a student and semester to view recovery plan metrics.
    """
    redirect_response = _require_teacher(request)
    if redirect_response:
        return redirect_response

    students = (
        UserProfile.objects
        .filter(role='student')
        .select_related('user')
        .order_by('user__username')
    )
    semesters = Semester.objects.order_by('number').all()

    return render(request, 'adminpanel/recovery_overview.html', {
        'students': students,
        'semesters': semesters,
    })


@login_required
def recovery_student_semester(request, user_id, semester_number):
    """
    Teacher: show subject-wise required final + urgency for a specific student and semester.
    """
    redirect_response = _require_teacher(request)
    if redirect_response:
        return redirect_response

    user = User.objects.filter(id=user_id).first()
    if not user:
        return redirect('teacher_recovery_overview')

    semester = Semester.objects.filter(number=semester_number).first()
    if not semester:
        return redirect('teacher_recovery_overview')

    subjects = list(Subject.objects.filter(semester=semester).order_by('name'))
    perf_map = {
        p.subject_id: p
        for p in StudentSubjectPerformance.objects.filter(user=user, semester=semester)
    }

    rows = []
    for subj in subjects:
        perf = perf_map.get(subj.id)
        if not perf:
            rows.append({'subject': subj, 'has_data': False})
            continue

        B = prefinal_total(perf.hand_marks, perf.attendance_marks, perf.ct_marks, perf.midterm_marks)
        target = best_feasible_grade(B)
        band, tier, steps = generate_recovery_plan(perf.midterm_marks, target['required_final'])

        required = {
            'd': required_final_for_cutoff(B, 40),
            'c': required_final_for_cutoff(B, 50),
            'c_plus': required_final_for_cutoff(B, 55),
            'b': required_final_for_cutoff(B, 60),
            'b_plus': required_final_for_cutoff(B, 65),
            'a_minus': required_final_for_cutoff(B, 70),
            'a': required_final_for_cutoff(B, 75),
            'a_plus': required_final_for_cutoff(B, 80),
        }

        rows.append({
            'subject': subj,
            'has_data': True,
            'perf': perf,
            'prefinal': B,
            'required': required,
            'target': target,
            'midterm_band': band,
            'urgency': tier,
            'plan_steps': steps,
        })

    return render(request, 'adminpanel/recovery_student_semester.html', {
        'student_user': user,
        'semester': semester,
        'rows': rows,
    })


@login_required
def recovery_student_semester_export(request, user_id, semester_number):
    """
    Teacher: export a student's semester recovery table as CSV for report submission.
    """
    redirect_response = _require_teacher(request)
    if redirect_response:
        return redirect_response

    user = User.objects.filter(id=user_id).first()
    semester = Semester.objects.filter(number=semester_number).first()
    if not user or not semester:
        return redirect('teacher_recovery_overview')

    subjects = list(Subject.objects.filter(semester=semester).order_by('name'))
    perf_map = {
        p.subject_id: p
        for p in StudentSubjectPerformance.objects.filter(user=user, semester=semester)
    }

    resp = HttpResponse(content_type='text/csv')
    resp['Content-Disposition'] = f'attachment; filename=\"recovery_{user.username}_S{semester.number}.csv\"'
    writer = csv.writer(resp)
    writer.writerow([
        'semester', 'subject',
        'hand', 'attendance', 'ct', 'midterm', 'final', 'total',
        'prefinal',
        'req_d', 'req_c', 'req_c_plus', 'req_b', 'req_b_plus', 'req_a_minus', 'req_a', 'req_a_plus',
        'best_target', 'req_final_target',
        'midterm_band', 'urgency',
    ])

    for subj in subjects:
        perf = perf_map.get(subj.id)
        if not perf:
            writer.writerow([semester.number, subj.name] + [''] * 16)
            continue

        B = prefinal_total(perf.hand_marks, perf.attendance_marks, perf.ct_marks, perf.midterm_marks)
        target = best_feasible_grade(B)
        band, tier, _ = generate_recovery_plan(perf.midterm_marks, target['required_final'])

        writer.writerow([
            semester.number, subj.name,
            perf.hand_marks, perf.attendance_marks, perf.ct_marks, perf.midterm_marks, perf.final_marks, perf.total,
            B,
            required_final_for_cutoff(B, 40),
            required_final_for_cutoff(B, 50),
            required_final_for_cutoff(B, 55),
            required_final_for_cutoff(B, 60),
            required_final_for_cutoff(B, 65),
            required_final_for_cutoff(B, 70),
            required_final_for_cutoff(B, 75),
            required_final_for_cutoff(B, 80),
            target['grade'], target['required_final'],
            band, tier,
        ])

    return resp
