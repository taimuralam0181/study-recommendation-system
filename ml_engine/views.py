from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.urls import reverse
import logging
from .models import StudyMaterial
from accounts.models import UserProfile
from userpanel.models import Student
from userpanel.models import StudentSubjectPerformance
from userpanel.academic_logic import required_final_for_cutoff, grade_from_total
from .ml_model import predict_student_outcome
from .academic_ml import SubjectRecord, recommend_student_level
from userpanel.models import Semester, Subject
from django.db.models import Q


logger = logging.getLogger(__name__)


def _render_input(request, error=None, form=None):
    # Syllabus-aligned subjects for CSE-only validation and safe selection.
    subjects = Subject.objects.filter(is_active=True).select_related('semester').order_by('semester__number', 'name')
    ctx = {'subjects': subjects}
    if error:
        ctx['error'] = error
    if form:
        ctx['form'] = form
    return render(request, 'ml_engine/input.html', ctx)


@login_required
def add_material(request):
    # Delegate to teacher panel to keep a single source of truth
    from adminpanel.views import add_material as teacher_add_material
    return teacher_add_material(request)


@login_required
def index(request):
    return redirect('get_recommendation')


def _require_student(request):
    profile, _ = UserProfile.objects.get_or_create(
        user=request.user,
        defaults={'role': 'student'}
    )
    if profile.role != 'student':
        return redirect('teacher_dashboard')
    return None


@login_required
def get_recommendation(request):
    redirect_response = _require_student(request)
    if redirect_response:
        return redirect_response

    if request.method == 'POST':
        form_vals = {
            'semester': request.POST.get('semester', ''),
            'subject_id': request.POST.get('subject_id', ''),
            'subject': request.POST.get('subject', ''),
            'assignment_marks': request.POST.get('assignment_marks', ''),
            'quiz_marks': request.POST.get('quiz_marks', ''),
            'midterm_marks': request.POST.get('midterm_marks', ''),
            'attendance_percentage': request.POST.get('attendance_percentage', ''),
        }
        try:
            semester = int(request.POST.get('semester'))
            # Prefer syllabus-aligned subject selection.
            subject_id_raw = (request.POST.get('subject_id') or '').strip()
            subject_id = int(subject_id_raw) if subject_id_raw else None
            subject = (request.POST.get('subject') or '').strip()  # fallback for older forms
            assignment_marks = int(request.POST.get('assignment_marks'))
            quiz_marks = int(request.POST.get('quiz_marks'))
            midterm_marks = int(request.POST.get('midterm_marks'))
            attendance_percentage = float(request.POST.get('attendance_percentage'))
        except (TypeError, ValueError):
            return _render_input(request, error='Please enter valid numeric values for all fields.', form=form_vals)

        if not (1 <= semester <= 8):
            return _render_input(request, error='Semester must be between 1 and 8.', form=form_vals)

        # Enforce "CSE only": subject must exist in seeded CS syllabus for the selected semester.
        sem_obj = Semester.objects.filter(number=semester).first()
        if not sem_obj:
            return _render_input(request, error='Semester is not configured yet. Ask admin to seed the syllabus.', form=form_vals)

        selected_subject = None
        if subject_id is not None:
            selected_subject = Subject.objects.filter(id=subject_id, is_active=True).select_related('semester').first()
            if not selected_subject:
                return _render_input(request, error='Invalid subject selection. Please select a subject from the syllabus list.', form=form_vals)
            if selected_subject.semester_id != sem_obj.id:
                return _render_input(request, error='Selected subject does not match the chosen semester. Please re-select both fields.', form=form_vals)
            subject = selected_subject.name
        else:
            if not subject:
                return _render_input(request, error='Subject is required.', form=form_vals)
            selected_subject = Subject.objects.filter(semester=sem_obj, name__iexact=subject, is_active=True).first()
            if not selected_subject:
                return _render_input(request, error='Subject not found for this semester. Select a CSE subject from the syllabus.', form=form_vals)

        # Academic scheme:
        # Assignment(5) + Attendance(5, scaled from %) + Quiz/CT(10) + Midterm(30) + Final(50 predicted) = 100
        if not (0 <= assignment_marks <= 5 and 0 <= quiz_marks <= 10 and 0 <= midterm_marks <= 30 and 0 <= attendance_percentage <= 100):
            return _render_input(request, error='Ranges: Assignment 0-5, Attendance% 0-100, Quiz/CT 0-10, Midterm 0-30.', form=form_vals)

        attendance_marks = int(round(5.0 * (attendance_percentage / 100.0)))
        cgpa_prev = float(Student.objects.filter(user=request.user).values_list('cgpa', flat=True).first() or 0.0)

        try:
            prediction = predict_student_outcome({
                'semester': semester,
                'subject': subject,
                'assignment_marks': assignment_marks,
                'attendance_percentage': attendance_percentage,
                'quiz_marks': quiz_marks,
                'midterm_marks': midterm_marks,
                'previous_cgpa': cgpa_prev,
            })
        except Exception as exc:
            logger.exception("Prediction failed for user=%s: %s", request.user.username, exc)
            return _render_input(
                request,
                error='Prediction model is unavailable or input data is invalid. Please contact your teacher/admin.',
                form=form_vals,
            )

        final_pred = float(prediction['predicted_final_marks'])
        total_pred = float(prediction['predicted_total'])
        predicted_grade = str(prediction['predicted_grade']) or grade_from_total(total_pred)
        probabilities = prediction.get('grade_probabilities', {})
        prob_a = float(probabilities.get('A', 0.0)) + float(probabilities.get('A+', 0.0))
        prob_aplus = float(probabilities.get('A+', 0.0))
        prob_fail = float(probabilities.get('F', 0.0))

        # Study support level (explainable mapping)
        if total_pred >= 80:
            study_level = "Advanced"
        elif total_pred >= 60:
            study_level = "Intermediate"
        else:
            study_level = "Beginner"

        explanation = (
            "Prediction uses trained regression + multi-class classification models. "
            "Inputs: semester, subject, assignment, quiz, attendance, midterm, previous CGPA. "
            "Final marks and grade probabilities are generated from active model versions."
        )

        required = {
            'd': required_final_for_cutoff(total_pred - final_pred, 40),
            'c': required_final_for_cutoff(total_pred - final_pred, 50),
            'c_plus': required_final_for_cutoff(total_pred - final_pred, 55),
            'b': required_final_for_cutoff(total_pred - final_pred, 60),
            'b_plus': required_final_for_cutoff(total_pred - final_pred, 65),
            'a_minus': required_final_for_cutoff(total_pred - final_pred, 70),
            'a': required_final_for_cutoff(total_pred - final_pred, 75),
            'a_plus': required_final_for_cutoff(total_pred - final_pred, 80),
        }

        materials_qs = (
            StudyMaterial.objects
            .select_related('subject', 'subject__semester')
            .filter(subject=selected_subject, subject__is_active=True)
            .order_by('-created_at')
        )
        # Prefer PDF materials if available.
        pdf_q = Q(material_type__iexact='PDF') | Q(link__iendswith='.pdf') | Q(file__iendswith='.pdf')
        materials_pdf = materials_qs.filter(pdf_q)
        materials = materials_pdf if materials_pdf.exists() else materials_qs

        weak_subject_rows = (
            StudentSubjectPerformance.objects
            .select_related('semester', 'subject')
            .filter(user=request.user, total__lt=50)
            .order_by('total')[:5]
        )
        weak_subjects = [
            {
                'semester': p.semester.number,
                'subject': p.subject.name,
                'total': p.total,
                'predicted_grade': p.predicted_grade or '-',
                'plan_url': reverse(
                    'recovery_plan',
                    kwargs={'semester_number': p.semester.number, 'subject_id': p.subject_id},
                ),
            }
            for p in weak_subject_rows
        ]

        personalized_links = []
        for ws in weak_subjects:
            personalized_links.append(
                {
                    'title': f"Recovery Plan - S{ws['semester']} {ws['subject']}",
                    'url': ws['plan_url'],
                }
            )
        for material in materials[:5]:
            material_url = material.link or (material.file.url if material.file else '')
            if material_url:
                personalized_links.append(
                    {
                        'title': f"{material.subject.name}: {material.title}",
                        'url': material_url,
                    }
                )

        academic_ml = _academic_ml_summary_for_user(request.user)

        return render(request, 'ml_engine/result.html', {
            'semester': semester,
            'subject': subject,
            'study_level': study_level,
            'prefinal': total_pred - final_pred,
            'final_pred': final_pred,
            'total_pred': total_pred,
            'predicted_grade': predicted_grade,
            'prob_a': prob_a,
            'prob_aplus': prob_aplus,
            'prob_fail': prob_fail,
            'prob_a_pct': int(round(prob_a * 100.0)),
            'prob_aplus_pct': int(round(prob_aplus * 100.0)),
            'prob_fail_pct': int(round(prob_fail * 100.0)),
            'grade_probabilities': probabilities,
            'required': required,
            'explanation': explanation,
            'materials': materials,
            'weak_subjects': weak_subjects,
            'personalized_links': personalized_links,
            'model_versions': prediction.get('model_versions', {}),
            'academic_ml': academic_ml,
        })

    return _render_input(request)


def _academic_ml_summary_for_user(user):
    qs = StudentSubjectPerformance.objects.filter(user=user).select_related('semester', 'subject')
    records = [
        SubjectRecord(
            semester=p.semester.number,
            subject=p.subject.name,
            midterm=float(p.midterm_marks),
            total=float(p.total),
        )
        for p in qs
    ]
    cohort = _build_cohort_records()
    rec = recommend_student_level(student_records=records, cohort_records=cohort)
    return {
        'performance_cluster': rec.performance_cluster,
        'study_level': rec.study_level,
        'features': rec.features,
        'explanation': rec.explanation,
    }


def _build_cohort_records():
    qs = StudentSubjectPerformance.objects.select_related('semester', 'subject', 'user')
    cohort = {}
    for p in qs:
        cohort.setdefault(p.user_id, []).append(
            SubjectRecord(
                semester=p.semester.number,
                subject=p.subject.name,
                midterm=float(p.midterm_marks),
                total=float(p.total),
            )
        )
    return cohort
