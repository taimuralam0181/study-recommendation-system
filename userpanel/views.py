from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from accounts.models import UserProfile
from .models import (
    Student,
    StudentResultHistory,
    Semester,
    Subject,
    StudentSubjectPerformance,
)
from ml_engine.ml_model import explain_recommendation
from .academic_logic import (
    prefinal_total,
    required_final_for_cutoff,
    best_feasible_grade,
    midterm_band,
    urgency_tier,
    explain_target,
    generate_recovery_plan,
)
from ml_engine.academic_ml import SubjectRecord, recommend_student_level
from ml_engine.inference import predict_final_and_probs
from ml_engine.models import StudyMaterial


def _require_student_role(request):
    profile, _ = UserProfile.objects.get_or_create(
        user=request.user,
        defaults={'role': 'student'}
    )
    if profile.role != 'student':
        return redirect('teacher_dashboard')
    return None


@login_required
def dashboard(request):
    redirect_response = _require_student_role(request)
    if redirect_response:
        return redirect_response

    student = Student.objects.filter(user=request.user).first()
    history = StudentResultHistory.objects.filter(user=request.user).order_by('-created_at')[:5]
    ml_explanation = ''
    if student:
        _, _, ml_explanation = explain_recommendation(
            student.math_marks, student.physics_marks, student.cs_marks
        )

    return render(request, 'userpanel/dashboard.html', {
        'student': student,
        'history': history,
        'ml_explanation': ml_explanation,
        'academic_ml': _build_academic_ml_for_user(request.user),
    })


def _build_academic_ml_for_user(user, semester_number=None):
    qs = StudentSubjectPerformance.objects.filter(user=user).select_related('semester', 'subject')
    if semester_number is not None:
        qs = qs.filter(semester__number=semester_number)

    records = [
        SubjectRecord(
            semester=p.semester.number,
            subject=p.subject.name,
            midterm=float(p.midterm_marks),
            total=float(p.total),
        )
        for p in qs
    ]

    cohort = _build_cohort_records(semester_number=semester_number)
    rec = recommend_student_level(student_records=records, cohort_records=cohort)
    return {
        'performance_cluster': rec.performance_cluster,
        'study_level': rec.study_level,
        'features': rec.features,
        'explanation': rec.explanation,
    }


def _build_cohort_records(semester_number=None):
    """
    Builds cohort_records for KMeans-style clustering:
      {user_id: [SubjectRecord, ...]}

    If semester_number is provided, cohort is restricted to that semester to measure
    relative performance within the same semester cohort.
    """
    qs = StudentSubjectPerformance.objects.select_related('semester', 'subject', 'user')
    if semester_number is not None:
        qs = qs.filter(semester__number=semester_number)

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


def _recommended_materials_for_subject(subject, limit=5):
    qs = (
        StudyMaterial.objects
        .filter(subject=subject)
        .order_by('-created_at')
    )
    pdf_qs = qs.filter(material_type__iexact='PDF')
    if pdf_qs.exists():
        qs = pdf_qs
    return list(qs[:limit])


@login_required
def performance_overview(request):
    redirect_response = _require_student_role(request)
    if redirect_response:
        return redirect_response

    semesters = list(Semester.objects.order_by('number').all())
    perf = (
        StudentSubjectPerformance.objects
        .filter(user=request.user)
        .select_related('semester')
    )

    # Per-semester totals summary (weak/average/strong counts) for academic presentation
    summary = {}
    for p in perf:
        s = p.semester.number
        summary.setdefault(s, {'weak': 0, 'average': 0, 'strong': 0, 'count': 0})
        summary[s]['count'] += 1
        if p.total < 50:
            summary[s]['weak'] += 1
        elif p.total >= 80:
            summary[s]['strong'] += 1
        else:
            summary[s]['average'] += 1

    sem_cards = []
    for sem in semesters:
        stats = summary.get(sem.number)
        sem_cards.append({'semester': sem, 'stats': stats})

    return render(request, 'userpanel/performance_overview.html', {
        'sem_cards': sem_cards,
    })


@login_required
def semester_detail(request, semester_number):
    redirect_response = _require_student_role(request)
    if redirect_response:
        return redirect_response

    semester = Semester.objects.filter(number=semester_number).first()
    if not semester:
        return redirect('performance_overview')

    subjects = Subject.objects.filter(semester=semester, is_active=True).order_by('name')
    performances = {
        p.subject_id: p
        for p in StudentSubjectPerformance.objects.filter(user=request.user, semester=semester)
    }

    rows = []
    for subj in subjects:
        p = performances.get(subj.id)
        total = p.total if p else None
        if total is None:
            strength = 'No Data'
        elif total < 50:
            strength = 'Weak'
        elif total >= 80:
            strength = 'Strong'
        else:
            strength = 'Average'

        if p:
            B = prefinal_total(p.hand_marks, p.attendance_marks, p.ct_marks, p.midterm_marks)
            req = {
                'd': required_final_for_cutoff(B, 40),
                'c': required_final_for_cutoff(B, 50),
                'c_plus': required_final_for_cutoff(B, 55),
                'b': required_final_for_cutoff(B, 60),
                'b_plus': required_final_for_cutoff(B, 65),
                'a_minus': required_final_for_cutoff(B, 70),
                'a': required_final_for_cutoff(B, 75),
                'a_plus': required_final_for_cutoff(B, 80),
            }
            target = best_feasible_grade(B)
            band = midterm_band(p.midterm_marks)
            tier = urgency_tier(target['required_final'])
            explanation = explain_target(B, target['grade'], target['required_final'])
        else:
            B = None
            req = None
            target = None
            band = None
            tier = None
            explanation = None

        rows.append({
            'subject': subj,
            'performance': p,
            'strength': strength,
            'prefinal': B,
            'required': req,
            'target': target,
            'midterm_band': band,
            'urgency': tier,
            'explanation': explanation,
        })

    return render(request, 'userpanel/semester_detail.html', {
        'semester': semester,
        'rows': rows,
        'academic_ml': _build_academic_ml_for_user(request.user, semester_number=semester.number),
    })


@login_required
def enter_subject_marks(request, semester_number, subject_id):
    redirect_response = _require_student_role(request)
    if redirect_response:
        return redirect_response

    semester = Semester.objects.filter(number=semester_number).first()
    subject = Subject.objects.filter(id=subject_id, semester=semester, is_active=True).first()
    if not semester or not subject:
        return redirect('performance_overview')

    existing = StudentSubjectPerformance.objects.filter(
        user=request.user, semester=semester, subject=subject
    ).first()

    if request.method == 'POST':
        try:
            hand = int(request.POST.get('hand_marks'))
            ct = int(request.POST.get('ct_marks'))
            mid = int(request.POST.get('midterm_marks'))
            attendance_pct = float(request.POST.get('attendance_percentage'))
        except (TypeError, ValueError):
            return render(request, 'userpanel/enter_subject_marks.html', {
                'semester': semester,
                'subject': subject,
                'existing': existing,
                'attendance_pct': request.POST.get('attendance_percentage', '0'),
                'error': 'Please enter valid numeric marks.'
            })

        # Academic scheme (Total=100): Assignment(5) + Attendance(5) + CT(10) + Midterm(30) + Final(50)
        if not (0 <= hand <= 5 and 0 <= ct <= 10 and 0 <= mid <= 30 and 0 <= attendance_pct <= 100):
            return render(request, 'userpanel/enter_subject_marks.html', {
                'semester': semester,
                'subject': subject,
                'existing': existing,
                'attendance_pct': attendance_pct,
                'error': 'Marks must be within: Assignment 0-5, Attendance% 0-100, CT 0-10, Midterm 0-30.'
            })

        # Convert attendance percentage to attendance marks out of 5 (examiner-friendly scaling).
        attendance = int(round(5.0 * (attendance_pct / 100.0)))

        # Students do NOT enter final marks. We predict final marks using ML.
        cgpa_prev = float(Student.objects.filter(user=request.user).values_list('cgpa', flat=True).first() or 0.0)
        final_pred, prob_a, prob_aplus, prob_fail = predict_final_and_probs({
            'semester': semester.number,
            'subject': subject.name,
            'assignment_marks': hand,
            'attendance_percentage': attendance_pct,
            'quiz_marks': ct,
            'midterm_marks': mid,
            'previous_cgpa': cgpa_prev,
        })
        total_pred = hand + attendance + ct + mid + final_pred
        total = int(round(total_pred))
        from .academic_logic import grade_from_total
        predicted_grade = grade_from_total(total_pred)

        StudentSubjectPerformance.objects.update_or_create(
            user=request.user,
            semester=semester,
            subject=subject,
            defaults={
                'hand_marks': hand,
                'attendance_marks': attendance,
                'attendance_percentage': attendance_pct,
                'ct_marks': ct,
                'midterm_marks': mid,
                'total': total,
                'final_pred': final_pred,
                'total_pred': total_pred,
                'prob_a': prob_a,
                'prob_aplus': prob_aplus,
                'prob_fail': prob_fail,
                'predicted_grade': predicted_grade,
            }
        )

        return redirect('recovery_plan', semester_number=semester.number, subject_id=subject.id)

    return render(request, 'userpanel/enter_subject_marks.html', {
        'semester': semester,
        'subject': subject,
        'existing': existing,
        'attendance_pct': (existing.attendance_percentage if existing else 0),
    })


@login_required
def recovery_plan_view(request, semester_number, subject_id):
    redirect_response = _require_student_role(request)
    if redirect_response:
        return redirect_response

    semester = Semester.objects.filter(number=semester_number).first()
    subject = Subject.objects.filter(id=subject_id, semester=semester, is_active=True).first()
    if not semester or not subject:
        return redirect('performance_overview')

    perf = StudentSubjectPerformance.objects.filter(
        user=request.user, semester=semester, subject=subject
    ).first()
    if not perf:
        return redirect('enter_subject_marks', semester_number=semester.number, subject_id=subject.id)

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

    materials = _recommended_materials_for_subject(subject, limit=6)

    return render(request, 'userpanel/recovery_plan.html', {
        'semester': semester,
        'subject': subject,
        'perf': perf,
        'prefinal': B,
        'required': required,
        'target': target,
        'midterm_band': band,
        'urgency': tier,
        'plan_steps': steps,
        'explanation': explain_target(B, target['grade'], target['required_final']),
        'materials': materials,
    })


@login_required
def semester_plan(request, semester_number):
    """
    Student-facing: show recovery plan metrics for all subjects in a semester.
    """
    redirect_response = _require_student_role(request)
    if redirect_response:
        return redirect_response

    semester = Semester.objects.filter(number=semester_number).first()
    if not semester:
        return redirect('performance_overview')

    subjects = list(Subject.objects.filter(semester=semester, is_active=True).order_by('name'))
    perf_map = {
        p.subject_id: p
        for p in StudentSubjectPerformance.objects.filter(user=request.user, semester=semester)
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
            'pass': required_final_for_cutoff(B, 40),
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

    return render(request, 'userpanel/semester_plan.html', {
        'semester': semester,
        'rows': rows,
    })
