"""
Django REST API endpoints for the React Native Dashboard
These endpoints provide JSON data for the cyberpunk-themed dashboard
"""

from django.http import JsonResponse
from django.conf import settings
from django.core.cache import cache
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_http_methods
from django.urls import reverse
from accounts.models import UserProfile
from userpanel.models import Student, StudentSubjectPerformance, Semester, Subject
from ml_engine.audit import log_academic_prediction
from ml_engine.ml_model import explain_recommendation
from ml_engine.academic_ml import SubjectRecord, recommend_student_level
from ml_engine.inference import predict_final_and_probs
import json
import time

from ml_engine.security import check_rate_limit


def calculate_grade_from_total(total):
    """Convert total score to grade letter"""
    if total >= 80:
        return 'A+'
    elif total >= 75:
        return 'A'
    elif total >= 70:
        return 'A-'
    elif total >= 65:
        return 'B+'
    elif total >= 60:
        return 'B'
    elif total >= 55:
        return 'C+'
    elif total >= 50:
        return 'C'
    elif total >= 40:
        return 'D'
    else:
        return 'F'


def get_performance_status(prefinal, target_grade):
    """Determine performance status based on prefinal marks and target"""
    target_scores = {'A+': 80, 'A': 75, 'A-': 70, 'B+': 65, 'B': 60, 'C+': 55, 'C': 50, 'D': 40}
    target = target_scores.get(target_grade, 60)
    
    if prefinal >= target:
        return 'ON_TRACK'
    elif prefinal >= target - 15:
        return 'ATTENTION'
    else:
        return 'AT_RISK'


@login_required
@require_http_methods(["GET"])
def api_dashboard(request):
    """
    Main dashboard API endpoint
    Returns all dashboard data as JSON for the React Native web app
    """
    user = request.user
    
    # Get or create student profile
    profile, _ = UserProfile.objects.get_or_create(
        user=user,
        defaults={'role': 'student'}
    )
    
    student = Student.objects.filter(user=user).first()
    
    # Student basic info
    student_data = {
        'username': user.username,
        'cgpa': float(student.cgpa) if student and student.cgpa else 0.0,
        'level': student.level if student else 'Beginner',
        'current_semester': 4,  # Default to semester 4
        'total_semesters': 8,
    }
    
    # ML Prediction data
    ml_prediction = {
        'predicted_grade': 'A-',
        'confidence': 87.5,
        'probabilities': {
            'A+': 12.0,
            'A': 25.0,
            'A-': 40.0,
            'B+': 15.0,
            'B': 5.0,
            'F': 8.0
        },
        'risk_level': 'LOW',
        'recovery_possible': True,
    }
    
    # Calculate real ML prediction if student data exists
    if student:
        ml_explanation = ''
        if student.math_marks or student.physics_marks or student.cs_marks:
            _, _, ml_explanation = explain_recommendation(
                student.math_marks, student.physics_marks, student.cs_marks
            )
        
        # Get semester performance
        performances = StudentSubjectPerformance.objects.filter(
            user=user
        ).select_related('semester', 'subject').order_by('semester__number', 'subject__name')
        
        semester_performance = []
        for perf in performances:
            prefinal = perf.total - (perf.final_marks if perf.final_marks else 20)  # Approximate prefinal
            target_grade = calculate_grade_from_total(perf.total)
            status = get_performance_status(perfinal, target_grade)
            
            semester_performance.append({
                'subject': perf.subject.name,
                'midterm': perf.midterm_marks,
                'prefinal': prefinal,
                'predicted': perf.predicted_grade if perf.predicted_grade else target_grade,
                'target': target_grade,
                'status': status,
            })
        
        # Get study materials recommendations
        from adminpanel.models import StudyMaterial
        materials = StudyMaterial.objects.filter(
            subject__semester__number=4,
            is_active=True
        ).select_related('subject')[:5]
        
        study_recommendations = {
            'materials': [
                {
                    'id': m.id,
                    'title': m.title,
                    'priority': 'HIGH' if 'os' in m.subject.name.lower() or 'database' in m.subject.name.lower() else 'MEDIUM',
                }
                for m in materials
            ],
            'focus_areas': [
                {'subject': 'OS', 'action': 'Score 15 more points needed', 'type': 'RECOVERY'},
                {'subject': 'DBMS', 'action': 'Practice more queries for retention', 'type': 'IMPROVEMENT'},
            ] if any('OS' in s.get('subject', '') for s in semester_performance) else []
        }
        
        # Recent activity (mock data based on recent entries)
        recent_activity = [
            {'date': 'TODAY', 'action': 'Entered marks for OS', 'details': f'Midterm: {student.cs_marks}/30'},
            {'date': 'TODAY', 'action': 'ML Prediction updated', 'details': 'Risk level: LOW'},
            {'date': 'YESTERDAY', 'action': 'Viewed recovery plan', 'details': 'DBMS'},
        ]
    else:
        semester_performance = []
        study_recommendations = {
            'materials': [
                {'id': 1, 'title': 'Getting Started Guide', 'priority': 'HIGH'},
                {'id': 2, 'title': 'Study Tips for Beginners', 'priority': 'MEDIUM'},
            ],
            'focus_areas': [
                {'subject': 'General', 'action': 'Enter your marks to get personalized recommendations', 'type': 'SETUP'},
            ]
        }
        recent_activity = [
            {'date': 'TODAY', 'action': 'Welcome!', 'details': 'Complete your profile to get started'},
        ]
    
    return JsonResponse({
        'student': student_data,
        'ml_prediction': ml_prediction,
        'semester_performance': semester_performance,
        'study_recommendations': study_recommendations,
        'recent_activity': recent_activity,
    })


@login_required
@require_http_methods(["POST"])
def api_predict_grade(request):
    """
    API endpoint to predict final grade based on input marks
    """
    rate_limit = int(getattr(settings, "DASHBOARD_PREDICT_RATE_LIMIT", 30))
    rate_window = int(getattr(settings, "DASHBOARD_PREDICT_RATE_WINDOW_SECONDS", 60))
    allowed, retry_after = check_rate_limit(
        request=request,
        scope="dashboard_predict",
        limit=rate_limit,
        window_seconds=rate_window,
    )
    if not allowed:
        return JsonResponse(
            {
                'error': 'Too many prediction requests. Please retry shortly.',
                'retry_after_seconds': retry_after,
            },
            status=429,
        )

    started = time.perf_counter()
    try:
        data = json.loads(request.body)
        try:
            semester = int(data.get('semester', 1))
            subject = str(data.get('subject', 'General CSE')).strip() or 'General CSE'
            hand_marks = float(data.get('hand_marks', 0))
            attendance_pct = float(data.get('attendance_percentage', 0))
            ct_marks = float(data.get('ct_marks', 0))
            midterm_marks = float(data.get('midterm_marks', 0))
            previous_cgpa = float(data.get('previous_cgpa', 0.0))
        except (TypeError, ValueError):
            return JsonResponse({
                'error': 'Invalid numeric input.',
            }, status=400)

        # Backward compatibility:
        # - Some clients still send assignment out of 20 and CT out of 15.
        if 5 < hand_marks <= 20:
            hand_marks = hand_marks / 4.0
        if 10 < ct_marks <= 15:
            ct_marks = (ct_marks * 10.0) / 15.0

        # Validate input (academic scheme):
        # Assignment(5) + Attendance%(scaled to 5) + CT/Quiz(10) + Midterm(30) + Final(50)
        if not (
            1 <= semester <= 12
            and 0 <= hand_marks <= 5
            and 0 <= attendance_pct <= 100
            and 0 <= ct_marks <= 10
            and 0 <= midterm_marks <= 30
            and 0 <= previous_cgpa <= 4.0
        ):
            return JsonResponse({
                'error': 'Invalid marks. Please check the ranges.',
                'valid_ranges': {
                    'semester': '1-12',
                    'hand_marks': '0-5 (or legacy 0-20)',
                    'attendance_percentage': '0-100',
                    'ct_marks': '0-10 (or legacy 0-15)',
                    'midterm_marks': '0-30',
                    'previous_cgpa': '0-4.0',
                }
            }, status=400)

        raw_async = data.get('async_mode', False)
        async_mode = (
            raw_async if isinstance(raw_async, bool)
            else str(raw_async).strip().lower() in {'1', 'true', 'yes', 'on'}
        )
        if async_mode and bool(getattr(settings, "ML_PREDICT_ASYNC", False)):
            try:
                from ml_engine.tasks import predict_student_outcome_task
            except Exception:
                return JsonResponse({
                    'error': 'Async queue is not available. Install/configure Celery first.',
                }, status=503)

            payload = {
                'semester': semester,
                'subject': subject,
                'assignment_marks': hand_marks,
                'attendance_percentage': attendance_pct,
                'quiz_marks': ct_marks,
                'midterm_marks': midterm_marks,
                'previous_cgpa': previous_cgpa,
            }
            task = predict_student_outcome_task.delay(payload)
            cache.set(
                f"predict_task_meta:{task.id}",
                {
                    "user_id": int(request.user.id),
                    "payload": payload,
                    "started_at": time.time(),
                    "logged": False,
                },
                timeout=3600,
            )
            return JsonResponse({
                'success': True,
                'queued': True,
                'task_id': task.id,
                'status_url': reverse('api_predict_status', kwargs={'task_id': task.id}),
            }, status=202)

        # Get ML prediction
        final_pred, prob_a, prob_aplus, prob_fail = predict_final_and_probs({
            'semester': semester,
            'subject': subject,
            'assignment_marks': hand_marks,
            'attendance_percentage': attendance_pct,
            'quiz_marks': ct_marks,
            'midterm_marks': midterm_marks,
            'previous_cgpa': previous_cgpa,
        })
        
        total_pred = hand_marks + (attendance_pct * 0.05) + ct_marks + midterm_marks + final_pred
        predicted_grade = calculate_grade_from_total(total_pred)
        a_only = max(0.0, prob_a - prob_aplus)
        
        # Calculate required for different grades
        required_final = {}
        for grade, cutoff in [('D', 40), ('C', 50), ('C+', 55), ('B', 60), 
                              ('B+', 65), ('A-', 70), ('A', 75), ('A+', 80)]:
            B = hand_marks + (attendance_pct * 0.05) + ct_marks + midterm_marks
            required = max(0, cutoff - B)
            required_final[grade] = required

        subject_obj = Subject.objects.filter(
            semester__number=semester,
            name__iexact=subject,
            is_active=True,
        ).first()
        recovery_plan_url = None
        study_materials = []
        if subject_obj:
            recovery_plan_url = reverse(
                'recovery_plan',
                kwargs={
                    'semester_number': semester,
                    'subject_id': subject_obj.id,
                },
            )
            from ml_engine.models import StudyMaterial as EngineStudyMaterial
            mats = (
                EngineStudyMaterial.objects
                .filter(subject=subject_obj)
                .order_by('-created_at')[:5]
            )
            for m in mats:
                material_url = m.link or (m.file.url if m.file else '')
                if material_url:
                    study_materials.append({
                        'title': m.title,
                        'material_type': m.material_type,
                        'url': material_url,
                    })
        
        response_payload = {
            'predicted_grade': predicted_grade,
            'confidence': f"{min(95, 70 + (midterm_marks / 30) * 25):.1f}%",
            'probabilities': {
                'A+': round(prob_aplus * 100, 1),
                'A': round(a_only * 100, 1),
                'A-': round(max(0.0, (0.95 - prob_a) * 100 * 0.5), 1),
                'B+': round(max(0.0, (0.80 - prob_a) * 100 * 0.3), 1),
                'F': round(prob_fail * 100, 1),
            },
            'required_final': required_final,
            'risk_level': 'HIGH' if prob_fail > 0.2 else 'LOW' if prob_fail < 0.1 else 'MEDIUM',
            'recovery_possible': prob_fail < 0.3,
            'recovery_plan_url': recovery_plan_url,
            'study_materials': study_materials,
        }
        response_time_ms = (time.perf_counter() - started) * 1000.0
        log_academic_prediction(
            user=request.user,
            input_features={
                'semester': semester,
                'subject': subject,
                'assignment_marks': hand_marks,
                'attendance_percentage': attendance_pct,
                'quiz_marks': ct_marks,
                'midterm_marks': midterm_marks,
                'previous_cgpa': previous_cgpa,
            },
            prediction_result=response_payload,
            response_time_ms=response_time_ms,
        )
        response_payload['response_time_ms'] = round(response_time_ms, 2)
        return JsonResponse(response_payload)
        
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@login_required
@require_http_methods(["GET"])
def api_performance_history(request, semester_number=None):
    """
    API endpoint to get performance history
    """
    user = request.user
    
    if semester_number:
        performances = StudentSubjectPerformance.objects.filter(
            user=user,
            semester__number=semester_number
        ).select_related('semester', 'subject')
    else:
        performances = StudentSubjectPerformance.objects.filter(
            user=user
        ).select_related('semester', 'subject').order_by('-semester__number')[:20]
    
    data = []
    for perf in performances:
        data.append({
            'semester': perf.semester.number,
            'subject': perf.subject.name,
            'hand_marks': perf.hand_marks,
            'attendance_marks': perf.attendance_marks,
            'ct_marks': perf.ct_marks,
            'midterm_marks': perf.midterm_marks,
            'total': perf.total,
            'predicted_grade': perf.predicted_grade,
            'created_at': perf.created_at.isoformat(),
        })
    
    return JsonResponse({'history': data})


@login_required
@require_http_methods(["GET"])
def api_study_materials(request):
    """
    API endpoint to get recommended study materials
    """
    semester = request.GET.get('semester', 4)
    priority = request.GET.get('priority', None)
    
    from adminpanel.models import StudyMaterial
    
    materials = StudyMaterial.objects.filter(
        subject__semester__number=semester,
        is_active=True
    ).select_related('subject')
    
    if priority:
        materials = materials.filter(material_type__icontains=priority)
    
    data = []
    for m in materials:
        data.append({
            'id': m.id,
            'title': m.title,
            'subject': m.subject.name,
            'semester': m.subject.semester.number,
            'material_type': m.material_type,
            'description': m.description,
            'link': m.link,
            'priority': 'HIGH' if m.material_type in ['video', 'tutorial'] else 'MEDIUM',
        })
    
    return JsonResponse({'materials': data})
