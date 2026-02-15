"""
Django REST Framework API Views for ML Engine

Provides REST endpoints for all ML predictions and analyses.
Requires authentication via JWT or session.
"""

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from django.utils import timezone

from ml_engine.unified_inference import MLInference, quick_predict
from ml_engine.burnout_prediction import BurnoutFeatures
from ml_engine.study_duration import predict_optimal_duration
from ml_engine.learning_style import LearningStyleInput, detect_learning_style
from ml_engine.anxiety_detection import AnxietyFeatures, predict_anxiety
from ml_engine.spaced_repetition import (
    ReviewItem, ReviewQuality, review_item, generate_daily_schedule
)
from ml_engine.knowledge_gaps import KnowledgeGapInput, analyze_knowledge_gaps
from ml_engine.adaptive_difficulty import AdaptiveDifficultyEngine, StudentResponse


# ============ Academic Analysis ============

class AcademicAnalysisView(APIView):
    """
    POST /api/ml/academic/
    
    Analyze academic performance and predict outcomes.
    
    Request body:
    {
        "assignment_marks": float,      # 0-5
        "attendance_percentage": float,  # 0-100
        "quiz_marks": float,           # 0-10
        "midterm_marks": float,         # 0-30
        "previous_cgpa": float          # 0-4
    }
    """
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        try:
            data = request.data
            engine = MLInference(student_id=request.user.username)
            
            result = engine.analyze_academic_performance(
                assignment_marks=float(data.get('assignment_marks', 3.0)),
                attendance_percentage=float(data.get('attendance_percentage', 80.0)),
                quiz_marks=float(data.get('quiz_marks', 7.0)),
                midterm_marks=float(data.get('midterm_marks', 20.0)),
                previous_cgpa=float(data.get('previous_cgpa', 3.0)),
            )
            
            return Response({
                'success': True,
                'data': {
                    'performance_cluster': result.performance_cluster,
                    'study_level': result.study_level,
                    'predicted_final': result.predicted_final,
                    'prob_grade_a': result.prob_grade_a,
                    'prob_grade_aplus': result.prob_grade_aplus,
                    'prob_fail': result.prob_fail,
                    'recommendations': result.recommendations,
                    'explanation': result.explanation,
                }
            })
        except Exception as e:
            return Response({
                'success': False,
                'error': str(e)
            }, status=status.HTTP_400_BAD_REQUEST)


# ============ Burnout Analysis ============

class BurnoutAnalysisView(APIView):
    """
    POST /api/ml/burnout/
    
    Analyze burnout risk.
    
    Request body:
    {
        "avg_daily_study_hours": float,
        "consecutive_study_days": int,
        "recent_grade_change": float,
        "sleep_hours_per_day": float,
        "assignment_completion_rate": float,  # optional
        "days_until_exam": int               # optional
    }
    """
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        try:
            data = request.data
            features = BurnoutFeatures(
                avg_daily_study_hours=float(data['avg_daily_study_hours']),
                study_session_count=int(data.get('study_session_count', 7)),
                avg_session_duration_minutes=float(data.get('avg_session_duration_minutes', 60)),
                consecutive_study_days=int(data['consecutive_study_days']),
                recent_grade_change=float(data['recent_grade_change']),
                sleep_hours_per_day=float(data['sleep_hours_per_day']),
                assignment_completion_rate=float(data.get('assignment_completion_rate', 0.8)),
                attendance_rate=float(data.get('attendance_rate', 0.9)),
                activity_level=float(data.get('activity_level', 0.7)),
                days_until_exam=int(data.get('days_until_exam', 14)),
            )
            
            result = quick_predict('burnout', **data)
            
            return Response({
                'success': True,
                'data': {
                    'risk_level': result.risk_level,
                    'risk_score': result.risk_score,
                    'confidence': result.confidence,
                    'top_factors': result.top_factors,
                    'recommendations': result.recommendations,
                }
            })
        except Exception as e:
            return Response({
                'success': False,
                'error': str(e)
            }, status=status.HTTP_400_BAD_REQUEST)


# ============ Study Duration ============

class DurationOptimizationView(APIView):
    """
    POST /api/ml/duration/
    
    Get optimal study duration.
    
    Request body:
    {
        "avg_session_length": float,
        "avg_completion_rate": float,
        "avg_retention": float,
        "preferred_time": str,
        "difficulty": float
    }
    """
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        try:
            data = request.data
            result = quick_predict('duration', **data)
            
            return Response({
                'success': True,
                'data': {
                    'optimal_minutes': result.optimal_duration_minutes,
                    'min_minutes': result.min_duration_minutes,
                    'max_minutes': result.max_duration_minutes,
                    'break_interval': result.break_interval_minutes,
                    'expected_retention': result.expected_retention,
                    'confidence': result.confidence,
                    'reasoning': result.reasoning,
                    'productivity_curve': result.productivity_curve,
                }
            })
        except Exception as e:
            return Response({
                'success': False,
                'error': str(e)
            }, status=status.HTTP_400_BAD_REQUEST)


# ============ Learning Style ============

class LearningStyleView(APIView):
    """
    POST /api/ml/learning-style/
    
    Detect learning style (VARK).
    
    Request body:
    {
        "visual_score": float,
        "auditory_score": float,
        "reading_score": float,
        "kinesthetic_score": float
    }
    """
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        try:
            data = request.data
            result = quick_predict('learning_style', **data)
            
            return Response({
                'success': True,
                'data': {
                    'primary_style': result.primary_style,
                    'scores': result.scores,
                    'confidence': result.confidence,
                    'recommendations': result.recommendations,
                }
            })
        except Exception as e:
            return Response({
                'success': False,
                'error': str(e)
            }, status=status.HTTP_400_BAD_REQUEST)


# ============ Anxiety Analysis ============

class AnxietyAnalysisView(APIView):
    """
    POST /api/ml/anxiety/
    
    Analyze exam anxiety levels.
    
    Request body:
    {
        "exam_frequency_per_month": float,
        "avg_preparation_hours": float,
        "sleep_night_before": float,
        "caffeine_intake": float,
        "previous_exam_score": float,
        "target_score": float,
        "time_until_exam_hours": float
    }
    """
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        try:
            data = request.data
            result = quick_predict('anxiety', **data)
            
            return Response({
                'success': True,
                'data': {
                    'anxiety_level': result.anxiety_level,
                    'anxiety_score': result.anxiety_score,
                    'confidence': result.confidence,
                    'contributing_factors': result.contributing_factors,
                    'recommendations': result.recommendations,
                }
            })
        except Exception as e:
            return Response({
                'success': False,
                'error': str(e)
            }, status=status.HTTP_400_BAD_REQUEST)


# ============ Spaced Repetition ============

class SpacedRepetitionView(APIView):
    """
    POST /api/ml/spaced-repetition/
    
    Manage spaced repetition reviews.
    
    Request body:
    {
        "action": "schedule" | "review",
        "items": [...],
        "reviews": [{"item_id": str, "quality": int}],
        "max_items": int,
        "max_time_minutes": int
    }
    """
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        try:
            data = request.data
            action = data.get('action', 'schedule')
            
            engine = MLInference(student_id=request.user.username)
            
            if action == 'schedule':
                result = engine.manage_spaced_repetition(
                    items=data.get('items'),
                    target_date=timezone.now(),
                    max_items=int(data.get('max_items', 20)),
                    max_time_minutes=int(data.get('max_time_minutes', 30)),
                )
                
                return Response({
                    'success': True,
                    'data': {
                        'items_due': result.items_due,
                        'daily_schedule': result.daily_schedule,
                        'retention_rate': result.retention_rate,
                        'mastered_items': result.mastered_items,
                    }
                })
            
            elif action == 'review':
                reviews = data.get('reviews', [])
                result = engine.manage_spaced_repetition(
                    item_reviews=[(r['item_id'], r['quality']) for r in reviews]
                )
                
                return Response({
                    'success': True,
                    'data': {
                        'processed_reviews': len(reviews),
                        'retention_rate': result.retention_rate,
                    }
                })
            
            else:
                return Response({
                    'success': False,
                    'error': f"Unknown action: {action}"
                }, status=status.HTTP_400_BAD_REQUEST)
                
        except Exception as e:
            return Response({
                'success': False,
                'error': str(e)
            }, status=status.HTTP_400_BAD_REQUEST)


# ============ Knowledge Gaps ============

class KnowledgeGapsView(APIView):
    """
    POST /api/ml/knowledge-gaps/
    
    Analyze knowledge gaps.
    
    Request body:
    {
        "current_marks": {"topic1": 70, "topic2": 50},
        "target_marks": {"topic1": 80, "topic2": 80},
        "syllabus_topics": ["topic1", "topic2", "topic3"],
        "time_available_hours": float
    }
    """
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        try:
            data = request.data
            engine = MLInference(student_id=request.user.username)
            
            result = engine.analyze_knowledge_gaps(
                current_marks=data.get('current_marks', {}),
                target_marks=data.get('target_marks', {}),
                syllabus_topics=data.get('syllabus_topics', []),
                recent_quiz_scores=data.get('recent_quiz_scores', {}),
                time_available_hours=float(data.get('time_available_hours', 10.0)),
            )
            
            return Response({
                'success': True,
                'data': {
                    'gap_count': result.gap_count,
                    'gaps': result.gaps,
                    'prioritized_learning': result.prioritized_learning,
                    'estimated_time_hours': result.estimated_time_hours,
                    'focus_areas': result.focus_areas,
                }
            })
        except Exception as e:
            return Response({
                'success': False,
                'error': str(e)
            }, status=status.HTTP_400_BAD_REQUEST)


# ============ Adaptive Difficulty ============

class AdaptiveDifficultyView(APIView):
    """
    POST /api/ml/adaptive-difficulty/
    
    Get adaptive difficulty recommendation.
    
    Request body:
    {
        "action": "recommend" | "record",
        "subject": str,
        "recent_success_rate": float,
        "total_problems": int,
        // For record action:
        "problem_id": str,
        "selected_answer": str,
        "is_correct": bool,
        "response_time": float,
        "difficulty": float
    }
    """
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        try:
            data = request.data
            action = data.get('action', 'recommend')
            subject = data.get('subject', 'General')
            
            engine = MLInference(student_id=request.user.username)
            
            if action == 'recommend':
                result = engine.get_adaptive_difficulty(
                    student_id=request.user.username,
                    subject=subject,
                    recent_success_rate=float(data.get('recent_success_rate', 0.5)),
                    total_problems=int(data.get('total_problems', 0)),
                )
                
                return Response({
                    'success': True,
                    'data': {
                        'recommended_difficulty': result.recommended_difficulty,
                        'confidence': result.confidence,
                        'exploration_rate': result.exploration_rate,
                        'reason': result.reason,
                        'alternatives': result.alternatives,
                        'estimated_success': result.estimated_success,
                    }
                })
            
            elif action == 'record':
                engine.record_response_adaptive(
                    student_id=request.user.username,
                    subject=subject,
                    problem_id=data['problem_id'],
                    selected_answer=data['selected_answer'],
                    is_correct=data['is_correct'],
                    response_time=float(data['response_time']),
                    difficulty=float(data['difficulty']),
                    confidence=data.get('confidence'),
                )
                
                return Response({
                    'success': True,
                    'message': 'Response recorded successfully'
                })
            
            else:
                return Response({
                    'success': False,
                    'error': f"Unknown action: {action}"
                }, status=status.HTTP_400_BAD_REQUEST)
                
        except Exception as e:
            return Response({
                'success': False,
                'error': str(e)
            }, status=status.HTTP_400_BAD_REQUEST)


# ============ Full Analysis ============

class FullAnalysisView(APIView):
    """
    POST /api/ml/full-analysis/
    
    Run comprehensive analysis across all ML modules.
    
    Request body includes all parameters from individual endpoints.
    """
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        try:
            data = request.data
            engine = MLInference(student_id=request.user.username)
            
            result = engine.run_full_analysis(
                student_id=request.user.username,
                # Academic
                assignment_marks=float(data.get('assignment_marks', 3.0)),
                attendance_percentage=float(data.get('attendance_percentage', 80.0)),
                quiz_marks=float(data.get('quiz_marks', 7.0)),
                midterm_marks=float(data.get('midterm_marks', 20.0)),
                previous_cgpa=float(data.get('previous_cgpa', 3.0)),
                # Burnout
                avg_daily_study_hours=float(data.get('avg_daily_study_hours', 5.0)),
                consecutive_study_days=int(data.get('consecutive_study_days', 3)),
                recent_grade_change=float(data.get('recent_grade_change', 0.0)),
                sleep_hours_per_day=float(data.get('sleep_hours_per_day', 7.0)),
                assignment_completion_rate=float(data.get('assignment_completion_rate', 0.8)),
                days_until_exam=int(data.get('days_until_exam', 14)),
                # Duration
                preferred_time=data.get('preferred_time', 'evening'),
                avg_session_length=float(data.get('avg_session_length', 45.0)),
                avg_completion_rate=float(data.get('avg_completion_rate', 0.8)),
                avg_retention=float(data.get('avg_retention', 0.7)),
                longest_streak=int(data.get('longest_streak', 7)),
                performance_trend=float(data.get('performance_trend', 0.0)),
                difficulty=float(data.get('difficulty', 5.0)),
                total_study_time=int(data.get('total_study_time', 120)),
                # Learning Style
                visual_score=float(data.get('visual_score', 0.0)),
                auditory_score=float(data.get('auditory_score', 0.0)),
                reading_score=float(data.get('reading_score', 0.0)),
                kinesthetic_score=float(data.get('kinesthetic_score', 0.0)),
                # Anxiety
                exam_frequency=float(data.get('exam_frequency', 1.0)),
                prep_hours=float(data.get('prep_hours', 5.0)),
                sleep_before=float(data.get('sleep_before', 7.0)),
                caffeine=float(data.get('caffeine', 1.0)),
                activity_hours=float(data.get('activity_hours', 1.0)),
                social_support=float(data.get('social_support', 0.7)),
                prev_exam_score=float(data.get('prev_exam_score', 75.0)),
                target_score=float(data.get('target_score', 80.0)),
                time_until_exam=float(data.get('time_until_exam', 24.0)),
            )
            
            return Response({
                'success': True,
                'data': {
                    'timestamp': result.timestamp.isoformat(),
                    'student_id': result.student_id,
                    'academic': result.academic.__dict__ if result.academic else None,
                    'burnout': result.burnout.__dict__ if result.burnout else None,
                    'duration': result.duration.__dict__ if result.duration else None,
                    'learning_style': result.learning_style.__dict__ if result.learning_style else None,
                    'anxiety': result.anxiety.__dict__ if result.anxiety else None,
                    'priority_recommendations': result.priority_recommendations,
                    'overall_health_score': result.overall_health_score,
                }
            })
        except Exception as e:
            return Response({
                'success': False,
                'error': str(e)
            }, status=status.HTTP_400_BAD_REQUEST)


# ============ Health Check ============

class HealthCheckView(APIView):
    """GET /api/ml/health/ - Health check endpoint."""
    permission_classes = [AllowAny]
    
    def get(self, request):
        return Response({
            'status': 'healthy',
            'service': 'ML Engine API',
            'timestamp': timezone.now().isoformat(),
        })
