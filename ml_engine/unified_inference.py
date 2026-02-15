"""
Unified ML Inference API

Aggregates all ML modules into a single, easy-to-use interface for Django views.

Modules integrated:
- Academic ML (grade prediction, clustering)
- Burnout Prediction (risk detection)
- Study Duration Optimization
- Learning Style Detection (VARK)
- Anxiety Detection
- Spaced Repetition (SM-2)
- Knowledge Gap Analysis
- Content Recommender
- Trend Prediction
- Adaptive Difficulty (MAB)
- Drift Detection

Usage:
    from ml_engine.unified_inference import MLInference
    
    engine = MLInference(student_id="student123")
    results = engine.run_full_analysis(
        academic_features={...},
        burnout_features={...},
        study_data=[...],
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import json


# ============ Academic ML ============
from ml_engine.academic_ml import (
    SubjectRecord,
    CohortRecord,
    recommend_student_level,
    get_performance_cluster,
    predict_final_grade_range,
)


# ============ Burnout Prediction ============
from ml_engine.burnout_prediction import (
    BurnoutFeatures,
    BurnoutPrediction,
    predict_burnout,
)


# ============ Study Duration ============
from ml_engine.study_duration import (
    StudySessionData,
    StudyDurationFeatures,
    predict_optimal_duration,
    generate_session_schedule,
)


# ============ Learning Style ============
from ml_engine.learning_style import (
    LearningStyleInput,
    LearningStylePrediction,
    detect_learning_style,
    get_style_recommendations,
)


# ============ Anxiety Detection ============
from ml_engine.anxiety_detection import (
    AnxietyFeatures,
    AnxietyPrediction,
    predict_anxiety,
)


# ============ Spaced Repetition ============
from ml_engine.spaced_repetition import (
    ReviewItem,
    ReviewQuality,
    ReviewResult,
    create_review_item,
    review_item as sm2_review_item,
    generate_daily_schedule,
    calculate_retention_stats,
)


# ============ Knowledge Gaps ============
from ml_engine.knowledge_gaps import (
    ConceptMastery,
    KnowledgeGapInput,
    KnowledgeGapResult,
    analyze_knowledge_gaps,
    prioritize_gaps,
)


# ============ Content Recommender ============
from ml_engine.content_recommender import (
    ContentItem,
    StudentInteraction,
    ContentRecommendation,
    recommend_content,
    get_popular_content,
)


# ============ Trend Prediction ============
from ml_engine.trend_prediction import (
    PerformancePoint,
    TrendPrediction,
    predict_performance_trend,
)


# ============ Adaptive Difficulty ============
from ml_engine.adaptive_difficulty import (
    Problem,
    StudentResponse,
    AdaptiveSession,
    AdaptiveDifficultyEngine,
)


# ============ Drift Detection ============
from ml_engine.drift_detection import (
    DataStatistics,
    ModelPerformance,
    DriftResult,
    detect_data_drift,
)


# ============ Result Dataclasses ============

@dataclass
class AcademicAnalysis:
    """Academic performance analysis results."""
    performance_cluster: str
    study_level: str
    predicted_final: float
    prob_grade_a: float
    prob_grade_aplus: float
    prob_fail: float
    features_used: Dict[str, float]
    explanation: str
    recommendations: List[str]


@dataclass
class BurnoutAnalysis:
    """Burnout risk analysis results."""
    risk_level: str  # Low, Medium, High
    risk_score: float
    confidence: float
    top_factors: List[Tuple[str, float]]
    recommendations: List[str]


@dataclass
class DurationAnalysis:
    """Study duration optimization results."""
    optimal_minutes: int
    min_minutes: int
    max_minutes: int
    break_interval: int
    expected_retention: float
    confidence: float
    reasoning: List[str]
    productivity_curve: List[Tuple[int, float]]
    session_schedule: List[Dict]


@dataclass
class LearningStyleAnalysis:
    """Learning style detection results."""
    primary_style: str
    scores: Dict[str, float]
    confidence: float
    recommendations: List[str]


@dataclass
class AnxietyAnalysis:
    """Anxiety level analysis results."""
    anxiety_level: str  # Low, Moderate, High
    anxiety_score: float
    confidence: float
    contributing_factors: List[Tuple[str, float]]
    recommendations: List[str]


@dataclass
class SpacedRepetitionAnalysis:
    """Spaced repetition analysis results."""
    items_due: int
    daily_schedule: Optional[Dict]
    retention_stats: Dict
    retention_rate: float
    mastered_items: int


@dataclass
class KnowledgeGapAnalysis:
    """Knowledge gap analysis results."""
    gap_count: int
    gaps: List[Dict]
    prioritized_learning: List[Dict]
    estimated_time_hours: float
    focus_areas: List[str]


@dataclass
class ContentAnalysis:
    """Content recommendation results."""
    recommendations: List[Dict]
    popular_content: List[Dict]
    based_on_interactions: int


@dataclass
class TrendAnalysis:
    """Performance trend analysis results."""
    trend: str  # Improving, Stable, Declining
    change_percent: float
    confidence: float
    predictions: List[Dict]


@dataclass
class AdaptiveDifficultyAnalysis:
    """Adaptive difficulty results."""
    recommended_difficulty: float
    confidence: float
    exploration_rate: float
    reason: str
    alternatives: List[Tuple[float, float]]
    estimated_success: float


@dataclass
class DriftAnalysis:
    """Data drift detection results."""
    has_drift: bool
    drift_score: float
    drift_features: List[str]
    recommendation: str
    model_performance: Optional[Dict]


@dataclass
class UnifiedMLResults:
    """Combined results from all ML modules."""
    timestamp: datetime
    student_id: str
    
    # Core analyses
    academic: Optional[AcademicAnalysis]
    burnout: Optional[BurnoutAnalysis]
    duration: Optional[DurationAnalysis]
    learning_style: Optional[LearningStyleAnalysis]
    anxiety: Optional[AnxietyAnalysis]
    spaced_repetition: Optional[SpacedRepetitionAnalysis]
    knowledge_gaps: Optional[KnowledgeGapAnalysis]
    content: Optional[ContentAnalysis]
    trends: Optional[TrendAnalysis]
    adaptive_difficulty: Optional[AdaptiveDifficultyAnalysis]
    drift: Optional[DriftAnalysis]
    
    # Summary
    priority_recommendations: List[str]
    overall_health_score: float
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "student_id": self.student_id,
            "academic": self.academic.__dict__ if self.academic else None,
            "burnout": self.burnout.__dict__ if self.burnout else None,
            "duration": self.duration.__dict__ if self.duration else None,
            "learning_style": self.learning_style.__dict__ if self.learning_style else None,
            "anxiety": self.anxiety.__dict__ if self.anxiety else None,
            "spaced_repetition": self.spaced_repetition.__dict__ if self.spaced_repetition else None,
            "knowledge_gaps": self.knowledge_gaps.__dict__ if self.knowledge_gaps else None,
            "content": self.content.__dict__ if self.content else None,
            "trends": self.trends.__dict__ if self.trends else None,
            "adaptive_difficulty": self.adaptive_difficulty.__dict__ if self.adaptive_difficulty else None,
            "drift": self.drift.__dict__ if self.drift else None,
            "priority_recommendations": self.priority_recommendations,
            "overall_health_score": self.overall_health_score,
        }


class MLInference:
    """
    Unified ML Inference Engine.
    
    Provides a single interface for all ML predictions.
    Maintains state for session-based predictions.
    """
    
    def __init__(self, student_id: str = None):
        self.student_id = student_id
        self.timestamp = datetime.now()
        
        # Initialize engines that maintain state
        self.adaptive_engine = AdaptiveDifficultyEngine()
        
        # Cache for results
        self._results_cache: Dict[str, Any] = {}
    
    def analyze_academic_performance(
        self,
        assignment_marks: float,
        attendance_percentage: float,
        quiz_marks: float,
        midterm_marks: float,
        previous_cgpa: float,
        student_records: List[Dict] = None,
        cohort_records: Dict[str, List[Dict]] = None,
    ) -> AcademicAnalysis:
        """
        Analyze academic performance and predict outcomes.
        """
        # Build cohort records if provided
        cohort = {}
        if cohort_records:
            for user_id, records in cohort_records.items():
                cohort[user_id] = [
                    SubjectRecord(**r) for r in records
                ]
        
        # Build student records
        student_recs = []
        if student_records:
            student_recs = [SubjectRecord(**r) for r in student_records]
        
        # Get ML recommendation
        rec = recommend_student_level(
            student_records=student_recs,
            cohort_records=cohort if cohort else None,
        )
        
        # Predict final grade (simplified)
        predicted_final = min(50.0, max(0.0, 
            assignment_marks * 2 +  # Scaled to 10
            quiz_marks * 2 +         # Scaled to 20
            midterm_marks * 1 +     # Scaled to 30
            10                       # Baseline
        ))
        
        # Calculate grade probabilities (simplified)
        prob_a = max(0.0, min(1.0, (predicted_final + 10) / 60))
        prob_aplus = max(0.0, min(1.0, (predicted_final + 15) / 65))
        prob_fail = max(0.0, min(1.0, (40 - predicted_final) / 40))
        
        recommendations = [
            f"📚 Performance level: {rec.study_level}",
            f"📊 Cluster: {rec.performance_cluster}",
        ]
        if rec.study_level == "Beginner":
            recommendations.extend([
                "Start with foundational concepts",
                "Use visual aids and practice problems",
            ])
        elif rec.study_level == "Intermediate":
            recommendations.extend([
                "Focus on weak areas identified",
                "Increase practice difficulty gradually",
            ])
        else:
            recommendations.extend([
                "Challenge yourself with advanced problems",
                "Help peers who are struggling",
            ])
        
        return AcademicAnalysis(
            performance_cluster=rec.performance_cluster,
            study_level=rec.study_level,
            predicted_final=round(predicted_final, 1),
            prob_grade_a=round(prob_a, 2),
            prob_grade_aplus=round(prob_aplus, 2),
            prob_fail=round(prob_fail, 2),
            features_used={
                "assignment_marks": assignment_marks,
                "attendance": attendance_percentage,
                "quiz_marks": quiz_marks,
                "midterm_marks": midterm_marks,
                "previous_cgpa": previous_cgpa,
            },
            explanation=rec.explanation,
            recommendations=recommendations,
        )
    
    def analyze_burnout_risk(
        self,
        avg_daily_study_hours: float,
        consecutive_study_days: int,
        recent_grade_change: float,
        sleep_hours_per_day: float,
        assignment_completion_rate: float = 0.8,
        study_session_count: int = 7,
        avg_session_duration: float = 60,
        attendance_rate: float = 0.9,
        activity_level: float = 0.7,
        days_until_exam: int = 14,
    ) -> BurnoutAnalysis:
        """
        Analyze burnout risk and generate recommendations.
        """
        features = BurnoutFeatures(
            avg_daily_study_hours=avg_daily_study_hours,
            study_session_count=study_session_count,
            avg_session_duration_minutes=avg_session_duration,
            consecutive_study_days=consecutive_study_days,
            recent_grade_change=recent_grade_change,
            sleep_hours_per_day=sleep_hours_per_day,
            assignment_completion_rate=assignment_completion_rate,
            attendance_rate=attendance_rate,
            activity_level=activity_level,
            days_until_exam=days_until_exam,
        )
        
        result = predict_burnout(features)
        
        return BurnoutAnalysis(
            risk_level=result.risk_level,
            risk_score=result.risk_score,
            confidence=result.confidence,
            top_factors=result.top_factors,
            recommendations=result.recommendations,
        )
    
    def optimize_study_duration(
        self,
        avg_session_length: float = 45.0,
        avg_completion_rate: float = 0.8,
        avg_quiz_score: float = 75.0,
        avg_retention: float = 0.7,
        preferred_time: str = "evening",
        longest_streak: int = 7,
        performance_trend: float = 0.0,
        difficulty: float = 5.0,
        total_study_time: int = 120,
    ) -> DurationAnalysis:
        """
        Calculate optimal study session duration.
        """
        result = predict_optimal_duration(
            avg_session_length=avg_session_length,
            avg_completion_rate=avg_completion_rate,
            avg_quiz_score=avg_quiz_score,
            avg_retention=avg_retention,
            preferred_time=preferred_time,
            longest_streak=longest_streak,
            performance_trend=performance_trend,
            difficulty=difficulty,
        )
        
        schedule = generate_session_schedule(
            total_study_time_minutes=total_study_time,
            optimal_duration=result.optimal_duration_minutes,
            break_interval=result.break_interval_minutes,
        )
        
        return DurationAnalysis(
            optimal_minutes=result.optimal_duration_minutes,
            min_minutes=result.min_duration_minutes,
            max_minutes=result.max_duration_minutes,
            break_interval=result.break_interval_minutes,
            expected_retention=result.expected_retention,
            confidence=result.confidence,
            reasoning=result.reasoning,
            productivity_curve=result.productivity_curve,
            session_schedule=schedule,
        )
    
    def detect_learning_style(
        self,
        visual_score: float = 0.0,
        auditory_score: float = 0.0,
        reading_score: float = 0.0,
        kinesthetic_score: float = 0.0,
        study_habits: Dict[str, float] = None,
    ) -> LearningStyleAnalysis:
        """
        Detect preferred learning style using VARK model.
        """
        input_data = LearningStyleInput(
            visual_score=visual_score,
            auditory_score=auditory_score,
            reading_score=reading_score,
            kinesthetic_score=kinesthetic_score,
            study_habits=study_habits or {},
        )
        
        result = detect_learning_style(input_data)
        recommendations = get_style_recommendations(result.primary_style)
        
        return LearningStyleAnalysis(
            primary_style=result.primary_style,
            scores=result.scores,
            confidence=result.confidence,
            recommendations=recommendations,
        )
    
    def analyze_anxiety(
        self,
        exam_frequency_per_month: float = 1.0,
        avg_preparation_hours: float = 5.0,
        sleep_night_before: float = 7.0,
        caffeine_intake: float = 1.0,
        physical_activity_hours: float = 1.0,
        social_support_score: float = 0.7,
        previous_exam_score: float = 75.0,
        target_score: float = 80.0,
        time_until_exam_hours: float = 24.0,
        assignment_completion_rate: float = 0.85,
    ) -> AnxietyAnalysis:
        """
        Analyze exam anxiety levels.
        """
        features = AnxietyFeatures(
            exam_frequency_per_month=exam_frequency_per_month,
            avg_preparation_hours=avg_preparation_hours,
            sleep_night_before=sleep_night_before,
            caffeine_intake=caffeine_intake,
            physical_activity_hours=physical_activity_hours,
            social_support_score=social_support_score,
            previous_exam_score=previous_exam_score,
            target_score=target_score,
            time_until_exam_hours=time_until_exam_hours,
            assignment_completion_rate=assignment_completion_rate,
        )
        
        result = predict_anxiety(features)
        
        return AnxietyAnalysis(
            anxiety_level=result.anxiety_level,
            anxiety_score=result.anxiety_score,
            confidence=result.confidence,
            contributing_factors=result.contributing_factors,
            recommendations=result.recommendations,
        )
    
    def manage_spaced_repetition(
        self,
        items: List[Dict] = None,
        review_history: List[Dict] = None,
        item_reviews: List[Tuple[str, int]] = None,  # (item_id, quality)
        target_date: datetime = None,
        max_items: int = 20,
        max_time_minutes: int = 30,
    ) -> SpacedRepetitionAnalysis:
        """
        Manage spaced repetition reviews.
        """
        # Convert dicts to ReviewItem objects
        review_items = []
        if items:
            for item in items:
                review = ReviewItem(
                    item_id=item["item_id"],
                    subject=item["subject"],
                    topic=item["topic"],
                    question=item["question"],
                    answer=item["answer"],
                    difficulty_rating=item.get("difficulty_rating", 2.5),
                    interval_days=item.get("interval_days", 1),
                    repetitions=item.get("repetitions", 0),
                    ease_factor=item.get("ease_factor", 2.5),
                )
                review_items.append(review)
        
        # Process reviews if provided
        if item_reviews:
            for item_id, quality in item_reviews:
                for item in review_items:
                    if item.item_id == item_id:
                        sm2_review_item(item, quality)
                        break
        
        # Generate schedule
        schedule = generate_daily_schedule(
            items=review_items,
            target_date=target_date,
            max_items=max_items,
            max_time_minutes=max_time_minutes,
        )
        
        # Calculate stats
        stats = calculate_retention_stats(review_items, review_history or [])
        
        schedule_dict = None
        if schedule:
            schedule_dict = {
                "date": schedule.date.isoformat(),
                "total_items": schedule.total_items,
                "estimated_time": schedule.estimated_time_minutes,
                "by_subject": schedule.by_subject,
                "new_items": schedule.new_items_count,
                "review_items": schedule.review_items_count,
            }
        
        return SpacedRepetitionAnalysis(
            items_due=len(schedule.items_due) if schedule else 0,
            daily_schedule=schedule_dict,
            retention_stats={
                "total_reviews": stats.total_reviews,
                "average_quality": stats.average_quality,
                "ease_factor": stats.average_ease_factor,
            },
            retention_rate=stats.retention_rate,
            mastered_items=stats.mastered_items_count,
        )
    
    def analyze_knowledge_gaps(
        self,
        current_marks: Dict[str, float],
        target_marks: Dict[str, float],
        syllabus_topics: List[str],
        recent_quiz_scores: Dict[str, float] = None,
        time_available_hours: float = 10.0,
    ) -> KnowledgeGapAnalysis:
        """
        Analyze knowledge gaps and prioritize learning.
        """
        input_data = KnowledgeGapInput(
            current_marks=current_marks,
            target_marks=target_marks,
            syllabus_topics=syllabus_topics,
            recent_quiz_scores=recent_quiz_scores or {},
        )
        
        result = analyze_knowledge_gaps(input_data)
        prioritized = prioritize_gaps(result.concept_gaps, time_available_hours)
        
        gaps_data = [
            {
                "concept": gap.concept,
                "current_level": gap.current_level,
                "target_level": gap.target_level,
                "gap_size": gap.gap_size,
                "priority": gap.priority,
                "estimated_hours": gap.estimated_hours_to_master,
            }
            for gap in result.concept_gaps
        ]
        
        prioritized_data = [
            {
                "concept": p.concept,
                "hours_allocated": p.hours_allocated,
                "order": p.order,
            }
            for p in prioritized
        ]
        
        return KnowledgeGapAnalysis(
            gap_count=len(result.concept_gaps),
            gaps=gaps_data,
            prioritized_learning=prioritized_data,
            estimated_time_hours=sum(g.estimated_hours_to_master for g in result.concept_gaps),
            focus_areas=result.focus_areas,
        )
    
    def recommend_content(
        self,
        student_id: str,
        subject: str,
        interactions: List[Dict] = None,
        content_items: List[Dict] = None,
        top_n: int = 5,
    ) -> ContentAnalysis:
        """
        Get personalized content recommendations.
        """
        # Convert to domain objects
        student_interactions = []
        if interactions:
            for i in interactions:
                student_interactions.append(StudentInteraction(
                    student_id=i["student_id"],
                    content_id=i["content_id"],
                    interaction_type=i.get("interaction_type", "view"),
                    rating=i.get("rating"),
                    timestamp=datetime.fromisoformat(i["timestamp"]) if i.get("timestamp") else datetime.now(),
                ))
        
        items = []
        if content_items:
            for c in content_items:
                items.append(ContentItem(
                    content_id=c["content_id"],
                    title=c["title"],
                    subject=c["subject"],
                    content_type=c.get("content_type", "video"),
                    difficulty=c.get("difficulty", 5.0),
                    rating=c.get("rating", 0.0),
                    views=c.get("views", 0),
                ))
        
        # Get recommendations
        recs = recommend_content(
            student_id=student_id,
            subject=subject,
            interactions=student_interactions,
            available_content=items,
            top_n=top_n,
        )
        
        # Get popular content
        popular = get_popular_content(items, top_n=top_n)
        
        rec_data = [
            {
                "content_id": r.content_id,
                "title": r.title,
                "subject": r.subject,
                "reason": r.reason,
                "score": r.score,
            }
            for r in recs
        ]
        
        pop_data = [
            {
                "content_id": p.content_id,
                "title": p.title,
                "views": p.views,
                "rating": p.rating,
            }
            for p in popular
        ]
        
        return ContentAnalysis(
            recommendations=rec_data,
            popular_content=pop_data,
            based_on_interactions=len(student_interactions),
        )
    
    def analyze_trends(
        self,
        performance_history: List[Dict],
        subject: str = None,
    ) -> TrendAnalysis:
        """
        Analyze performance trends over time.
        """
        points = []
        for p in performance_history:
            points.append(PerformancePoint(
                date=datetime.fromisoformat(p["date"]) if isinstance(p["date"], str) else p["date"],
                value=p["value"],
                subject=p.get("subject"),
            ))
        
        result = predict_performance_trend(points, subject=subject)
        
        predictions = [
            {
                "date": pred.date.strftime("%Y-%m-%d"),
                "predicted_value": pred.predicted_value,
                "confidence_lower": pred.confidence_lower,
                "confidence_upper": pred.confidence_upper,
            }
            for pred in result.predictions
        ]
        
        return TrendAnalysis(
            trend=result.trend,
            change_percent=result.change_percent,
            confidence=result.confidence,
            predictions=predictions,
        )
    
    def get_adaptive_difficulty(
        self,
        student_id: str,
        subject: str,
        recent_success_rate: float = 0.5,
        total_problems: int = 0,
    ) -> AdaptiveDifficultyAnalysis:
        """
        Get adaptive difficulty recommendation using MAB.
        """
        # Update learning state
        key = f"{student_id}_{subject}"
        if key in self.adaptive_engine.student_states:
            state = self.adaptive_engine.student_states[key]
            state.recent_success_rate = recent_success_rate
            state.total_problems_attempted = total_problems
        
        result = self.adaptive_engine.get_recommended_difficulty(
            student_id=student_id,
            subject=subject,
        )
        
        return AdaptiveDifficultyAnalysis(
            recommended_difficulty=result.difficulty,
            confidence=result.confidence,
            exploration_rate=result.exploration_rate,
            reason=result.reason,
            alternatives=result.alternative_difficulties,
            estimated_success=result.estimated_success_rate,
        )
    
    def record_response_adaptive(
        self,
        student_id: str,
        subject: str,
        problem_id: str,
        selected_answer: str,
        is_correct: bool,
        response_time: float,
        difficulty: float,
        confidence: int = None,
    ):
        """
        Record a response for adaptive difficulty system.
        """
        response = StudentResponse(
            problem_id=problem_id,
            student_id=student_id,
            selected_answer=selected_answer,
            is_correct=is_correct,
            response_time_seconds=response_time,
            confidence_rating=confidence,
        )
        
        self.adaptive_engine.record_response(
            student_id=student_id,
            subject=subject,
            response=response,
            difficulty=difficulty,
        )
    
    def detect_drift(
        self,
        reference_stats: Dict[str, float],
        current_stats: Dict[str, float],
        reference_performance: Dict[str, float] = None,
        current_performance: Dict[str, float] = None,
    ) -> DriftAnalysis:
        """
        Detect data drift in student performance.
        """
        ref_data = DataStatistics(**reference_stats)
        curr_data = DataStatistics(**current_stats)
        
        perf_ref = None
        perf_curr = None
        if reference_performance and current_performance:
            perf_ref = ModelPerformance(**reference_performance)
            perf_curr = ModelPerformance(**current_performance)
        
        result = detect_data_drift(
            reference_stats=ref_data,
            current_stats=curr_data,
            reference_performance=perf_ref,
            current_performance=perf_curr,
        )
        
        perf_dict = None
        if result.model_performance:
            perf_dict = {
                "accuracy": result.model_performance.accuracy,
                "precision": result.model_performance.precision,
                "recall": result.model_performance.recall,
                "f1_score": result.model_performance.f1_score,
            }
        
        return DriftAnalysis(
            has_drift=result.has_drift,
            drift_score=result.drift_score,
            drift_features=result.drift_features,
            recommendation=result.recommendation,
            model_performance=perf_dict,
        )
    
    def run_full_analysis(
        self,
        student_id: str,
        # Academic features
        assignment_marks: float = 3.0,
        attendance_percentage: float = 80.0,
        quiz_marks: float = 7.0,
        midterm_marks: float = 20.0,
        previous_cgpa: float = 3.0,
        # Burnout features
        avg_daily_study_hours: float = 5.0,
        consecutive_study_days: int = 3,
        recent_grade_change: float = 0.0,
        sleep_hours_per_day: float = 7.0,
        assignment_completion_rate: float = 0.8,
        days_until_exam: int = 14,
        # Duration features
        preferred_time: str = "evening",
        avg_session_length: float = 45.0,
        avg_completion_rate: float = 0.8,
        avg_retention: float = 0.7,
        longest_streak: int = 7,
        performance_trend: float = 0.0,
        difficulty: float = 5.0,
        total_study_time: int = 120,
        # Learning style
        visual_score: float = 0.0,
        auditory_score: float = 0.0,
        reading_score: float = 0.0,
        kinesthetic_score: float = 0.0,
        # Anxiety
        exam_frequency: float = 1.0,
        prep_hours: float = 5.0,
        sleep_before: float = 7.0,
        caffeine: float = 1.0,
        activity_hours: float = 1.0,
        social_support: float = 0.7,
        prev_exam_score: float = 75.0,
        target_score: float = 80.0,
        time_until_exam: float = 24.0,
    ) -> UnifiedMLResults:
        """
        Run comprehensive analysis across all ML modules.
        """
        self.student_id = student_id
        
        # Run all analyses
        academic = self.analyze_academic_performance(
            assignment_marks=assignment_marks,
            attendance_percentage=attendance_percentage,
            quiz_marks=quiz_marks,
            midterm_marks=midterm_marks,
            previous_cgpa=previous_cgpa,
        )
        
        burnout = self.analyze_burnout_risk(
            avg_daily_study_hours=avg_daily_study_hours,
            consecutive_study_days=consecutive_study_days,
            recent_grade_change=recent_grade_change,
            sleep_hours_per_day=sleep_hours_per_day,
            assignment_completion_rate=assignment_completion_rate,
            days_until_exam=days_until_exam,
        )
        
        duration = self.optimize_study_duration(
            avg_session_length=avg_session_length,
            avg_completion_rate=avg_completion_rate,
            avg_retention=avg_retention,
            preferred_time=preferred_time,
            longest_streak=longest_streak,
            performance_trend=performance_trend,
            difficulty=difficulty,
            total_study_time=total_study_time,
        )
        
        learning_style = self.detect_learning_style(
            visual_score=visual_score,
            auditory_score=auditory_score,
            reading_score=reading_score,
            kinesthetic_score=kinesthetic_score,
        )
        
        anxiety = self.analyze_anxiety(
            exam_frequency_per_month=exam_frequency,
            avg_preparation_hours=prep_hours,
            sleep_night_before=sleep_before,
            caffeine_intake=caffeine,
            physical_activity_hours=activity_hours,
            social_support_score=social_support,
            previous_exam_score=prev_exam_score,
            target_score=target_score,
            time_until_exam_hours=time_until_exam,
        )
        
        # Generate priority recommendations
        priority_recommendations = []
        
        # Check burnout
        if burnout.risk_level == "High":
            priority_recommendations.append("🔥 URGENT: Burnout risk is HIGH. Take immediate break.")
        elif burnout.risk_level == "Medium":
            priority_recommendations.append("⚠️ Monitor burnout levels. Schedule breaks.")
        
        # Check anxiety
        if anxiety.anxiety_level == "High":
            priority_recommendations.append("😰 High anxiety detected. Try relaxation techniques.")
        
        # Check academic
        if academic.prob_fail > 0.3:
            priority_recommendations.append("📚 At risk of failing. Focus on weak areas.")
        
        # Add style-based recommendation
        priority_recommendations.append(
            f"🎯 Use {learning_style.primary_style.upper()} learning methods for best results."
        )
        
        # Add duration recommendation
        priority_recommendations.append(
            f"⏱️ Optimal study session: {duration.optimal_minutes} minutes with {duration.break_interval}-min breaks."
        )
        
        # Calculate overall health score (inverse of risks)
        health = 1.0
        health -= burnout.risk_score * 0.3
        health -= (anxiety.anxiety_score / 100) * 0.2
        health *= (1 + academic.prob_grade_a * 0.3)
        health = max(0.0, min(1.0, health))
        
        # Other analyses (optional, with defaults)
        spaced_repetition = None
        knowledge_gaps = None
        content = None
        trends = None
        adaptive_difficulty = None
        drift = None
        
        return UnifiedMLResults(
            timestamp=datetime.now(),
            student_id=student_id,
            academic=academic,
            burnout=burnout,
            duration=duration,
            learning_style=learning_style,
            anxiety=anxiety,
            spaced_repetition=spaced_repetition,
            knowledge_gaps=knowledge_gaps,
            content=content,
            trends=trends,
            adaptive_difficulty=adaptive_difficulty,
            drift=drift,
            priority_recommendations=priority_recommendations,
            overall_health_score=round(health, 2),
        )


# Convenience function for quick predictions
def quick_predict(
    prediction_type: str,
    **kwargs
) -> Any:
    """
    Quick prediction using a single ML module.
    
    Usage:
        result = quick_predict("burnout", avg_daily_study_hours=10, ...)
        result = quick_predict("duration", avg_session_length=45, ...)
        result = quick_predict("anxiety", exam_frequency=2, ...)
    """
    engine = MLInference()
    
    if prediction_type == "burnout":
        return engine.analyze_burnout_risk(**kwargs)
    elif prediction_type == "duration":
        return engine.optimize_study_duration(**kwargs)
    elif prediction_type == "learning_style":
        return engine.detect_learning_style(**kwargs)
    elif prediction_type == "anxiety":
        return engine.analyze_anxiety(**kwargs)
    elif prediction_type == "academic":
        return engine.analyze_academic_performance(**kwargs)
    else:
        raise ValueError(f"Unknown prediction type: {prediction_type}")


# Demo
if __name__ == "__main__":
    print("=" * 70)
    print("🧠 UNIFIED ML INFERENCE ENGINE - DEMO")
    print("=" * 70)
    
    # Initialize engine
    engine = MLInference(student_id="student_001")
    
    # Run full analysis
    results = engine.run_full_analysis(
        student_id="student_001",
        # Academic
        assignment_marks=4.0,
        attendance_percentage=90.0,
        quiz_marks=8.0,
        midterm_marks=25.0,
        previous_cgpa=3.5,
        # Burnout (showing signs of burnout)
        avg_daily_study_hours=12.0,
        consecutive_study_days=10,
        recent_grade_change=-10.0,
        sleep_hours_per_day=5.0,
        assignment_completion_rate=0.7,
        days_until_exam=5,
        # Duration
        preferred_time="morning",
        avg_session_length=60.0,
        avg_completion_rate=0.85,
        avg_retention=0.75,
        longest_streak=14,
        performance_trend=-5.0,
        difficulty=7.0,
        total_study_time=180,
        # Learning style
        visual_score=8.0,
        auditory_score=4.0,
        reading_score=6.0,
        kinesthetic_score=5.0,
        # Anxiety (moderate)
        exam_frequency=2.0,
        prep_hours=3.0,
        sleep_before=6.0,
        caffeine=2.0,
        activity_hours=0.5,
        social_support=0.6,
        prev_exam_score=72.0,
        target_score=85.0,
        time_until_exam=12.0,
    )
    
    print(f"\n📊 ACADEMIC ANALYSIS")
    print(f"   Performance Cluster: {results.academic.performance_cluster}")
    print(f"   Study Level: {results.academic.study_level}")
    print(f"   Predicted Final: {results.academic.predicted_final}/50")
    print(f"   Prob Grade A: {results.academic.prob_grade_a:.0%}")
    print(f"   Prob Fail: {results.academic.prob_fail:.0%}")
    
    print(f"\n🔥 BURNOUT ANALYSIS")
    print(f"   Risk Level: {results.burnout.risk_level}")
    print(f"   Risk Score: {results.burnout.risk_score:.2f}")
    print(f"   Recommendation: {results.burnout.recommendation}")
