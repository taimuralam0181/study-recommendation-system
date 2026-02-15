"""
Exam Anxiety Detection Model

Predicts student anxiety levels before/during exams based on:
- Historical performance patterns
- Study behavior signals
- Time until exam
- Previous exam performance variance
- Sleep and stress indicators

Provides anxiety level classification and personalized coping strategies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class AnxietyFeatures:
    """Input features for anxiety detection."""
    # Performance patterns
    recent_grade_variance: float       # Variance in last 5 grades (0-100)
    performance_trend: float          # Recent trend (positive = improving)
    historical_exam_score_gap: float  # Gap between practice and actual exams
    avg_preparation_time: float        # Hours spent preparing
    
    # Behavioral signals
    study_procrastination_rate: float  # 0-1.0 (1 = high procrastination)
    last_minute_study_ratio: float      # % of study done in last 24h
    sleep_quality_index: float         # 0-1.0 (1 = good sleep)
    physical_activity_rate: float      # 0-1.0
    
    # Context
    days_until_exam: int               # 0-30
    exam_weight_percentage: float      # How much exam counts (0-100)
    num_past_exams_failed: int         # Failed exams in history
    
    # Self-reported (optional)
    stress_self_rating: Optional[int] = None  # 1-10 self-reported stress


@dataclass(frozen=True)
class AnxietyPrediction:
    """Anxiety prediction output."""
    anxiety_level: str           # "Low" | "Moderate" | "High" | "Very High"
    anxiety_score: float         # 0-1.0
    primary_triggers: List[Tuple[str, float]]  # (trigger, contribution)
    coping_strategies: List[str]
    recommendations: List[str]
    confidence: float
    model_version: str = "1.0.0"


# Anxiety thresholds
ANXIETY_THRESHOLDS = {
    "very_high": 0.75,
    "high": 0.55,
    "moderate": 0.35,
}


def _calculate_anxiety_score(features: AnxietyFeatures) -> Tuple[float, List[Tuple[str, float]]]:
    """
    Calculate anxiety score based on multiple factors.
    Returns (anxiety_score, factor_contributions).
    """
    factors = []
    score = 0.0
    
    # Performance variance factor (weight: 0.20)
    # High variance = unpredictable = more anxiety
    variance_factor = min(features.recent_grade_variance / 400, 1.0)
    variance_contrib = variance_factor * 0.20
    score += variance_contrib
    factors.append(("performance_variance", variance_contrib))
    
    # Performance trend factor (weight: 0.15)
    # Declining trend = anxiety about continuing decline
    trend_factor = max(0, -features.performance_trend / 50)
    trend_contrib = trend_factor * 0.15
    score += trend_contrib
    factors.append(("declining_performance", trend_contrib))
    
    # Exam gap factor (weight: 0.15)
    # Large gap between practice and actual = anxiety about unknown
    gap_factor = min(features.historical_exam_score_gap / 30, 1.0)
    gap_contrib = gap_factor * 0.15
    score += gap_contrib
    factors.append(("practice_exam_gap", gap_contrib))
    
    # Procrastination factor (weight: 0.15)
    procrastination_factor = features.study_procrastination_rate
    procrast_contrib = procrastination_factor * 0.15
    score += procrast_contrib
    factors.append(("procrastination", procrast_contrib))
    
    # Last-minute study factor (weight: 0.10)
    last_minute_factor = features.last_minute_study_ratio
    last_minute_contrib = last_minute_factor * 0.10
    score += last_minute_contrib
    factors.append(("last_minute_cramming", last_minute_contrib))
    
    # Sleep quality factor (weight: 0.10)
    # Poor sleep = higher anxiety
    sleep_factor = 1.0 - features.sleep_quality_index
    sleep_contrib = sleep_factor * 0.10
    score += sleep_contrib
    factors.append(("poor_sleep", sleep_contrib))
    
    # Exam weight factor (weight: 0.08)
    weight_factor = features.exam_weight_percentage / 100
    weight_contrib = weight_factor * 0.08
    score += weight_contrib
    factors.append(("high_stakes_exam", weight_contrib))
    
    # Time pressure factor (weight: 0.07)
    time_factor = max(0, 1.0 - (features.days_until_exam / 14))
    time_contrib = time_factor * 0.07
    score += time_contrib
    factors.append(("time_pressure", time_contrib))
    
    # Failure history factor (weight: 0.05)
    failure_factor = min(features.num_past_exams_failed / 5, 1.0)
    failure_contrib = failure_factor * 0.05
    score += failure_contrib
    factors.append(("past_failures", failure_contrib))
    
    # Self-reported stress (weight: 0.10 override)
    if features.stress_self_rating is not None:
        self_report = (features.stress_self_rating - 1) / 9.0  # 0-1 scale
        # Blend with calculated score
        score = score * 0.7 + self_report * 0.3
        factors.append(("self_reported_stress", self_report * 0.10))
    
    return score, factors


def _anxiety_level(score: float) -> str:
    """Convert score to anxiety level."""
    if score >= ANXIETY_THRESHOLDS["very_high"]:
        return "Very High"
    elif score >= ANXIETY_THRESHOLDS["high"]:
        return "High"
    elif score >= ANXIETY_THRESHOLDS["moderate"]:
        return "Moderate"
    return "Low"


def _get_coping_strategies(
    primary_triggers: List[Tuple[str, float]],
    anxiety_level: str,
) -> List[str]:
    """Generate personalized coping strategies."""
    strategies = []
    
    trigger_dict = {t[0]: t[1] for t in primary_triggers}
    
    # Level-based immediate strategies
    if anxiety_level in ["Very High", "High"]:
        strategies.extend([
            "🧘 Practice 5-minute deep breathing (4-7-8 technique)",
            "📝 Write down worries for 10 minutes, then set aside",
            "🚶 Take a 15-minute walk to release tension",
            "💬 Talk to someone you trust about your concerns",
        ])
    else:
        strategies.extend([
            "🧘 Try a 2-minute mindfulness exercise",
            "📋 Review what you've prepared to build confidence",
        ])
    
    # Trigger-specific strategies
    if trigger_dict.get("performance_variance", 0) > 0.10:
        strategies.append("📊 Focus on consistency over perfection - small daily wins matter")
    
    if trigger_dict.get("declining_performance", 0) > 0.08:
        strategies.append("📈 Identify one topic to master today, build momentum gradually")
    
    if trigger_dict.get("practice_exam_gap", 0) > 0.08:
        strategies.append("📝 Do a full practice exam under realistic conditions")
    
    if trigger_dict.get("procrastination", 0) > 0.08:
        strategies.append("⏰ Use the 2-minute rule: if it takes <2 min, do it now")
    
    if trigger_dict.get("last_minute_cramming", 0) > 0.05:
        strategies.append("📅 Plan tomorrow's study session tonight (just 15 min)")
    
    if trigger_dict.get("poor_sleep", 0) > 0.05:
        strategies.append("😴 Aim for 7-8 hours sleep - memory consolidation happens during sleep")
    
    if trigger_dict.get("high_stakes_exam", 0) > 0.04:
        strategies.append("💡 Remember: one exam doesn't define your worth or future")
    
    if trigger_dict.get("time_pressure", 0) > 0.04:
        strategies.append("📋 Create a realistic 30-minute study plan for tomorrow")
    
    if trigger_dict.get("past_failures", 0) > 0.03:
        strategies.append("🌟 Past failures are learning opportunities, not permanent labels")
    
    # General anxiety management
    strategies.extend([
        "💪 Trust your preparation - you've got this!",
    ])
    
    return strategies


def _get_recommendations(
    anxiety_level: str,
    days_until_exam: int,
) -> List[str]:
    """Generate preparation recommendations."""
    recommendations = []
    
    if anxiety_level == "Very High":
        recommendations = [
            "⚠️ Consider talking to a counselor or mental health professional",
            "📞 Reach out to your instructor for additional support",
            "🧘 Practice relaxation techniques daily",
            "📅 Break study into very small, manageable chunks",
            "💤 Prioritize sleep above all else",
        ]
    elif anxiety_level == "High":
        recommendations = [
            "📋 Focus on high-yield topics, don't try to learn everything",
            "🧘 Practice breathing exercises before starting study",
            "📞 Don't hesitate to ask for help from peers or instructors",
            "🎯 Set realistic expectations for this exam",
        ]
    elif anxiety_level == "Moderate":
        recommendations = [
            "📊 Create a balanced study schedule",
            "🧘 Include short mindfulness breaks in study sessions",
            "📝 Do practice questions to build confidence",
            "💪 Maintain your healthy habits (sleep, food, exercise)",
        ]
    else:
        recommendations = [
            "✅ Great job managing your stress!",
            "📊 Keep up your consistent study habits",
            "💪 Trust your preparation process",
        ]
    
    # Time-based adjustments
    if days_until_exam <= 1:
        recommendations.append("🌙 Tonight: Light review, then relax. Cramming now will hurt more than help.")
    elif days_until_exam <= 3:
        recommendations.append("📋 Focus on review and practice, minimize new content")
    
    return recommendations


def detect_anxiety(features: AnxietyFeatures) -> AnxietyPrediction:
    """
    Main function to detect exam anxiety level.
    """
    anxiety_score, factor_contributions = _calculate_anxiety_score(features)
    
    # Sort factors by contribution
    factor_contributions.sort(key=lambda x: x[1], reverse=True)
    
    anxiety_level = _anxiety_level(anxiety_score)
    
    # Calculate confidence
    data_points = 0
    data_points += 1 if features.recent_grade_variance > 0 else 0
    data_points += 1 if features.study_procrastination_rate > 0 else 0
    data_points += 1 if features.sleep_quality_index > 0 else 0
    data_points += 1 if features.stress_self_rating is not None else 0
    
    confidence = min(0.9, 0.5 + data_points * 0.1)
    
    coping_strategies = _get_coping_strategies(factor_contributions, anxiety_level)
    recommendations = _get_recommendations(anxiety_level, features.days_until_exam)
    
    return AnxietyPrediction(
        anxiety_level=anxiety_level,
        anxiety_score=round(anxiety_score, 3),
        primary_triggers=factor_contributions[:5],
        coping_strategies=coping_strategies[:6],
        recommendations=recommendations,
        confidence=round(confidence, 2),
    )


def detect_anxiety_simple(
    grade_variance: float,
    trend: float,
    procrastination_rate: float,
    sleep_quality: float,
    days_until_exam: int,
    self_rating: Optional[int] = None,
) -> AnxietyPrediction:
    """
    Simplified interface for anxiety detection.
    """
    features = AnxietyFeatures(
        recent_grade_variance=grade_variance,
        performance_trend=trend,
        historical_exam_score_gap=10,
        avg_preparation_time=5,
        study_procrastination_rate=procrastination_rate,
        last_minute_study_ratio=0.3,
        sleep_quality_index=sleep_quality,
        physical_activity_rate=0.5,
        days_until_exam=days_until_exam,
        exam_weight_percentage=30,
        num_past_exams_failed=0,
        stress_self_rating=self_rating,
    )
    return detect_anxiety(features)


# Alias for compatibility with unified_inference.py
predict_anxiety = detect_anxiety


# Anxiety Self-Assessment Questionnaire
ANXIETY_QUESTIONS = [
    {
        "id": 1,
        "text": "How often do you feel nervous or anxious before exams?",
        "options": [
            ("Never", 0.0),
            ("Rarely", 0.25),
            ("Sometimes", 0.5),
            ("Often", 0.75),
            ("Always", 1.0),
        ],
    },
    {
        "id": 2,
        "text": "Do you have trouble sleeping before an important exam?",
        "options": [
            ("Never", 0.0),
            ("Rarely", 0.25),
            ("Sometimes", 0.5),
            ("Often", 0.75),
            ("Always", 1.0),
        ],
    },
    {
        "id": 3,
        "text": "Do you feel your mind goes blank during exams?",
        "options": [
            ("Never", 0.0),
            ("Rarely", 0.25),
            ("Sometimes", 0.5),
            ("Often", 0.75),
            ("Always", 1.0),
        ],
    },
    {
        "id": 4,
        "text": "How much do exams affect your daily life?",
        "options": [
            ("Not at all", 0.0),
            ("A little", 0.25),
            ("Moderately", 0.5),
            ("Very much", 0.75),
            ("Extremely", 1.0),
        ],
    },
    {
        "id": 5,
        "text": "Do you avoid studying because of exam anxiety?",
        "options": [
            ("Never", 0.0),
            ("Rarely", 0.25),
            ("Sometimes", 0.5),
            ("Often", 0.75),
            ("Always", 1.0),
        ],
    },
]


def assess_from_questionnaire(answers: Dict[int, float]) -> Dict:
    """
    Calculate anxiety score from self-assessment questionnaire.
    answers: {question_id: score_value}
    """
    if not answers:
        return {"error": "No answers provided"}
    
    total = sum(answers.values())
    count = len(answers)
    avg_score = total / count if count > 0 else 0
    
    level = _anxiety_level(avg_score)
    
    return {
        "anxiety_score": round(avg_score, 3),
        "anxiety_level": level,
        "question_count": count,
        "recommendations": _get_recommendations(level, 7)[:4],
    }


# Demo / Testing
if __name__ == "__main__":
    print("=" * 60)
    print("🧠 EXAM ANXIETY DETECTION")
    print("=" * 60)
    
    # Example: Student with high anxiety signs
    student = AnxietyFeatures(
        recent_grade_variance=350,      # High variance
        performance_trend=-15,          # Declining
        historical_exam_score_gap=20,   # 20 point drop in actual exams
        avg_preparation_time=3,
        study_procrastination_rate=0.8,  # High procrastination
        last_minute_study_ratio=0.7,    # 70% studying last minute
        sleep_quality_index=0.4,         # Poor sleep
        physical_activity_rate=0.2,
        days_until_exam=3,
        exam_weight_percentage=50,
        num_past_exams_failed=2,
        stress_self_rating=8,           # High self-reported stress
    )
    
    result = detect_anxiety(student)
    
    print(f"\n😰 Anxiety Level: {result.anxiety_level}")
    print(f"   Anxiety Score: {result.anxiety_score:.0%}")
    print(f"   Confidence: {result.confidence:.0%}")
    
    print(f"\n🔍 Primary Triggers:")
    for trigger, contrib in result.primary_triggers[:5]:
        print(f"   - {trigger.replace('_', ' ').title()}: {contrib:.0%}")
    
    print(f"\n💡 Coping Strategies:")
    for strategy in result.coping_strategies[:4]:
        print(f"   {strategy}")
    
    print(f"\n📋 Recommendations:")
    for rec in result.recommendations[:4]:
        print(f"   {rec}")
    
    print("=" * 60)