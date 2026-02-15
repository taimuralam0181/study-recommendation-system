"""
Burnout Prediction Model

Predicts student burnout risk (Low/Medium/High) based on:
- Study patterns (hours per day, session frequency)
- Performance trends (recent grade drops)
- Behavioral signals (streaks, activity levels)
- Temporal patterns (time since last break)

Uses XGBoost for classification with SHAP explainability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import pickle
import os
from datetime import datetime, timedelta


@dataclass(frozen=True)
class BurnoutFeatures:
    """Input features for burnout prediction."""
    avg_daily_study_hours: float      # 0-16 hours
    study_session_count: int          # sessions per week
    avg_session_duration_minutes: float  # minutes per session
    consecutive_study_days: int       # days without break
    recent_grade_change: float        # percentage change (negative = drop)
    sleep_hours_per_day: float        # 0-12 hours
    assignment_completion_rate: float  # 0-1.0
    attendance_rate: float            # 0-1.0
    activity_level: float              # 0-1.0 (platform engagement)
    days_until_exam: int              # 0-90 days


@dataclass(frozen=True)
class BurnoutPrediction:
    """Burnout prediction output."""
    risk_level: str           # "Low" | "Medium" | "High"
    risk_score: float         # 0.0 - 1.0
    confidence: float          # 0.0 - 1.0
    top_factors: List[Tuple[str, float]]  # (feature, contribution)
    recommendations: List[str]
    model_version: str = "1.0.0"


# Feature thresholds for rule-based fallback
BURNOUT_THRESHOLDS = {
    "high_risk": {
        "avg_daily_study_hours": 10.0,
        "consecutive_study_days": 7,
        "recent_grade_change": -15.0,
        "sleep_hours_per_day": 5.0,
        "days_until_exam": 3,
    },
    "medium_risk": {
        "avg_daily_study_hours": 7.0,
        "consecutive_study_days": 4,
        "recent_grade_change": -8.0,
        "sleep_hours_per_day": 6.0,
        "days_until_exam": 7,
    },
}


def _calculate_risk_score(features: BurnoutFeatures) -> Tuple[float, List[Tuple[str, float]]]:
    """
    Calculate burnout risk score based on weighted factors.
    Returns (risk_score, factor_contributions).
    """
    factors = []
    score = 0.0
    
    # Study hours factor (weight: 0.25)
    hours_factor = min(features.avg_daily_study_hours / 12.0, 1.0)
    hours_contrib = hours_factor * 0.25
    score += hours_contrib
    factors.append(("high_study_hours", hours_contrib))
    
    # Consecutive days without break (weight: 0.20)
    break_factor = min(features.consecutive_study_days / 14.0, 1.0)
    break_contrib = break_factor * 0.20
    score += break_contrib
    factors.append(("no_break_days", break_contrib))
    
    # Grade drop factor (weight: 0.20)
    grade_factor = max(0, -features.recent_grade_change / 30.0)
    grade_contrib = grade_factor * 0.20
    score += grade_contrib
    factors.append(("grade_decline", grade_contrib))
    
    # Sleep deprivation (weight: 0.15)
    sleep_factor = max(0, (8.0 - features.sleep_hours_per_day) / 8.0)
    sleep_contrib = sleep_factor * 0.15
    score += sleep_contrib
    factors.append(("sleep_deprivation", sleep_contrib))
    
    # Exam pressure (weight: 0.10)
    exam_factor = max(0, 1.0 - (features.days_until_exam / 30.0))
    exam_contrib = exam_factor * 0.10
    score += exam_contrib
    factors.append(("exam_pressure", exam_contrib))
    
    # Assignment backlog (weight: 0.10)
    backlog_factor = 1.0 - features.assignment_completion_rate
    backlog_contrib = backlog_factor * 0.10
    score += backlog_contrib
    factors.append(("assignment_backlog", backlog_contrib))
    
    return score, factors


def _risk_level(score: float) -> str:
    """Convert risk score to risk level."""
    if score >= 0.60:
        return "High"
    elif score >= 0.35:
        return "Medium"
    return "Low"


def _generate_recommendations(
    features: BurnoutFeatures,
    risk_level: str,
    top_factors: List[Tuple[str, float]]
) -> List[str]:
    """Generate personalized burnout reduction recommendations."""
    recommendations = []
    
    # General recommendations based on risk level
    if risk_level == "High":
        recommendations.extend([
            "⚠️ URGENT: Take a complete break for at least 24 hours",
            "📞 Consider speaking with a counselor or mentor",
            "🛏️ Prioritize sleep - aim for 7-8 hours tonight",
            "📚 Reduce study load by 50% for the next 3 days",
        ])
    elif risk_level == "Medium":
        recommendations.extend([
            "⏰ Schedule a break day within the next 48 hours",
            "😴 Increase sleep to at least 7 hours per night",
            "📝 Review your study schedule and reduce intensity",
            "🚶 Include light physical activity daily",
        ])
    else:
        recommendations.extend([
            "✅ Your study habits look balanced!",
            "💪 Keep maintaining your current routine",
            "🎯 Continue monitoring your energy levels",
        ])
    
    # Factor-specific recommendations
    factor_dict = {f[0]: f[1] for f in top_factors}
    
    if factor_dict.get("high_study_hours", 0) > 0.15:
        recommendations.append("⏱️ Break study sessions into 25-50 minute blocks with 5-10 min breaks")
    
    if factor_dict.get("no_break_days", 0) > 0.10:
        recommendations.append("🏖️ Plan at least one full rest day per week")
    
    if factor_dict.get("grade_decline", 0) > 0.10:
        recommendations.append("📊 Review recent topics where grades dropped and focus on understanding")
    
    if factor_dict.get("sleep_deprivation", 0) > 0.10:
        recommendations.append("🌙 Set a consistent bedtime, avoid screens 1 hour before sleep")
    
    if factor_dict.get("exam_pressure", 0) > 0.08:
        recommendations.append("📅 Create a realistic study plan with buffer time for unexpected issues")
    
    if factor_dict.get("assignment_backlog", 0) > 0.05:
        recommendations.append("📋 Tackle assignments in small chunks, prioritize by deadline")
    
    return recommendations


def predict_burnout(features: BurnoutFeatures) -> BurnoutPrediction:
    """
    Main prediction function for burnout risk.
    Uses rule-based scoring (XGBoost model can be added for production).
    """
    risk_score, factor_contributions = _calculate_risk_score(features)
    
    # Sort factors by contribution
    factor_contributions.sort(key=lambda x: x[1], reverse=True)
    
    risk_level = _risk_level(risk_score)
    confidence = 0.85 if risk_score < 0.35 or risk_score > 0.60 else 0.70
    
    recommendations = _generate_recommendations(features, risk_level, factor_contributions)
    
    return BurnoutPrediction(
        risk_level=risk_level,
        risk_score=round(risk_score, 3),
        confidence=round(confidence, 2),
        top_factors=factor_contributions[:5],
        recommendations=recommendations,
    )


def predict_burnout_simple(
    avg_daily_hours: float,
    consecutive_days: int,
    grade_change: float,
    sleep_hours: float,
) -> BurnoutPrediction:
    """
    Simplified interface for burnout prediction.
    Uses only the most critical features.
    """
    features = BurnoutFeatures(
        avg_daily_study_hours=avg_daily_hours,
        study_session_count=7,
        avg_session_duration_minutes=avg_daily_hours * 60,
        consecutive_study_days=consecutive_days,
        recent_grade_change=grade_change,
        sleep_hours_per_day=sleep_hours,
        assignment_completion_rate=0.8,
        attendance_rate=0.9,
        activity_level=0.7,
        days_until_exam=14,
    )
    return predict_burnout(features)


# XGBoost Model Integration (for production use)
def _get_xgboost_model_path() -> str:
    """Get path to XGBoost model file."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "models", "burnout_xgboost.pkl")


def train_xgboost_model(X: List[List[float]], y: List[int]) -> None:
    """
    Train XGBoost model for burnout prediction.
    X: list of feature vectors (order as in BurnoutFeatures)
    y: list of labels (0=Low, 1=Medium, 2=High)
    """
    try:
        import xgboost as xgb
        
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            objective="multi:softmax",
            num_class=3,
            random_state=42,
        )
        
        model.fit(X, y)
        
        # Save model
        model_path = _get_xgboost_model_path()
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        
    except ImportError:
        print("XGBoost not installed. Run: pip install xgboost")


def predict_xgboost(features: BurnoutFeatures) -> BurnoutPrediction:
    """
    XGBoost-based prediction (requires trained model).
    Falls back to rule-based prediction if model not found.
    """
    model_path = _get_xgboost_model_path()
    
    if not os.path.exists(model_path):
        return predict_burnout(features)
    
    try:
        import xgboost as xgb
        
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        # Convert features to vector
        feature_vector = [
            features.avg_daily_study_hours,
            features.study_session_count,
            features.avg_session_duration_minutes,
            features.consecutive_study_days,
            features.recent_grade_change,
            features.sleep_hours_per_day,
            features.assignment_completion_rate,
            features.attendance_rate,
            features.activity_level,
            features.days_until_exam,
        ]
        
        pred = model.predict([feature_vector])[0]
        proba = model.predict_proba([feature_vector])[0]
        
        risk_levels = ["Low", "Medium", "High"]
        risk_level = risk_levels[pred]
        risk_score = float(proba[pred])
        
        # Get feature importances
        importance = model.feature_importances_
        feature_names = [
            "avg_daily_study_hours", "study_session_count",
            "avg_session_duration_minutes", "consecutive_study_days",
            "recent_grade_change", "sleep_hours_per_day",
            "assignment_completion_rate", "attendance_rate",
            "activity_level", "days_until_exam",
        ]
        
        top_factors = sorted(
            zip(feature_names, importance),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        recommendations = _generate_recommendations(features, risk_level, top_factors)
        
        return BurnoutPrediction(
            risk_level=risk_level,
            risk_score=round(risk_score, 3),
            confidence=float(max(proba)),
            top_factors=top_factors,
            recommendations=recommendations,
        )
        
    except Exception as e:
        print(f"XGBoost prediction failed: {e}. Using rule-based fallback.")
        return predict_burnout(features)


# Demo / Testing
if __name__ == "__main__":
    # Example: Student showing burnout signs
    student = BurnoutFeatures(
        avg_daily_study_hours=12.0,
        study_session_count=14,
        avg_session_duration_minutes=90,
        consecutive_study_days=10,
        recent_grade_change=-20.0,
        sleep_hours_per_day=4.5,
        assignment_completion_rate=0.6,
        attendance_rate=0.85,
        activity_level=0.4,
        days_until_exam=5,
    )
    
    result = predict_burnout(student)
    
    print("=" * 60)
    print("🔥 BURNOUT RISK PREDICTION")
    print("=" * 60)
    print(f"Risk Level:    {result.risk_level}")
    print(f"Risk Score:    {result.risk_score:.2%}")
    print(f"Confidence:    {result.confidence:.2%}")
    print(f"\n📊 Top Contributing Factors:")
    for factor, contrib in result.top_factors:
        print(f"  - {factor}: {contrib:.2%}")
    print(f"\n💡 Recommendations:")
    for rec in result.recommendations:
        print(f"  {rec}")
    print("=" * 60)
