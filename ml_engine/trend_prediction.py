"""
Trend Prediction Module

Predicts student performance trends using time series analysis:
- Linear trend analysis
- Moving average smoothing
- Exponential smoothing
- Seasonality detection
- Performance trajectory prediction

Forecasts future grades and identifies at-risk students.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import math
import json


@dataclass(frozen=True)
class PerformanceDataPoint:
    """A single performance data point."""
    date: datetime
    score: float              # 0-100
    subject: str
    activity_type: str        # "quiz", "exam", "assignment", "practice"
    weight: float            # Importance weight (exam = 1.0, quiz = 0.5)


@dataclass(frozen=True)
class TrendFeatures:
    """Extracted features for trend analysis."""
    slope: float                      # Overall trend slope
    slope_significant: bool           # Is trend statistically significant?
    recent_trend: float              # Recent (last 5 points) trend
    volatility: float                 # Standard deviation of scores
    seasonality_strength: float      # 0-1 seasonal pattern strength
    cyclicity: float                 # 0-1 recurring cycle strength
    trend_direction: str              # "improving", "declining", "stable"
    acceleration: float              # Is the trend accelerating?


@dataclass(frozen=True)
class TrendPrediction:
    """Trend prediction result."""
    student_id: str
    subject: str
    current_score: float               # Latest score
    predicted_score_1w: float         # 1 week ahead
    predicted_score_2w: float         # 2 weeks ahead
    predicted_score_4w: float         # 4 weeks ahead
    trend_features: TrendFeatures
    risk_level: str                   # "low", "medium", "high"
    confidence: float
    insights: List[str]
    recommendations: List[str]


@dataclass(frozen=True)
class MultiSubjectReport:
    """Complete trend analysis across all subjects."""
    student_id: str
    analysis_date: datetime
    subject_reports: Dict[str, TrendPrediction]
    overall_trend: str
    at_risk_subjects: List[str]
    improving_subjects: List[str]
    priority_actions: List[str]


@dataclass(frozen=True)
class SimpleTrendResult:
    """Simplified trend result for lightweight callers/tests."""
    trend_direction: str
    details: Optional[TrendPrediction]


def calculate_moving_average(
    data: List[float],
    window: int,
) -> List[float]:
    """Calculate moving average with centered window."""
    if len(data) < window:
        return data
    
    result = []
    half = window // 2
    
    for i in range(len(data)):
        start = max(0, i - half)
        end = min(len(data), i + half + 1)
        window_data = data[start:end]
        result.append(sum(window_data) / len(window_data))
    
    return result


def linear_trend_slope(
    dates: List[datetime],
    scores: List[float],
) -> Tuple[float, float]:
    """
    Calculate linear trend using least squares.
    Returns (slope, r_squared).
    """
    n = len(dates)
    if n < 2:
        return 0.0, 0.0
    
    # Convert dates to numeric (days since first date)
    base_date = min(dates)
    x = [(d - base_date).days for d in dates]
    y = scores
    
    # Calculate means
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    # Calculate slope and intercept
    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denominator = sum((xi - mean_x) ** 2 for xi in x)
    
    if denominator == 0:
        return 0.0, 0.0
    
    slope = numerator / denominator
    
    # Calculate R-squared
    ss_tot = sum((yi - mean_y) ** 2 for yi in y)
    ss_res = sum((yi - (slope * xi + mean_y)) ** 2 for xi, yi in zip(x, y))
    
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return slope, r_squared


def exponential_smoothing(
    data: List[float],
    alpha: float = 0.3,
) -> List[float]:
    """Apply exponential smoothing."""
    if not data:
        return []
    
    result = [data[0]]
    
    for i in range(1, len(data)):
        smoothed = alpha * data[i] + (1 - alpha) * result[i - 1]
        result.append(smoothed)
    
    return result


def detect_seasonality(
    dates: List[datetime],
    scores: List[float],
    cycle_length: int = 7,  # Weekly cycle
) -> float:
    """Detect seasonal patterns in the data."""
    if len(scores) < cycle_length * 2:
        return 0.0
    
    # Group by position in cycle
    cycle_sums = [0.0] * cycle_length
    cycle_counts = [0] * cycle_length
    
    for i, score in enumerate(scores):
        pos = i % cycle_length
        cycle_sums[pos] += score
        cycle_counts[pos] += 1
    
    cycle_avgs = [s / c if c > 0 else 0 for s, c in zip(cycle_sums, cycle_counts)]
    overall_avg = sum(scores) / len(scores)
    
    # Calculate variance due to seasonality
    variance = sum((avg - overall_avg) ** 2 for avg in cycle_avgs) / cycle_length
    total_variance = sum((s - overall_avg) ** 2 for s in scores) / len(scores)
    
    if total_variance == 0:
        return 0.0
    
    return min(1.0, variance / total_variance)


def extract_trend_features(
    dates: List[datetime],
    scores: List[float],
    weights: Optional[List[float]] = None,
) -> TrendFeatures:
    """Extract comprehensive trend features from performance data."""
    if not dates or not scores:
        return TrendFeatures(
            slope=0, slope_significant=False, recent_trend=0,
            volatility=0, seasonality_strength=0, cyclicity=0,
            trend_direction="stable", acceleration=0,
        )
    
    if weights is None:
        weights = [1.0] * len(scores)
    
    # Calculate overall trend
    slope, r_squared = linear_trend_slope(dates, scores)
    
    # Significant if R² > 0.3 and slope magnitude > 0.5 points/day
    slope_significant = r_squared > 0.3 and abs(slope) > 0.5
    
    # Recent trend (last 5 points)
    recent_scores = scores[-5:] if len(scores) >= 5 else scores
    if len(recent_scores) >= 2:
        recent_slope, _ = linear_trend_slope(
            dates[-len(recent_scores):],
            recent_scores
        )
    else:
        recent_slope = 0
    
    # Volatility (standard deviation)
    mean_score = sum(scores) / len(scores)
    variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
    volatility = math.sqrt(variance)
    
    # Seasonality
    seasonality = detect_seasonality(dates, scores)
    
    # Trend direction
    if slope > 0.5:
        trend_direction = "improving"
    elif slope < -0.5:
        trend_direction = "declining"
    else:
        trend_direction = "stable"
    
    # Acceleration (change in slope)
    acceleration = recent_slope - slope
    
    return TrendFeatures(
        slope=round(slope, 3),
        slope_significant=slope_significant,
        recent_trend=round(recent_slope, 3),
        volatility=round(volatility, 2),
        seasonality_strength=round(seasonality, 3),
        cyclicity=round(seasonality, 3),  # Simplified
        trend_direction=trend_direction,
        acceleration=round(acceleration, 3),
    )


def predict_future_scores(
    dates: List[datetime],
    scores: List[float],
    weeks_ahead: int = 4,
) -> List[float]:
    """
    Predict future scores using trend extrapolation.
    """
    if len(scores) < 3:
        # Not enough data, return last score
        return [scores[-1]] * weeks_ahead
    
    slope, r_squared = linear_trend_slope(dates, scores)
    
    # Use weighted combination of trend and exponential smoothing
    smoothed = exponential_smoothing(scores)
    last_smoothed = smoothed[-1]
    
    predictions = []
    last_date = dates[-1]
    last_score = scores[-1]
    
    for w in range(1, weeks_ahead + 1):
        # Days ahead
        days = w * 7
        
        # Trend prediction
        trend_pred = last_score + slope * days
        
        # Smoothed average contribution
        smoothed_pred = last_smoothed * 0.7 + trend_pred * 0.3
        
        # Combine based on R² (higher R² = more trust in trend)
        combined = smoothed_pred * (1 - r_squared) + trend_pred * r_squared
        
        # Blend toward recent performance
        final_pred = last_score * 0.2 + combined * 0.8
        
        predictions.append(max(0, min(100, final_pred)))
    
    return predictions


def analyze_performance_trend(
    student_id: str,
    subject: str,
    data_points: List[PerformanceDataPoint],
) -> TrendPrediction:
    """
    Main function to analyze performance trends.
    """
    if not data_points:
        return TrendPrediction(
            student_id=student_id,
            subject=subject,
            current_score=0,
            predicted_score_1w=0,
            predicted_score_2w=0,
            predicted_score_4w=0,
            trend_features=TrendFeatures(0, False, 0, 0, 0, 0, "stable", 0),
            risk_level="high",
            confidence=0.0,
            insights=["No performance data available"],
            recommendations=["Start tracking your scores to enable trend analysis"],
        )
    
    # Sort by date
    sorted_points = sorted(data_points, key=lambda x: x.date)
    
    dates = [p.date for p in sorted_points]
    scores = [p.score for p in sorted_points]
    weights = [p.weight for p in sorted_points]
    
    # Extract features
    features = extract_trend_features(dates, scores, weights)
    
    # Current score (latest)
    current_score = scores[-1]
    
    # Predictions
    predictions = predict_future_scores(dates, scores, weeks_ahead=4)
    
    # Risk level
    if features.trend_direction == "declining" and abs(features.slope) > 1.0:
        risk_level = "high"
    elif features.trend_direction == "declining" or features.slope_significant and features.slope < 0:
        risk_level = "medium"
    else:
        risk_level = "low"
    
    # Confidence based on data quantity and quality
    confidence = min(0.9, 0.4 + len(data_points) * 0.05)
    if len(data_points) < 5:
        confidence -= 0.1
    
    # Generate insights
    insights = []
    recommendations = []
    
    if features.trend_direction == "improving":
        insights.append(f"📈 Performance is improving ({features.slope:.2f} points/day)")
        recommendations.append("Great momentum! Keep up the consistent effort")
    elif features.trend_direction == "declining":
        insights.append(f"📉 Performance is declining ({features.slope:.2f} points/day)")
        recommendations.append("Consider adjusting your study strategy")
    else:
        insights.append("📊 Performance is stable")
    
    if features.recent_trend > features.slope:
        insights.append("🚀 Recent improvement accelerating!")
    elif features.recent_trend < features.slope:
        insights.append("⚠️ Recent performance slowing down")
    
    if features.volatility > 15:
        insights.append("📊 High score variability - focus on consistency")
        recommendations.append("Practice under varied conditions to improve consistency")
    
    if features.seasonality_strength > 0.3:
        insights.append(f"🔄 Detected weekly patterns in your performance")
        recommendations.append("Align your study schedule with your peak performance times")
    
    if features.slope_significant:
        recommendations.append(f"Trend is statistically significant - take action now")
    
    # Add risk-specific recommendations
    if risk_level == "high":
        recommendations.extend([
            "⚠️ Consider seeking extra help",
            "📅 Increase study time for this subject",
            "💬 Talk to your instructor about difficulties",
        ])
    elif risk_level == "medium":
        recommendations.extend([
            "📊 Monitor your progress closely",
            "📝 Focus on weak areas identified in recent tests",
        ])
    
    return TrendPrediction(
        student_id=student_id,
        subject=subject,
        current_score=round(current_score, 1),
        predicted_score_1w=round(predictions[0], 1),
        predicted_score_2w=round(predictions[1], 1),
        predicted_score_4w=round(predictions[2], 1),
        trend_features=features,
        risk_level=risk_level,
        confidence=round(confidence, 2),
        insights=insights,
        recommendations=recommendations,
    )


def predict_performance_trend(
    data_points: List[PerformanceDataPoint],
) -> SimpleTrendResult:
    """
    Backwards-compatible helper returning only the trend direction.
    """
    if not data_points:
        return SimpleTrendResult(trend_direction="Stable", details=None)

    subject = data_points[-1].subject or "Subject"
    prediction = analyze_performance_trend("unknown", subject, data_points)
    direction = prediction.trend_features.trend_direction
    normalized = {
        "improving": "Improving",
        "declining": "Declining",
        "stable": "Stable",
    }.get(direction, "Stable")
    return SimpleTrendResult(trend_direction=normalized, details=prediction)


def generate_multi_subject_report(
    student_id: str,
    subject_data: Dict[str, List[PerformanceDataPoint]],
) -> MultiSubjectReport:
    """
    Generate trend report across all subjects.
    """
    reports = {}
    at_risk = []
    improving = []
    
    for subject, data in subject_data.items():
        report = analyze_performance_trend(student_id, subject, data)
        reports[subject] = report
        
        if report.risk_level == "high":
            at_risk.append(subject)
        if report.trend_features.trend_direction == "improving":
            improving.append(subject)
    
    # Overall trend
    improving_count = sum(
        1 for r in reports.values()
        if r.trend_features.trend_direction == "improving"
    )
    declining_count = sum(
        1 for r in reports.values()
        if r.trend_features.trend_direction == "declining"
    )
    
    if improving_count > declining_count:
        overall_trend = "improving"
    elif declining_count > improving_count:
        overall_trend = "declining"
    else:
        overall_trend = "stable"
    
    # Priority actions
    priority_actions = []
    for subject in at_risk:
        priority_actions.append(f"🔴 Focus on {subject} - at risk of decline")
    for subject in improving[:2]:
        priority_actions.append(f"🟢 Maintain momentum in {subject}")
    
    return MultiSubjectReport(
        student_id=student_id,
        analysis_date=datetime.now(),
        subject_reports=reports,
        overall_trend=overall_trend,
        at_risk_subjects=at_risk,
        improving_subjects=improving,
        priority_actions=priority_actions,
    )


def export_trend_report_json(report: TrendPrediction) -> str:
    """Export trend prediction to JSON."""
    def serialize_datetime(dt: datetime) -> str:
        return dt.isoformat()
    
    report_dict = {
        "student_id": report.student_id,
        "subject": report.subject,
        "current_score": report.current_score,
        "predictions": {
            "1_week": report.predicted_score_1w,
            "2_weeks": report.predicted_score_2w,
            "4_weeks": report.predicted_score_4w,
        },
        "trend_features": {
            "slope": report.trend_features.slope,
            "significant": report.trend_features.slope_significant,
            "recent_trend": report.trend_features.recent_trend,
            "volatility": report.trend_features.volatility,
            "seasonality": report.trend_features.seasonality_strength,
            "direction": report.trend_features.trend_direction,
            "acceleration": report.trend_features.acceleration,
        },
        "risk_level": report.risk_level,
        "confidence": report.confidence,
        "insights": report.insights,
        "recommendations": report.recommendations,
    }
    
    return json.dumps(report_dict, indent=2)


# Demo / Testing
if __name__ == "__main__":
    print("=" * 60)
    print("📈 PERFORMANCE TREND PREDICTION")
    print("=" * 60)
    
    # Generate sample performance data
    base_date = datetime.now()
    math_scores = []
    
    # Simulate improving trend with some volatility
    for i in range(12):
        date = base_date - timedelta(days=(11 - i) * 7)
        # Base trend + weekly variation
        base = 60 + i * 1.5  # Improving
        noise = random.uniform(-5, 5)
        # Weekly pattern
        weekly = math.sin(i * math.pi / 2) * 3
        score = max(0, min(100, base + noise + weekly))
        
        weight = 1.0 if i % 4 == 3 else 0.5  # Exams have higher weight
        math_scores.append(PerformanceDataPoint(
            date=date,
            score=round(score, 1),
            subject="Mathematics",
            activity_type="exam" if weight == 1.0 else "quiz",
            weight=weight,
        ))
    
    # Physics - declining trend
    physics_scores = []
    for i in range(10):
        date = base_date - timedelta(days=(9 - i) * 7)
        base = 80 - i * 2  # Declining
        noise = random.uniform(-3, 3)
        score = max(0, min(100, base + noise))
        
        physics_scores.append(PerformanceDataPoint(
            date=date,
            score=round(score, 1),
            subject="Physics",
            activity_type="exam" if i % 3 == 2 else "quiz",
            weight=1.0 if i % 3 == 2 else 0.5,
        ))
    
    # Analyze Mathematics
    print("\n--- Mathematics Trend Analysis ---")
    print(predict_performance_trend(math_scores))
    print("\n--- Physics Trend Analysis ---")
    print(predict_performance_trend(physics_scores))
