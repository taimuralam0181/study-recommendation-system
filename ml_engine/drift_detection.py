"""
Model Drift Detection Module

Monitors ML model performance over time and detects:
- Prediction drift (distribution shift in predictions)
- Feature drift (distribution shift in input features)
- Performance drift (degradation in accuracy/metrics)
- Concept drift (changing relationships between features and target)

Provides alerts and recommendations when drift is detected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import math
import json


@dataclass(frozen=True)
class DistributionStats:
    """Statistics for a feature distribution."""
    mean: float
    std: float
    count: int = 0
    min_value: float = 0.0
    max_value: float = 0.0
    median: float = 0.0
    q25: float = 0.0
    q75: float = 0.0


@dataclass(frozen=True)
class DataDriftResult:
    """Lightweight drift result for simple comparisons."""
    drift_detected: bool
    mean_change: float
    std_change: float


@dataclass(frozen=True)
class DriftMetrics:
    """Drift detection metrics for a single feature."""
    feature: str
    ks_statistic: float          # Kolmogorov-Smirnov statistic
    ks_p_value: float            # P-value for KS test
    psi: float                   # Population Stability Index
    kl_divergence: float         # KL divergence
    wasserstein_distance: float  # Earth Mover's Distance
    drift_detected: bool
    drift_severity: str          # "none" | "low" | "medium" | "high"
    timestamp: datetime


@dataclass(frozen=True)
class PerformanceMetrics:
    """Model performance metrics over time."""
    date: datetime
    accuracy: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    f1_score: Optional[float]
    mse: Optional[float]
    mae: Optional[float]
    sample_size: int


@dataclass(frozen=True)
class DriftReport:
    """Complete drift detection report."""
    model_name: str
    analysis_date: datetime
    feature_drift: List[DriftMetrics]
    overall_drift_score: float
    drift_level: str
    performance_metrics: List[PerformanceMetrics]
    recent_performance_change: float
    alerts: List[str]
    recommendations: List[str]


def calculate_stats(data: List[float]) -> DistributionStats:
    """Calculate distribution statistics."""
    if not data:
        return DistributionStats(0, 0, 0, 0, 0, 0, 0, 0)
    
    sorted_data = sorted(data)
    n = len(sorted_data)
    
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / n
    std = math.sqrt(variance)
    
    return DistributionStats(
        mean=mean,
        std=std,
        count=n,
        min_value=sorted_data[0],
        max_value=sorted_data[-1],
        median=sorted_data[n // 2],
        q25=sorted_data[n // 4],
        q75=sorted_data[3 * n // 4],
    )


def detect_data_drift(
    reference: DistributionStats,
    current: DistributionStats,
    mean_threshold: float = 0.2,
    std_threshold: float = 0.3,
) -> DataDriftResult:
    """
    Simple drift detector for tests and lightweight usage.
    """
    mean_diff = abs(current.mean - reference.mean)
    std_diff = abs(current.std - reference.std)
    mean_rel = mean_diff / (abs(reference.mean) + 1e-9)
    std_rel = std_diff / (abs(reference.std) + 1e-9)
    drift = mean_rel > mean_threshold or std_rel > std_threshold
    return DataDriftResult(
        drift_detected=drift,
        mean_change=round(mean_rel, 4),
        std_change=round(std_rel, 4),
    )


def kolmogorov_smirnov(
    reference: List[float],
    current: List[float],
) -> Tuple[float, float]:
    """
    Calculate Kolmogorov-Smirnov statistic and p-value.
    Tests if two samples come from the same distribution.
    """
    if not reference or not current:
        return 0.0, 1.0
    
    # Create empirical CDFs
    ref_sorted = sorted(reference)
    curr_sorted = sorted(current)
    
    n1, n2 = len(ref_sorted), len(curr_sorted)
    
    # Calculate KS statistic
    ks_stat = 0.0
    i = j = 0
    
    while i < n1 and j < n2:
        d1 = ref_sorted[i]
        d2 = curr_sorted[j]
        
        # Approximate CDF values
        cdf1 = (i + 1) / n1
        cdf2 = (j + 1) / n2
        
        ks_stat = max(ks_stat, abs(cdf1 - cdf2))
        
        if d1 < d2:
            i += 1
        else:
            j += 1
    
    # Approximate p-value
    # Using asymptotic approximation
    n_eff = (n1 * n2) / (n1 + n2)
    sqrt_n = math.sqrt(n_eff)
    
    # Standard KS distribution approximation
    lambda_stat = ks_stat * (sqrt_n + 0.12 + 0.11 / sqrt_n)
    
    # Approximate p-value (Kolmogorov distribution)
    if lambda_stat > 0:
        # Sum of exp(-2 * k^2 * lambda^2) for k=1 to infinity
        p_value = 0.0
        for k in range(1, 20):  # Sufficient approximation
            p_value += 2 * (-1) ** (k - 1) * math.exp(-2 * k ** 2 * lambda_stat ** 2)
    else:
        p_value = 1.0
    
    p_value = max(0.0, min(1.0, p_value))
    
    return ks_stat, p_value


def population_stability_index(
    reference: List[float],
    current: List[float],
    buckets: int = 10,
) -> float:
    """
    Calculate Population Stability Index (PSI).
    
    PSI < 0.1: No significant change
    0.1 <= PSI < 0.2: Moderate change (monitor)
    PSI >= 0.2: Significant change (investigate)
    """
    if not reference or not current:
        return 0.0
    
    # Create buckets
    all_data = reference + current
    min_val, max_val = min(all_data), max(all_data)
    
    if min_val == max_val:
        return 0.0
    
    bucket_width = (max_val - min_val) / buckets
    
    # Count in each bucket
    ref_counts = [0] * buckets
    curr_counts = [0] * buckets
    
    for val in reference:
        bucket = int((val - min_val) / bucket_width)
        bucket = min(bucket, buckets - 1)
        ref_counts[bucket] += 1
    
    for val in current:
        bucket = int((val - min_val) / bucket_width)
        bucket = min(bucket, buckets - 1)
        curr_counts[bucket] += 1
    
    # Calculate percentages (with smoothing)
    ref_total = sum(ref_counts) or 1
    curr_total = sum(curr_counts) or 1
    
    psi = 0.0
    for i in range(buckets):
        ref_pct = (ref_counts[i] + 0.5) / (ref_total + 0.5 * buckets)
        curr_pct = (curr_counts[i] + 0.5) / (curr_total + 0.5 * buckets)
        
        if ref_pct > 0 and curr_pct > 0:
            psi += (curr_pct - ref_pct) * math.log(curr_pct / ref_pct)
    
    return psi


def kl_divergence(
    reference: List[float],
    current: List[float],
    buckets: int = 10,
) -> float:
    """
    Calculate KL Divergence from reference to current.
    Measures how different current distribution is from reference.
    """
    if not reference or not current:
        return 0.0
    
    # Create buckets
    all_data = reference + current
    min_val, max_val = min(all_data), max(all_data)
    
    if min_val == max_val:
        return 0.0
    
    bucket_width = (max_val - min_val) / buckets
    
    # Calculate distributions
    ref_counts = [0] * buckets
    curr_counts = [0] * buckets
    
    for val in reference:
        bucket = int((val - min_val) / bucket_width)
        bucket = min(bucket, buckets - 1)
        ref_counts[bucket] += 1
    
    for val in current:
        bucket = int((val - min_val) / bucket_width)
        bucket = min(bucket, buckets - 1)
        curr_counts[bucket] += 1
    
    # Add smoothing
    ref_total = sum(ref_counts) + buckets * 0.1
    curr_total = sum(curr_counts) + buckets * 0.1
    
    kl = 0.0
    for i in range(buckets):
        ref_pct = ref_counts[i] / ref_total
        curr_pct = curr_counts[i] / curr_total
        
        if ref_pct > 0 and curr_pct > 0:
            kl += ref_pct * math.log(ref_pct / curr_pct)
    
    return kl


def wasserstein_distance(
    reference: List[float],
    current: List[float],
) -> float:
    """
    Calculate Earth Mover's Distance (Wasserstein-1).
    Measures minimum work to transform one distribution to another.
    """
    if not reference or not current:
        return 0.0
    
    ref_sorted = sorted(reference)
    curr_sorted = sorted(current)
    n = len(ref_sorted)
    m = len(current)
    
    # Calculate cumulative distributions
    ref_cdf = [(i + 1) / n for i in range(n)]
    curr_cdf = [(i + 1) / m for i in range(m)]
    
    # Interpolate and integrate
    total_distance = 0.0
    
    i = j = 0
    while i < n and j < m:
        x1 = ref_sorted[i]
        x2 = curr_sorted[j]
        
        if ref_cdf[i] < curr_cdf[j]:
            total_distance += abs(x1 - x2) * (ref_cdf[i] - (curr_cdf[j - 1] if j > 0 else 0))
            i += 1
        else:
            total_distance += abs(x1 - x2) * (curr_cdf[j] - (ref_cdf[i - 1] if i > 0 else 0))
            j += 1
    
    return total_distance


def detect_feature_drift(
    feature_name: str,
    reference_data: List[float],
    current_data: List[float],
    threshold_psi: float = 0.1,
    threshold_ks: float = 0.3,
) -> DriftMetrics:
    """
    Detect drift for a single feature.
    """
    # Calculate metrics
    ks_stat, ks_p_value = kolmogorov_smirnov(reference_data, current_data)
    psi = population_stability_index(reference_data, current_data)
    kl = kl_divergence(reference_data, current_data)
    wd = wasserstein_distance(reference_data, current_data)
    
    # Determine if drift is detected
    drift_detected = psi > threshold_psi or ks_stat > threshold_ks
    
    # Determine severity
    if psi > 0.3 or ks_stat > 0.5:
        severity = "high"
    elif psi > 0.2 or ks_stat > 0.35:
        severity = "medium"
    elif psi > 0.1 or ks_stat > 0.25:
        severity = "low"
    else:
        severity = "none"
    
    return DriftMetrics(
        feature=feature_name,
        ks_statistic=round(ks_stat, 4),
        ks_p_value=round(ks_p_value, 4),
        psi=round(psi, 4),
        kl_divergence=round(kl, 4),
        wasserstein_distance=round(wd, 4),
        drift_detected=drift_detected,
        drift_severity=severity,
        timestamp=datetime.now(),
    )


def detect_concept_drift(
    predictions: List[float],
    actuals: List[float],
    window_size: int = 100,
) -> Tuple[float, bool]:
    """
    Detect concept drift by comparing recent vs. historical performance.
    
    Returns:
        (drift_score, drift_detected)
    """
    if len(predictions) < window_size:
        return 0.0, False
    
    # Split into historical and recent
    historical = predictions[:-window_size]
    recent = predictions[-window_size:]
    
    # Calculate error rates
    if len(actuals) >= window_size:
        hist_actuals = actuals[:-window_size]
        rec_actuals = actuals[-window_size:]
        
        hist_errors = sum(1 for p, a in zip(historical, hist_actuals) if p != a) / len(historical)
        rec_errors = sum(1 for p, a in zip(recent, rec_actuals) if p != a) / len(recent)
        
        # Drift score based on error rate change
        drift_score = abs(rec_errors - hist_errors) / (hist_errors + 0.01)
    else:
        drift_score = 0.0
    
    drift_detected = drift_score > 0.2  # 20% change threshold
    
    return drift_score, drift_detected


def analyze_performance_drift(
    metrics_history: List[PerformanceMetrics],
) -> Tuple[float, List[str]]:
    """
    Analyze performance metrics for drift.
    
    Returns:
        (overall_change_score, alerts)
    """
    alerts = []
    
    if len(metrics_history) < 2:
        return 0.0, alerts
    
    # Compare recent vs. historical
    recent = metrics_history[-5:]  # Last 5 entries
    historical = metrics_history[:-5]
    
    if not recent or not historical:
        return 0.0, alerts
    
    # Calculate average metrics
    def avg_metric(metrics_list, attr_name):
        values = [getattr(m, attr_name) for m in metrics_list if getattr(m, attr_name) is not None]
        return sum(values) / len(values) if values else None
    
    recent_acc = avg_metric(recent, 'accuracy')
    hist_acc = avg_metric(historical, 'accuracy')
    
    recent_f1 = avg_metric(recent, 'f1_score')
    hist_f1 = avg_metric(historical, 'f1_score')
    
    # Calculate changes
    changes = []
    if recent_acc is not None and hist_acc is not None and hist_acc > 0:
        acc_change = (recent_acc - hist_acc) / hist_acc
        changes.append(('accuracy', acc_change))
    
    if recent_f1 is not None and hist_f1 is not None and hist_f1 > 0:
        f1_change = (recent_f1 - hist_f1) / hist_f1
        changes.append(('f1_score', f1_change))
    
    # Overall change score
    total_change = sum(abs(c[1]) for c in changes)
    avg_change = total_change / len(changes) if changes else 0.0
    
    # Generate alerts
    for metric, change in changes:
        if change < -0.1:  # 10% decline
            alerts.append(f"⚠️ {metric.upper()} dropped by {abs(change):.1%}")
        elif change > 0.1:
            alerts.append(f"✅ {metric.upper()} improved by {change:.1%}")
    
    return avg_change, alerts


def generate_drift_report(
    model_name: str,
    reference_features: Dict[str, List[float]],
    current_features: Dict[str, List[float]],
    performance_history: Optional[List[PerformanceMetrics]] = None,
) -> DriftReport:
    """
    Generate comprehensive drift detection report.
    """
    # Detect feature drift
    feature_drift = []
    for feature in set(reference_features.keys()) | set(current_features.keys()):
        ref_data = reference_features.get(feature, [])
        curr_data = current_features.get(feature, [])
        
        if ref_data and curr_data:
            drift = detect_feature_drift(feature, ref_data, curr_data)
            feature_drift.append(drift)
    
    # Calculate overall drift score
    if feature_drift:
        psi_values = [d.psi for d in feature_drift]
        ks_values = [d.ks_statistic for d in feature_drift]
        
        overall_score = (sum(psi_values) / len(psi_values) + 
                        sum(ks_values) / len(ks_values)) / 2
    else:
        overall_score = 0.0
    
    # Determine drift level
    if overall_score > 0.3:
        drift_level = "high"
    elif overall_score > 0.15:
        drift_level = "medium"
    elif overall_score > 0.05:
        drift_level = "low"
    else:
        drift_level = "none"
    
    # Analyze performance drift
    recent_change = 0.0
    performance_alerts = []
    if performance_history:
        recent_change, performance_alerts = analyze_performance_drift(performance_history)
    
    # Generate alerts
    alerts = []
    
    # Feature drift alerts
    for drift in feature_drift:
        if drift.drift_severity == "high":
            alerts.append(f"🔴 HIGH DRIFT in {drift.feature}: PSI={drift.psi:.3f}, KS={drift.ks_statistic:.3f}")
        elif drift.drift_severity == "medium":
            alerts.append(f"🟡 MEDIUM DRIFT in {drift.feature}: PSI={drift.psi:.3f}")
    
    # Performance drift alerts
    alerts.extend(performance_alerts)
    
    # Generate recommendations
    recommendations = []
    
    if drift_level in ["high", "medium"]:
        recommendations.append("📊 Consider retraining the model with recent data")
        recommendations.append("🔍 Investigate the root cause of drift")
    
    if any(d.drift_severity == "high" for d in feature_drift):
        recommendations.append("⚠️ High-priority: Feature distributions have shifted significantly")
    
    if recent_change > 0.1:
        recommendations.append("📉 Model performance is degrading - schedule retraining")
    
    if drift_level == "none":
        recommendations.append("✅ Model is performing stably - no action needed")
        recommendations.append("📅 Continue monitoring on weekly basis")
    
    return DriftReport(
        model_name=model_name,
        analysis_date=datetime.now(),
        feature_drift=sorted(feature_drift, key=lambda x: x.psi, reverse=True),
        overall_drift_score=round(overall_score, 4),
        drift_level=drift_level,
        performance_metrics=performance_history or [],
        recent_performance_change=round(recent_change, 4),
        alerts=alerts,
        recommendations=recommendations,
    )


def export_drift_report_json(report: DriftReport) -> str:
    """Export drift report to JSON."""
    def serialize_datetime(dt: datetime) -> str:
        return dt.isoformat()
    
    report_dict = {
        "model_name": report.model_name,
        "analysis_date": serialize_datetime(report.analysis_date),
        "overall_drift_score": report.overall_drift_score,
        "drift_level": report.drift_level,
        "feature_drift": [
            {
                "feature": d.feature,
                "ks_statistic": d.ks_statistic,
                "ks_p_value": d.ks_p_value,
                "psi": d.psi,
                "kl_divergence": d.kl_divergence,
                "wasserstein_distance": d.wasserstein_distance,
                "drift_detected": d.drift_detected,
                "drift_severity": d.drift_severity,
                "timestamp": serialize_datetime(d.timestamp),
            }
            for d in report.feature_drift
        ],
        "performance_metrics": [
            {
                "date": serialize_datetime(m.date),
                "accuracy": m.accuracy,
                "precision": m.precision,
                "recall": m.recall,
                "f1_score": m.f1_score,
                "mse": m.mse,
                "mae": m.mae,
                "sample_size": m.sample_size,
            }
            for m in report.performance_metrics
        ],
        "recent_performance_change": report.recent_performance_change,
        "alerts": report.alerts,
        "recommendations": report.recommendations,
    }
    
    return json.dumps(report_dict, indent=2)


# Demo / Testing
if __name__ == "__main__":
    print("=" * 60)
    print("📊 MODEL DRIFT DETECTION REPORT")
    print("=" * 60)
    
    # Sample data - reference (historical) distribution
    reference_data = {
        "midterm_marks": [22, 24, 21, 25, 23, 22, 24, 20, 23, 25] * 10,
        "assignment_marks": [4.0, 4.2, 3.8, 4.5, 4.0, 3.9, 4.3, 4.1, 4.0, 4.4] * 10,
        "attendance_percentage": [85, 90, 88, 92, 87, 89, 91, 86, 88, 93] * 10,
    }
    
    # Current data (slightly shifted distribution)
    current_data = {
        "midterm_marks": [18, 20, 19, 21, 17, 18, 20, 16, 19, 21] * 10,  # Lower
        "assignment_marks": [3.5, 3.8, 3.2, 4.0, 3.5, 3.4, 3.9, 3.7, 3.5, 4.0] * 10,  # Lower
        "attendance_percentage": [80, 85, 82, 88, 78, 81, 86, 79, 83, 89] * 10,  # Lower
    }
    
    # Sample performance history
    perf_history = [
        PerformanceMetrics(
            date=datetime.now() - timedelta(days=30),
            accuracy=0.85,
            precision=0.82,
            recall=0.80,
            f1_score=0.81,
            mse=0.15,
            mae=0.12,
            sample_size=100,
        ),
        PerformanceMetrics(
            date=datetime.now() - timedelta(days=20),
            accuracy=0.84,
            precision=0.81,
            recall=0.79,
            f1_score=0.80,
            mse=0.16,
            mae=0.13,
            sample_size=120,
        ),
        PerformanceMetrics(
            date=datetime.now() - timedelta(days=10),
            accuracy=0.78,
            precision=0.75,
            recall=0.72,
            f1_score=0.73,
            mse=0.22,
            mae=0.18,
            sample_size=110,
        ),
    ]
    
    # Generate drift report
    report = generate_drift_report(
        model_name="Academic Performance Predictor",
        reference_features=reference_data,
        current_features=current_data,
        performance_history=perf_history,
    )
    
    print(f"\n📊 Model: {report.model_name}")
    print(f"   Analysis Date: {report.analysis_date.strftime('%Y-%m-%d %H:%M')}")
    
    print(f"\n📈 Overall Drift Score: {report.overall_drift_score:.4f}")
    print(f"   Drift Level: {report.drift_level.upper()}")
    
    print(f"\n🔍 Feature Drift Analysis:")
    for drift in report.feature_drift:
        status = "🔴" if drift.drift_severity == "high" else "🟡" if drift.drift_severity == "medium" else "🟢"
        print(f"   {status} {drift.feature}:")
        print(f"      PSI: {drift.psi:.4f} | KS: {drift.ks_statistic:.4f} | KL: {drift.kl_divergence:.4f}")
    
    print(f"\n⚠️ Alerts:")
    for alert in report.alerts:
        print(f"   {alert}")
    
    print(f"\n💡 Recommendations:")
    for rec in report.recommendations:
        print(f"   {rec}")
    
    # Show recent performance trend
    print(f"\n📉 Performance Trend:")
    for m in report.performance_metrics:
        print(f"   {m.date.strftime('%Y-%m-%d')}: Accuracy={m.accuracy:.1%} | F1={m.f1_score:.1%}")
    
    print("\n" + "=" * 60)
