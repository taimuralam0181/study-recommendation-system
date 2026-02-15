"""
Knowledge Gap Analysis Module

Identifies missing concepts and knowledge gaps using:
- Performance analysis by topic/subtopic
- Prerequisite mapping between concepts
- Error pattern recognition
- Bayesian knowledge tracing
- Concept dependency graphs

Outputs prioritized list of topics to review.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime
import math
import json


@dataclass(frozen=True)
class TopicScore:
    """Performance score for a specific topic."""
    topic: str
    subject: str
    score: float                    # 0-100
    attempts: int
    last_attempt: datetime
    error_rate: float               # 0-1.0
    avg_time_per_question: float    # seconds
    mastery_level: str              # "Beginner" / "Intermediate" / "Advanced" / "Mastered"


@dataclass(frozen=True)
class ConceptDependency:
    """Dependency between concepts (prerequisite -> dependent)."""
    prerequisite: str
    dependent: str
    strength: float                 # 0-1.0 (strong dependency = must learn first)
    subject: str


@dataclass(frozen=True)
class KnowledgeGap:
    """A identified knowledge gap."""
    topic: str
    subject: str
    gap_severity: float           # 0-1.0 (higher = more urgent)
    current_score: float         # Current performance
    target_score: float          # Desired performance
    estimated_hours: float       # Hours needed to fill gap
    prerequisite_gaps: List[str]  # Dependencies also needed
    recommended_resources: List[str]
    reasoning: str                # Why this was identified


@dataclass(frozen=True)
class KnowledgeGapReport:
    """Complete knowledge gap analysis report."""
    student_id: str
    analysis_date: datetime
    overall_weak_areas: List[KnowledgeGap]
    priority_topics: List[str]     # Top 5 topics to focus on
    subject_breakdown: Dict[str, Dict]
    estimated_total_hours: float
    strengths: List[str]          # Areas performing well
    recommendations: List[str]
    prerequisite_chain: Dict[str, List[str]]  # Topic -> [prerequisites]


@dataclass(frozen=True)
class KnowledgeGapInput:
    """Input data for knowledge gap analysis."""
    current_marks: Dict[str, float]      # {topic: score}
    target_marks: Dict[str, float]       # {topic: target_score}
    syllabus_topics: List[str]           # All topics in syllabus
    recent_quiz_scores: Dict[str, float]  # {topic: quiz_score}
    time_available_hours: float = 10.0    # Available study time


@dataclass(frozen=True)
class KnowledgeGapAnalysisResult:
    """Result of knowledge gap analysis."""
    concept_gaps: List[KnowledgeGap]
    focus_areas: List[str]
    estimated_hours_needed: float


# Subject-specific concept dependencies
PREREQUISITE_MAP = {
    "Mathematics": {
        "calculus": ["algebra", "trigonometry", "limits"],
        "differential_equations": ["calculus", "linear_algebra"],
        "linear_algebra": ["algebra", "geometry"],
        "statistics": ["algebra", "probability"],
        "probability": ["algebra", "counting"],
        "geometry": ["algebra", "trigonometry"],
        "trigonometry": ["algebra", "geometry"],
    },
    "Physics": {
        "mechanics": ["calculus", "vectors"],
        "electromagnetism": ["calculus", "vectors", "physics_mechanics"],
        "thermodynamics": ["calculus", "physics_mechanics"],
        "optics": ["geometry", "trigonometry"],
        "quantum_mechanics": ["calculus", "physics_mechanics", "linear_algebra"],
        "waves_and_oscillations": ["calculus", "trigonometry"],
    },
    "Computer Science": {
        "algorithms": ["programming_fundamentals", "discrete_math"],
        "data_structures": ["programming_fundamentals", "algorithms"],
        "machine_learning": ["linear_algebra", "statistics", "algorithms"],
        "databases": ["data_structures", "programming_fundamentals"],
        "networking": ["programming_fundamentals", "operating_systems"],
        "compilers": ["data_structures", "algorithms", "formal_languages"],
    },
}


def _calculate_topic_score(
    scores: List[float],
    errors: List[bool],
    times: List[float],
) -> TopicScore:
    """Calculate aggregated topic score from individual attempts."""
    if not scores:
        return None
    
    n = len(scores)
    avg_score = sum(scores) / n
    avg_time = sum(times) / n if times else 0
    
    error_count = sum(1 for e in errors if e)
    error_rate = error_count / n if n > 0 else 1.0
    
    # Determine mastery level
    if avg_score >= 90 and n >= 3:
        mastery = "Mastered"
    elif avg_score >= 75 and n >= 2:
        mastery = "Advanced"
    elif avg_score >= 60:
        mastery = "Intermediate"
    else:
        mastery = "Beginner"
    
    return TopicScore(
        topic="unknown",
        subject="unknown",
        score=round(avg_score, 1),
        attempts=n,
        last_attempt=datetime.now(),
        error_rate=round(error_rate, 2),
        avg_time_per_question=round(avg_time, 1),
        mastery_level=mastery,
    )


def _identify_prerequisite_gaps(
    topic: str,
    subject: str,
    all_scores: Dict[str, TopicScore],
) -> List[str]:
    """Identify which prerequisites for a topic are also weak."""
    gaps = []
    
    prereqs = PREREQUISITE_MAP.get(subject, {}).get(topic, [])
    
    for prereq in prereqs:
        if prereq in all_scores:
            if all_scores[prereq].score < 70:
                gaps.append(prereq)
        else:
            # Unknown prerequisite - assume gap
            gaps.append(prereq)
    
    return gaps


def _calculate_gap_severity(
    topic_score: TopicScore,
    prerequisite_gaps: List[str],
) -> float:
    """Calculate how severe a knowledge gap is."""
    # Base severity from low score
    score_severity = max(0, (70 - topic_score.score) / 70)
    
    # Amplify if prerequisites are also weak
    prereq_factor = min(0.3, len(prerequisite_gaps) * 0.1)
    
    # Reduce if student has many attempts (familiar but struggling)
    familiarity_factor = min(0.2, topic_score.attempts * 0.02)
    
    severity = score_severity + prereq_factor - familiarity_factor
    return min(1.0, max(0.0, severity))


def _estimate_learning_time(
    current_score: float,
    target_score: float,
    difficulty: float,  # 1-10
    prerequisites_needed: int,
) -> float:
    """
    Estimate hours needed to fill a knowledge gap.
    """
    score_improvement = target_score - current_score
    
    if score_improvement <= 0:
        return 0
    
    # Base hours per percentage point (varies by difficulty)
    base_hours = difficulty * 0.5
    
    # Diminishing returns for higher scores
    efficiency = 1.0 - (current_score / 200)
    
    hours = score_improvement * base_hours * efficiency
    
    # Add time for prerequisites
    hours += prerequisites_needed * 2.0
    
    return round(hours, 1)


def analyze_knowledge_gaps(
    input_data: KnowledgeGapInput | str,
    topic_scores: Optional[Dict[Tuple[str, str], TopicScore]] = None,
    target_score: float = 75.0,
) -> KnowledgeGapReport:
    """
    Main function to analyze knowledge gaps.
    
    Args:
        input_data: KnowledgeGapInput or student_id
        topic_scores: Dict of {(subject, topic): TopicScore} when student_id is passed
        target_score: Desired mastery level (default 75%)
    
    Returns:
        KnowledgeGapReport with identified gaps and recommendations
    """
    if isinstance(input_data, KnowledgeGapInput):
        student_id = "student"
        topic_scores = {}
        for topic in input_data.syllabus_topics:
            current = input_data.current_marks.get(topic, 0)
            quiz = input_data.recent_quiz_scores.get(topic, current)
            target = input_data.target_marks.get(topic, target_score)
            score = TopicScore(
                topic=topic,
                subject="General",
                score=float(current),
                attempts=1 if topic in input_data.recent_quiz_scores else 0,
                last_attempt=datetime.now(),
                error_rate=round(max(0.0, 1.0 - (quiz / 100)), 2),
                avg_time_per_question=0.0,
                mastery_level="Beginner" if current < target_score else "Advanced",
            )
            topic_scores[("General", topic)] = score
    else:
        student_id = input_data
        if topic_scores is None:
            topic_scores = {}

    gaps: List[KnowledgeGap] = []
    strengths: List[str] = []
    
    # Group by subject
    subjects: Dict[str, Dict[str, TopicScore]] = {}
    for (subject, topic), score in topic_scores.items():
        if subject not in subjects:
            subjects[subject] = {}
        subjects[subject][topic] = score
    
    # Analyze each topic
    all_scores_flat = {}
    for (subject, topic), score in topic_scores.items():
        all_scores_flat[topic] = score
    
    for (subject, topic), score in topic_scores.items():
        # Skip if already mastered
        if score.score >= target_score and score.attempts >= 2:
            strengths.append(f"{topic.title()} in {subject}")
            continue
        
        # Find prerequisite gaps
        prereq_gaps = _identify_prerequisite_gaps(
            topic, subject, all_scores_flat
        )
        
        # Calculate severity
        severity = _calculate_gap_severity(score, prereq_gaps)
        
        # Estimate learning time
        difficulty_map = {
            "Mathematics": 7,
            "Physics": 6,
            "Computer Science": 8,
        }
        difficulty = difficulty_map.get(subject, 5)
        hours = _estimate_learning_time(
            score.score, target_score, difficulty, len(prereq_gaps)
        )
        
        # Generate reasoning
        reasoning_parts = []
        if score.score < 60:
            reasoning_parts.append(f"Current score ({score.score:.0f}%) is below target")
        if score.error_rate > 0.3:
            reasoning_parts.append(f"High error rate ({score.error_rate:.0%})")
        if len(prereq_gaps) > 0:
            reasoning_parts.append(f"Missing prerequisites: {', '.join(prereq_gaps)}")
        if score.attempts < 2:
            reasoning_parts.append("Limited practice history")
        
        reasoning = ". ".join(reasoning_parts) if reasoning_parts else "Below target performance"
        
        # Generate recommendations
        resources = _get_resources_for_topic(subject, topic)
        
        if severity > 0:
            gap = KnowledgeGap(
                topic=topic,
                subject=subject,
                gap_severity=round(severity, 3),
                current_score=score.score,
                target_score=target_score,
                estimated_hours=hours,
                prerequisite_gaps=prereq_gaps,
                recommended_resources=resources,
                reasoning=reasoning,
            )
            gaps.append(gap)
    
    # Sort by severity
    gaps.sort(key=lambda x: x.gap_severity, reverse=True)
    
    # Get priority topics
    priority_topics = [g.topic for g in gaps[:5]]
    
    # Calculate total hours
    total_hours = sum(g.estimated_hours for g in gaps)
    
    # Subject breakdown
    subject_breakdown: Dict[str, Dict] = {}
    for gap in gaps:
        if gap.subject not in subject_breakdown:
            subject_breakdown[gap.subject] = {
                "gaps_count": 0,
                "total_hours": 0,
                "avg_severity": 0,
            }
        subject_breakdown[gap.subject]["gaps_count"] += 1
        subject_breakdown[gap.subject]["total_hours"] += gap.estimated_hours
        subject_breakdown[gap.subject]["avg_severity"] += gap.gap_severity
    
    for subj in subject_breakdown:
        count = subject_breakdown[subj]["gaps_count"]
        subject_breakdown[subj]["avg_severity"] = round(
            subject_breakdown[subj]["avg_severity"] / count, 2
        )
    
    # Build prerequisite chain
    prerequisite_chain: Dict[str, List[str]] = {}
    for subject, topic_map in PREREQUISITE_MAP.items():
        for topic, prereqs in topic_map.items():
            prerequisite_chain[topic] = prereqs
    
    # Generate overall recommendations
    recommendations = _generate_gap_recommendations(gaps, priority_topics)
    
    return KnowledgeGapReport(
        student_id=student_id,
        analysis_date=datetime.now(),
        overall_weak_areas=gaps,
        priority_topics=priority_topics,
        subject_breakdown=subject_breakdown,
        estimated_total_hours=round(total_hours, 1),
        strengths=strengths,
        recommendations=recommendations,
        prerequisite_chain=prerequisite_chain,
    )


def _get_resources_for_topic(subject: str, topic: str) -> List[str]:
    """Get recommended resources for a topic."""
    resource_map = {
        "Mathematics": {
            "calculus": ["Khan Academy Calculus", "3Blue1Brown Calculus Series", " Paul's Online Math Notes"],
            "algebra": ["Khan Academy Algebra", "PatrickJMT", "MathIsFun Algebra"],
            "statistics": ["Khan Academy Statistics", "StatQuest", "Seeing Theory"],
            "linear_algebra": ["3Blue1Brown Linear Algebra", "Gilbert Strang Lectures", "Khan Academy"],
        },
        "Physics": {
            "mechanics": ["Khan Academy Physics", "Flipping Physics", "Michel van Biezen"],
            "electromagnetism": ["PhysicsGirl", "ElectroBOOM", "MIT OpenCourseWare"],
            "thermodynamics": ["ThermoID", "LearnChemE", "Khan Academy"],
        },
        "Computer Science": {
            "algorithms": ["CLRS Introduction to Algorithms", "GeeksforGeeks", "LeetCode"],
            "data_structures": ["CS50", "Interview Cake", "Visualgo"],
            "machine_learning": ["Andrew Ng's ML Course", "Fast.ai", "Scikit-learn Documentation"],
            "programming_fundamentals": ["CS50", "Codecademy", "FreeCodeCamp"],
        },
    }
    
    return resource_map.get(subject, {}).get(topic, [
        "Khan Academy",
        "Coursera",
        "edX",
        "YouTube tutorials",
    ])


def _generate_gap_recommendations(
    gaps: List[KnowledgeGap],
    priority_topics: List[str],
) -> List[str]:
    """Generate overall study recommendations."""
    recommendations = []
    
    if not gaps:
        recommendations.append("✅ Excellent! No significant knowledge gaps detected.")
        recommendations.append("💪 Continue with advanced topics and maintenance review.")
        return recommendations
    
    # Prioritization recommendation
    if priority_topics:
        recommendations.append(
            f"🎯 Focus first on: {', '.join(p.title() for p in priority_topics[:3])}"
        )
    
    # Prerequisite strategy
    prereq_heavy = [g for g in gaps if len(g.prerequisite_gaps) > 0]
    if prereq_heavy:
        recommendations.append(
            "📚 Address prerequisite gaps before advanced topics for faster progress"
        )
    
    # Time management
    total_hours = sum(g.estimated_hours for g in gaps)
    if total_hours > 10:
        recommendations.append(
            f"⏰ Total estimated study time: {total_hours:.0f} hours. "
            "Consider spreading across 2-3 weeks."
        )
    elif total_hours > 0:
        recommendations.append(
            f"⏱️ Total estimated study time: {total_hours:.0f} hours. "
            "Plan daily 1-2 hour focused sessions."
        )
    
    # Subject balance
    subjects = set(g.subject for g in gaps)
    if len(subjects) > 1:
        recommendations.append(
            "🔄 Balance study time across subjects for optimal retention"
        )
    
    # Active learning
    recommendations.append(
        "🧪 Practice with active recall and spaced repetition"
    )
    
    return recommendations


def create_topic_graph(
    subjects: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """
    Create a visualization-friendly topic dependency graph.
    
    Returns:
        Dict mapping each topic to its prerequisites
    """
    graph = {}
    
    for subject, topics in subjects.items():
        for topic in topics:
            prereqs = PREREQUISITE_MAP.get(subject, {}).get(topic, [])
            graph[topic] = prereqs
    
    return graph


def find_learning_path(
    target_topic: str,
    current_scores: Dict[str, float],  # {topic: score}
    subject: str,
) -> List[str]:
    """
    Find the optimal learning path to master a target topic.
    
    Returns:
        Ordered list of topics to study in sequence
    """
    path = []
    visited = set()
    
    def add_prerequisites(topic: str):
        if topic in visited:
            return
        prereqs = PREREQUISITE_MAP.get(subject, {}).get(topic, [])
        for prereq in prereqs:
            if prereq not in visited:
                # Check if already mastered
                if current_scores.get(prereq, 0) >= 80:
                    visited.add(prereq)
                else:
                    add_prerequisites(prereq)
                    if prereq not in path:
                        path.append(prereq)
        if topic not in path:
            path.append(topic)
        visited.add(topic)
    
    add_prerequisites(target_topic)
    
    return path


def export_gap_report_json(report: KnowledgeGapReport) -> str:
    """Export knowledge gap report to JSON."""
    def serialize_datetime(dt: datetime) -> str:
        return dt.isoformat()
    
    report_dict = {
        "student_id": report.student_id,
        "analysis_date": serialize_datetime(report.analysis_date),
        "overall_weak_areas": [
            {
                "topic": g.topic,
                "subject": g.subject,
                "gap_severity": g.gap_severity,
                "current_score": g.current_score,
                "target_score": g.target_score,
                "estimated_hours": g.estimated_hours,
                "prerequisite_gaps": g.prerequisite_gaps,
                "recommended_resources": g.recommended_resources,
                "reasoning": g.reasoning,
            }
            for g in report.overall_weak_areas
        ],
        "priority_topics": report.priority_topics,
        "subject_breakdown": report.subject_breakdown,
        "estimated_total_hours": report.estimated_total_hours,
        "strengths": report.strengths,
        "recommendations": report.recommendations,
        "prerequisite_chain": report.prerequisite_chain,
    }
    
    return json.dumps(report_dict, indent=2)


# Demo / Testing
if __name__ == "__main__":
    print("=" * 60)
    print("📚 KNOWLEDGE GAP ANALYSIS")
    print("=" * 60)
    
    # Sample topic scores
    sample_scores = {
        ("Mathematics", "calculus"): TopicScore(
            topic="calculus",
            subject="Mathematics",
            score=55,
            attempts=5,
            last_attempt=datetime.now(),
            error_rate=0.45,
            avg_time_per_question=120,
            mastery_level="Beginner",
        ),
        ("Mathematics", "algebra"): TopicScore(
            topic="algebra",
            subject="Mathematics",
            score=78,
            attempts=8,
            last_attempt=datetime.now(),
            error_rate=0.22,
            avg_time_per_question=90,
            mastery_level="Advanced",
        ),
        ("Physics", "mechanics"): TopicScore(
            topic="mechanics",
            subject="Physics",
            score=42,
            attempts=3,
            last_attempt=datetime.now(),
            error_rate=0.58,
            avg_time_per_question=150,
            mastery_level="Beginner",
        ),
        ("Computer Science", "algorithms"): TopicScore(
            topic="algorithms",
            subject="Computer Science",
            score=65,
            attempts=6,
            last_attempt=datetime.now(),
            error_rate=0.35,
            avg_time_per_question=180,
            mastery_level="Intermediate",
        ),
        ("Computer Science", "data_structures"): TopicScore(
            topic="data_structures",
            subject="Computer Science",
            score=72,
            attempts=7,
            last_attempt=datetime.now(),
            error_rate=0.28,
            avg_time_per_question=160,
            mastery_level="Advanced",
        ),
    }
    
    # Run analysis
    report = analyze_knowledge_gaps(
        student_id="student_001",
        topic_scores=sample_scores,
        target_score=75.0,
    )
    
    print(f"\n📊 Analysis Date: {report.analysis_date.strftime('%Y-%m-%d %H:%M')}")
    print(f"📚 Total Topics Analyzed: {len(sample_scores)}")
    print(f"⚠️ Knowledge Gaps Found: {len(report.overall_weak_areas)}")
    print(f"💪 Strengths Identified: {len(report.strengths)}")
    print(f"⏰ Estimated Total Study Time: {report.estimated_total_hours} hours")
    
    print("\n" + "=" * 60)
    print("🎯 PRIORITY TOPICS TO FOCUS ON")
    print("=" * 60)
    
    for i, gap in enumerate(report.overall_weak_areas[:5], 1):
        print(f"\n{i}. {gap.topic.title()} ({gap.subject})")
        print(f"   Severity: {'🔴' if gap.gap_severity > 0.5 else '🟡' if gap.gap_severity > 0.3 else '🟢'} "
              f"{gap.gap_severity:.0%}")
        print(f"   Current Score: {gap.current_score:.0f}% → Target: {gap.target_score:.0f}%")
        print(f"   Est. Time: {gap.estimated_hours:.1f} hours")
        if gap.prerequisite_gaps:
            print(f"   Prerequisites Needed: {', '.join(gap.prerequisite_gaps)}")
        print(f"   Reasoning: {gap.reasoning}")
    
    print("\n" + "=" * 60)
    print("💪 STRENGTHS")
    print("=" * 60)
    for strength in report.strengths:
        print(f"   ✅ {strength}")
    
    print("\n" + "=" * 60)
    print("📋 RECOMMENDATIONS")
    print("=" * 60)
    for rec in report.recommendations:
        print(f"   {rec}")
    
    print("\n" + "=" * 60)
    print("📈 SUBJECT BREAKDOWN")
    print("=" * 60)
    for subject, data in report.subject_breakdown.items():
        print(f"\n{subject}:")
        print(f"   Gaps: {data['gaps_count']}")
        print(f"   Total Hours: {data['total_hours']:.1f}")
        print(f"   Avg Severity: {data['avg_severity']:.0%}")
    
    # Find learning path example
    print("\n" + "=" * 60)
    print("🛤️ LEARNING PATH EXAMPLE")
    print("=" * 60)
    
    current = {"algebra": 78, "programming_fundamentals": 60}
    path = find_learning_path("machine_learning", current, "Computer Science")
    print(f"\nPath to Machine Learning: {' → '.join(p.replace('_', ' ').title() for p in path)}")
    
    print("\n" + "=" * 60)
