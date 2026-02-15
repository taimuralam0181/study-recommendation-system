"""
Learning Style Detection Model

Detects student learning preferences using the VARK model:
- Visual (V): Diagrams, charts, videos, spatial understanding
- Auditory (A): Listening, discussions, verbal explanations
- Reading (R): Text, written instructions, notes
- Kinesthetic (K): Hands-on, practice, experiments, real-world examples

Uses behavioral signals and questionnaire data to classify learning style.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import random
from enum import Enum


class LearningStyle(Enum):
    VISUAL = "Visual"
    AUDITORY = "Auditory"
    READING = "Reading"
    KINESTHETIC = "Kinesthetic"


@dataclass(frozen=True)
class LearningStyleFeatures:
    """Behavioral features for learning style detection."""
    # Content preference scores (0-1)
    video_watch_ratio: float           # Videos watched vs total content
    diagram_view_ratio: float           # Diagrams/charts viewed
    text_read_ratio: float              # Text content read
    audio_listen_ratio: float           # Audio content consumed
    
    # Interaction patterns
    quiz_performance_by_type: Dict[str, float]  # {video: 0.8, text: 0.6, ...}
    avg_session_duration: float        # Minutes per session
    study_time_preference: str         # morning/afternoon/evening/night
    
    # Engagement metrics
    completion_rate_by_format: Dict[str, float]  # Format-specific completion
    revisit_rate: float                # How often material is reviewed
    practice_problem_rate: float      # Hands-on problem solving ratio
    
    # Subject preferences
    best_performed_subjects: List[str]
    preferred_activities: List[str]    # reading, listening, watching, doing
    
    # Self-reported preferences (optional)
    self_reported_style: Optional[str] = None  # If user took VARK questionnaire


@dataclass(frozen=True)
class LearningStyleResult:
    """Learning style detection result."""
    primary_style: str                 # V/A/R/K
    secondary_style: Optional[str]     # Second preference (if score difference > 0.1)
    scores: Dict[str, float]          # {Visual: 0.7, Auditory: 0.2, ...}
    confidence: float                 # 0-1.0
    recommended_formats: List[str]   # Best content formats for this learner
    study_tips: List[str]            # Personalized study strategies
    explanation: str                 # Why this style was detected


@dataclass(frozen=True)
class VARKQuestion:
    """Single VARK questionnaire question."""
    question_id: int
    question_text: str
    options: Dict[str, str]          # {a: "Option A", b: "Option B", ...}
    style_mapping: Dict[str, str]    # {a: "Visual", b: "Auditory", ...}


# Default VARK questionnaire
VARK_QUESTIONS = [
    VARKQuestion(
        question_id=1,
        question_text="You are learning a new skill at work. How do you prefer to learn?",
        options={
            "a": "Look at diagrams and charts that show how it's done",
            "b": "Listen to someone explain it verbally",
            "c": "Read the written instructions and procedures",
            "d": "Watch a demonstration of the skill",
        },
        style_mapping={"a": "Visual", "b": "Auditory", "c": "Reading", "d": "Kinesthetic"},
    ),
    VARKQuestion(
        question_id=2,
        question_text="You're planning a vacation. What helps you most?",
        options={
            "a": "Looking at photos and videos of the destination",
            "b": "Talking to someone who's been there",
            "c": "Reading a detailed travel guide/book",
            "d": "Just going and exploring when you get there",
        },
        style_mapping={"a": "Visual", "b": "Auditory", "c": "Reading", "d": "Kinesthetic"},
    ),
    VARKQuestion(
        question_id=3,
        question_text="When solving a difficult problem, you prefer to:",
        options={
            "a": "Draw a diagram or sketch to visualize",
            "b": "Talk through the problem aloud",
            "c": "Write out the steps in detail",
            "d": "Try different approaches by doing",
        },
        style_mapping={"a": "Visual", "b": "Auditory", "c": "Reading", "d": "Kinesthetic"},
    ),
    VARKQuestion(
        question_id=4,
        question_text="When learning a new game, you prefer:",
        options={
            "a": "Looking at the board/pieces and imagining moves",
            "b": "Having someone explain the rules verbally",
            "c": "Reading the rulebook carefully",
            "d": "Playing the game and learning by doing",
        },
        style_mapping={"a": "Visual", "b": "Auditory", "c": "Reading", "d": "Kinesthetic"},
    ),
    VARKQuestion(
        question_id=5,
        question_text="You remember things best when you:",
        options={
            "a": "See images and visual associations",
            "b": "Hear the information spoken",
            "c": "Write or read the information",
            "d": "Do activities related to the information",
        },
        style_mapping={"a": "Visual", "b": "Auditory", "c": "Reading", "d": "Kinesthetic"},
    ),
]


def _normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """Normalize scores to sum to 1.0."""
    total = sum(scores.values())
    if total == 0:
        return {"Visual": 0.25, "Auditory": 0.25, "Reading": 0.25, "Kinesthetic": 0.25}
    return {k: v / total for k, v in scores.items()}


def _calculate_behavioral_scores(
    features: LearningStyleFeatures,
) -> Dict[str, float]:
    """
    Calculate learning style scores from behavioral data.
    """
    scores = {"Visual": 0.0, "Auditory": 0.0, "Reading": 0.0, "Kinesthetic": 0.0}
    
    # Video and diagram content (Visual)
    visual_score = (
        features.video_watch_ratio * 0.6 +
        features.diagram_view_ratio * 0.4
    )
    scores["Visual"] = visual_score
    
    # Audio listening (Auditory)
    scores["Auditory"] = features.audio_listen_ratio
    
    # Text reading (Reading)
    reading_score = (
        features.text_read_ratio * 0.7 +
        features.quiz_performance_by_type.get("text", 0.5) * 0.3
    )
    scores["Reading"] = reading_score
    
    # Practice/hands-on (Kinesthetic)
    kinesthetic_score = (
        features.practice_problem_rate * 0.5 +
        (1 - features.avg_session_duration / 120) * 0.2 +  # Shorter sessions = more active
        features.completion_rate_by_format.get("interactive", 0.5) * 0.3
    )
    scores["Kinesthetic"] = kinesthetic_score
    
    return _normalize_scores(scores)


def _calculate_questionnaire_scores(
    answers: Dict[int, str],
) -> Dict[str, float]:
    """
    Calculate learning style scores from VARK questionnaire answers.
    answers: {question_id: selected_option_letter}
    """
    style_counts = {"Visual": 0, "Auditory": 0, "Reading": 0, "Kinesthetic": 0}
    total = 0
    
    for q_id, answer in answers.items():
        question = next((q for q in VARK_QUESTIONS if q.question_id == q_id), None)
        if question and answer in question.style_mapping:
            style = question.style_mapping[answer]
            style_counts[style] += 1
            total += 1
    
    if total == 0:
        return {"Visual": 0.25, "Auditory": 0.25, "Reading": 0.25, "Kinesthetic": 0.25}
    
    return {k: v / total for k, v in style_counts.items()}


def _combine_scores(
    behavioral: Dict[str, float],
    questionnaire: Dict[str, float],
    has_questionnaire: bool,
) -> Dict[str, float]:
    """
    Combine behavioral and questionnaire scores.
    Questionnaire is weighted higher if available.
    """
    if not has_questionnaire:
        return behavioral
    
    # Weight: 40% behavioral, 60% questionnaire (questionnaire is more reliable)
    weights = {"Visual": 0.4, "Auditory": 0.4, "Reading": 0.4, "Kinesthetic": 0.4}
    
    combined = {}
    for style in ["Visual", "Auditory", "Reading", "Kinesthetic"]:
        combined[style] = (
            behavioral[style] * weights[style] +
            questionnaire[style] * (1 - weights[style])
        )
    
    return _normalize_scores(combined)


def _get_primary_secondary(
    scores: Dict[str, float],
) -> Tuple[str, Optional[str]]:
    """Determine primary and secondary learning styles."""
    sorted_styles = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    primary = sorted_styles[0][0]
    
    # Secondary if score difference is less than 0.15
    secondary = None
    if len(sorted_styles) > 1:
        diff = sorted_styles[0][1] - sorted_styles[1][1]
        if diff < 0.15:
            secondary = sorted_styles[1][0]
    
    return primary, secondary


def _get_recommended_formats(style: str) -> List[str]:
    """Get recommended content formats for each learning style."""
    formats = {
        "Visual": [
            "📊 Infographics and charts",
            "🎬 Video tutorials with animations",
            "🗺️ Mind maps and concept diagrams",
            "📈 Graphs and data visualizations",
            "🎨 Color-coded notes",
        ],
        "Auditory": [
            "🎧 Podcast-style lectures",
            "🔊 Recorded verbal explanations",
            "💬 Group discussions and debates",
            "📢 Teaching others verbally",
            "🎵 Background music while studying",
        ],
        "Reading": [
            "📚 Detailed textbooks and articles",
            "📝 Comprehensive written notes",
            "📖 Step-by-step written instructions",
            "🔤 Flashcards with text",
            "📋 Summaries and outlines",
        ],
        "Kinesthetic": [
            "🔧 Interactive simulations",
            "✍️ Practice problems and exercises",
            "🧪 Hands-on experiments",
            "🎮 Gamified learning activities",
            "📍 Real-world examples and case studies",
        ],
    }
    return formats.get(style, formats["Reading"])


def _get_study_tips(
    primary_style: str,
    secondary_style: Optional[str],
    scores: Dict[str, float],
) -> List[str]:
    """Generate personalized study tips."""
    tips = []
    
    general_tips = {
        "Visual": [
            "🖌️ Create your own diagrams and mind maps",
            "📺 Watch video explanations before reading text",
            "🎨 Use colors to organize notes by topic",
            "🗺️ Draw connections between concepts visually",
        ],
        "Auditory": [
            "🎙️ Record yourself explaining concepts",
            "💬 Form study groups for verbal discussion",
            "🎧 Listen to educational podcasts",
            "🗣️ Read material aloud while studying",
        ],
        "Reading": [
            "📚 Take detailed written notes",
            "📖 Create comprehensive summaries",
            "🔤 Use flashcards with detailed text",
            "📝 Write practice essays and explanations",
        ],
        "Kinesthetic": [
            "🔧 Apply concepts through practice problems",
            "🧪 Conduct experiments or simulations",
            "✍️ Stand up and move while reviewing",
            "🎮 Use interactive learning platforms",
        ],
    }
    
    tips.extend(general_tips.get(primary_style, []))
    
    if secondary_style and secondary_style != primary_style:
        tips.append(f"💡 Also works well with {secondary_style}-style activities")
    
    # Add a tip based on low scores
    low_style = min(scores.items(), key=lambda x: x[1])[0]
    tips.append(f"💪 Challenge yourself with {low_style}-based activities to diversify learning")
    
    return tips


def _generate_explanation(
    primary_style: str,
    secondary_style: Optional[str],
    scores: Dict[str, float],
    has_questionnaire: bool,
) -> str:
    """Generate human-readable explanation."""
    method = "questionnaire responses" if has_questionnaire else "learning behavior patterns"
    
    explanation = (
        f"Based on your {method}, your primary learning style is ** {primary_style}** "
        f"with a score of {scores[primary_style]:.0%}."
    )
    
    if secondary_style:
        explanation += (
            f" You also show significant ** {secondary_style}** tendencies ({scores[secondary_style]:.0%}), "
            f"suggesting a multimodal learning preference."
        )
    
    explanation += (
        f"\n\nThis means you'll learn most effectively when content is presented "
        f"in {primary_style.lower()}-friendly formats."
    )
    
    return explanation


def detect_learning_style(
    features: LearningStyleFeatures,
    answers: Optional[Dict[int, str]] = None,
) -> LearningStyleResult:
    """
    Main function to detect learning style.
    
    Args:
        features: Behavioral learning data
        answers: Optional VARK questionnaire answers {q_id: option_letter}
    
    Returns:
        LearningStyleResult with detected style and recommendations
    """
    # Calculate scores from both sources
    behavioral_scores = _calculate_behavioral_scores(features)
    has_questionnaire = answers is not None and len(answers) >= 3
    questionnaire_scores = _calculate_questionnaire_scores(answers) if has_questionnaire else behavioral_scores
    
    # Combine scores
    combined_scores = _combine_scores(behavioral_scores, questionnaire_scores, has_questionnaire)
    
    # Determine primary and secondary styles
    primary, secondary = _get_primary_secondary(combined_scores)
    
    # Calculate confidence based on score separation and data quality
    score_range = max(combined_scores.values()) - min(combined_scores.values())
    confidence = min(0.9, 0.5 + score_range * 0.3 + (0.2 if has_questionnaire else 0))
    
    # Generate recommendations
    recommended_formats = _get_recommended_formats(primary)
    study_tips = _get_study_tips(primary, secondary, combined_scores)
    explanation = _generate_explanation(primary, secondary, combined_scores, has_questionnaire)
    
    return LearningStyleResult(
        primary_style=primary,
        secondary_style=secondary,
        scores={k: round(v, 3) for k, v in combined_scores.items()},
        confidence=round(confidence, 2),
        recommended_formats=recommended_formats,
        study_tips=study_tips,
        explanation=explanation,
    )


def detect_from_questionnaire_only(
    answers: Dict[int, str],
) -> LearningStyleResult:
    """
    Detect learning style from VARK questionnaire alone.
    Useful when no behavioral data is available.
    """
    features = LearningStyleFeatures(
        video_watch_ratio=0.5,
        diagram_view_ratio=0.5,
        text_read_ratio=0.5,
        audio_listen_ratio=0.5,
        quiz_performance_by_type={},
        avg_session_duration=45,
        study_time_preference="evening",
        completion_rate_by_format={},
        revisit_rate=0.5,
        practice_problem_rate=0.5,
        self_reported_style=None,
        best_performed_subjects=[],
        preferred_activities=[],
    )
    return detect_learning_style(features, answers)


def get_vark_questions() -> List[VARKQuestion]:
    """Return the VARK questionnaire questions."""
    return VARK_QUESTIONS


def generate_vark_form() -> List[Dict]:
    """Generate VARK questionnaire in form-friendly format."""
    return [
        {
            "id": q.question_id,
            "text": q.question_text,
            "options": [
                {"value": k, "label": v, "style": q.style_mapping[k]}
                for k, v in q.options.items()
            ],
        }
        for q in VARK_QUESTIONS
    ]


# Demo / Testing
if __name__ == "__main__":
    print("=" * 60)
    print("🎯 LEARNING STYLE DETECTION - VARK Model")
    print("=" * 60)
    
    # Example 1: Visual learner
    visual_features = LearningStyleFeatures(
        video_watch_ratio=0.9,
        diagram_view_ratio=0.85,
        text_read_ratio=0.3,
        audio_listen_ratio=0.2,
        quiz_performance_by_type={"video": 0.9, "text": 0.6, "audio": 0.5},
        avg_session_duration=40,
        study_time_preference="morning",
        completion_rate_by_format={"video": 0.95, "text": 0.7, "interactive": 0.6},
        revisit_rate=0.7,
        practice_problem_rate=0.3,
        best_performed_subjects=["Physics", "Geography"],
        preferred_activities=["watching_videos", "drawing_diagrams"],
    )
    
    result = detect_learning_style(visual_features)
    
    print(f"\n📊 Learning Style Scores:")
    for style, score in result.scores.items():
        bar = "█" * int(score * 30)
        print(f"   {style:<12} {score:.0%} {bar}")
    
    print(f"\n🎯 Primary Style:   {result.primary_style}")
    if result.secondary_style:
        print(f"   Secondary Style: {result.secondary_style}")
    print(f"   Confidence:      {result.confidence:.0%}")
    
    print(f"\n📺 Recommended Formats:")
    for fmt in result.recommended_formats[:3]:
        print(f"   {fmt}")
    
    print(f"\n💡 Study Tips:")
    for tip in result.study_tips[:3]:
        print(f"   {tip}")
    
    print("=" * 60)