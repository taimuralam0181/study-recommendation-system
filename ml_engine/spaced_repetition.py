"""
Spaced Repetition Scheduler (SM-2 Algorithm Implementation)

Implements the SM-2 spaced repetition algorithm (used by Anki) for optimal
review scheduling. Adapts to student's retention performance.

Features:
- SM-2 algorithm for calculating optimal review intervals
- Subject-specific difficulty adjustments
- Forgetting curve modeling
- Daily review queue generation
- Retention rate tracking
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import math
import json


class ReviewQuality(Enum):
    """Quality rating for review response."""
    COMPLETE_BLACKOUT = 0    # Complete failure
    INCORRECT_REMEMBERED = 1  # Incorrect but remembered after
    INCORRECT_EASY = 2       # Incorrect but easy recall once seen
    CORRECT_DIFFICULT = 3    # Correct with serious difficulty
    CORRECT_HESITATION = 4   # Correct with some hesitation
    PERFECT = 5              # Perfect response


@dataclass(frozen=True)
class ReviewItem:
    """A single item for spaced repetition review."""
    item_id: str
    subject: str
    topic: str
    question: str
    answer: str
    difficulty_rating: float = 2.5  # Initial difficulty (1-5 scale)
    interval_days: int = 1         # Current interval
    repetitions: int = 0           # Times reviewed
    ease_factor: float = 2.5       # SM-2 ease factor
    next_review_date: datetime = field(default_factory=datetime.now)
    last_review_date: Optional[datetime] = None
    created_date: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ReviewResult:
    """Result after reviewing an item."""
    item_id: str
    quality: int                    # Quality rating (0-5)
    new_interval_days: int
    new_ease_factor: float
    new_repetitions: int
    next_review_date: datetime
    retention_predicted: float      # Predicted retention rate
    should_review_again_today: bool


@dataclass(frozen=True)
class DailySchedule:
    """Generated daily review schedule."""
    date: datetime
    items_due: List[ReviewItem]
    total_items: int
    estimated_time_minutes: int
    by_subject: Dict[str, int]
    new_items_count: int
    review_items_count: int


@dataclass(frozen=True)
class RetentionStats:
    """Retention statistics for a student."""
    total_reviews: int
    average_quality: float
    average_ease_factor: float
    retention_rate: float          # Overall retention rate
    subject_retention: Dict[str, float]
    streak_days: int
    due_items_count: int
    mastered_items_count: int      # Items with interval > 21 days


# SM-2 Algorithm Constants
MIN_EASE_FACTOR = 1.3
DEFAULT_EASE_FACTOR = 2.5
MAX_EASE_FACTOR = 3.5
MIN_INTERVAL = 1
MAX_INTERVAL = 365


def _quality_to_retention(quality: int) -> float:
    """Convert quality rating to retention probability."""
    # Based on SM-2 research
    return {
        0: 0.0,
        1: 0.1,
        2: 0.3,
        3: 0.6,
        4: 0.85,
        5: 0.95,
    }.get(quality, 0.5)


def calculate_sm2_interval(
    quality: int,
    repetitions: int,
    interval_days: int,
    ease_factor: float,
) -> Tuple[int, float, int]:
    """
    Calculate new interval using SM-2 algorithm.
    
    Args:
        quality: Response quality (0-5)
        repetitions: Number of previous successful reviews
        interval_days: Current interval in days
        ease_factor: SM-2 ease factor
    
    Returns:
        (new_interval_days, new_ease_factor, new_repetitions)
    """
    # If quality < 3, reset repetitions
    if quality < 3:
        new_repetitions = 0
        new_interval = 1
    else:
        new_repetitions = repetitions + 1
        
        if new_repetitions == 1:
            new_interval = 1
        elif new_repetitions == 2:
            new_interval = 6
        else:
            new_interval = round(interval_days * ease_factor)
    
    # Update ease factor
    # EF' = EF + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
    new_ease_factor = ease_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    new_ease_factor = max(MIN_EASE_FACTOR, min(MAX_EASE_FACTOR, new_ease_factor))
    
    # Clamp interval
    new_interval = max(MIN_INTERVAL, min(MAX_INTERVAL, new_interval))
    
    return new_interval, new_ease_factor, new_repetitions


def predict_forgetting_curve(
    ease_factor: float,
    interval_days: int,
    days_since_review: int,
) -> float:
    """
    Predict retention probability based on forgetting curve.
    
    Uses the formula: R = e^(-days / (interval * ease_factor))
    where R is retention probability.
    """
    if interval_days == 0:
        return 1.0
    
    retention = math.exp(-days_since_review / (interval_days * ease_factor))
    return max(0.0, min(1.0, retention))


def calculate_optimal_interval(
    target_retention: float,
    ease_factor: float,
) -> int:
    """
    Calculate interval needed to achieve target retention.
    
    Rearranged forgetting curve formula:
    interval = -days / ln(R) / ease_factor
    """
    if target_retention >= 1.0:
        return MAX_INTERVAL
    if target_retention <= 0:
        return MIN_INTERVAL
    
    # Target 90% retention
    ln_retention = math.log(target_retention)
    interval = -1 / (ln_retention * ease_factor)
    
    return max(MIN_INTERVAL, min(MAX_INTERVAL, int(interval)))


def review_item(
    item: ReviewItem,
    quality: int,
    review_date: Optional[datetime] = None,
) -> ReviewResult:
    """
    Process a review of an item with given quality rating.
    
    Args:
        item: The item being reviewed
        quality: Quality rating (0-5)
        review_date: When the review occurred (default: now)
    
    Returns:
        ReviewResult with updated scheduling info
    """
    if review_date is None:
        review_date = datetime.now()
    
    # Calculate new interval using SM-2
    new_interval, new_ease_factor, new_repetitions = calculate_sm2_interval(
        quality=quality,
        repetitions=item.repetitions,
        interval_days=item.interval_days,
        ease_factor=item.ease_factor,
    )
    
    # Calculate next review date
    next_review_date = review_date + timedelta(days=new_interval)
    
    # Predict retention
    days_since = (review_date - (item.last_review_date or item.created_date)).days
    retention_predicted = predict_forgetting_curve(
        new_ease_factor, new_interval, days_since
    )
    
    # Determine if should review again today
    should_repeat = quality < 3  # If failed, review again today
    
    return ReviewResult(
        item_id=item.item_id,
        quality=quality,
        new_interval_days=new_interval,
        new_ease_factor=round(new_ease_factor, 2),
        new_repetitions=new_repetitions,
        next_review_date=next_review_date,
        retention_predicted=round(retention_predicted, 3),
        should_review_again_today=should_repeat,
    )


def create_review_item(
    item_id: str,
    subject: str,
    topic: str,
    question: str,
    answer: str,
    difficulty: float = 2.5,
    tags: Optional[List[str]] = None,
) -> ReviewItem:
    """
    Create a new review item with initial settings.
    """
    return ReviewItem(
        item_id=item_id,
        subject=subject,
        topic=topic,
        question=question,
        answer=answer,
        difficulty_rating=difficulty,
        interval_days=1,
        repetitions=0,
        ease_factor=DEFAULT_EASE_FACTOR,
        next_review_date=datetime.now() + timedelta(days=1),
        tags=tags or [],
    )


def generate_daily_schedule(
    items: List[ReviewItem],
    target_date: Optional[datetime] = None,
    max_items: int = 50,
    max_time_minutes: int = 60,
) -> DailySchedule:
    """
    Generate a daily review schedule.
    
    Prioritizes:
    1. Overdue items
    2. Items close to forgetting (based on ease factor)
    3. New items
    """
    if target_date is None:
        target_date = datetime.now()
    
    # Find items due today or earlier
    due_items = [
        item for item in items
        if item.next_review_date <= target_date
    ]
    
    # Sort by priority (overdue first, then by predicted retention)
    def priority(item: ReviewItem) -> Tuple[int, float, int]:
        overdue_days = (target_date - item.next_review_date).days
        
        # Calculate predicted retention
        days_since = (target_date - (item.last_review_date or item.created_date)).days
        retention = predict_forgetting_curve(
            item.ease_factor, item.interval_days, days_since
        )
        
        # Priority: overdue (higher = more urgent), low retention (lower = more urgent)
        return (-overdue_days, retention, -item.repetitions)
    
    due_items.sort(key=priority)
    
    # Limit by max items and time
    # Assume ~1 minute per review + 30 seconds for new items
    selected_items = []
    total_time = 0
    
    for item in due_items:
        estimated_time = 60 if item.repetitions == 0 else 45
        if len(selected_items) < max_items and total_time + estimated_time <= max_time_minutes * 60:
            selected_items.append(item)
            total_time += estimated_time
    
    # Group by subject
    by_subject: Dict[str, int] = {}
    new_count = 0
    review_count = 0
    
    for item in selected_items:
        by_subject[item.subject] = by_subject.get(item.subject, 0) + 1
        if item.repetitions == 0:
            new_count += 1
        else:
            review_count += 1
    
    return DailySchedule(
        date=target_date,
        items_due=selected_items,
        total_items=len(selected_items),
        estimated_time_minutes=int(total_time / 60),
        by_subject=by_subject,
        new_items_count=new_count,
        review_items_count=review_count,
    )


def calculate_retention_stats(
    items: List[ReviewItem],
    review_history: List[Dict],
) -> RetentionStats:
    """
    Calculate retention statistics from all items and review history.
    """
    if not items:
        return RetentionStats(
            total_reviews=0,
            average_quality=0,
            average_ease_factor=2.5,
            retention_rate=0,
            subject_retention={},
            streak_days=0,
            due_items_count=0,
            mastered_items_count=0,
        )
    
    # Calculate metrics
    total_reviews = sum(item.repetitions for item in items)
    
    # Average ease factor
    avg_ease = sum(item.ease_factor for item in items) / len(items)
    
    # Calculate retention rates per subject
    subject_retention: Dict[str, float] = {}
    for item in items:
        days_since = (datetime.now() - (item.last_review_date or item.created_date)).days
        retention = predict_forgetting_curve(
            item.ease_factor, item.interval_days, days_since
        )
        if item.subject not in subject_retention:
            subject_retention[item.subject] = []
        subject_retention[item.subject].append(retention)
    
    # Average retention per subject
    for subj in subject_retention:
        subject_retention[subj] = sum(subject_retention[subj]) / len(subject_retention[subj])
    
    # Overall retention rate
    all_retentions = [
        predict_forgetting_curve(
            item.ease_factor, item.interval_days,
            (datetime.now() - (item.last_review_date or item.created_date)).days
        )
        for item in items
    ]
    avg_retention = sum(all_retentions) / len(all_retentions)
    
    # Mastered items (interval > 21 days)
    mastered = sum(1 for item in items if item.interval_days > 21)
    
    # Due items
    due = sum(1 for item in items if item.next_review_date <= datetime.now())
    
    # Streak calculation (simplified - would need daily review logs)
    streak = 0
    
    # Average quality from review history
    avg_quality = 0
    if review_history:
        qualities = [r.get("quality", 3) for r in review_history]
        avg_quality = sum(qualities) / len(qualities) if qualities else 3
    
    return RetentionStats(
        total_reviews=total_reviews,
        average_quality=round(avg_quality, 2),
        average_ease_factor=round(avg_ease, 2),
        retention_rate=round(avg_retention, 2),
        subject_retention={k: round(v, 2) for k, v in subject_retention.items()},
        streak_days=streak,
        due_items_count=due,
        mastered_items_count=mastered,
    )


def export_items_json(items: List[ReviewItem]) -> str:
    """Export review items to JSON format."""
    def serialize(item: ReviewItem) -> Dict:
        return {
            "item_id": item.item_id,
            "subject": item.subject,
            "topic": item.topic,
            "question": item.question,
            "answer": item.answer,
            "difficulty_rating": item.difficulty_rating,
            "interval_days": item.interval_days,
            "repetitions": item.repetitions,
            "ease_factor": item.ease_factor,
            "next_review_date": item.next_review_date.isoformat(),
            "last_review_date": item.last_review_date.isoformat() if item.last_review_date else None,
            "created_date": item.created_date.isoformat(),
            "tags": item.tags,
        }
    
    return json.dumps([serialize(i) for i in items], indent=2)


def import_items_json(json_data: str) -> List[ReviewItem]:
    """Import review items from JSON format."""
    data = json.loads(json_data)
    
    items = []
    for d in data:
        item = ReviewItem(
            item_id=d["item_id"],
            subject=d["subject"],
            topic=d["topic"],
            question=d["question"],
            answer=d["answer"],
            difficulty_rating=d.get("difficulty_rating", 2.5),
            interval_days=d.get("interval_days", 1),
            repetitions=d.get("repetitions", 0),
            ease_factor=d.get("ease_factor", 2.5),
            next_review_date=datetime.fromisoformat(d["next_review_date"]),
            last_review_date=datetime.fromisoformat(d["last_review_date"]) if d.get("last_review_date") else None,
            created_date=datetime.fromisoformat(d["created_date"]),
            tags=d.get("tags", []),
        )
        items.append(item)
    
    return items


# Demo / Testing
if __name__ == "__main__":
    print("=" * 60)
    print("🧠 SPACED REPETITION SCHEDULER (SM-2)")
    print("=" * 60)
    
    # Create some sample items
    items = [
        create_review_item(
            item_id="math_001",
            subject="Mathematics",
            topic="Calculus",
            question="What is the derivative of x^n?",
            answer="n * x^(n-1)",
            difficulty=2.0,
            tags=["calculus", "derivatives"],
        ),
        create_review_item(
            item_id="physics_001",
            subject="Physics",
            topic="Mechanics",
            question="What is Newton's Second Law?",
            answer="F = ma (Force = mass × acceleration)",
            difficulty=2.5,
            tags=["mechanics", "forces"],
        ),
        create_review_item(
            item_id="cs_001",
            subject="Computer Science",
            topic="Data Structures",
            question="What is the time complexity of binary search?",
            answer="O(log n)",
            difficulty=3.0,
            tags=["algorithms", "searching"],
        ),
    ]
    
    print(f"\n📚 Created {len(items)} review items")
    
    # Simulate reviews
    print("\n📝 Simulating Reviews:")
    
    # First review (easy)
    result1 = review_item(items[0], quality=5)
    print(f"\nItem: {items[0].question[:30]}...")
    print(f"   Quality: 5 (Perfect)")
    print(f"   New Interval: {result1.new_interval_days} days")
    print(f"   New Ease Factor: {result1.new_ease_factor}")
    print(f"   Next Review: {result1.next_review_date.strftime('%Y-%m-%d')}")
    print(f"   Predicted Retention: {result1.retention_predicted:.0%}")
    
    # Second review (difficult)
    result2 = review_item(items[1], quality=3)
    print(f"\nItem: {items[1].question[:30]}...")
    print(f"   Quality: 3 (Correct with difficulty)")
    print(f"   New Interval: {result2.new_interval_days} days")
    print(f"   New Ease Factor: {result2.new_ease_factor}")
    
    # Third review (failed)
    result3 = review_item(items[2], quality=1)
    print(f"\nItem: {items[2].question[:30]}...")
    print(f"   Quality: 1 (Incorrect, remembered after)")
    print(f"   New Interval: {result3.new_interval_days} days (reset)")
    print(f"   Should Review Again Today: {result3.should_review_again_today}")
    
    # Generate daily schedule
    print("\n" + "=" * 60)
    print("📅 DAILY REVIEW SCHEDULE")
    print("=" * 60)
    
    # Simulate some items with various states
    items[0].repetitions = 3
    items[0].interval_days = 7
    items[0].ease_factor = 2.6
    
    items[1].repetitions = 1
    items[1].interval_days = 6
    items[1].next_review_date = datetime.now()
    
    schedule = generate_daily_schedule(items, max_items=10, max_time_minutes=30)
    
    print(f"\nDate: {schedule.date.strftime('%Y-%m-%d')}")
    print(f"Total Items: {schedule.total_items}")
    print(f"Estimated Time: {schedule.estimated_time_minutes} minutes")
    print(f"New Items: {schedule.new_items_count}")
    print(f"Review Items: {schedule.review_items_count}")
    
    print(f"\nBy Subject:")
    for subject, count in schedule.by_subject.items():
        print(f"   {subject}: {count} items")
    
    print("\n" + "=" * 60)
    print("📊 RETENTION STATISTICS")
    print("=" * 60)
    
    stats = calculate_retention_stats(items, [])
    
    print(f"\nTotal Reviews: {stats.total_reviews}")
    print(f"Average Quality: {stats.average_quality:.1f}")
    print(f"Average Ease Factor: {stats.average_ease_factor:.2f}")
    print(f"Overall Retention Rate: {stats.retention_rate:.0%}")
    print(f"Mastered Items (>21 days): {stats.mastered_items_count}")
    print(f"Items Due Today: {stats.due_items_count}")
    
    print("\nSubject Retention:")
    for subject, rate in stats.subject_retention.items():
        print(f"   {subject}: {rate:.0%}")
    
    print("\n" + "=" * 60)