"""
Content Recommender System

Implements collaborative filtering for study content recommendations:
- User-Item collaborative filtering
- Matrix factorization with SVD
- Content-based filtering hybrid
- Real-time recommendation updates

Predicts which study materials a student will find most useful.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime
import math
import random
import json


@dataclass(frozen=True)
class StudyContent:
    """A piece of study content."""
    content_id: str
    title: str
    subject: str
    topic: str
    content_type: str            # "video", "article", "quiz", "exercise", "notes"
    difficulty: float            # 1-10 scale
    duration_minutes: int
    quality_rating: float       # 0-5 from users
    popularity: int              # views/completions
    tags: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    created_date: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True)
class UserInteraction:
    """User interaction with content."""
    user_id: str
    content_id: str
    interaction_type: str        # "viewed", "completed", "rated", "bookmarked"
    rating: Optional[float]       # 1-5 if rated
    time_spent_seconds: int
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True)
class ContentRecommendation:
    """A content recommendation for a user."""
    content: StudyContent
    predicted_rating: float
    confidence: float
    recommendation_reason: str
    similar_users_also_viewed: int
    match_score: float          # How well it matches user preferences


@dataclass(frozen=True)
class UserPreferences:
    """Learned user preferences."""
    preferred_subjects: Dict[str, float]
    preferred_content_types: Dict[str, float]
    preferred_difficulty_range: Tuple[float, float]
    avg_session_duration: float
    learning_style: str         # From VARK detection
    peak_learning_hours: int
    engagement_score: float


# Aliases for compatibility with unified_inference.py
ContentItem = StudyContent
StudentInteraction = UserInteraction


@dataclass(frozen=True)
class RecommendationReport:
    """Complete recommendation report."""
    user_id: str
    generated_at: datetime
    recommendations: List[ContentRecommendation]
    new_discovery_count: int
    review_count: int
    estimated_total_time: int
    by_subject: Dict[str, int]
    by_type: Dict[str, int]


# Sample content database
CONTENT_DATABASE: Dict[str, StudyContent] = {}


def init_sample_content():
    """Initialize sample content for demonstration."""
    contents = [
        StudyContent(
            content_id="math_calc_001",
            title="Introduction to Derivatives",
            subject="Mathematics",
            topic="Calculus",
            content_type="video",
            difficulty=5,
            duration_minutes=15,
            quality_rating=4.5,
            popularity=1500,
            tags=["calculus", "derivatives", "basics"],
        ),
        StudyContent(
            content_id="math_calc_002",
            title="Chain Rule Practice Problems",
            subject="Mathematics",
            topic="Calculus",
            content_type="exercise",
            difficulty=7,
            duration_minutes=30,
            quality_rating=4.2,
            popularity=800,
            tags=["calculus", "chain_rule", "practice"],
        ),
        StudyContent(
            content_id="phys_mech_001",
            title="Newton's Laws Explained",
            subject="Physics",
            topic="Mechanics",
            content_type="video",
            difficulty=4,
            duration_minutes=20,
            quality_rating=4.7,
            popularity=2500,
            tags=["mechanics", "newton", "forces"],
        ),
        StudyContent(
            content_id="cs_algo_001",
            title="Binary Search Tutorial",
            subject="Computer Science",
            topic="Algorithms",
            content_type="video",
            difficulty=5,
            duration_minutes=25,
            quality_rating=4.6,
            popularity=1800,
            tags=["algorithms", "binary_search", "searching"],
        ),
        StudyContent(
            content_id="cs_algo_002",
            title="Sorting Algorithms Comparison",
            subject="Computer Science",
            topic="Algorithms",
            content_type="article",
            difficulty=6,
            duration_minutes=15,
            quality_rating=4.3,
            popularity=1200,
            tags=["algorithms", "sorting", "comparison"],
        ),
        StudyContent(
            content_id="math_stat_001",
            title="Probability Fundamentals",
            subject="Mathematics",
            topic="Statistics",
            content_type="video",
            difficulty=4,
            duration_minutes=20,
            quality_rating=4.4,
            popularity=2000,
            tags=["probability", "statistics", "basics"],
        ),
    ]
    
    for c in contents:
        CONTENT_DATABASE[c.content_id] = c


# Collaborative Filtering
class UserItemMatrix:
    """User-Item rating matrix for collaborative filtering."""
    
    def __init__(self):
        self.ratings: Dict[str, Dict[str, float]] = {}  # user -> content -> rating
        self.users: Set[str] = set()
        self.items: Set[str] = set()
    
    def add_rating(self, user_id: str, content_id: str, rating: float):
        """Add or update a rating."""
        if user_id not in self.ratings:
            self.ratings[user_id] = {}
        self.ratings[user_id][content_id] = rating
        self.users.add(user_id)
        self.items.add(content_id)
    
    def get_user_ratings(self, user_id: str) -> Dict[str, float]:
        """Get all ratings for a user."""
        return self.ratings.get(user_id, {})
    
    def get_item_ratings(self, content_id: str) -> Dict[str, float]:
        """Get all ratings for an item."""
        item_ratings = {}
        for user, ratings in self.ratings.items():
            if content_id in ratings:
                item_ratings[user] = ratings[content_id]
        return item_ratings
    
    def similarity(self, user1: str, user2: str) -> float:
        """Calculate cosine similarity between two users."""
        ratings1 = self.get_user_ratings(user1)
        ratings2 = self.get_user_ratings(user2)
        
        common_items = set(ratings1.keys()) & set(ratings2.keys())
        if not common_items:
            return 0.0
        
        # Cosine similarity
        vec1 = [ratings1.get(item, 0) for item in common_items]
        vec2 = [ratings2.get(item, 0) for item in common_items]
        
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot / (norm1 * norm2)
    
    def find_similar_users(self, user_id: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find most similar users to the given user."""
        similarities = []
        for other_user in self.users:
            if other_user != user_id:
                sim = self.similarity(user_id, other_user)
                if sim > 0:
                    similarities.append((other_user, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def predict_rating(
        self,
        user_id: str,
        content_id: str,
        similar_users: List[Tuple[str, float]],
    ) -> float:
        """Predict rating for a user-item pair using collaborative filtering."""
        numerator = 0.0
        denominator = 0.0
        
        user_ratings = self.get_user_ratings(user_id)
        
        for similar_user, similarity in similar_users:
            similar_rating = self.ratings.get(similar_user, {}).get(content_id)
            if similar_rating is not None:
                # Weight by similarity
                numerator += similarity * (similar_rating - 3.0)  # Normalize around 3
                denominator += abs(similarity)
        
        if denominator == 0:
            return 3.0  # Default middle rating
        
        return 3.0 + (numerator / denominator)
    
    def to_matrix(self) -> Tuple[List[List[float]], List[str], List[str]]:
        """Convert to dense matrix for SVD."""
        user_list = sorted(self.users)
        item_list = sorted(self.items)
        
        matrix = []
        for user in user_list:
            row = []
            for item in item_list:
                row.append(self.ratings.get(user, {}).get(item, 0))
            matrix.append(row)
        
        return matrix, user_list, item_list


class ContentRecommender:
    """Main content recommendation system."""
    
    def __init__(self):
        self.user_matrix = UserItemMatrix()
        self.content_db: Dict[str, StudyContent] = {}
        self.interactions: List[UserInteraction] = []
    
    def add_content(self, content: StudyContent):
        """Add content to the database."""
        self.content_db[content.content_id] = content
    
    def add_interaction(self, interaction: UserInteraction):
        """Record a user interaction."""
        self.interactions.append(interaction)
        
        # Convert to implicit rating
        if interaction.rating is not None:
            rating = interaction.rating
        elif interaction.interaction_type == "completed":
            rating = 4.0
        elif interaction.interaction_type == "viewed":
            rating = 2.0 + min(2.0, interaction.time_spent_seconds / 300)
        else:
            rating = 3.0
        
        self.user_matrix.add_rating(
            interaction.user_id,
            interaction.content_id,
            min(5.0, rating),
        )
    
    def _get_content_features(self, content: StudyContent) -> Dict[str, float]:
        """Extract features from content for hybrid filtering."""
        return {
            f"subject_{content.subject}": 1.0,
            f"type_{content.content_type}": 1.0,
            f"difficulty_{content.difficulty}": 1.0,
            "popularity": content.popularity / 1000,
            "quality": content.quality_rating / 5.0,
        }
    
    def _calculate_content_similarity(
        self,
        content1: StudyContent,
        content2: StudyContent,
    ) -> float:
        """Calculate similarity between two content items."""
        score = 0.0
        
        # Same subject
        if content1.subject == content2.subject:
            score += 0.3
        
        # Same topic
        if content1.topic == content2.topic:
            score += 0.3
        
        # Same type
        if content1.content_type == content2.content_type:
            score += 0.1
        
        # Difficulty proximity (inverted)
        diff_diff = abs(content1.difficulty - content2.difficulty) / 10.0
        score += (1 - diff_diff) * 0.1
        
        # Tag overlap
        overlap = len(set(content1.tags) & set(content2.tags))
        score += min(0.2, overlap * 0.05)
        
        return score
    
    def _get_user_preferences(
        self,
        user_id: str,
    ) -> UserPreferences:
        """Learn user preferences from interactions."""
        user_ratings = self.user_matrix.get_user_ratings(user_id)
        
        if not user_ratings:
            return UserPreferences(
                preferred_subjects={},
                preferred_content_types={},
                preferred_difficulty_range=(4, 8),
                avg_session_duration=30,
                learning_style="Reading",
                peak_learning_hours=14,
                engagement_score=0.5,
            )
        
        # Calculate subject preferences
        subject_scores: Dict[str, float] = {}
        subject_counts: Dict[str, int] = {}
        
        type_scores: Dict[str, float] = {}
        type_counts: Dict[str, int] = {}
        
        difficulty_sum = 0
        difficulty_count = 0
        
        for content_id, rating in user_ratings.items():
            content = self.content_db.get(content_id)
            if content:
                subject_scores[content.subject] = subject_scores.get(content.subject, 0) + rating
                subject_counts[content.subject] = subject_counts.get(content.subject, 0) + 1
                
                type_scores[content.content_type] = type_scores.get(content.content_type, 0) + rating
                type_counts[content.content_type] = type_counts.get(content.content_type, 0) + 1
                
                difficulty_sum += content.difficulty * rating
                difficulty_count += rating
        
        # Normalize preferences
        preferred_subjects = {}
        for subj in subject_scores:
            avg = subject_scores[subj] / subject_counts[subj]
            preferred_subjects[subj] = avg / 5.0
        
        preferred_types = {}
        for t in type_scores:
            avg = type_scores[t] / type_counts[t]
            preferred_types[t] = avg / 5.0
        
        # Calculate difficulty preference
        avg_difficulty = difficulty_sum / difficulty_count if difficulty_count > 0 else 5.0
        
        # Calculate engagement score
        interactions = [i for i in self.interactions if i.user_id == user_id]
        avg_time = sum(i.time_spent_seconds for i in interactions) / len(interactions) if interactions else 300
        engagement_score = min(1.0, avg_time / 600)
        
        return UserPreferences(
            preferred_subjects=preferred_subjects,
            preferred_content_types=preferred_types,
            preferred_difficulty_range=(max(1, avg_difficulty - 2), min(10, avg_difficulty + 2)),
            avg_session_duration=30,
            learning_style="Reading",
            peak_learning_hours=14,
            engagement_score=engagement_score,
        )
    
    def recommend(
        self,
        user_id: str,
        n_recommendations: int = 10,
        include_discovery: bool = True,
    ) -> RecommendationReport:
        """
        Generate personalized content recommendations.
        
        Args:
            user_id: The user to recommend for
            n_recommendations: Number of recommendations to generate
            include_discovery: Whether to include new/unseen content
        
        Returns:
            RecommendationReport with ranked recommendations
        """
        # Get similar users
        similar_users = self.user_matrix.find_similar_users(user_id, top_k=10)
        
        # Get user preferences
        preferences = self._get_user_preferences(user_id)
        
        # Get user's already viewed content
        viewed = set(self.user_matrix.get_user_ratings(user_id).keys())
        
        candidates = []
        
        for content_id, content in self.content_db.items():
            if content_id in viewed and not include_discovery:
                continue
            
            # Calculate multiple scores
            scores = {}
            
            # 1. Collaborative filtering score
            if similar_users:
                collab_score = self.user_matrix.predict_rating(
                    user_id, content_id, similar_users
                )
                scores["collaborative"] = collab_score / 5.0
            else:
                scores["collaborative"] = 0.5
            
            # 2. Content-based score
            subject_pref = preferences.preferred_subjects.get(content.subject, 0.5)
            type_pref = preferences.preferred_content_types.get(content.content_type, 0.5)
            diff_min, diff_max = preferences.preferred_difficulty_range
            
            diff_penalty = 0
            if content.difficulty < diff_min:
                diff_penalty = (diff_min - content.difficulty) / 5.0
            elif content.difficulty > diff_max:
                diff_penalty = (content.difficulty - diff_max) / 5.0
            
            scores["content_match"] = (subject_pref + type_pref) / 2 - diff_penalty
            
            # 3. Quality score
            scores["quality"] = content.quality_rating / 5.0
            
            # 4. Popularity score
            scores["popularity"] = min(1.0, content.popularity / 3000)
            
            # Weighted combination
            final_score = (
                scores["collaborative"] * 0.35 +
                scores["content_match"] * 0.30 +
                scores["quality"] * 0.25 +
                scores["popularity"] * 0.10
            )
            
            # Calculate confidence
            confidence = 0.5
            if similar_users:
                confidence += 0.2
            if content.quality_rating >= 4.0:
                confidence += 0.1
            confidence = min(0.9, confidence)
            
            # Generate reason
            reason_parts = []
            if scores["collaborative"] > 0.7:
                reason_parts.append("Similar students liked this")
            if scores["content_match"] > 0.7:
                reason_parts.append(f"Matches your {content.subject} interests")
            if content.quality_rating >= 4.5:
                reason_parts.append("Highly rated")
            if content_id not in viewed:
                reason_parts.append("New discovery for you")
            
            reason = "; ".join(reason_parts) if reason_parts else "Recommended for you"
            
            # Count similar users who viewed
            similar_viewers = sum(
                1 for user, _ in similar_users
                if content_id in self.user_matrix.get_user_ratings(user)
            )
            
            candidates.append(ContentRecommendation(
                content=content,
                predicted_rating=final_score * 5,
                confidence=confidence,
                recommendation_reason=reason,
                similar_users_also_viewed=similar_viewers,
                match_score=final_score,
            ))
        
        # Sort by predicted rating
        candidates.sort(key=lambda x: x.predicted_rating, reverse=True)
        recommendations = candidates[:n_recommendations]
        
        # Generate summary stats
        new_count = sum(1 for r in recommendations if r.content.content_id not in viewed)
        review_count = len(recommendations) - new_count
        
        by_subject: Dict[str, int] = {}
        by_type: Dict[str, int] = {}
        total_time = 0
        
        for r in recommendations:
            by_subject[r.content.subject] = by_subject.get(r.content.subject, 0) + 1
            by_type[r.content.content_type] = by_type.get(r.content.content_type, 0) + 1
            total_time += r.content.duration_minutes
        
        return RecommendationReport(
            user_id=user_id,
            generated_at=datetime.now(),
            recommendations=recommendations,
            new_discovery_count=new_count,
            review_count=review_count,
            estimated_total_time=total_time,
            by_subject=by_subject,
            by_type=by_type,
        )
    
    def get_similar_content(self, content_id: str, top_k: int = 5) -> List[Tuple[StudyContent, float]]:
        """Find similar content to the given content."""
        target = self.content_db.get(content_id)
        if not target:
            return []
        
        similarities = []
        for other_id, content in self.content_db.items():
            if other_id != content_id:
                sim = self._calculate_content_similarity(target, content)
                similarities.append((content, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


def export_recommendations_json(report: RecommendationReport) -> str:
    """Export recommendations to JSON."""
    def serialize_datetime(dt: datetime) -> str:
        return dt.isoformat()
    
    report_dict = {
        "user_id": report.user_id,
        "generated_at": serialize_datetime(report.generated_at),
        "recommendations": [
            {
                "content_id": r.content.content_id,
                "title": r.content.title,
                "subject": r.content.subject,
                "topic": r.content.topic,
                "content_type": r.content.content_type,
                "difficulty": r.content.difficulty,
                "duration_minutes": r.content.duration_minutes,
                "predicted_rating": r.predicted_rating,
                "confidence": r.confidence,
                "reason": r.recommendation_reason,
            }
            for r in report.recommendations
        ],
        "new_discovery_count": report.new_discovery_count,
        "review_count": report.review_count,
        "estimated_total_time": report.estimated_total_time,
        "by_subject": report.by_subject,
        "by_type": report.by_type,
    }
    
    return json.dumps(report_dict, indent=2)


# Demo / Testing
if __name__ == "__main__":
    print("=" * 60)
    print("📚 CONTENT RECOMMENDATION SYSTEM")
    print("=" * 60)
    
    # Initialize recommender
    recommender = ContentRecommender()
    
    # Add sample content
    init_sample_content()
    for content in CONTENT_DATABASE.values():
        recommender.add_content(content)
    
    # Add sample user interactions
    sample_interactions = [
        UserInteraction(
            user_id="student_001",
            content_id="math_calc_001",
            interaction_type="completed",
            rating=5.0,
            time_spent_seconds=900,
        ),
        UserInteraction(
            user_id="student_001",
            content_id="math_calc_002",
            interaction_type="rated",
            rating=4.0,
            time_spent_seconds=1800,
        ),
        UserInteraction(
            user_id="student_002",
            content_id="math_calc_001",
            interaction_type="completed",
            rating=5.0,
            time_spent_seconds=1000,
        ),
        UserInteraction(
            user_id="student_002",
            content_id="phys_mech_001",
            interaction_type="completed",
            rating=4.5,
            time_spent_seconds=1200,
        ),
    ]
    
    for interaction in sample_interactions:
        recommender.add_interaction(interaction)
    
    # Get recommendations for student_001
    print("\n👤 Generating recommendations for student_001...")
    report = recommender.recommend("student_001", n_recommendations=5)
    
    print(f"\n📊 Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M')}")
    print(f"🎯 Total Recommendations: {len(report.recommendations)}")
    print(f"🆕 New Discoveries: {report.new_discovery_count}")
    print(f"📖 Reviews: {report.review_count}")
    print(f"⏱️ Estimated Time: {report.estimated_total_time} minutes")
    
    print("\n" + "=" * 60)
    print("📋 RECOMMENDED CONTENT")
    print("=" * 60)
    
    for i, rec in enumerate(report.recommendations, 1):
        content = rec.content
        print(f"\n{i}. {content.title}")
        print(f"   📚 Subject: {content.subject} | {content.topic}")
        print(f"   📖 Type: {content.content_type} | ⏱️ {content.duration_minutes} min | 📊 Difficulty: {content.difficulty}/10")
        print(f"   ⭐ Rating: {content.quality_rating}/5 | 👁️ {content.popularity} views")
        print(f"   🎯 Predicted: {rec.predicted_rating:.1f}/5 | Confidence: {rec.confidence:.0%}")
        print(f"   💡 Reason: {rec.recommendation_reason}")
    
    print("\n" + "=" * 60)
    print("📈 RECOMMENDATION BREAKDOWN")
    print("=" * 60)
    
    print("\nBy Subject:")
    for subject, count in report.by_subject.items():
        print(f"   {subject}: {count}")
    
    print("\nBy Type:")
    for ctype, count in report.by_type.items():
        print(f"   {ctype}: {count}")
    
    # Show similar content
    print("\n" + "=" * 60)
    print("🔗 SIMILAR CONTENT EXAMPLE")
    print("=" * 60)
    
    similar = recommender.get_similar_content("math_calc_001", top_k=3)
    print(f"\nContent similar to 'Introduction to Derivatives':")
    for content, score in similar:
        print(f"   → {content.title} (similarity: {score:.0%})")
    
    print("\n" + "=" * 60)