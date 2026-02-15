"""
Adaptive Difficulty System (Reinforcement Learning)

Implements a Multi-Armed Bandit (MAB) approach for adaptive difficulty:
- Epsilon-Greedy exploration strategy
- Thompson Sampling for probability matching
- UCB (Upper Confidence Bound) for exploration
- Contextual bandits with student features

Automatically adjusts problem difficulty for optimal learning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import math
import random
import json


@dataclass(frozen=True)
class Problem:
    """A practice problem with difficulty level."""
    problem_id: str
    subject: str
    topic: str
    difficulty: float           # 1-10 scale
    content: str              # Question text
    correct_answer: str
    options: Optional[List[str]] = None
    explanation: Optional[str] = None
    time_limit_seconds: int = 300


@dataclass(frozen=True)
class StudentResponse:
    """Student's response to a problem."""
    problem_id: str
    student_id: str
    selected_answer: str
    is_correct: bool
    response_time_seconds: float
    confidence_rating: Optional[int] = None  # 1-5 self-rating
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True)
class DifficultyRecommendation:
    """Recommended difficulty for next problem."""
    difficulty: float          # 1-10 scale
    confidence: float          # 0-1
    exploration_rate: float    # 0-1 (higher = more exploration)
    reason: str
    alternative_difficulties: List[Tuple[float, float]]  # (difficulty, confidence)
    estimated_success_rate: float


@dataclass(frozen=True)
class LearningState:
    """Student's current learning state."""
    student_id: str
    subject: str
    current_difficulty: float
    recent_success_rate: float
    streak: int
    total_problems_attempted: int
    total_correct: int
    avg_response_time: float
    last_interaction: datetime
    learning_rate: float       # How fast they improve
    plateaus: int             # Number of plateaus detected


@dataclass(frozen=True)
class AdaptiveSession:
    """A learning session with adaptive difficulty."""
    session_id: str
    student_id: str
    subject: str
    problems: List[Problem]
    difficulties: List[float]
    responses: List[StudentResponse]
    start_time: datetime
    end_time: Optional[datetime]
    total_score: float
    average_difficulty: float
    recommendations: List[str]


# Bandit Algorithms
class EpsilonGreedy:
    """Epsilon-Greedy bandit algorithm."""
    
    def __init__(self, epsilon: float = 0.1):
        self.epsilon = epsilon
        self.arm_rewards: Dict[float, List[float]] = {}  # difficulty -> rewards
        self.arm_counts: Dict[float, int] = {}
    
    def select_arm(self, difficulties: List[float]) -> float:
        """Select an arm (difficulty) using epsilon-greedy strategy."""
        if random.random() < self.epsilon:
            # Explore: random choice
            return random.choice(difficulties)
        else:
            # Exploit: choose best average
            best_arm = None
            best_avg = float('-inf')
            
            for arm in difficulties:
                if arm in self.arm_rewards and self.arm_rewards[arm]:
                    avg = sum(self.arm_rewards[arm]) / len(self.arm_rewards[arm])
                    if avg > best_avg:
                        best_avg = avg
                        best_arm = arm
            
            if best_arm is None:
                return difficulties[len(difficulties) // 2]  # Middle difficulty
        
    def update(self, arm: float, reward: float):
        """Update arm statistics with observed reward."""
        if arm not in self.arm_rewards:
            self.arm_rewards[arm] = []
            self.arm_counts[arm] = 0
        self.arm_rewards[arm].append(reward)
        self.arm_counts[arm] += 1
    
    def get_expected_reward(self, arm: float) -> float:
        """Get expected reward for an arm."""
        if arm in self.arm_rewards and self.arm_rewards[arm]:
            return sum(self.arm_rewards[arm]) / len(self.arm_rewards[arm])
        return 0.5  # Default expected reward


class ThompsonSampling:
    """Thompson Sampling for contextual bandits."""
    
    def __init__(self):
        self.arm_posterior: Dict[float, Tuple[float, float]] = {}  # (alpha, beta) for Beta distribution
    
    def select_arm(self, difficulties: List[float]) -> float:
        """Sample from posterior and select best arm."""
        samples = {}
        
        for arm in difficulties:
            if arm in self.arm_posterior:
                alpha, beta = self.arm_posterior[arm]
            else:
                alpha, beta = 1, 1  # Uniform prior
            
            samples[arm] = random.betavariate(alpha, beta)
        
        return max(samples.items(), key=lambda x: x[1])[0]
    
    def update(self, arm: float, success: bool):
        """Update posterior with observed outcome."""
        if arm not in self.arm_posterior:
            self.arm_posterior[arm] = (1, 1)
        
        alpha, beta = self.arm_posterior[arm]
        
        if success:
            alpha += 1
        else:
            beta += 1
        
        self.arm_posterior[arm] = (alpha, beta)
    
    def get_probability_of_success(self, arm: float) -> float:
        """Get estimated probability of success for an arm."""
        if arm in self.arm_posterior:
            alpha, beta = self.arm_posterior[arm]
            return alpha / (alpha + beta)
        return 0.5


class UCB1:
    """Upper Confidence Bound (UCB1) algorithm."""
    
    def __init__(self):
        self.arm_rewards: Dict[float, List[float]] = {}
        self.arm_counts: Dict[float, int] = {}
        self.total_count: int = 0
    
    def select_arm(self, difficulties: List[float]) -> float:
        """Select arm using UCB1 formula."""
        self.total_count += 1
        
        for arm in difficulties:
            if arm not in self.arm_counts:
                return arm  # Explore unseen arms first
        
        # Calculate UCB for each arm
        ucb_values = {}
        total_reward = sum(
            sum(rewards) for rewards in self.arm_rewards.values()
        )
        
        for arm in difficulties:
            n = self.arm_counts[arm]
            avg_reward = sum(self.arm_rewards[arm]) / n
            exploration = math.sqrt(2 * math.log(self.total_count) / n)
            ucb_values[arm] = avg_reward + exploration
        
        return max(ucb_values.items(), key=lambda x: x[1])[0]
    
    def update(self, arm: float, reward: float):
        """Update arm statistics."""
        if arm not in self.arm_rewards:
            self.arm_rewards[arm] = []
            self.arm_counts[arm] = 0
        self.arm_rewards[arm].append(reward)
        self.arm_counts[arm] += 1
    
    def get_confidence_interval(self, arm: float) -> Tuple[float, float]:
        """Get 95% confidence interval for an arm."""
        if arm not in self.arm_rewards or self.arm_counts[arm] < 2:
            return (0, 1)
        
        n = self.arm_counts[arm]
        avg = sum(self.arm_rewards[arm]) / n
        std = math.sqrt(sum((r - avg) ** 2 for r in self.arm_rewards[arm]) / n)
        margin = 1.96 * std / math.sqrt(n)
        
        return (max(0, avg - margin), min(1, avg + margin))


class AdaptiveDifficultyEngine:
    """Main engine for adaptive difficulty selection."""
    
    def __init__(self, algorithm: str = "thompson"):
        self.algorithm = algorithm
        self.student_states: Dict[str, LearningState] = {}
        self.bandits: Dict[str, Dict[str, object]] = {}  # (student_id, subject) -> bandit
        
        # Difficulty levels (1-10 scale)
        self.difficulty_levels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        # Target success rate (zone of proximal development)
        self.target_success_rate = 0.7  # 70% success rate
    
    def get_bandit(self, student_id: str, subject: str, algorithm: str = None) -> object:
        """Get or create bandit for student-subject pair."""
        key = f"{student_id}_{subject}"
        
        if key not in self.bandits:
            algo = algorithm or self.algorithm
            if algo == "epsilon_greedy":
                self.bandits[key] = EpsilonGreedy(epsilon=0.1)
            elif algo == "thompson":
                self.bandits[key] = ThompsonSampling()
            else:
                self.bandits[key] = UCB1()
        
        return self.bandits[key]
    
    def get_learning_state(
        self,
        student_id: str,
        subject: str,
    ) -> LearningState:
        """Get or create learning state for student."""
        key = f"{student_id}_{subject}"
        
        if key not in self.student_states:
            self.student_states[key] = LearningState(
                student_id=student_id,
                subject=subject,
                current_difficulty=5.0,
                recent_success_rate=0.5,
                streak=0,
                total_problems_attempted=0,
                total_correct=0,
                avg_response_time=60.0,
                last_interaction=datetime.now(),
                learning_rate=0.01,
                plateaus=0,
            )
        
        return self.student_states[key]
    
    def calculate_reward(
        self,
        response: StudentResponse,
        difficulty: float,
    ) -> float:
        """
        Calculate reward for the bandit.
        Rewards should balance correctness and engagement.
        """
        reward = 0.0
        
        # Base reward for correctness
        if response.is_correct:
            reward += 0.7
        
        # Bonus for faster responses (within time limit)
        time_ratio = min(response.response_time_seconds / 300, 1.0)
        if response.is_correct:
            reward += 0.3 * (1 - time_ratio)  # Faster = more reward
        else:
            # Penalty for slow incorrect responses
            reward -= 0.1 * time_ratio
        
        # Confidence calibration bonus
        if response.confidence_rating is not None:
            if response.is_correct and response.confidence_rating >= 4:
                reward += 0.1  # Good calibration
            elif not response.is_correct and response.confidence_rating <= 2:
                reward += 0.1  # Good calibration
        
        # Scale by difficulty (harder problems = higher potential reward)
        reward *= (0.5 + difficulty / 20)
        
        return max(0, min(1, reward))
    
    def get_recommended_difficulty(
        self,
        student_id: str,
        subject: str,
        algorithm: Optional[str] = None,
    ) -> DifficultyRecommendation:
        """
        Get recommended difficulty for next problem.
        """
        state = self.get_learning_state(student_id, subject)
        bandit = self.get_bandit(student_id, subject, algorithm)
        
        # Get success probability for each difficulty level
        success_probs = {}
        for diff in self.difficulty_levels:
            if hasattr(bandit, 'get_probability_of_success'):
                success_probs[diff] = bandit.get_probability_of_success(diff)
            elif hasattr(bandit, 'get_expected_reward'):
                success_probs[diff] = bandit.get_expected_reward(diff)
            else:
                success_probs[diff] = 0.5
        
        # closest to target success Find difficulty rate
        best_diff = min(
            self.difficulty_levels,
            key=lambda d: abs(success_probs[d] - self.target_success_rate)
        )
        
        # Calculate confidence based on data
        if hasattr(bandit, 'arm_counts'):
            total_count = sum(bandit.arm_counts.values())
            if total_count > 20:
                confidence = min(0.9, 0.5 + total_count * 0.02)
            else:
                confidence = 0.4 + total_count * 0.02
        else:
            confidence = 0.5
        
        # Exploration rate (decreases with more data)
        if hasattr(bandit, 'epsilon'):
            exploration_rate = bandit.epsilon
        elif hasattr(bandit, 'total_count'):
            exploration_rate = max(0.05, 1 / math.sqrt(bandit.total_count))
        else:
            exploration_rate = 0.1
        
        # Generate alternative recommendations
        alternatives = sorted(
            [(d, success_probs[d]) for d in self.difficulty_levels],
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        # Estimate success rate
        estimated_success = success_probs[best_diff]
        
        # Generate reason
        if state.total_problems_attempted < 5:
            reason = "Initial learning phase - gathering data for personalized difficulty"
        elif estimated_success > 0.8:
            reason = f"Strong performance detected - ready for more challenge"
        elif estimated_success < 0.4:
            reason = f"Difficulty detected - adjusting to build confidence"
        else:
            reason = f"Optimal challenge level for steady progress"
        
        # Adjust based on recent trend
        if state.recent_success_rate < estimated_success - 0.2:
            best_diff = max(1, best_diff - 1)
            reason = f"Recent struggles noted - reducing difficulty slightly"
        elif state.recent_success_rate > estimated_success + 0.2:
            best_diff = min(10, best_diff + 1)
            reason = f"Recent success suggests readiness for harder problems"
        
        return DifficultyRecommendation(
            difficulty=best_diff,
            confidence=round(confidence, 2),
            exploration_rate=round(exploration_rate, 2),
            reason=reason,
            alternative_difficulties=alternatives,
            estimated_success_rate=round(estimated_success, 2),
        )
    
    def record_response(
        self,
        student_id: str,
        subject: str,
        response: StudentResponse,
        difficulty: float,
    ):
        """Record a student response and update models."""
        # Calculate reward
        reward = self.calculate_reward(response, difficulty)
        
        # Update bandit
        bandit = self.get_bandit(student_id, subject)
        bandit.update(difficulty, reward)
        
        # Update learning state
        state = self.get_learning_state(student_id, subject)
        
        state.total_problems_attempted += 1
        if response.is_correct:
            state.total_correct += 1
            state.streak += 1
        else:
            state.streak = 0
        
        # Update rolling success rate
        state.recent_success_rate = (
            0.3 * (1 if response.is_correct else 0) +
            0.7 * state.recent_success_rate
        )
        
        # Update response time (exponential moving average)
        state.avg_response_time = (
            0.3 * response.response_time_seconds +
            0.7 * state.avg_response_time
        )
        
        # Detect plateau (success rate stagnating)
        if state.recent_success_rate > 0.6 and state.recent_success_rate < 0.8:
            if abs(state.recent_success_rate - 0.7) < 0.05:
                state.plateaus += 1
                if state.plateaus >= 3:
                    # Suggest difficulty change
                    state.plateaus = 0
        
        # Adjust learning rate based on performance
        if response.is_correct:
            state.learning_rate = min(0.05, state.learning_rate * 1.05)
        else:
            state.learning_rate = max(0.005, state.learning_rate * 0.95)
        
        state.current_difficulty = difficulty
        state.last_interaction = datetime.now()
    
    def create_session(
        self,
        student_id: str,
        subject: str = "General",
        num_problems: int = 10,
    ) -> AdaptiveSession:
        """Create an adaptive learning session."""
        session_id = f"{student_id}_{subject}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        problems = []
        difficulties = []
        responses = []
        
        for i in range(num_problems):
            recommendation = self.get_recommended_difficulty(student_id, subject)
            
            # Create a sample problem (in real system, this would fetch from database)
            problem = Problem(
                problem_id=f"{session_id}_{i}",
                subject=subject,
                topic=f"Topic_{i % 3 + 1}",
                difficulty=recommendation.difficulty,
                content=f"Practice problem {i+1} (Difficulty: {recommendation.difficulty}/10)",
                correct_answer="A",
                options=["A", "B", "C", "D"],
                explanation=f"Solution to problem {i+1}",
            )
            
            problems.append(problem)
            difficulties.append(recommendation.difficulty)
        
        return AdaptiveSession(
            session_id=session_id,
            student_id=student_id,
            subject=subject,
            problems=problems,
            difficulties=difficulties,
            responses=responses,
            start_time=datetime.now(),
            end_time=None,
            total_score=0,
            average_difficulty=sum(difficulties) / len(difficulties),
            recommendations=[],
        )
    
    def complete_session(self, session: AdaptiveSession):
        """Mark a session as complete."""
        session.end_time = datetime.now()
        
        if session.responses:
            session.total_score = sum(
                1 for r in session.responses if r.is_correct
            ) / len(session.responses)
    
    def get_session_summary(self, session: AdaptiveSession) -> Dict:
        """Get summary of a learning session."""
        if not session.responses:
            return {"message": "No responses recorded"}
        
        correct = sum(1 for r in session.responses if r.is_correct)
        avg_time = sum(
            r.response_time_seconds for r in session.responses
        ) / len(session.responses)
        accuracy = correct / len(session.responses)
        end_time = session.end_time or datetime.now()
        duration_seconds = max(0, (end_time - session.start_time).total_seconds())
        
        return {
            "session_id": session.session_id,
            "student_id": session.student_id,
            "subject": session.subject,
            "total_questions": len(session.responses),
            "correct": correct,
            "accuracy": round(accuracy, 2),
            "avg_response_time_seconds": round(avg_time, 2),
            "average_difficulty": round(session.average_difficulty, 2),
            "total_score": round(session.total_score, 2),
            "duration_minutes": round(duration_seconds / 60.0, 1),
        }
