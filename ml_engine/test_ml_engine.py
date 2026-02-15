"""Unit Tests for ML Engine Modules - Simplified"""

import unittest
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_engine.burnout_prediction import BurnoutFeatures, predict_burnout, predict_burnout_simple
from ml_engine.study_duration import predict_optimal_duration, generate_session_schedule
from ml_engine.learning_style import LearningStyleFeatures, detect_learning_style
from ml_engine.anxiety_detection import AnxietyFeatures, detect_anxiety
from ml_engine.spaced_repetition import (
    ReviewItem, create_review_item, review_item, calculate_sm2_interval,
    predict_forgetting_curve, export_items_json, import_items_json
)
from ml_engine.knowledge_gaps import KnowledgeGapInput, analyze_knowledge_gaps
from ml_engine.content_recommender import StudyContent
from ml_engine.trend_prediction import PerformanceDataPoint, predict_performance_trend
from ml_engine.adaptive_difficulty import AdaptiveDifficultyEngine, StudentResponse
from ml_engine.drift_detection import DistributionStats, detect_data_drift


class TestBurnoutPrediction(unittest.TestCase):
    def test_low_risk_prediction(self):
        features = BurnoutFeatures(
            avg_daily_study_hours=4.0, study_session_count=5, avg_session_duration_minutes=45,
            consecutive_study_days=2, recent_grade_change=5.0, sleep_hours_per_day=8.0,
            assignment_completion_rate=0.9, attendance_rate=0.95, activity_level=0.8, days_until_exam=30,
        )
        result = predict_burnout(features)
        self.assertEqual(result.risk_level, "Low")
        self.assertLess(result.risk_score, 0.35)
    
    def test_high_risk_prediction(self):
        features = BurnoutFeatures(
            avg_daily_study_hours=12.0, study_session_count=14, avg_session_duration_minutes=120,
            consecutive_study_days=10, recent_grade_change=-20.0, sleep_hours_per_day=4.5,
            assignment_completion_rate=0.5, attendance_rate=0.7, activity_level=0.3, days_until_exam=3,
        )
        result = predict_burnout(features)
        self.assertEqual(result.risk_level, "High")
        self.assertGreater(result.risk_score, 0.6)
    
    def test_simple_interface(self):
        result = predict_burnout_simple(avg_daily_hours=8.0, consecutive_days=5, grade_change=-10.0, sleep_hours=6.0)
        self.assertIsNotNone(result)
        self.assertIn(result.risk_level, ["Low", "Medium", "High"])


class TestStudyDuration(unittest.TestCase):
    def test_optimal_duration_prediction(self):
        result = predict_optimal_duration(
            avg_session_length=45.0, avg_completion_rate=0.8, avg_quiz_score=75.0,
            avg_retention=0.7, preferred_time="morning", longest_streak=7,
            performance_trend=0.0, difficulty=5.0,
        )
        self.assertIsNotNone(result)
        self.assertGreater(result.optimal_duration_minutes, 0)
    
    def test_duration_bounds(self):
        result = predict_optimal_duration(
            avg_session_length=30.0, avg_completion_rate=0.5, avg_quiz_score=50.0,
            avg_retention=0.5, preferred_time="night", longest_streak=1,
            performance_trend=-10.0, difficulty=9.0,
        )
        self.assertGreaterEqual(result.optimal_duration_minutes, 15)
        self.assertLessEqual(result.optimal_duration_minutes, 90)
    
    def test_session_schedule_generation(self):
        schedule = generate_session_schedule(total_study_time_minutes=120, optimal_duration=30, break_interval=10)
        self.assertIsInstance(schedule, list)
        self.assertGreater(len(schedule), 0)


class TestLearningStyle(unittest.TestCase):
    def test_visual_learner_detection(self):
        input_data = LearningStyleFeatures(
            video_watch_ratio=0.9, diagram_view_ratio=0.8, text_read_ratio=0.2, audio_listen_ratio=0.1,
            quiz_performance_by_type={"video": 0.9, "text": 0.6}, avg_session_duration=45,
            study_time_preference="morning", completion_rate_by_format={"video": 0.9},
            revisit_rate=0.5, practice_problem_rate=0.3, best_performed_subjects=["Geometry"],
            preferred_activities=["watching_videos", "using_diagrams"],
        )
        result = detect_learning_style(input_data)
        self.assertEqual(result.primary_style, "Visual")
    
    def test_score_normalization(self):
        input_data = LearningStyleFeatures(
            video_watch_ratio=1.0, diagram_view_ratio=1.0, text_read_ratio=1.0, audio_listen_ratio=1.0,
            quiz_performance_by_type={}, avg_session_duration=45, study_time_preference="morning",
            completion_rate_by_format={}, revisit_rate=1.0, practice_problem_rate=1.0,
            best_performed_subjects=[], preferred_activities=[],
        )
        result = detect_learning_style(input_data)
        for score in result.scores.values():
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1)


class TestAnxietyDetection(unittest.TestCase):
    def test_low_anxiety_prediction(self):
        features = AnxietyFeatures(
            recent_grade_variance=100, performance_trend=10, historical_exam_score_gap=5,
            avg_preparation_time=15, study_procrastination_rate=0.2, last_minute_study_ratio=0.1,
            sleep_quality_index=0.9, physical_activity_rate=0.8, days_until_exam=14,
            exam_weight_percentage=30, num_past_exams_failed=0, stress_self_rating=3,
        )
        result = detect_anxiety(features)
        self.assertEqual(result.anxiety_level, "Low")
    
    def test_high_anxiety_prediction(self):
        features = AnxietyFeatures(
            recent_grade_variance=400, performance_trend=-20, historical_exam_score_gap=25,
            avg_preparation_time=2, study_procrastination_rate=0.9, last_minute_study_ratio=0.8,
            sleep_quality_index=0.3, physical_activity_rate=0.1, days_until_exam=2,
            exam_weight_percentage=60, num_past_exams_failed=3, stress_self_rating=9,
        )
        result = detect_anxiety(features)
        self.assertIn(result.anxiety_level, ["High", "Very High"])


class TestSpacedRepetition(unittest.TestCase):
    def test_sm2_interval_calculation(self):
        new_interval, new_ease, new_rep = calculate_sm2_interval(quality=5, repetitions=0, interval_days=1, ease_factor=2.5)
        self.assertEqual(new_interval, 1)
        self.assertEqual(new_rep, 1)
    
    def test_failed_review_reset(self):
        new_interval, new_ease, new_rep = calculate_sm2_interval(quality=2, repetitions=5, interval_days=30, ease_factor=2.5)
        self.assertEqual(new_interval, 1)
        self.assertEqual(new_rep, 0)
    
    def test_ease_factor_update(self):
        _, new_easy_high, _ = calculate_sm2_interval(quality=5, repetitions=2, interval_days=6, ease_factor=2.5)
        _, new_easy_low, _ = calculate_sm2_interval(quality=3, repetitions=2, interval_days=6, ease_factor=2.5)
        self.assertGreater(new_easy_high, 2.5)
        self.assertLess(new_easy_low, 2.5)
    
    def test_review_item(self):
        item = create_review_item(item_id="test_001", subject="Math", topic="Algebra", question="x+2=5?", answer="x=3", difficulty=2.5)
        result = review_item(item, quality=4)
        self.assertIsNotNone(result)
        self.assertGreater(result.new_interval_days, 0)
    
    def test_forgetting_curve(self):
        retention = predict_forgetting_curve(ease_factor=2.5, interval_days=7, days_since_review=3)
        self.assertGreater(retention, 0.7)
    
    def test_item_import_export(self):
        items = [create_review_item(item_id="test_1", subject="Science", topic="Physics", question="F=ma?", answer="Force = mass x acceleration")]
        json_data = export_items_json(items)
        imported = import_items_json(json_data)
        self.assertEqual(len(imported), 1)
        self.assertEqual(imported[0].subject, "Science")


class TestKnowledgeGaps(unittest.TestCase):
    def test_knowledge_gap_analysis(self):
        input_data = KnowledgeGapInput(
            current_marks={"Algebra": 60, "Calculus": 70, "Statistics": 55},
            target_marks={"Algebra": 80, "Calculus": 85, "Statistics": 80},
            syllabus_topics=["Algebra", "Calculus", "Statistics"],
            recent_quiz_scores={"Algebra": 55, "Calculus": 65, "Statistics": 50},
        )
        result = analyze_knowledge_gaps(input_data)
        self.assertIsNotNone(result)


class TestContentRecommender(unittest.TestCase):
    def test_content_item_creation(self):
        item = StudyContent(content_id="test_001", title="Test Video", subject="Math", topic="Algebra", content_type="video", difficulty=5.0, duration_minutes=10, quality_rating=4.0, popularity=100)
        self.assertEqual(item.content_id, "test_001")
        self.assertEqual(item.title, "Test Video")


class TestTrendPrediction(unittest.TestCase):
    def test_trend_prediction_improving(self):
        points = [
            PerformanceDataPoint(date=datetime.now() - timedelta(days=7), score=70, subject="Math", activity_type="quiz", weight=0.5),
            PerformanceDataPoint(date=datetime.now() - timedelta(days=5), score=72, subject="Math", activity_type="quiz", weight=0.5),
            PerformanceDataPoint(date=datetime.now() - timedelta(days=3), score=75, subject="Math", activity_type="quiz", weight=0.5),
            PerformanceDataPoint(date=datetime.now() - timedelta(days=1), score=78, subject="Math", activity_type="quiz", weight=0.5),
        ]
        result = predict_performance_trend(points)
        self.assertEqual(result.trend_direction, "Improving")
    
    def test_trend_prediction_declining(self):
        points = [
            PerformanceDataPoint(date=datetime.now() - timedelta(days=7), score=80, subject="Math", activity_type="quiz", weight=0.5),
            PerformanceDataPoint(date=datetime.now() - timedelta(days=5), score=75, subject="Math", activity_type="quiz", weight=0.5),
            PerformanceDataPoint(date=datetime.now() - timedelta(days=3), score=70, subject="Math", activity_type="quiz", weight=0.5),
            PerformanceDataPoint(date=datetime.now() - timedelta(days=1), score=65, subject="Math", activity_type="quiz", weight=0.5),
        ]
        result = predict_performance_trend(points)
        self.assertEqual(result.trend_direction, "Declining")


class TestAdaptiveDifficulty(unittest.TestCase):
    def test_session_creation(self):
        engine = AdaptiveDifficultyEngine(algorithm="epsilon_greedy")
        session = engine.create_session(student_id="test_001")
        self.assertIsNotNone(session)
        self.assertEqual(session.student_id, "test_001")
    
    def test_bandit_algorithms(self):
        epsilon_greedy = AdaptiveDifficultyEngine(algorithm="epsilon_greedy")
        thompson = AdaptiveDifficultyEngine(algorithm="thompson_sampling")
        ucb1 = AdaptiveDifficultyEngine(algorithm="ucb1")
        self.assertIsNotNone(epsilon_greedy)
        self.assertIsNotNone(thompson)
        self.assertIsNotNone(ucb1)


class TestDriftDetection(unittest.TestCase):
    def test_no_drift_detected(self):
        stats1 = DistributionStats(mean=75.0, std=10.0, count=100, min_value=50, max_value=100)
        stats2 = DistributionStats(mean=76.0, std=10.5, count=100, min_value=52, max_value=98)
        result = detect_data_drift(stats1, stats2)
        self.assertFalse(result.drift_detected)
    
    def test_drift_detected(self):
        stats1 = DistributionStats(mean=75.0, std=10.0, count=100, min_value=50, max_value=100)
        stats2 = DistributionStats(mean=40.0, std=15.0, count=100, min_value=20, max_value=60)
        result = detect_data_drift(stats1, stats2)
        self.assertTrue(result.drift_detected)


class TestEdgeCases(unittest.TestCase):
    def test_empty_inputs(self):
        empty_input = KnowledgeGapInput(current_marks={}, target_marks={}, syllabus_topics=[], recent_quiz_scores={})
        result = analyze_knowledge_gaps(empty_input)
        self.assertIsNotNone(result)
    
    def test_negative_values(self):
        result = predict_optimal_duration(
            avg_session_length=-10.0, avg_completion_rate=0.5, avg_quiz_score=50.0,
            avg_retention=0.5, preferred_time="morning", longest_streak=1,
            performance_trend=0, difficulty=5.0,
        )
        self.assertGreater(result.optimal_duration_minutes, 0)
    
    def test_extreme_values(self):
        result = predict_optimal_duration(
            avg_session_length=100.0, avg_completion_rate=0.99, avg_quiz_score=99.0,
            avg_retention=0.95, preferred_time="morning", longest_streak=365,
            performance_trend=50.0, difficulty=1.0,
        )
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
