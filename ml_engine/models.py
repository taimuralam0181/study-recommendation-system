"""
Database Models for ML Engine

Tracks ML predictions, student interactions, and model performance.
"""

from django.db import models
from django.contrib.auth import get_user_model
from django.utils import timezone


User = get_user_model()


class UploadedDataset(models.Model):
    STATUS_PENDING = "pending"
    STATUS_RUNNING = "running"
    STATUS_SUCCESS = "success"
    STATUS_FAILED = "failed"
    TRAIN_STATUS_CHOICES = [
        (STATUS_PENDING, "Pending"),
        (STATUS_RUNNING, "Running"),
        (STATUS_SUCCESS, "Success"),
        (STATUS_FAILED, "Failed"),
    ]

    file = models.FileField(upload_to='datasets/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    uploaded_by = models.ForeignKey(User, on_delete=models.CASCADE)
    checksum = models.CharField(max_length=64, blank=True)
    file_size = models.BigIntegerField(default=0)
    train_status = models.CharField(
        max_length=20,
        choices=TRAIN_STATUS_CHOICES,
        default=STATUS_PENDING,
    )
    trained_at = models.DateTimeField(null=True, blank=True)
    error_message = models.TextField(blank=True)

    class Meta:
        ordering = ['-uploaded_at']

    def __str__(self):
        username = getattr(self.uploaded_by, "username", None) or str(self.uploaded_by)
        return f"Dataset {self.id} by {username} [{self.train_status}]"


class StudyMaterial(models.Model):
    """Teacher-uploaded study materials."""

    MATERIAL_TYPES = [
        ('PDF', 'PDF'),
        ('Image', 'Image'),
        ('Link', 'Link'),
    ]

    subject = models.ForeignKey('userpanel.Subject', on_delete=models.CASCADE)
    title = models.CharField(max_length=200)
    description = models.TextField()
    material_type = models.CharField(max_length=30, choices=MATERIAL_TYPES)
    file = models.FileField(upload_to='materials/', blank=True)
    link = models.URLField(blank=True)
    added_by = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.subject} - {self.title}"


class StudentResult(models.Model):
    """Single-shot student result used by legacy pipelines."""

    student = models.OneToOneField(User, on_delete=models.CASCADE)
    subject = models.CharField(max_length=100)
    marks = models.IntegerField()

    def __str__(self):
        return f"{self.student.username} - {self.subject}: {self.marks}"


class DatasetStudentPerformance(models.Model):
    """Imported dataset rows for offline training."""

    source = models.CharField(max_length=200)
    uploaded_dataset = models.ForeignKey(
        UploadedDataset,
        on_delete=models.CASCADE,
        related_name='performance_rows',
        null=True,
        blank=True,
    )
    student_id = models.CharField(max_length=64, blank=True, default="")
    semester = models.PositiveSmallIntegerField(default=1)
    subject = models.CharField(max_length=120, blank=True, default="")
    assignment_marks = models.FloatField(default=0.0)
    quiz_marks = models.FloatField(default=0.0)
    attendance_percentage = models.FloatField(default=0.0)
    midterm_marks = models.FloatField(default=0.0)
    previous_cgpa = models.FloatField(default=0.0)
    final_marks = models.FloatField(default=0.0)
    final_grade = models.CharField(max_length=5, default='F')
    final_marks_norm = models.FloatField(default=0.0)
    previous_cgpa_norm = models.FloatField(default=0.0)
    math_marks = models.FloatField()
    physics_marks = models.FloatField()
    cs_marks = models.FloatField()
    avg_score = models.FloatField()
    level = models.CharField(max_length=20)
    math_norm = models.FloatField()
    physics_norm = models.FloatField()
    cs_norm = models.FloatField()
    avg_norm = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['semester', 'subject']),
            models.Index(fields=['student_id']),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=['uploaded_dataset', 'student_id', 'semester', 'subject'],
                name='uniq_dataset_student_sem_subject',
            ),
        ]

    def __str__(self):
        if self.student_id:
            return f"{self.student_id} S{self.semester} {self.subject}"
        return f"{self.source} - {self.level}"


class MLModelRun(models.Model):
    """Log model training runs for reproducibility."""

    kind = models.CharField(max_length=50)
    created_at = models.DateTimeField(auto_now_add=True)
    params = models.JSONField(default=dict)
    metrics = models.JSONField(default=dict)

    def __str__(self):
        return f"{self.kind} - {self.created_at:%Y-%m-%d}"


class CSETrainingExample(models.Model):
    """Training row used for CSE prediction models."""

    uploaded_dataset = models.ForeignKey(
        UploadedDataset,
        on_delete=models.CASCADE,
        related_name='training_examples',
        null=True,
        blank=True,
    )
    student_id = models.CharField(max_length=64, blank=True, default="")
    semester = models.PositiveSmallIntegerField()
    subject = models.CharField(max_length=120)
    assignment_marks = models.FloatField(default=0.0)
    attendance_percentage = models.FloatField(default=0.0)
    midterm_marks = models.FloatField(default=0.0)
    quiz_marks = models.FloatField(default=0.0)
    previous_cgpa = models.FloatField(default=0.0)
    final_marks = models.FloatField()
    total = models.FloatField()
    final_grade = models.CharField(max_length=5, default='F')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['semester', 'subject']),
            models.Index(fields=['final_grade']),
            models.Index(fields=['student_id']),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=['uploaded_dataset', 'student_id', 'semester', 'subject'],
                name='uniq_training_student_sem_subject',
            ),
        ]

    def __str__(self):
        return f"S{self.semester} {self.subject} - {self.total}"


class TrainedModel(models.Model):
    """Metadata registry for trained model artifacts."""

    TASK_REGRESSION = "regression"
    TASK_CLASSIFICATION = "classification"
    TASK_CHOICES = [
        (TASK_REGRESSION, "Final Marks Regression"),
        (TASK_CLASSIFICATION, "Final Grade Classification"),
    ]

    task_type = models.CharField(max_length=20, choices=TASK_CHOICES)
    algorithm = models.CharField(max_length=80)
    version = models.CharField(max_length=40)
    model_path = models.CharField(max_length=255)
    preprocessor_path = models.CharField(max_length=255, blank=True)
    label_encoder_path = models.CharField(max_length=255, blank=True)
    feature_columns = models.JSONField(default=list)
    target_column = models.CharField(max_length=80)
    metrics = models.JSONField(default=dict)
    confusion_matrix = models.JSONField(default=list, blank=True)
    classification_report = models.JSONField(default=dict, blank=True)
    is_active = models.BooleanField(default=True)
    trained_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    trained_on = models.ForeignKey(
        UploadedDataset,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='trained_models',
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['task_type', 'is_active']),
            models.Index(fields=['version']),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=['task_type', 'version'],
                name='uniq_task_type_version',
            ),
        ]

    def __str__(self):
        return f"{self.task_type}:{self.algorithm} ({self.version})"


class MLModelVersion(models.Model):
    """Track ML model versions and their performance."""
    
    MODEL_TYPES = [
        ('academic', 'Academic Prediction'),
        ('burnout', 'Burnout Prediction'),
        ('duration', 'Study Duration'),
        ('learning_style', 'Learning Style'),
        ('anxiety', 'Anxiety Detection'),
        ('spaced_repetition', 'Spaced Repetition'),
        ('knowledge_gaps', 'Knowledge Gap Analysis'),
        ('content_recommender', 'Content Recommender'),
        ('trend_prediction', 'Trend Prediction'),
        ('adaptive_difficulty', 'Adaptive Difficulty'),
        ('drift_detection', 'Drift Detection'),
    ]
    
    model_type = models.CharField(max_length=50, choices=MODEL_TYPES)
    version = models.CharField(max_length=20)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)
    
    # Model metrics
    accuracy = models.FloatField(null=True, blank=True)
    precision = models.FloatField(null=True, blank=True)
    recall = models.FloatField(null=True, blank=True)
    f1_score = models.FloatField(null=True, blank=True)
    
    # Path to saved model file
    model_path = models.CharField(max_length=255, blank=True)
    
    class Meta:
        unique_together = ['model_type', 'version']
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.model_type} v{self.version}"


class PredictionLog(models.Model):
    """Log of all ML predictions made."""
    
    MODEL_TYPES = [
        ('academic', 'Academic Prediction'),
        ('burnout', 'Burnout Prediction'),
        ('duration', 'Study Duration'),
        ('learning_style', 'Learning Style'),
        ('anxiety', 'Anxiety Detection'),
        ('spaced_repetition', 'Spaced Repetition'),
        ('knowledge_gaps', 'Knowledge Gap Analysis'),
        ('content_recommender', 'Content Recommender'),
        ('trend_prediction', 'Trend Prediction'),
        ('adaptive_difficulty', 'Adaptive Difficulty'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='ml_predictions')
    model_type = models.CharField(max_length=50, choices=MODEL_TYPES)
    created_at = models.DateTimeField(auto_now_add=True)
    
    # Input features (JSON)
    input_features = models.JSONField(default=dict)
    
    # Prediction result (JSON)
    prediction_result = models.JSONField(default=dict)
    
    # Response time
    response_time_ms = models.FloatField()
    
    # Model version used
    model_version = models.ForeignKey(
        MLModelVersion, 
        on_delete=models.SET_NULL, 
        null=True, 
        blank=True
    )
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['user', 'model_type']),
            models.Index(fields=['created_at']),
        ]
    
    def __str__(self):
        return f"{self.user.username} - {self.model_type} at {self.created_at}"


class StudentLearningState(models.Model):
    """Track student's current learning state for adaptive systems."""
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='learning_states')
    subject = models.CharField(max_length=100)
    
    # Adaptive difficulty state
    current_difficulty = models.FloatField(default=5.0)
    recent_success_rate = models.FloatField(default=0.5)
    streak = models.IntegerField(default=0)
    total_problems_attempted = models.IntegerField(default=0)
    total_correct = models.IntegerField(default=0)
    avg_response_time = models.FloatField(default=60.0)
    learning_rate = models.FloatField(default=0.01)
    plateaus = models.IntegerField(default=0)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        unique_together = ['user', 'subject']
        ordering = ['-updated_at']
    
    def __str__(self):
        return f"{self.user.username} - {self.subject}"


class BurnoutHistory(models.Model):
    """Track burnout predictions over time."""
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='burnout_history')
    risk_level = models.CharField(max_length=20)  # Low, Medium, High
    risk_score = models.FloatField()
    confidence = models.FloatField()
    
    # Key factors that contributed
    top_factors = models.JSONField(default=list)
    
    # Context
    avg_daily_study_hours = models.FloatField()
    consecutive_study_days = models.IntegerField()
    sleep_hours = models.FloatField()
    days_until_exam = models.IntegerField()
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.user.username} - {self.risk_level} at {self.created_at.date()}"


class StudySession(models.Model):
    """Track study sessions for duration optimization."""
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='study_sessions')
    subject = models.CharField(max_length=100)
    topic = models.CharField(max_length=100, blank=True)
    
    # Session details
    duration_minutes = models.IntegerField()
    completion_rate = models.FloatField()  # 0-1.0
    break_count = models.IntegerField(default=0)
    
    # Time of day
    time_of_day = models.CharField(max_length=20)  # morning, afternoon, evening, night
    day_of_week = models.CharField(max_length=20)  # weekday, weekend
    
    # Results
    quiz_score = models.FloatField(null=True, blank=True)  # 0-100
    retention_score = models.FloatField(null=True, blank=True)  # 0-1.0
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.user.username} - {self.subject} ({self.duration_minutes}min)"


class SpacedRepetitionItem(models.Model):
    """Track items for spaced repetition learning."""
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='sr_items')
    
    # Item content
    item_id = models.CharField(max_length=100)
    subject = models.CharField(max_length=100)
    topic = models.CharField(max_length=100)
    question = models.TextField()
    answer = models.TextField()
    tags = models.JSONField(default=list)
    
    # SM-2 state
    difficulty_rating = models.FloatField(default=2.5)  # 1-5
    interval_days = models.IntegerField(default=1)
    repetitions = models.IntegerField(default=0)
    ease_factor = models.FloatField(default=2.5)
    
    # Scheduling
    next_review_date = models.DateField()
    last_review_date = models.DateField(null=True, blank=True)
    created_date = models.DateField(auto_now_add=True)
    
    # Status
    is_active = models.BooleanField(default=True)
    is_mastered = models.BooleanField(default=False)  # interval > 21 days
    
    class Meta:
        unique_together = ['user', 'item_id']
        ordering = ['next_review_date']
    
    def __str__(self):
        return f"{self.user.username} - {self.item_id}"


class SpacedRepetitionReview(models.Model):
    """Log of all reviews."""
    
    item = models.ForeignKey(SpacedRepetitionItem, on_delete=models.CASCADE, related_name='reviews')
    
    # Review details
    quality = models.IntegerField()  # 0-5 (SM-2 quality rating)
    response_time_seconds = models.FloatField()
    confidence = models.IntegerField(null=True, blank=True)  # 1-5 self-rating
    
    # Result
    new_interval_days = models.IntegerField()
    new_ease_factor = models.FloatField()
    new_repetitions = models.IntegerField()
    predicted_retention = models.FloatField()
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.item.item_id} - Quality {self.quality}"


class LearningStyleProfile(models.Model):
    """Store detected learning style preferences."""
    
    STYLE_CHOICES = [
        ('visual', 'Visual'),
        ('auditory', 'Auditory'),
        ('reading', 'Reading/Writing'),
        ('kinesthetic', 'Kinesthetic'),
        ('multimodal', 'Multimodal'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='learning_style_profiles')
    
    # VARK scores
    visual_score = models.FloatField(default=0.0)
    auditory_score = models.FloatField(default=0.0)
    reading_score = models.FloatField(default=0.0)
    kinesthetic_score = models.FloatField(default=0.0)
    
    # Detected style
    primary_style = models.CharField(max_length=20, choices=STYLE_CHOICES)
    confidence = models.FloatField()
    
    # Study habits that informed this
    study_habits = models.JSONField(default=dict)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-updated_at']
    
    def __str__(self):
        return f"{self.user.username} - {self.primary_style}"


class AnxietyLog(models.Model):
    """Track anxiety predictions over time."""
    
    LEVEL_CHOICES = [
        ('Low', 'Low'),
        ('Moderate', 'Moderate'),
        ('High', 'High'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='anxiety_logs')
    anxiety_level = models.CharField(max_length=20, choices=LEVEL_CHOICES)
    anxiety_score = models.FloatField()
    confidence = models.FloatField()
    
    # Contributing factors
    contributing_factors = models.JSONField(default=list)
    
    # Context
    days_until_exam = models.FloatField()
    target_score = models.FloatField()
    previous_exam_score = models.FloatField()
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.user.username} - {self.anxiety_level} at {self.created_at.date()}"


class ContentInteraction(models.Model):
    """Track student interactions with learning content."""
    
    INTERACTION_TYPES = [
        ('view', 'View'),
        ('like', 'Like'),
        ('bookmark', 'Bookmark'),
        ('complete', 'Complete'),
        ('rate', 'Rate'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='content_interactions')
    content_id = models.CharField(max_length=100)
    
    interaction_type = models.CharField(max_length=20, choices=INTERACTION_TYPES)
    rating = models.FloatField(null=True, blank=True)  # 1-5 if rated
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['user', 'content_id']),
        ]
    
    def __str__(self):
        return f"{self.user.username} - {self.content_id} ({self.interaction_type})"


class PerformanceTrend(models.Model):
    """Track performance trends over time."""
    
    TREND_TYPES = [
        ('Improving', 'Improving'),
        ('Stable', 'Stable'),
        ('Declining', 'Declining'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='performance_trends')
    subject = models.CharField(max_length=100, blank=True)
    
    trend = models.CharField(max_length=20, choices=TREND_TYPES)
    change_percent = models.FloatField()
    confidence = models.FloatField()
    
    # Data range
    start_date = models.DateField()
    end_date = models.DateField()
    data_points = models.IntegerField()
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.user.username} - {self.trend} ({self.change_percent:+.1f}%)"


class KnowledgeGapRecord(models.Model):
    """Record of identified knowledge gaps."""
    
    PRIORITY_CHOICES = [
        ('critical', 'Critical'),
        ('high', 'High'),
        ('medium', 'Medium'),
        ('low', 'Low'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='knowledge_gaps')
    subject = models.CharField(max_length=100)
    topic = models.CharField(max_length=100)
    concept = models.CharField(max_length=200)
    
    current_level = models.FloatField()
    target_level = models.FloatField()
    gap_size = models.FloatField()
    priority = models.CharField(max_length=20, choices=PRIORITY_CHOICES)
    estimated_hours = models.FloatField()
    
    # Status
    is_addressed = models.BooleanField(default=False)
    addressed_date = models.DateField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['priority', '-created_at']
    
    def __str__(self):
        return f"{self.user.username} - {self.concept} ({self.priority})"


class DataDriftLog(models.Model):
    """Log detected data drift events."""
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='drift_logs')
    model_type = models.CharField(max_length=50)
    
    has_drift = models.BooleanField()
    drift_score = models.FloatField()
    drift_features = models.JSONField(default=list)
    
    # Performance comparison
    reference_accuracy = models.FloatField(null=True, blank=True)
    current_accuracy = models.FloatField(null=True, blank=True)
    
    recommendation = models.TextField()
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.model_type} - Drift: {self.has_drift} at {self.created_at.date()}"
