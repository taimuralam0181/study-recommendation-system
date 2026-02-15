from django.db import models
from django.contrib.auth.models import User


class Semester(models.Model):
    number = models.PositiveSmallIntegerField(unique=True)
    title = models.CharField(max_length=100, blank=True)

    def __str__(self):
        return f"Semester {self.number}"


class Subject(models.Model):
    semester = models.ForeignKey(Semester, on_delete=models.CASCADE)
    name = models.CharField(max_length=120)
    department = models.CharField(max_length=50, default='CSE')
    is_active = models.BooleanField(default=True)

    class Meta:
        unique_together = ('semester', 'name', 'department')
        ordering = ('semester__number', 'name')

    def __str__(self):
        return f"S{self.semester.number}: {self.name}"


class Student(models.Model):
    LEVEL_CHOICES = [
        ('Beginner', 'Beginner'),
        ('Intermediate', 'Intermediate'),
        ('Advanced', 'Advanced'),
    ]

    user = models.OneToOneField(User, on_delete=models.CASCADE)
    math_marks = models.IntegerField(default=0)
    physics_marks = models.IntegerField(default=0)
    cs_marks = models.IntegerField(default=0)
    avg_score = models.FloatField(default=0.0)
    level = models.CharField(max_length=20, choices=LEVEL_CHOICES, default='Beginner')
    cgpa = models.FloatField(default=0.0)

    def __str__(self):
        return self.user.username


class StudentSubjectPerformance(models.Model):
    """
    Stores per-subject performance using the academic evaluation scheme (Total = 100):
    - Hand/Teacher assessment: 20
    - Attendance: 5
    - Class Test (CT): 15
    - Midterm: 30
    - Final: 50

    Academic note:
    - We store the components separately for transparency and viva evaluation.
    - The system computes Total = sum of all components (0..100).
    """

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    semester = models.ForeignKey(Semester, on_delete=models.CASCADE)
    subject = models.ForeignKey(Subject, on_delete=models.CASCADE)

    hand_marks = models.PositiveSmallIntegerField(default=0)        # 0..20
    attendance_marks = models.PositiveSmallIntegerField(default=0)  # 0..5
    attendance_percentage = models.FloatField(default=0.0)          # 0..100 (stored for academic reporting)
    ct_marks = models.PositiveSmallIntegerField(default=0)          # 0..15
    midterm_marks = models.PositiveSmallIntegerField(default=0)     # 0..30
    # Final marks are NOT entered by students; can be stored later as ground truth by teachers/exams.
    final_marks = models.PositiveSmallIntegerField(default=0, blank=True)  # 0..50 (optional for students)
    total = models.PositiveSmallIntegerField(default=0)                    # 0..100 (sum; may use predicted final)

    # ML outputs (explainable predictions)
    final_pred = models.FloatField(default=0.0)       # predicted final out of 50
    total_pred = models.FloatField(default=0.0)       # predicted total out of 100
    prob_a = models.FloatField(default=0.0)           # P(total >= 75)
    prob_aplus = models.FloatField(default=0.0)       # P(total >= 80)
    prob_fail = models.FloatField(default=0.0)        # P(total < 40)
    predicted_grade = models.CharField(max_length=5, blank=True, default="")  # e.g., A+, A, B+, C, D

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('user', 'semester', 'subject')
        ordering = ('semester__number', 'subject__name')

    def __str__(self):
        return f"{self.user.username} - S{self.semester.number} - {self.subject.name}: {self.total}"


class StudentResultHistory(models.Model):
    LEVEL_CHOICES = [
        ('Beginner', 'Beginner'),
        ('Intermediate', 'Intermediate'),
        ('Advanced', 'Advanced'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    math_marks = models.IntegerField()
    physics_marks = models.IntegerField()
    cs_marks = models.IntegerField()
    avg_score = models.FloatField()
    level = models.CharField(max_length=20, choices=LEVEL_CHOICES)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.level} ({self.created_at:%Y-%m-%d})"
