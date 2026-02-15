from django.db import models

class StudyMaterial(models.Model):
    LEVEL_CHOICES = [
        ('Basic', 'Basic'),
        ('Intermediate', 'Intermediate'),
        ('Advanced', 'Advanced'),
    ]

    subject = models.CharField(max_length=100)
    title = models.CharField(max_length=200)
    material_link = models.URLField()
    level = models.CharField(max_length=20, choices=LEVEL_CHOICES)

    def __str__(self):
        return f"{self.subject} - {self.level}"
