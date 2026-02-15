from django.contrib import admin

from . import models


admin.site.register(
    [
        models.UploadedDataset,
        models.DatasetStudentPerformance,
        models.CSETrainingExample,
        models.TrainedModel,
        models.MLModelVersion,
        models.PredictionLog,
        models.StudentLearningState,
        models.BurnoutHistory,
        models.StudySession,
        models.SpacedRepetitionItem,
        models.SpacedRepetitionReview,
        models.LearningStyleProfile,
        models.AnxietyLog,
        models.ContentInteraction,
        models.PerformanceTrend,
        models.KnowledgeGapRecord,
        models.DataDriftLog,
    ]
)
