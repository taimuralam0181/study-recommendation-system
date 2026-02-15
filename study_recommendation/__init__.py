try:
    from study_recommendation.celery import app as celery_app
except Exception:  # Optional dependency in local/dev before celery install
    celery_app = None

__all__ = ("celery_app",)
