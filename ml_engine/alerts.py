from __future__ import annotations

import logging

from django.conf import settings
from django.core.mail import mail_admins


logger = logging.getLogger(__name__)


def send_ml_alert(subject: str, message: str) -> None:
    logger.warning("ML ALERT - %s | %s", subject, message)
    if not bool(getattr(settings, "ML_ALERT_EMAIL_ENABLED", False)):
        return
    try:
        mail_admins(subject=subject, message=message, fail_silently=True)
    except Exception as exc:
        logger.exception("Failed to send ML alert email: %s", exc)
