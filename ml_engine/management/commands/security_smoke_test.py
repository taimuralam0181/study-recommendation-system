import json

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Run lightweight production security smoke checks."

    def add_arguments(self, parser):
        parser.add_argument("--strict", action="store_true")

    def handle(self, *args, **options):
        strict = bool(options["strict"])
        findings = []

        if settings.DEBUG:
            findings.append("DEBUG is True.")
        if "dev-only-key" in str(settings.SECRET_KEY) or "django-insecure" in str(settings.SECRET_KEY):
            findings.append("SECRET_KEY appears to be a development key.")
        if not settings.ALLOWED_HOSTS:
            findings.append("ALLOWED_HOSTS is empty.")
        if not bool(getattr(settings, "SESSION_COOKIE_SECURE", False)):
            findings.append("SESSION_COOKIE_SECURE is disabled.")
        if not bool(getattr(settings, "CSRF_COOKIE_SECURE", False)):
            findings.append("CSRF_COOKIE_SECURE is disabled.")
        if int(getattr(settings, "SECURE_HSTS_SECONDS", 0)) <= 0:
            findings.append("SECURE_HSTS_SECONDS is not set.")

        report = {
            "status": "ok" if not findings else "warning",
            "strict": strict,
            "findings": findings,
        }
        self.stdout.write(json.dumps(report, indent=2))

        if strict and findings:
            raise CommandError("Security smoke test failed in strict mode.")
