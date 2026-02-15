from __future__ import annotations

import time
from typing import Tuple

from django.core.cache import cache


def _client_ip(request) -> str:
    forwarded = request.META.get("HTTP_X_FORWARDED_FOR", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.META.get("REMOTE_ADDR", "unknown")


def check_rate_limit(
    request,
    scope: str,
    limit: int,
    window_seconds: int,
) -> Tuple[bool, int]:
    """
    Returns (allowed, retry_after_seconds).
    """
    user_key = f"user:{request.user.id}" if getattr(request.user, "is_authenticated", False) else f"ip:{_client_ip(request)}"
    now = int(time.time())
    key = f"rate_limit:{scope}:{user_key}"
    state = cache.get(key)

    if not state or int(state.get("reset_at", 0)) <= now:
        state = {"count": 0, "reset_at": now + int(window_seconds)}

    if int(state["count"]) >= int(limit):
        retry_after = max(1, int(state["reset_at"]) - now)
        return False, retry_after

    state["count"] = int(state["count"]) + 1
    ttl = max(1, int(state["reset_at"]) - now)
    cache.set(key, state, timeout=ttl)
    return True, ttl
