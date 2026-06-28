import time
from fastapi import HTTPException, status
from app.config import get_settings

settings = get_settings()

# In-memory per-key sliding window counters. For production, swap to Redis.
_windows: dict[str, dict] = {}

WINDOW_SECONDS = 60


def check_rate_limit(api_key: str) -> None:
    now = time.time()
    state = _windows.setdefault(api_key, {"count": 0, "window_start": now})

    if now - state["window_start"] >= WINDOW_SECONDS:
        state["count"] = 0
        state["window_start"] = now

    state["count"] += 1

    if state["count"] > settings.rate_limit_rpm:
        retry_after = int(WINDOW_SECONDS - (now - state["window_start"])) + 1
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Retry after {retry_after}s.",
            headers={"Retry-After": str(retry_after)},
        )
