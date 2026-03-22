"""Rate limiting configuration for Phase 10."""

import os

from slowapi import Limiter
from slowapi.util import get_remote_address

# Configuration from environment
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
RATE_LIMIT_PERIOD = int(os.getenv("RATE_LIMIT_PERIOD", "60"))  # seconds

# Create limiter instance
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[f"{RATE_LIMIT_REQUESTS}/{RATE_LIMIT_PERIOD}second"],
    storage_uri="memory://",  # Use Redis in production: "redis://localhost:6379"
)


def get_limiter() -> Limiter:
    """Return configured limiter instance."""
    return limiter


def rate_limit(limit_string: str):
    """Decorator factory for custom rate limits."""

    def decorator(func):
        return limiter.limit(limit_string)(func)

    return decorator
