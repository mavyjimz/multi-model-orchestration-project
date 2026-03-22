"""Rate limit exception handler for FastAPI."""

from fastapi import Request, status
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded


async def rate_limit_exception_handler(
    request: Request,
    exc: RateLimitExceeded,
) -> JSONResponse:
    """Handle rate limit exceeded exceptions."""
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "detail": "Rate limit exceeded",
            "message": "Too many requests. Please try again later.",
            "retry_after": (exc.headers or {}).get("Retry-After", "60"),
        },
        headers={"Retry-After": (exc.headers or {}).get("Retry-After", "60")},
    )
