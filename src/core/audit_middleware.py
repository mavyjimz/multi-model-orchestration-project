"""FastAPI middleware for automatic audit logging."""

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.core.audit_logger import AuditLogger

audit_logger = AuditLogger()


class AuditLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to automatically log API requests/responses."""

    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip logging for health/metrics endpoints
        if request.url.path in ["/health", "/metrics", "/docs", "/openapi.json"]:
            return await call_next(request)

        # Capture request details
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")

        # Get user from auth context if available
        user = "anonymous"
        if hasattr(request.state, "user") and request.state.user:
            user = request.state.user.get("sub", "unknown")

        # Log request start
        audit_logger.log(
            event="api_request",
            user=user,
            action=f"{request.method} {request.url.path}",
            resource=request.url.path,
            details={
                "method": request.method,
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "user_agent": user_agent,
            },
            ip_address=client_ip,
        )

        # Process request
        response = await call_next(request)

        # Log response
        audit_logger.log(
            event="api_response",
            user=user,
            action=f"{request.method} {request.url.path}",
            resource=request.url.path,
            details={
                "status_code": response.status_code,
                "method": request.method,
                "path": request.url.path,
            },
            ip_address=client_ip,
        )

        return response
