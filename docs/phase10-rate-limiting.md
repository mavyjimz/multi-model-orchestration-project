# Phase 10.2: Rate Limiting Configuration

## Overview
Implements request throttling to prevent abuse and ensure fair resource usage.

## Default Limits
- 100 requests per minute per IP address
- Configurable via environment variables: RATE_LIMIT_REQUESTS, RATE_LIMIT_PERIOD

## Endpoints
- `/auth/login`: 5 requests/minute (prevents brute force)
- `/predict`: 30 requests/minute (model inference is expensive)
- `/health`, `/metrics`: No limit (monitoring endpoints)

## Production Considerations
1. Switch storage_uri from "memory://" to Redis for multi-instance deployments
2. Add X-Real-IP header parsing if behind proxy/load balancer
3. Consider user-based limits (not just IP) for authenticated endpoints

## Testing
```bash
# Test rate limiting with curl loop
for i in {1..110}; do
    curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000/health
done
# Should see 429 responses after ~100 requests
