#!/bin/bash
# rebuild_and_test.sh - Rebuild container with structured logging middleware
# Exit on any error, undefined variable, or pipe failure
set -euo pipefail

# Configuration
CONTAINER_NAME="mlops-intent-classifier"
IMAGE_NAME="multi-model-orchestration"
IMAGE_TAG="v1.0.2-phase9"
PORT="8000"
HEALTH_ENDPOINT="/health"
TEST_CORRELATION_ID="partner-test-phase9"

echo "=== Starting rebuild and test sequence ==="

# Step 1: Stop and remove existing container
echo "[1/6] Stopping existing container..."
docker stop "${CONTAINER_NAME}" 2>/dev/null || echo "  (container not running)"
docker rm "${CONTAINER_NAME}" 2>/dev/null || echo "  (container not found)"

# Step 2: Build new image
echo "[2/6] Building image ${IMAGE_NAME}:${IMAGE_TAG}..."
docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" .

# Step 3: Verify image built
echo "[3/6] Verifying image..."
if ! docker images | grep -q "${IMAGE_NAME}.*${IMAGE_TAG}"; then
    echo "ERROR: Image build failed"
    exit 1
fi
echo "  Image verified: ${IMAGE_NAME}:${IMAGE_TAG}"

# Step 4: Start new container
echo "[4/6] Starting container ${CONTAINER_NAME}..."
docker run -d \
  --name "${CONTAINER_NAME}" \
  -p "${PORT}:${PORT}" \
  -e REGISTRY_DEV_MODE=true \
  -e PYTHONPATH=/app \
  --health-cmd "curl -f http://localhost:${PORT}${HEALTH_ENDPOINT} || exit 1" \
  --health-interval 30s \
  --health-timeout 10s \
  --health-retries 3 \
  "${IMAGE_NAME}:${IMAGE_TAG}"

# Step 5: Wait for startup and test endpoint
echo "[5/6] Waiting for app startup..."
sleep 15

echo "  Testing endpoint with correlation ID..."
CURL_OUTPUT=$(curl -s -i -H "X-Correlation-ID: ${TEST_CORRELATION_ID}" "http://localhost:${PORT}${HEALTH_ENDPOINT}")

if echo "${CURL_OUTPUT}" | grep -q "HTTP/1.1 200 OK"; then
    echo "  Endpoint test: PASSED"
else
    echo "ERROR: Endpoint test failed"
    echo "${CURL_OUTPUT}"
    exit 1
fi

if echo "${CURL_OUTPUT}" | grep -q "X-Correlation-ID: ${TEST_CORRELATION_ID}"; then
    echo "  Middleware header injection: PASSED"
else
    echo "WARNING: Middleware header not found in response (may need log check)"
fi

# Step 6: Check structured logs
echo "[6/6] Checking structured logs..."
sleep 2
LOG_OUTPUT=$(docker logs "${CONTAINER_NAME}" 2>&1 | grep "correlation_id" | tail -3)

if [ -n "${LOG_OUTPUT}" ]; then
    echo "  Structured logging: PASSED"
    echo "  Sample log entry:"
    echo "${LOG_OUTPUT}" | head -1 | python3 -m json.tool 2>/dev/null || echo "${LOG_OUTPUT}" | head -1
else
    echo "WARNING: No structured logs found yet (app may not have processed request)"
fi

echo "=== Rebuild and test sequence completed ==="
echo "Container ${CONTAINER_NAME} is running on port ${PORT}"
echo "Test with: curl -H 'X-Correlation-ID: my-id' http://localhost:${PORT}${HEALTH_ENDPOINT}"
