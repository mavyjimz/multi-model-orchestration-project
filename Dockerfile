# Stage 1: Build dependencies
FROM python:3.12-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt requirements-inference.txt ./
RUN pip install --user --no-cache-dir -r requirements.txt
RUN pip install --user --no-cache-dir -r requirements-inference.txt

# Stage 2: Runtime image (lightweight)
FROM python:3.12-slim as runtime

# Create non-root user for security
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

# Create necessary directories
RUN mkdir -p logs/audit artifacts/models mlruns && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port for FastAPI
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run the registry API
CMD ["uvicorn", "src.registry.api:app", "--host", "0.0.0.0", "--port", "8000"]
