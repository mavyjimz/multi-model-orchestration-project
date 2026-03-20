"""
Prometheus metrics for MLOps observability.
Custom counters, histograms, and gauges for API monitoring.
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client import REGISTRY

# =============================================================================
# REQUEST METRICS
# =============================================================================

REQUEST_COUNT = Counter(
    "mlops_request_total",
    "Total number of API requests",
    ["endpoint", "method", "status_code"]
)

REQUEST_LATENCY = Histogram(
    "mlops_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

ERROR_COUNT = Counter(
    "mlops_error_total",
    "Total number of API errors",
    ["endpoint", "error_type"]
)

# =============================================================================
# MODEL METRICS
# =============================================================================

MODEL_VERSION = Gauge(
    "mlops_model_version",
    "Current deployed model version number",
    ["model_name", "stage"]
)

MODEL_LOAD_COUNT = Counter(
    "mlops_model_load_total",
    "Total number of model loads",
    ["model_name", "version"]
)

MODEL_INFERENCE_COUNT = Counter(
    "mlops_model_inference_total",
    "Total number of model inferences",
    ["model_name", "status"]
)

# =============================================================================
# SYSTEM METRICS
# =============================================================================

DISK_USAGE = Gauge(
    "mlops_disk_usage_bytes",
    "Disk usage in bytes",
    ["path"]
)

MEMORY_USAGE = Gauge(
    "mlops_memory_usage_bytes",
    "Memory usage in bytes",
    ["type"]
)

# =============================================================================
# DRIFT METRICS (Phase 5 integration)
# =============================================================================

DRIFT_PSI_SCORE = Gauge(
    "mlops_drift_psi_score",
    "Population Stability Index for drift detection",
    ["feature_group"]
)

DRIFT_KS_SCORE = Gauge(
    "mlops_drift_ks_score",
    "Kolmogorov-Smirnov statistic for drift detection",
    ["feature_group"]
)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_metrics():
    """Generate Prometheus metrics in text format."""
    return generate_latest(REGISTRY)

def get_content_type():
    """Get Prometheus content type header."""
    return CONTENT_TYPE_LATEST

def set_model_version(model_name: str, stage: str, version: str) -> None:
    """Set the current model version gauge."""
    try:
        # Convert version string to float (e.g., "v1.0.2" -> 1.002)
        version_num = version.replace("v", "").replace(".", "")
        if len(version_num) < 4:
            version_num = version_num.ljust(4, "0")
        version_float = float(version_num[:1] + "." + version_num[1:])
        MODEL_VERSION.labels(model_name=model_name, stage=stage).set(version_float)
    except (ValueError, IndexError):
        MODEL_VERSION.labels(model_name=model_name, stage=stage).set(1.0)

def record_request(endpoint: str, method: str, status_code: int, latency: float) -> None:
    """Record a request with latency."""
    REQUEST_COUNT.labels(endpoint=endpoint, method=method, status_code=status_code).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)

def record_error(endpoint: str, error_type: str) -> None:
    """Record an error."""
    ERROR_COUNT.labels(endpoint=endpoint, error_type=error_type).inc()

def record_inference(model_name: str, status: str) -> None:
    """Record a model inference."""
    MODEL_INFERENCE_COUNT.labels(model_name=model_name, status=status).inc()

def set_drift_scores(psi: float, ks: float, feature_group: str = "default") -> None:
    """Set drift detection scores."""
    DRIFT_PSI_SCORE.labels(feature_group=feature_group).set(psi)
    DRIFT_KS_SCORE.labels(feature_group=feature_group).set(ks)
