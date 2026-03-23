"""
A/B Test Metrics Collector
Collects and compares metrics between baseline and candidate models
"""

import json
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ModelMetrics:
    """Metrics for a single model version"""

    model_name: str
    model_version: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    error_rate: float
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    accuracy: float | None
    collected_at: str


@dataclass
class ABTestComparison:
    """Comparison results between baseline and candidate"""

    deployment_id: str
    baseline_metrics: dict[str, Any]
    candidate_metrics: dict[str, Any]
    accuracy_delta: float
    latency_delta_ms: float
    error_rate_delta: float
    winner: str
    confidence: float
    recommendation: str
    collected_at: str


class ABMetricsCollector:
    """
    Collects and compares metrics for A/B testing

    Tracks performance metrics for both baseline and candidate
    models to determine which performs better.
    """

    def __init__(self, deployment_id: str):
        self.deployment_id = deployment_id
        self.metrics_dir = Path(f"results/phase11/canary_deployments/metrics/{deployment_id}")
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.baseline_metrics: list[dict[str, Any]] = []
        self.candidate_metrics: list[dict[str, Any]] = []

    def record_request(
        self,
        model_version: str,
        latency_ms: float,
        success: bool,
        correct_prediction: bool | None = None,
    ):
        """
        Record metrics for a single request

        Args:
            model_version: Version of the model that handled the request
            latency_ms: Request latency in milliseconds
            success: Whether the request was successful
            correct_prediction: Whether the prediction was correct (for accuracy)
        """
        metric = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "model_version": model_version,
            "latency_ms": latency_ms,
            "success": success,
            "correct_prediction": correct_prediction,
        }

        if model_version == "v1.0.2":  # Baseline
            self.baseline_metrics.append(metric)
        else:  # Candidate
            self.candidate_metrics.append(metric)

        # Periodically save to disk
        if len(self.baseline_metrics) % 100 == 0:
            self._save_metrics()

    def _save_metrics(self):
        """Save metrics to disk"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        baseline_file = self.metrics_dir / f"baseline_metrics_{timestamp}.json"
        with open(baseline_file, "w") as f:
            json.dump(self.baseline_metrics, f, indent=2)

        candidate_file = self.metrics_dir / f"candidate_metrics_{timestamp}.json"
        with open(candidate_file, "w") as f:
            json.dump(self.candidate_metrics, f, indent=2)

    def calculate_metrics(self, metrics_list: list[dict[str, Any]]) -> dict[str, Any]:
        """Calculate aggregate metrics from raw request data"""
        if not metrics_list:
            return {}

        total = len(metrics_list)
        successful = sum(1 for m in metrics_list if m["success"])
        failed = total - successful

        latencies = [m["latency_ms"] for m in metrics_list]
        latencies_sorted = sorted(latencies)

        # Calculate percentiles
        p50_idx = int(len(latencies_sorted) * 0.50)
        p95_idx = int(len(latencies_sorted) * 0.95)
        p99_idx = int(len(latencies_sorted) * 0.99)

        # Calculate accuracy if available
        accuracy = None
        correct_predictions = [m for m in metrics_list if m.get("correct_prediction") is not None]
        if correct_predictions:
            accuracy = sum(1 for m in correct_predictions if m["correct_prediction"]) / len(
                correct_predictions
            )

        return {
            "total_requests": total,
            "successful_requests": successful,
            "failed_requests": failed,
            "error_rate": failed / total if total > 0 else 0.0,
            "latency_mean_ms": statistics.mean(latencies) if latencies else 0.0,
            "latency_p50_ms": latencies_sorted[p50_idx] if latencies_sorted else 0.0,
            "latency_p95_ms": latencies_sorted[p95_idx] if latencies_sorted else 0.0,
            "latency_p99_ms": latencies_sorted[p99_idx] if latencies_sorted else 0.0,
            "accuracy": accuracy,
        }

    def get_baseline_metrics(self) -> dict[str, Any]:
        """Get aggregated baseline metrics"""
        return self.calculate_metrics(self.baseline_metrics)

    def get_candidate_metrics(self) -> dict[str, Any]:
        """Get aggregated candidate metrics"""
        return self.calculate_metrics(self.candidate_metrics)

    def compare_models(self) -> ABTestComparison:
        """
        Compare baseline and candidate model performance

        Returns:
            ABTestComparison with detailed analysis
        """
        baseline = self.get_baseline_metrics()
        candidate = self.get_candidate_metrics()

        # Calculate deltas (candidate - baseline)
        accuracy_delta = (candidate.get("accuracy") or 0) - (baseline.get("accuracy") or 0)
        latency_delta = candidate.get("latency_p99_ms", 0) - baseline.get("latency_p99_ms", 0)
        error_rate_delta = candidate.get("error_rate", 0) - baseline.get("error_rate", 0)

        # Determine winner
        # Candidate wins if: accuracy improved OR latency improved without accuracy loss
        candidate_wins = False
        confidence = 0.0

        if accuracy_delta > 0.02:  # 2% accuracy improvement
            candidate_wins = True
            confidence = min(0.95, 0.5 + accuracy_delta * 5)
        elif (
            accuracy_delta >= 0 and latency_delta < -10
        ):  # Same accuracy, 10ms+ latency improvement
            candidate_wins = True
            confidence = 0.7
        elif (
            accuracy_delta >= -0.01 and latency_delta < -20
        ):  # <1% accuracy loss, 20ms+ latency improvement
            candidate_wins = True
            confidence = 0.6

        winner = "candidate" if candidate_wins else "baseline"

        # Generate recommendation
        if candidate_wins:
            recommendation = "Promote candidate to production"
        elif accuracy_delta < -0.05:
            recommendation = "Reject candidate - significant accuracy degradation"
        elif error_rate_delta > 0.02:
            recommendation = "Reject candidate - higher error rate"
        else:
            recommendation = "Continue monitoring - insufficient difference"

        comparison = ABTestComparison(
            deployment_id=self.deployment_id,
            baseline_metrics=baseline,
            candidate_metrics=candidate,
            accuracy_delta=round(accuracy_delta, 4),
            latency_delta_ms=round(latency_delta, 2),
            error_rate_delta=round(error_rate_delta, 4),
            winner=winner,
            confidence=round(confidence, 2),
            recommendation=recommendation,
            collected_at=datetime.utcnow().isoformat() + "Z",
        )

        # Save comparison
        self._save_comparison(comparison)

        return comparison

    def _save_comparison(self, comparison: ABTestComparison):
        """Save comparison results to file"""
        filename = (
            self.metrics_dir / f"ab_comparison_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(filename, "w") as f:
            json.dump(asdict(comparison), f, indent=2)


if __name__ == "__main__":
    # Test the metrics collector
    import random

    collector = ABMetricsCollector(deployment_id="test-deployment-001")

    print("A/B Metrics Collector Test")
    print("=" * 50)

    # Simulate baseline requests
    for _i in range(200):
        collector.record_request(
            model_version="v1.0.2",
            latency_ms=random.gauss(50, 10),
            success=random.random() > 0.02,
            correct_prediction=random.random() > 0.28,
        )

    # Simulate candidate requests
    for _i in range(200):
        collector.record_request(
            model_version="v1.0.3",
            latency_ms=random.gauss(45, 8),
            success=random.random() > 0.03,
            correct_prediction=random.random() > 0.25,
        )

    # Get metrics
    baseline = collector.get_baseline_metrics()
    candidate = collector.get_candidate_metrics()

    print(f"Baseline: {baseline['total_requests']} requests, accuracy={baseline.get('accuracy')}")
    print(
        f"Candidate: {candidate['total_requests']} requests, accuracy={candidate.get('accuracy')}"
    )

    # Compare
    comparison = collector.compare_models()
    print(f"\nWinner: {comparison.winner}")
    print(f"Confidence: {comparison.confidence}")
    print(f"Recommendation: {comparison.recommendation}")
