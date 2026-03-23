"""
Rolling Window Calculator
Calculates rolling statistics for model performance metrics
"""

import json
import statistics
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


@dataclass
class WindowStats:
    """Statistics for a rolling window"""

    window_name: str
    window_days: int
    start_date: str
    end_date: str
    sample_count: int
    metrics: dict[str, Any]
    calculated_at: str


class RollingWindowCalculator:
    """
    Calculates rolling window statistics for performance metrics

    Supports multiple window sizes (7-day, 30-day, 90-day) with
    automatic metric aggregation and trend calculation.
    """

    def __init__(self, config_path: str = "src/baseline/baseline_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()

        self.data_dir = Path("results/phase11/baseline_data")
        self.windows_dir = self.data_dir / "windows"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.windows_dir.mkdir(parents=True, exist_ok=True)

        # In-memory storage for recent data points
        self.data_points: deque = deque(maxlen=10000)

    def _load_config(self) -> dict[str, Any]:
        """Load configuration from YAML file"""
        import yaml

        if not self.config_path.exists():
            return self._default_config()

        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def _default_config(self) -> dict[str, Any]:
        """Return default configuration"""
        return {
            "rolling_windows": {
                "short_term": {"window_days": 7, "min_samples": 100},
                "medium_term": {"window_days": 30, "min_samples": 500},
                "long_term": {"window_days": 90, "min_samples": 2000},
            }
        }

    def add_data_point(
        self,
        timestamp: str,
        accuracy: float,
        latency_ms: float,
        error_rate: float,
        model_version: str,
        request_count: int = 1,
    ):
        """
        Add a new data point to the rolling window

        Args:
            timestamp: ISO 8601 timestamp
            accuracy: Model accuracy (0-1)
            latency_ms: Request latency in milliseconds
            error_rate: Error rate (0-1)
            model_version: Model version string
            request_count: Number of requests this data point represents
        """
        data_point = {
            "timestamp": timestamp,
            "accuracy": accuracy,
            "latency_ms": latency_ms,
            "error_rate": error_rate,
            "model_version": model_version,
            "request_count": request_count,
        }

        self.data_points.append(data_point)

    def _filter_by_window(self, window_days: int) -> list[dict[str, Any]]:
        """Filter data points within the specified window"""
        cutoff = datetime.utcnow() - timedelta(days=window_days)
        cutoff_str = cutoff.isoformat()

        return [dp for dp in self.data_points if dp["timestamp"] >= cutoff_str]

    def _calculate_metrics(self, data_points: list[dict[str, Any]]) -> dict[str, Any]:
        """Calculate aggregate metrics from data points"""
        if not data_points:
            return {}

        accuracies = [dp["accuracy"] for dp in data_points]
        latencies = [dp["latency_ms"] for dp in data_points]
        error_rates = [dp["error_rate"] for dp in data_points]

        total_requests = sum(dp.get("request_count", 1) for dp in data_points)

        # Calculate statistics
        def safe_mean(values):
            return statistics.mean(values) if values else 0.0

        def safe_std(values):
            return statistics.stdev(values) if len(values) > 1 else 0.0

        def percentile(values, p):
            if not values:
                return 0.0
            sorted_values = sorted(values)
            k = (len(sorted_values) - 1) * p / 100
            f = int(k)
            c = f + 1 if f + 1 < len(sorted_values) else f
            return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)

        return {
            "accuracy": {
                "mean": round(safe_mean(accuracies), 4),
                "std": round(safe_std(accuracies), 4),
                "min": round(min(accuracies), 4) if accuracies else 0.0,
                "max": round(max(accuracies), 4) if accuracies else 0.0,
                "p50": round(percentile(accuracies, 50), 4),
                "p95": round(percentile(accuracies, 95), 4),
            },
            "latency_ms": {
                "mean": round(safe_mean(latencies), 2),
                "std": round(safe_std(latencies), 2),
                "min": round(min(latencies), 2) if latencies else 0.0,
                "max": round(max(latencies), 2) if latencies else 0.0,
                "p50": round(percentile(latencies, 50), 2),
                "p95": round(percentile(latencies, 95), 2),
                "p99": round(percentile(latencies, 99), 2),
            },
            "error_rate": {
                "mean": round(safe_mean(error_rates), 4),
                "std": round(safe_std(error_rates), 4),
                "min": round(min(error_rates), 4) if error_rates else 0.0,
                "max": round(max(error_rates), 4) if error_rates else 0.0,
            },
            "total_requests": total_requests,
            "sample_count": len(data_points),
        }

    def calculate_window(
        self, window_name: str, window_days: int, min_samples: int = 0
    ) -> WindowStats | None:
        """
        Calculate statistics for a specific rolling window

        Args:
            window_name: Name of the window (e.g., "short_term")
            window_days: Number of days in the window
            min_samples: Minimum samples required

        Returns:
            WindowStats object or None if insufficient data
        """
        data_points = self._filter_by_window(window_days)

        if len(data_points) < min_samples:
            return None

        metrics = self._calculate_metrics(data_points)

        cutoff = datetime.utcnow() - timedelta(days=window_days)

        stats = WindowStats(
            window_name=window_name,
            window_days=window_days,
            start_date=cutoff.isoformat(),
            end_date=datetime.utcnow().isoformat() + "Z",
            sample_count=len(data_points),
            metrics=metrics,
            calculated_at=datetime.utcnow().isoformat() + "Z",
        )

        # Save to file
        self._save_window_stats(stats)

        return stats

    def _save_window_stats(self, stats: WindowStats):
        """Save window statistics to file"""
        filename = (
            self.windows_dir
            / f"{stats.window_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(filename, "w") as f:
            json.dump(asdict(stats), f, indent=2)

    def calculate_all_windows(self) -> dict[str, WindowStats | None]:
        """Calculate statistics for all configured windows"""
        results = {}

        windows = self.config.get("rolling_windows", {})

        for window_name, window_config in windows.items():
            stats = self.calculate_window(
                window_name=window_name,
                window_days=window_config["window_days"],
                min_samples=window_config["min_samples"],
            )
            results[window_name] = stats

        return results

    def get_current_baseline(self) -> dict[str, Any]:
        """Get current baseline from the most recent long-term window"""
        stats = self.calculate_window(window_name="long_term", window_days=90, min_samples=100)

        if stats is None:
            return {"status": "insufficient_data"}

        return {
            "status": "available",
            "accuracy": stats.metrics.get("accuracy", {}).get("mean", 0.0),
            "latency_p99_ms": stats.metrics.get("latency_ms", {}).get("p99", 0.0),
            "error_rate": stats.metrics.get("error_rate", {}).get("mean", 0.0),
            "sample_count": stats.sample_count,
            "calculated_at": stats.calculated_at,
        }

    def export_data(self, output_path: str = None) -> str:
        """Export all data points to JSON file"""
        if output_path is None:
            output_path = str(
                self.data_dir / f"data_export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            )

        with open(output_path, "w") as f:
            json.dump(list(self.data_points), f, indent=2)

        return output_path


if __name__ == "__main__":
    import random

    calculator = RollingWindowCalculator()

    print("Rolling Window Calculator Test")
    print("=" * 50)

    # Simulate data points over 30 days
    base_date = datetime.utcnow()
    for _i in range(500):
        days_ago = random.randint(0, 30)
        timestamp = (base_date - timedelta(days=days_ago)).isoformat()

        calculator.add_data_point(
            timestamp=timestamp,
            accuracy=0.70 + random.gauss(0, 0.05),
            latency_ms=50 + random.gauss(0, 15),
            error_rate=0.02 + random.gauss(0, 0.01),
            model_version="v1.0.2",
            request_count=random.randint(10, 100),
        )

    # Calculate all windows
    results = calculator.calculate_all_windows()

    for window_name, stats in results.items():
        if stats:
            print(f"\n{window_name} ({stats.window_days} days):")
            print(f"  Samples: {stats.sample_count}")
            print(f"  Accuracy: {stats.metrics['accuracy']['mean']:.4f}")
            print(f"  Latency P99: {stats.metrics['latency_ms']['p99']:.2f}ms")
        else:
            print(f"\n{window_name}: Insufficient data")

    # Get baseline
    baseline = calculator.get_current_baseline()
    print(f"\nCurrent Baseline: {json.dumps(baseline, indent=2)}")
