"""
Baseline Comparator
Compares current performance against baseline and detects drift
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from baseline.rolling_window import RollingWindowCalculator, WindowStats


class ComparisonStatus(Enum):
    """Status of baseline comparison"""
    WITHIN_BASELINE = "within_baseline"
    WARNING = "warning"
    DEGRADED = "degraded"
    IMPROVED = "improved"


@dataclass
class BaselineComparison:
    """Results of comparing current performance to baseline"""
    comparison_id: str
    timestamp: str
    baseline_window: str
    current_window: str
    status: str
    metrics_comparison: Dict[str, Any]
    recommendations: List[str]


class BaselineComparator:
    """
    Compares current performance against established baseline
    
    Detects performance drift and generates recommendations
    for baseline updates or model retraining.
    """
    
    def __init__(
        self,
        calculator: Optional[RollingWindowCalculator] = None,
        config_path: str = "src/baseline/baseline_config.yaml"
    ):
        self.calculator = calculator or RollingWindowCalculator(config_path)
        self.config = self.calculator.config
        self.config_path = Path(config_path)
        
        self.comparisons_dir = Path("results/phase11/baseline_comparisons")
        self.comparisons_dir.mkdir(parents=True, exist_ok=True)
        
        self.comparison_history: List[BaselineComparison] = []
    
    def _generate_comparison_id(self) -> str:
        """Generate unique comparison ID"""
        import hashlib
        timestamp = datetime.utcnow().isoformat()
        content = f"comparison:{timestamp}"
        hash_suffix = hashlib.sha256(content.encode()).hexdigest()[:8]
        return f"cmp-{hash_suffix}"
    
    def compare(
        self,
        baseline_window: str = "long_term",
        current_window: str = "short_term"
    ) -> BaselineComparison:
        """
        Compare current performance against baseline
        
        Args:
            baseline_window: Window to use as baseline
            current_window: Window for current performance
        
        Returns:
            BaselineComparison with detailed analysis
        """
        import yaml
        
        # Load config
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        baseline_config = config["rolling_windows"].get(baseline_window, {})
        current_config = config["rolling_windows"].get(current_window, {})
        
        # Calculate window stats
        baseline_stats = self.calculator.calculate_window(
            window_name=baseline_window,
            window_days=baseline_config.get("window_days", 90),
            min_samples=baseline_config.get("min_samples", 100)
        )
        
        current_stats = self.calculator.calculate_window(
            window_name=current_window,
            window_days=current_config.get("window_days", 7),
            min_samples=current_config.get("min_samples", 100)
        )
        
        if baseline_stats is None or current_stats is None:
            return BaselineComparison(
                comparison_id=self._generate_comparison_id(),
                timestamp=datetime.utcnow().isoformat() + "Z",
                baseline_window=baseline_window,
                current_window=current_window,
                status="insufficient_data",
                metrics_comparison={"error": "Insufficient data for comparison"},
                recommendations=["Collect more data before comparing"]
            )
        
        # Compare metrics
        metrics_comparison = self._compare_metrics(baseline_stats, current_stats)
        
        # Determine status
        status = self._determine_status(metrics_comparison)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(status, metrics_comparison)
        
        comparison = BaselineComparison(
            comparison_id=self._generate_comparison_id(),
            timestamp=datetime.utcnow().isoformat() + "Z",
            baseline_window=baseline_window,
            current_window=current_window,
            status=status.value,
            metrics_comparison=metrics_comparison,
            recommendations=recommendations
        )
        
        self.comparison_history.append(comparison)
        self._save_comparison(comparison)
        
        return comparison
    
    def _compare_metrics(
        self,
        baseline: WindowStats,
        current: WindowStats
    ) -> Dict[str, Any]:
        """Compare metrics between baseline and current"""
        baseline_acc = baseline.metrics.get("accuracy", {}).get("mean", 0.0)
        current_acc = current.metrics.get("accuracy", {}).get("mean", 0.0)
        accuracy_delta = current_acc - baseline_acc
        
        baseline_lat = baseline.metrics.get("latency_ms", {}).get("p99", 0.0)
        current_lat = current.metrics.get("latency_ms", {}).get("p99", 0.0)
        latency_delta = current_lat - baseline_lat
        
        baseline_err = baseline.metrics.get("error_rate", {}).get("mean", 0.0)
        current_err = current.metrics.get("error_rate", {}).get("mean", 0.0)
        error_rate_delta = current_err - baseline_err
        
        return {
            "accuracy": {
                "baseline": round(baseline_acc, 4),
                "current": round(current_acc, 4),
                "delta": round(accuracy_delta, 4),
                "delta_percentage": round(accuracy_delta / baseline_acc * 100, 2) if baseline_acc > 0 else 0.0
            },
            "latency_p99_ms": {
                "baseline": round(baseline_lat, 2),
                "current": round(current_lat, 2),
                "delta": round(latency_delta, 2),
                "delta_percentage": round(latency_delta / baseline_lat * 100, 2) if baseline_lat > 0 else 0.0
            },
            "error_rate": {
                "baseline": round(baseline_err, 4),
                "current": round(current_err, 4),
                "delta": round(error_rate_delta, 4),
                "delta_percentage": round(error_rate_delta / baseline_err * 100, 2) if baseline_err > 0 else 0.0
            },
            "baseline_samples": baseline.sample_count,
            "current_samples": current.sample_count
        }
    
    def _determine_status(self, metrics: Dict[str, Any]) -> ComparisonStatus:
        """Determine comparison status based on metrics"""
        rules = self.config.get("baseline_update_rules", {})
        
        acc_delta = metrics["accuracy"]["delta"]
        lat_delta = metrics["latency_p99_ms"]["delta"]
        err_delta = metrics["error_rate"]["delta"]
        
        # Check for improvement
        if acc_delta > rules.get("accuracy_improvement_threshold", 0.03):
            return ComparisonStatus.IMPROVED
        
        # Check for degradation
        if acc_delta < -rules.get("accuracy_improvement_threshold", 0.03) * 2:
            return ComparisonStatus.DEGRADED
        
        if err_delta > rules.get("accuracy_improvement_threshold", 0.03):
            return ComparisonStatus.DEGRADED
        
        # Check for warning
        if acc_delta < -0.01 or err_delta > 0.01 or lat_delta > 20:
            return ComparisonStatus.WARNING
        
        return ComparisonStatus.WITHIN_BASELINE
    
    def _generate_recommendations(
        self,
        status: ComparisonStatus,
        metrics: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on comparison results"""
        recommendations = []
        
        if status == ComparisonStatus.IMPROVED:
            recommendations.append(
                "Performance has improved significantly. Consider updating baseline."
            )
            recommendations.append(
                "Document changes that led to improvement for future reference."
            )
        
        elif status == ComparisonStatus.DEGRADED:
            recommendations.append(
                "Performance degradation detected. Investigate root cause immediately."
            )
            recommendations.append(
                "Consider triggering model retraining (Phase 11.2)."
            )
            recommendations.append(
                "Review recent deployments or data changes."
            )
        
        elif status == ComparisonStatus.WARNING:
            recommendations.append(
                "Minor performance changes detected. Continue monitoring."
            )
            recommendations.append(
                "Set up alerts for further degradation."
            )
        
        else:
            recommendations.append(
                "Performance within expected baseline. No action required."
            )
        
        return recommendations
    
    def _save_comparison(self, comparison: BaselineComparison):
        """Save comparison to file"""
        filename = self.comparisons_dir / f"{comparison.comparison_id}.json"
        with open(filename, 'w') as f:
            json.dump(asdict(comparison), f, indent=2)
    
    def should_update_baseline(self, comparison: BaselineComparison) -> Tuple[bool, str]:
        """
        Determine if baseline should be updated
        
        Returns:
            Tuple of (should_update: bool, reason: str)
        """
        if comparison.status == "insufficient_data":
            return False, "Insufficient data"
        
        if comparison.status == "improved":
            return True, "Performance improvement detected"
        
        # Check staleness
        rules = self.config.get("baseline_update_rules", {})
        staleness_days = rules.get("staleness_days", 60)
        
        baseline_date = datetime.fromisoformat(comparison.timestamp.replace('Z', '+00:00'))
        days_since = (datetime.utcnow() - baseline_date.replace(tzinfo=None)).days
        
        if days_since > staleness_days:
            return True, f"Baseline is {days_since} days old (threshold: {staleness_days})"
        
        return False, "Baseline still valid"
    
    def get_comparison_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get comparison history"""
        return [asdict(c) for c in self.comparison_history[-limit:]]


if __name__ == "__main__":
    comparator = BaselineComparator()
    
    print("Baseline Comparator Test")
    print("=" * 50)
    
    comparison = comparator.compare()
    
    print(f"Status: {comparison.status}")
    print(f"Metrics Comparison:")
    print(json.dumps(comparison.metrics_comparison, indent=2))
    print(f"\nRecommendations:")
    for rec in comparison.recommendations:
        print(f"  - {rec}")
    
    should_update, reason = comparator.should_update_baseline(comparison)
    print(f"\nUpdate Baseline: {should_update}")
    print(f"Reason: {reason}")
