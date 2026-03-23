"""
Trend Analyzer
Analyzes performance trends and forecasts future metrics
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
import statistics


@dataclass
class TrendForecast:
    """Forecast results for a metric"""
    metric_name: str
    current_value: float
    trend_direction: str  # improving, stable, degrading
    trend_strength: float  # 0-1
    forecast_7d: float
    forecast_14d: float
    confidence: float
    alert_recommended: bool


class TrendAnalyzer:
    """
    Analyzes performance trends and generates forecasts
    
    Uses linear regression for trend detection and forecasting.
    """
    
    def __init__(self, data_dir: str = "results/phase11/baseline_data"):
        self.data_dir = Path(data_dir)
        self.forecasts_dir = self.data_dir / "forecasts"
        self.forecasts_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_points: List[Dict[str, Any]] = []
    
    def load_data(self, data_file: str = None) -> int:
        """Load data points from file"""
        if data_file is None:
            # Find most recent data export
            exports = list(self.data_dir.glob("data_export_*.json"))
            if not exports:
                return 0
            data_file = max(exports, key=lambda p: p.stat().st_mtime)
        
        with open(data_file, 'r') as f:
            self.data_points = json.load(f)
        
        return len(self.data_points)
    
    def _linear_regression(
        self,
        x_values: List[float],
        y_values: List[float]
    ) -> Tuple[float, float]:
        """
        Simple linear regression
        
        Returns:
            Tuple of (slope, intercept)
        """
        n = len(x_values)
        if n < 2:
            return 0.0, y_values[0] if y_values else 0.0
        
        x_mean = statistics.mean(x_values)
        y_mean = statistics.mean(y_values)
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        
        if denominator == 0:
            return 0.0, y_mean
        
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        
        return slope, intercept
    
    def analyze_trend(
        self,
        metric_name: str,
        threshold_degrading: float = -0.001,
        threshold_improving: float = 0.001
    ) -> Optional[TrendForecast]:
        """
        Analyze trend for a specific metric
        
        Args:
            metric_name: Name of the metric to analyze
            threshold_degrading: Slope threshold for degrading trend
            threshold_improving: Slope threshold for improving trend
        
        Returns:
            TrendForecast or None if insufficient data
        """
        if len(self.data_points) < 7:
            return None
        
        # Extract metric values with timestamps
        data = []
        for dp in self.data_points:
            if metric_name in dp or metric_name.replace("_", ".") in dp:
                try:
                    ts = datetime.fromisoformat(dp["timestamp"].replace('Z', '+00:00'))
                    value = dp.get(metric_name, dp.get(metric_name.replace("_", ".")))
                    if value is not None:
                        data.append((ts, float(value)))
                except (ValueError, TypeError):
                    continue
        
        if len(data) < 7:
            return None
        
        # Sort by timestamp
        data.sort(key=lambda x: x[0])
        
        # Convert to numeric x values (days from first point)
        first_ts = data[0][0]
        x_values = [(ts - first_ts).total_seconds() / 86400 for ts, _ in data]
        y_values = [value for _, value in data]
        
        # Calculate regression
        slope, intercept = self._linear_regression(x_values, y_values)
        
        # Determine trend direction
        if slope < threshold_degrading:
            trend_direction = "degrading"
        elif slope > threshold_improving:
            trend_direction = "improving"
        else:
            trend_direction = "stable"
        
        # Calculate trend strength (normalized slope)
        y_mean = statistics.mean(y_values)
        trend_strength = min(1.0, abs(slope) / (y_mean * 0.1)) if y_mean > 0 else 0.0
        
        # Forecast
        last_x = x_values[-1]
        forecast_7d = intercept + slope * (last_x + 7)
        forecast_14d = intercept + slope * (last_x + 14)
        
        # Confidence based on R-squared (simplified)
        y_pred = [intercept + slope * x for x in x_values]
        ss_res = sum((y - yp) ** 2 for y, yp in zip(y_values, y_pred))
        ss_tot = sum((y - y_mean) ** 2 for y in y_values)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        confidence = max(0.0, min(1.0, r_squared))
        
        # Alert recommendation
        alert_recommended = (
            (trend_direction == "degrading" and trend_strength > 0.5) or
            (forecast_14d < y_mean * 0.9 if "accuracy" in metric_name else forecast_14d > y_mean * 1.2)
        )
        
        current_value = y_values[-1]
        
        forecast = TrendForecast(
            metric_name=metric_name,
            current_value=round(current_value, 4),
            trend_direction=trend_direction,
            trend_strength=round(trend_strength, 4),
            forecast_7d=round(forecast_7d, 4),
            forecast_14d=round(forecast_14d, 4),
            confidence=round(confidence, 4),
            alert_recommended=alert_recommended
        )
        
        self._save_forecast(forecast)
        
        return forecast
    
    def _save_forecast(self, forecast: TrendForecast):
        """Save forecast to file"""
        filename = self.forecasts_dir / f"forecast_{forecast.metric_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(asdict(forecast), f, indent=2)
    
    def analyze_all_metrics(self) -> Dict[str, Optional[TrendForecast]]:
        """Analyze trends for all standard metrics"""
        metrics = ["accuracy", "latency_ms", "error_rate"]
        results = {}
        
        for metric in metrics:
            forecast = self.analyze_trend(metric)
            results[metric] = forecast
        
        return results
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate summary report of all trends"""
        forecasts = self.analyze_all_metrics()
        
        summary = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "data_points_analyzed": len(self.data_points),
            "forecasts": {},
            "alerts": [],
            "overall_health": "healthy"
        }
        
        degrading_count = 0
        
        for metric, forecast in forecasts.items():
            if forecast:
                summary["forecasts"][metric] = asdict(forecast)
                
                if forecast.trend_direction == "degrading":
                    degrading_count += 1
                    
                    if forecast.alert_recommended:
                        summary["alerts"].append({
                            "metric": metric,
                            "severity": "warning" if forecast.trend_strength < 0.7 else "critical",
                            "message": f"{metric} showing {forecast.trend_direction} trend (strength: {forecast.trend_strength})"
                        })
        
        # Determine overall health
        if degrading_count >= 2:
            summary["overall_health"] = "critical"
        elif degrading_count == 1:
            summary["overall_health"] = "warning"
        
        # Save report
        report_file = self.forecasts_dir / f"trend_summary_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary


if __name__ == "__main__":
    import random
    
    analyzer = TrendAnalyzer()
    
    print("Trend Analyzer Test")
    print("=" * 50)
    
    # Simulate data with slight degradation trend
    base_date = datetime.utcnow()
    for i in range(30):
        days_ago = 29 - i
        timestamp = (base_date - timedelta(days=days_ago)).isoformat()
        
        # Accuracy slowly decreasing
        accuracy = 0.75 - (i * 0.002) + random.gauss(0, 0.02)
        
        analyzer.data_points.append({
            "timestamp": timestamp,
            "accuracy": accuracy,
            "latency_ms": 50 + random.gauss(0, 10),
            "error_rate": 0.02 + (i * 0.001) + random.gauss(0, 0.005)
        })
    
    # Analyze trends
    forecasts = analyzer.analyze_all_metrics()
    
    for metric, forecast in forecasts.items():
        if forecast:
            print(f"\n{metric}:")
            print(f"  Current: {forecast.current_value}")
            print(f"  Trend: {forecast.trend_direction} (strength: {forecast.trend_strength})")
            print(f"  7-day forecast: {forecast.forecast_7d}")
            print(f"  14-day forecast: {forecast.forecast_14d}")
            print(f"  Alert recommended: {forecast.alert_recommended}")
    
    # Generate summary
    summary = analyzer.generate_summary_report()
    print(f"\nOverall Health: {summary['overall_health']}")
    if summary['alerts']:
        print("Alerts:")
        for alert in summary['alerts']:
            print(f"  - [{alert['severity']}] {alert['message']}")
