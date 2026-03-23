#!/usr/bin/env python3
"""
Baseline Updates Integration Tests
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
project_root = Path(__file__).parent.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

from baseline.rolling_window import RollingWindowCalculator
from baseline.baseline_comparator import BaselineComparator
from baseline.trend_analyzer import TrendAnalyzer


def test_rolling_window_calculation():
    """Test rolling window statistics calculation"""
    print("Test 1: Rolling Window Calculation")
    
    calculator = RollingWindowCalculator()
    
    # Add data points
    base_date = datetime.utcnow()
    for i in range(100):
        days_ago = i % 30
        timestamp = (base_date - timedelta(days=days_ago)).isoformat()
        
        calculator.add_data_point(
            timestamp=timestamp,
            accuracy=0.70 + (i % 10) * 0.01,
            latency_ms=50 + (i % 20),
            error_rate=0.02,
            model_version="v1.0.2",
            request_count=10
        )
    
    # Calculate windows
    results = calculator.calculate_all_windows()
    
    # Check short-term window has data
    assert results.get("short_term") is not None, "Short-term window should have data"
    
    stats = results["short_term"]
    assert stats.sample_count > 0
    assert "accuracy" in stats.metrics
    
    print(f"  Short-term samples: {stats.sample_count}")
    print(f"  Accuracy mean: {stats.metrics['accuracy']['mean']:.4f}")
    print("  ✓ Passed")


def test_baseline_comparison():
    """Test baseline comparison"""
    print("\nTest 2: Baseline Comparison")
    
    comparator = BaselineComparator()
    comparison = comparator.compare()
    
    assert comparison.comparison_id is not None
    assert comparison.status in ["insufficient_data", "within_baseline", "warning", "degraded", "improved"]
    assert "recommendations" in comparison.__dict__
    
    print(f"  Status: {comparison.status}")
    print(f"  Recommendations: {len(comparison.recommendations)}")
    print("  ✓ Passed")


def test_trend_analysis():
    """Test trend analysis"""
    print("\nTest 3: Trend Analysis")
    
    analyzer = TrendAnalyzer()
    
    # Add data with clear trend
    base_date = datetime.utcnow()
    for i in range(30):
        timestamp = (base_date - timedelta(days=29-i)).isoformat()
        analyzer.data_points.append({
            "timestamp": timestamp,
            "accuracy": 0.75 - (i * 0.003),  # Degrading trend
            "latency_ms": 50 + (i * 0.5),
            "error_rate": 0.02
        })
    
    forecast = analyzer.analyze_trend("accuracy")
    
    assert forecast is not None, "Should have forecast with 30 data points"
    assert forecast.trend_direction in ["improving", "stable", "degrading"]
    assert 0 <= forecast.confidence <= 1
    
    print(f"  Trend direction: {forecast.trend_direction}")
    print(f"  Confidence: {forecast.confidence}")
    print(f"  14-day forecast: {forecast.forecast_14d}")
    print("  ✓ Passed")


def test_baseline_update_decision():
    """Test baseline update decision logic"""
    print("\nTest 4: Baseline Update Decision")
    
    comparator = BaselineComparator()
    comparison = comparator.compare()
    
    should_update, reason = comparator.should_update_baseline(comparison)
    
    assert isinstance(should_update, bool)
    assert isinstance(reason, str)
    
    print(f"  Should update: {should_update}")
    print(f"  Reason: {reason}")
    print("  ✓ Passed")


def test_trend_summary():
    """Test trend summary generation"""
    print("\nTest 5: Trend Summary")
    
    analyzer = TrendAnalyzer()
    
    # Add varied data
    base_date = datetime.utcnow()
    for i in range(50):
        timestamp = (base_date - timedelta(days=49-i)).isoformat()
        analyzer.data_points.append({
            "timestamp": timestamp,
            "accuracy": 0.70 + (i % 5) * 0.01,
            "latency_ms": 50 + (i % 10),
            "error_rate": 0.02 + (i % 3) * 0.005
        })
    
    summary = analyzer.generate_summary_report()
    
    assert "generated_at" in summary
    assert "forecasts" in summary
    assert "overall_health" in summary
    assert summary["overall_health"] in ["healthy", "warning", "critical"]
    
    print(f"  Overall health: {summary['overall_health']}")
    print(f"  Alerts: {len(summary['alerts'])}")
    print("  ✓ Passed")


def main():
    print("=" * 60)
    print("Baseline Updates Integration Tests")
    print("=" * 60)
    
    try:
        test_rolling_window_calculation()
        test_baseline_comparison()
        test_trend_analysis()
        test_baseline_update_decision()
        test_trend_summary()
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\n[FAILED] {e}")
        return 1
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
