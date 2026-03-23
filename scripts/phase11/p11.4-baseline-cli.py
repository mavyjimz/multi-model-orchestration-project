#!/usr/bin/env python3
"""
Baseline Management CLI Tool
Command-line interface for baseline and trend analysis
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

from baseline.rolling_window import RollingWindowCalculator
from baseline.baseline_comparator import BaselineComparator
from baseline.trend_analyzer import TrendAnalyzer


def cmd_calculate_windows(args):
    """Calculate all rolling window statistics"""
    calculator = RollingWindowCalculator()
    
    # Load data if file provided
    if args.data_file:
        with open(args.data_file, 'r') as f:
            data = json.load(f)
        for dp in data:
            calculator.add_data_point(**dp)
    
    results = calculator.calculate_all_windows()
    
    output = {}
    for window_name, stats in results.items():
        if stats:
            output[window_name] = {
                "sample_count": stats.sample_count,
                "accuracy_mean": stats.metrics.get("accuracy", {}).get("mean", 0),
                "latency_p99": stats.metrics.get("latency_ms", {}).get("p99", 0),
                "error_rate": stats.metrics.get("error_rate", {}).get("mean", 0)
            }
        else:
            output[window_name] = {"status": "insufficient_data"}
    
    print(json.dumps(output, indent=2))
    return 0


def cmd_get_baseline(args):
    """Get current baseline"""
    calculator = RollingWindowCalculator()
    baseline = calculator.get_current_baseline()
    print(json.dumps(baseline, indent=2))
    return 0


def cmd_compare(args):
    """Compare current vs baseline"""
    comparator = BaselineComparator()
    comparison = comparator.compare(
        baseline_window=args.baseline_window,
        current_window=args.current_window
    )
    
    print(json.dumps({
        "comparison_id": comparison.comparison_id,
        "status": comparison.status,
        "metrics_comparison": comparison.metrics_comparison,
        "recommendations": comparison.recommendations
    }, indent=2))
    
    return 0


def cmd_trend(args):
    """Analyze performance trends"""
    analyzer = TrendAnalyzer()
    
    if args.data_file:
        analyzer.load_data(args.data_file)
    
    if args.metric:
        forecast = analyzer.analyze_trend(args.metric)
        if forecast:
            print(json.dumps({
                "metric": forecast.metric_name,
                "current_value": forecast.current_value,
                "trend_direction": forecast.trend_direction,
                "trend_strength": forecast.trend_strength,
                "forecast_7d": forecast.forecast_7d,
                "forecast_14d": forecast.forecast_14d,
                "confidence": forecast.confidence,
                "alert_recommended": forecast.alert_recommended
            }, indent=2))
        else:
            print("Insufficient data for trend analysis")
            return 1
    else:
        summary = analyzer.generate_summary_report()
        print(json.dumps(summary, indent=2))
    
    return 0


def cmd_export(args):
    """Export baseline data"""
    calculator = RollingWindowCalculator()
    output_file = calculator.export_data(args.output)
    print(f"Data exported to: {output_file}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Baseline Management CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # calculate-windows command
    p_calc = subparsers.add_parser("calculate-windows", help="Calculate rolling windows")
    p_calc.add_argument("--data-file", type=str, help="Input data file")
    p_calc.set_defaults(func=cmd_calculate_windows)
    
    # get-baseline command
    p_base = subparsers.add_parser("get-baseline", help="Get current baseline")
    p_base.set_defaults(func=cmd_get_baseline)
    
    # compare command
    p_cmp = subparsers.add_parser("compare", help="Compare current vs baseline")
    p_cmp.add_argument("--baseline-window", default="long_term", help="Baseline window")
    p_cmp.add_argument("--current-window", default="short_term", help="Current window")
    p_cmp.set_defaults(func=cmd_compare)
    
    # trend command
    p_trend = subparsers.add_parser("trend", help="Analyze trends")
    p_trend.add_argument("--data-file", type=str, help="Input data file")
    p_trend.add_argument("--metric", type=str, help="Specific metric to analyze")
    p_trend.set_defaults(func=cmd_trend)
    
    # export command
    p_export = subparsers.add_parser("export", help="Export data")
    p_export.add_argument("--output", type=str, help="Output file path")
    p_export.set_defaults(func=cmd_export)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
