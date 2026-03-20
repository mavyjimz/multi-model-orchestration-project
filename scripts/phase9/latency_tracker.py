"""
Latency Tracker - Phase 9.6
Calculate P50/P95/P99 latency percentiles.
"""

import numpy as np
from log_aggregator import aggregate_logs

def calculate_percentiles(entries):
    """Calculate P50, P95, P99 latency from log entries."""
    latencies = [e.get('latency_ms', 0) for e in entries if 'latency_ms' in e]

    if not latencies:
        return {'p50': 0, 'p95': 0, 'p99': 0, 'count': 0, 'avg': 0, 'max': 0, 'min': 0}

    return {
        'p50': float(np.percentile(latencies, 50)),
        'p95': float(np.percentile(latencies, 95)),
        'p99': float(np.percentile(latencies, 99)),
        'count': len(latencies),
        'avg': float(np.mean(latencies)),
        'max': float(np.max(latencies)),
        'min': float(np.min(latencies))
    }

if __name__ == "__main__":
    entries = aggregate_logs()
    percentiles = calculate_percentiles(entries)
    print("Latency Percentiles:")
    print(f"  P50: {percentiles['p50']:.2f}ms")
    print(f"  P95: {percentiles['p95']:.2f}ms")
    print(f"  P99: {percentiles['p99']:.2f}ms")
    print(f"  Avg: {percentiles['avg']:.2f}ms")
    print(f"  Count: {percentiles['count']}")
