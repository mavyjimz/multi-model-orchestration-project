"""
Alert Rules - Phase 9.7
Define and evaluate alerting rules for error rates and latency spikes.
"""

from log_aggregator import aggregate_logs
from latency_tracker import calculate_percentiles

# Alert thresholds
ERROR_RATE_THRESHOLD = 0.05  # 5% error rate
LATENCY_P95_THRESHOLD = 100  # 100ms P95 latency

def check_error_rate(entries):
    """Check if error rate exceeds threshold."""
    import json
    # Parse entries if they are strings
    parsed_entries = []
    for e in entries:
        if isinstance(e, str):
            try:
                parsed_entries.append(json.loads(e))
            except json.JSONDecodeError:
                continue
        else:
            parsed_entries.append(e)

    total = len(entries)
    errors = sum(1 for e in parsed_entries if isinstance(e, dict) and e.get('status_code', 200) >= 400)
    error_rate = errors / total if total > 0 else 0
    triggered = error_rate > ERROR_RATE_THRESHOLD
    return {
        'rule': 'error_rate',
        'threshold': ERROR_RATE_THRESHOLD,
        'current': error_rate,
        'triggered': triggered
    }

def check_latency(entries):
    """Check if P95 latency exceeds threshold."""
    percentiles = calculate_percentiles(entries)
    triggered = percentiles['p95'] > LATENCY_P95_THRESHOLD
    return {
        'rule': 'latency_p95',
        'threshold': LATENCY_P95_THRESHOLD,
        'current': percentiles['p95'],
        'triggered': triggered
    }

def evaluate_all_rules():
    """Evaluate all alert rules."""
    entries = aggregate_logs()
    results = [check_error_rate(entries), check_latency(entries)]

    for result in results:
        status = "TRIGGERED" if result['triggered'] else "OK"
        print(f"[{status}] {result['rule']}: {result['current']:.4f} (threshold: {result['threshold']})")

    return results

if __name__ == "__main__":
    evaluate_all_rules()
