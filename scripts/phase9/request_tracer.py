"""
Request Tracer - Phase 9.5
Filter logs by correlation_id for end-to-end tracing.
"""

import json
import sys
from log_aggregator import aggregate_logs, filter_by_correlation_id

def trace_request(correlation_id):
    """Trace a request by correlation ID."""
    all_entries = aggregate_logs()
    filtered = filter_by_correlation_id(all_entries, correlation_id)

    if not filtered:
        print(f"No entries found for correlation_id: {correlation_id}")
        return []

    print(f"Found {len(filtered)} entries for correlation_id: {correlation_id}")
    for entry in sorted(filtered, key=lambda x: x.get('timestamp', '')):
        print(json.dumps(entry, indent=2))

    return filtered

if __name__ == "__main__":
    if len(sys.argv) > 1:
        correlation_id = sys.argv[1]
        trace_request(correlation_id)
    else:
        print("Usage: python request_tracer.py <correlation_id>")
