"""
Structured Log Aggregator - Phase 9.4
Parses JSON logs and aggregates for dashboard view.
"""

import json
import os
from datetime import datetime
from collections import defaultdict

LOG_DIR = "logs"

def parse_log_file(filepath):
    """Parse a JSON log file and return structured entries."""
    entries = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    entries.append(entry)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass
    return entries

def aggregate_logs():
    """Aggregate all logs from the logs directory."""
    all_entries = []
    for filename in os.listdir(LOG_DIR):
        if filename.endswith('.log'):
            filepath = os.path.join(LOG_DIR, filename)
            entries = parse_log_file(filepath)
            all_entries.extend(entries)
    return all_entries

def filter_by_correlation_id(entries, correlation_id):
    """Filter log entries by correlation ID for request tracing."""
    return [e for e in entries if e.get('correlation_id') == correlation_id]

def get_error_summary(entries):
    """Get summary of errors from log entries."""
    error_count = sum(1 for e in entries if e.get('level') == 'error')
    return {'total_entries': len(entries), 'error_count': error_count}

if __name__ == "__main__":
    entries = aggregate_logs()
    summary = get_error_summary(entries)
    print(f"Total entries: {summary['total_entries']}")
    print(f"Error count: {summary['error_count']}")
