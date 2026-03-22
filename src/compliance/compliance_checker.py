"""GDPR compliance checker CLI."""

import json
import sys
from datetime import datetime

from src.compliance.data_retention import DataRetentionPolicy
from src.compliance.right_to_erase import RightToErasure


def run_compliance_check() -> dict:
    """Run full compliance check and return report."""
    retention = DataRetentionPolicy()
    erasure = RightToErasure()

    # Check retention policy
    old_files = retention.scan_for_old_files()

    # Check erasure requests
    pending_erasures = erasure.get_erasure_requests(status="pending")

    report = {
        "check_timestamp": datetime.utcnow().isoformat(),
        "retention_policy": {
            "retention_days": retention.retention_days,
            "files_exceeding_retention": len(old_files),
            "total_size_bytes": sum(f["size_bytes"] for f in old_files),
        },
        "erasure_requests": {
            "pending": len(pending_erasures),
            "requests": pending_erasures,
        },
        "compliance_status": "PASS"
        if len(old_files) == 0 and len(pending_erasures) == 0
        else "REVIEW_REQUIRED",
    }

    return report


if __name__ == "__main__":
    report = run_compliance_check()
    print(json.dumps(report, indent=2))

    # Exit with error code if compliance issues found
    if report["compliance_status"] != "PASS":
        sys.exit(1)
    sys.exit(0)
