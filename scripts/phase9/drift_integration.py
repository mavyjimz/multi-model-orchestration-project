"""
Drift Integration - Phase 9.8
Connect PSI/KS drift metrics from Phase 5 to observability dashboard.
"""

import json
import os

DRIFT_RESULTS_PATH = "results/phase5/drift_detection_psi.json"

def load_drift_results():
    """Load drift detection results from Phase 5."""
    if os.path.exists(DRIFT_RESULTS_PATH):
        with open(DRIFT_RESULTS_PATH, 'r') as f:
            return json.load(f)
    return None

def get_drift_status(psi_score, threshold=0.2):
    """Determine drift status based on PSI score."""
    if psi_score < 0.1:
        return "STABLE"
    elif psi_score < threshold:
        return "WARNING"
    else:
        return "DRIFT_DETECTED"

def export_drift_metrics():
    """Export drift metrics for Prometheus."""
    results = load_drift_results()
    if results:
        psi = results.get('psi_score', 0)
        ks = results.get('ks_score', 0)
        status = get_drift_status(psi)
        print(f"PSI Score: {psi:.4f}")
        print(f"KS Score: {ks:.4f}")
        print(f"Status: {status}")
        return {'psi': psi, 'ks': ks, 'status': status}
    else:
        print("No drift results found")
        return None

if __name__ == "__main__":
    export_drift_metrics()
