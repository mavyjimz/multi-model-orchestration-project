"""
Drift Integration Module
Connects retraining triggers with Phase 5 drift detection results
"""

import json
from pathlib import Path
from typing import Any

from retraining.trigger_engine import RetrainingTriggerEngine, TriggerEvent


class DriftIntegration:
    """
    Integrates drift detection results with retraining triggers

    Reads from Phase 5 drift detection outputs and triggers
    retraining when thresholds are exceeded.
    """

    def __init__(
        self,
        drift_results_path: str = "results/phase5/drift_detection_psi.json",
        trigger_engine: RetrainingTriggerEngine | None = None,
    ):
        self.drift_results_path = Path(drift_results_path)
        self.trigger_engine = trigger_engine or RetrainingTriggerEngine()
        self.model_name = "intent-classifier-sgd"
        self.model_version = "v1.0.2"

    def load_drift_results(self) -> dict[str, Any] | None:
        """Load drift detection results from Phase 5"""
        if not self.drift_results_path.exists():
            return None

        with open(self.drift_results_path) as f:
            return json.load(f)

    def check_and_trigger(self) -> tuple[bool, TriggerEvent | None]:
        """
        Check drift results and trigger retraining if needed

        Returns:
            Tuple of (triggered: bool, trigger_event: Optional[TriggerEvent])
        """
        results = self.load_drift_results()

        if results is None:
            return False, None

        triggered = False
        trigger_event = None

        # Check PSI score
        psi_score = results.get("psi_score", 0.0)
        psi_trigger = self.trigger_engine.check_psi_drift(
            psi_score=psi_score, model_name=self.model_name, model_version=self.model_version
        )

        if psi_trigger:
            triggered = True
            trigger_event = psi_trigger

        # Check KS p-value
        ks_pvalue = results.get("ks_pvalue", 1.0)
        ks_trigger = self.trigger_engine.check_ks_drift(
            ks_pvalue=ks_pvalue, model_name=self.model_name, model_version=self.model_version
        )

        if ks_trigger:
            triggered = True
            trigger_event = ks_trigger  # Use the most recent trigger

        return triggered, trigger_event

    def get_drift_status(self) -> dict[str, Any]:
        """Get current drift status summary"""
        results = self.load_drift_results()

        if results is None:
            return {"status": "unknown", "message": "Drift results not found"}

        psi_score = results.get("psi_score", 0.0)
        ks_pvalue = results.get("ks_pvalue", 1.0)

        # Determine status
        if psi_score > 0.2 or ks_pvalue < 0.01:
            status = "critical"
        elif psi_score > 0.1 or ks_pvalue < 0.05:
            status = "warning"
        else:
            status = "normal"

        return {
            "status": status,
            "psi_score": psi_score,
            "ks_pvalue": ks_pvalue,
            "last_checked": results.get("timestamp", "unknown"),
            "model_name": self.model_name,
            "model_version": self.model_version,
        }


if __name__ == "__main__":
    integration = DriftIntegration()

    print("Drift Integration Check")
    print("=" * 50)

    status = integration.get_drift_status()
    print(f"Status: {status['status']}")
    print(f"PSI Score: {status.get('psi_score', 'N/A')}")
    print(f"KS P-Value: {status.get('ks_pvalue', 'N/A')}")

    triggered, trigger = integration.check_and_trigger()

    if triggered:
        print("\n[TRIGGER] Retraining triggered!")
        print(f"Message: {trigger.message}")
        print(f"Action: {trigger.recommended_action}")
    else:
        print("\n[OK] No retraining needed")
