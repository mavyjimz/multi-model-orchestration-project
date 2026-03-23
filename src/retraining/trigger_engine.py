"""
Retraining Trigger Engine
Monitors model performance and data drift to automatically trigger retraining
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class TriggerSeverity(Enum):
    """Severity levels for retraining triggers"""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"


class TriggerType(Enum):
    """Types of retraining triggers"""
    PSI_DRIFT = "psi_drift"
    KS_DRIFT = "ks_drift"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    FEEDBACK_RATING = "feedback_rating"
    DATA_AGE = "data_age"
    MANUAL = "manual"


@dataclass
class TriggerEvent:
    """Represents a retraining trigger event"""
    trigger_id: str
    timestamp: str
    trigger_type: str
    severity: str
    model_name: str
    model_version: str
    metric_name: str
    current_value: float
    threshold_value: float
    baseline_value: float | None
    message: str
    recommended_action: str
    metadata: dict[str, Any]


class RetrainingTriggerEngine:
    """
    Engine for monitoring and triggering model retraining

    Integrates with:
    - Phase 5: Drift Detection (PSI/KS)
    - Phase 6: Model Registry
    - Phase 9: Monitoring & Observability
    - Phase 11.1: Feedback Collection (if enabled)
    """

    def __init__(self, config_path: str = "src/retraining/retraining_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()

        self.results_dir = Path("results/phase11/retraining_triggers")
        self.history_dir = self.results_dir / "trigger_history"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)

        self.trigger_history: list[TriggerEvent] = []

    def _load_config(self) -> dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")

        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def _generate_trigger_id(self, trigger_type: str, model_name: str) -> str:
        """Generate unique trigger ID"""
        timestamp = datetime.utcnow().isoformat()
        content = f"{trigger_type}:{model_name}:{timestamp}"
        hash_suffix = hashlib.sha256(content.encode()).hexdigest()[:8]
        return f"trigger-{trigger_type}-{hash_suffix}"

    def check_psi_drift(
        self,
        psi_score: float,
        model_name: str,
        model_version: str
    ) -> TriggerEvent | None:
        """
        Check PSI drift and trigger if threshold exceeded

        Args:
            psi_score: Current PSI score from drift detection
            model_name: Name of the model
            model_version: Version of the model

        Returns:
            TriggerEvent if threshold exceeded, None otherwise
        """
        config = self.config["trigger_conditions"]["psi"]

        if not config["enabled"]:
            return None

        severity = TriggerSeverity.NORMAL

        if psi_score >= config["critical_threshold"]:
            severity = TriggerSeverity.CRITICAL
        elif psi_score >= config["warning_threshold"]:
            severity = TriggerSeverity.WARNING
        else:
            return None  # No trigger needed

        trigger = TriggerEvent(
            trigger_id=self._generate_trigger_id("psi", model_name),
            timestamp=datetime.utcnow().isoformat() + "Z",
            trigger_type=TriggerType.PSI_DRIFT.value,
            severity=severity.value,
            model_name=model_name,
            model_version=model_version,
            metric_name="psi_score",
            current_value=psi_score,
            threshold_value=config["critical_threshold"] if severity == TriggerSeverity.CRITICAL else config["warning_threshold"],
            baseline_value=0.0,
            message=f"PSI drift detected: {psi_score:.4f} (threshold: {config['critical_threshold']})",
            recommended_action="Initiate model retraining with recent data" if severity == TriggerSeverity.CRITICAL else "Monitor closely, prepare retraining pipeline",
            metadata={
                "psi_score": psi_score,
                "warning_threshold": config["warning_threshold"],
                "critical_threshold": config["critical_threshold"]
            }
        )

        self._record_trigger(trigger)
        return trigger

    def check_ks_drift(
        self,
        ks_pvalue: float,
        model_name: str,
        model_version: str
    ) -> TriggerEvent | None:
        """
        Check KS test drift and trigger if threshold exceeded

        Args:
            ks_pvalue: P-value from KS test (lower = more drift)
            model_name: Name of the model
            model_version: Version of the model

        Returns:
            TriggerEvent if threshold exceeded, None otherwise
        """
        config = self.config["trigger_conditions"]["ks_test"]

        if not config["enabled"]:
            return None

        severity = TriggerSeverity.NORMAL

        if ks_pvalue <= config["critical_threshold"]:
            severity = TriggerSeverity.CRITICAL
        elif ks_pvalue <= config["warning_threshold"]:
            severity = TriggerSeverity.WARNING
        else:
            return None  # No trigger needed

        trigger = TriggerEvent(
            trigger_id=self._generate_trigger_id("ks", model_name),
            timestamp=datetime.utcnow().isoformat() + "Z",
            trigger_type=TriggerType.KS_DRIFT.value,
            severity=severity.value,
            model_name=model_name,
            model_version=model_version,
            metric_name="ks_pvalue",
            current_value=ks_pvalue,
            threshold_value=config["critical_threshold"] if severity == TriggerSeverity.CRITICAL else config["warning_threshold"],
            baseline_value=1.0,
            message=f"KS drift detected: p-value={ks_pvalue:.4f} (threshold: {config['critical_threshold']})",
            recommended_action="Initiate model retraining with recent data" if severity == TriggerSeverity.CRITICAL else "Monitor closely, prepare retraining pipeline",
            metadata={
                "ks_pvalue": ks_pvalue,
                "warning_threshold": config["warning_threshold"],
                "critical_threshold": config["critical_threshold"]
            }
        )

        self._record_trigger(trigger)
        return trigger

    def check_performance_degradation(
        self,
        current_accuracy: float,
        baseline_accuracy: float,
        model_name: str,
        model_version: str
    ) -> TriggerEvent | None:
        """
        Check performance degradation and trigger if threshold exceeded

        Args:
            current_accuracy: Current model accuracy
            baseline_accuracy: Baseline accuracy (rolling window average)
            model_name: Name of the model
            model_version: Version of the model

        Returns:
            TriggerEvent if degradation exceeds threshold, None otherwise
        """
        config = self.config["trigger_conditions"]["performance"]

        if not config["enabled"]:
            return None

        degradation = baseline_accuracy - current_accuracy

        if degradation <= config["degradation_threshold"]:
            return None  # No significant degradation

        severity = TriggerSeverity.CRITICAL if degradation > config["degradation_threshold"] * 2 else TriggerSeverity.WARNING

        trigger = TriggerEvent(
            trigger_id=self._generate_trigger_id("perf", model_name),
            timestamp=datetime.utcnow().isoformat() + "Z",
            trigger_type=TriggerType.PERFORMANCE_DEGRADATION.value,
            severity=severity.value,
            model_name=model_name,
            model_version=model_version,
            metric_name=config["metric"],
            current_value=current_accuracy,
            threshold_value=baseline_accuracy - config["degradation_threshold"],
            baseline_value=baseline_accuracy,
            message=f"Performance degradation: {current_accuracy:.4f} vs baseline {baseline_accuracy:.4f} (drop: {degradation:.4f})",
            recommended_action="Initiate model retraining immediately",
            metadata={
                "current_accuracy": current_accuracy,
                "baseline_accuracy": baseline_accuracy,
                "degradation": degradation,
                "degradation_threshold": config["degradation_threshold"]
            }
        )

        self._record_trigger(trigger)
        return trigger

    def check_feedback_rating(
        self,
        avg_rating: float,
        low_rating_percentage: float,
        feedback_count: int,
        model_name: str,
        model_version: str
    ) -> TriggerEvent | None:
        """
        Check feedback ratings and trigger if thresholds exceeded

        Args:
            avg_rating: Average user rating (1-5)
            low_rating_percentage: Percentage of 1-2 star ratings
            feedback_count: Total number of feedback records
            model_name: Name of the model
            model_version: Version of the model

        Returns:
            TriggerEvent if thresholds exceeded, None otherwise
        """
        config = self.config["trigger_conditions"]["feedback"]

        if not config["enabled"]:
            return None

        if feedback_count < config["min_feedback_count"]:
            return None  # Insufficient data

        severity = TriggerSeverity.NORMAL
        trigger_reason = []

        if avg_rating < config["avg_rating_threshold"]:
            severity = TriggerSeverity.CRITICAL
            trigger_reason.append(f"Low average rating: {avg_rating:.2f}")

        if low_rating_percentage > config["low_rating_percentage"]:
            if severity == TriggerSeverity.NORMAL:
                severity = TriggerSeverity.WARNING
            trigger_reason.append(f"High low-rating percentage: {low_rating_percentage:.2%}")

        if not trigger_reason:
            return None  # No trigger needed

        trigger = TriggerEvent(
            trigger_id=self._generate_trigger_id("feedback", model_name),
            timestamp=datetime.utcnow().isoformat() + "Z",
            trigger_type=TriggerType.FEEDBACK_RATING.value,
            severity=severity.value,
            model_name=model_name,
            model_version=model_version,
            metric_name="avg_rating",
            current_value=avg_rating,
            threshold_value=config["avg_rating_threshold"],
            baseline_value=5.0,
            message="; ".join(trigger_reason),
            recommended_action="Review feedback patterns and consider retraining",
            metadata={
                "avg_rating": avg_rating,
                "low_rating_percentage": low_rating_percentage,
                "feedback_count": feedback_count,
                "thresholds": config
            }
        )

        self._record_trigger(trigger)
        return trigger

    def _record_trigger(self, trigger: TriggerEvent):
        """Record trigger event to history and file"""
        self.trigger_history.append(trigger)

        # Save to file
        filename = self.history_dir / f"{trigger.trigger_id}.json"
        with open(filename, 'w') as f:
            json.dump(asdict(trigger), f, indent=2)

    def get_trigger_history(
        self,
        model_name: str | None = None,
        severity: str | None = None,
        limit: int = 100
    ) -> list[dict[str, Any]]:
        """Retrieve trigger history with optional filtering"""
        triggers = self.trigger_history

        if model_name:
            triggers = [t for t in triggers if t.model_name == model_name]

        if severity:
            triggers = [t for t in triggers if t.severity == severity]

        # Sort by timestamp descending
        triggers = sorted(triggers, key=lambda t: t.timestamp, reverse=True)

        return [asdict(t) for t in triggers[:limit]]

    def get_pending_triggers(self) -> list[dict[str, Any]]:
        """Get all critical triggers that haven't been actioned"""
        critical = [t for t in self.trigger_history if t.severity == TriggerSeverity.CRITICAL.value]
        return [asdict(t) for t in critical]

    def export_summary(self) -> dict[str, Any]:
        """Export summary of all triggers"""
        return {
            "total_triggers": len(self.trigger_history),
            "by_severity": {
                "critical": len([t for t in self.trigger_history if t.severity == "critical"]),
                "warning": len([t for t in self.trigger_history if t.severity == "warning"]),
                "normal": len([t for t in self.trigger_history if t.severity == "normal"])
            },
            "by_type": {
                t_type: len([t for t in self.trigger_history if t.trigger_type == t_type])
                for t_type in {t.trigger_type for t in self.trigger_history}
            },
            "recent_triggers": self.get_trigger_history(limit=10)
        }


if __name__ == "__main__":
    # Test the trigger engine
    engine = RetrainingTriggerEngine()

    print("Testing Retraining Trigger Engine")
    print("=" * 50)

    # Simulate PSI drift
    trigger = engine.check_psi_drift(
        psi_score=0.25,
        model_name="intent-classifier-sgd",
        model_version="v1.0.2"
    )

    if trigger:
        print(f"PSI Trigger: {trigger.message}")
        print(f"Severity: {trigger.severity}")
        print(f"Action: {trigger.recommended_action}")

    # Print summary
    print("\n" + "=" * 50)
    print("Summary:")
    print(json.dumps(engine.export_summary(), indent=2))
