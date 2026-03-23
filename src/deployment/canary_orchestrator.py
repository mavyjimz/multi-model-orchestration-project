"""
Canary Deployment Orchestrator
Manages automated A/B testing with controlled traffic splitting
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class DeploymentStatus(Enum):
    """Status of a canary deployment"""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


class StageStatus(Enum):
    """Status of a canary stage"""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class StageResult:
    """Results from a canary stage"""

    stage_name: str
    traffic_percentage: int
    status: str
    started_at: str
    completed_at: str | None
    requests_processed: int
    error_rate: float
    latency_p99_ms: float
    accuracy: float
    success: bool
    failure_reason: str | None = None


@dataclass
class CanaryDeployment:
    """Represents a canary deployment"""

    deployment_id: str
    baseline_model: str
    baseline_version: str
    candidate_model: str
    candidate_version: str
    status: str
    current_stage: int
    current_traffic_percentage: int
    started_at: str
    completed_at: str | None
    stage_results: list[dict[str, Any]] = field(default_factory=list)
    rollback_reason: str | None = None


class CanaryOrchestrator:
    """
    Orchestrates canary deployments with automated traffic management

    Features:
    - Gradual traffic increase (1% -> 5% -> 25% -> 50% -> 100%)
    - Automated rollback on performance degradation
    - Real-time metrics collection
    - Integration with model registry
    """

    def __init__(self, config_path: str = "src/deployment/canary_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()

        self.deployments_dir = Path("results/phase11/canary_deployments")
        self.metrics_dir = self.deployments_dir / "metrics"
        self.deployments_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.active_deployments: dict[str, CanaryDeployment] = {}
        self.completed_deployments: list[CanaryDeployment] = []

    def _load_config(self) -> dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")

        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def _generate_deployment_id(self, baseline_version: str, candidate_version: str) -> str:
        """Generate unique deployment ID"""
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        content = f"{baseline_version}:{candidate_version}:{timestamp}"
        hash_suffix = hashlib.sha256(content.encode()).hexdigest()[:8]
        return f"canary-{hash_suffix}"

    def start_deployment(
        self,
        baseline_model: str,
        baseline_version: str,
        candidate_model: str,
        candidate_version: str,
    ) -> CanaryDeployment:
        """
        Start a new canary deployment

        Args:
            baseline_model: Name of the baseline (production) model
            baseline_version: Version of the baseline model
            candidate_model: Name of the candidate model
            candidate_version: Version of the candidate model

        Returns:
            CanaryDeployment object
        """
        deployment_id = self._generate_deployment_id(baseline_version, candidate_version)

        deployment = CanaryDeployment(
            deployment_id=deployment_id,
            baseline_model=baseline_model,
            baseline_version=baseline_version,
            candidate_model=candidate_model,
            candidate_version=candidate_version,
            status=DeploymentStatus.PENDING.value,
            current_stage=0,
            current_traffic_percentage=0,
            started_at=datetime.utcnow().isoformat() + "Z",
            completed_at=None,
            stage_results=[],
        )

        self.active_deployments[deployment_id] = deployment
        self._save_deployment(deployment)

        return deployment

    def _save_deployment(self, deployment: CanaryDeployment):
        """Save deployment state to file"""
        filename = self.deployments_dir / f"{deployment.deployment_id}.json"
        with open(filename, "w") as f:
            json.dump(asdict(deployment), f, indent=2)

    def advance_stage(self, deployment_id: str, stage_result: StageResult) -> tuple[bool, str]:
        """
        Advance to the next canary stage

        Args:
            deployment_id: ID of the deployment
            stage_result: Results from the current stage

        Returns:
            Tuple of (success: bool, message: str)
        """
        if deployment_id not in self.active_deployments:
            return False, f"Deployment not found: {deployment_id}"

        deployment = self.active_deployments[deployment_id]
        stages = self.config["canary_stages"]

        # Record stage result
        deployment.stage_results.append(asdict(stage_result))

        # Check if stage failed
        if not stage_result.success:
            deployment.status = DeploymentStatus.ROLLED_BACK.value
            deployment.rollback_reason = stage_result.failure_reason
            deployment.completed_at = datetime.utcnow().isoformat() + "Z"
            self._save_deployment(deployment)
            return False, f"Stage failed: {stage_result.failure_reason}"

        # Move to next stage
        deployment.current_stage += 1

        if deployment.current_stage >= len(stages):
            # All stages completed - full rollout
            deployment.status = DeploymentStatus.COMPLETED.value
            deployment.current_traffic_percentage = 100
            deployment.completed_at = datetime.utcnow().isoformat() + "Z"
            self.completed_deployments.append(deployment)
            del self.active_deployments[deployment_id]
            self._save_deployment(deployment)
            return True, "Canary deployment completed - full rollout successful"

        # Update traffic percentage for next stage
        next_stage = stages[deployment.current_stage]
        deployment.current_traffic_percentage = next_stage["traffic_percentage"]
        deployment.status = DeploymentStatus.RUNNING.value

        self._save_deployment(deployment)

        return (
            True,
            f"Advanced to stage {deployment.current_stage}: {next_stage['name']} ({deployment.current_traffic_percentage}% traffic)",
        )

    def evaluate_stage(
        self, deployment_id: str, metrics: dict[str, Any]
    ) -> tuple[bool, str | None]:
        """
        Evaluate if current stage meets success criteria

        Args:
            deployment_id: ID of the deployment
            metrics: Current stage metrics (error_rate, latency_p99_ms, accuracy)

        Returns:
            Tuple of (success: bool, failure_reason: Optional[str])
        """
        if deployment_id not in self.active_deployments:
            return False, "Deployment not found"

        deployment = self.active_deployments[deployment_id]
        stages = self.config["canary_stages"]
        current_stage = stages[deployment.current_stage]
        criteria = current_stage["success_criteria"]

        failure_reasons = []

        # Check error rate
        error_rate = metrics.get("error_rate", 0.0)
        if error_rate > criteria["error_rate_max"]:
            failure_reasons.append(
                f"Error rate {error_rate:.4f} exceeds max {criteria['error_rate_max']}"
            )

        # Check latency
        latency_p99 = metrics.get("latency_p99_ms", 0.0)
        if latency_p99 > criteria["latency_p99_max_ms"]:
            failure_reasons.append(
                f"P99 latency {latency_p99:.2f}ms exceeds max {criteria['latency_p99_max_ms']}ms"
            )

        # Check accuracy (if provided)
        if "accuracy" in metrics:
            rollback_config = self.config["rollback_triggers"]
            accuracy_drop = 1.0 - metrics["accuracy"]
            if accuracy_drop > rollback_config["accuracy_drop_threshold"]:
                failure_reasons.append(
                    f"Accuracy drop {accuracy_drop:.4f} exceeds threshold {rollback_config['accuracy_drop_threshold']}"
                )

        if failure_reasons:
            return False, "; ".join(failure_reasons)

        return True, None

    def rollback(self, deployment_id: str, reason: str) -> bool:
        """
        Rollback a canary deployment

        Args:
            deployment_id: ID of the deployment
            reason: Reason for rollback

        Returns:
            True if rollback successful
        """
        if deployment_id not in self.active_deployments:
            return False

        deployment = self.active_deployments[deployment_id]
        deployment.status = DeploymentStatus.ROLLED_BACK.value
        deployment.rollback_reason = reason
        deployment.completed_at = datetime.utcnow().isoformat() + "Z"
        deployment.current_traffic_percentage = 0

        self._save_deployment(deployment)
        del self.active_deployments[deployment_id]

        return True

    def get_deployment_status(self, deployment_id: str) -> dict[str, Any] | None:
        """Get status of a specific deployment"""
        if deployment_id in self.active_deployments:
            return asdict(self.active_deployments[deployment_id])

        for deployment in self.completed_deployments:
            if deployment.deployment_id == deployment_id:
                return asdict(deployment)

        return None

    def get_all_deployments(self) -> list[dict[str, Any]]:
        """Get all deployments (active and completed)"""
        active = [asdict(d) for d in self.active_deployments.values()]
        completed = [asdict(d) for d in self.completed_deployments]
        return active + completed

    def get_traffic_split(self, deployment_id: str) -> dict[str, Any]:
        """Get current traffic split for a deployment"""
        if deployment_id not in self.active_deployments:
            return {"error": "Deployment not found or completed"}

        deployment = self.active_deployments[deployment_id]
        baseline_traffic = 100 - deployment.current_traffic_percentage
        candidate_traffic = deployment.current_traffic_percentage

        return {
            "deployment_id": deployment_id,
            "baseline": {
                "model": deployment.baseline_model,
                "version": deployment.baseline_version,
                "traffic_percentage": baseline_traffic,
            },
            "candidate": {
                "model": deployment.candidate_model,
                "version": deployment.candidate_version,
                "traffic_percentage": candidate_traffic,
            },
        }


if __name__ == "__main__":
    # Test the orchestrator
    orchestrator = CanaryOrchestrator()

    print("Canary Deployment Orchestrator Test")
    print("=" * 50)

    # Start a deployment
    deployment = orchestrator.start_deployment(
        baseline_model="intent-classifier-sgd",
        baseline_version="v1.0.2",
        candidate_model="intent-classifier-sgd",
        candidate_version="v1.0.3",
    )

    print(f"Deployment ID: {deployment.deployment_id}")
    print(f"Status: {deployment.status}")
    print(f"Traffic: {deployment.current_traffic_percentage}%")

    # Get traffic split
    split = orchestrator.get_traffic_split(deployment.deployment_id)
    print(f"\nTraffic Split: {json.dumps(split, indent=2)}")
