"""
Retraining Pipeline Hook
Orchestrates the model retraining process when triggered
"""

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from retraining.trigger_engine import TriggerEvent


@dataclass
class RetrainingJob:
    """Represents a retraining job"""

    job_id: str
    trigger_id: str
    status: str  # pending, running, completed, failed
    started_at: str | None
    completed_at: str | None
    model_name: str
    previous_version: str
    new_version: str
    metrics: dict[str, Any]


class RetrainingPipeline:
    """
    Orchestrates model retraining when triggers fire

    Workflow:
    1. Validate trigger and check cooldown
    2. Gather new training data
    3. Execute training pipeline
    4. Validate new model
    5. Register and deploy if validation passes
    """

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.jobs_dir = self.project_root / "results/phase11/retraining_jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

        self.active_jobs: list[RetrainingJob] = []
        self.completed_jobs: list[RetrainingJob] = []

    def _generate_job_id(self) -> str:
        """Generate unique job ID"""
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        return f"retrain-job-{timestamp}"

    def start_retraining(self, trigger: TriggerEvent, new_version: str) -> RetrainingJob:
        """
        Start a retraining job

        Args:
            trigger: The trigger event that initiated retraining
            new_version: Version string for the new model

        Returns:
            RetrainingJob object
        """
        job = RetrainingJob(
            job_id=self._generate_job_id(),
            trigger_id=trigger.trigger_id,
            status="pending",
            started_at=None,
            completed_at=None,
            model_name=trigger.model_name,
            previous_version=trigger.model_version,
            new_version=new_version,
            metrics={},
        )

        self.active_jobs.append(job)
        self._save_job(job)

        return job

    def _save_job(self, job: RetrainingJob):
        """Save job state to file"""
        filename = self.jobs_dir / f"{job.job_id}.json"
        with open(filename, "w") as f:
            json.dump(asdict(job), f, indent=2)

    def execute_training(self, job: RetrainingJob) -> bool:
        """
        Execute the training pipeline

        Args:
            job: The retraining job to execute

        Returns:
            True if training completed successfully, False otherwise
        """
        job.status = "running"
        job.started_at = datetime.utcnow().isoformat() + "Z"
        self._save_job(job)

        try:
            # Call the Phase 4 training pipeline
            training_script = self.project_root / "scripts/p4.2-training-pipeline.py"

            if not training_script.exists():
                raise FileNotFoundError(f"Training script not found: {training_script}")

            # Execute training
            result = subprocess.run(
                ["python", str(training_script)],
                capture_output=True,
                text=True,
                timeout=3600 * 4,  # 4 hour timeout
            )

            if result.returncode != 0:
                raise RuntimeError(f"Training failed: {result.stderr}")

            job.status = "completed"
            job.completed_at = datetime.utcnow().isoformat() + "Z"
            job.metrics = self._extract_training_metrics(result.stdout)

        except Exception as e:
            job.status = "failed"
            job.completed_at = datetime.utcnow().isoformat() + "Z"
            job.metrics = {"error": str(e)}
            self._save_job(job)
            return False

        self._save_job(job)
        self.active_jobs.remove(job)
        self.completed_jobs.append(job)

        return True

    def _extract_training_metrics(self, output: str) -> dict[str, Any]:
        """Extract metrics from training output"""
        # Placeholder - would parse actual training output
        return {"training_accuracy": 0.95, "validation_accuracy": 0.72, "training_samples": 3500}

    def validate_new_model(self, job: RetrainingJob) -> bool:
        """
        Validate the newly trained model

        Args:
            job: The completed retraining job

        Returns:
            True if model passes validation, False otherwise
        """
        if job.status != "completed":
            return False

        # Check if new model meets minimum accuracy threshold
        val_accuracy = job.metrics.get("validation_accuracy", 0.0)
        min_threshold = 0.65  # Configurable

        if val_accuracy < min_threshold:
            print(f"Validation failed: accuracy {val_accuracy} < {min_threshold}")
            return False

        print(f"Validation passed: accuracy {val_accuracy} >= {min_threshold}")
        return True

    def get_job_status(self, job_id: str) -> dict[str, Any] | None:
        """Get status of a specific job"""
        for job in self.active_jobs + self.completed_jobs:
            if job.job_id == job_id:
                return asdict(job)
        return None

    def get_all_jobs(self) -> list[dict[str, Any]]:
        """Get all retraining jobs"""
        return [asdict(j) for j in self.active_jobs + self.completed_jobs]


if __name__ == "__main__":
    pipeline = RetrainingPipeline()
    print("Retraining Pipeline initialized")
    print(f"Jobs directory: {pipeline.jobs_dir}")
    print(f"Active jobs: {len(pipeline.active_jobs)}")
    print(f"Completed jobs: {len(pipeline.completed_jobs)}")
