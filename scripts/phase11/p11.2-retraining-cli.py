#!/usr/bin/env python3
"""
Retraining CLI Tool
Command-line interface for managing retraining triggers and jobs
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

from retraining.drift_integration import DriftIntegration  # noqa: E402
from retraining.retraining_pipeline import RetrainingPipeline  # noqa: E402
from retraining.trigger_engine import RetrainingTriggerEngine  # noqa: E402


def cmd_check_drift(args):
    """Check drift and display status"""
    integration = DriftIntegration()
    status = integration.get_drift_status()

    print(json.dumps(status, indent=2))

    if status["status"] == "critical":
        return 1
    return 0


def cmd_trigger_status(args):
    """Display trigger history and status"""
    engine = RetrainingTriggerEngine()

    if args.model:
        history = engine.get_trigger_history(model_name=args.model, limit=args.limit)
    else:
        history = engine.get_trigger_history(limit=args.limit)

    print(f"Trigger History (last {args.limit}):")
    print(json.dumps(history, indent=2))


def cmd_pending_triggers(args):
    """Display pending critical triggers"""
    engine = RetrainingTriggerEngine()
    pending = engine.get_pending_triggers()

    if not pending:
        print("No pending critical triggers")
        return 0

    print(f"Pending Critical Triggers: {len(pending)}")
    print(json.dumps(pending, indent=2))
    return 0


def cmd_start_retrain(args):
    """Manually trigger retraining"""
    from retraining.trigger_engine import TriggerEvent, TriggerSeverity, TriggerType

    engine = RetrainingTriggerEngine()
    pipeline = RetrainingPipeline()

    # Create manual trigger
    trigger = TriggerEvent(
        trigger_id=engine._generate_trigger_id("manual", args.model),
        timestamp="manual",
        trigger_type=TriggerType.MANUAL.value,
        severity=TriggerSeverity.CRITICAL.value,
        model_name=args.model,
        model_version=args.version,
        metric_name="manual",
        current_value=0.0,
        threshold_value=0.0,
        baseline_value=None,
        message="Manual retraining trigger",
        recommended_action="Execute retraining pipeline",
        metadata={"triggered_by": "cli"}
    )

    # Start retraining job
    job = pipeline.start_retraining(trigger, args.new_version)

    print(f"Retraining job started: {job.job_id}")
    print(json.dumps({
        "job_id": job.job_id,
        "status": job.status,
        "model": job.model_name,
        "previous_version": job.previous_version,
        "new_version": job.new_version
    }, indent=2))

    return 0


def cmd_job_status(args):
    """Check status of a retraining job"""
    pipeline = RetrainingPipeline()

    if args.job_id:
        status = pipeline.get_job_status(args.job_id)
        if status:
            print(json.dumps(status, indent=2))
        else:
            print(f"Job not found: {args.job_id}")
            return 1
    else:
        jobs = pipeline.get_all_jobs()
        print(f"Total jobs: {len(jobs)}")
        print(json.dumps(jobs, indent=2))

    return 0


def main():
    parser = argparse.ArgumentParser(description="Retraining CLI Tool")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # check-drift command
    p_check = subparsers.add_parser("check-drift", help="Check drift status")
    p_check.set_defaults(func=cmd_check_drift)

    # trigger-status command
    p_trigger = subparsers.add_parser("trigger-status", help="Show trigger history")
    p_trigger.add_argument("--model", type=str, help="Filter by model name")
    p_trigger.add_argument("--limit", type=int, default=10, help="Number of records")
    p_trigger.set_defaults(func=cmd_trigger_status)

    # pending-triggers command
    p_pending = subparsers.add_parser("pending-triggers", help="Show pending critical triggers")
    p_pending.set_defaults(func=cmd_pending_triggers)

    # start-retrain command
    p_retrain = subparsers.add_parser("start-retrain", help="Manually trigger retraining")
    p_retrain.add_argument("--model", required=True, help="Model name")
    p_retrain.add_argument("--version", required=True, help="Current version")
    p_retrain.add_argument("--new-version", required=True, help="New version")
    p_retrain.set_defaults(func=cmd_start_retrain)

    # job-status command
    p_job = subparsers.add_parser("job-status", help="Check job status")
    p_job.add_argument("--job-id", type=str, help="Specific job ID")
    p_job.set_defaults(func=cmd_job_status)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
