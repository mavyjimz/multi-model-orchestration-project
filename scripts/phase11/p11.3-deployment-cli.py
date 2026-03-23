#!/usr/bin/env python3
"""
Canary Deployment CLI Tool
Command-line interface for managing A/B test deployments
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

from deployment.ab_metrics_collector import ABMetricsCollector  # noqa: E402
from deployment.canary_orchestrator import CanaryOrchestrator, StageResult  # noqa: E402


def cmd_start_deployment(args):
    """Start a new canary deployment"""
    orchestrator = CanaryOrchestrator()

    deployment = orchestrator.start_deployment(
        baseline_model=args.baseline_model,
        baseline_version=args.baseline_version,
        candidate_model=args.candidate_model,
        candidate_version=args.candidate_version
    )

    print("Canary deployment started")
    print(json.dumps({
        "deployment_id": deployment.deployment_id,
        "status": deployment.status,
        "baseline": f"{deployment.baseline_model}:{deployment.baseline_version}",
        "candidate": f"{deployment.candidate_model}:{deployment.candidate_version}",
        "initial_traffic": f"{deployment.current_traffic_percentage}%"
    }, indent=2))

    return 0


def cmd_status(args):
    """Get deployment status"""
    orchestrator = CanaryOrchestrator()

    if args.deployment_id:
        status = orchestrator.get_deployment_status(args.deployment_id)
        if status:
            print(json.dumps(status, indent=2))
        else:
            print(f"Deployment not found: {args.deployment_id}")
            return 1
    else:
        deployments = orchestrator.get_all_deployments()
        print(f"Total deployments: {len(deployments)}")
        print(json.dumps(deployments, indent=2))

    return 0


def cmd_traffic_split(args):
    """Get current traffic split"""
    orchestrator = CanaryOrchestrator()

    split = orchestrator.get_traffic_split(args.deployment_id)
    print(json.dumps(split, indent=2))

    return 0


def cmd_advance(args):
    """Manually advance to next stage"""
    orchestrator = CanaryOrchestrator()

    stage_result = StageResult(
        stage_name=args.stage_name,
        traffic_percentage=args.traffic_percentage,
        status="completed",
        started_at=args.started_at,
        completed_at=args.completed_at,
        requests_processed=args.requests,
        error_rate=args.error_rate,
        latency_p99_ms=args.latency_p99,
        accuracy=args.accuracy,
        success=True
    )

    success, message = orchestrator.advance_stage(args.deployment_id, stage_result)

    print(f"Stage advance: {'Success' if success else 'Failed'}")
    print(f"Message: {message}")

    return 0 if success else 1


def cmd_rollback(args):
    """Rollback a deployment"""
    orchestrator = CanaryOrchestrator()

    success = orchestrator.rollback(args.deployment_id, args.reason)

    if success:
        print(f"Deployment {args.deployment_id} rolled back")
        print(f"Reason: {args.reason}")
        return 0
    else:
        print(f"Failed to rollback deployment {args.deployment_id}")
        return 1


def cmd_metrics(args):
    """Get A/B test metrics"""
    collector = ABMetricsCollector(args.deployment_id)

    comparison = collector.compare_models()
    print(json.dumps({
        "deployment_id": comparison.deployment_id,
        "winner": comparison.winner,
        "confidence": comparison.confidence,
        "recommendation": comparison.recommendation,
        "accuracy_delta": comparison.accuracy_delta,
        "latency_delta_ms": comparison.latency_delta_ms,
        "error_rate_delta": comparison.error_rate_delta
    }, indent=2))

    return 0


def main():
    parser = argparse.ArgumentParser(description="Canary Deployment CLI Tool")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # start-deployment command
    p_start = subparsers.add_parser("start-deployment", help="Start new canary deployment")
    p_start.add_argument("--baseline-model", required=True, help="Baseline model name")
    p_start.add_argument("--baseline-version", required=True, help="Baseline version")
    p_start.add_argument("--candidate-model", required=True, help="Candidate model name")
    p_start.add_argument("--candidate-version", required=True, help="Candidate version")
    p_start.set_defaults(func=cmd_start_deployment)

    # status command
    p_status = subparsers.add_parser("status", help="Get deployment status")
    p_status.add_argument("--deployment-id", type=str, help="Specific deployment ID")
    p_status.set_defaults(func=cmd_status)

    # traffic-split command
    p_split = subparsers.add_parser("traffic-split", help="Get traffic split")
    p_split.add_argument("--deployment-id", required=True, help="Deployment ID")
    p_split.set_defaults(func=cmd_traffic_split)

    # advance command
    p_advance = subparsers.add_parser("advance", help="Advance to next stage")
    p_advance.add_argument("--deployment-id", required=True, help="Deployment ID")
    p_advance.add_argument("--stage-name", required=True, help="Stage name")
    p_advance.add_argument("--traffic-percentage", type=int, required=True, help="Traffic %")
    p_advance.add_argument("--started-at", required=True, help="Stage start time")
    p_advance.add_argument("--completed-at", required=True, help="Stage end time")
    p_advance.add_argument("--requests", type=int, required=True, help="Requests processed")
    p_advance.add_argument("--error-rate", type=float, required=True, help="Error rate")
    p_advance.add_argument("--latency-p99", type=float, required=True, help="P99 latency")
    p_advance.add_argument("--accuracy", type=float, required=True, help="Accuracy")
    p_advance.set_defaults(func=cmd_advance)

    # rollback command
    p_rollback = subparsers.add_parser("rollback", help="Rollback deployment")
    p_rollback.add_argument("--deployment-id", required=True, help="Deployment ID")
    p_rollback.add_argument("--reason", required=True, help="Rollback reason")
    p_rollback.set_defaults(func=cmd_rollback)

    # metrics command
    p_metrics = subparsers.add_parser("metrics", help="Get A/B test metrics")
    p_metrics.add_argument("--deployment-id", required=True, help="Deployment ID")
    p_metrics.set_defaults(func=cmd_metrics)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
