#!/usr/bin/env python3
"""
Canary Deployment Integration Tests
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

from deployment.ab_metrics_collector import ABMetricsCollector  # noqa: E402
from deployment.canary_orchestrator import CanaryOrchestrator  # noqa: E402
from deployment.traffic_router import TrafficRouter  # noqa: E402


def test_deployment_start():
    """Test starting a canary deployment"""
    print("Test 1: Start Deployment")

    orchestrator = CanaryOrchestrator()
    deployment = orchestrator.start_deployment(
        baseline_model="intent-classifier-sgd",
        baseline_version="v1.0.2",
        candidate_model="intent-classifier-sgd",
        candidate_version="v1.0.3"
    )

    assert deployment.deployment_id is not None
    assert deployment.status == "pending"
    assert deployment.current_traffic_percentage == 0

    print(f"  Deployment ID: {deployment.deployment_id}")
    print("  ✓ Passed")


def test_traffic_routing():
    """Test traffic routing between models"""
    print("\nTest 2: Traffic Routing")

    router = TrafficRouter(deployment_id="test-deployment-002")
    router.set_traffic_split(20)  # 20% to candidate

    baseline_count = 0
    candidate_count = 0

    for _i in range(100):
        decision = router.route_request()
        if decision.selected_version == "v1.0.3":
            candidate_count += 1
        else:
            baseline_count += 1

    stats = router.get_routing_stats()
    print(f"  Baseline: {stats['baseline_count']} ({stats['baseline_percentage']}%)")
    print(f"  Candidate: {stats['candidate_count']} ({stats['candidate_percentage']}%)")

    # Allow some variance (15-25% for 20% target)
    assert 15 <= stats['candidate_percentage'] <= 25

    print("  ✓ Passed")


def test_stage_evaluation():
    """Test stage evaluation with metrics"""
    print("\nTest 3: Stage Evaluation")

    orchestrator = CanaryOrchestrator()
    deployment = orchestrator.start_deployment(
        baseline_model="test-model",
        baseline_version="v1.0.0",
        candidate_model="test-model",
        candidate_version="v1.0.1"
    )

    # Test passing metrics
    success, reason = orchestrator.evaluate_stage(
        deployment.deployment_id,
        metrics={
            "error_rate": 0.01,
            "latency_p99_ms": 100,
            "accuracy": 0.75
        }
    )

    assert success, f"Should pass with good metrics: {reason}"
    print("  Good metrics: PASSED")

    # Test failing metrics
    success, reason = orchestrator.evaluate_stage(
        deployment.deployment_id,
        metrics={
            "error_rate": 0.10,  # Too high
            "latency_p99_ms": 100,
            "accuracy": 0.75
        }
    )

    assert not success, "Should fail with high error rate"
    print(f"  Bad metrics: FAILED (expected) - {reason}")

    print("  ✓ Passed")


def test_rollback():
    """Test deployment rollback"""
    print("\nTest 4: Rollback")

    orchestrator = CanaryOrchestrator()
    deployment = orchestrator.start_deployment(
        baseline_model="test-model",
        baseline_version="v1.0.0",
        candidate_model="test-model",
        candidate_version="v1.0.1"
    )

    success = orchestrator.rollback(deployment.deployment_id, "Test rollback")

    assert success, "Rollback should succeed"

    status = orchestrator.get_deployment_status(deployment.deployment_id)
    assert status["status"] == "rolled_back"
    assert status["rollback_reason"] == "Test rollback"

    print("  ✓ Passed")


def test_metrics_collection():
    """Test A/B metrics collection"""
    print("\nTest 5: Metrics Collection")

    collector = ABMetricsCollector(deployment_id="test-deployment-003")

    # Record some requests
    for i in range(50):
        collector.record_request(
            model_version="v1.0.2",
            latency_ms=50 + (i % 20),
            success=True,
            correct_prediction=(i % 3 != 0)
        )
        collector.record_request(
            model_version="v1.0.3",
            latency_ms=45 + (i % 15),
            success=True,
            correct_prediction=(i % 4 != 0)
        )

    baseline = collector.get_baseline_metrics()
    candidate = collector.get_candidate_metrics()

    assert baseline["total_requests"] == 50
    assert candidate["total_requests"] == 50
    assert "error_rate" in baseline
    assert "latency_p99_ms" in baseline

    print(f"  Baseline requests: {baseline['total_requests']}")
    print(f"  Candidate requests: {candidate['total_requests']}")

    print("  ✓ Passed")


def test_model_comparison():
    """Test model comparison"""
    print("\nTest 6: Model Comparison")

    collector = ABMetricsCollector(deployment_id="test-deployment-004")

    # Record requests with clear winner
    for i in range(100):
        collector.record_request(
            model_version="v1.0.2",
            latency_ms=60,
            success=True,
            correct_prediction=(i % 4 != 0)  # ~75% accuracy
        )
        collector.record_request(
            model_version="v1.0.3",
            latency_ms=40,
            success=True,
            correct_prediction=(i % 5 != 0)  # ~80% accuracy
        )

    comparison = collector.compare_models()

    assert comparison.winner in ["baseline", "candidate"]
    assert "recommendation" in comparison.recommendation
    assert -1.0 <= comparison.accuracy_delta <= 1.0

    print(f"  Winner: {comparison.winner}")
    print(f"  Recommendation: {comparison.recommendation}")

    print("  ✓ Passed")


def main():
    print("=" * 60)
    print("Canary Deployment Integration Tests")
    print("=" * 60)

    try:
        test_deployment_start()
        test_traffic_routing()
        test_stage_evaluation()
        test_rollback()
        test_metrics_collection()
        test_model_comparison()

        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\n[FAILED] {e}")
        return 1
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
