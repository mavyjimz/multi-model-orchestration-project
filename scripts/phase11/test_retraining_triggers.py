#!/usr/bin/env python3
"""
Retraining Triggers Integration Tests
"""

import json
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

from retraining.trigger_engine import (  # noqa: E402
    RetrainingTriggerEngine,
    TriggerSeverity,
    TriggerType,
)


def test_psi_trigger_critical():
    """Test PSI trigger at critical level"""
    print("Test 1: PSI Critical Trigger")

    engine = RetrainingTriggerEngine()
    trigger = engine.check_psi_drift(
        psi_score=0.25,
        model_name="test-model",
        model_version="v1.0.0"
    )

    assert trigger is not None, "Should trigger at PSI > 0.2"
    assert trigger.severity == TriggerSeverity.CRITICAL.value
    assert trigger.trigger_type == TriggerType.PSI_DRIFT.value

    print("  ✓ Passed")


def test_psi_trigger_warning():
    """Test PSI trigger at warning level"""
    print("\nTest 2: PSI Warning Trigger")

    engine = RetrainingTriggerEngine()
    trigger = engine.check_psi_drift(
        psi_score=0.15,
        model_name="test-model",
        model_version="v1.0.0"
    )

    assert trigger is not None, "Should trigger at PSI > 0.1"
    assert trigger.severity == TriggerSeverity.WARNING.value

    print("  ✓ Passed")


def test_psi_no_trigger():
    """Test no PSI trigger below threshold"""
    print("\nTest 3: PSI No Trigger")

    engine = RetrainingTriggerEngine()
    trigger = engine.check_psi_drift(
        psi_score=0.05,
        model_name="test-model",
        model_version="v1.0.0"
    )

    assert trigger is None, "Should not trigger at PSI < 0.1"

    print("  ✓ Passed")


def test_ks_trigger():
    """Test KS drift trigger"""
    print("\nTest 4: KS Drift Trigger")

    engine = RetrainingTriggerEngine()
    trigger = engine.check_ks_drift(
        ks_pvalue=0.005,
        model_name="test-model",
        model_version="v1.0.0"
    )

    assert trigger is not None, "Should trigger at KS p-value < 0.01"
    assert trigger.severity == TriggerSeverity.CRITICAL.value

    print("  ✓ Passed")


def test_performance_degradation():
    """Test performance degradation trigger"""
    print("\nTest 5: Performance Degradation Trigger")

    engine = RetrainingTriggerEngine()
    trigger = engine.check_performance_degradation(
        current_accuracy=0.65,
        baseline_accuracy=0.72,
        model_name="test-model",
        model_version="v1.0.0"
    )

    assert trigger is not None, "Should trigger at 7% degradation"
    assert trigger.trigger_type == TriggerType.PERFORMANCE_DEGRADATION.value

    print("  ✓ Passed")


def test_trigger_history():
    """Test trigger history retrieval"""
    print("\nTest 6: Trigger History")

    engine = RetrainingTriggerEngine()
    history = engine.get_trigger_history(limit=10)

    assert isinstance(history, list)
    assert len(history) > 0, "Should have recorded triggers"

    print(f"  Total triggers in history: {len(history)}")
    print("  ✓ Passed")


def test_export_summary():
    """Test summary export"""
    print("\nTest 7: Export Summary")

    engine = RetrainingTriggerEngine()
    summary = engine.export_summary()

    assert "total_triggers" in summary
    assert "by_severity" in summary
    assert "by_type" in summary

    print(f"  Summary: {json.dumps(summary, indent=2)}")
    print("  ✓ Passed")


def main():

    print("=" * 60)
    print("Retraining Triggers Integration Tests")
    print("=" * 60)

    try:
        test_psi_trigger_critical()
        test_psi_trigger_warning()
        test_psi_no_trigger()
        test_ks_trigger()
        test_performance_degradation()
        test_trigger_history()
        test_export_summary()

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
