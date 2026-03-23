# Phase 11.2: Model Retraining Triggers

## Overview
Automated system for detecting model performance degradation and data drift, triggering retraining when thresholds are exceeded.

## Architecture

### Components
1. **Trigger Engine** (`src/retraining/trigger_engine.py`)
   - Core logic for evaluating trigger conditions
   - Supports PSI, KS, performance, and feedback-based triggers
   - Records trigger history with severity levels

2. **Drift Integration** (`src/retraining/drift_integration.py`)
   - Connects with Phase 5 drift detection outputs
   - Automatic trigger evaluation from drift results

3. **Retraining Pipeline** (`src/retraining/retraining_pipeline.py`)
   - Orchestrates the retraining workflow
   - Validates new models before deployment

4. **CLI Tool** (`scripts/phase11/p11.2-retraining-cli.py`)
   - Command-line interface for manual operations

### Trigger Types

| Type | Metric | Warning Threshold | Critical Threshold |
|------|--------|-------------------|-------------------|
| PSI Drift | PSI Score | 0.10 | 0.20 |
| KS Drift | KS P-Value | 0.05 | 0.01 |
| Performance | Accuracy Drop | 5% | 10% |
| Feedback | Avg Rating | 3.5 | 3.0 |

## Configuration

Edit `src/retraining/retraining_config.yaml` to customize thresholds:

```yaml
trigger_conditions:
  psi:
    enabled: true
    warning_threshold: 0.1
    critical_threshold: 0.2
