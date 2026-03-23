# Phase 11.3: Automated A/B Test Deployment (Canary Releases)

## Overview
Automated canary deployment system for safely rolling out new model versions with controlled traffic splitting and automatic rollback on performance degradation.

## Architecture

### Components
1. **Canary Orchestrator** (`src/deployment/canary_orchestrator.py`)
   - Manages deployment lifecycle through 5 stages (1% → 5% → 25% → 50% → 100%)
   - Evaluates stage success criteria
   - Handles automatic rollback

2. **Traffic Router** (`src/deployment/traffic_router.py`)
   - Routes requests between baseline and candidate models
   - Supports session-based sticky routing
   - Logs all routing decisions

3. **A/B Metrics Collector** (`src/deployment/ab_metrics_collector.py`)
   - Collects performance metrics for both models
   - Compares accuracy, latency, error rates
   - Generates winner recommendations

4. **CLI Tool** (`scripts/phase11/p11.3-deployment-cli.py`)
   - Command-line interface for deployment management

### Deployment Stages

| Stage | Traffic % | Duration | Min Requests | Success Criteria |
|-------|-----------|----------|--------------|------------------|
| Initial | 1% | 30 min | 100 | Error < 1%, P99 < 200ms |
| Early | 5% | 60 min | 500 | Error < 2%, P99 < 150ms |
| Mid | 25% | 120 min | 2000 | Error < 2%, P99 < 100ms |
| Late | 50% | 240 min | 5000 | Error < 2%, P99 < 100ms |
| Full | 100% | - | - | Deployment complete |

## Usage

### Start Deployment
```bash
cd ~/MLOps/multi-model-orchestration-project
python scripts/phase11/p11.3-deployment-cli.py start-deployment \
  --baseline-model intent-classifier-sgd \
  --baseline-version v1.0.2 \
  --candidate-model intent-classifier-sgd \
  --candidate-version v1.0.3
