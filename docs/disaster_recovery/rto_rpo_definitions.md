# Recovery Time Objective (RTO) & Recovery Point Objective (RPO)

## Definitions
- **RTO (Recovery Time Objective)**: Maximum acceptable downtime. Target: 4 hours.
- **RPO (Recovery Point Objective)**: Maximum acceptable data loss. Target: 1 hour.

## Critical Systems
1. Model Registry (MLflow): RTO=1h, RPO=15m
2. API Service (FastAPI): RTO=30m, RPO=N/A (Stateless)
3. Database (PostgreSQL/SQLite): RTO=2h, RPO=1h
4. Vector Index (FAISS): RTO=1h, RPO=1h (Rebuildable from data)

## Escalation Matrix
- Level 1: On-call Engineer (0-30 min)
- Level 2: MLOps Lead (30-60 min)
- Level 3: CTO/Stakeholders (60+ min)
