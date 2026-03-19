# CI/CD Strategy — Multi-Model Orchestration Project

## Philosophy
- Fail fast, fail early: Catch linting and testing issues before build/deploy stages
- Immutable artifacts: Docker images built once, promoted across environments
- Security by default: Secrets never in code; dependency scanning on every commit
- Observability first: Every pipeline run produces auditable artifacts

## Workflow Design Decisions

### Triggers
| Event | Branches | Purpose |
|-------|----------|---------|
| push | main, develop | Full pipeline: lint → test → build → deploy-staging |
| pull_request | main, develop | Lint + test only (no deploy) |
| tag | v* | Release build with semantic versioning |
| workflow_dispatch | any | Manual trigger for debugging/re-runs |

### Job Dependencies
lint → test → build → deploy-staging

### Environment Strategy
| Environment | Trigger | Image Tag | Secrets Scope |
|-------------|---------|-----------|---------------|
| staging | push to main | main-<sha> | staging/* |
| production | manual approval | v* | production/* |

## Local Development Parity
- Use docker-compose.yml for local testing of CI steps
- Pre-commit hooks mirror CI linting rules
- Test locally: pytest --cov=src before pushing

## Hardware Target
- Laptop deployment: 8GB RAM, NVIDIA MX150
- Lightweight Docker: multi-stage build, minimal runtime
- Resource limits: 2GB RAM max for container

## Next Steps (Phase 7)
- p7.2: Add integration test matrix with model fixtures
- p7.4: Optimize Dockerfile with multi-stage builds
- p7.6: Implement retraining trigger on data drift detection
- p7.8: Configure Dependabot + secret scanning alerts

## Runbook Reference
See docs/CI_CD_RUNBOOK.md for troubleshooting pipeline failures.
