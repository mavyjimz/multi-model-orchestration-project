# Required GitHub Repository Secrets for Phase 10

Add these via: GitHub Repo -> Settings -> Secrets and variables -> Actions

| Secret Name | Purpose | Example Value |
|-------------|---------|---------------|
| JWT_SECRET_KEY | JWT token signing | $(openssl rand -hex 32) |
| AUDIT_TAMPER_HASH_SECRET | Audit log integrity | $(openssl rand -hex 32) |
| DATABASE_URL | Production database connection | postgresql://user:pass@host/db |
| MLFLOW_TRACKING_URI | Remote MLflow server | https://mlflow.example.com |

# Local Development
For local testing, copy `.env.example` to `.env` and fill with test values.
NEVER commit `.env` to version control.

# CI/CD Usage
In .github/workflows/*.yml, reference secrets as:
  env:
    JWT_SECRET_KEY: ${{ secrets.JWT_SECRET_KEY }}
