# Security Runbook

## Secrets Management

### GitHub Actions Secrets
Store sensitive values in GitHub Secrets:
1. Repository Settings > Secrets and variables > Actions
2. Add required secrets from `config/secrets.template`
3. Never commit actual secret values

### Required Secrets
| Secret | Description | Rotation |
|--------|-------------|----------|
| MLFLOW_TRACKING_URI | MLflow server URL | Infra change |
| API_SECRET_KEY | API encryption key | 180 days |
| DATABASE_URL | Database connection | Infra change |

### Local Development
```bash
cp config/secrets.template config/secrets.env
# Edit with actual values (never commit!)

### Step 5: Create .env.example and update .gitignore
```bash
cat > .env.example << 'EOF'
# Local Development Environment
MLFLOW_TRACKING_URI=http://localhost:5000
API_SECRET_KEY=your-dev-secret-key
DATABASE_URL=sqlite:///./registry.db
LOG_LEVEL=DEBUG
