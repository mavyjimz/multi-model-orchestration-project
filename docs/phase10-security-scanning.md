# Phase 10.7: Automated Security Scanning

## Overview
Implements multi-layered security scanning for code, dependencies, and containers.

## Scan Types

### 1. SAST (Static Application Security Testing)
- Tool: Bandit
- Target: `src/` Python code
- Command: `bash scripts/security/scan-code.sh`
- Output: `reports/security/bandit-report.json`

### 2. Dependency Scanning
- Tool: pip-audit
- Target: `requirements.txt` dependencies
- Command: `bash scripts/security/scan-deps.sh`
- Output: `reports/security/pip-audit-report.json`

### 3. Container Scanning
- Tool: Trivy
- Target: Docker image
- Command: `bash scripts/security/scan-container.sh <image_name>`
- Output: `reports/security/trivy-report.txt`

### 4. Full Scan Suite
- Command: `bash scripts/security/run-all-scans.sh`
- Output: Consolidated summary in `reports/security/security-summary-*.md`

## CI/CD Integration

Add to `.github/workflows/ci-cd.yml`:
```yaml
- name: Run security scans
  run: |
    bash scripts/security/scan-code.sh
    bash scripts/security/scan-deps.sh
    # Container scan requires built image
