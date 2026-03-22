# Phase 10.3: Audit Logging with Tamper Detection

## Overview
Implements cryptographically-linked audit logs to detect tampering and ensure compliance.

## Features
- SHA256 HMAC chain linking (each entry hashes previous entry)
- Automatic logging of API requests/responses via middleware
- Query interface for filtered audit trail retrieval
- Integrity verification CLI tool

## Usage

### Manual Logging
```python
from src.core.audit_logger import AuditLogger

audit = AuditLogger()
audit.log(
    event="model_promotion",
    user="admin",
    action="promote",
    resource="intent-classifier-sgd/v1.0.2",
    details={"from_stage": "staging", "to_stage": "production"},
    ip_address="192.168.1.100"
)
