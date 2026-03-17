"""
Audit logging utility for Model Registry
Phase 6.8: Deprecation & Retirement Policy
Logs to: logs/audit/deprecation.log (JSON structured)
"""
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

# Ensure audit log directory exists (respects your SOP: ./logs/audit/)
AUDIT_LOG_DIR = Path("logs/audit")
AUDIT_LOG_DIR.mkdir(parents=True, exist_ok=True)

# Configure structured JSON logger
audit_logger = logging.getLogger("registry.audit")
audit_logger.setLevel(logging.INFO)

# Prevent duplicate handlers if module reloaded
if not audit_logger.handlers:
    log_file = AUDIT_LOG_DIR / "deprecation.log"
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    
    class JSONFormatter(logging.Formatter):
        """Custom formatter for structured JSON audit logs"""
        def format(self, record: logging.LogRecord) -> str:
            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "action": getattr(record, "action", None),
                "model_name": getattr(record, "model_name", None),
                "version": getattr(record, "version", None),
                "actor": getattr(record, "actor", "system"),
                "metadata": getattr(record, "metadata", {}),
                "ip_address": getattr(record, "ip_address", None),
                "status": getattr(record, "status", "success"),
                "message": record.getMessage(),
            }
            return json.dumps({k: v for k, v in log_entry.items() if v is not None})
    
    file_handler.setFormatter(JSONFormatter())
    audit_logger.addHandler(file_handler)


def log_deprecation(
    model_name: str,
    version: str,
    reason: str,
    actor: str = "system",
    migration_guide: Optional[str] = None,
    ip_address: Optional[str] = None,
    **extra_metadata
) -> None:
    """Log a deprecation event with structured fields"""
    audit_logger.info(
        "Model deprecated: %s v%s",
        model_name,
        version,
        extra={
            "action": "deprecate",
            "model_name": model_name,
            "version": version,
            "actor": actor,
            "metadata": {
                "reason": reason,
                "migration_guide": migration_guide,
                **extra_metadata
            },
            "ip_address": ip_address,
        }
    )


def log_retirement(
    model_name: str,
    version: str,
    soft_delete: bool,
    actor: str = "system",
    archive_location: Optional[str] = None,
    ip_address: Optional[str] = None,
    **extra_metadata
) -> None:
    """Log a retirement event with structured fields"""
    audit_logger.info(
        "Model retired: %s v%s (soft_delete=%s)",
        model_name,
        version,
        soft_delete,
        extra={
            "action": "retire",
            "model_name": model_name,
            "version": version,
            "actor": actor,
            "metadata": {
                "soft_delete": soft_delete,
                "archive_location": archive_location,
                **extra_metadata
            },
            "ip_address": ip_address,
        }
    )


def log_lifecycle_event(
    action: str,
    model_name: str,
    version: str,
    status: str = "success",
    actor: str = "system",
    metadata: Optional[Dict[str, Any]] = None,
    ip_address: Optional[str] = None,
    error_message: Optional[str] = None
) -> None:
    """Generic lifecycle event logger for extendability"""
    level = logging.INFO if status == "success" else logging.ERROR
    audit_logger.log(
        level,
        "Lifecycle event: %s %s v%s [%s]",
        action,
        model_name,
        version,
        status,
        extra={
            "action": action,
            "model_name": model_name,
            "version": version,
            "actor": actor,
            "metadata": metadata or {},
            "ip_address": ip_address,
            "status": status,
            "error_message": error_message,
        }
    )


def query_audit_log(
    model_name: Optional[str] = None,
    action: Optional[str] = None,
    version: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 100
) -> list[Dict[str, Any]]:
    """Query audit log entries from JSONL file"""
    results = []
    log_file = AUDIT_LOG_DIR / "deprecation.log"
    
    if not log_file.exists():
        return results
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if model_name and entry.get('model_name') != model_name:
                    continue
                if action and entry.get('action') != action:
                    continue
                if version and entry.get('version') != version:
                    continue
                if start_date and entry.get('timestamp', '') < start_date:
                    continue
                if end_date and entry.get('timestamp', '') > end_date:
                    continue
                results.append(entry)
                if len(results) >= limit:
                    break
            except json.JSONDecodeError:
                continue
    return results
