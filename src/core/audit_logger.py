"""Immutable audit logging with tamper detection for Phase 10."""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path


class AuditLogger:
    """Audit logger with cryptographic integrity verification."""

    def __init__(
        self,
        log_path: str = "logs/audit/audit.log",
        hash_secret: str | None = None,
    ):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.hash_secret = hash_secret or os.getenv(
            "AUDIT_TAMPER_HASH_SECRET", "dev-hash-secret-change-in-production"
        )
        self._last_hash: str | None = None

        # Load last hash from existing log if present
        if self.log_path.exists():
            self._last_hash = self._get_last_log_hash()

    def _compute_entry_hash(self, entry: dict, prev_hash: str | None) -> str:
        """Compute HMAC-SHA256 hash for audit entry with chain linkage."""
        entry_copy = entry.copy()
        entry_copy["prev_hash"] = prev_hash
        entry_copy["timestamp"] = entry_copy["timestamp"].isoformat()
        content = json.dumps(entry_copy, sort_keys=True) + self.hash_secret
        return hashlib.sha256(content.encode()).hexdigest()

    def _get_last_log_hash(self) -> str | None:
        """Extract hash from last line of audit log."""
        if not self.log_path.exists():
            return None
        with open(self.log_path) as f:
            lines = f.readlines()
            if not lines:
                return None
            try:
                last_entry = json.loads(lines[-1].strip())
                return last_entry.get("entry_hash")
            except (json.JSONDecodeError, KeyError):
                return None

    def log(
        self,
        event: str,
        user: str,
        action: str,
        resource: str,
        details: dict | None = None,
        ip_address: str | None = None,
    ) -> dict:
        """Write immutable audit log entry."""
        entry = {
            "timestamp": datetime.utcnow(),
            "event": event,
            "user": user,
            "action": action,
            "resource": resource,
            "details": details or {},
            "ip_address": ip_address,
            "prev_hash": self._last_hash,
        }

        entry["entry_hash"] = self._compute_entry_hash(entry, self._last_hash)

        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

        self._last_hash = entry["entry_hash"]
        return entry

    def verify_integrity(self) -> tuple[bool, str | None]:
        """Verify entire audit log chain integrity."""
        if not self.log_path.exists():
            return True, None

        prev_hash = None
        with open(self.log_path) as f:
            for line_num, line in enumerate(f, 1):
                try:
                    entry = json.loads(line.strip())
                    expected_hash = self._compute_entry_hash(entry, prev_hash)

                    if entry.get("entry_hash") != expected_hash:
                        return False, f"Tampering detected at line {line_num}"

                    prev_hash = entry.get("entry_hash")
                except json.JSONDecodeError as e:
                    return False, f"Invalid JSON at line {line_num}: {e}"

        return True, None

    def get_entries(
        self,
        user: str | None = None,
        event: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[dict]:
        """Query audit log entries with optional filters."""
        entries = []
        if not self.log_path.exists():
            return entries

        with open(self.log_path) as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if user and entry.get("user") != user:
                        continue
                    if event and entry.get("event") != event:
                        continue
                    if start_time:
                        entry_time = datetime.fromisoformat(entry["timestamp"])
                        if entry_time < start_time:
                            continue
                    if end_time:
                        entry_time = datetime.fromisoformat(entry["timestamp"])
                        if entry_time > end_time:
                            continue
                    entries.append(entry)
                except (json.JSONDecodeError, KeyError):
                    continue

        return entries
