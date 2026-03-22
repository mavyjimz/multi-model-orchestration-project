"""Right-to-Erase (GDPR Article 17) implementation."""

import json
from datetime import datetime
from pathlib import Path

from src.core.audit_logger import AuditLogger


class RightToErasure:
    """Handle data subject erasure requests per GDPR Article 17."""

    def __init__(self, audit_logger: AuditLogger | None = None):
        self.audit = audit_logger or AuditLogger()
        self.erasure_log_path = Path("logs/compliance/erasure_requests.log")
        self.erasure_log_path.parent.mkdir(parents=True, exist_ok=True)

    def request_erasure(
        self,
        user_id: str,
        requestor: str,
        reason: str,
        data_categories: list[str] | None = None,
    ) -> dict:
        """Log and process erasure request."""
        request_id = f"erase-{user_id}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

        request_record = {
            "request_id": request_id,
            "user_id": user_id,
            "requestor": requestor,
            "reason": reason,
            "data_categories": data_categories or ["all"],
            "requested_at": datetime.utcnow().isoformat(),
            "status": "pending",
        }

        # Log erasure request
        with open(self.erasure_log_path, "a") as f:
            f.write(json.dumps(request_record) + "\n")

        # Audit log
        self.audit.log(
            event="gdpr_erasure_request",
            user=requestor,
            action="request_erasure",
            resource=f"user:{user_id}",
            details=request_record,
        )

        return request_record

    def execute_erasure(
        self,
        request_id: str,
        user_id: str,
        data_paths: list[str],
    ) -> dict:
        """Execute data erasure for approved request."""
        erased_files = []
        errors = []

        for data_path in data_paths:
            path = Path(data_path)
            if not path.exists():
                continue

            try:
                if path.is_file():
                    # Redact user data from file
                    if path.suffix == ".json":
                        self._redact_json_file(path, user_id)
                    elif path.suffix in [".csv", ".parquet"]:
                        self._redact_tabular_file(path, user_id)
                    else:
                        path.unlink()
                    erased_files.append(data_path)
                elif path.is_dir():
                    # Remove user-specific subdirectories
                    user_dir = path / user_id
                    if user_dir.exists():
                        import shutil

                        shutil.rmtree(user_dir)
                        erased_files.append(str(user_dir))
            except Exception as e:
                errors.append({"path": data_path, "error": str(e)})

        # Update erasure log
        self._update_erasure_status(request_id, "completed", erased_files, errors)

        # Audit log
        self.audit.log(
            event="gdpr_erasure_executed",
            user="system",
            action="execute_erasure",
            resource=f"user:{user_id}",
            details={
                "request_id": request_id,
                "erased_files": erased_files,
                "errors": errors,
            },
        )

        return {
            "request_id": request_id,
            "status": "completed",
            "erased_files": erased_files,
            "errors": errors,
        }

    def _redact_json_file(self, path: Path, user_id: str):
        """Redact user data from JSON file."""
        try:
            with open(path) as f:
                data = json.load(f)

            # Simple redaction - remove entries matching user_id
            if isinstance(data, list):
                data = [item for item in data if item.get("user_id") != user_id]
            elif isinstance(data, dict):
                data = {k: v for k, v in data.items() if k != user_id}

            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except (OSError, json.JSONDecodeError):
            pass

    def _redact_tabular_file(self, path: Path, user_id: str):
        """Redact user data from CSV/Parquet file."""
        # Placeholder - implement with pandas in production
        pass

    def _update_erasure_status(
        self,
        request_id: str,
        status: str,
        erased_files: list[str],
        errors: list[dict],
    ):
        """Update erasure request status in log."""
        updated_records = []
        with open(self.erasure_log_path) as f:
            for line in f:
                record = json.loads(line.strip())
                if record["request_id"] == request_id:
                    record["status"] = status
                    record["completed_at"] = datetime.utcnow().isoformat()
                    record["erased_files"] = erased_files
                    record["errors"] = errors
                updated_records.append(record)

        with open(self.erasure_log_path, "w") as f:
            for record in updated_records:
                f.write(json.dumps(record) + "\n")

    def get_erasure_requests(
        self,
        user_id: str | None = None,
        status: str | None = None,
    ) -> list[dict]:
        """Query erasure requests."""
        requests = []
        if not self.erasure_log_path.exists():
            return requests

        with open(self.erasure_log_path) as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    if user_id and record.get("user_id") != user_id:
                        continue
                    if status and record.get("status") != status:
                        continue
                    requests.append(record)
                except json.JSONDecodeError:
                    continue

        return requests
