"""Data retention policy enforcement for GDPR compliance."""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path


class DataRetentionPolicy:
    """Enforce data retention policies per GDPR requirements."""

    def __init__(
        self,
        retention_days: int = 365,
        data_dirs: list[str] | None = None,
    ):
        self.retention_days = retention_days or int(os.getenv("DATA_RETENTION_DAYS", "365"))
        self.data_dirs = data_dirs or [
            "logs/audit",
            "logs",
            "results",
            "mlruns",
        ]

    def get_retention_cutoff(self) -> datetime:
        """Calculate cutoff date for data retention."""
        return datetime.utcnow() - timedelta(days=self.retention_days)

    def scan_for_old_files(self) -> list[dict]:
        """Scan data directories for files exceeding retention period."""
        cutoff = self.get_retention_cutoff()
        old_files = []

        for dir_path in self.data_dirs:
            path = Path(dir_path)
            if not path.exists():
                continue

            for file_path in path.rglob("*"):
                if file_path.is_file():
                    try:
                        mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if mtime < cutoff:
                            old_files.append(
                                {
                                    "path": str(file_path),
                                    "modified": mtime.isoformat(),
                                    "size_bytes": file_path.stat().st_size,
                                    "days_old": (datetime.utcnow() - mtime).days,
                                }
                            )
                    except (OSError, ValueError):
                        continue

        return sorted(old_files, key=lambda x: x["days_old"], reverse=True)

    def generate_retention_report(
        self, output_path: str = "results/compliance/retention_report.json"
    ) -> dict:
        """Generate compliance report for audit purposes."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        old_files = self.scan_for_old_files()
        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "retention_days": self.retention_days,
            "cutoff_date": self.get_retention_cutoff().isoformat(),
            "files_exceeding_retention": len(old_files),
            "total_size_bytes": sum(f["size_bytes"] for f in old_files),
            "files": old_files[:100],  # Limit to first 100 for readability
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        return report

    def cleanup_old_files(self, dry_run: bool = True) -> list[str]:
        """Remove files exceeding retention period."""
        old_files = self.scan_for_old_files()
        deleted = []

        for file_info in old_files:
            file_path = Path(file_info["path"])
            if dry_run:
                print(f"[DRY RUN] Would delete: {file_path}")
            else:
                try:
                    file_path.unlink()
                    deleted.append(file_info["path"])
                    print(f"Deleted: {file_path}")
                except OSError as e:
                    print(f"Failed to delete {file_path}: {e}")

        return deleted
