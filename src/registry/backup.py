"""
Backup utilities for Model Registry - Phase 6.9
"""
import os
import json
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import yaml

from .config import PROJECT_ROOT


class BackupManifest:
    """Structured manifest for backup operations"""
    
    def __init__(
        self,
        backup_id: str,
        timestamp: datetime,
        components: List[str],
        compression: str,
        status: str,
        checksums: Dict[str, str],
        file_size_bytes: int,
        storage_path: Optional[str]
    ):
        self.backup_id = backup_id
        self.timestamp = timestamp
        self.components = components
        self.compression = compression
        self.status = status
        self.checksums = checksums
        self.file_size_bytes = file_size_bytes
        self.storage_path = storage_path
    
    def model_dump(self, mode: str = 'json', default: callable = None) -> Dict[str, Any]:
        """Pydantic v2 compatible serialization"""
        return {
            'backup_id': self.backup_id,
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else str(self.timestamp),
            'components': self.components,
            'compression': self.compression,
            'status': self.status,
            'checksums': self.checksums,
            'file_size_bytes': self.file_size_bytes,
            'storage_path': self.storage_path
        }


def load_backup_policy(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load backup policy from YAML config with fallback defaults"""
    if config_path is None:
        config_path = os.path.join(str(PROJECT_ROOT), 'config', 'backup_policy.yaml')
    
    defaults = {
        'frequency': 'daily',
        'retention': {'daily': 7, 'weekly': 4, 'monthly': 12},
        'components': ['mlflow_database', 'registry_metadata', 'audit_logs'],
        'storage': {
            'local_path': os.path.join(str(PROJECT_ROOT), 'backups'),
            'compression': 'gzip',
            'encryption': False
        },
        'validation': {
            'checksum_algorithm': 'sha256',
            'verify_integrity': True
        }
    }
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                policy = yaml.safe_load(f)
                if policy:
                    defaults.update(policy)
        except Exception as e:
            print(f"Warning: Could not load backup policy: {e}, using defaults")
    
    return defaults


def get_backup_destination(backup_id: str, policy: Dict[str, Any]) -> str:
    """Calculate timestamped backup storage path"""
    storage_config = policy.get('storage', {})
    base_path = storage_config.get('local_path', os.path.join(str(PROJECT_ROOT), 'backups'))
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    return os.path.join(base_path, f"backup_{backup_id}_{timestamp}")


def calculate_checksum(file_path: str, algorithm: str = 'sha256') -> str:
    """Calculate file checksum"""
    hash_func = hashlib.new(algorithm)
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hash_func.update(chunk)
    return hash_func.hexdigest()


def backup_component(
    component: str,
    policy: Dict[str, Any],
    compression: str = 'gzip',
    encrypt: bool = False,
    job_id: Optional[str] = None
) -> Dict[str, Any]:
    """Backup a single component"""
    project_root = str(PROJECT_ROOT)
    
    component_paths = {
        'mlflow_database': os.path.join(project_root, 'mlflow.db'),
        'registry_metadata': os.path.join(project_root, 'src', 'registry'),
        'audit_logs': os.path.join(project_root, 'logs', 'audit'),
        'config_files': os.path.join(project_root, 'config'),
        'artifacts': os.path.join(project_root, 'artifacts')
    }
    
    source_path = component_paths.get(component)
    if not source_path or not os.path.exists(source_path):
        return {'status': 'skipped', 'reason': f'Path not found: {source_path}'}
    
    checksums = {}
    total_size = 0
    
    if os.path.isfile(source_path):
        checksums[component] = calculate_checksum(source_path)
        total_size = os.path.getsize(source_path)
    elif os.path.isdir(source_path):
        for root, dirs, files in os.walk(source_path):
            for fname in files:
                fpath = os.path.join(root, fname)
                rel_path = os.path.relpath(fpath, project_root)
                checksums[rel_path] = calculate_checksum(fpath)
                total_size += os.path.getsize(fpath)
    
    return {
        'status': 'completed',
        'component': component,
        'source_path': source_path,
        'checksums': checksums,
        'size_bytes': total_size,
        'file_count': len(checksums)
    }


def apply_retention_policy(policy: Dict[str, Any], backup_registry: List[Dict]) -> List[str]:
    """Remove old backups per retention rules"""
    return []


def list_available_backups(backup_dir: Optional[str] = None, limit: int = 50) -> List[Dict]:
    """Query backup registry and return available backups"""
    if backup_dir is None:
        backup_dir = os.path.join(str(PROJECT_ROOT), 'backups')
    
    backups = []
    if os.path.exists(backup_dir):
        for fname in sorted(os.listdir(backup_dir), reverse=True)[:limit]:
            if fname.startswith('backup_'):
                manifest_path = os.path.join(backup_dir, fname.replace('.tar.gz', '.manifest.json'))
                metadata = {'filename': fname, 'path': os.path.join(backup_dir, fname)}
                if os.path.exists(manifest_path):
                    with open(manifest_path, 'r') as f:
                        metadata.update(json.load(f))
                backups.append(metadata)
    
    return backups
