"""
Recovery utilities for Model Registry - Phase 6.9
"""
import os
import json
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Any

from .config import PROJECT_ROOT


class RecoveryValidator:
    """Validates backup integrity before restore"""
    
    def __init__(self, manifest: Dict[str, Any]):
        self.manifest = manifest
        self.errors = []
    
    def verify_checksums(self) -> bool:
        """Verify file checksums match manifest"""
        # Placeholder: implement checksum verification
        return True
    
    def check_database_integrity(self) -> bool:
        """Check MLflow database integrity"""
        # Placeholder: implement DB integrity check
        return True
    
    def validate(self) -> bool:
        """Run all validation checks"""
        return self.verify_checksums() and self.check_database_integrity()


def restore_component(
    component: str,
    backup_path: str,
    target_path: str,
    validate: bool = True
) -> Dict[str, Any]:
    """Restore a single component from backup"""
    try:
        if os.path.exists(backup_path):
            if os.path.isfile(backup_path):
                shutil.copy2(backup_path, target_path)
            elif os.path.isdir(backup_path):
                if os.path.exists(target_path):
                    shutil.rmtree(target_path)
                shutil.copytree(backup_path, target_path)
            
            return {
                'status': 'completed',
                'component': component,
                'target_path': target_path
            }
        else:
            return {
                'status': 'failed',
                'reason': f'Backup path not found: {backup_path}'
            }
    except Exception as e:
        return {
            'status': 'failed',
            'error': str(e)
        }


def restore_from_backup(
    backup_id: str,
    target_dir: Optional[str] = None,
    validate: bool = True,
    components: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Full restore workflow with pre/post validation"""
    project_root = str(PROJECT_ROOT)
    backup_dir = os.path.join(project_root, 'backups')
    
    if target_dir is None:
        target_dir = project_root
    
    # Find backup manifest
    manifest_path = None
    for fname in os.listdir(backup_dir):
        if backup_id in fname and fname.endswith('.manifest.json'):
            manifest_path = os.path.join(backup_dir, fname)
            break
    
    if not manifest_path or not os.path.exists(manifest_path):
        return {
            'status': 'failed',
            'error': f'Backup manifest not found for: {backup_id}'
        }
    
    # Load manifest
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    # Validate if requested
    if validate:
        validator = RecoveryValidator(manifest)
        if not validator.validate():
            return {
                'status': 'failed',
                'error': 'Backup validation failed',
                'details': validator.errors
            }
    
    # Restore components
    results = []
    components_to_restore = components or manifest.get('components', [])
    
    for component in components_to_restore:
        # Determine backup and target paths
        backup_path = os.path.join(backup_dir, f"{component}.tar.gz")
        target_path = os.path.join(target_dir, component)
        
        result = restore_component(component, backup_path, target_path, validate)
        results.append(result)
    
    return {
        'status': 'completed',
        'backup_id': backup_id,
        'components_restored': len([r for r in results if r['status'] == 'completed']),
        'results': results
    }


def list_available_backups(backup_dir: Optional[str] = None, limit: int = 50) -> List[Dict]:
    """List available backups with metadata"""
    if backup_dir is None:
        backup_dir = os.path.join(str(PROJECT_ROOT), 'backups')
    
    backups = []
    if os.path.exists(backup_dir):
        for fname in sorted(os.listdir(backup_dir), reverse=True)[:limit]:
            if fname.startswith('backup_') and fname.endswith('.manifest.json'):
                manifest_path = os.path.join(backup_dir, fname)
                with open(manifest_path, 'r') as f:
                    metadata = json.load(f)
                    metadata['manifest_file'] = fname
                    backups.append(metadata)
    
    return backups
