#!/usr/bin/env python3
"""
p6.2-semantic-versioning.py
Phase 6.2: Model Versioning Strategy Implementation

Implements semantic versioning (SemVer) for model registry with:
- Version parsing and validation (v1.0.1 format)
- Git commit lineage tracking
- Version comparison utilities
- Automatic version bumping based on change type
- Integration with MLflow Model Registry
"""

import os
import sys
import re
import json
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
import mlflow
from mlflow.tracking import MlflowClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/p6.2-semantic-versioning.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
MLFLOWS_DIR = PROJECT_ROOT / 'mlruns'
CONFIG_DIR = PROJECT_ROOT / 'config' / 'registry'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'phase6'

MLFLOW_TRACKING_URI = f"file:{MLFLOWS_DIR.resolve()}"
MODEL_NAME = "intent-classifier-sgd"


class SemanticVersion:
    """Semantic versioning helper for model versions."""
    
    VERSION_PATTERN = re.compile(r'^v?(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9.-]+))?(?:\+([a-zA-Z0-9.-]+))?$')
    
    def __init__(self, major: int, minor: int, patch: int, 
                 prerelease: Optional[str] = None, build: Optional[str] = None):
        self.major = major
        self.minor = minor
        self.patch = patch
        self.prerelease = prerelease
        self.build = build
    
    @classmethod
    def parse(cls, version_str: str) -> 'SemanticVersion':
        """Parse version string like 'v1.0.1' or '1.0.1-rc1'."""
        match = cls.VERSION_PATTERN.match(version_str.strip())
        if not match:
            raise ValueError(f"Invalid version format: {version_str}")
        
        major, minor, patch = map(int, match.groups()[:3])
        prerelease, build = match.groups()[3:5]
        return cls(major, minor, patch, prerelease, build)
    
    def bump(self, level: str = 'patch') -> 'SemanticVersion':
        """Bump version by level: 'major', 'minor', or 'patch'."""
        if level == 'major':
            return SemanticVersion(self.major + 1, 0, 0)
        elif level == 'minor':
            return SemanticVersion(self.major, self.minor + 1, 0)
        else:  # patch
            return SemanticVersion(self.major, self.minor, self.patch + 1)
    
    def __str__(self) -> str:
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        if self.build:
            version += f"+{self.build}"
        return version
    
    def __repr__(self) -> str:
        return f"SemanticVersion(v{self})"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, SemanticVersion):
            return False
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)
    
    def __lt__(self, other) -> bool:
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)


def get_git_commit_info() -> dict:
    """Extract current git commit metadata for lineage tracking."""
    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
        short_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
        commit_msg = subprocess.check_output(['git', 'log', '-1', '--pretty=%B']).decode().strip()
        branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode().strip()
        tags = subprocess.check_output(['git', 'tag', '--points-at', 'HEAD']).decode().strip().split('\n')
        tags = [t for t in tags if t]  # Remove empty strings
        
        return {
            "commit_hash": commit_hash,
            "short_hash": short_hash,
            "commit_message": commit_msg,
            "branch": branch,
            "tags": tags,
            "timestamp": datetime.now().isoformat()
        }
    except subprocess.CalledProcessError as e:
        logger.warning(f"Could not retrieve git info: {e}")
        return {"error": str(e)}


def get_current_model_version(client: MlflowClient, model_name: str) -> Optional[SemanticVersion]:
    """Get the latest registered version of a model from MLflow."""
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            return None
        
        # Sort by version number descending
        latest = max(versions, key=lambda v: int(v.version))
        return SemanticVersion.parse(f"1.0.{latest.version}")  # MLflow uses integer versions
    except Exception as e:
        logger.warning(f"Could not retrieve model version: {e}")
        return None


def determine_version_bump(change_type: str, current_version: SemanticVersion) -> SemanticVersion:
    """Determine next version based on change type."""
    bump_rules = {
        'breaking': 'major',
        'feature': 'minor',
        'fix': 'patch',
        'docs': 'patch',
        'refactor': 'patch',
        'test': 'patch',
        'chore': 'patch'
    }
    
    bump_level = bump_rules.get(change_type, 'patch')
    return current_version.bump(bump_level)


def register_model_with_version(model_path: Path, version: SemanticVersion, 
                                metadata: dict, git_info: dict) -> dict:
    """Register a model version in MLflow with full lineage metadata."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    
    with mlflow.start_run(run_name=f"{MODEL_NAME}-v{version}"):
        # Log metadata
        for key, value in metadata.get('metrics', {}).items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)
        
        for key, value in metadata.get('hyperparameters', {}).items():
            mlflow.log_param(key, value)
        
        # Log lineage tags
        mlflow.set_tags({
            "model.version": str(version),
            "model.semver.major": version.major,
            "model.semver.minor": version.minor,
            "model.semver.patch": version.patch,
            "git.commit_hash": git_info.get('commit_hash', 'unknown'),
            "git.short_hash": git_info.get('short_hash', 'unknown'),
            "git.branch": git_info.get('branch', 'unknown'),
            "git.commit_message": git_info.get('commit_message', '')[:200],
            "registry.registered_at": datetime.now().isoformat()
        })
        
        # Log model if path exists
        if model_path.exists():
            try:
                mlflow.sklearn.log_model(
                    sk_model=str(model_path),
                    artifact_path="model",
                    registered_model_name=MODEL_NAME
                )
                logger.info(f"Logged model artifact: {model_path}")
            except Exception as e:
                logger.warning(f"Could not log model artifact: {e}")
        
        run_id = mlflow.active_run().info.run_id
    
    logger.info(f"Registered {MODEL_NAME} v{version} with run_id: {run_id}")
    return {"run_id": run_id, "version": str(version), "model_name": MODEL_NAME}


def generate_version_changelog(version: SemanticVersion, changes: list) -> str:
    """Generate changelog entry for a model version."""
    changelog = f"""## [{version}] - {datetime.now().strftime('%Y-%m-%d')}

### Changes
"""
    for change in changes:
        changelog += f"- {change}\n"
    
    changelog += f"""
### Lineage
- Git Commit: {get_git_commit_info().get('short_hash', 'unknown')}
- Branch: {get_git_commit_info().get('branch', 'unknown')}
- Registered: {datetime.now().isoformat()}
"""
    return changelog


def validate_versioning_setup() -> Tuple[bool, list]:
    """Validate versioning strategy configuration."""
    checks = []
    
    # Check 1: SemVer parser works
    try:
        v = SemanticVersion.parse("v1.0.1")
        checks.append(("SemVer parser", v.major == 1 and v.minor == 0 and v.patch == 1))
    except Exception as e:
        checks.append(("SemVer parser", False, str(e)))
    
    # Check 2: Git info retrievable
    git_info = get_git_commit_info()
    checks.append(("Git lineage tracking", "commit_hash" in git_info))
    
    # Check 3: MLflow client accessible
    try:
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        client.search_model_versions(f"name='{MODEL_NAME}'")
        checks.append(("MLflow model registry access", True))
    except Exception as e:
        checks.append(("MLflow model registry access", False, str(e)))
    
    # Check 4: Version bump logic
    try:
        current = SemanticVersion(1, 0, 1)
        next_ver = determine_version_bump('feature', current)
        checks.append(("Version bump logic", str(next_ver) == "1.1.0"))
    except Exception as e:
        checks.append(("Version bump logic", False, str(e)))
    
    return all(c[1] for c in checks), checks


def main():
    """Main execution for p6.2 semantic versioning."""
    parser = argparse.ArgumentParser(description='Phase 6.2: Semantic Versioning Strategy')
    parser.add_argument('--current-version', type=str, default='1.0.1',
                       help='Current model version (default: 1.0.1)')
    parser.add_argument('--change-type', type=str, default='fix',
                       choices=['breaking', 'feature', 'fix', 'docs', 'refactor', 'test', 'chore'],
                       help='Type of change for version bump')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without registering')
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info(f"Phase 6.2: Semantic Versioning Strategy")
    logger.info("=" * 60)
    
    try:
        # Initialize
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        
        # Parse current version
        current_version = SemanticVersion.parse(args.current_version)
        logger.info(f"Current version: v{current_version}")
        
        # Get git lineage
        git_info = get_git_commit_info()
        logger.info(f"Git commit: {git_info.get('short_hash', 'unknown')} on {git_info.get('branch', 'unknown')}")
        
        # Determine next version
        next_version = determine_version_bump(args.change_type, current_version)
        logger.info(f"Next version ({args.change_type}): v{next_version}")
        
        # Load existing model metadata if available
        manifest_path = PROJECT_ROOT / 'models' / 'phase4' / 'model_manifest_v1.0.1.json'
        metadata = {}
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Loaded metadata from: {manifest_path}")
        
        # Validate setup
        success, checks = validate_versioning_setup()
        for check in checks:
            status = "PASS" if check[1] else "FAIL"
            msg = f"[{status}] {check[0]}"
            if len(check) == 3:
                msg += f": {check[2]}"
            logger.info(msg)
        
        if not success:
            logger.warning("Some validation checks failed - proceeding with caution")
        
        # Register model (unless dry run)
        if args.dry_run:
            logger.info("[DRY RUN] Would register model with:")
            logger.info(f"  - Version: v{next_version}")
            logger.info(f"  - Model: {MODEL_NAME}")
            logger.info(f"  - Git: {git_info.get('short_hash')}")
        else:
            model_path = PROJECT_ROOT / 'models' / 'phase4' / 'sgd_v1.0.1.pkl'
            result = register_model_with_version(model_path, next_version, metadata, git_info)
            logger.info(f"Registration result: {result}")
        
        # Generate changelog
        changelog = generate_version_changelog(next_version, [
            f"{args.change_type.capitalize()} change per semantic versioning policy",
            f"Lineage tracked to git commit {git_info.get('short_hash')}"
        ])
        
        changelog_path = RESULTS_DIR / f'CHANGELOG_v{next_version}.md'
        with open(changelog_path, 'w') as f:
            f.write(changelog)
        logger.info(f"Changelog saved: {changelog_path}")
        
        # Summary
        summary = {
            "phase": "6.2",
            "current_version": str(current_version),
            "next_version": str(next_version),
            "change_type": args.change_type,
            "git_lineage": git_info,
            "validation_passed": success,
            "changelog_path": str(changelog_path),
            "timestamp": datetime.now().isoformat()
        }
        
        summary_path = RESULTS_DIR / 'p6.2_versioning_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary saved: {summary_path}")
        
        logger.info("=" * 60)
        logger.info(f"Phase 6.2 Complete: v{current_version} -> v{next_version}")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Versioning setup failed: {e}", exc_info=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
