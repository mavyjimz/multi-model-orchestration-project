#!/usr/bin/env python3
"""
Phase 1.5: Data Versioning & Lineage Tracking
Creates version manifests and checksums for all datasets.
"""
import os
import sys
import logging
import hashlib
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/p1.5-data-versioning.log')
    ]
)
logger = logging.getLogger(__name__)


def calculate_sha256(file_path: Path) -> str:
    """Calculate SHA256 checksum for a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def create_version_manifest():
    """Create VERSION.md with complete data lineage."""
    logger.info("Creating data version manifest...")
    
    manifest = []
    manifest.append("# Data Version Manifest\n")
    manifest.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    manifest.append(f"**Project**: Multi-Model Orchestration System\n\n")
    
    # Phase 1.1: Raw Data
    manifest.append("## Phase 1.1: Raw Data Sources\n\n")
    raw_dir = Path("input-data/raw")
    if raw_dir.exists():
        for file in sorted(raw_dir.glob("*.csv")):
            if not file.name.startswith('.'):
                checksum = calculate_sha256(file)
                size = file.stat().st_size / 1024  # KB
                manifest.append(f"- `{file.name}`: {size:.1f} KB | SHA256: `{checksum[:16]}...`\n")
    
    manifest.append("\n## Phase 1.2: Merged Dataset\n\n")
    merged_files = sorted(raw_dir.glob("intent_classification_merged*.csv"))
    for file in merged_files:
        checksum = calculate_sha256(file)
        size = file.stat().st_size / 1024
        manifest.append(f"- `{file.name}`: {size:.1f} KB | SHA256: `{checksum[:16]}...`\n")
    
    manifest.append("\n## Phase 1.3: Processed Features\n\n")
    processed_dir = Path("input-data/processed")
    if processed_dir.exists():
        for file in sorted(processed_dir.glob("intent_features*.csv")):
            checksum = calculate_sha256(file)
            size = file.stat().st_size / 1024
            manifest.append(f"- `{file.name}`: {size:.1f} KB | SHA256: `{checksum[:16]}...`\n")
    
    manifest.append("\n## Phase 1.4: Train/Val/Test Splits\n\n")
    splits_dir = Path("input-data/splits")
    if splits_dir.exists():
        for split_type in ['train', 'val', 'test']:
            files = sorted(splits_dir.glob(f"{split_type}_*.csv"))
            if files:
                latest = files[-1]
                checksum = calculate_sha256(latest)
                size = latest.stat().st_size / 1024
                manifest.append(f"- `{latest.name}`: {size:.1f} KB | SHA256: `{checksum[:16]}...`\n")
    
    manifest.append("\n## Version History\n\n")
    manifest.append("| Version | Date | Description | Samples |\n")
    manifest.append("|---------|------|-------------|---------|\n")
    manifest.append("| v1.0 | 2026-03-09 | Initial dataset (original) | 1,000 |\n")
    manifest.append("| v1.1 | 2026-03-09 | Multi-source merge (ATIS + Chatbots) | 6,121 |\n")
    manifest.append("| v1.2 | 2026-03-09 | Feature engineering | 4,786 |\n")
    manifest.append("| v1.3 | 2026-03-10 | Stratified splits | 4,774 |\n")
    
    manifest_path = Path("VERSION.md")
    with open(manifest_path, 'w') as f:
        f.writelines(manifest)
    
    logger.info(f"Version manifest saved to: {manifest_path}")
    return manifest_path


def create_checksum_file():
    """Create checksums.sha256 file for all datasets."""
    logger.info("Creating SHA256 checksum file...")
    
    checksums = []
    all_files = []
    
    # Collect all CSV files
    for pattern in [
        "input-data/raw/*.csv",
        "input-data/processed/*.csv",
        "input-data/splits/*.csv"
    ]:
        all_files.extend(Path(".").glob(pattern))
    
    for file in sorted(all_files):
        checksum = calculate_sha256(file)
        checksums.append(f"{checksum}  {file}\n")
    
    checksum_file = Path("input-data/checksums.sha256")
    with open(checksum_file, 'w') as f:
        f.writelines(checksums)
    
    logger.info(f"Checksums saved to: {checksum_file}")
    return checksum_file


def main():
    logger.info("=" * 60)
    logger.info("Phase 1.5: Data Versioning & Lineage Tracking - STARTED")
    logger.info("=" * 60)
    
    # Create version manifest
    create_version_manifest()
    
    # Create checksum file
    create_checksum_file()
    
    logger.info("=" * 60)
    logger.info("Phase 1.5: Data Versioning & Lineage Tracking - COMPLETED")
    logger.info("=" * 60)
    
    print("\n✅ Phase 1 COMPLETE!")
    print("   - VERSION.md: Data lineage manifest")
    print("   - checksums.sha256: File integrity verification")


if __name__ == "__main__":
    main()
