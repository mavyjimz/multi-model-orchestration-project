#!/usr/bin/env python3
"""
Phase 1.1: Data Ingestion & Validation
Objective: Establish automated, validated data pipelines for intent classification.
"""

import os
import sys
import hashlib
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/p1-ingestion.log')
    ]
)
logger = logging.getLogger(__name__)


class DataIngestionValidator:
    """Phase 1.1: Automated data ingestion with validation and versioning."""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.input_dir = self.project_root / 'input-data' / 'router'
        self.raw_dir = self.project_root / 'input-data' / 'raw'
        self.logs_dir = self.project_root / 'logs'
        
        # Ensure directories exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Expected schema for intent classification dataset
        self.expected_columns = ['user_input', 'intent']
        
    def compute_checksum(self, file_path: str) -> str:
        """Generate SHA256 checksum for data integrity verification."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Validate dataframe has expected columns."""
        missing_columns = set(self.expected_columns) - set(df.columns)
        if missing_columns:
            logger.error(f"Missing columns: {missing_columns}")
            return False
        logger.info(f"Schema validation passed. Columns: {list(df.columns)}")
        return True
    
    def validate_quality(self, df: pd.DataFrame) -> dict:
        """Perform data quality checks and return metrics."""
        quality_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'null_counts': df.isnull().sum().to_dict(),
            'duplicate_rows': int(df.duplicated().sum()),
            'null_percentage': float((df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100)
        }
        
        logger.info(f"Data quality report: {quality_report}")
        
        # Quality thresholds
        if quality_report['null_percentage'] > 5.0:
            logger.warning(f"High null percentage: {quality_report['null_percentage']:.2f}%")
        
        if quality_report['duplicate_rows'] > 0:
            logger.warning(f"Duplicate rows found: {quality_report['duplicate_rows']}")
        
        return quality_report
    
    def ingest_dataset(self) -> dict:
        """Main ingestion pipeline for intent classification dataset."""
        logger.info("=" * 60)
        logger.info("Phase 1.1: Data Ingestion & Validation - STARTED")
        logger.info("=" * 60)
        
        ingestion_report = {
            'timestamp': datetime.now().isoformat(),
            'status': 'failed',
            'file_path': None,
            'checksum': None,
            'quality_metrics': None
        }
        
        try:
            # Step 1: Locate source file
            source_file = self.input_dir / 'chatbot_intent_classification.csv'
            
            if not source_file.exists():
                raise FileNotFoundError(f"Source file not found: {source_file}")
            
            logger.info(f"Source file located: {source_file}")
            
            # Step 2: Compute checksum for integrity
            checksum = self.compute_checksum(source_file)
            ingestion_report['checksum'] = checksum
            logger.info(f"SHA256 checksum: {checksum[:16]}...")
            
            # Step 3: Load and validate data
            df = pd.read_csv(source_file)
            logger.info(f"Loaded {len(df)} rows from CSV")
            
            # Step 4: Schema validation
            if not self.validate_schema(df):
                raise ValueError("Schema validation failed")
            
            # Step 5: Quality validation
            quality_metrics = self.validate_quality(df)
            ingestion_report['quality_metrics'] = quality_metrics
            
            # Step 6: Copy to raw directory (versioned location)
            dest_file = self.raw_dir / f"intent_classification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(dest_file, index=False)
            ingestion_report['file_path'] = str(dest_file)
            
            logger.info(f"Data copied to versioned location: {dest_file}")
            
            # Step 7: Save checksum file
            checksum_file = self.raw_dir / f"intent_classification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sha256"
            with open(checksum_file, 'w') as f:
                f.write(f"{checksum}  {dest_file.name}\n")
            
            ingestion_report['status'] = 'success'
            logger.info("=" * 60)
            logger.info("Phase 1.1: Data Ingestion & Validation - COMPLETED")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Ingestion failed: {str(e)}")
            ingestion_report['error'] = str(e)
        
        return ingestion_report


def main():
    """Execute Phase 1.1 data ingestion pipeline."""
    validator = DataIngestionValidator()
    report = validator.ingest_dataset()
    
    # Print final report
    print("\n" + "=" * 60)
    print("INGESTION REPORT")
    print("=" * 60)
    for key, value in report.items():
        if key != 'quality_metrics':
            print(f"{key}: {value}")
    print("=" * 60)
    
    # Exit with appropriate code
    sys.exit(0 if report['status'] == 'success' else 1)


if __name__ == '__main__':
    main()
