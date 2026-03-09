#!/usr/bin/env python3
"""
Phase 1.2: Feature Engineering Pipeline
Objective: Transform raw data into ML-ready features for intent classification.
"""

import os
import sys
import re
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/p2-feature-engineering.log')
    ]
)
logger = logging.getLogger(__name__)


class FeatureEngineeringPipeline:
    """Phase 1.2: Transform raw data into ML-ready features."""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.raw_dir = self.project_root / 'input-data' / 'raw'
        self.processed_dir = self.project_root / 'input-data' / 'processed'
        self.logs_dir = self.project_root / 'logs'
        
        # Ensure directories exist
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize label encoder
        self.label_encoder = LabelEncoder()
        
    def load_latest_raw_data(self) -> pd.DataFrame:
        """Load the most recent versioned raw dataset."""
        raw_files = list(self.raw_dir.glob('intent_classification_*.csv'))
        
        if not raw_files:
            raise FileNotFoundError("No versioned raw data found in input-data/raw/")
        
        # Get latest file by modification time
        latest_file = max(raw_files, key=lambda p: p.stat().st_mtime)
        
        logger.info(f"Loading latest raw data: {latest_file}")
        df = pd.read_csv(latest_file)
        
        return df
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text data."""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract additional features from text data."""
        logger.info("Extracting text-based features...")
        
        # Text length
        df['text_length'] = df['user_input_clean'].apply(len)
        
        # Word count
        df['word_count'] = df['user_input_clean'].apply(lambda x: len(x.split()))
        
        # Average word length
        df['avg_word_length'] = df['user_input_clean'].apply(
            lambda x: np.mean([len(word) for word in x.split()]) if x else 0
        )
        
        # Exclamation mark count (emotion indicator)
        df['exclamation_count'] = df['user_input'].apply(
            lambda x: str(x).count('!')
        )
        
        # Question mark count
        df['question_count'] = df['user_input'].apply(
            lambda x: str(x).count('?')
        )
        
        logger.info(f"Features extracted: text_length, word_count, avg_word_length, exclamation_count, question_count")
        
        return df
    
    def encode_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode intent labels to numeric values."""
        logger.info("Encoding intent labels...")
        
        df['intent_encoded'] = self.label_encoder.fit_transform(df['intent'])
        
        # Create mapping dictionary for reference
        label_mapping = dict(zip(
            self.label_encoder.classes_,
            self.label_encoder.transform(self.label_encoder.classes_)
        ))
        
        logger.info(f"Label mapping: {label_mapping}")
        
        # Save label mapping
        mapping_file = self.processed_dir / 'intent_label_mapping.txt'
        with open(mapping_file, 'w') as f:
            for intent, encoded in label_mapping.items():
                f.write(f"{encoded}: {intent}\n")
        
        logger.info(f"Label mapping saved to: {mapping_file}")
        
        return df
    
    def validate_features(self, df: pd.DataFrame) -> bool:
        """Validate feature engineering output."""
        logger.info("Validating feature engineering output...")
        
        # Check for null values in critical columns
        critical_columns = ['user_input_clean', 'intent_encoded']
        null_counts = df[critical_columns].isnull().sum()
        
        if null_counts.any():
            logger.error(f"Null values found in critical columns: {null_counts[null_counts > 0]}")
            return False
        
        # Check for expected columns
        expected_columns = ['user_input_clean', 'intent', 'intent_encoded', 
                          'text_length', 'word_count', 'avg_word_length']
        missing_columns = set(expected_columns) - set(df.columns)
        
        if missing_columns:
            logger.error(f"Missing expected columns: {missing_columns}")
            return False
        
        logger.info("Feature validation passed")
        return True
    
    def engineer_features(self) -> dict:
        """Main feature engineering pipeline."""
        logger.info("=" * 60)
        logger.info("Phase 1.2: Feature Engineering Pipeline - STARTED")
        logger.info("=" * 60)
        
        engineering_report = {
            'timestamp': datetime.now().isoformat(),
            'status': 'failed',
            'input_rows': None,
            'output_rows': None,
            'duplicates_removed': None,
            'features_created': None,
            'output_file': None
        }
        
        try:
            # Step 1: Load raw data
            df = self.load_latest_raw_data()
            engineering_report['input_rows'] = len(df)
            logger.info(f"Loaded {len(df)} rows from raw data")
            
            # Step 2: Remove duplicates
            initial_rows = len(df)
            df = df.drop_duplicates(subset=['user_input', 'intent'], keep='first')
            duplicates_removed = initial_rows - len(df)
            engineering_report['duplicates_removed'] = duplicates_removed
            logger.info(f"Removed {duplicates_removed} duplicate rows")
            
            # Step 3: Clean text
            logger.info("Cleaning and normalizing text...")
            df['user_input_clean'] = df['user_input'].apply(self.clean_text)
            
            # Step 4: Extract features
            df = self.extract_features(df)
            
            # Step 5: Encode labels
            df = self.encode_labels(df)
            
            # Step 6: Validate features
            if not self.validate_features(df):
                raise ValueError("Feature validation failed")
            
            # Step 7: Save processed data
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = self.processed_dir / f'intent_features_{timestamp}.csv'
            df.to_csv(output_file, index=False)
            engineering_report['output_file'] = str(output_file)
            engineering_report['output_rows'] = len(df)
            
            # Count features created
            feature_columns = ['text_length', 'word_count', 'avg_word_length', 
                             'exclamation_count', 'question_count']
            engineering_report['features_created'] = len([c for c in feature_columns if c in df.columns])
            
            logger.info(f"Processed data saved to: {output_file}")
            logger.info(f"Final dataset: {len(df)} rows, {len(df.columns)} columns")
            
            engineering_report['status'] = 'success'
            logger.info("=" * 60)
            logger.info("Phase 1.2: Feature Engineering Pipeline - COMPLETED")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {str(e)}")
            engineering_report['error'] = str(e)
        
        return engineering_report


def main():
    """Execute Phase 1.2 feature engineering pipeline."""
    pipeline = FeatureEngineeringPipeline()
    report = pipeline.engineer_features()
    
    # Print final report
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING REPORT")
    print("=" * 60)
    for key, value in report.items():
        print(f"{key}: {value}")
    print("=" * 60)
    
    # Exit with appropriate code
    sys.exit(0 if report['status'] == 'success' else 1)


if __name__ == '__main__':
    main()
