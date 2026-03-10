#!/usr/bin/env python3
"""
Phase 1.4: Stratified Data Splitting Pipeline
Splits processed data into train/validation/test sets with stratified sampling.
"""
import os
import sys
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/p1.4-data-splitting.log')
    ]
)
logger = logging.getLogger(__name__)


class DataSplittingPipeline:
    """Phase 1.4: Split processed data into train/val/test sets."""
    
    def __init__(self):
        self.processed_dir = Path("input-data/processed")
        self.splits_dir = Path("input-data/splits")
        self.random_state = 42
        self.test_size = 0.15
        self.val_size = 0.15
        self.min_samples_per_class = 2
        
    def load_latest_processed_data(self) -> pd.DataFrame:
        """Load the most recent processed dataset."""
        processed_files = list(self.processed_dir.glob('intent_features_*.csv'))
        
        if not processed_files:
            raise FileNotFoundError(
                "No processed data found in input-data/processed/"
            )
        
        latest_file = max(processed_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"Loading latest processed  {latest_file}")
        
        df = pd.read_csv(latest_file)
        logger.info(f"Loaded {len(df)} rows from processed data")
        return df
    
    def filter_rare_classes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove classes with too few samples for stratified splitting."""
        class_counts = df['intent_encoded'].value_counts()
        rare_classes = class_counts[class_counts < self.min_samples_per_class].index.tolist()
        
        if rare_classes:
            logger.warning(f"Removing {len(rare_classes)} classes with < {self.min_samples_per_class} samples: {rare_classes}")
            df_filtered = df[~df['intent_encoded'].isin(rare_classes)]
            logger.info(f"Removed {len(df) - len(df_filtered)} samples from rare classes")
            return df_filtered
        return df
    
    def create_stratified_splits(self, df: pd.DataFrame):
        """Create stratified train/val/test splits."""
        logger.info("Creating stratified splits (70% train, 15% val, 15% test)")
        
        # Separate features and labels
        X = df.drop('intent_encoded', axis=1)
        y = df['intent_encoded']
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        # Second split: separate train and validation
        val_adjusted = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_adjusted,
            random_state=self.random_state,
            stratify=y_temp
        )
        
        # Reconstruct full DataFrames
        train_df = X_train.copy()
        train_df['intent_encoded'] = y_train
        
        val_df = X_val.copy()
        val_df['intent_encoded'] = y_val
        
        test_df = X_test.copy()
        test_df['intent_encoded'] = y_test
        
        logger.info(f"Train set: {len(train_df)} rows ({len(train_df)/len(df)*100:.1f}%)")
        logger.info(f"Validation set: {len(val_df)} rows ({len(val_df)/len(df)*100:.1f}%)")
        logger.info(f"Test set: {len(test_df)} rows ({len(test_df)/len(df)*100:.1f}%)")
        
        return train_df, val_df, test_df
    
    def validate_splits(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """Validate class distribution across splits."""
        logger.info("Validating class distribution across splits...")
        
        # Get unique classes from each split
        train_classes = set(int(x) for x in train_df['intent_encoded'].unique())
        val_classes = set(int(x) for x in val_df['intent_encoded'].unique())
        test_classes = set(int(x) for x in test_df['intent_encoded'].unique())
        all_classes = train_classes.union(val_classes).union(test_classes)
        
        for split_name, split_classes in [('train', train_classes), ('val', val_classes), ('test', test_classes)]:
            missing = all_classes - split_classes
            if missing:
                logger.warning(f"{split_name} set missing {len(missing)} classes")
            else:
                logger.info(f"{split_name} set: All {len(all_classes)} classes present")
        
        logger.info("Class distribution validation complete")
        return True
    
    def save_splits(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """Save splits to input-data/splits/."""
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        train_path = self.splits_dir / f"train_{timestamp}.csv"
        val_path = self.splits_dir / f"val_{timestamp}.csv"
        test_path = self.splits_dir / f"test_{timestamp}.csv"
        
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        logger.info(f"Train set saved to: {train_path}")
        logger.info(f"Validation set saved to: {val_path}")
        logger.info(f"Test set saved to: {test_path}")
        
        return train_path, val_path, test_path
    
    def generate_split_report(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, excluded_count: int):
        """Generate and save a split summary report."""
        total = len(train_df) + len(val_df) + len(test_df)
        
        report_path = self.splits_dir / "split_summary.txt"
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("DATA SPLITTING REPORT - PHASE 1.4\n")
            f.write("=" * 60 + "\n")
            f.write(f"timestamp: {datetime.now().isoformat()}\n")
            f.write(f"status: success\n")
            f.write(f"total_samples: {total}\n")
            f.write(f"train_samples: {len(train_df)} ({len(train_df)/total*100:.1f}%)\n")
            f.write(f"val_samples: {len(val_df)} ({len(val_df)/total*100:.1f}%)\n")
            f.write(f"test_samples: {len(test_df)} ({len(test_df)/total*100:.1f}%)\n")
            f.write(f"unique_intents: {train_df['intent_encoded'].nunique()}\n")
            f.write(f"random_state: {self.random_state}\n")
            f.write(f"excluded_classes_count: {excluded_count}\n")
            f.write("=" * 60 + "\n")
        
        logger.info(f"Split report saved to: {report_path}")
        return total
    
    def run(self):
        """Execute the full data splitting pipeline."""
        logger.info("=" * 60)
        logger.info("Phase 1.4: Stratified Data Splitting Pipeline - STARTED")
        logger.info("=" * 60)
        
        # Load processed data
        df = self.load_latest_processed_data()
        
        # Filter rare classes
        df_filtered = self.filter_rare_classes(df)
        excluded_count = len(df) - len(df_filtered)
        
        # Create stratified splits
        train_df, val_df, test_df = self.create_stratified_splits(df_filtered)
        
        # Validate splits
        self.validate_splits(train_df, val_df, test_df)
        
        # Save splits
        self.save_splits(train_df, val_df, test_df)
        
        # Generate report
        total = self.generate_split_report(train_df, val_df, test_df, excluded_count)
        
        logger.info("=" * 60)
        logger.info("Phase 1.4: Stratified Data Splitting Pipeline - COMPLETED")
        logger.info("=" * 60)
        
        # Print summary
        print("\n" + "=" * 60)
        print("DATA SPLITTING REPORT - PHASE 1.4")
        print("=" * 60)
        print(f"Total samples: {total}")
        print(f"Train: {len(train_df)} (70%)")
        print(f"Validation: {len(val_df)} (15%)")
        print(f"Test: {len(test_df)} (15%)")
        print(f"Unique intents: {train_df['intent_encoded'].nunique()}")
        print(f"Excluded classes: {excluded_count}")
        print("=" * 60)


if __name__ == "__main__":
    pipeline = DataSplittingPipeline()
    pipeline.run()
