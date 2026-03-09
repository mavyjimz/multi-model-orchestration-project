#!/usr/bin/env python3
"""
Phase 1.1: Multi-Source Dataset Merger
Merges original, ATIS, and chatbots datasets into unified CSV.
"""
import pandas as pd
import glob
from pathlib import Path
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_and_normalize_datasets():
    """Load all datasets and normalize to (user_input, intent, source) format."""
    all_data = []
    
    # 1. Load original dataset
    original_files = glob.glob("input-data/raw/intent_classification_*.csv")
    if original_files:
        df_orig = pd.read_csv(original_files[0])
        df_orig['source'] = 'original'
        all_data.append(df_orig)
        logger.info(f"Loaded original: {len(df_orig)} rows")
    
    # 2. Load ATIS dataset (swap columns: intent,user -> user_input,intent)
    atis_file = "input-data/raw/atis-temp/atis_intents.csv"
    if Path(atis_file).exists():
        df_atis = pd.read_csv(atis_file, header=None, names=['intent', 'user_input'])
        df_atis = df_atis[['user_input', 'intent']]  # Reorder columns
        df_atis['source'] = 'atis'
        all_data.append(df_atis)
        logger.info(f"Loaded ATIS: {len(df_atis)} rows")
    
    # 3. Load chatbots dataset
    chatbots_file = "input-data/raw/chatbots-temp/chatbots_intents.csv"
    if Path(chatbots_file).exists():
        df_chat = pd.read_csv(chatbots_file)
        df_chat['source'] = 'chatbots'
        all_data.append(df_chat)
        logger.info(f"Loaded chatbots: {len(df_chat)} rows")
    
    return all_data

def main():
    logger.info("=" * 60)
    logger.info("Phase 1.1: Multi-Source Dataset Merger")
    logger.info("=" * 60)
    
    # Load all datasets
    datasets = load_and_normalize_datasets()
    
    if not datasets:
        logger.error("No datasets found to merge!")
        return
    
    # Merge all
    merged_df = pd.concat(datasets, ignore_index=True)
    
    # Log statistics
    logger.info(f"\nTotal rows before deduplication: {len(merged_df)}")
    logger.info(f"Unique rows: {merged_df.drop_duplicates().shape[0]}")
    logger.info(f"Unique intents: {merged_df['intent'].nunique()}")
    logger.info(f"\nIntent distribution:")
    logger.info(f"{merged_df['intent'].value_counts()}")
    
    # Save merged dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"input-data/raw/intent_merged_{timestamp}.csv"
    merged_df.to_csv(output_path, index=False)
    
    logger.info(f"\nMerged dataset saved to: {output_path}")
    logger.info(f"Total samples: {len(merged_df)}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
