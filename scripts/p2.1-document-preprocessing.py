#!/usr/bin/env python3
"""
p2.1-document-preprocessing.py
==============================
Phase 2.1: Document Preprocessing for Multi-Model Orchestration

Cleans, normalizes, and enriches text data from Phase 1 splits.
Implements chunked processing for memory efficiency (8GB RAM constraint).

Author: mavyjimz
Created: 2026-03-10
Version: 2.1.0
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Generator, Optional, Tuple

import pandas as pd
import yaml
from tqdm import tqdm

# Optional: spaCy for lemmatization (install if needed)
try:
    import spacy
    NLP = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    NLP = None


class TextPreprocessor:
    """Memory-efficient text preprocessing with chunked processing."""
    
    def __init__(self, config: Dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.prep_config = config.get('preprocessing', {})
        
        # Preprocessing settings
        self.max_length = self.prep_config.get('max_length', 128)
        self.remove_urls = self.prep_config.get('remove_urls', True)
        self.remove_emails = self.prep_config.get('remove_emails', True)
        self.lemmatize = self.prep_config.get('lemmatize', True) and SPACY_AVAILABLE
        self.chunk_size = self.prep_config.get('chunk_size', 500)
        
        # Statistics tracking
        self.stats = {
            'total_rows': 0,
            'processed_rows': 0,
            'dropped_rows': 0,
            'drop_reasons': {
                'empty_text': 0,
                'too_short': 0,
                'error': 0
            }
        }
        
        if self.lemmatize and not SPACY_AVAILABLE:
            logger.warning("spaCy not available - lemmatization disabled")
        
        logger.info(f"Preprocessor initialized: max_length={self.max_length}, "
                   f"lemmatize={self.lemmatize}, chunk_size={self.chunk_size}")
    
    def clean_text(self, text: str) -> str:
        """Apply cleaning pipeline to a single text sample."""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower().strip()
        
        # Remove URLs
        if self.remove_urls:
            text = re.sub(r'https?://\S+|www\.\S+', '', text)
            text = re.sub(r'\S+\.(com|org|net|edu)\S*', '', text)
        
        # Remove email addresses
        if self.remove_emails:
            text = re.sub(r'\S+@\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove special characters but keep punctuation for NLP
        text = re.sub(r'[^a-zA-Z0-9\s\.\?\!\,]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Lemmatization (if enabled)
        if self.lemmatize and NLP is not None and len(text) > 0:
            try:
                doc = NLP(text)
                text = ' '.join([token.lemma_ for token in doc])
            except Exception as e:
                self.logger.debug(f"Lemmatization skipped: {e}")
        
        return text
    
    def enrich_features(self, row: pd.Series) -> pd.Series:
        """Add derived features to a row."""
        cleaned = row.get('cleaned_text', '')
        
        # Token-based features
        tokens = cleaned.split() if cleaned else []
        row['token_count'] = len(tokens)
        row['avg_token_length'] = (
            sum(len(t) for t in tokens) / len(tokens) 
            if tokens else 0.0
        )
        
        # Punctuation features
        row['has_question'] = 1 if '?' in cleaned else 0
        row['has_exclamation'] = 1 if '!' in cleaned else 0
        row['question_word_count'] = sum(
            1 for t in tokens[:5] 
            if t in ['what', 'where', 'when', 'why', 'how', 'who', 'which']
        )
        
        # Length validation
        row['is_valid_length'] = (
            1 if 0 < len(tokens) <= self.max_length else 0
        )
        
        return row
    
    def validate_row(self, row: pd.Series) -> Tuple[bool, Optional[str]]:
        """
        Validate a preprocessed row.
        Returns: (is_valid, drop_reason)
        """
        cleaned = row.get('cleaned_text', '')
        
        if not cleaned or len(cleaned.strip()) == 0:
            return False, 'empty_text'
        
        token_count = row.get('token_count', 0)
        if token_count == 0:
            return False, 'too_short'
        
        return True, None
    
    def process_chunk(self, chunk: pd.DataFrame, chunk_idx: int) -> pd.DataFrame:
        """Process a single chunk of data."""
        self.logger.debug(f"Processing chunk {chunk_idx} ({len(chunk)} rows)")
        
        processed_rows = []
        
        for idx, row in chunk.iterrows():
            self.stats['total_rows'] += 1
            
            try:
                # Extract original text
                original_text = row.get('user_input', '')
                
                # Clean text
                cleaned = self.clean_text(original_text)
                row['cleaned_text'] = cleaned
                
                # Enrich with features
                row = self.enrich_features(row)
                
                # Validate
                is_valid, drop_reason = self.validate_row(row)
                
                if is_valid:
                    processed_rows.append(row)
                    self.stats['processed_rows'] += 1
                else:
                    self.stats['dropped_rows'] += 1
                    self.stats['drop_reasons'][drop_reason] += 1
                    
            except Exception as e:
                self.stats['dropped_rows'] += 1
                self.stats['drop_reasons']['error'] += 1
                self.logger.debug(f"Row {idx} error: {e}")
        
        return pd.DataFrame(processed_rows) if processed_rows else pd.DataFrame()
    
    def process_file(self, input_path: Path, output_path: Path) -> Dict:
        """Process a single CSV file with chunked reading."""
        self.logger.info(f"Processing: {input_path} → {output_path}")
        
        chunk_results = []
        total_chunks = 0
        
        # Calculate total chunks for progress bar
        total_rows = sum(1 for _ in open(input_path, 'r', encoding='utf-8')) - 1
        total_chunks = (total_rows // self.chunk_size) + 1
        
        # Process in chunks
        with tqdm(
            total=total_chunks, 
            desc=f"Preprocessing {input_path.name}",
            unit="chunk"
        ) as pbar:
            for chunk_idx, chunk in enumerate(
                pd.read_csv(input_path, chunksize=self.chunk_size)
            ):
                processed_chunk = self.process_chunk(chunk, chunk_idx)
                
                if not processed_chunk.empty:
                    chunk_results.append(processed_chunk)
                
                pbar.update(1)
        
        # Combine all chunks
        if chunk_results:
            final_df = pd.concat(chunk_results, ignore_index=True)
        else:
            final_df = pd.DataFrame()
            self.logger.warning(f"No valid rows in {input_path}")
        
        # Save to disk
        final_df.to_csv(output_path, index=False, encoding='utf-8')
        self.logger.info(f"Saved {len(final_df)} rows to {output_path}")
        
        # Generate checksum
        checksum = self._generate_checksum(output_path)
        
        return {
            'input_file': str(input_path),
            'output_file': str(output_path),
            'input_rows': total_rows,
            'output_rows': len(final_df),
            'dropped_rows': self.stats['dropped_rows'],
            'checksum': checksum
        }
    
    def _generate_checksum(self, file_path: Path) -> str:
        """Generate SHA256 checksum for a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()


def setup_logging(log_path: Path) -> logging.Logger:
    """Configure structured logging."""
    logger = logging.getLogger('p2.1_preprocessing')
    logger.setLevel(logging.DEBUG)
    
    # File handler
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def load_config(config_path: Path) -> Dict:
    """Load configuration from YAML."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    """Main execution pipeline."""
    parser = argparse.ArgumentParser(
        description='Phase 2.1: Document Preprocessing'
    )
    parser.add_argument(
        '--config', 
        type=Path, 
        default=Path('config/config.yaml'),
        help='Path to configuration file'
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path('data/processed'),
        help='Input directory containing split CSV files'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/processed'),
        help='Output directory for cleaned CSV files'
    )
    parser.add_argument(
        '--log-dir',
        type=Path,
        default=Path('logs'),
        help='Directory for log files'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = args.log_dir / 'p2.1_preprocessing.log'
    logger = setup_logging(log_file)
    
    # Start execution
    start_time = datetime.now()
    logger.info("=" * 70)
    logger.info("PHASE 2.1: DOCUMENT PREPROCESSING STARTED")
    logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Loaded config from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(config, logger)
    
    # Process all splits
    split_files = ['split_train.csv', 'split_val.csv', 'split_test.csv']
    results = []
    all_stats = {
        'train': {'total': 0, 'processed': 0, 'dropped': 0},
        'val': {'total': 0, 'processed': 0, 'dropped': 0},
        'test': {'total': 0, 'processed': 0, 'dropped': 0}
    }
    
    for split_file in split_files:
        input_path = args.input_dir / split_file
        output_path = args.output_dir / f"cleaned_{split_file}"
        
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            continue
        
        # Reset stats for this file
        preprocessor.stats = {
            'total_rows': 0,
            'processed_rows': 0,
            'dropped_rows': 0,
            'drop_reasons': {
                'empty_text': 0,
                'too_short': 0,
                'error': 0
            }
        }
        
        # Process file
        result = preprocessor.process_file(input_path, output_path)
        results.append(result)
        
        # Track stats
        split_name = split_file.replace('split_', '').replace('.csv', '')
        all_stats[split_name] = {
            'total': result['input_rows'],
            'processed': result['output_rows'],
            'dropped': result['dropped_rows']
        }
    
    # Generate summary report
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    summary = {
        'phase': '2.1',
        'name': 'Document Preprocessing',
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'duration_seconds': duration,
        'config': {
            'max_length': preprocessor.max_length,
            'remove_urls': preprocessor.remove_urls,
            'remove_emails': preprocessor.remove_emails,
            'lemmatize': preprocessor.lemmatize,
            'chunk_size': preprocessor.chunk_size
        },
        'splits': all_stats,
        'checksums': {r['output_file']: r['checksum'] for r in results},
        'drop_reasons': preprocessor.stats['drop_reasons']
    }
    
    # Save summary
    summary_path = args.log_dir / 'p2.1_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    # Log summary
    logger.info("=" * 70)
    logger.info("PREPROCESSING SUMMARY")
    logger.info("=" * 70)
    for split_name, stats in all_stats.items():
        logger.info(
            f"{split_name.upper():8} | "
            f"Input: {stats['total']:5} | "
            f"Output: {stats['processed']:5} | "
            f"Dropped: {stats['dropped']:5}"
        )
    
    logger.info(f"\nTotal duration: {duration:.2f}s")
    logger.info(f"Drop reasons: {preprocessor.stats['drop_reasons']}")
    logger.info(f"Summary saved to: {summary_path}")
    logger.info("=" * 70)
    logger.info("PHASE 2.1 COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
