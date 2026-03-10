#!/usr/bin/env python3
"""
Phase 2.2: Text Embedding Generation
Generates TF-IDF embeddings for train/val/test splits
"""

import pandas as pd
import numpy as np
import yaml
import json
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime

def load_config():
    """Load configuration from config.yaml"""
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def main():
    print("=" * 60)
    print("PHASE 2.2: TEXT EMBEDDING GENERATION")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    embedding_config = config['embedding']
    
    # Set defaults for missing keys (based on project requirements)
    max_features = 5000  # Default from project spec
    ngram_range = (1, 2)  # Unigrams + bigrams
    
    print(f"\nConfiguration:")
    print(f"  Model: {embedding_config['model_name']}")
    print(f"  Backend: {embedding_config['backend']}")
    print(f"  Max features: {max_features}")
    print(f"  N-gram range: {ngram_range}")
    print(f"  Normalize: {embedding_config['normalize']}")
    
    # Create output directories
    os.makedirs('data/embeddings', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Initialize TF-IDF Vectorizer
    print("\nInitializing TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        norm='l2' if embedding_config['normalize'] else None,
        dtype=np.float32  # Memory optimization
    )
    
    # Load cleaned datasets
    print("\nLoading cleaned datasets...")
    train_df = pd.read_csv('data/processed/cleaned_split_train.csv')
    val_df = pd.read_csv('data/processed/cleaned_split_val.csv')
    test_df = pd.read_csv('data/processed/cleaned_split_test.csv')
    
    print(f"  Train samples: {len(train_df)}")
    print(f"  Val samples: {len(val_df)}")
    print(f"  Test samples: {len(test_df)}")
    
    # Extract text column (use user_input_clean from Phase 2.1)
    print("\nExtracting text features...")
    train_texts = train_df['user_input_clean'].fillna('').astype(str).tolist()
    val_texts = val_df['user_input_clean'].fillna('').astype(str).tolist()
    test_texts = test_df['user_input_clean'].fillna('').astype(str).tolist()
    
    # Fit on training data and transform all sets
    print("\nFitting vectorizer on training data...")
    train_embeddings = vectorizer.fit_transform(train_texts)
    
    print("Transforming validation data...")
    val_embeddings = vectorizer.transform(val_texts)
    
    print("Transforming test data...")
    test_embeddings = vectorizer.transform(test_texts)
    
    # Convert sparse matrices to dense float32 arrays
    print("\nConverting to dense arrays (float32)...")
    train_emb_dense = train_embeddings.toarray().astype(np.float32)
    val_emb_dense = val_embeddings.toarray().astype(np.float32)
    test_emb_dense = test_embeddings.toarray().astype(np.float32)
    
    # Save embeddings
    print("\nSaving embeddings...")
    np.save('data/embeddings/train_emb.npy', train_emb_dense)
    np.save('data/embeddings/val_emb.npy', val_emb_dense)
    np.save('data/embeddings/test_emb.npy', test_emb_dense)
    
    # Save vectorizer
    print("Saving vectorizer...")
    with open('data/embeddings/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # Create and save index mappings (text -> label)
    print("Creating index mappings...")
    label_mapping = {label: idx for idx, label in enumerate(vectorizer.get_feature_names_out())}
    with open('data/embeddings/all_index_maps.pkl', 'wb') as f:
        pickle.dump({
            'feature_names': vectorizer.get_feature_names_out().tolist(),
            'label_to_idx': label_mapping,
            'vocab_size': len(vectorizer.vocabulary_)
        }, f)
    
    # Calculate and save metrics
    print("\nCalculating embedding metrics...")
    metrics = {
        'phase': '2.2',
        'timestamp': datetime.now().isoformat(),
        'embedding_config': {
            'model_name': embedding_config['model_name'],
            'backend': embedding_config['backend'],
            'max_features': max_features,
            'ngram_range': list(ngram_range),
            'normalize': embedding_config['normalize']
        },
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'vocab_size': len(vectorizer.vocabulary_),
        'embedding_dim': train_emb_dense.shape[1],
        'train_emb_shape': list(train_emb_dense.shape),
        'val_emb_shape': list(val_emb_dense.shape),
        'test_emb_shape': list(test_emb_dense.shape),
        'train_emb_size_mb': round(train_emb_dense.nbytes / (1024 * 1024), 2),
        'val_emb_size_mb': round(val_emb_dense.nbytes / (1024 * 1024), 2),
        'test_emb_size_mb': round(test_emb_dense.nbytes / (1024 * 1024), 2),
        'total_size_mb': round((train_emb_dense.nbytes + val_emb_dense.nbytes + test_emb_dense.nbytes) / (1024 * 1024), 2),
        'dtype': 'float32',
        'files_created': [
            'data/embeddings/train_emb.npy',
            'data/embeddings/val_emb.npy',
            'data/embeddings/test_emb.npy',
            'data/embeddings/vectorizer.pkl',
            'data/embeddings/all_index_maps.pkl'
        ]
    }
    
    with open('logs/p2.2_summary.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\n" + "=" * 60)
    print("PHASE 2.2 COMPLETE")
    print("=" * 60)
    print(f"\nEmbedding Dimensions: {train_emb_dense.shape[1]}")
    print(f"Vocabulary Size: {len(vectorizer.vocabulary_)}")
    print(f"Total Size: {metrics['total_size_mb']} MB")
    print(f"\nFiles created:")
    for f in metrics['files_created']:
        print(f"  ✓ {f}")
    print(f"\nSummary saved to: logs/p2.2_summary.json")
    print("\nReady for Phase 2.3: Embedding Validation")

if __name__ == '__main__':
    main()
