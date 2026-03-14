#!/usr/bin/env python3
"""
Phase 4.6: Edge Case Resolution & Retraining
"""

import os
import sys
import json
import pickle
import logging
import argparse
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report
)

# Try to import imbalanced-learn for SMOTE
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("WARNING: imbalanced-learn not installed.")

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("WARNING: XGBoost not installed.")

def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "p4.6-edge-case-resolution.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()


class ClassMapper:
    """Map non-contiguous class labels to contiguous range."""
    
    def __init__(self, unique_classes: np.ndarray):
        self.original_classes = unique_classes
        self.n_classes = len(unique_classes)
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(unique_classes))}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
    
    def map_to_contiguous(self, y: np.ndarray) -> np.ndarray:
        """Map original labels to contiguous range."""
        return np.array([self.label_to_idx[label] for label in y])
    
    def map_to_original(self, y_contiguous: np.ndarray) -> np.ndarray:
        """Map contiguous labels back to original labels."""
        return np.array([self.idx_to_label[label] for label in y_contiguous])


def load_embeddings(embeddings_path: str, split: str) -> np.ndarray:
    """Load TF-IDF embeddings for a specific split."""
    emb_file = Path(embeddings_path) / f"{split}_emb.npy"
    if emb_file.exists():
        return np.load(emb_file)
    return None


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Phase 4.6: Edge Case Resolution')
    parser.add_argument('--use-smote', action='store_true', help='Apply SMOTE')
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU for XGBoost')
    parser.add_argument('--use-tfidf', action='store_true', help='Use TF-IDF embeddings instead of engineered features')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("PHASE 4.6: EDGE CASE RESOLUTION & RETRAINING")
    logger.info("="*80)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data
    logger.info("\nLoading preprocessed data...")
    data_cfg = config.get('data', {})
    
    train_df = pd.read_csv(data_cfg.get('train_path', 'data/processed/cleaned_split_train.csv'))
    val_df = pd.read_csv(data_cfg.get('val_path', 'data/processed/cleaned_split_val.csv'))
    test_df = pd.read_csv(data_cfg.get('test_path', 'data/processed/cleaned_split_test.csv'))
    
    logger.info(f"  Train: {len(train_df)} samples")
    logger.info(f"  Validation: {len(val_df)} samples")
    logger.info(f"  Test: {len(test_df)} samples")
    
    # Load TF-IDF embeddings if requested
    if args.use_tfidf:
        logger.info("\nUsing TF-IDF embeddings...")
        embeddings_path = data_cfg.get('embeddings_path', 'data/final/embeddings_v2.0')
        
        X_train = load_embeddings(embeddings_path, 'train')
        X_val = load_embeddings(embeddings_path, 'val')
        X_test = load_embeddings(embeddings_path, 'test')
        
        if X_train is None:
            logger.error("TF-IDF embeddings not found. Falling back to engineered features.")
            args.use_tfidf = False
    
    if not args.use_tfidf:
        logger.info("\nUsing engineered features...")
        # Define columns
        label_column = 'intent_encoded'
        exclude_cols = ['user_input', 'intent', 'source', 'user_input_clean', 'cleaned_text', label_column]
        feature_cols = [col for col in train_df.columns if col not in exclude_cols]
        
        logger.info(f"Using {len(feature_cols)} feature columns")
        
        # Extract features and labels
        X_train = train_df[feature_cols].values
        X_val = val_df[feature_cols].values
        X_test = test_df[feature_cols].values
    
    y_train_original = train_df['intent_encoded'].values
    y_val_original = val_df['intent_encoded'].values
    y_test_original = test_df['intent_encoded'].values
    
    logger.info(f"Feature matrix shape: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")
    
    # Create class mapper
    unique_classes = np.unique(y_train_original)
    class_mapper = ClassMapper(unique_classes)
    
    logger.info(f"\nOriginal classes: {len(unique_classes)} unique labels")
    logger.info(f"Class range: {unique_classes.min()} to {unique_classes.max()}")
    
    # Map to contiguous range
    y_train = class_mapper.map_to_contiguous(y_train_original)
    y_val = class_mapper.map_to_contiguous(y_val_original)
    y_test = class_mapper.map_to_contiguous(y_test_original)
    
    # Analyze class distribution
    logger.info("\n" + "="*80)
    logger.info("STEP 1: CLASS DISTRIBUTION ANALYSIS")
    logger.info("="*80)
    
    unique, counts = np.unique(y_train, return_counts=True)
    logger.info(f"Total samples: {len(y_train)}")
    logger.info(f"Number of classes: {len(unique)}")
    logger.info(f"Imbalance ratio: {counts.max() / counts.min():.2f}")
    logger.info(f"Min class count: {counts.min()}")
    logger.info(f"Max class count: {counts.max()}")
    
    os.makedirs("results/phase4", exist_ok=True)
    
    # Calculate class weights
    logger.info("\n" + "="*80)
    logger.info("STEP 2: CLASS WEIGHT CALCULATION")
    logger.info("="*80)
    
    weights = compute_class_weight(class_weight='balanced', classes=unique, y=y_train)
    class_weights = dict(zip(unique.tolist(), weights.tolist()))
    sample_weights = np.array([class_weights[y_i] for y_i in y_train])
    
    logger.info(f"Sample weights: min={sample_weights.min():.4f}, max={sample_weights.max():.4f}")
    
    # Apply SMOTE if requested
    if args.use_smote and SMOTE_AVAILABLE:
        logger.info("\n" + "="*80)
        logger.info("STEP 3: SMOTE OVERSAMPLING")
        logger.info("="*80)
        
        try:
            min_count = counts.min()
            k_neighbors = min(5, min_count - 1) if min_count > 1 else 1
            
            smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            
            logger.info(f"Before SMOTE: {X_train.shape}, After SMOTE: {X_train_balanced.shape}")
            logger.info(f"Oversampling ratio: {len(y_train_balanced) / len(y_train):.2f}x")
            
            sample_weights_balanced = None
        except Exception as e:
            logger.warning(f"SMOTE failed: {e}. Using original data.")
            X_train_balanced, y_train_balanced = X_train, y_train
            sample_weights_balanced = sample_weights
    else:
        X_train_balanced, y_train_balanced = X_train, y_train
        sample_weights_balanced = sample_weights
    
    # Train XGBoost
    logger.info("\n" + "="*80)
    logger.info("STEP 4: TRAINING XGBOOST")
    logger.info("="*80)
    
    if not XGBOOST_AVAILABLE:
        logger.error("XGBoost not available. Cannot proceed.")
        return 1
    
    n_classes = len(np.unique(y_train_balanced))
    logger.info(f"Number of classes: {n_classes}")
    logger.info(f"Training samples: {len(y_train_balanced)}")
    logger.info(f"Features: {X_train_balanced.shape[1]}")
    
    params = {
        'objective': 'multi:softprob',
        'num_class': n_classes,
        'eval_metric': 'mlogloss',
        'max_depth': 8,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'tree_method': 'gpu_hist' if args.use_gpu else 'hist',
        'scale_pos_weight': 1.0  # Will be overridden by sample_weights
    }
    
    model = xgb.XGBClassifier(**params)
    
    eval_set = [(X_train_balanced, y_train_balanced), (X_val, y_val)]
    if sample_weights_balanced is not None:
        model.fit(X_train_balanced, y_train_balanced, sample_weight=sample_weights_balanced,
                 eval_set=eval_set, verbose=False)
    else:
        model.fit(X_train_balanced, y_train_balanced, eval_set=eval_set, verbose=False)
    
    # Validation metrics
    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_f1_macro = f1_score(y_val, y_val_pred, average='macro', zero_division=0)
    
    logger.info(f"\nXGBoost Validation Metrics:")
    logger.info(f"  Accuracy: {val_accuracy:.4f}")
    logger.info(f"  F1-Score (macro): {val_f1_macro:.4f}")
    
    # Test set evaluation
    logger.info("\n" + "="*80)
    logger.info("STEP 5: TEST SET EVALUATION")
    logger.info("="*80)
    
    y_test_pred_contiguous = model.predict(X_test)
    y_test_pred = class_mapper.map_to_original(y_test_pred_contiguous)
    y_test_orig = class_mapper.map_to_original(y_test)
    
    test_metrics = {
        'accuracy': accuracy_score(y_test_orig, y_test_pred),
        'f1_macro': f1_score(y_test_orig, y_test_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_test_orig, y_test_pred, average='weighted', zero_division=0),
        'precision_macro': precision_score(y_test_orig, y_test_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_test_orig, y_test_pred, average='macro', zero_division=0)
    }
    
    logger.info(f"\nXGBoost Test Set Performance:")
    logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"  F1-Score (macro): {test_metrics['f1_macro']:.4f}")
    logger.info(f"  F1-Score (weighted): {test_metrics['f1_weighted']:.4f}")
    
    # Compare with SGD baseline
    evaluation_results = {
        'xgboost': {'metrics': test_metrics, 'predictions': y_test_pred}
    }
    
    try:
        sgd_df = pd.read_csv("results/phase4/training_results_v1.0.1.csv")
        sgd_row = sgd_df[sgd_df['model'] == 'SGD']
        if len(sgd_row) > 0:
            evaluation_results['sgd_baseline'] = {
                'metrics': {
                    'accuracy': float(sgd_row['test_accuracy'].values[0]),
                    'f1_macro': float(sgd_row['test_f1_macro'].values[0]),
                    'f1_weighted': float(sgd_row['test_f1_weighted'].values[0]),
                    'precision_macro': float(sgd_row['test_f1_macro'].values[0]),
                    'recall_macro': float(sgd_row['test_f1_macro'].values[0])
                },
                'predictions': None
            }
            logger.info("\nSGD Baseline loaded for comparison")
    except Exception as e:
        logger.warning(f"Could not load SGD baseline: {e}")
    
    # Save model
    logger.info("\n" + "="*80)
    logger.info("STEP 6: MODEL SERIALIZATION")
    logger.info("="*80)
    
    models_dir = Path("models/phase4")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    xgb_path = models_dir / "xgboost_v1.0.1_balanced.pkl"
    with open(xgb_path, 'wb') as f:
        pickle.dump({'model': model, 'class_mapper': class_mapper, 'use_tfidf': args.use_tfidf}, f)
    logger.info(f"XGBoost saved: {xgb_path}")
    
    # Save results CSV
    results_df = pd.DataFrame([
        {
            'model': model_name,
            'accuracy': result['metrics']['accuracy'],
            'f1_macro': result['metrics']['f1_macro'],
            'f1_weighted': result['metrics']['f1_weighted'],
            'precision_macro': result['metrics']['precision_macro'],
            'recall_macro': result['metrics']['recall_macro']
        }
        for model_name, result in evaluation_results.items()
    ])
    
    results_path = "results/phase4/edge_case_results_v1.0.1.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"Results saved: {results_path}")
    
    # Print final leaderboard
    logger.info("\n" + "="*80)
    logger.info("FINAL LEADERBOARD (Test Set)")
    logger.info("="*80)
    logger.info(results_df.to_string(index=False))
    
    logger.info("\n" + "="*80)
    logger.info("PHASE 4.6 COMPLETE")
    logger.info("="*80)
    logger.info("Note: LightGBM skipped due to sklearn version incompatibility")
    logger.info("Ready for Phase 4.7: Final Validation")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
