#!/usr/bin/env python3
"""
Phase 4.2: Production Training Pipeline
========================================
Trains selected models (SGD, Random Forest) with:
- MLflow experiment tracking
- Semantic versioning for model artifacts
- Comprehensive validation metrics
- Model manifest generation
- Checkpointing and serialization

Author: MLOps Team
Date: March 2026
Version: 1.0.0
"""

import os
import sys
import json
import pickle
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import warnings

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Third-party imports
import numpy as np
import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.calibration import CalibratedClassifierCV
import joblib

# Suppress non-critical warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/p4.2-training-pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)


class ModelTrainer:
    """Production-grade model training pipeline with MLflow integration."""
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """Initialize trainer with configuration."""
        self.config = self._load_config(config_path)
        self.project_root = Path(__file__).parent.parent
        self.model_version = self._get_next_model_version()
        self.trained_models: Dict[str, Any] = {}
        self.metrics: Dict[str, Dict[str, float]] = {}
        
        logger.info(f"Initialized ModelTrainer with config: {config_path}")
        logger.info(f"Model version: {self.model_version}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded successfully from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            raise
    
    def _get_next_model_version(self) -> str:
        """Generate semantic version for model artifacts."""
        # Check existing models to determine next version
        models_dir = self.project_root / 'models' / 'phase4'
        models_dir.mkdir(parents=True, exist_ok=True)
        
        existing_versions = []
        for model_file in models_dir.glob('*.pkl'):
            # Extract version from filename (e.g., sgd_v1.0.0.pkl)
            try:
                version_str = model_file.stem.split('_v')[-1]
                major, minor, patch = map(int, version_str.split('.'))
                existing_versions.append((major, minor, patch))
            except (ValueError, IndexError):
                continue
        
        if existing_versions:
            # Increment patch version
            latest = max(existing_versions)
            next_version = (latest[0], latest[1], latest[2] + 1)
        else:
            next_version = (1, 0, 0)
        
        return f"{next_version[0]}.{next_version[1]}.{next_version[2]}"
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                  np.ndarray, np.ndarray, np.ndarray]:
        """Load training, validation, and test data."""
        logger.info("Loading datasets...")
        
        try:
            # Load embeddings
            embeddings_path = self.project_root / self.config['data']['embeddings_path']
            
            train_emb = np.load(embeddings_path / 'train_emb.npy')
            val_emb = np.load(embeddings_path / 'val_emb.npy')
            test_emb = np.load(embeddings_path / 'test_emb.npy')
            
            # Load labels from CSV files
            train_df = pd.read_csv(self.project_root / self.config['data']['train_path'])
            val_df = pd.read_csv(self.project_root / self.config['data']['val_path'])
            test_df = pd.read_csv(self.project_root / self.config['data']['test_path'])
            
            train_labels = train_df['intent_encoded'].values
            val_labels = val_df['intent_encoded'].values
            test_labels = test_df['intent_encoded'].values
            
            logger.info(f"Train set: {train_emb.shape[0]} samples, {train_emb.shape[1]} features")
            logger.info(f"Val set: {val_emb.shape[0]} samples")
            logger.info(f"Test set: {test_emb.shape[0]} samples")
            logger.info(f"Number of classes: {len(np.unique(train_labels))}")
            
            return train_emb, val_emb, test_emb, train_labels, val_labels, test_labels
            
        except Exception as e:
            logger.error(f"Failed to load  {e}")
            raise
    
    def train_model(self, model_name: str, model: Any,
                    X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray,
                    X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Train a single model with comprehensive metrics."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Training Model: {model_name}")
        logger.info(f"{'='*60}")
        
        start_time = datetime.now()
        
        # Configure MLflow
        mlflow.set_experiment(self.config['phase4']['mlflow_experiment'])
        
        with mlflow.start_run(run_name=f"{model_name}_v{self.model_version}"):
            try:
                # Log hyperparameters
                if hasattr(model, 'get_params'):
                    params = model.get_params()
                    mlflow.log_params(params)
                
                # Train model
                logger.info(f"Training {model_name}...")
                model.fit(X_train, y_train)
                
                train_time = (datetime.now() - start_time).total_seconds()
                logger.info(f"Training completed in {train_time:.2f}s")
                
                # Generate predictions
                y_train_pred = model.predict(X_train)
                y_val_pred = model.predict(X_val)
                y_test_pred = model.predict(X_test)
                
                # Calculate metrics
                metrics = self._calculate_metrics(
                    y_train, y_train_pred,
                    y_val, y_val_pred,
                    y_test, y_test_pred,
                    model, X_test
                )
                
                # Log metrics to MLflow
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                
                # Log training time
                mlflow.log_metric("training_time_seconds", train_time)
                
                # Log model size (estimate)
                model_path = f"models/phase4/{model_name}_v{self.model_version}.pkl"
                model_size_mb = sys.getsizeof(pickle.dumps(model)) / (1024 * 1024)
                mlflow.log_metric("model_size_mb", model_size_mb)
                
                # Log artifacts
                mlflow.log_text(
                    classification_report(y_test, y_test_pred, zero_division=0),
                    f"{model_name}_classification_report.txt"
                )
                
                # Log confusion matrix as artifact
                cm = confusion_matrix(y_test, y_test_pred)
                np.save(f"/tmp/{model_name}_confusion_matrix.npy", cm)
                mlflow.log_artifact(f"/tmp/{model_name}_confusion_matrix.npy")
                
                logger.info(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
                logger.info(f"Test F1-Score: {metrics['test_f1_macro']:.4f}")
                logger.info(f"Training Time: {train_time:.2f}s")
                
                # Store trained model
                self.trained_models[model_name] = {
                    'model': model,
                    'metrics': metrics,
                    'train_time': train_time,
                    'version': self.model_version
                }
                
                return metrics
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                mlflow.log_param("training_failed", True)
                raise
    
    def _calculate_metrics(self, y_train: np.ndarray, y_train_pred: np.ndarray,
                          y_val: np.ndarray, y_val_pred: np.ndarray,
                          y_test: np.ndarray, y_test_pred: np.ndarray,
                          model: Any, X_test: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive metrics for all splits."""
        metrics = {}
        
        # Training metrics
        metrics['train_accuracy'] = float(accuracy_score(y_train, y_train_pred))
        metrics['train_f1_macro'] = float(f1_score(y_train, y_train_pred, average='macro'))
        
        # Validation metrics
        metrics['val_accuracy'] = float(accuracy_score(y_val, y_val_pred))
        metrics['val_f1_macro'] = float(f1_score(y_val, y_val_pred, average='macro'))
        
        # Test metrics
        metrics['test_accuracy'] = float(accuracy_score(y_test, y_test_pred))
        metrics['test_f1_macro'] = float(f1_score(y_test, y_test_pred, average='macro'))
        metrics['test_f1_weighted'] = float(f1_score(y_test, y_test_pred, average='weighted'))
        metrics['test_precision_macro'] = float(precision_score(y_test, y_test_pred, average='macro', zero_division=0))
        metrics['test_recall_macro'] = float(recall_score(y_test, y_test_pred, average='macro', zero_division=0))
        
        # ROC-AUC (only if probability predictions available)
        try:
            if hasattr(model, 'predict_proba'):
                y_test_proba = model.predict_proba(X_test)
                metrics['test_roc_auc_ovr'] = float(roc_auc_score(
                    y_test, y_test_proba, multi_class='ovr', average='macro'
                ))
        except Exception as e:
            logger.warning(f"Could not calculate ROC-AUC: {e}")
            metrics['test_roc_auc_ovr'] = 0.0
        
        return metrics
    
    def save_models(self) -> Dict[str, str]:
        """Save trained models with semantic versioning."""
        logger.info(f"\n{'='*60}")
        logger.info("Saving Trained Models")
        logger.info(f"{'='*60}")
        
        models_dir = self.project_root / 'models' / 'phase4'
        models_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = {}
        
        for model_name, model_data in self.trained_models.items():
            model = model_data['model']
            version = model_data['version']
            
            # Save model with version
            model_filename = f"{model_name}_v{version}.pkl"
            model_path = models_dir / model_filename
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Calculate checksum
            with open(model_path, 'rb') as f:
                checksum = hashlib.md5(f.read()).hexdigest()
            
            # Get file size
            file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
            
            saved_paths[model_name] = {
                'path': str(model_path),
                'filename': model_filename,
                'version': version,
                'checksum': checksum,
                'size_mb': round(file_size_mb, 2),
                'metrics': model_data['metrics']
            }
            
            logger.info(f"Saved: {model_filename} ({file_size_mb:.2f} MB)")
            logger.info(f"  Checksum: {checksum}")
            logger.info(f"  Test Accuracy: {model_data['metrics']['test_accuracy']:.4f}")
        
        return saved_paths
    
    def generate_model_manifest(self, saved_paths: Dict[str, str]) -> None:
        """Generate comprehensive model manifest."""
        logger.info(f"\n{'='*60}")
        logger.info("Generating Model Manifest")
        logger.info(f"{'='*60}")
        
        manifest = {
            'project': 'Multi-Model Orchestration System',
            'phase': '4.2 - Training Pipeline',
            'version': self.model_version,
            'generated_at': datetime.now().isoformat(),
            'config': {
                'target_accuracy': float(self.config['phase4']['target_accuracy']),
                'random_state': int(self.config['phase4']['random_state']),
                'cv_folds': int(self.config['phase4']['cross_validation_folds'])
            },
            'models': {},
            'summary': {
                'total_models_trained': len(self.trained_models),
                'models_exceeding_target': 0,
                'best_model': None,
                'best_accuracy': 0.0
            }
        }
        
        for model_name, model_info in saved_paths.items():
            meets_target = bool(model_info['metrics']['test_accuracy'] >= self.config['phase4']['target_accuracy'])
            
            manifest['models'][model_name] = {
                'filename': model_info['filename'],
                'version': model_info['version'],
                'path': model_info['path'],
                'checksum_md5': model_info['checksum'],
                'size_mb': float(model_info['size_mb']),
                'metrics': {k: float(v) for k, v in model_info['metrics'].items()},
                'meets_target': meets_target
            }
            
            # Update summary
            if meets_target:
                manifest['summary']['models_exceeding_target'] += 1
            
            if model_info['metrics']['test_accuracy'] > manifest['summary']['best_accuracy']:
                manifest['summary']['best_accuracy'] = float(model_info['metrics']['test_accuracy'])
                manifest['summary']['best_model'] = model_name
        
        # Save manifest
        manifest_path = self.project_root / 'models' / 'phase4' / f'model_manifest_v{self.model_version}.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2, cls=NumpyEncoder)
        
        logger.info(f"Manifest saved to: {manifest_path}")
        logger.info(f"Total models trained: {manifest['summary']['total_models_trained']}")
        logger.info(f"Models exceeding target ({self.config['phase4']['target_accuracy']*100}%): {manifest['summary']['models_exceeding_target']}")
        logger.info(f"Best model: {manifest['summary']['best_model']} ({manifest['summary']['best_accuracy']*100:.2f}%)")
    
    def save_results_summary(self) -> None:
        """Save training results summary to CSV."""
        logger.info(f"\n{'='*60}")
        logger.info("Saving Results Summary")
        logger.info(f"{'='*60}")
        
        results_dir = self.project_root / 'results' / 'phase4'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_data = []
        for model_name, model_data in self.trained_models.items():
            metrics = model_data['metrics']
            results_data.append({
                'model_name': model_name,
                'version': model_data['version'],
                'train_accuracy': metrics['train_accuracy'],
                'train_f1_macro': metrics['train_f1_macro'],
                'val_accuracy': metrics['val_accuracy'],
                'val_f1_macro': metrics['val_f1_macro'],
                'test_accuracy': metrics['test_accuracy'],
                'test_f1_macro': metrics['test_f1_macro'],
                'test_f1_weighted': metrics['test_f1_weighted'],
                'test_precision_macro': metrics['test_precision_macro'],
                'test_recall_macro': metrics['test_recall_macro'],
                'test_roc_auc_ovr': metrics.get('test_roc_auc_ovr', 0.0),
                'training_time_seconds': model_data['train_time'],
                'meets_target': metrics['test_accuracy'] >= self.config['phase4']['target_accuracy']
            })
        
        # Create DataFrame and save
        df_results = pd.DataFrame(results_data)
        results_path = results_dir / f'training_results_v{self.model_version}.csv'
        df_results.to_csv(results_path, index=False)
        
        logger.info(f"Results saved to: {results_path}")
        logger.info(f"\n{df_results[['model_name', 'test_accuracy', 'test_f1_macro', 'training_time_seconds']].to_string(index=False)}")


def main():
    """Main training pipeline execution."""
    logger.info("="*60)
    logger.info("PHASE 4.2: PRODUCTION TRAINING PIPELINE")
    logger.info("="*60)
    
    try:
        # Initialize trainer
        trainer = ModelTrainer(config_path='config/config.yaml')
        
        # Load data
        X_train, X_val, X_test, y_train, y_val, y_test = trainer.load_data()
        
        # Define models to train (based on Phase 4.1 selection)
        from sklearn.linear_model import SGDClassifier
        from sklearn.ensemble import RandomForestClassifier
        
        models_to_train = {
            'sgd': SGDClassifier(
                loss='hinge',  # SVM-like loss
                penalty='l2',
                alpha=1e-4,
                random_state=trainer.config['phase4']['random_state'],
                max_iter=1000,
                tol=1e-3,
                n_jobs=-1
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=trainer.config['phase4']['random_state'],
                n_jobs=-1,
                class_weight='balanced'
            )
        }
        
        # Train each model
        for model_name, model in models_to_train.items():
            trainer.train_model(
                model_name=model_name,
                model=model,
                X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val,
                X_test=X_test, y_test=y_test
            )
        
        # Save models with versioning
        saved_paths = trainer.save_models()
        
        # Generate model manifest
        trainer.generate_model_manifest(saved_paths)
        
        # Save results summary
        trainer.save_results_summary()
        
        logger.info("\n" + "="*60)
        logger.info("PHASE 4.2 TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        
        # Print final summary
        print("\n" + "="*60)
        print("TRAINING PIPELINE SUMMARY")
        print("="*60)
        for model_name, model_data in trainer.trained_models.items():
            print(f"\n{model_name.upper()}:")
            print(f"  Version: v{model_data['version']}")
            print(f"  Test Accuracy: {model_data['metrics']['test_accuracy']*100:.2f}%")
            print(f"  Test F1-Score: {model_data['metrics']['test_f1_macro']:.4f}")
            print(f"  Training Time: {model_data['train_time']:.2f}s")
            print(f"  Meets Target: {'✓' if model_data['metrics']['test_accuracy'] >= trainer.config['phase4']['target_accuracy'] else '✗'}")
        
        print("\n" + "="*60)
        print(f"Model artifacts saved to: models/phase4/")
        print(f"Manifest: models/phase4/model_manifest_v{trainer.model_version}.json")
        print(f"Results: results/phase4/training_results_v{trainer.model_version}.csv")
        print(f"MLflow tracking: mlruns/")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
