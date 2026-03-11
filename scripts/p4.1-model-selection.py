#!/usr/bin/env python3
"""
p4.1-model-selection.py - Model Selection & Baseline Establishment

Phase 4.1: Model Development & Experimentation
Multi-Model Orchestration System - MLOps Production Pipeline

Purpose:
    - Define candidate models for classification task
    - Establish baseline metrics to beat
    - Create model comparison framework
    - Document selection criteria
    - Initialize MLflow experiment tracking

Author: mavyjimz
Date: March 11, 2026
Version: 1.0.0
"""

import os
import sys
import json
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/p4.1-model-selection.log')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a single model candidate."""
    name: str
    model_class: str
    params: Dict[str, Any]
    description: str
    category: str  # 'linear', 'tree-based', 'ensemble', 'svm', 'bayesian'
    expected_training_time: str  # 'fast', 'medium', 'slow'
    scalability: str  # 'high', 'medium', 'low'


@dataclass
class BaselineMetrics:
    """Baseline metrics structure."""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    cross_val_mean: float
    cross_val_std: float
    training_time_sec: float
    model_size_kb: float
    timestamp: str


class ModelSelectionFramework:
    """
    Comprehensive model selection and baseline establishment framework.
    
    This class provides:
    - Candidate model definition and configuration
    - Baseline metric computation
    - Model comparison and ranking
    - Selection criteria documentation
    - MLflow experiment tracking integration
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the model selection framework.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.models: Dict[str, ModelConfig] = {}
        self.baseline_results: Dict[str, BaselineMetrics] = {}
        self.results_df: Optional[pd.DataFrame] = None
        
        # Create necessary directories
        Path("logs").mkdir(exist_ok=True)
        Path("results/phase4").mkdir(parents=True, exist_ok=True)
        Path("models/phase4").mkdir(parents=True, exist_ok=True)
        
        logger.info("Model Selection Framework initialized")
        logger.info(f"Configuration loaded from: {self.config_path}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info("Configuration loaded successfully")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file not found: {self.config_path}")
            logger.info("Using default configuration")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'phase4': {
                'target_accuracy': 0.90,
                'cross_validation_folds': 5,
                'random_state': 42,
                'mlflow_experiment': 'phase4_model_selection',
                'candidate_models': [
                    'logistic_regression',
                    'random_forest',
                    'svm',
                    'xgboost',
                    'gradient_boosting',
                    'naive_bayes',
                    'knn'
                ]
            },
            'data': {
                'train_path': 'data/processed/cleaned_split_train.csv',
                'val_path': 'data/processed/cleaned_split_val.csv',
                'test_path': 'data/processed/cleaned_split_test.csv',
                'embeddings_path': 'data/final/embeddings_v2.0'
            }
        }
    
    def define_candidate_models(self) -> Dict[str, ModelConfig]:
        """
        Define candidate models for evaluation.
        
        Returns:
            Dictionary of model configurations
        """
        logger.info("Defining candidate models...")
        
        self.models = {
            'logistic_regression': ModelConfig(
                name="Logistic Regression",
                model_class="LogisticRegression",
                params={
                    'C': 1.0,
                    'max_iter': 1000,
                    'random_state': 42,
                    'solver': 'lbfgs',
                    'multi_class': 'auto'
                },
                description="Linear model for classification with L2 regularization",
                category='linear',
                expected_training_time='fast',
                scalability='high'
            ),
            'random_forest': ModelConfig(
                name="Random Forest",
                model_class="RandomForestClassifier",
                params={
                    'n_estimators': 100,
                    'max_depth': None,
                    'random_state': 42,
                    'n_jobs': -1,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1
                },
                description="Ensemble of decision trees with bagging",
                category='ensemble',
                expected_training_time='medium',
                scalability='high'
            ),
            'svm': ModelConfig(
                name="Support Vector Machine",
                model_class="SVC",
                params={
                    'C': 1.0,
                    'kernel': 'rbf',
                    'probability': True,
                    'random_state': 42,
                    'gamma': 'scale'
                },
                description="Support Vector Machine with RBF kernel",
                category='svm',
                expected_training_time='slow',
                scalability='low'
            ),
            'xgboost': ModelConfig(
                name="XGBoost",
                model_class="XGBClassifier",
                params={
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': 42,
                    'n_jobs': -1,
                    'use_label_encoder': False,
                    'eval_metric': 'mlogloss'
                },
                description="Extreme Gradient Boosting classifier",
                category='ensemble',
                expected_training_time='medium',
                scalability='high'
            ),
            'gradient_boosting': ModelConfig(
                name="Gradient Boosting",
                model_class="GradientBoostingClassifier",
                params={
                    'n_estimators': 100,
                    'max_depth': 5,
                    'learning_rate': 0.1,
                    'random_state': 42,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1
                },
                description="Gradient Boosting classifier",
                category='ensemble',
                expected_training_time='medium',
                scalability='medium'
            ),
            'naive_bayes': ModelConfig(
                name="Multinomial Naive Bayes",
                model_class="MultinomialNB",
                params={
                    'alpha': 1.0,
                    'fit_prior': True
                },
                description="Naive Bayes classifier for text data",
                category='bayesian',
                expected_training_time='fast',
                scalability='high'
            ),
            'knn': ModelConfig(
                name="K-Nearest Neighbors",
                model_class="KNeighborsClassifier",
                params={
                    'n_neighbors': 5,
                    'weights': 'distance',
                    'n_jobs': -1,
                    'metric': 'euclidean'
                },
                description="K-Nearest Neighbors classifier",
                category='instance-based',
                expected_training_time='fast',
                scalability='low'
            ),
            'decision_tree': ModelConfig(
                name="Decision Tree",
                model_class="DecisionTreeClassifier",
                params={
                    'max_depth': None,
                    'random_state': 42,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'criterion': 'gini'
                },
                description="Decision Tree classifier",
                category='tree-based',
                expected_training_time='fast',
                scalability='medium'
            ),
            'sgd': ModelConfig(
                name="Stochastic Gradient Descent",
                model_class="SGDClassifier",
                params={
                    'loss': 'hinge',
                    'penalty': 'l2',
                    'random_state': 42,
                    'max_iter': 1000,
                    'tol': 1e-3,
                    'n_jobs': -1
                },
                description="Linear classifier with SGD optimization",
                category='linear',
                expected_training_time='fast',
                scalability='high'
            ),
            'lightgbm': ModelConfig(
                name="LightGBM",
                model_class="LGBMClassifier",
                params={
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': 42,
                    'n_jobs': -1,
                    'verbose': -1
                },
                description="Light Gradient Boosting Machine",
                category='ensemble',
                expected_training_time='medium',
                scalability='high'
            )
        }
        
        logger.info(f"Defined {len(self.models)} candidate models")
        for name, config in self.models.items():
            logger.info(f"  - {name}: {config.category} ({config.expected_training_time})")
        
        return self.models
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                  np.ndarray, np.ndarray, np.ndarray]:
        """
        Load training, validation, and test data.
        
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        logger.info("Loading data...")
        
        # Load embeddings
        embeddings_path = Path(self.config['data']['embeddings_path'])
        
        try:
            X_train = np.load(embeddings_path / 'train_emb.npy')
            X_val = np.load(embeddings_path / 'val_emb.npy')
            X_test = np.load(embeddings_path / 'test_emb.npy')
            logger.info(f"Loaded embeddings: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
        except FileNotFoundError as e:
            logger.error(f"Embedding files not found: {e}")
            raise
        
        # Load labels from CSV files
        train_df = pd.read_csv(self.config['data']['train_path'])
        val_df = pd.read_csv(self.config['data']['val_path'])
        test_df = pd.read_csv(self.config['data']['test_path'])
        
        y_train = train_df['intent_encoded'].values
        y_val = val_df['intent_encoded'].values
        y_test = test_df['intent_encoded'].values
        
        logger.info(f"Loaded labels: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")
        logger.info(f"Class distribution - train: {np.bincount(y_train)}, test: {np.bincount(y_test)}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def create_model(self, model_key: str) -> Any:
        """
        Create model instance from configuration.
        
        Args:
            model_key: Key of the model in self.models
            
        Returns:
            Instantiated model
        """
        if model_key not in self.models:
            raise ValueError(f"Unknown model: {model_key}")
        
        config = self.models[model_key]
        model_class = getattr(sys.modules[__name__], config.model_class)
        model = model_class(**config.params)
        
        logger.debug(f"Created model: {config.name}")
        return model
    
    def compute_baseline_metrics(self, model_key: str, model: Any,
                                  X_train: np.ndarray, y_train: np.ndarray,
                                  X_val: np.ndarray, y_val: np.ndarray,
                                  X_test: np.ndarray, y_test: np.ndarray,
                                  n_folds: int = 5) -> BaselineMetrics:
        """
        Compute baseline metrics for a model.
        
        Args:
            model_key: Model identifier
            model: Model instance
            X_train, y_train: Training data
            X_val, y_val: Validation data
            X_test, y_test: Test data
            n_folds: Number of CV folds
            
        Returns:
            BaselineMetrics object
        """
        import time
        
        logger.info(f"Computing baseline metrics for {model_key}...")
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        
        # Train model
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Predictions
        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred_test)
        precision = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
        
        # ROC-AUC (for multiclass)
        try:
            y_proba = model.predict_proba(X_test)
            roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
        except Exception as e:
            logger.warning(f"ROC-AUC computation failed: {e}")
            roc_auc = 0.0
        
        # Model size
        import pickle
        model_path = f"models/phase4/{model_key}_baseline.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        model_size_kb = os.path.getsize(model_path) / 1024
        
        metrics = BaselineMetrics(
            model_name=self.models[model_key].name,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=roc_auc,
            cross_val_mean=cv_scores.mean(),
            cross_val_std=cv_scores.std(),
            training_time_sec=training_time,
            model_size_kb=model_size_kb,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"  Accuracy: {accuracy:.4f} (CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f})")
        logger.info(f"  F1-Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        logger.info(f"  ROC-AUC: {roc_auc:.4f}")
        logger.info(f"  Training time: {training_time:.2f}s, Model size: {model_size_kb:.2f} KB")
        
        return metrics
    
    def initialize_mlflow(self):
        """Initialize MLflow experiment tracking."""
        experiment_name = self.config['phase4']['mlflow_experiment']
        
        try:
            mlflow.set_experiment(experiment_name)
            logger.info(f"MLflow experiment initialized: {experiment_name}")
        except Exception as e:
            logger.warning(f"MLflow initialization failed: {e}")
            logger.info("Continuing without MLflow tracking")
    
    def log_to_mlflow(self, model_key: str, metrics: BaselineMetrics, 
                      params: Dict[str, Any]):
        """
        Log metrics and parameters to MLflow.
        
        Args:
            model_key: Model identifier
            metrics: BaselineMetrics object
            params: Model parameters
        """
        try:
            with mlflow.start_run(run_name=model_key):
                # Log parameters
                mlflow.log_params(params)
                
                # Log metrics
                mlflow.log_metric("accuracy", metrics.accuracy)
                mlflow.log_metric("precision", metrics.precision)
                mlflow.log_metric("recall", metrics.recall)
                mlflow.log_metric("f1_score", metrics.f1_score)
                mlflow.log_metric("roc_auc", metrics.roc_auc)
                mlflow.log_metric("cv_mean", metrics.cross_val_mean)
                mlflow.log_metric("cv_std", metrics.cross_val_std)
                mlflow.log_metric("training_time_sec", metrics.training_time_sec)
                mlflow.log_metric("model_size_kb", metrics.model_size_kb)
                
                # Log model
                model = self.create_model(model_key)
                mlflow.sklearn.log_model(model, f"models/{model_key}")
                
            logger.debug(f"Logged {model_key} to MLflow")
        except Exception as e:
            logger.warning(f"MLflow logging failed for {model_key}: {e}")
    
    def create_comparison_framework(self) -> pd.DataFrame:
        """
        Create model comparison DataFrame.
        
        Returns:
            DataFrame with all model metrics
        """
        if not self.baseline_results:
            logger.warning("No baseline results available")
            return pd.DataFrame()
        
        # Convert to DataFrame
        data = []
        for model_key, metrics in self.baseline_results.items():
            row = asdict(metrics)
            row['model_key'] = model_key
            row['category'] = self.models[model_key].category
            row['training_time_category'] = self.models[model_key].expected_training_time
            row['scalability'] = self.models[model_key].scalability
            data.append(row)
        
        self.results_df = pd.DataFrame(data)
        
        # Sort by accuracy
        self.results_df = self.results_df.sort_values('accuracy', ascending=False)
        
        logger.info("\n" + "="*80)
        logger.info("MODEL COMPARISON RESULTS")
        logger.info("="*80)
        logger.info(self.results_df[['model_key', 'accuracy', 'f1_score', 
                                     'roc_auc', 'cross_val_mean']].to_string(index=False))
        logger.info("="*80)
        
        return self.results_df
    
    def document_selection_criteria(self) -> Dict[str, Any]:
        """
        Document model selection criteria and recommendations.
        
        Returns:
            Dictionary with selection criteria and recommendations
        """
        target_accuracy = self.config['phase4']['target_accuracy']
        
        # Find models meeting target
        models_meeting_target = self.results_df[
            self.results_df['accuracy'] >= target_accuracy
        ] if self.results_df is not None else pd.DataFrame()
        
        # Best model by accuracy
        best_accuracy_model = self.results_df.loc[
            self.results_df['accuracy'].idxmax()
        ] if self.results_df is not None and not self.results_df.empty else None
        
        # Best balanced model (F1-score)
        best_f1_model = self.results_df.loc[
            self.results_df['f1_score'].idxmax()
        ] if self.results_df is not None and not self.results_df.empty else None
        
        # Fastest model (training time)
        fastest_model = self.results_df.loc[
            self.results_df['training_time_sec'].idxmin()
        ] if self.results_df is not None and not self.results_df.empty else None
        
        # Smallest model
        smallest_model = self.results_df.loc[
            self.results_df['model_size_kb'].idxmin()
        ] if self.results_df is not None and not self.results_df.empty else None
        
        criteria = {
            'target_accuracy': target_accuracy,
            'models_meeting_target': len(models_meeting_target),
            'models_meeting_target_list': models_meeting_target['model_key'].tolist() if not models_meeting_target.empty else [],
            'best_accuracy': {
                'model': best_accuracy_model['model_key'] if best_accuracy_model is not None else None,
                'accuracy': float(best_accuracy_model['accuracy']) if best_accuracy_model is not None else None
            },
            'best_f1_score': {
                'model': best_f1_model['model_key'] if best_f1_model is not None else None,
                'f1_score': float(best_f1_model['f1_score']) if best_f1_model is not None else None
            },
            'fastest_training': {
                'model': fastest_model['model_key'] if fastest_model is not None else None,
                'time_sec': float(fastest_model['training_time_sec']) if fastest_model is not None else None
            },
            'smallest_model': {
                'model': smallest_model['model_key'] if smallest_model is not None else None,
                'size_kb': float(smallest_model['model_size_kb']) if smallest_model is not None else None
            },
            'selection_recommendation': self._generate_recommendation(
                best_accuracy_model, models_meeting_target
            ),
            'next_steps': [
                "Proceed to Phase 4.2: Training Pipeline Setup",
                "Implement hyperparameter optimization (Phase 4.5)",
                "Set up 5-fold cross-validation for top candidates",
                "Configure MLflow for experiment tracking"
            ]
        }
        
        # Save to file
        criteria_path = Path("results/phase4/selection_criteria.json")
        with open(criteria_path, 'w') as f:
            json.dump(criteria, f, indent=2)
        
        logger.info(f"\nSelection criteria documented: {criteria_path}")
        logger.info(f"Models meeting target accuracy (>{target_accuracy}): {criteria['models_meeting_target']}")
        if criteria['best_accuracy']['model']:
            logger.info(f"Best accuracy: {criteria['best_accuracy']['model']} ({criteria['best_accuracy']['accuracy']:.4f})")
        
        return criteria
    
    def _generate_recommendation(self, best_model: Optional[pd.Series],
                                  models_meeting_target: pd.DataFrame) -> str:
        """Generate model selection recommendation."""
        if models_meeting_target.empty:
            return "No models meet target accuracy. Consider hyperparameter tuning or feature engineering."
        
        if best_model is None:
            return "Unable to generate recommendation due to missing data."
        
        top_model = models_meeting_target.iloc[0]
        
        recommendation = (
            f"Recommend {top_model['model_key']} ({top_model['model_name']}) as primary candidate. "
            f"Achieves {top_model['accuracy']:.4f} accuracy with "
            f"{top_model['training_time_sec']:.2f}s training time. "
            f"Category: {top_model['category']}. "
            f"Proceed to hyperparameter optimization for further improvements."
        )
        
        return recommendation
    
    def run_full_evaluation(self):
        """
        Run complete model selection and baseline evaluation.
        """
        logger.info("="*80)
        logger.info("PHASE 4.1: MODEL SELECTION & BASELINE ESTABLISHMENT")
        logger.info("="*80)
        
        # Step 1: Define candidate models
        self.define_candidate_models()
        
        # Step 2: Load data
        X_train, y_train, X_val, y_val, X_test, y_test = self.load_data()
        
        # Step 3: Initialize MLflow
        self.initialize_mlflow()
        
        # Step 4: Evaluate each model
        for model_key in self.models.keys():
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"Evaluating: {self.models[model_key].name}")
                logger.info(f"{'='*60}")
                
                model = self.create_model(model_key)
                
                metrics = self.compute_baseline_metrics(
                    model_key, model,
                    X_train, y_train,
                    X_val, y_val,
                    X_test, y_test,
                    n_folds=self.config['phase4']['cross_validation_folds']
                )
                
                self.baseline_results[model_key] = metrics
                
                # Log to MLflow
                self.log_to_mlflow(model_key, metrics, self.models[model_key].params)
                
            except Exception as e:
                logger.error(f"Failed to evaluate {model_key}: {e}")
                continue
        
        # Step 5: Create comparison framework
        self.create_comparison_framework()
        
        # Step 6: Document selection criteria
        criteria = self.document_selection_criteria()
        
        # Step 7: Save results
        if self.results_df is not None:
            results_path = Path("results/phase4/baseline_results.csv")
            self.results_df.to_csv(results_path, index=False)
            logger.info(f"Results saved to: {results_path}")
        
        logger.info("\n" + "="*80)
        logger.info("PHASE 4.1 COMPLETE")
        logger.info("="*80)
        logger.info(f"Evaluated {len(self.baseline_results)} models")
        logger.info(f"Target accuracy: {self.config['phase4']['target_accuracy']}")
        logger.info(f"Models meeting target: {criteria['models_meeting_target']}")
        logger.info("="*80)
        
        return criteria


def main():
    """Main entry point for model selection."""
    logger.info("Starting Phase 4.1: Model Selection & Baseline Establishment")
    
    # Initialize framework
    framework = ModelSelectionFramework()
    
    # Run full evaluation
    criteria = framework.run_full_evaluation()
    
    logger.info("\nPhase 4.1 completed successfully!")
    logger.info("Ready to proceed to Phase 4.2: Training Pipeline Setup")
    
    return criteria


if __name__ == "__main__":
    main()
