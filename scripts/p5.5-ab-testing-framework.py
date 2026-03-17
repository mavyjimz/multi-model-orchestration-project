#!/usr/bin/env python3
"""
Phase 5.5: A/B Testing Framework
Traffic splitting, statistical significance testing, and rollback logic

Compares SGD v1.0.1 (Primary) vs XGBoost v1.0.1_balanced (Candidate)
"""

import os
import sys
import json
import time
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/p5.5-ab-testing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# ClassMapper - Exact copy from p4.6-edge-case-resolution.py
# Required for pickle deserialization of models trained with this class
# =============================================================================
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


class ABTestingFramework:
    """
    A/B Testing Framework for model comparison
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results: Dict[str, Any] = {}
        self.primary_model = None
        self.candidate_model = None
        self.primary_vectorizer = None
        
    def load_models(self) -> bool:
        """
        Load primary and candidate models with dictionary wrapper handling
        """
        try:
            logger.info("Loading primary model (SGD v1.0.1)...")
            with open(self.config['primary_model_path'], 'rb') as f:
                loaded = pickle.load(f)
            if isinstance(loaded, dict) and 'model' in loaded:
                self.primary_model = loaded['model']
            else:
                self.primary_model = loaded
            
            logger.info("Loading candidate model (XGBoost v1.0.1_balanced)...")
            with open(self.config['candidate_model_path'], 'rb') as f:
                loaded = pickle.load(f)
            if isinstance(loaded, dict) and 'model' in loaded:
                self.candidate_model = loaded['model']
            else:
                self.candidate_model = loaded
            
            logger.info("Loading vectorizer...")
            with open(self.config['vectorizer_path'], 'rb') as f:
                self.primary_vectorizer = pickle.load(f)
            
            logger.info("Models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def simulate_traffic(self, test_data: pd.DataFrame, split_ratio: float = 0.5) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split test data between primary and candidate models
        """
        logger.info(f"Simulating traffic with {split_ratio:.2f} split ratio")
        
        n_samples = len(test_data)
        assignments = np.random.rand(n_samples) < split_ratio
        
        primary_data = test_data[assignments].reset_index(drop=True)
        candidate_data = test_data[~assignments].reset_index(drop=True)
        
        logger.info(f"Primary traffic: {len(primary_data)} samples")
        logger.info(f"Candidate traffic: {len(candidate_data)} samples")
        
        return primary_data, candidate_data
    
    def predict_batch(self, model: Any, data: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
        """
        Run batch predictions and measure latency
        """
        start_time = time.perf_counter()
        
        # Transform features
        X_text = data['user_input_clean'].values
        X_transformed = self.primary_vectorizer.transform(X_text)
        
        # Predict
        predictions = model.predict(X_transformed)
        probabilities = model.predict_proba(X_transformed) if hasattr(model, 'predict_proba') else None
        
        end_time = time.perf_counter()
        latency = (end_time - start_time) * 1000  # ms
        avg_latency = latency / len(data) if len(data) > 0 else 0
        
        return predictions, probabilities, avg_latency
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate performance metrics
        """
        from sklearn.metrics import accuracy_score, f1_score
        
        accuracy = accuracy_score(y_true, y_pred)
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        
        return {
            'accuracy': float(accuracy),
            'f1_weighted': float(f1_weighted),
            'f1_macro': float(f1_macro)
        }
    
    def statistical_test(self, primary_metrics: Dict, candidate_metrics: Dict, 
                        primary_preds: np.ndarray, candidate_preds: np.ndarray,
                        y_true: np.ndarray) -> Dict[str, Any]:
        """
        Perform statistical significance testing
        """
        results = {}
        
        # Chi-square test for accuracy difference
        primary_correct = (primary_preds == y_true[:len(primary_preds)]).sum()
        candidate_correct = (candidate_preds == y_true[len(primary_preds):]).sum()
        
        n_primary = len(primary_preds)
        n_candidate = len(candidate_preds)
        
        contingency = np.array([
            [primary_correct, n_primary - primary_correct],
            [candidate_correct, n_candidate - candidate_correct]
        ])
        
        chi2, p_value_chi2, _, _ = stats.chi2_contingency(contingency)
        results['chi_square_test'] = {
            'statistic': float(chi2),
            'p_value': float(p_value_chi2),
            'significant': p_value_chi2 < 0.05
        }
        
        # Latency comparison
        latency_diff = candidate_metrics.get('avg_latency', 0) - primary_metrics.get('avg_latency', 0)
        results['latency_comparison'] = {
            'primary_avg_ms': primary_metrics.get('avg_latency', 0),
            'candidate_avg_ms': candidate_metrics.get('avg_latency', 0),
            'difference_ms': float(latency_diff),
            'candidate_slower': latency_diff > 0
        }
        
        return results
    
    def evaluate_rollback(self, primary_metrics: Dict, candidate_metrics: Dict) -> Dict[str, Any]:
        """
        Determine if rollback is triggered based on thresholds
        """
        thresholds = self.config.get('rollback_thresholds', {})
        
        accuracy_degradation = primary_metrics['accuracy'] - candidate_metrics['accuracy']
        f1_degradation = primary_metrics['f1_weighted'] - candidate_metrics['f1_weighted']
        latency_increase = candidate_metrics.get('avg_latency', 0) - primary_metrics.get('avg_latency', 0)
        
        rollback_triggered = False
        reasons = []
        
        if accuracy_degradation > thresholds.get('max_accuracy_drop', 0.05):
            rollback_triggered = True
            reasons.append(f"Accuracy drop {accuracy_degradation:.4f} > {thresholds.get('max_accuracy_drop', 0.05)}")
        
        if f1_degradation > thresholds.get('max_f1_drop', 0.05):
            rollback_triggered = True
            reasons.append(f"F1 drop {f1_degradation:.4f} > {thresholds.get('max_f1_drop', 0.05)}")
        
        if latency_increase > thresholds.get('max_latency_increase_ms', 50):
            rollback_triggered = True
            reasons.append(f"Latency increase {latency_increase:.2f}ms > {thresholds.get('max_latency_increase_ms', 50)}ms")
        
        return {
            'rollback_triggered': rollback_triggered,
            'reasons': reasons,
            'recommendation': 'ROLLBACK' if rollback_triggered else 'PROMOTE'
        }
    
    def run_ab_test(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute full A/B test pipeline
        """
        logger.info("Starting A/B Test...")
        
        # Traffic splitting
        primary_data, candidate_data = self.simulate_traffic(test_data, split_ratio=0.5)
        
        if len(primary_data) == 0 or len(candidate_data) == 0:
            logger.error("Traffic split resulted in empty datasets")
            return {'error': 'Empty traffic split'}
        
        # Primary model inference
        logger.info("Running primary model inference...")
        primary_preds, primary_probs, primary_latency = self.predict_batch(
            self.primary_model, primary_data
        )
        primary_metrics = self.calculate_metrics(
            primary_data['intent_encoded'].values, primary_preds
        )
        primary_metrics['avg_latency'] = primary_latency
        primary_metrics['sample_count'] = len(primary_data)
        
        # Candidate model inference
        logger.info("Running candidate model inference...")
        candidate_preds, candidate_probs, candidate_latency = self.predict_batch(
            self.candidate_model, candidate_data
        )
        candidate_metrics = self.calculate_metrics(
            candidate_data['intent_encoded'].values, candidate_preds
        )
        candidate_metrics['avg_latency'] = candidate_latency
        candidate_metrics['sample_count'] = len(candidate_data)
        
        # Statistical testing
        logger.info("Running statistical tests...")
        y_true_combined = np.concatenate([
            primary_data['intent_encoded'].values,
            candidate_data['intent_encoded'].values
        ])
        stats_results = self.statistical_test(
            primary_metrics, candidate_metrics,
            primary_preds, candidate_preds,
            y_true_combined
        )
        
        # Rollback evaluation
        logger.info("Evaluating rollback criteria...")
        rollback_decision = self.evaluate_rollback(primary_metrics, candidate_metrics)
        
        # Compile results
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'test_id': f"ab_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'primary_model': {
                'name': 'SGD v1.0.1',
                'metrics': primary_metrics
            },
            'candidate_model': {
                'name': 'XGBoost v1.0.1_balanced',
                'metrics': candidate_metrics
            },
            'statistical_tests': stats_results,
            'rollback_decision': rollback_decision,
            'config': self.config
        }
        
        logger.info(f"A/B Test Complete. Recommendation: {rollback_decision['recommendation']}")
        return self.results
    
    def save_results(self, output_path: str) -> None:
        """
        Save results to JSON file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"Results saved to {output_path}")


def main():
    """
    Main execution function
    """
    # Configuration
    config = {
        'primary_model_path': 'models/phase4/sgd_v1.0.1.pkl',
        'candidate_model_path': 'models/phase4/xgboost_v1.0.1_balanced.pkl',
        'vectorizer_path': 'data/final/embeddings_v2.0/vectorizer.pkl',
        'test_data_path': 'data/processed/cleaned_split_test.csv',
        'output_path': 'results/phase5/ab_test_analysis.json',
        'rollback_thresholds': {
            'max_accuracy_drop': 0.05,
            'max_f1_drop': 0.05,
            'max_latency_increase_ms': 50
        }
    }
    
    # Check INPUT file existence only
    input_files = [
        config['primary_model_path'],
        config['candidate_model_path'],
        config['vectorizer_path'],
        config['test_data_path']
    ]
    
    for path in input_files:
        if not os.path.exists(path):
            logger.error(f"Required file not found: {path}")
            sys.exit(1)
    
    # Load test data
    logger.info("Loading test data...")
    test_data = pd.read_csv(config['test_data_path'])
    
    # Initialize framework
    framework = ABTestingFramework(config)
    
    # Load models
    if not framework.load_models():
        logger.error("Failed to load models")
        sys.exit(1)
    
    # Run A/B test
    results = framework.run_ab_test(test_data)
    
    # Save results
    framework.save_results(config['output_path'])
    
    # Print summary
    print("\n" + "="*60)
    print("A/B TEST SUMMARY")
    print("="*60)
    print(f"Primary Model (SGD):     Accuracy={results['primary_model']['metrics']['accuracy']:.4f}")
    print(f"Candidate Model (XGB):   Accuracy={results['candidate_model']['metrics']['accuracy']:.4f}")
    print(f"Statistical Significance: {results['statistical_tests']['chi_square_test']['significant']}")
    print(f"Recommendation:          {results['rollback_decision']['recommendation']}")
    if results['rollback_decision']['reasons']:
        print(f"Reasons: {', '.join(results['rollback_decision']['reasons'])}")
    print("="*60)


if __name__ == '__main__':
    main()
