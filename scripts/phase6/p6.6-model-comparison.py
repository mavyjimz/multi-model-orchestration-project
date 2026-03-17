#!/usr/bin/env python3
"""
p6.6-model-comparison.py (PATCHED)
Phase 6.6: Model Comparison & Selection Framework
Added: Auto-detection for label column + robust error handling
"""
import json
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from datetime import datetime
import mlflow
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/p6.6-model-comparison.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def find_column(df: pd.DataFrame, preferred: str, alternatives: List[str]) -> Optional[str]:
    """Find column name by trying preferred name then alternatives."""
    if preferred in df.columns:
        return preferred
    for alt in alternatives:
        if alt in df.columns:
            logger.info(f"Auto-detected column '{preferred}' as '{alt}'")
            return alt
    return None


class ModelComparator:
    """Production-grade model comparison framework."""
    
    def __init__(self, mlflow_uri: str = 'file:./mlruns'):
        self.mlflow_uri = mlflow_uri
        mlflow.set_tracking_uri(mlflow_uri)
        self.client = mlflow.tracking.MlflowClient()
        self.results = {}
        
    def load_model_metrics(self, model_name: str, version: str) -> Dict:
        """Load comprehensive metrics for a model version from MLflow."""
        logger.info(f"Loading metrics for {model_name} v{version}")
        mv = self.client.get_model_version(model_name, version)
        run_id = mv.run_id
        run = self.client.get_run(run_id)
        
        return {
            'name': model_name, 'version': version, 'stage': mv.current_stage,
            'metrics': dict(run.data.metrics), 'params': dict(run.data.params),
            'tags': dict(run.data.tags),
            'artifacts': [a.path for a in self.client.list_artifacts(run_id)],
            'created': mv.creation_timestamp, 'last_updated': mv.last_updated_timestamp
        }
    
    def statistical_significance_test(self, model_a_preds: np.ndarray, model_b_preds: np.ndarray, 
                                     labels: np.ndarray, alpha: float = 0.05) -> Dict:
        """Perform McNemar's test for statistical significance."""
        both_correct = np.sum((model_a_preds == labels) & (model_b_preds == labels))
        a_correct_b_wrong = np.sum((model_a_preds == labels) & (model_b_preds != labels))
        a_wrong_b_correct = np.sum((model_a_preds != labels) & (model_b_preds == labels))
        both_wrong = np.sum((model_a_preds != labels) & (model_b_preds != labels))
        
        b, c = a_correct_b_wrong, a_wrong_b_correct
        if b + c == 0:
            chi2_stat, p_value = 0, 1.0
        else:
            chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
            p_value = 1 - stats.chi2.cdf(chi2_stat, df=1)
        
        return {
            'test': 'McNemar', 'contingency_table': [[both_correct, a_correct_b_wrong], [a_wrong_b_correct, both_wrong]],
            'chi2_statistic': float(chi2_stat), 'p_value': float(p_value),
            'significant': p_value < alpha, 'alpha': alpha,
            'interpretation': f"Models {'differ significantly' if p_value < alpha else 'do not differ significantly'} at alpha={alpha} (p={p_value:.4f})"
        }
    
    def bootstrap_confidence_interval(self, metric_func, predictions: np.ndarray, labels: np.ndarray,
                                     n_bootstraps: int = 1000, confidence: float = 0.95) -> Dict:
        """Compute bootstrap confidence interval for any metric."""
        n_samples = len(labels)
        bootstrap_scores = [metric_func(predictions[np.random.choice(n_samples, size=n_samples, replace=True)], 
                                       labels[np.random.choice(n_samples, size=n_samples, replace=True)]) 
                           for _ in range(n_bootstraps)]
        lower = np.percentile(bootstrap_scores, (1 - confidence) / 2 * 100)
        upper = np.percentile(bootstrap_scores, (1 + confidence) / 2 * 100)
        return {
            'metric': metric_func.__name__, 'point_estimate': metric_func(predictions, labels),
            'ci_lower': float(lower), 'ci_upper': float(upper),
            'confidence_level': confidence, 'n_bootstraps': n_bootstraps
        }
    
    def business_impact_analysis(self, model_metrics: Dict, cost_per_inference: float = 0.001,
                                revenue_per_correct: float = 0.10, cost_per_error: float = 0.50,
                                monthly_predictions: int = 100000) -> Dict:
        """Model business impact modeling."""
        accuracy = model_metrics['metrics'].get('accuracy', 0)
        latency_p95 = model_metrics['metrics'].get('latency_p95_ms', 100)
        inference_cost = monthly_predictions * cost_per_inference
        correct_predictions = monthly_predictions * accuracy
        errors = monthly_predictions * (1 - accuracy)
        revenue = correct_predictions * revenue_per_correct
        error_cost = errors * cost_per_error
        net_value = revenue - inference_cost - error_cost
        
        return {
            'monthly_inference_cost': inference_cost, 'monthly_revenue': revenue,
            'monthly_error_cost': error_cost, 'net_monthly_value': net_value,
            'annualized_net_value': net_value * 12,
            'cost_per_correct_prediction': cost_per_inference / accuracy if accuracy > 0 else float('inf'),
            'roi_percentage': ((net_value / inference_cost) - 1) * 100 if inference_cost > 0 else 0,
            'latency_compliance': latency_p95 <= 100,
            'assumptions': {'cost_per_inference': cost_per_inference, 'revenue_per_correct': revenue_per_correct,
                           'cost_per_error': cost_per_error, 'monthly_predictions': monthly_predictions}
        }
    
    def weighted_selection_score(self, model_metrics: Dict, business_impact: Dict,
                                weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate weighted selection score for model promotion decision."""
        if weights is None:
            weights = {'accuracy': 0.35, 'latency': 0.20, 'business_value': 0.25, 'stability': 0.10, 'maintainability': 0.10}
        
        accuracy_score = model_metrics['metrics'].get('accuracy', 0)
        latency_ms = model_metrics['metrics'].get('latency_p95_ms', 100)
        latency_score = max(0, 1 - (latency_ms / 200))
        business_score = min(1.0, max(0, business_impact['net_monthly_value'] / 10000))
        ci_width = model_metrics.get('accuracy_ci_upper', 1) - model_metrics.get('accuracy_ci_lower', 0)
        stability_score = max(0, 1 - ci_width / 0.1)
        model_size_mb = model_metrics.get('model_size_mb', 10)
        maintainability_score = max(0, 1 - (model_size_mb / 100))
        
        return round(weights['accuracy'] * accuracy_score + weights['latency'] * latency_score +
                    weights['business_value'] * business_score + weights['stability'] * stability_score +
                    weights['maintainability'] * maintainability_score, 4)
    
    def generate_recommendation(self, comparison_results: Dict, promotion_threshold: float = 0.75) -> Dict:
        """Generate promotion recommendation based on comparison results."""
        models = comparison_results['models']
        best_model = max(models, key=lambda m: m['weighted_score'])
        
        recommendation = {
            'recommended_model': best_model['name'], 'recommended_version': best_model['version'],
            'weighted_score': best_model['weighted_score'],
            'meets_threshold': best_model['weighted_score'] >= promotion_threshold,
            'promotion_ready': (best_model['weighted_score'] >= promotion_threshold and
                               best_model['business_impact']['latency_compliance']),
            'reasoning': [], 'risks': [], 'next_steps': []
        }
        
        if best_model['weighted_score'] >= promotion_threshold:
            recommendation['reasoning'].append(f"Weighted score {best_model['weighted_score']:.4f} exceeds threshold {promotion_threshold}")
        else:
            recommendation['reasoning'].append(f"Weighted score {best_model['weighted_score']:.4f} below threshold {promotion_threshold}")
        
        if best_model['business_impact']['latency_compliance']:
            recommendation['reasoning'].append("Meets latency SLA (P95 <= 100ms)")
        else:
            recommendation['risks'].append("Does not meet latency SLA")
        
        if recommendation['promotion_ready']:
            recommendation['next_steps'].extend([
                f"Promote {best_model['name']} v{best_model['version']} to PRODUCTION stage",
                "Update promotion_audit_log.jsonl with decision",
                "Trigger Phase 7 CI/CD pipeline for automated deployment"
            ])
        else:
            recommendation['next_steps'].extend([
                "Continue A/B testing with increased traffic allocation",
                "Review feature engineering for accuracy improvement"
            ])
        
        return recommendation
    
    def compare_models(self, model_specs: List[Dict], test_data_path: str, weights: Optional[Dict] = None) -> Dict:
        """Main comparison orchestration method."""
        logger.info(f"Starting comparison for {len(model_specs)} models")
        
        test_df = pd.read_csv(test_data_path)
        logger.info(f"Loaded test  {len(test_df)} rows, columns: {list(test_df.columns)}")
        
        # Auto-detect text and label columns
        text_col = find_column(test_df, 'cleaned_text', ['text', 'message', 'utterance', 'input', 'query'])
        label_col = find_column(test_df, 'label', ['intent', 'class', 'target', 'category', 'y'])
        
        if not text_col or not label_col:
            raise ValueError(f"Could not find required columns. Available: {list(test_df.columns)}")
        
        texts = test_df[text_col].fillna('').astype(str).tolist()
        labels = test_df[label_col].values
        
        comparison_results = {
            'timestamp': datetime.now().isoformat(), 'test_samples': len(labels),
            'models': [], 'pairwise_tests': [], 'recommendation': None
        }
        
        for spec in model_specs:
            logger.info(f"Evaluating {spec['model_name']} v{spec['version']}")
            metrics = self.load_model_metrics(spec['model_name'], spec['version'])
            
            preds_path = spec.get('predictions_path')
            predictions = np.load(preds_path) if preds_path and Path(preds_path).exists() else None
            
            if predictions is not None:
                ci_result = self.bootstrap_confidence_interval(lambda p, l: np.mean(p == l), predictions, labels)
                metrics['accuracy_ci_lower'] = ci_result['ci_lower']
                metrics['accuracy_ci_upper'] = ci_result['ci_upper']
            
            business_impact = self.business_impact_analysis(metrics)
            weighted_score = self.weighted_selection_score(metrics, business_impact, weights)
            
            comparison_results['models'].append({
                'name': spec['model_name'], 'version': spec['version'],
                'metrics': metrics, 'business_impact': business_impact,
                'weighted_score': weighted_score, 'predictions_available': predictions is not None
            })
        
        comparison_results['recommendation'] = self.generate_recommendation(comparison_results)
        self.results = comparison_results
        logger.info(f"Comparison complete. Recommended: {comparison_results['recommendation']['recommended_model']}")
        return comparison_results
    
    def save_results(self, output_path: str):
        """Save comparison results to JSON file."""
        def serialize(obj):
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, (np.integer, np.floating)): return obj.item()
            if hasattr(obj, 'isoformat'): return obj.isoformat()
            return str(obj)
        serializable = json.loads(json.dumps(self.results, default=serialize, indent=2))
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        logger.info(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Phase 6.6: Model Comparison Framework')
    parser.add_argument('--models', type=str, required=True, help='Comma-separated: name:version:preds_path')
    parser.add_argument('--test-data', type=str, default='data/processed/cleaned_split_test.csv')
    parser.add_argument('--output', type=str, default='results/phase6/p6.6-comparison-results.json')
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--mlflow-uri', type=str, default='file:./mlruns')
    args = parser.parse_args()
    
    model_specs = []
    for spec in args.models.split(','):
        parts = spec.strip().split(':')
        model_specs.append({'model_name': parts[0], 'version': parts[1],
                           'predictions_path': parts[2] if len(parts) > 2 else None})
    
    weights = json.loads(args.weights) if args.weights else None
    comparator = ModelComparator(mlflow_uri=args.mlflow_uri)
    results = comparator.compare_models(model_specs=model_specs, test_data_path=args.test_data, weights=weights)
    comparator.save_results(args.output)
    
    print(f"\n{'='*60}\nMODEL COMPARISON SUMMARY\n{'='*60}")
    rec = results['recommendation']
    print(f"Recommended Model: {rec['recommended_model']} v{rec['recommended_version']}")
    print(f"Weighted Score: {rec['weighted_score']:.4f}")
    print(f"Promotion Ready: {rec['promotion_ready']}")
    print(f"\nReasoning:")
    for r in rec['reasoning']: print(f"  - {r}")
    if rec['risks']:
        print(f"\nRisks:")
        for r in rec['risks']: print(f"  - {r}")
    print(f"\nNext Steps:")
    for step in rec['next_steps']: print(f"  - {step}")
    print(f"{'='*60}\n")
    return 0 if rec['promotion_ready'] else 1


if __name__ == '__main__':
    exit(main())
