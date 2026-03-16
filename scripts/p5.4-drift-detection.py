#!/usr/bin/env python3
"""
p5.4-drift-detection.py
Phase 5.4: Model Drift Detection for Multi-Model Orchestration System

Implements statistical drift detection using KS test and PSI to monitor
feature distribution changes and prediction shifts over time.

Author: Mavyjimz
Date: March 2026
Version: 1.0.0
"""

import os
import sys
import json
import pickle
import logging
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

CONFIG = {
    'model_path': 'models/phase4/sgd_v1.0.1.pkl',
    'vectorizer_path': 'data/final/embeddings_v2.0/vectorizer.pkl',
    'baseline_data_path': 'data/processed/cleaned_split_test.csv',
    'output_dir': 'results/phase5',
    'log_dir': 'logs',
    'log_level': 'INFO',
    'psi_bins': 10,  # Bins for Population Stability Index
    'ks_alpha': 0.05,  # Significance level for Kolmogorov-Smirnov test
    'drift_threshold_psi': 0.2,  # PSI > 0.2 indicates significant drift
    'drift_threshold_ks': 0.05,  # KS p-value < 0.05 indicates drift
    'sample_size': 500  # Max samples for drift analysis
}

Path(CONFIG['output_dir']).mkdir(parents=True, exist_ok=True)
Path(CONFIG['log_dir']).mkdir(parents=True, exist_ok=True)

def setup_logging():
    logger = logging.getLogger('p5.4-drift-detection')
    logger.setLevel(getattr(logging, CONFIG['log_level']))
    logger.handlers = []
    
    log_file = Path(CONFIG['log_dir']) / 'p5.4-drift-detection.log'
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s'))
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter('%(levelname)-8s | %(message)s'))
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

logger = setup_logging()


class DriftDetector:
    """Statistical drift detection using KS test and PSI."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vectorizer = None
        self.baseline_features = None
        self.baseline_predictions = None
        self.results = {}
        
    def load_baseline(self) -> bool:
        """Load baseline data for comparison."""
        try:
            logger.info(f"Loading vectorizer from {self.config['vectorizer_path']}")
            with open(self.config['vectorizer_path'], 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            logger.info(f"Loading baseline data from {self.config['baseline_data_path']}")
            baseline_df = pd.read_csv(self.config['baseline_data_path'])
            
            # Sample if too large
            if len(baseline_df) > self.config['sample_size']:
                baseline_df = baseline_df.sample(n=self.config['sample_size'], random_state=42)
                logger.info(f"Sampled {len(baseline_df)} rows for baseline")
            
            # Transform to features
            texts = baseline_df['user_input_clean'].astype(str).tolist()
            self.baseline_features = self.vectorizer.transform(texts)
            
            # Generate baseline predictions
            with open(self.config['model_path'], 'rb') as f:
                model_dict = pickle.load(f)
                model = model_dict.get('model', model_dict) if isinstance(model_dict, dict) else model_dict
            
            self.baseline_predictions = model.predict(self.baseline_features)
            
            logger.info(f"Baseline loaded: {self.baseline_features.shape[0]} samples, {self.baseline_features.shape[1]} features")
            return True
        except Exception as e:
            logger.error(f"Error loading baseline: {e}")
            return False
    
    def compute_psi(self, baseline: np.ndarray, current: np.ndarray, bins: int = None) -> float:
        """
        Compute Population Stability Index (PSI) between two distributions.
        
        PSI < 0.1: No significant change
        PSI 0.1-0.25: Moderate change
        PSI > 0.25: Significant change
        """
        if bins is None:
            bins = self.config['psi_bins']
        
        # Handle sparse matrix input
        if hasattr(baseline, 'toarray'):
            baseline = baseline.toarray().flatten()
        if hasattr(current, 'toarray'):
            current = current.toarray().flatten()
        
        # Remove zeros for PSI calculation (sparse features)
        baseline = baseline[baseline != 0]
        current = current[current != 0]
        
        if len(baseline) == 0 or len(current) == 0:
            return 0.0
        
        # Create bins based on baseline distribution
        min_val = min(baseline.min(), current.min())
        max_val = max(baseline.max(), current.max())
        
        if min_val == max_val:
            return 0.0
        
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        
        # Calculate percentages in each bin
        baseline_pct = np.histogram(baseline, bins=bin_edges)[0] / len(baseline)
        current_pct = np.histogram(current, bins=bin_edges)[0] / len(current)
        
        # Add small constant to avoid log(0)
        epsilon = 1e-5
        baseline_pct = np.where(baseline_pct == 0, epsilon, baseline_pct)
        current_pct = np.where(current_pct == 0, epsilon, current_pct)
        
        # Calculate PSI
        psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
        
        return float(psi)
    
    def compute_ks_test(self, baseline: np.ndarray, current: np.ndarray) -> Tuple[float, float]:
        """
        Compute Kolmogorov-Smirnov test between two distributions.
        
        Returns: (statistic, p-value)
        - p-value < alpha: distributions are significantly different (drift detected)
        """
        # Handle sparse matrix input
        if hasattr(baseline, 'toarray'):
            baseline = baseline.toarray().flatten()
        if hasattr(current, 'toarray'):
            current = current.toarray().flatten()
        
        # Remove zeros for KS test
        baseline = baseline[baseline != 0]
        current = current[current != 0]
        
        if len(baseline) < 2 or len(current) < 2:
            return 0.0, 1.0
        
        statistic, p_value = stats.ks_2samp(baseline, current)
        
        return float(statistic), float(p_value)
    
    def analyze_feature_drift(self, current_features: np.ndarray) -> Dict[str, Any]:
        """Analyze drift across feature dimensions."""
        logger.info("Analyzing feature drift...")
        
        drift_results = {
            'features_analyzed': 0,
            'features_with_drift_psi': 0,
            'features_with_drift_ks': 0,
            'avg_psi': 0.0,
            'max_psi': 0.0,
            'min_ks_pvalue': 1.0,
            'drift_summary': []
        }
        
        # Analyze top features by variance (limit for performance)
        n_features = min(100, self.baseline_features.shape[1])
        feature_indices = np.argsort(np.array(self.baseline_features.toarray().var(axis=0).flatten())[::-1])[:n_features]
        
        psis = []
        ks_pvalues = []
        
        for idx in feature_indices:
            baseline_feat = self.baseline_features[:, idx].toarray().flatten()
            current_feat = current_features[:, idx].toarray().flatten()
            
            psi = self.compute_psi(baseline_feat, current_feat)
            ks_stat, ks_pval = self.compute_ks_test(baseline_feat, current_feat)
            
            psis.append(psi)
            ks_pvalues.append(ks_pval)
            
            if psi > self.config['drift_threshold_psi'] or ks_pval < self.config['ks_alpha']:
                drift_results['drift_summary'].append({
                    'feature_index': int(idx),
                    'psi': psi,
                    'ks_statistic': ks_stat,
                    'ks_pvalue': ks_pval,
                    'drift_detected': psi > self.config['drift_threshold_psi'] or ks_pval < self.config['ks_alpha']
                })
        
        drift_results['features_analyzed'] = n_features
        drift_results['features_with_drift_psi'] = sum(1 for p in psis if p > self.config['drift_threshold_psi'])
        drift_results['features_with_drift_ks'] = sum(1 for p in ks_pvalues if p < self.config['ks_alpha'])
        drift_results['avg_psi'] = float(np.mean(psis)) if psis else 0.0
        drift_results['max_psi'] = float(np.max(psis)) if psis else 0.0
        drift_results['min_ks_pvalue'] = float(np.min(ks_pvalues)) if ks_pvalues else 1.0
        
        logger.info(f"Feature drift: {drift_results['features_with_drift_psi']}/{n_features} features with PSI drift")
        
        return drift_results
    
    def analyze_prediction_drift(self, current_predictions: np.ndarray) -> Dict[str, Any]:
        """Analyze drift in prediction distributions."""
        logger.info("Analyzing prediction drift...")
        
        # Compute class distribution for baseline and current
        baseline_counts = np.bincount(self.baseline_predictions, minlength=41)
        current_counts = np.bincount(current_predictions, minlength=41)
        
        baseline_dist = baseline_counts / baseline_counts.sum()
        current_dist = current_counts / current_counts.sum()
        
        # Compute PSI for prediction distribution
        psi = self.compute_psi(baseline_dist, current_dist, bins=len(baseline_dist))
        
        # Compute chi-square test for distribution difference
        chi2_stat, chi2_pval = stats.chisquare(current_counts, baseline_counts)
        
        return {
            'prediction_psi': float(psi),
            'chi2_statistic': float(chi2_stat),
            'chi2_pvalue': float(chi2_pval),
            'drift_detected': psi > self.config['drift_threshold_psi'] or chi2_pval < self.config['ks_alpha'],
            'baseline_distribution': baseline_dist.tolist(),
            'current_distribution': current_dist.tolist()
        }
    
    def generate_drift_report(self, feature_drift: Dict, prediction_drift: Dict) -> Dict[str, Any]:
        """Generate comprehensive drift detection report."""
        overall_drift = (
            feature_drift['max_psi'] > self.config['drift_threshold_psi'] or
            feature_drift['min_ks_pvalue'] < self.config['ks_alpha'] or
            prediction_drift['drift_detected']
        )
        
        return {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'psi_threshold': self.config['drift_threshold_psi'],
                'ks_alpha': self.config['ks_alpha'],
                'sample_size': self.config['sample_size']
            },
            'overall_drift_detected': overall_drift,
            'feature_drift': feature_drift,
            'prediction_drift': prediction_drift,
            'recommendations': self._generate_recommendations(feature_drift, prediction_drift)
        }
    
    def _generate_recommendations(self, feature_drift: Dict, prediction_drift: Dict) -> List[str]:
        """Generate actionable recommendations based on drift analysis."""
        recommendations = []
        
        if feature_drift['max_psi'] > self.config['drift_threshold_psi']:
            recommendations.append(f"High PSI detected ({feature_drift['max_psi']:.3f}): Review feature engineering pipeline")
        
        if feature_drift['min_ks_pvalue'] < self.config['ks_alpha']:
            recommendations.append(f"KS test significant (p={feature_drift['min_ks_pvalue']:.4f}): Investigate data source changes")
        
        if prediction_drift['drift_detected']:
            recommendations.append("Prediction distribution shifted: Consider model retraining or calibration")
        
        if not recommendations:
            recommendations.append("No significant drift detected: Continue monitoring")
        
        return recommendations
    
    def generate_drift_plot(self, output_path: str, feature_drift: Dict, prediction_drift: Dict) -> str:
        """Generate visualization of drift analysis."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # PSI distribution plot
        psi_values = [d['psi'] for d in feature_drift.get('drift_summary', [])]
        if psi_values:
            axes[0].hist(psi_values, bins=20, edgecolor='black', alpha=0.7)
            axes[0].axvline(x=self.config['drift_threshold_psi'], color='red', linestyle='--', label=f'Threshold ({self.config["drift_threshold_psi"]})')
            axes[0].set_xlabel('PSI Value')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title('Feature PSI Distribution')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # Prediction distribution comparison
        baseline_dist = prediction_drift.get('baseline_distribution', [])
        current_dist = prediction_drift.get('current_distribution', [])
        if baseline_dist and current_dist:
            x = np.arange(len(baseline_dist))
            width = 0.35
            axes[1].bar(x - width/2, baseline_dist, width, label='Baseline', alpha=0.7)
            axes[1].bar(x + width/2, current_dist, width, label='Current', alpha=0.7)
            axes[1].set_xlabel('Class Index')
            axes[1].set_ylabel('Proportion')
            axes[1].set_title('Prediction Distribution Comparison')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved drift plot: {output_path}")
        return output_path
    
    def run_drift_detection(self, current_data_path: Optional[str] = None) -> Dict[str, Any]:
        """Execute full drift detection analysis."""
        logger.info("="*60)
        logger.info("PHASE 5.4: MODEL DRIFT DETECTION")
        logger.info("="*60)
        
        if not self.load_baseline():
            return {'error': 'Failed to load baseline'}
        
        # If current data provided, analyze it; otherwise use baseline as "current" for demo
        if current_data_path and os.path.exists(current_data_path):
            logger.info(f"Loading current data from {current_data_path}")
            current_df = pd.read_csv(current_data_path)
            if len(current_df) > self.config['sample_size']:
                current_df = current_df.sample(n=self.config['sample_size'], random_state=42)
            texts = current_df['user_input_clean'].astype(str).tolist()
            current_features = self.vectorizer.transform(texts)
            
            with open(self.config['model_path'], 'rb') as f:
                model_dict = pickle.load(f)
                model = model_dict.get('model', model_dict) if isinstance(model_dict, dict) else model_dict
            current_predictions = model.predict(current_features)
        else:
            logger.info("No current data provided; using baseline for self-comparison demo")
            current_features = self.baseline_features
            current_predictions = self.baseline_predictions
        
        # Analyze drift
        feature_drift = self.analyze_feature_drift(current_features)
        prediction_drift = self.analyze_prediction_drift(current_predictions)
        
        # Generate report
        report = self.generate_drift_report(feature_drift, prediction_drift)
        
        # Export results
        files = self.export_results(report, feature_drift, prediction_drift)
        
        # Print summary
        logger.info("-"*60)
        logger.info("DRIFT DETECTION SUMMARY")
        logger.info("-"*60)
        logger.info(f"Overall Drift Detected: {report['overall_drift_detected']}")
        logger.info(f"Feature PSI - Avg: {feature_drift['avg_psi']:.4f}, Max: {feature_drift['max_psi']:.4f}")
        logger.info(f"Feature KS - Min p-value: {feature_drift['min_ks_pvalue']:.4f}")
        logger.info(f"Prediction PSI: {prediction_drift['prediction_psi']:.4f}")
        logger.info(f"Recommendations: {len(report['recommendations'])}")
        for rec in report['recommendations']:
            logger.info(f"  • {rec}")
        logger.info(f"Output: {list(files.keys())}")
        logger.info("="*60)
        
        return report
    
    def export_results(self, report: Dict, feature_drift: Dict, prediction_drift: Dict) -> Dict[str, str]:
        """Export drift detection results."""
        files = {}
        
        # JSON report
        json_path = Path(self.config['output_dir']) / 'drift_detection_results.json'
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        files['json_report'] = str(json_path)
        
        # Feature drift CSV
        if feature_drift.get('drift_summary'):
            csv_path = Path(self.config['output_dir']) / 'feature_drift_details.csv'
            pd.DataFrame(feature_drift['drift_summary']).to_csv(csv_path, index=False)
            files['csv_details'] = str(csv_path)
        
        # Drift plot
        plot_path = Path(self.config['output_dir']) / 'drift_analysis_v1.0.1.png'
        self.generate_drift_plot(str(plot_path), feature_drift, prediction_drift)
        files['drift_plot'] = str(plot_path)
        
        # Alert config template
        alert_path = Path(self.config['output_dir']) / 'drift_alert_config.yaml'
        with open(alert_path, 'w') as f:
            f.write(f"""# Drift Alerting Configuration
# Auto-generated: {datetime.now().isoformat()}

alerting:
  enabled: true
  check_interval_hours: 24
  
thresholds:
  psi_warning: 0.1
  psi_critical: {self.config['drift_threshold_psi']}
  ks_pvalue_critical: {self.config['ks_alpha']}
  
notifications:
  email: []
  slack_webhook: ""
  pagerduty_key: ""
""")
        files['alert_config'] = str(alert_path)
        
        return files


def main():
    """Main execution entry point."""
    start_time = datetime.now()
    logger.info(f"Drift detection started at {start_time.isoformat()}")
    
    detector = DriftDetector(CONFIG)
    report = detector.run_drift_detection()
    
    if 'error' in report:
        logger.error(f"Drift detection failed: {report['error']}")
        sys.exit(1)
    
    duration = (datetime.now() - start_time).total_seconds()
    logger.info(f"Completed in {duration:.2f}s")
    
    # Exit with appropriate code
    sys.exit(0 if not report['overall_drift_detected'] else 1)


if __name__ == '__main__':
    main()
