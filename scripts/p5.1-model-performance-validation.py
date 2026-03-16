#!/usr/bin/env python3
"""
p5.1-model-performance-validation.py
Phase 5.1: Model Performance Validation
"""

import os
import sys
import json
import pickle
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)

warnings.filterwarnings('ignore')

CONFIG = {
    'model_path': 'models/phase4/sgd_v1.0.1.pkl',
    'vectorizer_path': 'data/final/embeddings_v2.0/vectorizer.pkl',
    'test_data_path': 'data/processed/cleaned_split_test.csv',
    'baseline_results_path': 'results/phase4/validation_results_v1.0.1.json',
    'output_dir': 'results/phase5',
    'log_dir': 'logs',
    'log_level': 'INFO',
    'weak_class_threshold': 0.5,
    'min_samples_for_analysis': 5
}

Path(CONFIG['output_dir']).mkdir(parents=True, exist_ok=True)
Path(CONFIG['log_dir']).mkdir(parents=True, exist_ok=True)

def setup_logging(script_name):
    logger = logging.getLogger(script_name)
    logger.setLevel(getattr(logging, CONFIG['log_level']))
    logger.handlers = []
    
    log_file = Path(CONFIG['log_dir']) / f"{script_name}.log"
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s'))
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(levelname)-8s | %(message)s'))
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

logger = setup_logging('p5.1-model-performance-validation')

class ModelValidator:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.vectorizer = None
        self.test_data = None
        self.y_true = None
        self.y_pred = None
        self.metrics = {}
        self.class_mapper = None
        
    def load_artifacts(self):
        try:
            logger.info(f"Loading model from {self.config['model_path']}")
            with open(self.config['model_path'], 'rb') as f:
                model_dict = pickle.load(f)
            
            # Extract model from dictionary wrapper
            if isinstance(model_dict, dict):
                self.model = model_dict['model']
                self.class_mapper = model_dict.get('class_mapper', None)
                logger.info("Model extracted from dictionary wrapper")
            else:
                self.model = model_dict
            
            logger.info(f"Loading vectorizer from {self.config['vectorizer_path']}")
            with open(self.config['vectorizer_path'], 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            logger.info(f"Loading test data from {self.config['test_data_path']}")
            self.test_data = pd.read_csv(self.config['test_data_path'])
            
            # Use 'intent_encoded' column
            self.y_true = self.test_data['intent_encoded'].values
            logger.info(f"Loaded: {len(self.test_data)} samples, {len(np.unique(self.y_true))} classes")
            return True
        except Exception as e:
            logger.error(f"Error loading artifacts: {e}")
            return False
    
    def generate_predictions(self):
        try:
            logger.info("Vectorizing test texts...")
            X_test = self.vectorizer.transform(self.test_data['user_input_clean'].astype(str).tolist())
            
            logger.info("Generating predictions...")
            self.y_pred = self.model.predict(X_test)
            logger.info(f"Generated {len(self.y_pred)} predictions")
            return True
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            return False
    
    def compute_metrics(self):
        logger.info("Computing metrics...")
        
        accuracy = accuracy_score(self.y_true, self.y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            self.y_true, self.y_pred, average=None, zero_division=0
        )
        
        macro = precision_recall_fscore_support(self.y_true, self.y_pred, average='macro', zero_division=0)
        weighted = precision_recall_fscore_support(self.y_true, self.y_pred, average='weighted', zero_division=0)
        
        classes = sorted(list(set(self.y_true) | set(self.y_pred)))
        per_class = {}
        for i, cls in enumerate(classes):
            if i < len(precision):
                per_class[int(cls)] = {
                    'precision': float(precision[i]),
                    'recall': float(recall[i]),
                    'f1_score': float(f1[i]),
                    'support': int(support[i])
                }
        
        self.metrics = {
            'overall': {
                'accuracy': float(accuracy),
                'precision_macro': float(macro[0]),
                'recall_macro': float(macro[1]),
                'f1_macro': float(macro[2]),
                'precision_weighted': float(weighted[0]),
                'recall_weighted': float(weighted[1]),
                'f1_weighted': float(weighted[2]),
                'total_samples': int(len(self.y_true)),
                'n_classes': int(len(classes))
            },
            'per_class': per_class,
            'confusion_matrix': confusion_matrix(self.y_true, self.y_pred).tolist(),
            'timestamp': datetime.now().isoformat(),
            'model_version': 'sgd_v1.0.1'
        }
        
        logger.info(f"Accuracy: {accuracy:.4f}, F1-Macro: {macro[2]:.4f}")
        return self.metrics
    
    def identify_weak_classes(self):
        weak = []
        for cls_id, m in self.metrics['per_class'].items():
            if m['support'] >= self.config['min_samples_for_analysis'] and m['f1_score'] < self.config['weak_class_threshold']:
                weak.append({
                    'class_id': cls_id,
                    'f1_score': m['f1_score'],
                    'precision': m['precision'],
                    'recall': m['recall'],
                    'support': m['support']
                })
        return sorted(weak, key=lambda x: x['f1_score'])
    
    def generate_confusion_plot(self, output_path):
        cm = np.array(self.metrics['confusion_matrix'])
        classes = list(self.metrics['per_class'].keys())
        
        if len(classes) > 20:
            top = sorted(self.metrics['per_class'].items(), key=lambda x: x[1]['support'], reverse=True)[:20]
            classes = [c[0] for c in top]
            idx = [i for i, c in enumerate(sorted(self.metrics['per_class'].keys())) if c in classes]
            cm = cm[np.ix_(idx, idx)]
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix - SGD v1.0.1')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved confusion matrix: {output_path}")
    
    def compare_baseline(self):
        path = self.config['baseline_results_path']
        if not os.path.exists(path):
            logger.warning(f"Baseline not found: {path}")
            return None
        
        try:
            with open(path, 'r') as f:
                base = json.load(f)
            
            return {
                'baseline_accuracy': base.get('overall', {}).get('accuracy'),
                'current_accuracy': self.metrics['overall']['accuracy'],
                'accuracy_delta': self.metrics['overall']['accuracy'] - base.get('overall', {}).get('accuracy', 0),
                'baseline_f1_macro': base.get('overall', {}).get('f1_macro'),
                'current_f1_macro': self.metrics['overall']['f1_macro'],
                'f1_delta': self.metrics['overall']['f1_macro'] - base.get('overall', {}).get('f1_macro', 0)
            }
        except Exception as e:
            logger.error(f"Baseline comparison error: {e}")
            return None
    
    def export_results(self):
        files = {}
        
        # JSON report
        json_path = Path(self.config['output_dir']) / 'validation_report_v1.0.1.json'
        with open(json_path, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
        files['json_report'] = str(json_path)
        
        # CSV metrics
        csv_path = Path(self.config['output_dir']) / 'per_class_metrics_v1.0.1.csv'
        pd.DataFrame.from_dict(self.metrics['per_class'], orient='index').to_csv(csv_path)
        files['csv_metrics'] = str(csv_path)
        
        # Confusion plot
        plot_path = Path(self.config['output_dir']) / 'confusion_matrix_v1.0.1.png'
        self.generate_confusion_plot(str(plot_path))
        files['confusion_plot'] = str(plot_path)
        
        # Text report
        txt_path = Path(self.config['output_dir']) / 'classification_report_v1.0.1.txt'
        with open(txt_path, 'w') as f:
            f.write(classification_report(self.y_true, self.y_pred))
        files['text_report'] = str(txt_path)
        
        # Weak classes
        weak = self.identify_weak_classes()
        weak_path = Path(self.config['output_dir']) / 'weak_classes_analysis_v1.0.1.json'
        with open(weak_path, 'w') as f:
            json.dump({'weak_classes': weak, 'count': len(weak)}, f, indent=2)
        files['weak_classes'] = str(weak_path)
        
        return files
    
    def run(self):
        logger.info("="*60)
        logger.info("PHASE 5.1: MODEL PERFORMANCE VALIDATION")
        logger.info("="*60)
        
        if not self.load_artifacts():
            return False
        
        if not self.generate_predictions():
            return False
        
        self.compute_metrics()
        
        baseline = self.compare_baseline()
        if baseline:
            self.metrics['baseline_comparison'] = baseline
        
        files = self.export_results()
        
        logger.info("-"*60)
        logger.info("VALIDATION COMPLETE")
        logger.info(f"Accuracy:  {self.metrics['overall']['accuracy']:.4f}")
        logger.info(f"F1-Macro:  {self.metrics['overall']['f1_macro']:.4f}")
        logger.info(f"Samples:   {self.metrics['overall']['total_samples']}")
        if baseline:
            logger.info(f"Δ Accuracy: {baseline['accuracy_delta']:+.4f}")
            logger.info(f"Δ F1-Macro: {baseline['f1_delta']:+.4f}")
        logger.info(f"Output: {list(files.keys())}")
        logger.info("="*60)
        
        return True

def main():
    start = datetime.now()
    validator = ModelValidator(CONFIG)
    success = validator.run()
    duration = (datetime.now() - start).total_seconds()
    logger.info(f"Completed in {duration:.2f}s")
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
