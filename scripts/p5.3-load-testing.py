#!/usr/bin/env python3
"""
p5.3-load-testing.py
Phase 5.3: Load & Stress Testing for Multi-Model Orchestration System

Tests model inference latency and throughput under concurrent load
without requiring API server. Direct model/vectorizer benchmarking.

Author: Mavyjimz
Date: March 2026
Version: 1.0.0
"""

import os
import sys
import json
import time
import pickle
import logging
import warnings
import statistics
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

CONFIG = {
    'model_path': 'models/phase4/sgd_v1.0.1.pkl',
    'vectorizer_path': 'data/final/embeddings_v2.0/vectorizer.pkl',
    'test_data_path': 'data/processed/cleaned_split_test.csv',
    'output_dir': 'results/phase5',
    'log_dir': 'logs',
    'log_level': 'INFO',
    'concurrency_levels': [1, 5, 10, 25, 50],  # Simulated concurrent requests
    'requests_per_level': 100,  # Requests to send per concurrency level
    'warmup_requests': 10,  # Warmup requests before timing
    'timeout_seconds': 300  # Max time for entire test suite
}

Path(CONFIG['output_dir']).mkdir(parents=True, exist_ok=True)
Path(CONFIG['log_dir']).mkdir(parents=True, exist_ok=True)

def setup_logging():
    logger = logging.getLogger('p5.3-load-testing')
    logger.setLevel(getattr(logging, CONFIG['log_level']))
    logger.handlers = []
    
    log_file = Path(CONFIG['log_dir']) / 'p5.3-load-testing.log'
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s'))
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter('%(levelname)-8s | %(message)s'))
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

logger = setup_logging()


class LoadTester:
    """Load and stress tester for model inference performance."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.vectorizer = None
        self.test_texts = []
        self.results = {}
        self._lock = threading.Lock()
        
    def load_artifacts(self) -> bool:
        """Load model and vectorizer for testing."""
        try:
            logger.info(f"Loading model from {self.config['model_path']}")
            with open(self.config['model_path'], 'rb') as f:
                model_dict = pickle.load(f)
                self.model = model_dict.get('model', model_dict) if isinstance(model_dict, dict) else model_dict
            
            logger.info(f"Loading vectorizer from {self.config['vectorizer_path']}")
            with open(self.config['vectorizer_path'], 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            logger.info(f"Loading test data from {self.config['test_data_path']}")
            test_df = pd.read_csv(self.config['test_data_path'])
            self.test_texts = test_df['user_input_clean'].astype(str).tolist()
            
            logger.info(f"Loaded: {len(self.test_texts)} test samples")
            return True
        except Exception as e:
            logger.error(f"Error loading artifacts: {e}")
            return False
    
    def _single_inference(self, text: str) -> Dict[str, Any]:
        """Execute single inference and measure latency."""
        start = time.perf_counter()
        try:
            X = self.vectorizer.transform([text])
            pred = self.model.predict(X)[0]
            latency_ms = (time.perf_counter() - start) * 1000
            return {'success': True, 'latency_ms': latency_ms, 'prediction': pred}
        except Exception as e:
            return {'success': False, 'error': str(e), 'latency_ms': (time.perf_counter() - start) * 1000}
    
    def _concurrent_test(self, concurrency: int, n_requests: int) -> Dict[str, Any]:
        """Run concurrent inference test."""
        logger.info(f"Testing concurrency={concurrency}, requests={n_requests}")
        
        # Select random test texts
        texts = np.random.choice(self.test_texts, size=min(n_requests, len(self.test_texts)), replace=True).tolist()
        
        # Warmup
        for _ in range(self.config['warmup_requests']):
            self._single_inference(texts[0])
        
        # Concurrent execution
        latencies = []
        successes = 0
        errors = []
        
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(self._single_inference, text) for text in texts]
            
            for future in as_completed(futures):
                result = future.result()
                if result['success']:
                    successes += 1
                    latencies.append(result['latency_ms'])
                else:
                    errors.append(result.get('error', 'Unknown error'))
        
        # Calculate metrics
        if latencies:
            return {
                'concurrency': concurrency,
                'total_requests': n_requests,
                'successful_requests': successes,
                'failed_requests': n_requests - successes,
                'success_rate': successes / n_requests * 100,
                'latency_avg_ms': statistics.mean(latencies),
                'latency_p50_ms': statistics.median(latencies),
                'latency_p95_ms': np.percentile(latencies, 95),
                'latency_p99_ms': np.percentile(latencies, 99),
                'latency_min_ms': min(latencies),
                'latency_max_ms': max(latencies),
                'throughput_rps': successes / (max(latencies) / 1000) if latencies else 0,
                'errors': errors[:5]  # Limit error samples
            }
        else:
            return {
                'concurrency': concurrency,
                'total_requests': n_requests,
                'successful_requests': 0,
                'failed_requests': n_requests,
                'success_rate': 0,
                'error': 'All requests failed',
                'errors': errors[:5]
            }
    
    def run_load_tests(self) -> Dict[str, Any]:
        """Execute full load test suite."""
        logger.info("="*60)
        logger.info("PHASE 5.3: LOAD & STRESS TESTING")
        logger.info("="*60)
        
        if not self.load_artifacts():
            return {'error': 'Failed to load artifacts'}
        
        results = []
        start_time = time.time()
        
        for concurrency in self.config['concurrency_levels']:
            # Check timeout
            if time.time() - start_time > self.config['timeout_seconds']:
                logger.warning(f"Timeout reached, skipping concurrency={concurrency}")
                break
            
            result = self._concurrent_test(
                concurrency=concurrency,
                n_requests=self.config['requests_per_level']
            )
            results.append(result)
            logger.info(f"✓ concurrency={concurrency}: avg={result['latency_avg_ms']:.2f}ms, p95={result['latency_p95_ms']:.2f}ms, success={result['success_rate']:.1f}%")
        
        self.results = {
            'test_config': {
                'concurrency_levels': self.config['concurrency_levels'],
                'requests_per_level': self.config['requests_per_level'],
                'warmup_requests': self.config['warmup_requests']
            },
            'model_info': {
                'version': 'sgd_v1.0.1',
                'features': 5000,
                'classes': 41
            },
            'results': results,
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': time.time() - start_time
        }
        
        return self.results
    
    def generate_performance_plot(self, output_path: str) -> str:
        """Generate latency vs concurrency plot."""
        if not self.results.get('results'):
            return ''
        
        concurrencies = [r['concurrency'] for r in self.results['results']]
        avg_latencies = [r['latency_avg_ms'] for r in self.results['results']]
        p95_latencies = [r['latency_p95_ms'] for r in self.results['results']]
        throughputs = [r['throughput_rps'] for r in self.results['results']]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Latency plot
        ax1.plot(concurrencies, avg_latencies, 'b-o', label='Avg Latency', marker='o')
        ax1.plot(concurrencies, p95_latencies, 'r--s', label='P95 Latency', marker='s')
        ax1.axhline(y=100, color='gray', linestyle=':', label='Budget: 100ms')
        ax1.set_xlabel('Concurrent Requests')
        ax1.set_ylabel('Latency (ms)')
        ax1.set_title('Latency vs Concurrency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Throughput plot
        ax2.plot(concurrencies, throughputs, 'g-^', label='Throughput', marker='^')
        ax2.set_xlabel('Concurrent Requests')
        ax2.set_ylabel('Requests/Second')
        ax2.set_title('Throughput vs Concurrency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved performance plot: {output_path}")
        return output_path
    
    def export_results(self) -> Dict[str, str]:
        """Export test results to files."""
        files = {}
        
        # JSON report
        json_path = Path(self.config['output_dir']) / 'load_test_results.json'
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        files['json_report'] = str(json_path)
        
        # CSV summary
        csv_path = Path(self.config['output_dir']) / 'load_test_summary.csv'
        if self.results.get('results'):
            df = pd.DataFrame(self.results['results'])
            df.to_csv(csv_path, index=False)
            files['csv_summary'] = str(csv_path)
        
        # Performance plot
        plot_path = Path(self.config['output_dir']) / 'performance_plot_v1.0.1.png'
        self.generate_performance_plot(str(plot_path))
        files['performance_plot'] = str(plot_path)
        
        # Benchmark report
        report_path = Path(self.config['output_dir']) / 'benchmark_report_v1.0.1.txt'
        with open(report_path, 'w') as f:
            f.write("LOAD & STRESS TESTING BENCHMARK REPORT\n")
            f.write("="*50 + "\n\n")
            f.write(f"Model: SGD v1.0.1\n")
            f.write(f"Features: 5000 TF-IDF dimensions\n")
            f.write(f"Classes: 41 intent classes\n\n")
            f.write("PERFORMANCE METRICS:\n")
            f.write("-"*50 + "\n")
            for r in self.results.get('results', []):
                f.write(f"\nConcurrency: {r['concurrency']}\n")
                f.write(f"  Success Rate: {r['success_rate']:.1f}%\n")
                f.write(f"  Avg Latency:  {r['latency_avg_ms']:.2f} ms\n")
                f.write(f"  P95 Latency:  {r['latency_p95_ms']:.2f} ms\n")
                f.write(f"  P99 Latency:  {r['latency_p99_ms']:.2f} ms\n")
                f.write(f"  Throughput:   {r['throughput_rps']:.1f} RPS\n")
            f.write("\n" + "="*50 + "\n")
            f.write(f"Test Duration: {self.results.get('duration_seconds', 0):.2f} seconds\n")
        files['text_report'] = str(report_path)
        
        return files
    
    def validate_budgets(self) -> Dict[str, bool]:
        """Validate results against performance budgets."""
        budgets = {
            'latency_p95_under_100ms': True,
            'success_rate_above_99': True,
            'throughput_scaling': True
        }
        
        for r in self.results.get('results', []):
            if r['latency_p95_ms'] > 100:
                budgets['latency_p95_under_100ms'] = False
                logger.warning(f"⚠ P95 latency {r['latency_p95_ms']:.2f}ms exceeds 100ms budget at concurrency={r['concurrency']}")
            
            if r['success_rate'] < 99:
                budgets['success_rate_above_99'] = False
                logger.warning(f"⚠ Success rate {r['success_rate']:.1f}% below 99% target at concurrency={r['concurrency']}")
        
        # Check throughput scaling (should increase or plateau, not decrease sharply)
        throughputs = [r['throughput_rps'] for r in self.results.get('results', [])]
        if len(throughputs) >= 2:
            if throughputs[-1] < throughputs[0] * 0.5:  # More than 50% drop
                budgets['throughput_scaling'] = False
                logger.warning("⚠ Throughput degradation detected at high concurrency")
        
        return budgets


def main():
    """Main execution entry point."""
    start_time = datetime.now()
    logger.info(f"Load testing started at {start_time.isoformat()}")
    
    tester = LoadTester(CONFIG)
    results = tester.run_load_tests()
    
    if 'error' in results:
        logger.error(f"Load testing failed: {results['error']}")
        sys.exit(1)
    
    # Export results
    files = tester.export_results()
    
    # Validate budgets
    budgets = tester.validate_budgets()
    
    # Print summary
    logger.info("-"*60)
    logger.info("LOAD TEST SUMMARY")
    logger.info("-"*60)
    
    for r in results.get('results', []):
        status = "✓" if r['latency_p95_ms'] <= 100 and r['success_rate'] >= 99 else "⚠"
        logger.info(f"{status} concurrency={r['concurrency']:2d}: avg={r['latency_avg_ms']:6.2f}ms, p95={r['latency_p95_ms']:6.2f}ms, {r['success_rate']:5.1f}% success")
    
    logger.info(f"\nBudget Validation:")
    for budget, passed in budgets.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"  {budget}: {status}")
    
    logger.info(f"\nOutput files: {list(files.keys())}")
    logger.info(f"Duration: {results.get('duration_seconds', 0):.2f} seconds")
    logger.info("="*60)
    
    # Exit with appropriate code
    all_passed = all(budgets.values())
    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
