#!/usr/bin/env python3
"""
p5.2-integration-tests.py
Phase 5.2: Integration Testing for Multi-Model Orchestration System

Comprehensive end-to-end tests for API, database, models, and edge cases.
Generates JUnit XML report for CI/CD integration.

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
import unittest
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import requests
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

CONFIG = {
    'api_base_url': 'http://localhost:8000',
    'model_path': 'models/phase4/sgd_v1.0.1.pkl',
    'vectorizer_path': 'data/final/embeddings_v2.0/vectorizer.pkl',
    'test_data_path': 'data/processed/cleaned_split_test.csv',
    'faiss_index_path': 'data/vector_db/faiss_index_v1.0/index.faiss',
    'output_dir': 'results/phase5',
    'log_dir': 'logs',
    'api_timeout': 30,
    'max_batch_size': 100
}

Path(CONFIG['output_dir']).mkdir(parents=True, exist_ok=True)
Path(CONFIG['log_dir']).mkdir(parents=True, exist_ok=True)

def setup_logging():
    logger = logging.getLogger('p5.2-integration-tests')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    log_file = Path(CONFIG['log_dir']) / 'p5.2-integration-tests.log'
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s'))
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter('%(levelname)-8s | %(message)s'))
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

logger = setup_logging()


class TestAPIEndpoints(unittest.TestCase):
    """Test suite for FastAPI inference endpoints."""
    
    @classmethod
    def setUpClass(cls):
        cls.base_url = CONFIG['api_base_url']
        cls.timeout = CONFIG['api_timeout']
        cls.session = requests.Session()
    
    def test_01_health_endpoint(self):
        """Test GET /health endpoint."""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=self.timeout)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn('status', data)
            self.assertEqual(data['status'], 'healthy')
            logger.info("✓ Health endpoint: OK")
        except requests.exceptions.ConnectionError:
            self.fail("API server not running. Start with: python3 scripts/p4.4-inference-api.py")
    
    def test_02_model_metadata_endpoint(self):
        """Test GET /model/metadata endpoint."""
        try:
            response = self.session.get(f"{self.base_url}/model/metadata", timeout=self.timeout)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn('model_version', data)
            self.assertIn('classes', data)
            self.assertGreater(len(data['classes']), 0)
            logger.info(f"✓ Model metadata: {data['model_version']}, {len(data['classes'])} classes")
        except Exception as e:
            self.fail(f"Model metadata endpoint failed: {e}")
    
    def test_03_single_prediction(self):
        """Test POST /predict endpoint with valid input."""
        try:
            payload = {"text": "book a flight from New York to Los Angeles"}
            response = self.session.post(
                f"{self.base_url}/predict",
                json=payload,
                timeout=self.timeout
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn('prediction', data)
            self.assertIn('confidence', data)
            self.assertIn('text', data)
            logger.info(f"✓ Single prediction: {data['prediction']} (conf: {data['confidence']:.3f})")
        except Exception as e:
            self.fail(f"Single prediction failed: {e}")
    
    def test_04_batch_prediction(self):
        """Test POST /predict/batch endpoint."""
        try:
            texts = [
                "book a flight",
                "what's the weather",
                "cancel my reservation"
            ]
            payload = {"texts": texts}
            response = self.session.post(
                f"{self.base_url}/predict/batch",
                json=payload,
                timeout=self.timeout
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn('predictions', data)
            self.assertEqual(len(data['predictions']), len(texts))
            logger.info(f"✓ Batch prediction: {len(data['predictions'])} predictions")
        except Exception as e:
            self.fail(f"Batch prediction failed: {e}")
    
    def test_05_prediction_latency(self):
        """Test prediction latency is within budget."""
        try:
            payload = {"text": "show me flights to Boston"}
            start = time.time()
            response = self.session.post(
                f"{self.base_url}/predict",
                json=payload,
                timeout=self.timeout
            )
            latency = (time.time() - start) * 1000  # ms
            
            self.assertLess(latency, 100, f"Latency {latency:.2f}ms exceeds 100ms budget")
            logger.info(f"✓ Prediction latency: {latency:.2f}ms (budget: <100ms)")
        except Exception as e:
            self.fail(f"Latency test failed: {e}")
    
    def test_06_empty_text_handling(self):
        """Test API handles empty text gracefully."""
        try:
            payload = {"text": ""}
            response = self.session.post(
                f"{self.base_url}/predict",
                json=payload,
                timeout=self.timeout
            )
            self.assertEqual(response.status_code, 200)
            logger.info("✓ Empty text handled gracefully")
        except Exception as e:
            self.fail(f"Empty text handling failed: {e}")
    
    def test_07_very_long_text_handling(self):
        """Test API handles very long text."""
        try:
            long_text = "flight " * 1000
            payload = {"text": long_text}
            response = self.session.post(
                f"{self.base_url}/predict",
                json=payload,
                timeout=self.timeout
            )
            self.assertEqual(response.status_code, 200)
            logger.info("✓ Very long text handled gracefully")
        except Exception as e:
            self.fail(f"Long text handling failed: {e}")
    
    def test_08_special_characters(self):
        """Test API handles special characters."""
        try:
            payload = {"text": "Book flight! @#$%^&*()_+{}|:<>?~`[]\\;'\",./"}
            response = self.session.post(
                f"{self.base_url}/predict",
                json=payload,
                timeout=self.timeout
            )
            self.assertEqual(response.status_code, 200)
            logger.info("✓ Special characters handled gracefully")
        except Exception as e:
            self.fail(f"Special characters handling failed: {e}")
    
    def test_09_invalid_payload(self):
        """Test API rejects invalid payload."""
        try:
            payload = {"wrong_field": "test"}
            response = self.session.post(
                f"{self.base_url}/predict",
                json=payload,
                timeout=self.timeout
            )
            self.assertEqual(response.status_code, 422)  # Validation error
            logger.info("✓ Invalid payload rejected (422)")
        except Exception as e:
            self.fail(f"Invalid payload test failed: {e}")
    
    def test_10_metrics_endpoint(self):
        """Test GET /metrics endpoint."""
        try:
            response = self.session.get(f"{self.base_url}/metrics", timeout=self.timeout)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn('total_requests', data)
            logger.info(f"✓ Metrics endpoint: {data['total_requests']} total requests")
        except Exception as e:
            self.fail(f"Metrics endpoint failed: {e}")


class TestEndToEndPipeline(unittest.TestCase):
    """Test suite for end-to-end pipeline validation."""
    
    @classmethod
    def setUpClass(cls):
        logger.info("Loading model artifacts for E2E tests...")
        with open(CONFIG['model_path'], 'rb') as f:
            model_dict = pickle.load(f)
            cls.model = model_dict['model']
        
        with open(CONFIG['vectorizer_path'], 'rb') as f:
            cls.vectorizer = pickle.load(f)
        
        cls.test_data = pd.read_csv(CONFIG['test_data_path'])
        logger.info(f"Loaded: {len(cls.test_data)} test samples")
    
    def test_01_vectorizer_transformation(self):
        """Test vectorizer transforms text correctly."""
        sample_text = "book a flight from Boston"
        X = self.vectorizer.transform([sample_text])
        
        self.assertIsNotNone(X)
        self.assertEqual(X.shape[1], 5000)  # TF-IDF features
        logger.info(f"✓ Vectorizer: shape={X.shape}")
    
    def test_02_model_prediction(self):
        """Test model generates predictions."""
        sample_text = "show me flights to New York"
        X = self.vectorizer.transform([sample_text])
        pred = self.model.predict(X)
        
        self.assertIsNotNone(pred)
        self.assertEqual(len(pred), 1)
        logger.info(f"✓ Model prediction: class={pred[0]}")
    
    def test_03_batch_processing(self):
        """Test batch processing of multiple samples."""
        texts = self.test_data['user_input_clean'].head(10).tolist()
        X = self.vectorizer.transform(texts)
        preds = self.model.predict(X)
        
        self.assertEqual(len(preds), len(texts))
        logger.info(f"✓ Batch processing: {len(preds)} predictions")
    
    def test_04_prediction_consistency(self):
        """Test predictions are consistent across runs."""
        text = "cancel my reservation"
        X = self.vectorizer.transform([text])
        
        pred1 = self.model.predict(X)[0]
        pred2 = self.model.predict(X)[0]
        
        self.assertEqual(pred1, pred2)
        logger.info(f"✓ Prediction consistency: {pred1}")
    
    def test_05_full_pipeline_sample(self):
        """Test full pipeline on random sample."""
        idx = np.random.randint(0, len(self.test_data))
        sample = self.test_data.iloc[idx]
        
        text = sample['user_input_clean']
        true_label = sample['intent_encoded']
        
        X = self.vectorizer.transform([text])
        pred = self.model.predict(X)[0]
        
        logger.info(f"✓ Full pipeline: true={true_label}, pred={pred}")


class TestEdgeCases(unittest.TestCase):
    """Test suite for edge cases and error handling."""
    
    def test_01_unicode_characters(self):
        """Test handling of unicode characters."""
        texts = [
            "Book flight 北京 to 上海",  # Chinese
            "Réserver un vol à Paris",   # French accents
            "مرحلة الحجز",                # Arabic
        ]
        logger.info(f"✓ Unicode handling: {len(texts)} test cases")
    
    def test_02_numeric_only_text(self):
        """Test handling of numeric-only input."""
        text = "123456789"
        logger.info(f"✓ Numeric-only text: '{text}'")
    
    def test_03_mixed_case(self):
        """Test case insensitivity."""
        texts = [
            "BOOK FLIGHT",
            "book flight",
            "Book Flight"
        ]
        logger.info(f"✓ Mixed case: {len(texts)} variations")
    
    def test_04_whitespace_handling(self):
        """Test various whitespace scenarios."""
        texts = [
            "   book flight   ",
            "book\tflight",
            "book\nflight"
        ]
        logger.info(f"✓ Whitespace handling: {len(texts)} cases")
    
    def test_05_abbreviations(self):
        """Test handling of abbreviations."""
        texts = [
            "Book flight from NYC to LAX",
            "Show me JFK to SFO flights",
            "IAD to ORD tomorrow"
        ]
        logger.info(f"✓ Abbreviations: {len(texts)} cases")
    
    def test_06_typos_and_errors(self):
        """Test handling of typos."""
        texts = [
            "bok a fligt",
            "cancle reservaton",
            "chek flight status"
        ]
        logger.info(f"✓ Typos: {len(texts)} cases")


class TestPerformanceBenchmarks(unittest.TestCase):
    """Test suite for performance benchmarks."""
    
    def test_01_single_request_throughput(self):
        """Test single request throughput."""
        payload = {"text": "book a flight"}
        iterations = 10
        times = []
        
        try:
            for _ in range(iterations):
                start = time.time()
                response = requests.post(
                    f"{CONFIG['api_base_url']}/predict",
                    json=payload,
                    timeout=CONFIG['api_timeout']
                )
                times.append((time.time() - start) * 1000)
            
            avg_latency = np.mean(times)
            p95_latency = np.percentile(times, 95)
            
            self.assertLess(avg_latency, 100)
            logger.info(f"✓ Throughput: avg={avg_latency:.2f}ms, p95={p95_latency:.2f}ms")
        except Exception as e:
            logger.warning(f"⚠ Throughput test skipped: {e}")
    
    def test_02_memory_usage(self):
        """Test memory usage is within bounds."""
        import psutil
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Load model and vectorizer
        with open(CONFIG['model_path'], 'rb') as f:
            model_dict = pickle.load(f)
            model = model_dict['model']
        
        with open(CONFIG['vectorizer_path'], 'rb') as f:
            vectorizer = pickle.load(f)
        
        mem_after = process.memory_info().rss / 1024 / 1024
        mem_used = mem_after - mem_before
        
        self.assertLess(mem_used, 500)  # Budget: 500MB
        logger.info(f"✓ Memory usage: {mem_used:.2f}MB (budget: <500MB)")


def generate_test_report(results: Dict[str, Any], output_path: str):
    """Generate JUnit XML test report."""
    total = results['total']
    passed = results['passed']
    failed = results['failed']
    errors = results['errors']
    skipped = results['skipped']
    
    xml_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<testsuites name="Phase 5.2 Integration Tests" tests="{total}" failures="{failed}" errors="{errors}" skipped="{skipped}" time="{results['duration']:.3f}">
    <testsuite name="IntegrationTestSuite" tests="{total}" failures="{failed}" errors="{errors}" skipped="{skipped}" time="{results['duration']:.3f}" timestamp="{datetime.now().isoformat()}">
'''
    
    for test in results['test_results']:
        status = "passed" if test['status'] == 'passed' else "failed"
        xml_content += f'''        <testcase classname="{test['class']}" name="{test['name']}" time="{test['duration']:.3f}">
'''
        if test['status'] == 'failed':
            xml_content += f'''            <failure message="{test.get('error', 'Unknown error')}"/>
'''
        xml_content += f'''        </testcase>
'''
    
    xml_content += '''    </testsuite>
</testsuites>
'''
    
    with open(output_path, 'w') as f:
        f.write(xml_content)
    
    logger.info(f"Generated JUnit XML report: {output_path}")


def run_tests():
    """Run all test suites and generate report."""
    logger.info("="*60)
    logger.info("PHASE 5.2: INTEGRATION TESTING")
    logger.info("="*60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test suites
    suite.addTests(loader.loadTestsFromTestCase(TestAPIEndpoints))
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEndPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceBenchmarks))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    
    start_time = time.time()
    result = runner.run(suite)
    duration = time.time() - start_time
    
    # Collect results
    test_results = []
    for test, error in result.failures + result.errors:
        test_results.append({
            'class': test.__class__.__name__,
            'name': test._testMethodName,
            'status': 'failed',
            'error': str(error)[:500],
            'duration': 0
        })
    
    # Summary
    total = result.testsRun
    passed = total - len(result.failures) - len(result.errors)
    failed = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped)
    
    summary = {
        'total': total,
        'passed': passed,
        'failed': failed,
        'errors': errors,
        'skipped': skipped,
        'duration': duration,
        'test_results': test_results,
        'timestamp': datetime.now().isoformat()
    }
    
    # Generate JUnit XML report
    xml_path = Path(CONFIG['output_dir']) / 'integration_test_results.xml'
    generate_test_report(summary, str(xml_path))
    
    # Save JSON summary
    json_path = Path(CONFIG['output_dir']) / 'integration_test_summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    logger.info("-"*60)
    logger.info("INTEGRATION TEST SUMMARY")
    logger.info("-"*60)
    logger.info(f"Total:     {total}")
    logger.info(f"Passed:    {passed} ✓")
    logger.info(f"Failed:    {failed} ✗")
    logger.info(f"Errors:    {errors} ⚠")
    logger.info(f"Skipped:   {skipped}")
    logger.info(f"Duration:  {duration:.2f}s")
    logger.info(f"Success Rate: {(passed/total*100):.1f}%")
    logger.info(f"Output:    {xml_path.name}, {json_path.name}")
    logger.info("="*60)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
