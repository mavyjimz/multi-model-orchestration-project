#!/usr/bin/env python3
"""
Test script for Phase 4.4 Inference API
========================================
Validates API endpoints before deployment
"""

import requests
import json
import time
from typing import List

API_BASE = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("\n[TEST] Health Check")
    print("-" * 40)
    response = requests.get(f"{API_BASE}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_root():
    """Test root endpoint"""
    print("\n[TEST] Root Endpoint")
    print("-" * 40)
    response = requests.get(f"{API_BASE}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_metadata():
    """Test model metadata endpoint"""
    print("\n[TEST] Model Metadata")
    print("-" * 40)
    response = requests.get(f"{API_BASE}/model/metadata")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_single_prediction():
    """Test single prediction endpoint"""
    print("\n[TEST] Single Prediction")
    print("-" * 40)
    
    test_texts = [
        "How do I reset my password?",
        "I can't access my account",
        "What are your business hours?",
        "I want to cancel my subscription"
    ]
    
    results = []
    for text in test_texts:
        payload = {"text": text, "request_id": f"test_{int(time.time() * 1000)}"}
        response = requests.post(f"{API_BASE}/predict", json=payload)
        result = response.json()
        results.append(result)
        print(f"Text: {text[:50]}...")
        print(f"Prediction: {result.get('prediction')}")
        print(f"Confidence: {result.get('confidence')}")
        print(f"Latency: {result.get('latency_ms')}ms")
        print()
    
    return all(r.get('prediction') for r in results)

def test_batch_prediction():
    """Test batch prediction endpoint"""
    print("\n[TEST] Batch Prediction")
    print("-" * 40)
    
    payload = {
        "texts": [
            "How do I reset my password?",
            "I can't access my account",
            "What are your business hours?"
        ],
        "request_id": "batch_test_001"
    }
    
    response = requests.post(f"{API_BASE}/predict/batch", json=payload)
    result = response.json()
    
    print(f"Status: {response.status_code}")
    print(f"Total Latency: {result.get('total_latency_ms')}ms")
    print(f"Average Latency: {result.get('average_latency_ms')}ms")
    print(f"Predictions: {len(result.get('predictions', []))}")
    
    for pred in result.get('predictions', []):
        print(f"  - {pred.get('prediction')} ({pred.get('confidence')})")
    
    return response.status_code == 200

def test_metrics():
    """Test metrics endpoint"""
    print("\n[TEST] API Metrics")
    print("-" * 40)
    response = requests.get(f"{API_BASE}/metrics")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def run_all_tests():
    """Run all API tests"""
    print("=" * 60)
    print("Phase 4.4: Inference API Test Suite")
    print("=" * 60)
    
    tests = [
        ("Root", test_root),
        ("Health", test_health),
        ("Metadata", test_metadata),
        ("Single Prediction", test_single_prediction),
        ("Batch Prediction", test_batch_prediction),
        ("Metrics", test_metrics)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"[FAILED] {name}: {str(e)}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
    
    print("-" * 60)
    print(f"Total: {passed}/{total} tests passed")
    print("=" * 60)
    
    return passed == total

if __name__ == "__main__":
    print("\nStarting API tests...")
    print("Ensure the API server is running: python3 scripts/p4.4-inference-api.py")
    print("\nWaiting 3 seconds for server to initialize...")
    time.sleep(3)
    
    success = run_all_tests()
    exit(0 if success else 1)
