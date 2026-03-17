#!/usr/bin/env python3
"""
Phase 5.7: Final Validation & Handoff
Complete validation of Phase 5, API testing, and release preparation
"""

import os
import sys
import json
import subprocess
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/p5.7-final-validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def fix_p44_api_loading():
    """
    Fix p4.4-inference-api.py model loading issue
    Extract model from dictionary wrapper
    """
    logger.info("Fixing p4.4-inference-api.py model loading...")
    
    file_path = 'scripts/p4.4-inference-api.py'
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if already fixed
    if 'Extract model from dictionary wrapper' in content:
        logger.info("p4.4 already fixed, skipping...")
        return True
    
    # Find and replace the load_model method
    old_code = """with open(self.model_path, 'rb') as f:
            loaded = pickle.load(f)"""
    
    new_code = """with open(self.model_path, 'rb') as f:
            loaded = pickle.load(f)
        
        # Extract model from dictionary wrapper (consistent with training scripts)
        if isinstance(loaded, dict) and 'model' in loaded:
            self.model = loaded['model']
            self.class_mapper = loaded.get('class_mapper')
        else:
            self.model = loaded
            self.class_mapper = None
        
        # Safe access to classes attribute
        if hasattr(self.model, 'classes_'):
            self.classes = self.model.classes_.tolist()
        elif hasattr(self.model, 'class_mapper') and self.model.class_mapper:
            self.classes = list(self.model.class_mapper.idx_to_label.values())
        else:
            self.classes = []"""
    
    if old_code in content:
        content = content.replace(old_code, new_code)
        
        with open(file_path, 'w') as f:
            f.write(content)
        
        logger.info("✓ p4.4-inference-api.py fixed successfully")
        return True
    else:
        logger.warning("⚠ Could not find model loading code to fix")
        return False


def test_api_server():
    """Test API server startup and endpoints"""
    logger.info("Testing API server...")
    
    # Start API server in background
    api_process = subprocess.Popen(
        ['python3', 'scripts/p4.4-inference-api.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for startup
    import time
    time.sleep(5)
    
    # Test health endpoint
    try:
        import requests
        response = requests.get('http://localhost:8000/health', timeout=5)
        if response.status_code == 200:
            logger.info("✓ API health endpoint responding")
            api_success = True
        else:
            logger.error(f"✗ API health check failed: {response.status_code}")
            api_success = False
    except Exception as e:
        logger.error(f"✗ API not responding: {str(e)}")
        api_success = False
    
    # Terminate API server
    api_process.terminate()
    api_process.wait(timeout=5)
    
    return api_success


def generate_phase5_summary():
    """Generate comprehensive Phase 5 summary report"""
    logger.info("Generating Phase 5 summary report...")
    
    # Collect results from all Phase 5 scripts
    results = {
        'phase': '5',
        'title': 'Model Validation & Testing',
        'timestamp': datetime.now().isoformat(),
        'components': {}
    }
    
    # p5.1: Model Performance Validation
    p51_path = 'results/phase5/validation_report_v1.0.1.json'
    if os.path.exists(p51_path):
        with open(p51_path, 'r') as f:
            results['components']['p5.1_performance_validation'] = json.load(f)
    
    # p5.2: Integration Testing
    p52_path = 'results/phase5/integration_test_summary.json'
    if os.path.exists(p52_path):
        with open(p52_path, 'r') as f:
            results['components']['p5.2_integration_testing'] = json.load(f)
    
    # p5.3: Load Testing
    p53_path = 'results/phase5/load_test_results.json'
    if os.path.exists(p53_path):
        with open(p53_path, 'r') as f:
            results['components']['p5.3_load_testing'] = json.load(f)
    
    # p5.4: Drift Detection
    p54_path = 'results/phase5/drift_detection_results.json'
    if os.path.exists(p54_path):
        with open(p54_path, 'r') as f:
            results['components']['p5.4_drift_detection'] = json.load(f)
    
    # p5.5: A/B Testing
    p55_path = 'results/phase5/ab_test_analysis.json'
    if os.path.exists(p55_path):
        with open(p55_path, 'r') as f:
            results['components']['p5.5_ab_testing'] = json.load(f)
    
    # Save summary
    output_path = 'results/phase5/phase5_summary.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"✓ Phase 5 summary saved to {output_path}")
    return results


def create_release_tag():
    """Create git release tag for v1.0-phase5-complete"""
    logger.info("Creating release tag v1.0-phase5-complete...")
    
    try:
        # Create tag
        subprocess.run(['git', 'tag', '-a', 'v1.0-phase5-complete', '-m', 
                       'Phase 5 Complete: Model Validation & Testing'],
                      check=True, capture_output=True)
        
        # Push tag
        subprocess.run(['git', 'push', 'origin', 'v1.0-phase5-complete'],
                      check=True, capture_output=True)
        
        logger.info("✓ Release tag created and pushed")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Failed to create tag: {str(e)}")
        return False


def generate_handoff_documentation():
    """Generate Phase 6 handoff documentation"""
    logger.info("Generating Phase 6 handoff documentation...")
    
    handoff_doc = """# PHASE 6: PRODUCTION DEPLOYMENT - HANDOFF DOCUMENTATION

Generated: {timestamp}
Project: Multi-Model Orchestration System
Repository: https://github.com/mavyjimz/multi-model-orchestration-project

## PHASE 5 COMPLETION STATUS

### Completed Components:
- p5.1: Model Performance Validation - COMPLETE
- p5.2: Integration Testing - COMPLETE (core validation passed)
- p5.3: Load & Stress Testing - COMPLETE (P95 < 100ms, 100% success)
- p5.4: Model Drift Detection - COMPLETE (PSI/KS implemented)
- p5.5: A/B Testing Framework - COMPLETE (SGD vs XGBoost comparison)
- p5.6: Model Monitoring Dashboard - COMPLETE (Streamlit dashboard)
- p5.7: Final Validation & Handoff - IN PROGRESS

### Key Metrics:
- Primary Model: SGD v1.0.1
- Test Accuracy: 71.69% (717 samples, 39 classes)
- Inference Latency: 1.72ms average
- Load Test P95: 21.75ms (Target <100ms) ✓
- Success Rate: 100% across all concurrency levels ✓

### Known Issues:
1. XGBoost model shows 0% accuracy in A/B test (label mapping issue)
   - Recommendation: Investigate ClassMapper integration in p4.6
   - Workaround: Use SGD as primary model for production

2. p4.4-inference-api.py model loading
   - Status: Fixed in p5.7 (dictionary wrapper extraction)
   - Action: Verify API starts successfully after fix

## PHASE 6 OBJECTIVES

### p6.1: API Deployment
- Deploy FastAPI inference server to production
- Configure Gunicorn/Uvicorn workers
- Set up reverse proxy (Nginx)
- Configure SSL/TLS certificates

### p6.2: Containerization
- Create Dockerfile for API server
- Build and test Docker image
- Push to container registry (Docker Hub/ECR)
- Create docker-compose.yml for local testing

### p6.3: CI/CD Pipeline
- Set up GitHub Actions workflow
- Configure automated testing
- Implement automated deployment
- Add rollback mechanisms

### p6.4: Production Monitoring
- Deploy monitoring dashboard (Streamlit/Grafana)
- Configure alerting (email/Slack/PagerDuty)
- Set up log aggregation (ELK/Loki)
- Implement distributed tracing

### p6.5: Scaling & Optimization
- Configure horizontal pod autoscaling (Kubernetes)
- Implement caching layer (Redis)
- Optimize model inference (batching, quantization)
- Load balancing configuration

### p6.6: Documentation & Training
- Create deployment runbook
- Document monitoring procedures
- Train operations team
- Conduct disaster recovery drill

## CRITICAL FILES FOR PHASE 6

### Models:
- models/phase4/sgd_v1.0.1.pkl (Primary model - STAGING)
- data/final/embeddings_v2.0/vectorizer.pkl (TF-IDF vectorizer)

### Scripts:
- scripts/p4.4-inference-api.py (Fixed in p5.7)
- scripts/p5.6-monitoring-dashboard.py (Monitoring UI)
- scripts/test-inference-api.py (API integration tests)

### Configuration:
- config/config.yaml (Main configuration)
- models/registry/registry_index.json (Model registry)

### Results:
- results/phase5/phase5_summary.json (Complete Phase 5 results)
- results/phase5/ab_test_analysis.json (A/B test comparison)

## DEPLOYMENT CHECKLIST

[ ] API server starts successfully
[ ] Health endpoint responds (GET /health)
[ ] Prediction endpoint works (POST /predict)
[ ] Batch endpoint works (POST /predict/batch)
[ ] Metrics endpoint works (GET /metrics)
[ ] Load testing passes (P95 < 100ms)
[ ] Monitoring dashboard accessible
[ ] Alerts configured and tested
[ ] Documentation complete
[ ] Team training completed

## ROLLBACK PROCEDURE

If deployment fails:
1. Stop new deployment
2. Revert to previous model version in registry
3. Restart API server with previous version
4. Investigate root cause
5. Document incident and resolution

## CONTACTS & RESOURCES

- Project Lead: mavyjimz
- Repository: https://github.com/mavyjimz/multi-model-orchestration-project
- Documentation: PHASE4_README.md, PHASE5_README.md (to be created)
- Monitoring: http://localhost:8501 (Streamlit dashboard)

## SUCCESS CRITERIA FOR PHASE 6

- API uptime > 99.9%
- P95 latency < 100ms
- Error rate < 1%
- Successful deployment to production environment
- Monitoring and alerting operational
- Team trained and documentation complete

---
End of Phase 6 Handoff Documentation
"""
    
    output_path = 'PHASE6_HANDOFF.md'
    
    with open(output_path, 'w') as f:
        f.write(handoff_doc.format(timestamp=datetime.now().isoformat()))
    
    logger.info(f"✓ Phase 6 handoff documentation saved to {output_path}")
    return output_path


def main():
    """Main execution function"""
    logger.info("="*80)
    logger.info("PHASE 5.7: FINAL VALIDATION & HANDOFF")
    logger.info("="*80)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'steps': {}
    }
    
    # Step 1: Fix p4.4 API
    logger.info("\n[Step 1/5] Fixing p4.4-inference-api.py...")
    results['steps']['p44_fix'] = fix_p44_api_loading()
    
    # Step 2: Test API server
    logger.info("\n[Step 2/5] Testing API server...")
    results['steps']['api_test'] = test_api_server()
    
    # Step 3: Generate Phase 5 summary
    logger.info("\n[Step 3/5] Generating Phase 5 summary...")
    phase5_summary = generate_phase5_summary()
    results['steps']['phase5_summary'] = True
    
    # Step 4: Create release tag
    logger.info("\n[Step 4/5] Creating release tag...")
    results['steps']['release_tag'] = create_release_tag()
    
    # Step 5: Generate handoff documentation
    logger.info("\n[Step 5/5] Generating handoff documentation...")
    handoff_path = generate_handoff_documentation()
    results['steps']['handoff_doc'] = True
    
    # Save validation results
    output_path = 'results/phase5/final_validation_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "="*80)
    print("PHASE 5.7 FINAL VALIDATION SUMMARY")
    print("="*80)
    print(f"p4.4 API Fix: {'✓ PASS' if results['steps']['p44_fix'] else '✗ FAIL'}")
    print(f"API Server Test: {'✓ PASS' if results['steps'].get('api_test', False) else '✗ FAIL'}")
    print(f"Phase 5 Summary: {'✓ PASS' if results['steps']['phase5_summary'] else '✗ FAIL'}")
    print(f"Release Tag: {'✓ PASS' if results['steps']['release_tag'] else '✗ FAIL'}")
    print(f"Handoff Doc: {'✓ PASS' if results['steps']['handoff_doc'] else '✗ FAIL'}")
    print("="*80)
    
    if all(results['steps'].values()):
        print("\n🎉 PHASE 5 COMPLETE! Ready for Phase 6 deployment.")
    else:
        print("\n⚠️ Some validation steps failed. Review logs before proceeding.")
    
    logger.info(f"\nValidation results saved to {output_path}")


if __name__ == '__main__':
    main()
