#!/usr/bin/env python3
"""
Phase 5.6: Model Monitoring Dashboard
Real-time inference metrics, performance tracking, and drift alerts
"""

import os
import sys
import json
import time
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
from collections import deque
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/p5.6-monitoring.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ModelMonitor:
    """
    Real-time model performance and drift monitoring
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.inference_metrics = deque(maxlen=1000)
        self.performance_history = deque(maxlen=100)
        self.drift_alerts = deque(maxlen=50)
        self.start_time = datetime.now()
        
    def log_inference(self, prediction: Any, latency_ms: float, 
                     confidence: float, success: bool = True) -> None:
        """Log a single inference request"""
        self.inference_metrics.append({
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction,
            'latency_ms': latency_ms,
            'confidence': confidence,
            'success': success
        })
    
    def calculate_realtime_metrics(self) -> Dict[str, float]:
        """Calculate real-time performance metrics"""
        if not self.inference_metrics:
            return {
                'total_requests': 0,
                'success_rate': 0.0,
                'avg_latency_ms': 0.0,
                'p50_latency_ms': 0.0,
                'p95_latency_ms': 0.0,
                'p99_latency_ms': 0.0,
                'requests_per_minute': 0.0
            }
        
        metrics_list = list(self.inference_metrics)
        latencies = [m['latency_ms'] for m in metrics_list]
        successes = sum(1 for m in metrics_list if m['success'])
        
        # Calculate time window
        time_window = (datetime.now() - self.start_time).total_seconds() / 60.0
        rpm = len(metrics_list) / time_window if time_window > 0 else 0
        
        return {
            'total_requests': len(metrics_list),
            'success_rate': successes / len(metrics_list) * 100,
            'avg_latency_ms': np.mean(latencies),
            'p50_latency_ms': np.percentile(latencies, 50),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'requests_per_minute': rpm
        }
    
    def check_drift(self, current_predictions: np.ndarray, 
                   baseline_distribution: Dict[int, float],
                   threshold: float = 0.1) -> Dict[str, Any]:
        """Check for prediction distribution drift"""
        if len(current_predictions) == 0:
            return {'drift_detected': False, 'psi': 0.0}
        
        # Calculate current distribution
        unique, counts = np.unique(current_predictions, return_counts=True)
        current_dist = {int(u): c / len(current_predictions) for u, c in zip(unique, counts)}
        
        # Calculate PSI (Population Stability Index)
        psi = 0.0
        for label in set(baseline_distribution.keys()) | set(current_dist.keys()):
            baseline_prob = baseline_distribution.get(label, 0.001)
            current_prob = current_dist.get(label, 0.001)
            psi += (current_prob - baseline_prob) * np.log(current_prob / baseline_prob)
        
        drift_detected = abs(psi) > threshold
        
        if drift_detected:
            self.drift_alerts.append({
                'timestamp': datetime.now().isoformat(),
                'type': 'distribution_drift',
                'psi': float(psi),
                'threshold': threshold,
                'severity': 'HIGH' if abs(psi) > 0.25 else 'MEDIUM'
            })
        
        return {
            'drift_detected': drift_detected,
            'psi': float(psi),
            'threshold': threshold
        }
    
    def log_performance(self, accuracy: float, f1_weighted: float, 
                       f1_macro: float) -> None:
        """Log periodic performance metrics"""
        self.performance_history.append({
            'timestamp': datetime.now().isoformat(),
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro
        })
    
    def get_alerts(self) -> List[Dict[str, Any]]:
        """Get recent drift alerts"""
        return list(self.drift_alerts)
    
    def get_performance_trend(self) -> pd.DataFrame:
        """Get performance history as DataFrame"""
        return pd.DataFrame(list(self.performance_history))


class MonitoringDashboard:
    """
    Interactive monitoring dashboard using Streamlit
    """
    
    def __init__(self, monitor: ModelMonitor, config: Dict[str, Any]):
        self.monitor = monitor
        self.config = config
        
    def render_dashboard(self):
        """Render the Streamlit dashboard"""
        try:
            import streamlit as st
        except ImportError:
            logger.error("Streamlit not installed. Install with: pip install streamlit")
            sys.exit(1)
        
        st.set_page_config(
            page_title="Model Monitoring Dashboard",
            page_icon="📊",
            layout="wide"
        )
        
        st.title("📊 Multi-Model Orchestration - Monitoring Dashboard")
        st.markdown("---")
        
        # Sidebar configuration
        st.sidebar.header("⚙️ Configuration")
        refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 1, 60, 5)
        show_drift = st.sidebar.checkbox("Show Drift Analysis", True)
        show_performance = st.sidebar.checkbox("Show Performance Trends", True)
        
        # Main metrics
        st.header("📈 Real-Time Inference Metrics")
        metrics = self.monitor.calculate_realtime_metrics()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Requests", f"{metrics['total_requests']:,}")
        with col2:
            st.metric("Success Rate", f"{metrics['success_rate']:.2f}%",
                     delta=f"{metrics['success_rate'] - 99:.2f}%" if metrics['success_rate'] < 99 else None)
        with col3:
            st.metric("Avg Latency", f"{metrics['avg_latency_ms']:.2f} ms",
                     delta=f"{metrics['avg_latency_ms'] - 100:.2f} ms" if metrics['avg_latency_ms'] > 100 else None)
        with col4:
            st.metric("Requests/Min", f"{metrics['requests_per_minute']:.1f}")
        
        # Latency percentiles
        st.subheader("Latency Distribution")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("P50", f"{metrics['p50_latency_ms']:.2f} ms")
        with col2:
            st.metric("P95", f"{metrics['p95_latency_ms']:.2f} ms",
                     delta=f"{metrics['p95_latency_ms'] - 100:.2f} ms" if metrics['p95_latency_ms'] > 100 else "OK")
        with col3:
            st.metric("P99", f"{metrics['p99_latency_ms']:.2f} ms")
        
        st.markdown("---")
        
        # Performance trends
        if show_performance:
            st.header("📉 Performance Trends")
            perf_df = self.monitor.get_performance_trend()
            
            if not perf_df.empty:
                col1, col2 = st.columns(2)
                with col1:
                    st.line_chart(perf_df.set_index('timestamp')['accuracy'])
                with col2:
                    st.line_chart(perf_df.set_index('timestamp')[['f1_weighted', 'f1_macro']])
            else:
                st.info("No performance data available yet. Run periodic validation to populate.")
        
        st.markdown("---")
        
        # Drift detection
        if show_drift:
            st.header("🚨 Drift Detection")
            alerts = self.monitor.get_alerts()
            
            if alerts:
                st.warning(f"⚠️ {len(alerts)} drift alert(s) detected")
                for alert in alerts[-5:]:  # Show last 5 alerts
                    st.error(f"**{alert['timestamp']}** - {alert['type'].upper()} "
                            f"(PSI: {alert['psi']:.4f}, Severity: {alert['severity']})")
            else:
                st.success("✅ No drift detected")
        
        st.markdown("---")
        
        # Model comparison
        st.header("🔄 Model Comparison (A/B Test Results)")
        try:
            ab_test_path = self.config.get('ab_test_results', 'results/phase5/ab_test_analysis.json')
            if os.path.exists(ab_test_path):
                with open(ab_test_path, 'r') as f:
                    ab_results = json.load(f)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader(ab_results['primary_model']['name'])
                    st.metric("Accuracy", f"{ab_results['primary_model']['metrics']['accuracy']:.2%}")
                    st.metric("F1 Weighted", f"{ab_results['primary_model']['metrics']['f1_weighted']:.4f}")
                
                with col2:
                    st.subheader(ab_results['candidate_model']['name'])
                    st.metric("Accuracy", f"{ab_results['candidate_model']['metrics']['accuracy']:.2%}")
                    st.metric("F1 Weighted", f"{ab_results['candidate_model']['metrics']['f1_weighted']:.4f}")
                
                st.info(f"**Recommendation:** {ab_results['rollback_decision']['recommendation']}")
                if ab_results['rollback_decision']['reasons']:
                    for reason in ab_results['rollback_decision']['reasons']:
                        st.warning(reason)
            else:
                st.info("A/B test results not found. Run p5.5-ab-testing-framework.py first.")
        except Exception as e:
            st.error(f"Error loading A/B test results: {str(e)}")
        
        # Auto-refresh
        if refresh_rate > 0:
            time.sleep(refresh_rate)
            st.rerun()


def simulate_inference(monitor: ModelMonitor, duration_seconds: int = 60):
    """Simulate inference requests for testing"""
    logger.info(f"Starting inference simulation for {duration_seconds} seconds...")
    
    start_time = time.time()
    request_count = 0
    
    while time.time() - start_time < duration_seconds:
        # Simulate inference
        latency = np.random.exponential(5)  # Average 5ms latency
        success = np.random.random() > 0.01  # 99% success rate
        confidence = np.random.beta(8, 2)  # High confidence distribution
        
        monitor.log_inference(
            prediction=np.random.randint(0, 41),
            latency_ms=latency,
            confidence=confidence,
            success=success
        )
        
        request_count += 1
        time.sleep(0.1)  # 10 requests per second
    
    logger.info(f"Simulation complete. Processed {request_count} requests.")


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Phase 5.6: Model Monitoring Dashboard')
    parser.add_argument('--mode', choices=['dashboard', 'simulate', 'api'], default='dashboard',
                       help='Run mode: dashboard (Streamlit), simulate (test data), or api (background monitoring)')
    parser.add_argument('--duration', type=int, default=60, help='Simulation duration in seconds')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    args = parser.parse_args()
    
    # Configuration
    config = {
        'ab_test_results': 'results/phase5/ab_test_analysis.json',
        'drift_threshold': 0.1,
        'latency_threshold_ms': 100,
        'success_rate_threshold': 99.0
    }
    
    # Initialize monitor
    monitor = ModelMonitor(config)
    
    if args.mode == 'simulate':
        # Run simulation
        simulate_inference(monitor, args.duration)
        
        # Print summary
        metrics = monitor.calculate_realtime_metrics()
        print("\n" + "="*60)
        print("SIMULATION SUMMARY")
        print("="*60)
        print(f"Total Requests: {metrics['total_requests']}")
        print(f"Success Rate: {metrics['success_rate']:.2f}%")
        print(f"Avg Latency: {metrics['avg_latency_ms']:.2f} ms")
        print(f"P95 Latency: {metrics['p95_latency_ms']:.2f} ms")
        print("="*60)
    
    elif args.mode == 'dashboard':
        # Run Streamlit dashboard
        dashboard = MonitoringDashboard(monitor, config)
        dashboard.render_dashboard()
    
    elif args.mode == 'api':
        # Run background API monitoring
        logger.info("Starting API monitoring mode...")
        # This would integrate with the inference API to log real requests
        # Implementation depends on p4.4-inference-api.py structure
        logger.warning("API mode requires integration with inference API - not yet implemented")


if __name__ == '__main__':
    main()
