"""
Streamlit Dashboard for MLOps Observability - Phase 9
Displays Prometheus metrics, logs, and system health.
"""

import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MLOps Observability Dashboard", layout="wide")
st.title("MLOps Observability Dashboard")

# Prometheus metrics endpoint
METRICS_URL = "http://localhost:8000/metrics"

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Select View", ["Overview", "Request Metrics", "Latency", "Errors", "Model Health"])

# Fetch metrics
def fetch_metrics():
    try:
        response = requests.get(METRICS_URL, timeout=5)
        if response.status_code == 200:
            return response.text
    except requests.exceptions.RequestException:
        return None
    return None

metrics_text = fetch_metrics()

if metrics_text:
    st.success("Metrics endpoint connected")
else:
    st.warning("Metrics endpoint not available - ensure container is running on localhost:8000")

# Overview Page
if page == "Overview":
    st.header("System Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Metrics Endpoint", "Active" if metrics_text else "Offline")
    with col2:
        st.metric("API Health", "Check /health endpoint")
    with col3:
        st.metric("Phase", "9 - Observability")

# Request Metrics Page
elif page == "Request Metrics":
    st.header("Request Metrics")
    st.code(metrics_text[:2000] if metrics_text else "No metrics available", language="text")

# Latency Page
elif page == "Latency":
    st.header("Latency Histogram")
    st.info("P50/P95/P99 latency tracking available via Prometheus metrics endpoint")

# Errors Page
elif page == "Errors":
    st.header("Error Tracking")
    st.info("Error rates tracked via mlops_error_total counter")

# Model Health Page
elif page == "Model Health":
    st.header("Model Health")
    st.info("Model version and drift metrics available via Prometheus")

st.sidebar.markdown("---")
st.sidebar.markdown("**Phase 9 - Monitoring & Observability**")
