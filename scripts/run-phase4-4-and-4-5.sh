#!/bin/bash
# =============================================================================
# Phase 4.4 & 4.5: Complete Execution Script
# =============================================================================
# This script will:
# 1. Install all required dependencies
# 2. Start the inference API server
# 3. Run API tests
# 4. Run model serialization and optimization
# 5. Generate final summary
# =============================================================================

set -e  # Exit on error

echo "=============================================================================="
echo "Phase 4.4 & 4.5: Complete Execution Pipeline"
echo "=============================================================================="
echo "Started at: $(date)"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_header() {
    echo -e "${BLUE}=============================================================================="
    echo -e "$1"
    echo -e "==============================================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}→ $1${NC}"
}

# Step 1: Install dependencies
print_header "STEP 1: Installing Dependencies"

echo ""
print_info "Installing Phase 4.4 dependencies (FastAPI, uvicorn, etc.)..."
pip install -r requirements-inference.txt

echo ""
print_info "Installing Phase 4.5 dependencies (ONNX, skl2onnx, onnxruntime)..."
pip install onnx skl2onnx onnxruntime joblib

echo ""
print_success "All dependencies installed successfully!"

# Step 2: Create logs directory
print_header "STEP 2: Setting Up Environment"

mkdir -p logs
mkdir -p results/benchmarks
mkdir -p models/optimized

print_success "Directories created"

# Step 3: Start API server in background
print_header "STEP 3: Starting Inference API Server"

print_info "Starting server on http://localhost:8000..."
python3 scripts/p4.4-inference-api.py > logs/api-server.log 2>&1 &
API_PID=$!

print_info "API Server PID: $API_PID"
print_info "Waiting 5 seconds for server to initialize..."

sleep 5

# Check if server is running
if ps -p $API_PID > /dev/null; then
    print_success "API server is running (PID: $API_PID)"
else
    print_error "API server failed to start. Check logs/api-server.log"
    exit 1
fi

# Step 4: Run API tests
print_header "STEP 4: Running API Tests"

echo ""
python3 scripts/test-inference-api.py

if [ $? -eq 0 ]; then
    print_success "All API tests passed!"
else
    print_error "Some API tests failed"
fi

# Step 5: Run model serialization
print_header "STEP 5: Running Model Serialization & Optimization"

echo ""
python3 scripts/p4.5-model-serialization.py

if [ $? -eq 0 ]; then
    print_success "Model serialization completed successfully!"
else
    print_error "Model serialization encountered errors"
fi

# Step 6: Generate summary
print_header "STEP 6: Execution Summary"

echo ""
echo "Artifacts Created:"
echo "------------------"

# API logs
if [ -f "logs/p4.4-inference-api.log" ]; then
    print_info "API Server Logs: logs/p4.4-inference-api.log"
fi

# Serialization logs
if [ -f "logs/p4.5-model-serialization.log" ]; then
    print_info "Serialization Logs: logs/p4.5-model-serialization.log"
fi

# Optimized models
echo ""
echo "Optimized Models:"
if [ -f "models/optimized/sgd_v1.0.1.onnx" ]; then
    ONNX_SIZE=$(du -h models/optimized/sgd_v1.0.1.onnx | cut -f1)
    print_success "  ONNX Model: models/optimized/sgd_v1.0.1.onnx ($ONNX_SIZE)"
fi

if [ -f "models/optimized/sgd_v1.0.1_int8.pkl" ]; then
    INT8_SIZE=$(du -h models/optimized/sgd_v1.0.1_int8.pkl | cut -f1)
    print_success "  Quantized Model: models/optimized/sgd_v1.0.1_int8.pkl ($INT8_SIZE)"
fi

# Benchmark results
echo ""
echo "Benchmark Results:"
if [ -f "results/benchmarks/model_benchmark_v1.0.1.json" ]; then
    print_success "  Performance Benchmarks: results/benchmarks/model_benchmark_v1.0.1.json"
    echo ""
    echo "  Quick View:"
    cat results/benchmarks/model_benchmark_v1.0.1.json | python3 -m json.tool 2>/dev/null || cat results/benchmarks/model_benchmark_v1.0.1.json
fi

if [ -f "results/benchmarks/serialization_results_v1.0.1.json" ]; then
    print_success "  Serialization Results: results/benchmarks/serialization_results_v1.0.1.json"
fi

# API endpoints
echo ""
print_header "API Server Status"
echo ""
echo "Server is running at: http://localhost:8000"
echo ""
echo "Available Endpoints:"
echo "  - Swagger UI:     http://localhost:8000/docs"
echo "  - ReDoc:          http://localhost:8000/redoc"
echo "  - Health Check:   http://localhost:8000/health"
echo "  - Model Metadata: http://localhost:8000/model/metadata"
echo "  - Metrics:        http://localhost:8000/metrics"
echo ""
echo "To stop the API server, run:"
echo "  kill $API_PID"
echo ""

print_header "EXECUTION COMPLETE"
echo ""
echo "Completed at: $(date)"
echo ""
print_success "Phase 4.4 (Inference API) - COMPLETE"
print_success "Phase 4.5 (Model Serialization) - COMPLETE"
echo ""
echo "Next: Phase 4.6 (Edge Case Resolution & Retraining)"
echo "=============================================================================="

# Keep script running to show API server logs
echo ""
print_info "API server is still running. Press Ctrl+C to stop."
echo ""

# Wait for user to press Ctrl+C
wait $API_PID
