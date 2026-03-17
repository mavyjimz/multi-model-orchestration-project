#!/usr/bin/env python3
"""
Phase 4.4: Inference API & Serving Prototype
=============================================
Production-grade FastAPI wrapper for model inference with:
- Model loading from registry (SGD v1.0.1 from staging)
- Input validation and preprocessing
- Prediction endpoint with confidence scores
- Error handling and logging middleware
- Health check and model metadata endpoints
- Performance monitoring (latency, request counting)

Author: Multi-Model Orchestration Team
Date: March 14, 2026
Version: 4.4.1 (Fixed label mapping)
"""

import os
import sys
import json
import time
import logging
import pickle
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Centralized configuration for inference API"""
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    MODEL_REGISTRY_PATH = PROJECT_ROOT / "models" / "registry" / "staging"
    MODEL_PHASE4_PATH = PROJECT_ROOT / "models" / "phase4"
    EMBEDDING_PATH = PROJECT_ROOT / "data" / "final" / "embeddings_v2.0"
    CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
    
    # Model settings
    MODEL_NAME = "sgd_v1.0.1"
    MODEL_FILE = f"{MODEL_NAME}.pkl"
    VECTORIZER_FILE = "vectorizer.pkl"
    
    # API settings
    HOST = "0.0.0.0"
    PORT = 8000
    WORKERS = 1
    RELOAD = False
    
    # Performance settings
    MAX_REQUEST_SIZE = 1024 * 1024  # 1MB
    REQUEST_TIMEOUT = 30  # seconds
    MAX_CONCURRENT_REQUESTS = 100
    
    # Logging settings
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE = PROJECT_ROOT / "logs" / "p4.4-inference-api.log"

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging() -> logging.Logger:
    """Configure production-grade logging"""
    
    # Ensure log directory exists
    Config.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("inference_api")
    logger.setLevel(getattr(logging, Config.LOG_LEVEL))
    
    # Clear existing handlers
    logger.handlers = []
    
    # File handler
    file_handler = logging.FileHandler(Config.LOG_FILE)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(Config.LOG_FORMAT))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(Config.LOG_FORMAT))
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

# =============================================================================
# PYDANTIC MODELS (Input/Output Validation)
# =============================================================================

class PredictionRequest(BaseModel):
    """Request schema for single prediction"""
    
    text: str = Field(..., min_length=1, max_length=10000, description="Input text for classification")
    request_id: Optional[str] = Field(None, description="Optional client request ID for tracing")
    
    @validator('text')
    def validate_text(cls, v):
        """Validate and clean input text"""
        if not v or not v.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        return v.strip()
    
    class Config:
        schema_extra = {
            "example": {
                "text": "How do I reset my password?",
                "request_id": "req_12345"
            }
        }

class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions"""
    
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of input texts")
    request_id: Optional[str] = Field(None, description="Optional batch request ID")
    
    @validator('texts')
    def validate_texts(cls, v):
        """Validate batch texts"""
        if not v:
            raise ValueError("Texts list cannot be empty")
        for i, text in enumerate(v):
            if not text or not text.strip():
                raise ValueError(f"Text at index {i} cannot be empty")
            if len(text) > 10000:
                raise ValueError(f"Text at index {i} exceeds 10000 characters")
        return [t.strip() for t in v]

class PredictionResponse(BaseModel):
    """Response schema for single prediction"""
    
    request_id: str
    prediction: str
    confidence: float
    all_probabilities: Dict[str, float]
    latency_ms: float
    model_version: str
    timestamp: str
    
    class Config:
        schema_extra = {
            "example": {
                "request_id": "req_12345",
                "prediction": "password_reset",
                "confidence": 0.96,
                "all_probabilities": {"password_reset": 0.96, "account_access": 0.03, "other": 0.01},
                "latency_ms": 12.5,
                "model_version": "sgd_v1.0.1",
                "timestamp": "2026-03-14T10:30:00Z"
            }
        }

class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions"""
    
    request_id: str
    predictions: List[PredictionResponse]
    total_latency_ms: float
    average_latency_ms: float
    model_version: str
    timestamp: str

class HealthResponse(BaseModel):
    """Health check response"""
    
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float
    timestamp: str

class ModelMetadataResponse(BaseModel):
    """Model metadata response"""
    
    model_name: str
    model_version: str
    model_stage: str
    accuracy: float
    f1_score: float
    training_date: str
    feature_count: int
    classes: List[str]

# =============================================================================
# MODEL LOADER (FIXED)
# =============================================================================

class ModelLoader:
    """Handles model and vectorizer loading from registry"""
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.metadata = None
        self.classes = None
        self.is_loaded = False
        self.load_time = None
    
    def load(self) -> bool:
        """Load model, vectorizer, and metadata"""
        try:
            logger.info("Loading model and vectorizer...")
            
            # Load model
            model_path = Config.MODEL_PHASE4_PATH / Config.MODEL_FILE
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            with open(model_path, 'rb') as f:
                loaded = pickle.load(f)
            
            # Extract model from dictionary wrapper if present
            if isinstance(loaded, dict):
                self.model = loaded.get("model", loaded)
                logger.info("  Model extracted from dictionary wrapper")
            else:
                self.model = loaded
            logger.info(f"✓ Model loaded: {model_path}")
            logger.info(f"  Model type: {self.model.__class__.__name__}")
            logger.info(f"  Model classes: {getattr(self.model, 'classes_', 'N/A')}")
            
            # Load vectorizer
            vectorizer_path = Config.EMBEDDING_PATH / Config.VECTORIZER_FILE
            if not vectorizer_path.exists():
                raise FileNotFoundError(f"Vectorizer not found: {vectorizer_path}")
            
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            logger.info(f"✓ Vectorizer loaded: {vectorizer_path}")
            logger.info(f"  Features: {self.vectorizer.get_feature_names_out().shape[0]}")
            
            # Load metadata
            metadata_path = Config.MODEL_PHASE4_PATH / "model_manifest_v1.0.1.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"✓ Metadata loaded: {metadata_path}")
            
            # Extract classes safely - MUST BE INSIDE TRY BLOCK
            if hasattr(self.model, "classes_"):
                self.classes = self.model.classes_.tolist()
            else:
                self.classes = []
                logger.warning("Model has no classes_ attribute")
            logger.info(f"✓ Classes loaded: {len(self.classes)} classes")
            
            self.is_loaded = True
            self.load_time = datetime.utcnow()
            
            logger.info("✓ Model loading complete")
            return True
            
        except Exception as e:
            logger.error(f"✗ Model loading failed: {str(e)}", exc_info=True)
            return False
    def predict(self, text: str) -> Dict[str, Any]:
        """Make prediction for single text"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        try:
            # Transform text using vectorizer
            text_vector = self.vectorizer.transform([text])
            
            # Get prediction and probabilities
            prediction_idx = self.model.predict(text_vector)[0]
            probabilities = self.model.predict_proba(text_vector)[0]
            
            # Get prediction label from model classes
            prediction = str(self.classes[prediction_idx]) if prediction_idx < len(self.classes) else f"class_{prediction_idx}"
            
            # Create probability dictionary
            class_probs = {}
            for i, prob in enumerate(probabilities):
                class_name = str(self.classes[i]) if i < len(self.classes) else f"class_{i}"
                class_probs[class_name] = float(prob)
            
            # Sort by probability
            class_probs = dict(sorted(class_probs.items(), key=lambda x: x[1], reverse=True))
            
            # Get confidence (max probability)
            confidence = float(np.max(probabilities))
            
            return {
                "prediction": prediction,
                "confidence": confidence,
                "all_probabilities": class_probs
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}", exc_info=True)
            raise
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Make predictions for batch of texts"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        try:
            # Transform texts using vectorizer
            text_vectors = self.vectorizer.transform(texts)
            
            # Get predictions and probabilities
            prediction_indices = self.model.predict(text_vectors)
            probabilities = self.model.predict_proba(text_vectors)
            
            results = []
            for pred_idx, probs in zip(prediction_indices, probabilities):
                # Get prediction label
                prediction = str(self.classes[pred_idx]) if pred_idx < len(self.classes) else f"class_{pred_idx}"
                
                # Create probability dictionary
                class_probs = {}
                for i, prob in enumerate(probs):
                    class_name = str(self.classes[i]) if i < len(self.classes) else f"class_{i}"
                    class_probs[class_name] = float(prob)
                
                # Sort by probability
                class_probs = dict(sorted(class_probs.items(), key=lambda x: x[1], reverse=True))
                
                # Get confidence
                confidence = float(np.max(probs))
                
                results.append({
                    "prediction": prediction,
                    "confidence": confidence,
                    "all_probabilities": class_probs
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Batch prediction error: {str(e)}", exc_info=True)
            raise
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get model metadata"""
        if not self.metadata:
            return {}
        return self.metadata

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

# Initialize model loader
model_loader = ModelLoader()

# Create FastAPI app
app = FastAPI(
    title="Multi-Model Orchestration Inference API",
    description="Production-grade inference API for intent classification model",
    version="4.4.1",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Track startup time
START_TIME = time.time()

# Request counter for monitoring
request_count = {"total": 0, "success": 0, "error": 0}

# =============================================================================
# MIDDLEWARE
# =============================================================================

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Log all requests with timing and status"""
    start_time = time.time()
    request_id = request.headers.get("X-Request-ID", f"req_{int(time.time() * 1000)}")
    
    # Add request ID to response
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    
    # Log request
    duration_ms = (time.time() - start_time) * 1000
    logger.info(
        f"Request: {request.method} {request.url.path} | "
        f"Status: {response.status_code} | "
        f"Duration: {duration_ms:.2f}ms | "
        f"Request-ID: {request_id}"
    )
    
    # Update counters
    request_count["total"] += 1
    if response.status_code < 400:
        request_count["success"] += 1
    else:
        request_count["error"] += 1
    
    return response

@app.middleware("http")
async def size_limit_middleware(request: Request, call_next):
    """Enforce maximum request size"""
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > Config.MAX_REQUEST_SIZE:
        logger.warning(f"Request size exceeded limit: {content_length} bytes")
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content={"error": "Request size exceeds maximum allowed (1MB)"}
        )
    return await call_next(request)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# EXCEPTION HANDLERS
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    logger.error(f"HTTP Error {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "status_code": 500}
    )

# =============================================================================
# ENDPOINTS
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Starting Inference API...")
    success = model_loader.load()
    if not success:
        logger.error("Failed to load model on startup")
        raise RuntimeError("Model loading failed")
    logger.info("Inference API started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Inference API...")
    logger.info(f"Total requests served: {request_count['total']}")
    logger.info(f"Successful: {request_count['success']}, Errors: {request_count['error']}")

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Multi-Model Orchestration Inference API",
        "version": "4.4.1",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "metadata": "/model/metadata"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    uptime = time.time() - START_TIME
    return HealthResponse(
        status="healthy" if model_loader.is_loaded else "unhealthy",
        model_loaded=model_loader.is_loaded,
        model_version=Config.MODEL_NAME,
        uptime_seconds=round(uptime, 2),
        timestamp=datetime.utcnow().isoformat() + "Z"
    )

@app.get("/model/metadata", response_model=ModelMetadataResponse, tags=["Model"])
async def get_model_metadata():
    """Get model metadata and performance metrics"""
    metadata = model_loader.get_metadata()
    classes = model_loader.classes or []
    
    return ModelMetadataResponse(
        model_name=metadata.get("model_name", "sgd"),
        model_version=metadata.get("model_version", "1.0.1"),
        model_stage=metadata.get("stage", "staging"),
        accuracy=metadata.get("metrics", {}).get("accuracy", 0.9693),
        f1_score=metadata.get("metrics", {}).get("f1_score", 0.9652),
        training_date=metadata.get("training_date", "2026-03-14"),
        feature_count=5000,
        classes=[str(c) for c in classes]
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(request: PredictionRequest):
    """
    Single text prediction endpoint
    
    - **text**: Input text for classification (1-10000 characters)
    - **request_id**: Optional client request ID for tracing
    
    Returns prediction with confidence score and all class probabilities.
    """
    start_time = time.time()
    request_id = request.request_id or f"req_{int(time.time() * 1000)}"
    
    try:
        # Make prediction
        result = model_loader.predict(request.text)
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            request_id=request_id,
            prediction=result["prediction"],
            confidence=round(result["confidence"], 4),
            all_probabilities={k: round(v, 4) for k, v in result["all_probabilities"].items()},
            latency_ms=round(latency_ms, 2),
            model_version=Config.MODEL_NAME,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Batch prediction endpoint
    
    - **texts**: List of input texts (1-100 items, each 1-10000 characters)
    - **request_id**: Optional batch request ID
    
    Returns predictions for all texts with individual latencies.
    """
    start_time = time.time()
    request_id = request.request_id or f"batch_{int(time.time() * 1000)}"
    
    try:
        # Make batch predictions
        results = model_loader.predict_batch(request.texts)
        
        # Calculate latencies
        total_latency_ms = (time.time() - start_time) * 1000
        avg_latency_ms = total_latency_ms / len(results) if results else 0
        
        # Build individual responses
        predictions = []
        for i, result in enumerate(results):
            predictions.append(PredictionResponse(
                request_id=f"{request_id}_{i}",
                prediction=result["prediction"],
                confidence=round(result["confidence"], 4),
                all_probabilities={k: round(v, 4) for k, v in result["all_probabilities"].items()},
                latency_ms=round(avg_latency_ms, 2),
                model_version=Config.MODEL_NAME,
                timestamp=datetime.utcnow().isoformat() + "Z"
            ))
        
        return BatchPredictionResponse(
            request_id=request_id,
            predictions=predictions,
            total_latency_ms=round(total_latency_ms, 2),
            average_latency_ms=round(avg_latency_ms, 2),
            model_version=Config.MODEL_NAME,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )

@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """Get API performance metrics"""
    uptime = time.time() - START_TIME
    return {
        "uptime_seconds": round(uptime, 2),
        "total_requests": request_count["total"],
        "successful_requests": request_count["success"],
        "failed_requests": request_count["error"],
        "success_rate": round(request_count["success"] / max(request_count["total"], 1) * 100, 2),
        "model_loaded": model_loader.is_loaded,
        "model_version": Config.MODEL_NAME
    }

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Phase 4.4: Inference API & Serving Prototype")
    logger.info("=" * 60)
    logger.info(f"Host: {Config.HOST}")
    logger.info(f"Port: {Config.PORT}")
    logger.info(f"Model: {Config.MODEL_NAME}")
    logger.info(f"Registry: {Config.MODEL_REGISTRY_PATH}")
    logger.info("=" * 60)
    
    # Run server
    uvicorn.run(
        app,
        host=Config.HOST,
        port=Config.PORT,
        workers=Config.WORKERS,
        reload=Config.RELOAD,
        log_level=Config.LOG_LEVEL.lower()
    )
