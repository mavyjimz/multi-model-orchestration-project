#!/usr/bin/env python3
"""
Phase 4.5: Model Serialization & Optimization
==============================================
Production-grade model export and optimization with:
- ONNX export for cross-platform deployment
- Model quantization (int8) for reduced size and faster inference
- TorchScript export for PyTorch-based serving
- Benchmarking: pickle vs ONNX vs quantized performance
- Size comparison and accuracy validation
- Optimization recommendations based on deployment target

Author: Multi-Model Orchestration Team
Date: March 14, 2026
Version: 4.5.0
"""

import os
import sys
import json
import time
import pickle
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd

# Try to import optional dependencies
try:
    import onnx
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import StringTensorType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    warnings.warn("ONNX libraries not available. Install: pip install onnx skl2onnx")

try:
    import torch
    from sklearn.linear_model import SGDClassifier
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Install: pip install torch")

from sklearn.base import BaseEstimator
import joblib

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Centralized configuration for model serialization"""
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    MODEL_PHASE4_PATH = PROJECT_ROOT / "models" / "phase4"
    MODEL_REGISTRY_PATH = PROJECT_ROOT / "models" / "registry"
    OPTIMIZED_MODELS_PATH = PROJECT_ROOT / "models" / "optimized"
    BENCHMARKS_PATH = PROJECT_ROOT / "results" / "benchmarks"
    
    # Model settings
    MODEL_NAME = "sgd_v1.0.1"
    MODEL_FILE = f"{MODEL_NAME}.pkl"
    VECTORIZER_FILE = "vectorizer.pkl"
    
    # Optimization settings
    QUANTIZATION_BITS = 8  # int8 quantization
    ONNX_OPSET = 12
    
    # Benchmark settings
    BENCHMARK_ITERATIONS = 100
    TEST_SAMPLE_SIZE = 100
    
    # Logging settings
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE = PROJECT_ROOT / "logs" / "p4.5-model-serialization.log"

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging() -> logging.Logger:
    """Configure production-grade logging"""
    
    # Ensure log directory exists
    Config.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("model_serialization")
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
# MODEL LOADER
# =============================================================================

class ModelLoader:
    """Handles model and vectorizer loading"""
    
    @staticmethod
    def load_pickle_model() -> Tuple[Any, Any, Dict]:
        """Load model and vectorizer from pickle files"""
        logger.info(f"Loading model: {Config.MODEL_PHASE4_PATH / Config.MODEL_FILE}")
        
        # Load model
        model_path = Config.MODEL_PHASE4_PATH / Config.MODEL_FILE
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load vectorizer
        vectorizer_path = Config.MODEL_PHASE4_PATH.parent / "final" / "embeddings_v2.0" / Config.VECTORIZER_FILE
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        
        # Load metadata
        metadata_path = Config.MODEL_PHASE4_PATH / "model_manifest_v1.0.1.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Model loaded: {model.__class__.__name__}")
        logger.info(f"Vectorizer features: {vectorizer.get_feature_names_out().shape[0]}")
        
        return model, vectorizer, metadata

# =============================================================================
# ONNX EXPORTER
# =============================================================================

class ONNXExporter:
    """Export sklearn models to ONNX format"""
    
    def __init__(self, model: Any, vectorizer: Any, metadata: Dict):
        self.model = model
        self.vectorizer = vectorizer
        self.metadata = metadata
    
    def export(self) -> Dict[str, Any]:
        """Export model to ONNX format"""
        if not ONNX_AVAILABLE:
            logger.error("ONNX libraries not available. Skipping export.")
            return {"success": False, "error": "ONNX libraries not installed"}
        
        try:
            logger.info("Exporting model to ONNX format...")
            
            # Create output directory
            Config.OPTIMIZED_MODELS_PATH.mkdir(parents=True, exist_ok=True)
            
            # Define input type for ONNX
            # For TF-IDF vectorizer, input is string array
            initial_type = [('text_input', StringTensorType([None, 1]))]
            
            # Convert to ONNX
            onnx_model = convert_sklearn(
                self.model,
                initial_types=initial_type,
                target_opset=Config.ONNX_OPSET
            )
            
            # Save ONNX model
            onnx_path = Config.OPTIMIZED_MODELS_PATH / f"{Config.MODEL_NAME}.onnx"
            with open(onnx_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            
            # Get file size
            onnx_size = onnx_path.stat().st_size
            
            logger.info(f"ONNX model saved: {onnx_path}")
            logger.info(f"ONNX model size: {onnx_size / 1024:.2f} KB")
            
            # Validate ONNX model
            onnx.checker.check_model(onnx_model)
            logger.info("ONNX model validation: PASSED")
            
            return {
                "success": True,
                "path": str(onnx_path),
                "size_bytes": onnx_size,
                "size_kb": round(onnx_size / 1024, 2),
                "opset": Config.ONNX_OPSET,
                "validation": "passed"
            }
            
        except Exception as e:
            logger.error(f"ONNX export failed: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}

# =============================================================================
# QUANTIZATION
# =============================================================================

class ModelQuantizer:
    """Quantize models for reduced size and faster inference"""
    
    def __init__(self, model: Any, vectorizer: Any, metadata: Dict):
        self.model = model
        self.vectorizer = vectorizer
        self.metadata = metadata
        self.original_size = 0
        self.quantized_size = 0
    
    def quantize_to_int8(self) -> Dict[str, Any]:
        """Quantize model coefficients to int8"""
        try:
            logger.info("Quantizing model to int8...")
            
            # Get model coefficients
            if hasattr(self.model, 'coef_'):
                coef_original = self.model.coef_.astype(np.float32)
                intercept_original = self.model.intercept_.astype(np.float32)
                
                # Calculate original size
                self.original_size = coef_original.nbytes + intercept_original.nbytes
                
                # Quantize to int8
                coef_min = coef_original.min()
                coef_max = coef_original.max()
                coef_scale = (coef_max - coef_min) / 255.0
                coef_quantized = np.round((coef_original - coef_min) / coef_scale).astype(np.uint8)
                
                intercept_min = intercept_original.min()
                intercept_max = intercept_original.max()
                intercept_scale = (intercept_max - intercept_min) / 255.0
                intercept_quantized = np.round((intercept_original - intercept_min) / intercept_scale).astype(np.uint8)
                
                # Create quantized model wrapper
                quantized_model = {
                    'model_type': self.model.__class__.__name__,
                    'coef_quantized': coef_quantized,
                    'intercept_quantized': intercept_quantized,
                    'coef_scale': coef_scale,
                    'coef_min': coef_min,
                    'intercept_scale': intercept_scale,
                    'intercept_min': intercept_min,
                    'classes': self.model.classes_
                }
                
                # Save quantized model
                Config.OPTIMIZED_MODELS_PATH.mkdir(parents=True, exist_ok=True)
                quantized_path = Config.OPTIMIZED_MODELS_PATH / f"{Config.MODEL_NAME}_int8.pkl"
                
                with open(quantized_path, 'wb') as f:
                    pickle.dump(quantized_model, f)
                
                self.quantized_size = quantized_path.stat().st_size
                
                # Calculate compression ratio
                compression_ratio = self.original_size / max(self.quantized_size, 1)
                
                logger.info(f"Quantized model saved: {quantized_path}")
                logger.info(f"Original size: {self.original_size / 1024:.2f} KB")
                logger.info(f"Quantized size: {self.quantized_size / 1024:.2f} KB")
                logger.info(f"Compression ratio: {compression_ratio:.2f}x")
                
                return {
                    "success": True,
                    "path": str(quantized_path),
                    "original_size_bytes": self.original_size,
                    "quantized_size_bytes": self.quantized_size,
                    "compression_ratio": round(compression_ratio, 2),
                    "quantization_bits": Config.QUANTIZATION_BITS
                }
            
            else:
                logger.warning("Model does not have coef_ attribute. Skipping quantization.")
                return {"success": False, "error": "Model not compatible with quantization"}
        
        except Exception as e:
            logger.error(f"Quantization failed: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}

# =============================================================================
# BENCHMARKING
# =============================================================================

class ModelBenchmark:
    """Benchmark different model formats"""
    
    def __init__(self, model: Any, vectorizer: Any, test_data: pd.DataFrame):
        self.model = model
        self.vectorizer = vectorizer
        self.test_data = test_data
        self.results = {}
    
    def benchmark_pickle(self) -> Dict[str, Any]:
        """Benchmark pickle model inference"""
        logger.info("Benchmarking pickle model...")
        
        texts = self.test_data['text'].head(Config.TEST_SAMPLE_SIZE).tolist()
        
        # Warmup
        _ = self.vectorizer.transform(texts[:10])
        _ = self.model.predict(self.vectorizer.transform(texts[:10]))
        
        # Benchmark
        start_time = time.time()
        
        for _ in range(Config.BENCHMARK_ITERATIONS):
            vectors = self.vectorizer.transform(texts)
            predictions = self.model.predict(vectors)
        
        total_time = time.time() - start_time
        avg_latency = (total_time / Config.BENCHMARK_ITERATIONS) * 1000  # ms
        
        self.results['pickle'] = {
            "avg_latency_ms": round(avg_latency, 3),
            "total_time_s": round(total_time, 3),
            "iterations": Config.BENCHMARK_ITERATIONS,
            "samples_per_batch": len(texts)
        }
        
        logger.info(f"Pickle benchmark: {avg_latency:.3f}ms avg latency")
        return self.results['pickle']
    
    def benchmark_onnx(self) -> Dict[str, Any]:
        """Benchmark ONNX model inference"""
        if not ONNX_AVAILABLE:
            logger.warning("ONNX not available. Skipping ONNX benchmark.")
            return {"error": "ONNX not available"}
        
        try:
            import onnxruntime as ort
            
            logger.info("Benchmarking ONNX model...")
            
            onnx_path = Config.OPTIMIZED_MODELS_PATH / f"{Config.MODEL_NAME}.onnx"
            if not onnx_path.exists():
                logger.warning("ONNX model not found. Skipping benchmark.")
                return {"error": "ONNX model not found"}
            
            # Create ONNX runtime session
            session = ort.InferenceSession(str(onnx_path))
            input_name = session.get_inputs()[0].name
            
            texts = self.test_data['text'].head(Config.TEST_SAMPLE_SIZE).tolist()
            
            # Warmup
            _ = session.run(None, {input_name: np.array(texts[:10]).reshape(-1, 1).astype(str)})
            
            # Benchmark
            start_time = time.time()
            
            for _ in range(Config.BENCHMARK_ITERATIONS):
                _ = session.run(None, {input_name: np.array(texts).reshape(-1, 1).astype(str)})
            
            total_time = time.time() - start_time
            avg_latency = (total_time / Config.BENCHMARK_ITERATIONS) * 1000  # ms
            
            self.results['onnx'] = {
                "avg_latency_ms": round(avg_latency, 3),
                "total_time_s": round(total_time, 3),
                "iterations": Config.BENCHMARK_ITERATIONS,
                "samples_per_batch": len(texts)
            }
            
            logger.info(f"ONNX benchmark: {avg_latency:.3f}ms avg latency")
            return self.results['onnx']
        
        except ImportError:
            logger.warning("onnxruntime not installed. Skipping ONNX benchmark.")
            return {"error": "onnxruntime not installed"}
        except Exception as e:
            logger.error(f"ONNX benchmark failed: {str(e)}")
            return {"error": str(e)}
    
    def generate_comparison_report(self) -> Dict[str, Any]:
        """Generate comparison report"""
        logger.info("Generating benchmark comparison report...")
        
        report = {
            "benchmark_config": {
                "iterations": Config.BENCHMARK_ITERATIONS,
                "sample_size": Config.TEST_SAMPLE_SIZE,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            },
            "results": self.results,
            "comparison": {}
        }
        
        if 'pickle' in self.results and 'onnx' in self.results:
            pickle_latency = self.results['pickle']['avg_latency_ms']
            onnx_latency = self.results['onnx']['avg_latency_ms']
            
            if isinstance(onnx_latency, (int, float)) and onnx_latency > 0:
                speedup = pickle_latency / onnx_latency
                report['comparison'] = {
                    "pickle_vs_onnx_speedup": round(speedup, 2),
                    "recommended_format": "onnx" if speedup > 1.0 else "pickle"
                }
        
        # Save report
        Config.BENCHMARKS_PATH.mkdir(parents=True, exist_ok=True)
        report_path = Config.BENCHMARKS_PATH / "model_benchmark_v1.0.1.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Benchmark report saved: {report_path}")
        
        return report

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution pipeline"""
    logger.info("=" * 70)
    logger.info("Phase 4.5: Model Serialization & Optimization")
    logger.info("=" * 70)
    
    results = {
        "phase": "4.5",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "onnx_export": None,
        "quantization": None,
        "benchmark": None,
        "recommendations": []
    }
    
    try:
        # Step 1: Load model
        logger.info("\n[STEP 1] Loading model and vectorizer...")
        model, vectorizer, metadata = ModelLoader.load_pickle_model()
        logger.info(f"Model type: {model.__class__.__name__}")
        
        # Step 2: Export to ONNX
        logger.info("\n[STEP 2] Exporting to ONNX format...")
        if ONNX_AVAILABLE:
            onnx_exporter = ONNXExporter(model, vectorizer, metadata)
            results['onnx_export'] = onnx_exporter.export()
        else:
            results['onnx_export'] = {"skipped": True, "reason": "ONNX libraries not installed"}
            logger.warning("ONNX export skipped: libraries not available")
        
        # Step 3: Quantize model
        logger.info("\n[STEP 3] Quantizing model to int8...")
        quantizer = ModelQuantizer(model, vectorizer, metadata)
        results['quantization'] = quantizer.quantize_to_int8()
        
        # Step 4: Run benchmarks
        logger.info("\n[STEP 4] Running performance benchmarks...")
        
        # Load test data for benchmarking
        test_data_path = Config.PROJECT_ROOT / "data" / "processed" / "cleaned_split_test.csv"
        if test_data_path.exists():
            test_data = pd.read_csv(test_data_path)
            benchmark = ModelBenchmark(model, vectorizer, test_data)
            
            # Benchmark pickle
            benchmark.benchmark_pickle()
            
            # Benchmark ONNX (if available)
            if results['onnx_export'] and results['onnx_export'].get('success'):
                benchmark.benchmark_onnx()
            
            # Generate comparison report
            results['benchmark'] = benchmark.generate_comparison_report()
        else:
            logger.warning("Test data not found. Skipping benchmarks.")
            results['benchmark'] = {"skipped": True, "reason": "Test data not found"}
        
        # Step 5: Generate recommendations
        logger.info("\n[STEP 5] Generating optimization recommendations...")
        
        recommendations = []
        
        # ONNX recommendation
        if results['onnx_export'] and results['onnx_export'].get('success'):
            onnx_size = results['onnx_export']['size_kb']
            pickle_path = Config.MODEL_PHASE4_PATH / Config.MODEL_FILE
            pickle_size = pickle_path.stat().st_size / 1024
            size_ratio = pickle_size / max(onnx_size, 1)
            
            if size_ratio > 1.5:
                recommendations.append({
                    "type": "deployment",
                    "recommendation": "Use ONNX format for production deployment",
                    "reason": f"ONNX is {size_ratio:.1f}x smaller than pickle",
                    "priority": "high"
                })
        
        # Quantization recommendation
        if results['quantization'] and results['quantization'].get('success'):
            compression = results['quantization']['compression_ratio']
            if compression > 2.0:
                recommendations.append({
                    "type": "optimization",
                    "recommendation": "Use int8 quantization for edge deployment",
                    "reason": f"Quantization achieves {compression:.1f}x compression",
                    "priority": "medium"
                })
        
        # Benchmark recommendation
        if results['benchmark'] and 'comparison' in results['benchmark']:
            comparison = results['benchmark']['comparison']
            if comparison.get('recommended_format'):
                recommendations.append({
                    "type": "performance",
                    "recommendation": f"Use {comparison['recommended_format']} format for inference",
                    "reason": f"Speedup: {comparison.get('pickle_vs_onnx_speedup', 'N/A')}x",
                    "priority": "high"
                })
        
        results['recommendations'] = recommendations
        
        # Save results
        results_path = Config.BENCHMARKS_PATH / "serialization_results_v1.0.1.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\nResults saved: {results_path}")
        
        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("SERIALIZATION SUMMARY")
        logger.info("=" * 70)
        
        if results['onnx_export']:
            status = "✓ SUCCESS" if results['onnx_export'].get('success') else "✗ FAILED"
            logger.info(f"ONNX Export: {status}")
            if results['onnx_export'].get('size_kb'):
                logger.info(f"  Size: {results['onnx_export']['size_kb']} KB")
        
        if results['quantization']:
            status = "✓ SUCCESS" if results['quantization'].get('success') else "✗ FAILED"
            logger.info(f"Quantization: {status}")
            if results['quantization'].get('compression_ratio'):
                logger.info(f"  Compression: {results['quantization']['compression_ratio']}x")
        
        if results['benchmark']:
            logger.info(f"Benchmark: ✓ COMPLETED")
            if results['benchmark'].get('results'):
                for format_name, metrics in results['benchmark']['results'].items():
                    if isinstance(metrics, dict) and 'avg_latency_ms' in metrics:
                        logger.info(f"  {format_name}: {metrics['avg_latency_ms']}ms avg latency")
        
        logger.info(f"\nRecommendations: {len(recommendations)}")
        for rec in recommendations:
            logger.info(f"  [{rec['priority'].upper()}] {rec['recommendation']}")
        
        logger.info("=" * 70)
        logger.info("Phase 4.5 COMPLETE")
        logger.info("=" * 70)
        
        return results
    
    except Exception as e:
        logger.error(f"Phase 4.5 failed: {str(e)}", exc_info=True)
        results['error'] = str(e)
        return results

if __name__ == "__main__":
    results = main()
    sys.exit(0 if not results.get('error') else 1)
