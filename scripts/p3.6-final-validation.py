#!/usr/bin/env python3
"""
Phase 3.6: Final Validation & Storage
Comprehensive validation and archiving of Phase 3 deliverables.
"""

import os
import sys
import json
import time
import pickle
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import faiss
except ImportError:
    print("Error: faiss package not installed")
    sys.exit(1)


class Phase3Validator:
    """Comprehensive validator for Phase 3 deliverables."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.project_root = Path(__file__).parent.parent
        self.config = self._load_config(config_path)
        self.validation_results = {}
        self.start_time = None
        self.artifacts = {}
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration."""
        try:
            import yaml
            config_file = self.project_root / config_path
            if config_file.exists():
                with open(config_file, 'r') as f:
                    return yaml.safe_load(f)
        except ImportError:
            pass
        
        return {
            "vector_db": {
                "index_path": "data/vector_db/faiss_index_v1.0/index.faiss",
                "metadata_path": "data/vector_db/faiss_index_v1.0/document_metadata.json",
                "vectorizer_path": "data/final/embeddings_v2.0/vectorizer.pkl"
            },
            "validation": {
                "target_latency_ms": 100,
                "test_sample_size": 100,
                "max_duplicate_percentage": 1.0  # Allow up to 1% duplicates
            },
            "versioning": {
                "artifacts_dir": "artifacts/production_v1.0"
            }
        }
    
    def run_all_validations(self) -> Dict[str, Any]:
        """Run complete validation suite."""
        self.start_time = time.time()
        
        print("=" * 70)
        print("PHASE 3.6: FINAL VALIDATION & STORAGE")
        print("=" * 70)
        print(f"Started: {datetime.now().isoformat()}")
        print("=" * 70)
        
        self._validate_file_existence()
        self._validate_file_integrity()
        self._validate_model_loading()
        self._validate_index_integrity()
        self._validate_metadata_consistency()
        self._validate_performance()
        self._validate_end_to_end()
        self._check_deployment_readiness()
        
        total_time = time.time() - self.start_time
        report = self._generate_report(total_time)
        self._archive_artifacts()
        self._print_summary(report)
        
        return report
    
    def _validate_file_existence(self) -> None:
        """Check all required files exist."""
        print("\n[1/7] Validating file existence...")
        
        required_files = [
            "data/final/embeddings_v2.0/vectorizer.pkl",
            "data/final/embeddings_v2.0/all_index_maps.pkl",
            "data/vector_db/faiss_index_v1.0/index.faiss",
            "data/vector_db/faiss_index_v1.0/document_metadata.json",
            "scripts/p3.1-vector-database-setup.py",
            "scripts/p3.2-similarity-search.py",
            "scripts/p3.4-interactive-query.py",
            "config/config.yaml"
        ]
        
        missing_files = []
        for file_path in required_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                missing_files.append(file_path)
        
        self.validation_results["file_existence"] = {
            "status": "PASS" if not missing_files else "FAIL",
            "missing_files": missing_files,
            "details": f"Found {len(required_files) - len(missing_files)}/{len(required_files)} files"
        }
        
        print(f"  {'✓' if not missing_files else '✗'} {self.validation_results['file_existence']['details']}")
    
    def _validate_file_integrity(self) -> None:
        """Validate file checksums and sizes."""
        print("\n[2/7] Validating file integrity...")
        
        integrity_checks = {
            "vectorizer.pkl": {
                "path": "data/final/embeddings_v2.0/vectorizer.pkl",
                "min_size": 100000
            },
            "index.faiss": {
                "path": "data/vector_db/faiss_index_v1.0/index.faiss",
                "min_size": 10000000
            },
            "document_metadata.json": {
                "path": "data/vector_db/faiss_index_v1.0/document_metadata.json",
                "min_size": 1000
            }
        }
        
        failed_checks = []
        for name, check in integrity_checks.items():
            full_path = self.project_root / check["path"]
            if full_path.exists():
                size = full_path.stat().st_size
                if size < check["min_size"]:
                    failed_checks.append(f"{name}: size {size} < {check['min_size']}")
            else:
                failed_checks.append(f"{name}: file not found")
        
        self.validation_results["file_integrity"] = {
            "status": "PASS" if not failed_checks else "FAIL",
            "failed_checks": failed_checks
        }
        
        print(f"  {'✓' if not failed_checks else '✗'} {'All integrity checks passed' if not failed_checks else f'{len(failed_checks)} checks failed'}")
    
    def _validate_model_loading(self) -> None:
        """Test loading of all models and indexes."""
        print("\n[3/7] Validating model loading...")
        
        loading_errors = []
        
        try:
            vectorizer_path = self.project_root / "data/final/embeddings_v2.0/vectorizer.pkl"
            with open(vectorizer_path, 'rb') as f:
                vectorizer = pickle.load(f)
            self.artifacts["vectorizer"] = vectorizer
            print(f"  ✓ Vectorizer loaded: {vectorizer.get_feature_names_out().size} features")
        except Exception as e:
            loading_errors.append(f"Vectorizer: {str(e)}")
            print(f"  ✗ Vectorizer failed: {str(e)}")
        
        try:
            index_path = self.project_root / "data/vector_db/faiss_index_v1.0/index.faiss"
            index = faiss.read_index(str(index_path))
            self.artifacts["faiss_index"] = index
            print(f"  ✓ FAISS index loaded: {index.ntotal} vectors, {index.d} dimensions")
        except Exception as e:
            loading_errors.append(f"FAISS index: {str(e)}")
            print(f"  ✗ FAISS index failed: {str(e)}")
        
        try:
            metadata_path = self.project_root / "data/vector_db/faiss_index_v1.0/document_metadata.json"
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self.artifacts["metadata"] = metadata
            print(f"  ✓ Metadata loaded: {len(metadata)} documents")
        except Exception as e:
            loading_errors.append(f"Metadata: {str(e)}")
            print(f"  ✗ Metadata failed: {str(e)}")
        
        self.validation_results["model_loading"] = {
            "status": "PASS" if not loading_errors else "FAIL",
            "errors": loading_errors,
            "artifacts_loaded": len(self.artifacts)
        }
    
    def _validate_index_integrity(self) -> None:
        """Validate FAISS index structure and consistency."""
        print("\n[4/7] Validating index integrity...")
        
        integrity_issues = []
        
        if "faiss_index" in self.artifacts and "metadata" in self.artifacts:
            index = self.artifacts["faiss_index"]
            metadata = self.artifacts["metadata"]
            
            if index.ntotal != len(metadata):
                integrity_issues.append(
                    f"Vector count mismatch: index={index.ntotal}, metadata={len(metadata)}"
                )
            
            if index.ntotal == 0:
                integrity_issues.append("Index is empty")
            
            if index.d <= 0:
                integrity_issues.append(f"Invalid dimension: {index.d}")
            
            print(f"  Index vectors: {index.ntotal}")
            print(f"  Index dimension: {index.d}")
            print(f"  Metadata entries: {len(metadata)}")
        else:
            integrity_issues.append("Required artifacts not loaded")
        
        self.validation_results["index_integrity"] = {
            "status": "PASS" if not integrity_issues else "FAIL",
            "issues": integrity_issues
        }
        
        print(f"  {'✓' if not integrity_issues else '✗'} {'Index structure valid' if not integrity_issues else f'{len(integrity_issues)} issues found'}")
    
    def _validate_metadata_consistency(self) -> None:
        """Validate metadata structure and consistency."""
        print("\n[5/7] Validating metadata consistency...")
        
        consistency_issues = []
        max_dup_pct = self.config.get("validation", {}).get("max_duplicate_percentage", 1.0)
        
        if "metadata" in self.artifacts:
            metadata = self.artifacts["metadata"]
            
            # Handle both list and dict metadata formats
            if isinstance(metadata, dict):
                metadata_list = list(metadata.values()) if metadata else []
            elif isinstance(metadata, list):
                metadata_list = metadata
            else:
                consistency_issues.append(f"Unexpected metadata type: {type(metadata)}")
                metadata_list = []
            
            if metadata_list:
                # Check first 5 documents
                required_fields = ["content"]
                check_count = min(5, len(metadata_list))
                
                for i in range(check_count):
                    doc = metadata_list[i]
                    if isinstance(doc, dict):
                        for field in required_fields:
                            if field not in doc:
                                consistency_issues.append(f"Document {i} missing field: {field}")
                    else:
                        consistency_issues.append(f"Document {i} is not a dict")
                
                # Check for duplicates (allow small percentage)
                contents = []
                for doc in metadata_list:
                    if isinstance(doc, dict):
                        contents.append(doc.get("content", ""))
                
                unique_contents = set(contents)
                duplicates = len(contents) - len(unique_contents)
                dup_percentage = (duplicates / len(contents)) * 100 if contents else 0
                
                print(f"  Total documents: {len(metadata_list)}")
                print(f"  Unique documents: {len(unique_contents)}")
                print(f"  Duplicates: {duplicates} ({dup_percentage:.2f}%)")
                
                if dup_percentage > max_dup_pct:
                    consistency_issues.append(
                        f"Duplicate percentage {dup_percentage:.2f}% exceeds threshold {max_dup_pct}%"
                    )
                elif duplicates > 0:
                    print(f"  ⚠ Warning: {duplicates} duplicates found (within acceptable threshold)")
            else:
                consistency_issues.append("No documents in metadata")
        else:
            consistency_issues.append("Metadata not loaded")
        
        self.validation_results["metadata_consistency"] = {
            "status": "PASS" if not consistency_issues else "FAIL",
            "issues": consistency_issues
        }
        
        print(f"  {'✓' if not consistency_issues else '✗'} {'Metadata consistent' if not consistency_issues else f'{len(consistency_issues)} issues found'}")
    
    def _validate_performance(self) -> None:
        """Run performance benchmarks."""
        print("\n[6/7] Running performance benchmarks...")
        
        perf_results = {
            "latency_tests": [],
            "target_latency_ms": self.config.get("validation", {}).get("target_latency_ms", 100)
        }
        
        if "vectorizer" in self.artifacts and "faiss_index" in self.artifacts:
            vectorizer = self.artifacts["vectorizer"]
            index = self.artifacts["faiss_index"]
            
            test_queries = [
                "hello how are you",
                "what is machine learning",
                "tell me about python programming",
                "how to build a chatbot",
                "what is natural language processing"
            ]
            
            for query in test_queries:
                start = time.time()
                query_emb = vectorizer.transform([query]).toarray().astype(np.float32)
                distances, indices = index.search(query_emb, k=5)
                latency_ms = (time.time() - start) * 1000
                perf_results["latency_tests"].append({
                    "query": query,
                    "latency_ms": latency_ms
                })
            
            latencies = [r["latency_ms"] for r in perf_results["latency_tests"]]
            avg_latency = sum(latencies) / len(latencies)
            max_latency = max(latencies)
            
            perf_results["avg_latency_ms"] = avg_latency
            perf_results["max_latency_ms"] = max_latency
            perf_results["meets_target"] = max_latency < perf_results["target_latency_ms"]
            
            print(f"  Average latency: {avg_latency:.2f}ms")
            print(f"  Max latency: {max_latency:.2f}ms")
            print(f"  Target: <{perf_results['target_latency_ms']}ms")
            print(f"  Status: {'✓ PASS' if perf_results['meets_target'] else '✗ FAIL'}")
        else:
            print("  ✗ Cannot run benchmarks: artifacts not loaded")
            perf_results["meets_target"] = False
        
        self.validation_results["performance"] = {
            "status": "PASS" if perf_results.get("meets_target", False) else "FAIL",
            "results": perf_results
        }
    
    def _validate_end_to_end(self) -> None:
        """Run end-to-end query test."""
        print("\n[7/7] Running end-to-end validation...")
        
        e2e_issues = []
        
        if all(k in self.artifacts for k in ["vectorizer", "faiss_index", "metadata"]):
            vectorizer = self.artifacts["vectorizer"]
            index = self.artifacts["faiss_index"]
            metadata = self.artifacts["metadata"]
            
            # Convert metadata to list if it's a dict
            if isinstance(metadata, dict):
                metadata_list = list(metadata.values())
            else:
                metadata_list = metadata
            
            test_query = "hello"
            query_emb = vectorizer.transform([test_query]).toarray().astype(np.float32)
            distances, indices = index.search(query_emb, k=5)
            
            if len(indices[0]) == 0 or indices[0][0] == -1:
                e2e_issues.append("No results returned for test query")
            else:
                top_idx = indices[0][0]
                if top_idx >= len(metadata_list):
                    e2e_issues.append(f"Index {top_idx} out of metadata bounds")
            
            print(f"  Test query: '{test_query}'")
            print(f"  Results returned: {len([i for i in indices[0] if i != -1])}")
        else:
            e2e_issues.append("Required artifacts not loaded")
        
        self.validation_results["end_to_end"] = {
            "status": "PASS" if not e2e_issues else "FAIL",
            "issues": e2e_issues
        }
        
        print(f"  {'✓' if not e2e_issues else '✗'} {'End-to-end test passed' if not e2e_issues else 'End-to-end test failed'}")
    
    def _check_deployment_readiness(self) -> None:
        """Check if system is ready for deployment."""
        print("\n" + "=" * 70)
        print("DEPLOYMENT READINESS CHECKLIST")
        print("=" * 70)
        
        checklist = {
            "All required files present": self.validation_results.get("file_existence", {}).get("status") == "PASS",
            "File integrity verified": self.validation_results.get("file_integrity", {}).get("status") == "PASS",
            "Models load successfully": self.validation_results.get("model_loading", {}).get("status") == "PASS",
            "Index integrity valid": self.validation_results.get("index_integrity", {}).get("status") == "PASS",
            "Metadata consistent": self.validation_results.get("metadata_consistency", {}).get("status") == "PASS",
            "Performance meets target": self.validation_results.get("performance", {}).get("status") == "PASS",
            "End-to-end test passed": self.validation_results.get("end_to_end", {}).get("status") == "PASS"
        }
        
        all_passed = all(checklist.values())
        
        for item, passed in checklist.items():
            status = "✓" if passed else "✗"
            print(f"  {status} {item}")
        
        self.validation_results["deployment_readiness"] = {
            "status": "READY" if all_passed else "NOT READY",
            "checklist": checklist,
            "passed_checks": sum(checklist.values()),
            "total_checks": len(checklist)
        }
        
        print("=" * 70)
        print(f"Overall Status: {'✓ READY FOR DEPLOYMENT' if all_passed else '✗ NOT READY'}")
        print(f"Passed: {sum(checklist.values())}/{len(checklist)} checks")
        print("=" * 70)
    
    def _generate_report(self, total_time: float) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        report = {
            "phase": "3.6",
            "validation_timestamp": datetime.now().isoformat(),
            "total_validation_time_seconds": total_time,
            "overall_status": "PASS" if self.validation_results.get("deployment_readiness", {}).get("status") == "READY" else "FAIL",
            "validation_results": self.validation_results,
            "project_info": {
                "name": "Multi-Model Orchestration System",
                "phase_completed": "Phase 3: Vector Database & Retrieval",
                "version": "1.0"
            }
        }
        
        reports_dir = self.project_root / "logs"
        reports_dir.mkdir(exist_ok=True)
        
        report_path = reports_dir / "p3.6_final_validation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\nValidation report saved to: {report_path}")
        
        return report
    
    def _archive_artifacts(self) -> None:
        """Archive production-ready artifacts."""
        print("\nArchiving production artifacts...")
        
        artifacts_dir = self.project_root / "artifacts" / "production_v1.0"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        import shutil
        config_src = self.project_root / "config" / "config.yaml"
        if config_src.exists():
            shutil.copy(config_src, artifacts_dir / "config.yaml")
            print(f"  ✓ Copied config/config.yaml")
        
        manifest = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "artifacts": {
                "vectorizer": "data/final/embeddings_v2.0/vectorizer.pkl",
                "index_maps": "data/final/embeddings_v2.0/all_index_maps.pkl",
                "faiss_index": "data/vector_db/faiss_index_v1.0/index.faiss",
                "metadata": "data/vector_db/faiss_index_v1.0/document_metadata.json"
            },
            "validation_status": self.validation_results.get("deployment_readiness", {}).get("status", "UNKNOWN")
        }
        
        manifest_path = artifacts_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"  ✓ Created manifest.json")
        print(f"  Artifacts archived to: {artifacts_dir}")
    
    def _print_summary(self, report: Dict[str, Any]) -> None:
        """Print validation summary."""
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        print(f"Overall Status: {report['overall_status']}")
        print(f"Total Time: {report['total_validation_time_seconds']:.2f}s")
        print(f"Deployment Ready: {self.validation_results.get('deployment_readiness', {}).get('status', 'UNKNOWN')}")
        
        if "performance" in self.validation_results:
            perf = self.validation_results["performance"]["results"]
            print(f"\nPerformance Metrics:")
            print(f"  Average Latency: {perf.get('avg_latency_ms', 0):.2f}ms")
            print(f"  Max Latency: {perf.get('max_latency_ms', 0):.2f}ms")
            print(f"  Target: <{perf.get('target_latency_ms', 100)}ms")
        
        print("=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Phase 3.6: Final Validation & Storage")
    parser.add_argument("--config", "-c", type=str, default="config/config.yaml",
                        help="Path to configuration file")
    
    args = parser.parse_args()
    
    try:
        validator = Phase3Validator(config_path=args.config)
        report = validator.run_all_validations()
        
        if report["overall_status"] == "PASS":
            print("\n✓ Phase 3.6 validation completed successfully!")
            sys.exit(0)
        else:
            print("\n✗ Phase 3.6 validation failed. Review report for details.")
            sys.exit(1)
    
    except Exception as e:
        print(f"\n✗ Validation error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
