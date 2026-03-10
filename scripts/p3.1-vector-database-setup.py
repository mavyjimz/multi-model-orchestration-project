#!/usr/bin/env python3
"""
p3.1: Vector Database Setup with FAISS
=======================================
Creates FAISS index for similarity search using versioned embeddings.

Inputs:
  - data/final/embeddings_v2.0/train_emb.npy
  - data/final/embeddings_v2.0/val_emb.npy
  - data/final/embeddings_v2.0/test_emb.npy
  - data/final/embeddings_v2.0/all_index_maps.pkl
  
Outputs:
  - data/vector_db/faiss_index_v1.0/index.faiss
  - data/vector_db/faiss_index_v1.0/index_metadata.json
  - logs/p3.1_index_creation_report.json

Author: MLOps Team
Date: 2026-03-10
"""

import os
import sys
import json
import time
import hashlib
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import faiss
    print(f"FAISS version: {faiss.__version__}")
except ImportError:
    print("ERROR: FAISS not installed. Installing...")
    os.system("pip install faiss-cpu")
    import faiss


class FAISSVectorDatabase:
    """FAISS vector database for similarity search."""
    
    def __init__(self, config):
        self.config = config
        self.index = None
        self.metadata = {}
        self.index_maps = None
        
    def load_embeddings(self):
        """Load versioned embeddings from storage."""
        embed_dir = Path(self.config['embeddings_path'])
        
        print(f"Loading embeddings from {embed_dir}...")
        
        # Load embedding arrays
        train_emb = np.load(embed_dir / 'train_emb.npy')
        val_emb = np.load(embed_dir / 'val_emb.npy')
        test_emb = np.load(embed_dir / 'test_emb.npy')
        
        # Load index maps
        with open(embed_dir / 'all_index_maps.pkl', 'rb') as f:
            self.index_maps = pickle.load(f)
        
        print(f"  Train embeddings: {train_emb.shape}")
        print(f"  Val embeddings: {val_emb.shape}")
        print(f"  Test embeddings: {test_emb.shape}")
        
        # Ensure float32 for FAISS
        train_emb = train_emb.astype(np.float32)
        val_emb = val_emb.astype(np.float32)
        test_emb = test_emb.astype(np.float32)
        
        return train_emb, val_emb, test_emb
    
    def create_index(self, dimension):
        """Create FAISS index based on configuration."""
        index_type = self.config.get('index_type', 'flat')
        
        print(f"Creating {index_type.upper()} FAISS index...")
        
        if index_type == 'flat':
            # Exact search using L2 distance
            self.index = faiss.IndexFlatL2(dimension)
            print(f"  Index type: IndexFlatL2 (exact search)")
            
        elif index_type == 'ivf':
            # Approximate search using IVF
            nlist = self.config.get('nlist', 100)  # Number of Voronoi cells
            
            # Create IVF index
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist, 
                                           faiss.METRIC_L2)
            print(f"  Index type: IndexIVFFlat (approximate search)")
            print(f"  Number of cells (nlist): {nlist}")
            
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        return self.index
    
    def train_index(self, train_embeddings):
        """Train index if required (for IVF indices)."""
        if isinstance(self.index, faiss.IndexIVFFlat):
            print("Training IVF index...")
            self.index.train(train_embeddings)
            print("  Training complete")
    
    def add_embeddings(self, embeddings):
        """Add embeddings to the index."""
        print(f"Adding {len(embeddings)} embeddings to index...")
        self.index.add(embeddings)
        print(f"  Index size: {self.index.ntotal} vectors")
    
    def save_index(self, output_path):
        """Save FAISS index to disk."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        index_file = output_path / 'index.faiss'
        print(f"Saving index to {index_file}...")
        
        faiss.write_index(self.index, str(index_file))
        
        # Get file size
        file_size = index_file.stat().st_size
        print(f"  Index size: {file_size / 1024 / 1024:.2f} MB")
        
        # Calculate checksum
        checksum = self._calculate_checksum(index_file)
        
        return index_file, checksum
    
    def _calculate_checksum(self, filepath):
        """Calculate SHA256 checksum of file."""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def create_metadata(self, train_emb, val_emb, test_emb, checksum):
        """Create metadata for the index."""
        self.metadata = {
            'version': '1.0',
            'created_at': datetime.now().isoformat(),
            'index_type': self.config['index_type'],
            'embedding_dimension': train_emb.shape[1],
            'total_vectors': self.index.ntotal,
            'dataset_splits': {
                'train': len(train_emb),
                'val': len(val_emb),
                'test': len(test_emb)
            },
            'index_maps': {
                'train_start': 0,
                'train_end': len(train_emb),
                'val_start': len(train_emb),
                'val_end': len(train_emb) + len(val_emb),
                'test_start': len(train_emb) + len(val_emb),
                'test_end': self.index.ntotal
            },
            'config': self.config,
            'checksum': checksum,
            'hardware': {
                'ram_gb': self.config.get('ram_gb', 8),
                'cpu_only': True
            }
        }
        
        return self.metadata
    
    def save_metadata(self, output_path):
        """Save metadata to JSON file."""
        output_path = Path(output_path)
        metadata_file = output_path / 'index_metadata.json'
        
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"Metadata saved to {metadata_file}")
        return metadata_file
    
    def validate_index(self, test_embeddings):
        """Validate index with basic sanity checks."""
        print("Validating index...")
        
        validation_results = {
            'index_size': self.index.ntotal,
            'expected_size': self.metadata['total_vectors'],
            'dimension': self.index.d,
            'is_trained': self.index.is_trained if hasattr(self.index, 'is_trained') else True,
            'test_queries': []
        }
        
        # Test a few random queries
        n_test_queries = min(5, len(test_embeddings))
        test_indices = np.random.choice(len(test_embeddings), n_test_queries, replace=False)
        
        for idx in test_indices:
            query = test_embeddings[idx:idx+1]
            D, I = self.index.search(query, k=5)
            
            validation_results['test_queries'].append({
                'query_index': int(idx),
                'distances': D[0].tolist(),
                'neighbors': I[0].tolist()
            })
        
        # Check if validation passed
        validation_results['passed'] = (
            validation_results['index_size'] == validation_results['expected_size']
            and validation_results['dimension'] == self.metadata['embedding_dimension']
        )
        
        print(f"  Index size: {validation_results['index_size']} (expected: {validation_results['expected_size']})")
        print(f"  Dimension: {validation_results['dimension']}")
        print(f"  Validation: {'PASSED' if validation_results['passed'] else 'FAILED'}")
        
        return validation_results


def main():
    """Main execution function."""
    print("=" * 70)
    print("Phase 3.1: Vector Database Setup with FAISS")
    print("=" * 70)
    
    start_time = time.time()
    
    # Configuration
    config = {
        'embeddings_path': 'data/final/embeddings_v2.0',
        'output_path': 'data/vector_db/faiss_index_v1.0',
        'index_type': 'flat',  # 'flat' or 'ivf'
        'nlist': 100,  # For IVF index
        'ram_gb': 8
    }
    
    # Load config from YAML if available
    config_file = project_root / 'config' / 'config.yaml'
    if config_file.exists():
        import yaml
        with open(config_file, 'r') as f:
            yaml_config = yaml.safe_load(f)
            # Merge YAML config with defaults
            if 'vector_db' in yaml_config:
                config.update(yaml_config['vector_db'])
    
    # Create vector database
    vdb = FAISSVectorDatabase(config)
    
    # Step 1: Load embeddings
    print("\n[1/6] Loading embeddings...")
    train_emb, val_emb, test_emb = vdb.load_embeddings()
    
    # Step 2: Create index
    print("\n[2/6] Creating FAISS index...")
    dimension = train_emb.shape[1]
    vdb.create_index(dimension)
    
    # Step 3: Train index (if needed)
    print("\n[3/6] Training index...")
    vdb.train_index(train_emb)
    
    # Step 4: Add embeddings
    print("\n[4/6] Adding embeddings to index...")
    # Combine all embeddings
    all_embeddings = np.vstack([train_emb, val_emb, test_emb])
    vdb.add_embeddings(all_embeddings)
    
    # Step 5: Save index
    print("\n[5/6] Saving index...")
    index_file, checksum = vdb.save_index(config['output_path'])
    
    # Step 6: Create and save metadata
    print("\n[6/6] Creating metadata...")
    vdb.create_metadata(train_emb, val_emb, test_emb, checksum)
    metadata_file = vdb.save_metadata(config['output_path'])
    
    # Validate index
    print("\n" + "=" * 70)
    print("Validation")
    print("=" * 70)
    validation_results = vdb.validate_index(test_emb)
    
    # Generate creation report
    elapsed_time = time.time() - start_time
    
    creation_report = {
        'phase': '3.1',
        'timestamp': datetime.now().isoformat(),
        'elapsed_time_seconds': elapsed_time,
        'config': config,
        'metadata': vdb.metadata,
        'validation': validation_results,
        'files_created': [
            str(index_file),
            str(metadata_file)
        ]
    }
    
    # Save creation report
    logs_dir = project_root / 'logs'
    logs_dir.mkdir(exist_ok=True)
    report_file = logs_dir / 'p3.1_index_creation_report.json'
    
    with open(report_file, 'w') as f:
        json.dump(creation_report, f, indent=2)
    
    print(f"\nCreation report saved to {report_file}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Phase 3.1 Complete")
    print("=" * 70)
    print(f"Index type: {config['index_type'].upper()}")
    print(f"Total vectors: {vdb.index.ntotal}")
    print(f"Dimension: {dimension}")
    print(f"Index size: {index_file.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print(f"Validation: {'PASSED' if validation_results['passed'] else 'FAILED'}")
    print("=" * 70)
    
    return 0 if validation_results['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
