#!/usr/bin/env python3
"""
Phase 3.2: Similarity Search Implementation
Implements k-NN retrieval using FAISS index with comprehensive evaluation metrics.
"""

import os
import sys
import json
import time
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import faiss
except ImportError:
    print("Installing faiss-cpu...")
    os.system("pip install faiss-cpu")
    import faiss

from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


class FAISSSimilaritySearch:
    """FAISS-based similarity search with intent retrieval."""
    
    def __init__(self, index_path: str, embeddings_dir: str):
        self.index_path = index_path
        self.embeddings_dir = embeddings_dir
        self.index = None
        self.id_to_intent = {}
        self.vectorizer = None
        
    def load_index(self):
        """Load FAISS index and build intent mapping from original data."""
        print("=" * 70)
        print("LOADING FAISS INDEX AND BUILDING INTENT MAPPING")
        print("=" * 70)
        
        # Load FAISS index
        print(f"\n[1/3] Loading FAISS index from: {self.index_path}")
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        self.index = faiss.read_index(self.index_path)
        print(f"      Index loaded: {self.index.ntotal} vectors, dimension {self.index.d}")
        
        # Load vectorizer
        vectorizer_path = Path(self.embeddings_dir) / "vectorizer.pkl"
        print(f"\n[2/3] Loading vectorizer from: {vectorizer_path}")
        if os.path.exists(vectorizer_path):
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            print("      Vectorizer loaded successfully")
        else:
            print(f"      Warning: Vectorizer not found")
            self.vectorizer = None
        
        # Build intent mapping from original CSV files
        print(f"\n[3/3] Building intent mapping from original data files...")
        self._build_intent_mapping()
        print(f"      Intent mapping built for {len(self.id_to_intent)} documents")
        
        print("\n" + "=" * 70)
        print("INDEX LOADING COMPLETE")
        print("=" * 70)
    
    def _build_intent_mapping(self):
        """Build mapping from FAISS index position to intent label."""
        import pandas as pd
        
        # Load all splits and concatenate
        base_dir = project_root / "data" / "processed"
        splits = ['train', 'val', 'test']
        
        all_intents = []
        current_idx = 0
        
        for split in splits:
            csv_path = base_dir / f"cleaned_split_{split}.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                if 'intent' in df.columns:
                    intents = df['intent'].tolist()
                elif 'label' in df.columns:
                    intents = df['label'].tolist()
                else:
                    raise ValueError(f"CSV must have 'intent' or 'label' column: {csv_path}")
                
                # Map indices to intents
                for intent in intents:
                    self.id_to_intent[str(current_idx)] = intent
                    current_idx += 1
        
        print(f"      Mapped {current_idx} documents to intents")
            
    def search(self, query_embedding: np.ndarray, k: int = 5) -> Dict[str, Any]:
        """Search for k nearest neighbors given a query embedding."""
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        distances, indices = self.index.search(query_embedding.astype(np.float32), k)
        
        results = {
            'indices': indices[0].tolist(),
            'distances': distances[0].tolist(),
            'intents': [self.id_to_intent.get(str(idx), 'unknown') for idx in indices[0]]
        }
        
        return results
    
    def search_by_text(self, query_text: str, k: int = 5) -> Dict[str, Any]:
        """Search using raw text query."""
        if self.vectorizer is None:
            raise ValueError("Vectorizer not loaded.")
        
        query_embedding = self.vectorizer.transform([query_text]).toarray()[0]
        return self.search(query_embedding, k)
    
    def retrieve_intent(self, query_embedding: np.ndarray, k: int = 1) -> str:
        """Retrieve the most similar intent for a query."""
        results = self.search(query_embedding, k=k)
        intents = results['intents']
        if not intents:
            return 'unknown'
        
        intent_counts = defaultdict(int)
        for intent in intents:
            intent_counts[intent] += 1
        
        return max(intent_counts.keys(), key=lambda x: intent_counts[x])


class RetrievalEvaluator:
    """Evaluator for retrieval system performance."""
    
    def __init__(self, search_engine: FAISSSimilaritySearch):
        self.search_engine = search_engine
        
    def evaluate_on_dataset(
        self,
        embeddings_path: str,
        labels_path: str,
        dataset_name: str,
        k_values: List[int] = [1, 3, 5]
    ) -> Dict[str, Any]:
        """Evaluate retrieval performance on a dataset."""
        import pandas as pd
        
        print(f"\n{'=' * 70}")
        print(f"EVALUATING ON {dataset_name.upper()} SET")
        print(f"{'=' * 70}")
        
        # Load embeddings
        print(f"\nLoading embeddings from: {embeddings_path}")
        embeddings = np.load(embeddings_path)
        print(f"Loaded {len(embeddings)} embeddings")
        
        # Load labels
        print(f"\nLoading labels from: {labels_path}")
        df = pd.read_csv(labels_path)
        if 'intent' in df.columns:
            true_intents = df['intent'].tolist()
        elif 'label' in df.columns:
            true_intents = df['label'].tolist()
        else:
            raise ValueError("CSV must have 'intent' or 'label' column")
        print(f"Loaded {len(true_intents)} intent labels")
        
        min_len = min(len(embeddings), len(true_intents))
        embeddings = embeddings[:min_len]
        true_intents = true_intents[:min_len]
        
        metrics = {
            'dataset': dataset_name,
            'num_queries': min_len,
            'accuracy_at_k': {},
            'recall_at_k': {},
            'mrr': 0.0,
            'latency_ms': 0.0,
            'predictions': [],
            'true_labels': true_intents
        }
        
        all_predictions_at_k = {k: [] for k in k_values}
        mrr_scores = []
        latencies = []
        
        print(f"\nRunning retrieval for {min_len} queries...")
        start_time = time.time()
        
        for i, (query_emb, true_intent) in enumerate(zip(embeddings, true_intents)):
            query_start = time.time()
            results = self.search_engine.search(query_emb, k=max(k_values))
            query_time = (time.time() - query_start) * 1000
            latencies.append(query_time)
            
            for k in k_values:
                all_predictions_at_k[k].append(results['intents'][:k])
            
            # Calculate MRR
            reciprocal_rank = 0.0
            for rank, pred_intent in enumerate(results['intents'], 1):
                if pred_intent == true_intent:
                    reciprocal_rank = 1.0 / rank
                    break
            mrr_scores.append(reciprocal_rank)
            
            metrics['predictions'].append(results['intents'][0] if results['intents'] else 'unknown')
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{min_len} queries...")
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        for k in k_values:
            correct_at_k = [
                1.0 if true_intent in preds_at_k
                else 0.0
                for true_intent, preds_at_k in zip(true_intents, all_predictions_at_k[k])
            ]
            metrics['accuracy_at_k'][k] = np.mean(correct_at_k)
            metrics['recall_at_k'][k] = metrics['accuracy_at_k'][k]
        
        metrics['mrr'] = np.mean(mrr_scores)
        metrics['latency_ms'] = np.mean(latencies)
        metrics['total_time_seconds'] = total_time
        
        # Print results
        print(f"\n{'=' * 70}")
        print(f"{dataset_name.upper()} SET RESULTS")
        print(f"{'=' * 70}")
        print(f"\nAccuracy@k:")
        for k in k_values:
            acc = metrics['accuracy_at_k'][k]
            threshold = 0.7 if k == 1 else 0.9
            status = "PASS" if acc > threshold else "FAIL"
            print(f"  Accuracy@{k}: {acc:.4f} ({acc*100:.2f}%) {status}")
        
        print(f"\nMRR: {metrics['mrr']:.4f} {'PASS' if metrics['mrr'] > 0.75 else 'FAIL'}")
        print(f"Average latency: {metrics['latency_ms']:.2f}ms {'PASS' if metrics['latency_ms'] < 100 else 'FAIL'}")
        print(f"Total time: {total_time:.2f}s")
        
        return metrics
    
    def generate_confusion_matrix(self, true_labels: List[str], pred_labels: List[str], output_path: str):
        """Generate and save confusion matrix."""
        print(f"\nGenerating confusion matrix...")
        
        all_labels = sorted(list(set(true_labels + pred_labels)))
        cm = confusion_matrix(true_labels, pred_labels, labels=all_labels)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=all_labels, yticklabels=all_labels)
        plt.title('Confusion Matrix - Intent Retrieval')
        plt.ylabel('True Intent')
        plt.xlabel('Predicted Intent')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to: {output_path}")
        
    def save_metrics(self, metrics: Dict[str, Any], output_path: str):
        """Save metrics to JSON."""
        def convert(obj):
            if isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert(item) for item in obj]
            return obj
        
        serializable = {k: convert(v) for k, v in metrics.items()}
        
        with open(output_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        
        print(f"Metrics saved to: {output_path}")


def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print("PHASE 3.2: SIMILARITY SEARCH IMPLEMENTATION")
    print("=" * 70)
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    base_dir = project_root
    vector_db_dir = base_dir / "data" / "vector_db" / "faiss_index_v1.0"
    embeddings_dir = base_dir / "data" / "final" / "embeddings_v2.0"
    logs_dir = base_dir / "logs"
    
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize search engine
    search_engine = FAISSSimilaritySearch(
        index_path=str(vector_db_dir / "index.faiss"),
        embeddings_dir=str(embeddings_dir)
    )
    
    search_engine.load_index()
    
    # Initialize evaluator
    evaluator = RetrievalEvaluator(search_engine)
    
    # Evaluate on validation set
    val_metrics = evaluator.evaluate_on_dataset(
        embeddings_path=str(embeddings_dir / "val_emb.npy"),
        labels_path=str(base_dir / "data" / "processed" / "cleaned_split_val.csv"),
        dataset_name="validation",
        k_values=[1, 3, 5]
    )
    
    # Evaluate on test set
    test_metrics = evaluator.evaluate_on_dataset(
        embeddings_path=str(embeddings_dir / "test_emb.npy"),
        labels_path=str(base_dir / "data" / "processed" / "cleaned_split_test.csv"),
        dataset_name="test",
        k_values=[1, 3, 5]
    )
    
    # Generate confusion matrix
    evaluator.generate_confusion_matrix(
        true_labels=test_metrics['true_labels'],
        pred_labels=test_metrics['predictions'],
        output_path=str(logs_dir / "p3.2_confusion_matrix.png")
    )
    
    # Combine metrics
    all_metrics = {
        'phase': '3.2',
        'description': 'Similarity Search Evaluation',
        'validation': val_metrics,
        'test': test_metrics,
        'success_criteria': {
            'top1_accuracy_gt_70': test_metrics['accuracy_at_k'][1] > 0.70,
            'top5_accuracy_gt_90': test_metrics['accuracy_at_k'][5] > 0.90,
            'latency_lt_100ms': test_metrics['latency_ms'] < 100,
            'mrr_gt_075': test_metrics['mrr'] > 0.75
        },
        'all_criteria_met': bool(all([
            test_metrics['accuracy_at_k'][1] > 0.70,
            test_metrics['accuracy_at_k'][5] > 0.90,
            test_metrics['latency_ms'] < 100,
            test_metrics['mrr'] > 0.75
        ]))
    }
    
    # Save metrics
    metrics_path = logs_dir / "p3.2_retrieval_metrics.json"
    evaluator.save_metrics(all_metrics, str(metrics_path))
    
    # Print summary
    print("\n" + "=" * 70)
    print("PHASE 3.2 EVALUATION SUMMARY")
    print("=" * 70)
    print(f"\nTest Set Performance:")
    print(f"  Top-1 Accuracy:  {test_metrics['accuracy_at_k'][1]:.4f} ({test_metrics['accuracy_at_k'][1]*100:.2f}%)")
    print(f"  Top-5 Accuracy:  {test_metrics['accuracy_at_k'][5]:.4f} ({test_metrics['accuracy_at_k'][5]*100:.2f}%)")
    print(f"  MRR:             {test_metrics['mrr']:.4f}")
    print(f"  Avg Latency:     {test_metrics['latency_ms']:.2f}ms")
    
    print(f"\nSuccess Criteria:")
    for criterion, met in all_metrics['success_criteria'].items():
        status = "PASS" if met else "FAIL"
        print(f"  {criterion}: {status}")
    
    print(f"\nOverall: {'ALL CRITERIA MET' if all_metrics['all_criteria_met'] else 'SOME CRITERIA NOT MET'}")
    print(f"\nEnd time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")
    
    return all_metrics


if __name__ == "__main__":
    try:
        metrics = main()
        sys.exit(0 if metrics['all_criteria_met'] else 1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
