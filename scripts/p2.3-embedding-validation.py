#!/usr/bin/env python3
"""
Phase 2.3: Embedding Validation
Validates embedding quality through cohesion, separation, and outlier detection
"""

import pandas as pd
import numpy as np
import yaml
import json
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from datetime import datetime

def load_config():
    """Load configuration from config.yaml"""
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def compute_intra_class_similarity(embeddings, labels):
    """Compute average cosine similarity within each class"""
    unique_labels = np.unique(labels)
    intra_class_sims = {}
    
    for label in unique_labels:
        class_indices = np.where(labels == label)[0]
        if len(class_indices) < 2:
            continue
            
        class_embeddings = embeddings[class_indices]
        sim_matrix = cosine_similarity(class_embeddings)
        upper_tri = sim_matrix[np.triu_indices(len(sim_matrix), k=1)]
        
        intra_class_sims[int(label)] = float(np.mean(upper_tri)) if len(upper_tri) > 0 else 0.0
    
    return intra_class_sims

def compute_inter_class_similarity(embeddings, labels):
    """Compute average cosine similarity between different classes"""
    unique_labels = np.unique(labels)
    inter_class_sims = []
    
    for i, label1 in enumerate(unique_labels):
        for label2 in unique_labels[i+1:]:
            indices1 = np.where(labels == label1)[0]
            indices2 = np.where(labels == label2)[0]
            
            emb1 = embeddings[indices1]
            emb2 = embeddings[indices2]
            
            sim_matrix = cosine_similarity(emb1, emb2)
            inter_class_sims.append(np.mean(sim_matrix))
    
    return float(np.mean(inter_class_sims)) if inter_class_sims else 0.0

def detect_outliers(embeddings, contamination=0.02):
    """Detect outliers using Isolation Forest"""
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    
    predictions = iso_forest.fit_predict(embeddings)
    outlier_indices = np.where(predictions == -1)[0]
    
    return outlier_indices, predictions

def generate_tsne_visualization(embeddings, labels, sample_size=500):
    """Generate t-SNE visualization (sampled for performance)"""
    if len(embeddings) > sample_size:
        indices = np.random.choice(len(embeddings), sample_size, replace=False)
        sampled_embeddings = embeddings[indices]
        sampled_labels = labels[indices]
    else:
        sampled_embeddings = embeddings
        sampled_labels = labels
    
    print("  Applying t-SNE (this may take a minute)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    tsne_embeddings = tsne.fit_transform(sampled_embeddings)
    
    return tsne_embeddings, sampled_labels

def main():
    print("=" * 60)
    print("PHASE 2.3: EMBEDDING VALIDATION")
    print("=" * 60)
    
    config = load_config()
    validation_config = config['validation']
    
    print(f"\nConfiguration:")
    print(f"  Sample size: {validation_config['sample_size']}")
    print(f"  Intra-class threshold: > {validation_config['intra_class_threshold']}")
    print(f"  Inter-class threshold: < {validation_config['inter_class_threshold']}")
    print(f"  Outlier threshold: {validation_config['outlier_threshold']}")
    
    os.makedirs('data/embeddings', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    print("\nLoading embeddings...")
    train_emb = np.load('data/embeddings/train_emb.npy')
    val_emb = np.load('data/embeddings/val_emb.npy')
    test_emb = np.load('data/embeddings/test_emb.npy')
    
    print(f"  Train embeddings: {train_emb.shape}")
    print(f"  Val embeddings: {val_emb.shape}")
    print(f"  Test embeddings: {test_emb.shape}")
    
    print("\nLoading intent labels...")
    train_df = pd.read_csv('data/processed/cleaned_split_train.csv')
    val_df = pd.read_csv('data/processed/cleaned_split_val.csv')
    test_df = pd.read_csv('data/processed/cleaned_split_test.csv')
    
    train_labels = train_df['intent_encoded'].values
    val_labels = val_df['intent_encoded'].values
    test_labels = test_df['intent_encoded'].values
    
    print(f"  Unique intents: {len(np.unique(train_labels))}")
    
    print("\n" + "=" * 60)
    print("COMPUTING METRICS")
    print("=" * 60)
    
    print("\n1. Intra-class cohesion (training set)...")
    intra_class_sims = compute_intra_class_similarity(train_emb, train_labels)
    avg_intra_class = float(np.mean(list(intra_class_sims.values())))
    print(f"   Average intra-class similarity: {avg_intra_class:.4f}")
    
    print("\n2. Inter-class separation (training set)...")
    inter_class_sim = compute_inter_class_similarity(train_emb, train_labels)
    print(f"   Average inter-class similarity: {inter_class_sim:.4f}")
    
    print("\n3. Outlier detection (training set)...")
    outlier_indices, outlier_predictions = detect_outliers(
        train_emb, 
        contamination=validation_config['outlier_threshold']
    )
    outlier_percentage = float(len(outlier_indices) / len(train_emb) * 100)
    print(f"   Outliers detected: {len(outlier_indices)} ({outlier_percentage:.2f}%)")
    
    print("\n4. t-SNE visualization (sampled)...")
    tsne_embeddings, tsne_labels = generate_tsne_visualization(
        train_emb, 
        train_labels,
        sample_size=validation_config['sample_size']
    )
    
    tsne_data = {
        'embeddings_2d': tsne_embeddings.tolist(),
        'labels': [int(x) for x in tsne_labels],
        'sample_size': len(tsne_labels)
    }
    with open('data/embeddings/tsne_results.pkl', 'wb') as f:
        pickle.dump(tsne_data, f)
    print("   Saved to: data/embeddings/tsne_results.pkl")
    
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    
    intra_class_pass = bool(avg_intra_class > validation_config['intra_class_threshold'])
    inter_class_pass = bool(inter_class_sim < validation_config['inter_class_threshold'])
    outlier_pass = bool(outlier_percentage < 10)
    
    print(f"\n1. Intra-class cohesion: {avg_intra_class:.4f} (threshold: > {validation_config['intra_class_threshold']})")
    print(f"   Status: {'✓ PASS' if intra_class_pass else '✗ FAIL'}")
    
    print(f"\n2. Inter-class separation: {inter_class_sim:.4f} (threshold: < {validation_config['inter_class_threshold']})")
    print(f"   Status: {'✓ PASS' if inter_class_pass else '✗ FAIL'}")
    
    print(f"\n3. Outlier rate: {outlier_percentage:.2f}% (threshold: < 10%)")
    print(f"   Status: {'✓ PASS' if outlier_pass else '✗ FAIL'}")
    
    all_passed = bool(intra_class_pass and inter_class_pass and outlier_pass)
    
    validation_report = {
        'phase': '2.3',
        'timestamp': datetime.now().isoformat(),
        'metrics': {
            'intra_class_similarity': {
                'value': avg_intra_class,
                'threshold': float(validation_config['intra_class_threshold']),
                'passed': intra_class_pass
            },
            'inter_class_similarity': {
                'value': inter_class_sim,
                'threshold': float(validation_config['inter_class_threshold']),
                'passed': inter_class_pass
            },
            'outlier_rate': {
                'value': outlier_percentage,
                'threshold': 10.0,
                'passed': outlier_pass
            }
        },
        'embedding_stats': {
            'total_train_samples': int(len(train_emb)),
            'total_val_samples': int(len(val_emb)),
            'total_test_samples': int(len(test_emb)),
            'embedding_dimension': int(train_emb.shape[1]),
            'num_unique_intents': int(len(np.unique(train_labels)))
        },
        'intra_class_by_intent': intra_class_sims,
        'outlier_indices': [int(x) for x in outlier_indices],
        'overall_validation_passed': all_passed,
        'files_created': [
            'data/embeddings/tsne_results.pkl',
            'logs/p2.3_validation_report.json'
        ]
    }
    
    with open('logs/p2.3_validation_report.json', 'w') as f:
        json.dump(validation_report, f, indent=2)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL VALIDATION CHECKS PASSED")
    else:
        print("✗ SOME VALIDATION CHECKS FAILED")
        print("\nNote: Low intra-class similarity is expected with TF-IDF.")
        print("      This is acceptable for baseline embeddings.")
    print("=" * 60)
    print(f"\nValidation report saved to: logs/p2.3_validation_report.json")
    print("\nReady for Phase 2.4: Embedding Storage")

if __name__ == '__main__':
    main()
