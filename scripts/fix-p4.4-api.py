#!/usr/bin/env python3
"""
Quick fix for Phase 4.4 API prediction issues
"""

import pickle
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# Check what's in the model
model_path = PROJECT_ROOT / "models" / "phase4" / "sgd_v1.0.1.pkl"
print(f"Loading model from: {model_path}")

with open(model_path, 'rb') as f:
    model = pickle.load(f)

print(f"Model type: {type(model)}")
print(f"Model classes: {model.classes_}")
print(f"Model coef shape: {model.coef_.shape}")

# Check vectorizer
vectorizer_path = PROJECT_ROOT / "data" / "final" / "embeddings_v2.0" / "vectorizer.pkl"
print(f"\nLoading vectorizer from: {vectorizer_path}")

with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

print(f"Vectorizer type: {type(vectorizer)}")
print(f"Vectorizer features: {vectorizer.get_feature_names_out().shape[0]}")

# Check label mapping
index_maps_path = PROJECT_ROOT / "data" / "final" / "embeddings_v2.0" / "all_index_maps.pkl"
print(f"\nLoading index maps from: {index_maps_path}")

with open(index_maps_path, 'rb') as f:
    maps = pickle.load(f)

print(f"Maps keys: {maps.keys()}")
if 'label_mapping' in maps:
    print(f"Label mapping: {maps['label_mapping']}")
    print(f"Number of classes: {len(maps['label_mapping'])}")

# Check metadata
metadata_path = PROJECT_ROOT / "models" / "phase4" / "model_manifest_v1.0.1.json"
print(f"\nLoading metadata from: {metadata_path}")

with open(metadata_path, 'r') as f:
    metadata = json.load(f)

print(f"Metadata keys: {metadata.keys()}")
if 'label_mapping' in metadata:
    print(f"Label mapping in metadata: {len(metadata['label_mapping'])} classes")

print("\n✓ Diagnostic complete")
