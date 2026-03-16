#!/usr/bin/env python3
"""Retrain SGD model with current sklearn version."""

import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
import os

print("="*70)
print("RETRAINING SGD MODEL - Sklearn Version Compatibility Fix")
print("="*70)

# Load data
print("\nLoading data...")
train_df = pd.read_csv('data/processed/cleaned_split_train.csv')
test_df = pd.read_csv('data/processed/cleaned_split_test.csv')

# Load TF-IDF embeddings
X_train = np.load('data/final/embeddings_v2.0/train_emb.npy')
X_test = np.load('data/final/embeddings_v2.0/test_emb.npy')
y_train = train_df['intent_encoded'].values
y_test = test_df['intent_encoded'].values

print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
print(f"  Classes: {len(np.unique(y_train))}")

# Train SGD
print("\nTraining SGDClassifier...")
model = SGDClassifier(
    loss='log_loss',
    penalty='l2',
    alpha=0.0001,
    max_iter=1000,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average='macro')

print(f"\nSGD Results:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  F1-Macro: {f1_macro:.4f}")

# Save model
os.makedirs('models/phase4', exist_ok=True)
model_path = 'models/phase4/sgd_v1.0.1.pkl'

with open(model_path, 'wb') as f:
    pickle.dump({'model': model, 'class_mapper': None}, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"\n✓ Model saved: {model_path}")
print(f"  Size: {os.path.getsize(model_path) / 1024:.2f} KB")

# Verify
with open(model_path, 'rb') as f:
    loaded = pickle.load(f)
    print(f"✓ Model verified - loads successfully")

print("\nReady for Phase 4.7 validation!")
