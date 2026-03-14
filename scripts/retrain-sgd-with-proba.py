#!/usr/bin/env python3
"""
Retrain SGD model with log_loss for probability estimates
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

# Load data
train_df = pd.read_csv("data/processed/cleaned_split_train.csv")
X_text = train_df['user_input_clean'].fillna('').tolist()
y = train_df['intent_encoded'].tolist()

# Load vectorizer
with open("data/final/embeddings_v2.0/vectorizer.pkl", 'rb') as f:
    vectorizer = pickle.load(f)

# Transform
X = vectorizer.transform(X_text)

# Train SGD with log_loss (supports predict_proba)
model = SGDClassifier(
    loss='log_loss',  # Changed from 'hinge'
    penalty='l2',
    alpha=0.0001,
    max_iter=1000,
    tol=1e-3,
    random_state=42,
    class_weight='balanced'
)

print("Training SGD with log_loss...")
model.fit(X, y)

print(f"✓ Model trained: {model.__class__.__name__}")
print(f"✓ Loss: {model.loss}")
print(f"✓ Classes: {len(model.classes_)}")
print(f"✓ Supports predict_proba: {hasattr(model, 'predict_proba')}")

# Save
with open("models/phase4/sgd_v1.0.1.pkl", 'wb') as f:
    pickle.dump(model, f)

print("✓ Model saved: models/phase4/sgd_v1.0.1.pkl")
