#!/usr/bin/env python3
"""
Regenerate fitted vectorizer to match model's 5000 features
"""

import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load training data
train_df = pd.read_csv("data/processed/cleaned_split_train.csv")
print(f"Loaded {len(train_df)} training samples")

# Use user_input_clean column
texts = train_df['user_input_clean'].fillna('').tolist()
print(f"Text column: user_input_clean")

# Create and fit vectorizer with 5000 features
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=1,
    max_df=1.0,
    sublinear_tf=True
)

print("Fitting vectorizer...")
vectorizer.fit(texts)

num_features = len(vectorizer.get_feature_names_out())
print(f"✓ Vectorizer fitted with {num_features} features")

# Save
with open("data/final/embeddings_v2.0/vectorizer.pkl", 'wb') as f:
    pickle.dump(vectorizer, f)

print(f"✓ Saved to: data/final/embeddings_v2.0/vectorizer.pkl")
print(f"✓ Has idf_: {hasattr(vectorizer, 'idf_')}")
