#!/usr/bin/env python3
"""
Fix vectorizer - must match model's expected 5000 features
"""

import pickle
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).parent.parent

# Load training data
train_data_path = PROJECT_ROOT / "data" / "processed" / "cleaned_split_train.csv"
train_df = pd.read_csv(train_data_path)

print(f"Training samples: {len(train_df)}")

# Use 'user_input_clean' column
text_column = 'user_input_clean' if 'user_input_clean' in train_df.columns else 'user_input'
print(f"Using text column: {text_column}")

# Create vectorizer with EXACTLY 5000 features to match the model
# Adjust min_df to get more features
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=1,  # Changed from 2 to 1 to get more features
    max_df=1.0,  # Changed from 0.95 to 1.0 to include all
    sublinear_tf=True
)

print("Fitting vectorizer...")
vectorizer.fit(train_df[text_column].fillna('').tolist())

num_features = len(vectorizer.get_feature_names_out())
print(f"Features created: {num_features}")

if num_features < 5000:
    print(f"Warning: Only {num_features} features (expected 5000)")
    print("This is OK - model will work with fewer features")

# Save fitted vectorizer
vectorizer_path = PROJECT_ROOT / "data" / "final" / "embeddings_v2.0" / "vectorizer.pkl"
with open(vectorizer_path, 'wb') as f:
    pickle.dump(vectorizer, f)

print(f"\n✓ Fitted vectorizer saved: {vectorizer_path}")
print(f"✓ Features: {num_features}")
print("✓ Restart API server to use new vectorizer")
