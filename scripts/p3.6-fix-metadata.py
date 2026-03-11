#!/usr/bin/env python3
"""
Regenerate document metadata for FAISS index.
Combines train/val/test data to match the 4,774 vectors in the index.
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime

def regenerate_document_metadata():
    """Load all processed data and create proper document metadata."""
    
    print("=" * 70)
    print("Regenerating Document Metadata for FAISS Index")
    print("=" * 70)
    
    # Load all split files
    data_dir = Path("data/processed")
    splits = {
        "train": data_dir / "cleaned_split_train.csv",
        "val": data_dir / "cleaned_split_val.csv",
        "test": data_dir / "cleaned_split_test.csv"
    }
    
    all_documents = []
    total_count = 0
    
    for split_name, split_path in splits.items():
        if split_path.exists():
            df = pd.read_csv(split_path)
            print(f"\n{split_name.upper()}:")
            print(f"  Rows: {len(df)}")
            print(f"  Columns: {list(df.columns)}")
            
            # Create document metadata for each row
            for idx, row in df.iterrows():
                doc = {
                    "index": total_count,
                    "split": split_name,
                    "original_index": idx,
                    "content": row.get("cleaned_text", row.get("user_input", "")),
                    "intent": row.get("intent", "unknown"),
                    "source": row.get("source", "unknown"),
                    "user_input": row.get("user_input", ""),
                    "text_length": row.get("text_length", 0),
                    "word_count": row.get("word_count", 0)
                }
                all_documents.append(doc)
                total_count += 1
            
            print(f"  Documents added: {len(df)}")
    
    print(f"\n{'=' * 70}")
    print(f"Total documents: {len(all_documents)}")
    print(f"{'=' * 70}")
    
    # Save document metadata
    output_path = Path("data/vector_db/faiss_index_v1.0/document_metadata.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_documents, f, indent=2)
    
    print(f"\nDocument metadata saved to: {output_path}")
    
    # Verify
    print(f"\nSample document (first):")
    print(json.dumps(all_documents[0], indent=2))
    
    print(f"\nSample document (middle):")
    mid_idx = len(all_documents) // 2
    print(json.dumps(all_documents[mid_idx], indent=2))
    
    # Check for duplicates
    contents = [doc["content"] for doc in all_documents]
    unique_contents = set(contents)
    print(f"\nDuplicate check:")
    print(f"  Total contents: {len(contents)}")
    print(f"  Unique contents: {len(unique_contents)}")
    print(f"  Duplicates: {len(contents) - len(unique_contents)}")
    
    return all_documents


if __name__ == "__main__":
    documents = regenerate_document_metadata()
    print("\n✓ Metadata regeneration complete!")
