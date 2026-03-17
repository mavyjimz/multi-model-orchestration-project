#!/usr/bin/env python3
"""
p6.6-generate-predictions.py (DICT-MODEL HANDLER)
Handles model pickle that may be: 
- Direct estimator (SGDClassifier)
- Dict with keys: {'model': ..., 'vectorizer': ..., 'class_mapper': ...}
"""
import argparse
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/p6.6-generate-predictions.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def extract_model_from_pickle(loaded_obj):
    """Extract the actual estimator from various pickle structures."""
    # Case 1: Direct estimator (has predict method)
    if hasattr(loaded_obj, 'predict'):
        return loaded_obj
    
    # Case 2: Dict with 'model' or 'classifier' key
    if isinstance(loaded_obj, dict):
        for key in ['model', 'classifier', 'estimator', 'sgd_model']:
            if key in loaded_obj and hasattr(loaded_obj[key], 'predict'):
                logger.info(f"Extracted model from dict key: '{key}'")
                return loaded_obj[key]
        # If dict has no predict-capable object, return the first one that does
        for val in loaded_obj.values():
            if hasattr(val, 'predict'):
                return val
    
    # Case 3: Object with .model attribute (wrapper pattern)
    if hasattr(loaded_obj, 'model') and hasattr(loaded_obj.model, 'predict'):
        return loaded_obj.model
    
    raise TypeError(
        f"Loaded object has no predict method. Type: {type(loaded_obj)}. "
        f"Keys if dict: {list(loaded_obj.keys()) if isinstance(loaded_obj, dict) else 'N/A'}"
    )


def load_vectorizer_from_artifacts(model_name: str, version: str, base_path: str = 'artifacts'):
    """Load vectorizer from Phase 6.3 artifact structure."""
    if version == '1':
        version = '1.0.2'
    elif version.startswith('v'):
        version = version[1:]
    
    version_dirs = [version]
    if version == '1.0.2':
        version_dirs.extend(['1.0.1', 'v1.0.2', 'v1.0.1'])
    
    for ver in version_dirs:
        for subdir in ['model', 'serialized', '']:
            candidate = Path(base_path) / 'models' / model_name / ver / subdir / 'vectorizer.pkl'
            if candidate.exists():
                logger.info(f"Loading vectorizer from: {candidate}")
                return joblib.load(candidate)
    
    model_dir = Path(base_path) / 'models' / model_name
    if model_dir.exists():
        for candidate in model_dir.rglob('vectorizer.pkl'):
            logger.info(f"Found vectorizer via search: {candidate}")
            return joblib.load(candidate)
    
    raise FileNotFoundError(f"Vectorizer not found for {model_name}")


def load_model_from_artifacts(model_name: str, version: str, base_path: str = 'artifacts'):
    """Load and extract model from Phase 6.3 artifact structure."""
    if version == '1':
        version = '1.0.2'
    elif version.startswith('v'):
        version = version[1:]
    
    version_dirs = [version]
    if version == '1.0.2':
        version_dirs.extend(['1.0.1', 'v1.0.2', 'v1.0.1'])
    
    for ver in version_dirs:
        for subdir in ['model', 'serialized', '']:
            patterns = [
                f'{model_name}_{ver}.pkl',
                f'{model_name}_v{ver}.pkl',
                'model.pkl', 'classifier.pkl', 'sgd_model.pkl'
            ]
            for pattern in patterns:
                candidate = Path(base_path) / 'models' / model_name / ver / subdir / pattern
                if candidate.exists():
                    logger.info(f"Loading model artifact: {candidate}")
                    loaded = joblib.load(candidate)
                    return extract_model_from_pickle(loaded)
    
    model_dir = Path(base_path) / 'models' / model_name
    if model_dir.exists():
        for candidate in model_dir.rglob('*.pkl'):
            if 'vectorizer' not in candidate.name.lower():
                logger.info(f"Trying model candidate: {candidate}")
                loaded = joblib.load(candidate)
                try:
                    return extract_model_from_pickle(loaded)
                except TypeError:
                    continue
    
    raise FileNotFoundError(f"Model not found for {model_name} v{version}")


def generate_predictions(model_name: str, version: str, test_data_path: str, output_path: str,
                        text_column: str = 'cleaned_text', label_column: str = 'label',
                        artifacts_path: str = 'artifacts') -> dict:
    """Generate predictions using artifact storage."""
    logger.info(f"Generating predictions for {model_name} v{version}")
    
    test_df = pd.read_csv(test_data_path)
    logger.info(f"Loaded test  {len(test_df)} rows")
    
    if text_column not in test_df.columns:
        for col in ['text', 'message', 'utterance', 'input', 'query']:
            if col in test_df.columns:
                text_column = col
                logger.info(f"Using text column: {text_column}")
                break
    
    texts = test_df[text_column].fillna('').astype(str).tolist()
    true_labels = test_df[label_column].values if label_column in test_df.columns else None
    
    logger.info("Loading model from artifact storage...")
    model = load_model_from_artifacts(model_name, version, artifacts_path)
    logger.info(f"Model type: {type(model).__name__}, has predict: {hasattr(model, 'predict')}")
    
    logger.info("Loading vectorizer...")
    vectorizer = load_vectorizer_from_artifacts(model_name, version, artifacts_path)
    
    logger.info(f"Vectorizing {len(texts)} samples...")
    X = vectorizer.transform(texts)
    logger.info("Running inference...")
    predictions = model.predict(X)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, predictions)
    logger.info(f"Saved predictions to {output_path}")
    
    result = {
        'model': model_name, 'version': version,
        'predictions_path': output_path,
        'n_samples': len(predictions),
        'unique_predictions': int(np.unique(predictions).size)
    }
    
    if true_labels is not None:
        if predictions.dtype != true_labels.dtype:
            try:
                true_labels = true_labels.astype(predictions.dtype)
            except:
                pass
        result['accuracy'] = float(np.mean(predictions == true_labels))
        logger.info(f"Test accuracy: {result['accuracy']:.4f}")
    
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--version', required=True)
    parser.add_argument('--test-data', default='data/processed/cleaned_split_test.csv')
    parser.add_argument('--output', required=True)
    parser.add_argument('--text-column', default='cleaned_text')
    parser.add_argument('--label-column', default='label')
    parser.add_argument('--artifacts-path', default='artifacts')
    args = parser.parse_args()
    
    result = generate_predictions(
        model_name=args.model, version=args.version,
        test_data_path=args.test_data, output_path=args.output,
        text_column=args.text_column, label_column=args.label_column,
        artifacts_path=args.artifacts_path
    )
    
    print(f"\n{'='*50}\nPREDICTION GENERATION COMPLETE\n{'='*50}")
    print(f"Model: {result['model']} v{result['version']}")
    print(f"Samples: {result['n_samples']}")
    print(f"Output: {result['predictions_path']}")
    if 'accuracy' in result:
        print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"{'='*50}\n")
    return 0


if __name__ == '__main__':
    exit(main())
