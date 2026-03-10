# Data Version Manifest
**Generated**: 2026-03-10 09:36:13
**Project**: Multi-Model Orchestration System

## Phase 1.1: Raw Data Sources

- `intent_classification_20260309_193000.csv`: 42.1 KB | SHA256: `0aec78010766c7ea...`
- `intent_classification_merged_20260309_210129.csv`: 463.6 KB | SHA256: `6346d4832d79599d...`

## Phase 1.2: Merged Dataset

- `intent_classification_merged_20260309_210129.csv`: 463.6 KB | SHA256: `6346d4832d79599d...`

## Phase 1.3: Processed Features

- `intent_features_20260309_194815.csv`: 1.0 KB | SHA256: `5d2d6fb46ee439e9...`
- `intent_features_20260309_212131.csv`: 806.3 KB | SHA256: `8f6b3254bb9dbca8...`

## Phase 1.4: Train/Val/Test Splits

- `train_20260310_093157.csv`: 562.4 KB | SHA256: `47324edb07323aef...`
- `val_20260310_093157.csv`: 120.6 KB | SHA256: `a3ac52f39a46d742...`
- `test_20260310_093157.csv`: 121.6 KB | SHA256: `33a7c9f918e2019c...`

## Version History

| Version | Date | Description | Samples |
|---------|------|-------------|---------|
| v1.0 | 2026-03-09 | Initial dataset (original) | 1,000 |
| v1.1 | 2026-03-09 | Multi-source merge (ATIS + Chatbots) | 6,121 |
| v1.2 | 2026-03-09 | Feature engineering | 4,786 |
| v1.3 | 2026-03-10 | Stratified splits | 4,774 |
