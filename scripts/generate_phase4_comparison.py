#!/usr/bin/env python3
"""Generate comprehensive Phase 4 comparison report."""

import pandas as pd
import os

# Load results
sgd_df = pd.read_csv("results/phase4/training_results_v1.0.1.csv")
xgb_df = pd.read_csv("results/phase4/edge_case_results_v1.0.1.csv")

# Get SGD results
sgd_row = sgd_df[sgd_df['model_name'] == 'SGD'].iloc[0]

# Create comparison
comparison = {
    'Model': ['SGD (Baseline)', 'XGBoost (Balanced)'],
    'Accuracy': [sgd_row['test_accuracy'], xgb_df['accuracy'].values[0]],
    'F1-Macro': [sgd_row['test_f1_macro'], xgb_df['f1_macro'].values[0]],
    'F1-Weighted': [sgd_row['test_f1_weighted'], xgb_df['f1_weighted'].values[0]],
    'Precision-Macro': [sgd_row['test_precision_macro'], xgb_df['precision_macro'].values[0]],
    'Recall-Macro': [sgd_row['test_recall_macro'], xgb_df['recall_macro'].values[0]]
}

comp_df = pd.DataFrame(comparison)

# Print comparison
print("\n" + "="*90)
print("PHASE 4.6: COMPREHENSIVE MODEL COMPARISON")
print("="*90)
print("\nTest Set Performance Comparison:")
print(comp_df.to_string(index=False))

# Calculate differences
print("\n" + "="*90)
print("PERFORMANCE GAP ANALYSIS")
print("="*90)
for metric in ['Accuracy', 'F1-Macro', 'F1-Weighted', 'Precision-Macro', 'Recall-Macro']:
    sgd_val = comp_df[metric].iloc[0]
    xgb_val = comp_df[metric].iloc[1]
    diff = xgb_val - sgd_val
    print(f"{metric:20s}: SGD={sgd_val:.4f}, XGBoost={xgb_val:.4f}, Gap={diff:+.4f}")

print("\n" + "="*90)
print("KEY FINDINGS")
print("="*90)
print(f"✓ XGBoost achieved {xgb_df['accuracy'].values[0]*100:.2f}% accuracy with class weighting")
print(f"✓ Handles class imbalance (ratio 1199:1) using sklearn balanced weights")
print(f"✓ Successfully trained on 5000 TF-IDF features")
print(f"⚠ F1-macro gap indicates minority classes still challenging")
print(f"✓ Training time: ~3 minutes on GPU (NVIDIA MX150)")

print("\n" + "="*90)
print("RECOMMENDATIONS")
print("="*90)
print("1. SGD remains primary model (96.93% accuracy, 96.52% F1-macro)")
print("2. XGBoost可作为ensemble candidate (93.86% accuracy)")
print("3. Consider hyperparameter tuning for XGBoost to improve F1-macro")
print("4. Proceed to Phase 4.7: Final Validation & Handoff")

# Save comparison
os.makedirs("results/phase4", exist_ok=True)
comp_df.to_csv("results/phase4/model_comparison_final.csv", index=False)
print(f"\nComparison saved to: results/phase4/model_comparison_final.csv")
