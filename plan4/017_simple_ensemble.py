#!/usr/bin/env python3
"""
Simple ensemble of available submissions.
Weighted average of predictions.
"""
import pandas as pd
import numpy as np
import os

print("Loading submissions...")

# Available submission files
submissions = {
    '007_xgboost': '007_xgboost_submission.csv',
    '008_calibrated': '008_calibrated_submission.csv',
    '010_raw': '010_raw_predictions.csv',
    '010_mild': '010_mild_scaled.csv'
}

# Load all available submissions
loaded = {}
for name, file in submissions.items():
    if os.path.exists(file):
        df = pd.read_csv(file)
        loaded[name] = df
        print(f"  {name}: mean={df['clicked'].mean():.5f}, std={df['clicked'].std():.5f}")
    else:
        print(f"  {name}: Not found")

if len(loaded) < 2:
    print("Error: Need at least 2 submissions for ensemble")
    exit(1)

# Get reference for IDs
reference = list(loaded.values())[0]
ids = reference['ID']

print(f"\nFound {len(loaded)} submissions for ensemble")

# Method 1: Simple average
print("\nMethod 1: Simple average")
preds_list = [df['clicked'].values for df in loaded.values()]
avg_preds = np.mean(preds_list, axis=0)

print(f"  Mean: {avg_preds.mean():.5f}")
print(f"  Std:  {avg_preds.std():.5f}")

submission_avg = pd.DataFrame({
    'ID': ids,
    'clicked': avg_preds
})
submission_avg.to_csv('017_ensemble_avg.csv', index=False)
print(f"  Saved to 017_ensemble_avg.csv")

# Method 2: Weighted average (give more weight to raw predictions)
print("\nMethod 2: Weighted average")

# Weights based on expected performance
weights = {
    '007_xgboost': 0.5,      # Original XGBoost (before guardrail)
    '008_calibrated': 0.1,   # Calibrated (after guardrail)
    '010_raw': 0.3,          # Raw predictions
    '010_mild': 0.1          # Mild scaled
}

weighted_sum = np.zeros(len(ids))
total_weight = 0

for name, df in loaded.items():
    weight = weights.get(name, 0.25)  # Default weight if not specified
    weighted_sum += df['clicked'].values * weight
    total_weight += weight

weighted_preds = weighted_sum / total_weight

print(f"  Mean: {weighted_preds.mean():.5f}")
print(f"  Std:  {weighted_preds.std():.5f}")

submission_weighted = pd.DataFrame({
    'ID': ids,
    'clicked': weighted_preds
})
submission_weighted.to_csv('017_ensemble_weighted.csv', index=False)
print(f"  Saved to 017_ensemble_weighted.csv")

# Method 3: Rank average (more robust)
print("\nMethod 3: Rank average")

ranks_list = []
for name, df in loaded.items():
    # Convert predictions to ranks
    ranks = df['clicked'].rank(pct=True).values
    ranks_list.append(ranks)

avg_ranks = np.mean(ranks_list, axis=0)

# Convert ranks back to prediction scale
# Use original distribution as reference
ref_sorted = np.sort(loaded['007_xgboost']['clicked'].values)
rank_indices = (avg_ranks * (len(ref_sorted) - 1)).astype(int)
rank_preds = ref_sorted[rank_indices]

print(f"  Mean: {rank_preds.mean():.5f}")
print(f"  Std:  {rank_preds.std():.5f}")

submission_rank = pd.DataFrame({
    'ID': ids,
    'clicked': rank_preds
})
submission_rank.to_csv('017_ensemble_rank.csv', index=False)
print(f"  Saved to 017_ensemble_rank.csv")

print("\n" + "="*50)
print("Ensemble complete!")
print("Generated 3 ensemble submissions:")
print("  1. 017_ensemble_avg.csv (simple average)")
print("  2. 017_ensemble_weighted.csv (weighted average)")
print("  3. 017_ensemble_rank.csv (rank average)")
print("\nRecommendation: Try submitting 017_ensemble_weighted.csv first")
print("="*50)