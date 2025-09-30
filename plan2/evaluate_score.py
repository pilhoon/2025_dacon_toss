#!/usr/bin/env python3
"""
evaluate_score.py
Calculate the actual competition score
Score = 0.5 * AP + 0.5 * (1/(1+WLL))
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss
import glob
import sys

def calculate_weighted_log_loss(y_true, y_pred, eps=1e-15):
    """
    Calculate Weighted Log Loss with 50:50 class balance
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)

    # Calculate class weights for 50:50 balance
    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos

    # Weight to balance to 50:50
    pos_weight = 0.5 / (n_pos / len(y_true)) if n_pos > 0 else 1.0
    neg_weight = 0.5 / (n_neg / len(y_true)) if n_neg > 0 else 1.0

    # Normalize weights
    total_weight = pos_weight * n_pos + neg_weight * n_neg
    pos_weight = pos_weight * len(y_true) / total_weight
    neg_weight = neg_weight * len(y_true) / total_weight

    # Calculate weighted log loss
    loss = 0.0
    for i in range(len(y_true)):
        if y_true[i] == 1:
            loss += -pos_weight * np.log(y_pred[i])
        else:
            loss += -neg_weight * np.log(1 - y_pred[i])

    return loss / len(y_true)

def calculate_competition_score(y_true, y_pred):
    """
    Calculate the competition score
    Score = 0.5 * AP + 0.5 * (1/(1+WLL))
    """
    # Average Precision
    ap = average_precision_score(y_true, y_pred)

    # Weighted Log Loss
    wll = calculate_weighted_log_loss(y_true, y_pred)

    # Final score
    score = 0.5 * ap + 0.5 * (1 / (1 + wll))

    return score, ap, wll

def evaluate_submission(submission_file, y_true=None):
    """
    Evaluate a submission file
    """
    # Load submission
    df = pd.read_csv(submission_file)
    y_pred = df['clicked'].values

    print(f"\nEvaluating: {submission_file}")
    print(f"Prediction stats:")
    print(f"  Mean: {y_pred.mean():.6f}")
    print(f"  Std: {y_pred.std():.6f}")
    print(f"  Min: {y_pred.min():.6f}")
    print(f"  Max: {y_pred.max():.6f}")
    print(f"  >0.5: {(y_pred > 0.5).sum()} ({(y_pred > 0.5).mean()*100:.2f}%)")

    # If we have true labels (for validation)
    if y_true is not None:
        score, ap, wll = calculate_competition_score(y_true, y_pred)
        print(f"\nEvaluation Metrics:")
        print(f"  AP (Average Precision): {ap:.6f}")
        print(f"  WLL (Weighted LogLoss): {wll:.6f}")
        print(f"  Competition Score: {score:.6f}")
        print(f"    = 0.5 × {ap:.4f} + 0.5 × (1/(1+{wll:.4f}))")
        print(f"    = {0.5*ap:.4f} + {0.5*(1/(1+wll)):.4f}")
        print(f"    = {score:.6f}")
        return score

    return None

def main():
    """
    Evaluate all submission files
    """
    print("="*60)
    print("Competition Score Evaluation")
    print("Score = 0.5 × AP + 0.5 × (1/(1+WLL))")
    print("="*60)

    # Find all submission files
    submission_files = glob.glob('plan2/*_submission.csv')
    submission_files.sort()

    if not submission_files:
        print("No submission files found!")
        return

    # For actual evaluation, we would need validation labels
    # Here we just show the prediction distribution
    scores = []
    for file in submission_files:
        score = evaluate_submission(file)
        if score:
            scores.append((file, score))

    # Simulate scores based on prediction distribution
    print("\n" + "="*60)
    print("Estimated Scores (based on prediction distribution):")
    print("="*60)

    for file in submission_files:
        df = pd.read_csv(file)
        y_pred = df['clicked'].values

        # Estimate based on distribution
        # Higher variance and reasonable mean = better AP
        # Lower prediction values = lower WLL

        mean_pred = y_pred.mean()
        std_pred = y_pred.std()

        # Rough estimation
        estimated_ap = min(0.4, std_pred * 2)  # Variance indicates ranking ability
        estimated_wll = -np.log(1 - mean_pred) * 10  # Rough WLL estimate
        estimated_score = 0.5 * estimated_ap + 0.5 * (1 / (1 + estimated_wll))

        model_name = file.split('/')[-1].replace('_submission.csv', '')
        print(f"\n{model_name}:")
        print(f"  Estimated AP: ~{estimated_ap:.3f}")
        print(f"  Estimated WLL: ~{estimated_wll:.3f}")
        print(f"  Estimated Score: ~{estimated_score:.3f}")

    print("\n" + "="*60)
    print("Note: These are estimates. Actual scores require true labels.")
    print("Better scores need: Higher AP (better ranking) + Lower WLL")
    print("="*60)

if __name__ == "__main__":
    main()