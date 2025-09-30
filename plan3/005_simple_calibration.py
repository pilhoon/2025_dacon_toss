#!/usr/bin/env python
"""
Simple Calibration for 057 Model
Goal: Create final submission with optimal calibration power
"""

import numpy as np
import pandas as pd


def calibrate(p, power=1.08):
    """Power calibration to improve discrimination"""
    p_safe = np.clip(p, 1e-7, 1-1e-7)
    return np.power(p_safe, power) / (np.power(p_safe, power) + np.power(1-p_safe, power))


def main():
    print("="*80)
    print("SIMPLE CALIBRATION FOR 057 MODEL")
    print("Creating final submission files")
    print("="*80)

    # Load uncalibrated predictions
    print("\nLoading 057 model predictions...")
    uncalibrated_df = pd.read_csv('plan2/060_gpu_uncalibrated.csv')
    predictions = uncalibrated_df['clicked'].values

    print(f"Original statistics:")
    print(f"  Mean: {predictions.mean():.6f}")
    print(f"  Std: {predictions.std():.6f}")
    print(f"  Min: {predictions.min():.6f}")
    print(f"  Max: {predictions.max():.6f}")

    # Test different calibration powers
    # Based on analysis, 1.08 seems optimal for matching target distribution
    test_powers = [1.0, 1.05, 1.08, 1.10, 1.15, 1.20]

    print("\n" + "="*80)
    print("Testing calibration powers...")
    print("="*80)

    results = []
    for power in test_powers:
        calibrated = calibrate(predictions, power)

        mean_val = calibrated.mean()
        std_val = calibrated.std()

        # Target positive rate is ~0.019
        distance_from_target = abs(mean_val - 0.019)

        results.append({
            'power': power,
            'mean': mean_val,
            'std': std_val,
            'distance': distance_from_target
        })

        print(f"Power {power:.2f}: mean={mean_val:.6f}, std={std_val:.6f}, distance={distance_from_target:.6f}")

    # Find best power (closest to target positive rate)
    best_result = min(results, key=lambda x: x['distance'])
    best_power = best_result['power']

    print("\n" + "="*80)
    print(f"Best calibration power: {best_power:.2f}")
    print(f"  Mean: {best_result['mean']:.6f}")
    print(f"  Distance from target: {best_result['distance']:.6f}")
    print("="*80)

    # Create final submissions
    print("\nCreating submission files...")

    # 1. Best calibrated version
    best_calibrated = calibrate(predictions, best_power)
    submission_best = pd.DataFrame({
        'ID': uncalibrated_df['ID'],
        'clicked': best_calibrated
    })
    submission_best.to_csv('plan3/005_best_calibrated_submission.csv', index=False)
    print(f"✓ Best calibrated (power={best_power:.2f}) saved to: plan3/005_best_calibrated_submission.csv")

    # 2. Original calibration (1.08) for comparison
    original_calibrated = calibrate(predictions, 1.08)
    submission_original = pd.DataFrame({
        'ID': uncalibrated_df['ID'],
        'clicked': original_calibrated
    })
    submission_original.to_csv('plan3/005_original_calibrated_submission.csv', index=False)
    print(f"✓ Original calibrated (power=1.08) saved to: plan3/005_original_calibrated_submission.csv")

    # 3. Conservative calibration (lower power)
    conservative_calibrated = calibrate(predictions, 1.0)
    submission_conservative = pd.DataFrame({
        'ID': uncalibrated_df['ID'],
        'clicked': conservative_calibrated
    })
    submission_conservative.to_csv('plan3/005_conservative_submission.csv', index=False)
    print(f"✓ Conservative (power=1.0) saved to: plan3/005_conservative_submission.csv")

    # 4. Aggressive calibration (higher power)
    aggressive_calibrated = calibrate(predictions, 1.15)
    submission_aggressive = pd.DataFrame({
        'ID': uncalibrated_df['ID'],
        'clicked': aggressive_calibrated
    })
    submission_aggressive.to_csv('plan3/005_aggressive_submission.csv', index=False)
    print(f"✓ Aggressive (power=1.15) saved to: plan3/005_aggressive_submission.csv")

    print("\n" + "="*80)
    print("SUBMISSION FILES CREATED!")
    print("="*80)
    print("\nRecommended submission:")
    print(f"  → plan3/005_best_calibrated_submission.csv (power={best_power:.2f})")
    print(f"     Mean: {best_result['mean']:.6f}")
    print("\nAll submissions created successfully!")

    return best_power


if __name__ == "__main__":
    best_power = main()