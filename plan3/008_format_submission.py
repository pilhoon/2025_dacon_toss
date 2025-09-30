#!/usr/bin/env python
"""
Format submission file to ensure compatibility
"""

import pandas as pd
import numpy as np

def format_submission(input_path, output_path):
    """Format submission file for Dacon platform"""
    print(f"Reading {input_path}...")
    df = pd.read_csv(input_path)

    print(f"Original shape: {df.shape}")
    print(f"Original clicked stats:")
    print(f"  Mean: {df['clicked'].mean():.6f}")
    print(f"  Min: {df['clicked'].min():.10f}")
    print(f"  Max: {df['clicked'].max():.10f}")

    # Ensure ID is integer
    df['ID'] = df['ID'].astype(int)

    # Round predictions to 10 decimal places (Dacon standard)
    df['clicked'] = df['clicked'].round(10)

    # Clip to valid probability range
    df['clicked'] = np.clip(df['clicked'], 0.0, 1.0)

    # Remove any scientific notation
    pd.options.display.float_format = '{:.10f}'.format

    print(f"\nFormatted clicked stats:")
    print(f"  Mean: {df['clicked'].mean():.6f}")
    print(f"  Min: {df['clicked'].min():.10f}")
    print(f"  Max: {df['clicked'].max():.10f}")

    # Save with specific formatting
    df.to_csv(output_path, index=False, float_format='%.10f')
    print(f"\nSaved formatted submission to: {output_path}")

    # Verify the saved file
    df_check = pd.read_csv(output_path, nrows=5)
    print("\nFirst 5 rows of formatted file:")
    print(df_check)

    return df

# Format the best submission
print("="*80)
print("FORMATTING SUBMISSION FOR DACON")
print("="*80)

# Process the best calibrated submission
df = format_submission(
    'plan3/005_best_calibrated_submission.csv',
    'plan3/008_formatted_submission.csv'
)

print("\n✓ Submission formatted successfully!")
print("File ready for upload: plan3/008_formatted_submission.csv")