#!/usr/bin/env python
"""
Format submission file exactly like successful 046 submission
"""

import pandas as pd
import numpy as np

def format_like_046(input_path, output_path):
    """Format submission file to match 046 FT Transformer format"""
    print(f"Reading {input_path}...")
    df = pd.read_csv(input_path)

    print(f"Original shape: {df.shape}")
    print(f"Original ID format: {df['ID'].iloc[0]}")
    print(f"Original clicked range: [{df['clicked'].min():.6f}, {df['clicked'].max():.6f}]")

    # Convert ID to TEST_ format with 7 digit padding
    df['ID'] = df['ID'].apply(lambda x: f'TEST_{int(x):07d}')

    print(f"\nFormatted ID format: {df['ID'].iloc[0]}")
    print(f"Clicked range: [{df['clicked'].min():.6f}, {df['clicked'].max():.6f}]")

    # Save without scientific notation
    df.to_csv(output_path, index=False)
    print(f"\nSaved formatted submission to: {output_path}")

    # Verify the saved file
    df_check = pd.read_csv(output_path, nrows=5)
    print("\nFirst 5 rows of formatted file:")
    print(df_check)

    return df

# Format the best submission exactly like 046
print("="*80)
print("FORMATTING SUBMISSION LIKE 046 FT TRANSFORMER")
print("="*80)

# Process the best calibrated submission
df = format_like_046(
    'plan3/008_formatted_submission.csv',
    'plan3/009_dacon_submission.csv'
)

print("\n✓ Submission formatted successfully!")
print("File ready for upload: plan3/009_dacon_submission.csv")