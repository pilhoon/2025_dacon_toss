#!/usr/bin/env python
"""
Format submission files with TEST_ prefix for Dacon platform
"""

import pandas as pd
import os

def format_submission(input_path, output_path):
    """Format a submission file with TEST_ prefix"""
    print(f"Formatting: {input_path}")

    # Read submission
    df = pd.read_csv(input_path)

    # Check current format
    if 'ID' in df.columns:
        first_id = str(df['ID'].iloc[0])
        if first_id.startswith('TEST_'):
            print(f"  Already formatted with TEST_ prefix")
            return

        # Convert ID to TEST_ format
        df['ID'] = df['ID'].apply(lambda x: f'TEST_{int(x):07d}')

        # Save formatted version
        df.to_csv(output_path, index=False)
        print(f"  ✓ Saved to: {output_path}")

        # Print stats
        print(f"  Predictions - Mean: {df['clicked'].mean():.6f}, Std: {df['clicked'].std():.6f}")
    else:
        print(f"  ERROR: No ID column found")

def main():
    """Format all recent submission files"""
    print("="*80)
    print("FORMATTING SUBMISSIONS FOR DACON")
    print("="*80)

    # List of files to format
    files_to_format = [
        # Probing submissions
        ('plan3/014_probe_temporal.csv', 'plan3/015_probe_temporal_formatted.csv'),
        ('plan3/014_probe_low_ctr.csv', 'plan3/015_probe_low_ctr_formatted.csv'),
        ('plan3/014_probe_no_f1.csv', 'plan3/015_probe_no_f1_formatted.csv'),
        # GPU maximized model
        ('plan3/013_gpu_maximized_submission.csv', 'plan3/015_gpu_maximized_formatted.csv'),
    ]

    # Process each file
    for input_file, output_file in files_to_format:
        if os.path.exists(input_file):
            format_submission(input_file, output_file)
        else:
            print(f"File not found: {input_file}")

    print("\n" + "="*80)
    print("FORMATTING COMPLETE")
    print("="*80)
    print("\nFormatted files ready for submission:")
    print("1. plan3/015_probe_temporal_formatted.csv - Test temporal distribution hypothesis")
    print("2. plan3/015_probe_low_ctr_formatted.csv - Test low CTR hypothesis")
    print("3. plan3/015_probe_no_f1_formatted.csv - Test feature importance hypothesis")
    print("4. plan3/015_gpu_maximized_formatted.csv - GPU maximized deep model")
    print("\nSubmit these files to Dacon and record the scores!")

if __name__ == "__main__":
    main()