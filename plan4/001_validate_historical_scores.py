# plan4/001_validate_historical_scores.py
"""
Validate historical submission scores to determine correct formula
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
sys.path.append('plan4/src')
from score import competition_score, alternative_score, calculate_prediction_stats, check_prediction_guardrails


def load_train_labels():
    """Load training labels for score calculation"""
    train_path = "data/train.parquet"
    if not os.path.exists(train_path):
        print(f"Warning: {train_path} not found")
        return None

    # Note: train data doesn't have user_id, using index as proxy
    df = pd.read_parquet(train_path, columns=['clicked'])
    df = df.rename(columns={'clicked': 'is_click'})
    df['user_id'] = df.index
    return df


def analyze_submission(submission_path, label_df=None):
    """Analyze a submission file"""

    if not os.path.exists(submission_path):
        print(f"Submission file not found: {submission_path}")
        return None

    sub_df = pd.read_csv(submission_path)

    # Check for different submission formats
    pred_col = None
    id_col = None

    if 'clicked' in sub_df.columns:
        pred_col = 'clicked'
        if 'ID' in sub_df.columns:
            id_col = 'ID'
        elif 'user_id' in sub_df.columns:
            id_col = 'user_id'
    elif 'is_click' in sub_df.columns:
        pred_col = 'is_click'
        if 'user_id' in sub_df.columns:
            id_col = 'user_id'
        elif 'ID' in sub_df.columns:
            id_col = 'ID'

    if pred_col is None:
        print(f"Invalid submission format: {submission_path}")
        return None

    result = {
        'file': submission_path,
        'n_samples': len(sub_df),
        'n_unique_ids': sub_df[id_col].nunique() if id_col else len(sub_df)
    }

    # Get prediction statistics
    y_prob = sub_df[pred_col].values
    stats = calculate_prediction_stats(y_prob)
    result['stats'] = stats

    # Check guardrails with updated requirements (std >= 0.055)
    guardrails = check_prediction_guardrails(y_prob, mean_range=(0.017, 0.021), min_std=0.055)
    result['guardrails'] = guardrails

    # Skip label matching for now since we don't have proper ID mapping
    # Focus on prediction distribution analysis

    return result


def find_all_submissions():
    """Find all submission files in the project"""
    patterns = [
        'plan*/submission*.csv',
        'plan*/*submission*.csv',
        'plan*/output/*submission*.csv',
        'submission*.csv'
    ]

    submissions = []
    for pattern in patterns:
        for path in Path('.').glob(pattern):
            if path.is_file():
                submissions.append(str(path))

    # Remove duplicates
    submissions = list(set(submissions))
    submissions.sort()

    return submissions


def main():
    print("=" * 70)
    print("Historical Submission Score Validation")
    print("=" * 70)

    # Load training labels if available
    label_df = load_train_labels()

    # Find all submissions
    submissions = find_all_submissions()
    print(f"\nFound {len(submissions)} submission files")

    if len(submissions) == 0:
        print("No submission files found")
        return

    results = []
    for sub_path in submissions:
        print(f"\nAnalyzing: {sub_path}")
        result = analyze_submission(sub_path, label_df)
        if result:
            results.append(result)

            # Print summary
            print(f"  Samples: {result['n_samples']:,}")
            print(f"  Mean: {result['stats']['mean']:.5f}")
            print(f"  Std: {result['stats']['std']:.5f}")
            print(f"  Guardrails (std>=0.055): {'PASSED' if result['guardrails']['passed'] else 'FAILED'}")

            if 'scores' in result:
                print(f"  Official Score: {result['scores']['official']['score']:.5f}")
                print(f"  Alternative Score: {result['scores']['alternative']['score']:.5f}")

    # Save results
    output_path = 'plan4/metrics_validation.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)

    print(f"\nResults saved to {output_path}")

    # Create markdown report
    report_path = 'plan4/metrics_validation.md'
    with open(report_path, 'w') as f:
        f.write("# Metrics Validation Report\n\n")
        f.write("## Summary\n\n")
        f.write(f"- Analyzed {len(results)} submission files\n")
        f.write(f"- Official Formula: `Score = 0.5*AP + 0.5*(1/(1+WLL))`\n")
        f.write(f"- Alternative Formula: `Score = 0.7*AP + 0.3/WLL`\n\n")

        f.write("## Guardrail Requirements\n\n")
        f.write("- Mean: [0.017, 0.021]\n")
        f.write("- Std: >= 0.055 (updated from 0.05)\n\n")

        f.write("## Submission Analysis\n\n")
        f.write("| File | Samples | Mean | Std | Guardrails |\n")
        f.write("|------|---------|------|-----|------------|\n")

        for r in results:
            guardrail_status = '✓' if r['guardrails']['passed'] else '✗'
            f.write(f"| {r['file']} | {r['n_samples']:,} | {r['stats']['mean']:.5f} | {r['stats']['std']:.5f} | {guardrail_status} |\n")

        f.write("\n## Recommendations\n\n")
        f.write("1. Use official formula: `0.5*AP + 0.5*(1/(1+WLL))`\n")
        f.write("2. Ensure predictions have std >= 0.055 for better AP\n")
        f.write("3. Target mean prediction around 0.019 ± 0.002\n")
        f.write("4. Apply calibration methods to balance AP and WLL\n")

    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()