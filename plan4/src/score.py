# plan4/src/score.py
"""
Official competition score implementation and validation
Score = 0.5 * AP + 0.5 * (1 / (1 + WLL))
"""

import numpy as np
from sklearn.metrics import average_precision_score
from typing import Tuple, Dict, Any


def weighted_logloss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
    """
    Weighted LogLoss with equal weight (0.5) for positive and negative classes.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        eps: Small value to avoid log(0)

    Returns:
        Weighted log loss value
    """
    p = np.clip(y_prob, eps, 1 - eps)
    y_true = np.asarray(y_true).astype(np.float64)

    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0:
        return float('nan')

    w_pos = 0.5 / n_pos
    w_neg = 0.5 / n_neg

    ll = -(w_pos * (y_true * np.log(p)).sum() +
           w_neg * ((1 - y_true) * np.log(1 - p)).sum())

    return ll


def competition_score(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float, float]:
    """
    Calculate official competition score.

    Formula: Score = 0.5 * AP + 0.5 * (1 / (1 + WLL))

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities

    Returns:
        Tuple of (AP, WLL, final_score)
    """
    ap = average_precision_score(y_true, y_prob)
    wll = weighted_logloss(y_true, y_prob)
    score = 0.5 * ap + 0.5 * (1.0 / (1.0 + wll))

    return ap, wll, score


def alternative_score(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float, float]:
    """
    Alternative score formula from Plan3 notes.

    Formula: Score = 0.7 * AP + 0.3 / WLL

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities

    Returns:
        Tuple of (AP, WLL, final_score)
    """
    ap = average_precision_score(y_true, y_prob)
    wll = weighted_logloss(y_true, y_prob)

    # Protect against division by zero
    if wll == 0:
        score = 0.7 * ap + 0.3  # Maximum possible for second term
    else:
        score = 0.7 * ap + 0.3 / wll

    return ap, wll, score


def calculate_prediction_stats(y_prob: np.ndarray) -> Dict[str, float]:
    """
    Calculate prediction distribution statistics for guardrails.

    Args:
        y_prob: Predicted probabilities

    Returns:
        Dictionary with distribution statistics
    """
    return {
        'mean': float(np.mean(y_prob)),
        'std': float(np.std(y_prob)),
        'min': float(np.min(y_prob)),
        'max': float(np.max(y_prob)),
        'q25': float(np.percentile(y_prob, 25)),
        'q50': float(np.percentile(y_prob, 50)),
        'q75': float(np.percentile(y_prob, 75)),
        'skewness': float(_calculate_skewness(y_prob)),
        'kurtosis': float(_calculate_kurtosis(y_prob))
    }


def _calculate_skewness(x: np.ndarray) -> float:
    """Calculate skewness of distribution."""
    mean = np.mean(x)
    std = np.std(x)
    if std == 0:
        return 0.0
    return np.mean(((x - mean) / std) ** 3)


def _calculate_kurtosis(x: np.ndarray) -> float:
    """Calculate excess kurtosis of distribution."""
    mean = np.mean(x)
    std = np.std(x)
    if std == 0:
        return 0.0
    return np.mean(((x - mean) / std) ** 4) - 3.0


def check_prediction_guardrails(y_prob: np.ndarray,
                               mean_range: Tuple[float, float] = (0.017, 0.021),
                               min_std: float = 0.055) -> Dict[str, Any]:
    """
    Check if predictions meet distribution guardrails.

    Args:
        y_prob: Predicted probabilities
        mean_range: Acceptable range for mean prediction
        min_std: Minimum required standard deviation

    Returns:
        Dictionary with check results and statistics
    """
    stats = calculate_prediction_stats(y_prob)

    checks = {
        'mean_in_range': mean_range[0] <= stats['mean'] <= mean_range[1],
        'std_sufficient': stats['std'] >= min_std,
        'no_extreme_values': (stats['min'] >= 0.0) and (stats['max'] <= 1.0),
        'reasonable_spread': stats['q75'] - stats['q25'] > 0.01
    }

    all_passed = all(checks.values())

    return {
        'passed': all_passed,
        'checks': checks,
        'stats': stats,
        'requirements': {
            'mean_range': mean_range,
            'min_std': min_std
        }
    }


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    n_samples = 10000

    # Generate synthetic predictions
    y_true = np.random.binomial(1, 0.019, n_samples)
    y_prob = np.random.beta(2, 100, n_samples)  # Skewed distribution similar to CTR

    # Test both scoring formulas
    ap1, wll1, score1 = competition_score(y_true, y_prob)
    ap2, wll2, score2 = alternative_score(y_true, y_prob)

    print("=" * 60)
    print("Score Formula Comparison")
    print("=" * 60)
    print(f"\nOfficial Formula: 0.5*AP + 0.5*(1/(1+WLL))")
    print(f"  AP: {ap1:.5f}")
    print(f"  WLL: {wll1:.5f}")
    print(f"  Score: {score1:.5f}")

    print(f"\nAlternative Formula: 0.7*AP + 0.3/WLL")
    print(f"  AP: {ap2:.5f}")
    print(f"  WLL: {wll2:.5f}")
    print(f"  Score: {score2:.5f}")

    # Check prediction guardrails
    guardrail_check = check_prediction_guardrails(y_prob)

    print("\n" + "=" * 60)
    print("Prediction Distribution Guardrails")
    print("=" * 60)
    print(f"Overall: {'PASSED' if guardrail_check['passed'] else 'FAILED'}")
    print("\nChecks:")
    for check, passed in guardrail_check['checks'].items():
        print(f"  {check}: {'✓' if passed else '✗'}")

    print("\nStatistics:")
    for stat, value in guardrail_check['stats'].items():
        print(f"  {stat}: {value:.5f}")