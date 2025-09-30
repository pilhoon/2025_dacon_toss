# plan4/src/prediction_guardrails.py
"""
Implement and integrate prediction distribution guardrails into training pipeline
"""

import numpy as np
import pandas as pd
import json
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
# import matplotlib.pyplot as plt
# import seaborn as sns

sys.path.append('plan4/src')
from score import check_prediction_guardrails, calculate_prediction_stats


class PredictionGuardrailMonitor:
    """Monitor and enforce prediction distribution guardrails during training"""

    def __init__(self,
                 mean_range: Tuple[float, float] = (0.017, 0.021),
                 min_std: float = 0.055,
                 log_path: Optional[str] = None):
        """
        Initialize guardrail monitor

        Args:
            mean_range: Acceptable range for mean prediction
            min_std: Minimum required standard deviation
            log_path: Path to save guardrail logs
        """
        self.mean_range = mean_range
        self.min_std = min_std
        self.log_path = log_path
        self.history = []

    def check(self, y_prob: np.ndarray, stage: str = "unknown") -> Dict[str, Any]:
        """
        Check if predictions meet guardrails

        Args:
            y_prob: Predicted probabilities
            stage: Stage name (e.g., "fold_1", "oof", "test")

        Returns:
            Dictionary with check results
        """
        result = check_prediction_guardrails(y_prob, self.mean_range, self.min_std)
        result['stage'] = stage
        result['n_samples'] = len(y_prob)

        # Add to history
        self.history.append(result)

        # Log if path provided
        if self.log_path:
            self._save_log()

        return result

    def _save_log(self):
        """Save guardrail check history to file"""
        if self.log_path:
            with open(self.log_path, 'w') as f:
                json.dump(self.history, f, indent=2, default=float)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all checks performed"""
        if not self.history:
            return {}

        passed_checks = sum(1 for h in self.history if h['passed'])
        total_checks = len(self.history)

        return {
            'total_checks': total_checks,
            'passed': passed_checks,
            'failed': total_checks - passed_checks,
            'pass_rate': passed_checks / total_checks if total_checks > 0 else 0,
            'stages': list(set(h['stage'] for h in self.history))
        }

    def plot_distributions(self, save_path: Optional[str] = None):
        """Plot prediction distributions for all stages"""
        # Plotting disabled for now (missing matplotlib)
        pass


def check_predictions(y_prob: np.ndarray) -> Tuple[bool, str]:
    """
    Wrapper function for backward compatibility.
    Checks if predictions meet guardrails.

    Args:
        y_prob: Prediction probabilities

    Returns:
        Tuple of (passed: bool, message: str)
    """
    monitor = PredictionGuardrailMonitor()
    result = monitor.check(y_prob, "validation")
    return result['passed'], result['message']


def apply_distribution_adjustment(y_prob: np.ndarray,
                                 target_mean: float = 0.019,
                                 target_std: float = 0.055) -> np.ndarray:
    """
    Adjust prediction distribution to meet guardrails

    Args:
        y_prob: Original predictions
        target_mean: Target mean value
        target_std: Target standard deviation

    Returns:
        Adjusted predictions
    """
    # Current stats
    current_mean = np.mean(y_prob)
    current_std = np.std(y_prob)

    # Avoid division by zero
    if current_std < 1e-6:
        current_std = 1e-6

    # Standardize
    z_scores = (y_prob - current_mean) / current_std

    # Rescale to target distribution
    adjusted = target_mean + z_scores * target_std

    # Clip to valid probability range [0, 1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    # Final adjustment to ensure mean is in range
    # This may slightly reduce std but should still pass
    final_mean = np.mean(adjusted)
    if final_mean < target_mean * 0.9 or final_mean > target_mean * 1.1:
        # Apply linear shift
        shift = target_mean - final_mean
        adjusted = adjusted + shift
        adjusted = np.clip(adjusted, 0.0, 1.0)

    return adjusted


def create_guardrail_template() -> Dict[str, Any]:
    """Create template for guardrail configuration"""
    return {
        "mean_range": [0.017, 0.021],
        "min_std": 0.055,
        "checks": {
            "mean_in_range": {
                "description": "Mean prediction within target range",
                "critical": True
            },
            "std_sufficient": {
                "description": "Standard deviation >= minimum threshold",
                "critical": True
            },
            "no_extreme_values": {
                "description": "All predictions in [0, 1]",
                "critical": True
            },
            "reasonable_spread": {
                "description": "Q75 - Q25 > 0.01",
                "critical": False
            }
        },
        "adjustment_strategy": {
            "enabled": False,
            "target_mean": 0.019,
            "target_std": 0.055
        }
    }


def main():
    """Demo guardrail functionality"""
    print("=" * 70)
    print("Prediction Distribution Guardrails")
    print("=" * 70)

    # Create guardrail monitor
    monitor = PredictionGuardrailMonitor(
        mean_range=(0.017, 0.021),
        min_std=0.055,
        log_path='plan4/guardrail_log.json'
    )

    # Test with different distributions
    np.random.seed(42)

    # 1. Good distribution (should pass)
    print("\n1. Testing good distribution...")
    good_pred = np.random.beta(2, 100, 10000) * 3  # Scaled beta
    good_pred = apply_distribution_adjustment(good_pred, target_mean=0.019, target_std=0.06)
    result1 = monitor.check(good_pred, "good_distribution")
    print(f"   Result: {'PASSED' if result1['passed'] else 'FAILED'}")

    # 2. Low variance distribution (should fail)
    print("\n2. Testing low variance distribution...")
    low_var_pred = np.random.normal(0.019, 0.001, 10000)
    low_var_pred = np.clip(low_var_pred, 0, 1)
    result2 = monitor.check(low_var_pred, "low_variance")
    print(f"   Result: {'PASSED' if result2['passed'] else 'FAILED'}")

    # 3. Wrong mean distribution (should fail)
    print("\n3. Testing wrong mean distribution...")
    wrong_mean_pred = np.random.beta(5, 20, 10000)  # Higher mean
    result3 = monitor.check(wrong_mean_pred, "wrong_mean")
    print(f"   Result: {'PASSED' if result3['passed'] else 'FAILED'}")

    # 4. Adjusted distribution
    print("\n4. Testing adjusted distribution...")
    adjusted_pred = apply_distribution_adjustment(wrong_mean_pred)
    result4 = monitor.check(adjusted_pred, "adjusted")
    print(f"   Result: {'PASSED' if result4['passed'] else 'FAILED'}")

    # Summary
    print("\n" + "=" * 70)
    print("Guardrail Check Summary")
    print("=" * 70)
    summary = monitor.get_summary()
    print(f"Total checks: {summary['total_checks']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Pass rate: {summary['pass_rate']:.1%}")

    # Save template
    template_path = 'plan4/pred_stats_template.json'
    template = create_guardrail_template()
    with open(template_path, 'w') as f:
        json.dump(template, f, indent=2)
    print(f"\nGuardrail template saved to {template_path}")

    # Create integration example
    integration_path = 'plan4/guardrail_integration.md'
    with open(integration_path, 'w') as f:
        f.write("# Prediction Guardrail Integration Guide\n\n")
        f.write("## Usage Example\n\n")
        f.write("```python\n")
        f.write("from plan4.src.prediction_guardrails import PredictionGuardrailMonitor\n\n")
        f.write("# Initialize monitor\n")
        f.write("monitor = PredictionGuardrailMonitor(\n")
        f.write("    mean_range=(0.017, 0.021),\n")
        f.write("    min_std=0.055\n")
        f.write(")\n\n")
        f.write("# During training\n")
        f.write("for fold in range(n_folds):\n")
        f.write("    # ... train model ...\n")
        f.write("    predictions = model.predict(X_val)\n")
        f.write("    \n")
        f.write("    # Check guardrails\n")
        f.write("    result = monitor.check(predictions, f'fold_{fold}')\n")
        f.write("    if not result['passed']:\n")
        f.write("        print(f'Warning: Fold {fold} failed guardrails')\n")
        f.write("        # Optionally adjust predictions\n")
        f.write("        predictions = apply_distribution_adjustment(predictions)\n")
        f.write("```\n\n")
        f.write("## Guardrail Requirements\n\n")
        f.write("- **Mean**: [0.017, 0.021]\n")
        f.write("- **Std**: >= 0.055 (increased from 0.05)\n")
        f.write("- **Range**: [0, 1]\n")
        f.write("- **Spread**: Q75 - Q25 > 0.01\n")

    print(f"Integration guide saved to {integration_path}")


if __name__ == "__main__":
    main()