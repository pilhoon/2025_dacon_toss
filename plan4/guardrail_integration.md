# Prediction Guardrail Integration Guide

## Usage Example

```python
# Import from the src module
from plan4.src.prediction_guardrails import (
    PredictionGuardrailMonitor,
    apply_distribution_adjustment
)

# Initialize monitor
monitor = PredictionGuardrailMonitor(
    mean_range=(0.017, 0.021),
    min_std=0.055
)

# During training
for fold in range(n_folds):
    # ... train model ...
    predictions = model.predict(X_val)

    # Check guardrails
    result = monitor.check(predictions, f'fold_{fold}')
    if not result['passed']:
        print(f'Warning: Fold {fold} failed guardrails')
        # Optionally adjust predictions
        predictions = apply_distribution_adjustment(predictions)
```

## Guardrail Requirements

- **Mean**: [0.017, 0.021]
- **Std**: >= 0.055 (increased from 0.05)
- **Range**: [0, 1]
- **Spread**: Q75 - Q25 > 0.01
