
from typing import Optional, Dict
import numpy as np
from sklearn.metrics import average_precision_score, log_loss

def score_components(y_true: np.ndarray, y_prob: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> Dict[str, float]:
    ap = float(average_precision_score(y_true, y_prob, sample_weight=sample_weight))
    wll = float(log_loss(y_true, y_prob, sample_weight=sample_weight, labels=[0,1]))
    return {'AP': ap, 'WLL': wll, 'Score': 0.5 * ap + 0.5 * (1.0 / (1.0 + wll))}
