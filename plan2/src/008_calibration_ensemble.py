from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression

from plan2.src.log_utils import append_md_entry


def logloss(y: np.ndarray, p: np.ndarray, eps: float = 1e-7) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def temperature_scale(y: np.ndarray, logits: np.ndarray) -> Tuple[float, float]:
    # simple 1D search over temperature to minimize logloss on OOF
    best_t, best_ll = 1.0, 1e9
    for t in np.linspace(0.2, 5.0, 97):
        p = 1 / (1 + np.exp(-logits / max(1e-6, t)))
        ll = logloss(y, p)
        if ll < best_ll:
            best_t, best_ll = float(t), float(ll)
    return best_t, best_ll


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", required=True, help="Path to OOF probabilities (.npy) or logits")
    parser.add_argument("--labels", required=True, help="Path to labels (.npy)")
    parser.add_argument("--outdir", default="plan2/experiments/008_calibration", help="Output directory")
    args = parser.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    # load arrays
    probs_or_logits = np.load(args.oof)
    y = np.load(args.labels)

    # detect if input looks like logits or probs
    is_logit = np.any(probs_or_logits < 0) or np.any(probs_or_logits > 1)
    if is_logit:
        logits = probs_or_logits.astype(np.float64)
        probs = 1 / (1 + np.exp(-logits))
    else:
        probs = probs_or_logits.astype(np.float64)
        # back out logits for temperature scaling
        logits = np.log(np.clip(probs, 1e-7, 1 - 1e-7)) - np.log(1 - np.clip(probs, 1e-7, 1 - 1e-7))

    base_ll = logloss(y, probs)

    # temperature scaling
    t, t_ll = temperature_scale(y, logits)
    p_t = 1 / (1 + np.exp(-logits / max(1e-6, t)))

    # isotonic regression
    iso = IsotonicRegression(out_of_bounds="clip")
    p_iso = iso.fit_transform(probs, y)
    iso_ll = logloss(y, p_iso)

    # choose better
    method = "temperature" if t_ll <= iso_ll else "isotonic"
    chosen_ll = min(t_ll, iso_ll)
    out_metrics = {
        "base_logloss": base_ll,
        "temp_logloss": t_ll,
        "iso_logloss": iso_ll,
        "chosen": method,
        "temperature": t,
    }
    with (out / "001_calibration_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(out_metrics, f, indent=2)

    append_md_entry(
        "plan2/000_EXPERIMENT_LOG.md",
        "008 Calibration",
        f"base={base_ll:.6f} chosen={method} ll={chosen_ll:.6f}",
    )

    # save calibrated probabilities
    np.save(out / "001_probs_base.npy", probs)
    np.save(out / "001_probs_temp.npy", p_t)
    np.save(out / "001_probs_iso.npy", p_iso)


if __name__ == "__main__":
    main()

