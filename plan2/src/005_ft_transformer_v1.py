from __future__ import annotations

import json
import subprocess
from pathlib import Path

from plan2.src.log_utils import append_md_entry


def run() -> None:
    out = Path("plan2/experiments/005_ft_transformer_v1")
    out.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "plan2/src/train.py",
        "--config",
        "plan2/configs/ft_transformer.yaml",
        "--trainer",
        "plan2/configs/trainer.yaml",
        "--out",
        out.as_posix(),
    ]
    print("[RUN]", " ".join(cmd))
    import os
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", ".")
    subprocess.run(cmd, check=True, env=env)

    metrics_path = out / "oof_metrics.json"
    if not metrics_path.exists():
        metrics_path = out / "final_metrics.json"
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as f:
            m = json.load(f)
        bullets = [
            f"AP={m.get('ap'):.6f}",
            f"WLL={m.get('wll'):.6f}",
            f"ROC={m.get('roc_auc'):.6f}",
            f"composite={m.get('composite'):.6f}",
            f"out={out.as_posix()}",
        ]
        append_md_entry("plan2/000_EXPERIMENT_LOG.md", "005 FT-Transformer v1", "OOF or single snapshot", bullets)
    else:
        append_md_entry("plan2/000_EXPERIMENT_LOG.md", "005 FT-Transformer v1", "run finished; metrics missing", [f"out={out.as_posix()}"])


if __name__ == "__main__":
    run()
