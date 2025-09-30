from __future__ import annotations

import json
import subprocess
from pathlib import Path

from plan2.src.log_utils import append_md_entry


def run() -> None:
    out = Path("plan2/experiments/002_dcnv2_v1")
    out.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "plan2/src/train.py",
        "--config",
        "plan2/configs/dcnv2.yaml",
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

    final_metrics_path = out / "final_metrics.json"
    if final_metrics_path.exists():
        with final_metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        bullets = [
            f"AP={metrics.get('ap'):.6f}",
            f"WLL={metrics.get('wll'):.6f}",
            f"ROC={metrics.get('roc_auc'):.6f}",
            f"composite={metrics.get('composite'):.6f}",
            f"out={out.as_posix()}",
        ]
        append_md_entry("plan2/000_EXPERIMENT_LOG.md", "002 DCNv2 v1", "single-run snapshot after epochs", bullets)
    else:
        append_md_entry("plan2/000_EXPERIMENT_LOG.md", "002 DCNv2 v1", "run finished; metrics missing", [f"out={out.as_posix()}"])


if __name__ == "__main__":
    run()
