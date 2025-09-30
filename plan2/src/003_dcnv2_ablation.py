from __future__ import annotations

import itertools
import json
import subprocess
from pathlib import Path
from typing import Dict, Tuple

from plan2.src.utils import load_yaml, save_json
from plan2.src.log_utils import append_md_entry


GRID = {
    "embed_dim": [16, 32, 64],
    "cross_depth": [2, 3, 4],
}


def run_one(out_dir: Path, cfg_path: str, trainer_path: str, overrides: Dict[str, int]) -> Tuple[Path, Dict[str, float]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    # make a temp config copy with overrides
    base = load_yaml(cfg_path)
    base["model"]["embed_dim"] = int(overrides["embed_dim"])  # type: ignore
    base["model"]["cross_depth"] = int(overrides["cross_depth"])  # type: ignore
    tmp_cfg = out_dir / "config.overridden.yaml"
    with tmp_cfg.open("w", encoding="utf-8") as f:
        import yaml
        yaml.safe_dump(base, f)

    cmd = [
        "python",
        "plan2/src/train.py",
        "--config",
        tmp_cfg.as_posix(),
        "--trainer",
        trainer_path,
        "--out",
        out_dir.as_posix(),
    ]
    print("[RUN]", " ".join(cmd))
    import os
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", ".")
    subprocess.run(cmd, check=True, env=env)

    # read oof preferred; fall back to final
    metrics_path = out_dir / "oof_metrics.json"
    if not metrics_path.exists():
        metrics_path = out_dir / "final_metrics.json"
    metrics: Dict[str, float] = {}
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
    return metrics_path, metrics


def main() -> None:
    root = Path("plan2/experiments/003_dcnv2_ablation")
    root.mkdir(parents=True, exist_ok=True)
    best = {"composite": -1.0, "path": ""}
    results = []
    for ed, cd in itertools.product(GRID["embed_dim"], GRID["cross_depth"]):
        name = f"ed{ed}_cd{cd}"
        out = root / name
        mp, m = run_one(out, "plan2/configs/dcnv2.yaml", "plan2/configs/trainer.yaml", {"embed_dim": ed, "cross_depth": cd})
        composite = float(m.get("composite", -1))
        results.append({"name": name, **m})
        if composite > best["composite"]:
            best = {"composite": composite, "path": out.as_posix()}
        append_md_entry("plan2/000_EXPERIMENT_LOG.md", "003 DCNv2 ablation", f"{name}", [f"metrics={m}", f"out={out.as_posix()}"])

    save_json({"results": results, "best": best}, root / "summary.json")
    append_md_entry("plan2/000_EXPERIMENT_LOG.md", "003 DCNv2 ablation", "completed grid", [f"best_composite={best['composite']:.6f}", f"best_out={best['path']}"])


if __name__ == "__main__":
    main()
