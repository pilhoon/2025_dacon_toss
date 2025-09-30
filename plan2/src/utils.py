from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
import yaml


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def match_patterns(names: Iterable[str], include_patterns: Optional[List[str]], exclude_patterns: Optional[List[str]]) -> List[str]:
    names = list(names)
    if not include_patterns:
        included = names
    else:
        included = []
        for name in names:
            for pat in include_patterns:
                regex = re.compile("^" + pat.replace("*", ".*") + "$")
                if regex.match(name):
                    included.append(name)
                    break
    if exclude_patterns:
        filtered: List[str] = []
        for name in included:
            excluded = False
            for pat in exclude_patterns:
                regex = re.compile("^" + pat.replace("*", ".*") + "$")
                if regex.match(name):
                    excluded = True
                    break
            if not excluded:
                filtered.append(name)
        return filtered
    return included


def save_json(obj: dict, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


