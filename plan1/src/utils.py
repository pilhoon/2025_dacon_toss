import os
import re
from typing import Iterable, List


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def match_patterns(names: Iterable[str], include_patterns: List[str] | None, exclude_patterns: List[str] | None) -> List[str]:
    if include_patterns is None or len(include_patterns) == 0:
        included = list(names)
    else:
        included = []
        for name in names:
            for pat in include_patterns:
                regex = re.compile("^" + pat.replace("*", ".*") + "$")
                if regex.match(name):
                    included.append(name)
                    break
    if exclude_patterns:
        filtered = []
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


