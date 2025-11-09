"""Utility helpers for persisting reusable modelling artefacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .env import LogPaths, PROJECT_ROOT


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fp:
            return json.load(fp)
    except (json.JSONDecodeError, OSError):
        return {}


def load_oof_entry(path: Path, model_type: str) -> Optional[Dict[str, Any]]:
    """Load the cached OOF calibration for a specific model type."""

    data = _load_json(path)
    return data.get(model_type)


def update_oof_artifact(path: Path, model_type: str, payload: Dict[str, Any]) -> None:
    """Persist (or overwrite) the OOF artefact for ``model_type``."""

    path.parent.mkdir(parents=True, exist_ok=True)
    data = _load_json(path)
    data[model_type] = payload
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)


def load_first_available_oof(
    model_type: str, candidates: Iterable[Path]
) -> Tuple[Optional[Dict[str, Any]], Optional[Path]]:
    """Return the first available OOF artefact in ``candidates``.

    The helper is resilient to missing files (as happens on Kaggle where the
    writable ``/kaggle/working`` directory starts empty) by probing multiple
    potential locations and returning whichever yields an entry first.
    """

    for candidate in candidates:
        entry = load_oof_entry(candidate, model_type)
        if entry:
            return entry, candidate
    return None, None


def _deduplicate_paths(paths: Iterable[Path]) -> List[Path]:
    unique: List[Path] = []
    seen = set()
    for raw in paths:
        normalized = Path(raw).expanduser()
        key = normalized.resolve(strict=False)
        if key in seen:
            continue
        seen.add(key)
        unique.append(normalized)
    return unique


def oof_artifact_candidates(log_paths: LogPaths) -> List[Path]:
    """Return ordered candidate paths for locating the OOF artefact file."""

    override = os.getenv("HULL_OOF_PATH")
    packaged = PROJECT_ROOT / "working" / "artifacts" / "oof_summary.json"
    candidates = []
    if override:
        candidates.append(Path(override))
    candidates.extend([log_paths.oof_metrics, packaged])
    return _deduplicate_paths(candidates)


__all__ = [
    "load_oof_entry",
    "update_oof_artifact",
    "load_first_available_oof",
    "oof_artifact_candidates",
]
