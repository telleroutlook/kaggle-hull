"""Utility helpers for persisting reusable modelling artefacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


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


__all__ = ["load_oof_entry", "update_oof_artifact"]
