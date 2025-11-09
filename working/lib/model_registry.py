"""
Shared model configuration presets to keep training/inference in sync.
"""

from __future__ import annotations

import os
from typing import Dict, Any

DEFAULT_MODEL_TYPE = os.getenv("HULL_MODEL_TYPE", "lightgbm").lower()

MODEL_PRESETS: Dict[str, Any] = {
    "lightgbm": {
        "n_estimators": 2500,
        "learning_rate": 0.01,
        "num_leaves": 192,
        "subsample": 0.85,
        "colsample_bytree": 0.75,
        "reg_lambda": 3.0,
        "min_child_samples": 45,
    },
    "xgboost": {
        "n_estimators": 2000,
        "learning_rate": 0.01,
        "max_depth": 7,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_lambda": 2.0,
        "min_child_weight": 8,
    },
    "catboost": {
        "iterations": 4000,
        "learning_rate": 0.02,
        "depth": 8,
        "l2_leaf_reg": 5.0,
        "random_strength": 0.8,
        "bagging_temperature": 0.9,
        "loss_function": "RMSE",
    },
    "ensemble": {
        "weights": {"lightgbm": 0.5, "xgboost": 0.3, "catboost": 0.2},
        "lightgbm": {
            "n_estimators": 2000,
            "learning_rate": 0.008,
            "num_leaves": 128,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
            "reg_lambda": 2.5,
        },
        "xgboost": {
            "n_estimators": 1500,
            "learning_rate": 0.01,
            "max_depth": 6,
            "subsample": 0.85,
            "colsample_bytree": 0.75,
            "reg_lambda": 2.0,
        },
        "catboost": {
            "iterations": 2500,
            "learning_rate": 0.018,
            "depth": 7,
            "l2_leaf_reg": 4.0,
            "random_strength": 0.7,
            "bagging_temperature": 0.8,
        },
    },
    "baseline": {},
}


def resolve_model_type(requested: str | None = None) -> str:
    """Choose a supported model type."""

    model_type = (requested or DEFAULT_MODEL_TYPE or "lightgbm").lower()
    if model_type not in MODEL_PRESETS:
        return "lightgbm"
    return model_type


def get_model_params(model_type: str) -> Dict[str, Any]:
    """Return preset hyper-parameters for the requested model."""

    return MODEL_PRESETS.get(model_type, {})


__all__ = ["DEFAULT_MODEL_TYPE", "MODEL_PRESETS", "resolve_model_type", "get_model_params"]
