#!/usr/bin/env python3
"""
Quick offline training harness for iterative experiments.

Usage:
    python train_experiment.py --model-type lightgbm --n-splits 5
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from lib.artifacts import update_oof_artifact
from lib.data import get_feature_columns
from lib.evaluation import backtest_strategy
from lib.features import FeaturePipeline
from lib.env import detect_run_environment, get_log_paths
from lib.model_registry import get_model_params
from lib.models import HullModel
from lib.strategy import (
    scale_to_allocation,
    optimize_scale_with_rolling_cv,
    apply_volatility_overlay,
)
from lib.utils import save_metrics

DATA_ROOT = Path(__file__).resolve().parents[1] / "input" / "hull-tactical-market-prediction"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline trainer for Hull Tactical models.")
    parser.add_argument(
        "--model-type",
        choices=["baseline", "lightgbm", "xgboost", "catboost", "ensemble"],
        default="lightgbm",
    )
    parser.add_argument("--n-splits", type=int, default=5, help="Number of TimeSeriesSplit folds.")
    parser.add_argument("--sample-frac", type=float, default=1.0, help="Optional subsampling for fast debugging.")
    parser.add_argument("--clip-quantile", type=float, default=0.01)
    parser.add_argument("--missing-indicator-threshold", type=float, default=0.05)
    parser.add_argument("--no-standardize", action="store_true", help="Disable z-score scaling in FeaturePipeline.")
    parser.add_argument("--model-param", action="append", default=[], help="Override model params key=value.")
    parser.add_argument("--scale-cv-splits", type=int, default=4, help="Rolling folds for scale tuning.")
    parser.add_argument("--overlay-lookback", type=int, default=63, help="Lookback for volatility overlay.")
    parser.add_argument(
        "--overlay-min-periods",
        type=int,
        default=21,
        help="Min history before overlay becomes active.",
    )
    parser.add_argument(
        "--overlay-vol-cap",
        type=float,
        default=1.2,
        help="Max multiple of market vol before down-scaling allocations.",
    )
    parser.add_argument(
        "--std-guard-threshold",
        type=float,
        default=0.15,
        help="Minimum std of raw predictions before triggering fallback safeguards (<=0 disables).",
    )
    parser.add_argument(
        "--std-guard-fallback-model",
        choices=["baseline", "lightgbm", "xgboost", "catboost", "ensemble", "none"],
        default="baseline",
        help="Simpler model to fall back to when prediction std collapses (use 'none' to disable).",
    )
    parser.add_argument(
        "--std-guard-min-scale",
        type=float,
        default=35.0,
        help="Minimum leverage scale enforced when std guard fires.",
    )
    return parser.parse_args()


def parse_model_params(param_list: list[str]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}

    def _cast(raw: str) -> Any:
        lowered = raw.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        try:
            return float(raw) if "." in raw else int(raw)
        except ValueError:
            return raw

    for item in param_list:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        target = params
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})  # type: ignore[assignment]
        target[parts[-1]] = _cast(value)
    return params


def merge_params(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_params(merged[key], value)
        else:
            merged[key] = value
    return merged


def main() -> None:
    args = parse_args()
    if args.std_guard_fallback_model == "none":  # type: ignore[attr-defined]
        args.std_guard_fallback_model = None  # type: ignore[assignment]
    train_path = DATA_ROOT / "train.csv"
    df = pd.read_csv(train_path)
    if args.sample_frac < 1.0:
        df = df.sample(frac=args.sample_frac, random_state=42).sort_values("date_id")

    env = detect_run_environment()
    log_paths = get_log_paths(env)

    pipeline = FeaturePipeline(
        clip_quantile=args.clip_quantile,
        missing_indicator_threshold=args.missing_indicator_threshold,
        standardize=not args.no_standardize,
    )
    features = pipeline.fit_transform(df)
    target = df["forward_returns"].reset_index(drop=True)
    features = features.reset_index(drop=True)

    overrides = parse_model_params(args.model_param)
    base_params = get_model_params(args.model_type)
    model_params = merge_params(base_params, overrides)
    splitter = TimeSeriesSplit(n_splits=args.n_splits)
    fold_metrics = []
    n_rows = len(df)
    oof_preds = np.full(n_rows, np.nan, dtype=float)
    oof_allocations = np.full(n_rows, np.nan, dtype=float)
    oof_overlay_allocations = np.full(n_rows, np.nan, dtype=float)

    for fold, (train_idx, val_idx) in enumerate(splitter.split(features)):
        active_model_type = args.model_type
        model = HullModel(active_model_type, model_params=model_params)
        model.fit(features.iloc[train_idx], target.iloc[train_idx])
        train_preds = model.predict(features.iloc[train_idx], clip=False)

        min_splits = max(2, min(args.scale_cv_splits, len(train_idx) - 1))

        def _tune_scale(predictions: np.ndarray) -> tuple[float, Dict[str, Any]]:
            info = optimize_scale_with_rolling_cv(
                predictions,
                target.iloc[train_idx].values,
                min_splits=min_splits,
            )
            value = info.get("scale")
            scale_value = float(value) if value is not None else 20.0
            return scale_value, info

        scale, scale_info = _tune_scale(train_preds)
        preds = model.predict(features.iloc[val_idx], clip=False)
        std_prediction = float(np.std(preds))
        guard_info: Dict[str, Any] = {
            "triggered": False,
            "threshold": args.std_guard_threshold,
            "initial_std_prediction": std_prediction,
            "model_type": active_model_type,
        }

        if args.std_guard_threshold > 0 and std_prediction < args.std_guard_threshold:
            guard_info["triggered"] = True
            fallback_model_type = args.std_guard_fallback_model
            if fallback_model_type and fallback_model_type != active_model_type:
                fallback_params = get_model_params(fallback_model_type)
                fallback_model = HullModel(fallback_model_type, model_params=fallback_params)
                fallback_model.fit(features.iloc[train_idx], target.iloc[train_idx])
                model = fallback_model
                active_model_type = fallback_model_type
                train_preds = model.predict(features.iloc[train_idx], clip=False)
                scale, scale_info = _tune_scale(train_preds)
                preds = model.predict(features.iloc[val_idx], clip=False)
                guard_info["fallback_model_type"] = fallback_model_type
                std_prediction = float(np.std(preds))
            if args.std_guard_min_scale is not None:
                scale = max(scale, float(args.std_guard_min_scale))
                guard_info["min_scale_enforced"] = scale
            guard_info["final_std_prediction"] = float(np.std(preds))
            std_prediction = guard_info["final_std_prediction"]
            print(
                f"⚠️ Std guard triggered on fold {fold}: "
                f"std={std_prediction:.4f} (threshold={args.std_guard_threshold}), "
                f"model={active_model_type}"
            )

        allocations = scale_to_allocation(preds, scale=scale)
        std_allocation = float(np.std(allocations))
        mse = mean_squared_error(target.iloc[val_idx], preds)
        overlay = apply_volatility_overlay(
            allocations,
            target.iloc[val_idx].values,
            lookback=args.overlay_lookback,
            min_periods=args.overlay_min_periods,
            volatility_cap=args.overlay_vol_cap,
        )
        overlay_allocations = overlay["allocations"]
        std_overlay_allocation = float(np.std(overlay_allocations))
        bt = backtest_strategy(allocations, target.iloc[val_idx].values)
        bt_overlay = backtest_strategy(overlay_allocations, target.iloc[val_idx].values)
        oof_preds[val_idx] = preds
        oof_allocations[val_idx] = allocations
        oof_overlay_allocations[val_idx] = overlay_allocations
        fold_metrics.append(
            {
                "fold": fold,
                "mse": mse,
                "scale": scale,
                "scale_cv_sharpe": scale_info.get("cv_sharpe", np.nan),
                "sharpe": bt["strategy_sharpe"],
                "total_return": bt["strategy_total_return"],
                "overlay_sharpe": bt_overlay["strategy_sharpe"],
                "overlay_total_return": bt_overlay["strategy_total_return"],
                "overlay_mean_scale": float(np.mean(overlay["scaling_factors"])),
                "overlay_breaches": overlay.get("breaches", 0),
                "std_prediction": std_prediction,
                "std_allocation": std_allocation,
                "std_overlay_allocation": std_overlay_allocation,
                "std_guard_triggered": guard_info["triggered"],
                "model_type_used": active_model_type,
            }
        )
        print(
            f"Fold {fold}: "
            f"MSE={mse:.6f}, "
            f"Scale={scale:.2f}, "
            f"CVSharpe={scale_info.get('cv_sharpe', float('nan')):.4f}, "
            f"StdPred={std_prediction:.4f}, "
            f"StdAlloc={std_allocation:.4f}, "
            f"Sharpe={bt['strategy_sharpe']:.4f}->{bt_overlay['strategy_sharpe']:.4f}, "
            f"TotalReturn={bt['strategy_total_return']:.4f}"
        )

    metrics_df = pd.DataFrame(fold_metrics)
    print("\nAggregated metrics:")
    print(metrics_df.mean(numeric_only=True))

    valid_mask = ~np.isnan(oof_preds)
    if valid_mask.any():
        valid_idx = np.where(valid_mask)[0]
        y_true = target.iloc[valid_idx].values
        oof_mse = mean_squared_error(y_true, oof_preds[valid_mask])
        oof_bt = backtest_strategy(oof_allocations[valid_mask], y_true)
        oof_overlay_bt = backtest_strategy(oof_overlay_allocations[valid_mask], y_true)
        print(
            "\nOut-of-fold metrics: "
            f"MSE={oof_mse:.6f}, "
            f"Sharpe={oof_bt['strategy_sharpe']:.4f}, "
            f"OverlaySharpe={oof_overlay_bt['strategy_sharpe']:.4f}"
        )

        scale_values = [m.get("scale") for m in fold_metrics if m.get("scale") is not None]
        preferred_scale = float(np.median(scale_values)) if scale_values else None
        def _to_native(value: Any):
            if isinstance(value, (np.floating,)):
                return float(value)
            if isinstance(value, (np.integer,)):
                return int(value)
            return value

        fold_payload = [{k: _to_native(v) for k, v in entry.items()} for entry in fold_metrics]

        std_payload = {
            "predictions": float(np.std(oof_preds[valid_mask])),
            "allocations": float(np.std(oof_allocations[valid_mask])),
            "overlay_allocations": float(np.std(oof_overlay_allocations[valid_mask])),
        }

        payload = {
            "model_type": args.model_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_rows": int(n_rows),
            "n_splits": args.n_splits,
            "feature_count": int(features.shape[1]),
            "preferred_scale": preferred_scale,
            "fold_metrics": fold_payload,
            "oof_metrics": {
                "mse": float(oof_mse),
                "sharpe": float(oof_bt["strategy_sharpe"]),
                "overlay_sharpe": float(oof_overlay_bt["strategy_sharpe"]),
                "overlay_total_return": float(oof_overlay_bt["strategy_total_return"]),
            },
            "std_deviation": std_payload,
            "std_guard": {
                "threshold": args.std_guard_threshold,
                "fallback_model": args.std_guard_fallback_model,
                "min_scale": args.std_guard_min_scale,
                "triggered_folds": int(sum(1 for m in fold_metrics if m.get("std_guard_triggered"))),
            },
            "overlay_config": {
                "lookback": args.overlay_lookback,
                "min_periods": args.overlay_min_periods,
                "volatility_cap": args.overlay_vol_cap,
            },
        }
        update_oof_artifact(log_paths.oof_metrics, args.model_type, payload)
        save_metrics(
            {
                "context": "train_experiment",
                "model_type": args.model_type,
                "std_prediction": std_payload["predictions"],
                "std_allocation": std_payload["allocations"],
                "std_overlay_allocation": std_payload["overlay_allocations"],
                "oof_sharpe": float(oof_bt["strategy_sharpe"]),
                "oof_overlay_sharpe": float(oof_overlay_bt["strategy_sharpe"]),
                "preferred_scale": preferred_scale if preferred_scale is not None else float("nan"),
                "std_guard_triggered_folds": int(
                    sum(1 for m in fold_metrics if m.get("std_guard_triggered"))
                ),
                "std_guard_threshold": args.std_guard_threshold,
            },
            log_paths.metrics_csv,
        )
        print(f"💾 OOF artefact saved to {log_paths.oof_metrics} (scale={preferred_scale})")


if __name__ == "__main__":
    main()
