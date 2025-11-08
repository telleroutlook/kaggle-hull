#!/usr/bin/env python3
"""Inference server entrypoint for the Hull Tactical competition."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, cast

import numpy as np
import pandas as pd

# Ensure the Kaggle evaluation helpers are importable in both local and Kaggle environments.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_SEARCH_ROOTS = [
    Path("/kaggle/input/hull-tactical-market-prediction"),
    PROJECT_ROOT / "input" / "hull-tactical-market-prediction",
]

for candidate in EVALUATION_SEARCH_ROOTS:
    pkg_root = candidate / "kaggle_evaluation"
    if pkg_root.exists():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break
else:
    raise RuntimeError(
        "kaggle_evaluation package not found. Please add the official competition data set "
        "as an input when running this script."
    )

import kaggle_evaluation.default_inference_server  # type: ignore  # noqa: E402

from lib.data import load_train_data  # noqa: E402
from lib.env import detect_run_environment, get_data_paths  # noqa: E402
from lib.features import FeaturePipeline  # noqa: E402
from lib.model_registry import get_model_params, resolve_model_type  # noqa: E402
from lib.models import HullModel  # noqa: E402
from lib.strategy import (  # noqa: E402
    VolatilityOverlay,
    optimize_scale_with_rolling_cv,
    scale_to_allocation,
    tune_allocation_scale,
)


STATE: Dict[str, object] = {
    "model": None,
    "pipeline": None,
    "allocation_scale": 20.0,
    "overlay": None,
}


def _ensure_model_initialized() -> None:
    """Train the baseline model once and cache everything needed for inference."""

    if STATE["model"] is not None:
        return

    env = detect_run_environment()
    data_paths = get_data_paths(env)
    train_df = load_train_data(data_paths)
    pipeline = FeaturePipeline()
    features = pipeline.fit_transform(train_df)
    target = train_df["forward_returns"].fillna(train_df["forward_returns"].median())

    model_type = resolve_model_type(None)
    model_params = get_model_params(model_type)
    model = HullModel(model_type=model_type, model_params=model_params)
    model.fit(features, target)

    raw_predictions = model.predict(features, clip=False)
    tuning_result = optimize_scale_with_rolling_cv(raw_predictions, target.to_numpy())
    allocation_scale = tuning_result.get("scale", 20.0)
    if allocation_scale is None:
        tuning_fallback = tune_allocation_scale(raw_predictions, target.to_numpy())
        allocation_scale = tuning_fallback.get("scale", 20.0)
        tuning_result = tuning_fallback

    STATE["model"] = model
    STATE["pipeline"] = pipeline
    STATE["allocation_scale"] = allocation_scale

    print(
        f"✅ Trained {model_type} model on {len(train_df):,} rows "
        f"using {features.shape[1]} engineered features."
    )
    print(
        f"🎯 Calibrated allocation scale={allocation_scale:.2f} "
        f"(Sharpe {tuning_result.get('strategy_sharpe', tuning_result.get('cv_sharpe', 0)):.4f})"
    )


def _ensure_pandas(df) -> pd.DataFrame:
    """Convert Polars/Pandas batches to a pandas.DataFrame copy."""

    if isinstance(df, pd.DataFrame):
        return df.copy()

    to_pandas = getattr(df, "to_pandas", None)
    if callable(to_pandas):
        return to_pandas()

    raise TypeError(f"Unsupported batch type: {type(df)!r}")


def _prepare_features(batch_df: pd.DataFrame) -> pd.DataFrame:
    pipeline = cast(FeaturePipeline, STATE["pipeline"])
    return pipeline.transform(batch_df)


def predict(test_batch):
    """Return allocations for each row in the incoming batch."""

    _ensure_model_initialized()

    batch_df = _ensure_pandas(test_batch)
    if batch_df.empty:
        return pd.DataFrame({"prediction": []})

    features = _prepare_features(batch_df)
    model: HullModel = STATE["model"]  # type: ignore[assignment]
    raw_predictions = model.predict(features, clip=False)
    allocation_scale = cast(float, STATE.get("allocation_scale", 20.0))
    predictions = scale_to_allocation(raw_predictions, scale=allocation_scale)

    overlay_source = None
    if "lagged_forward_returns" in batch_df.columns:
        overlay_source = batch_df["lagged_forward_returns"].to_numpy()
    elif "lagged_market_forward_excess_returns" in batch_df.columns:
        overlay_source = batch_df["lagged_market_forward_excess_returns"].to_numpy()

    if overlay_source is not None:
        overlay_state = cast(
            VolatilityOverlay | None,
            STATE.get("overlay"),
        )
        if overlay_state is None:
            overlay_state = VolatilityOverlay(reference_is_lagged=True)
            STATE["overlay"] = overlay_state
        overlay_result = overlay_state.transform(predictions, overlay_source)
        predictions = overlay_result["allocations"]

    return pd.DataFrame({"prediction": predictions.astype(np.float32)})


def _ensure_writable_workdir(env: str) -> Path:
    """On Kaggle the code may live in /kaggle/input (read-only). Move to /kaggle/working."""

    if env != "kaggle":
        return Path.cwd()

    writable_root = Path("/kaggle/working")
    try:
        writable_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:  # pragma: no cover - very unlikely on Kaggle
        print(f"⚠️ Unable to prepare {writable_root}: {exc}")
        return Path.cwd()

    current = Path.cwd()
    if current.resolve() != writable_root.resolve():
        print(f"📁 Switching working directory from {current} to {writable_root} for writable outputs.")
        os.chdir(writable_root)

    return writable_root


def main() -> None:
    inference_server = kaggle_evaluation.default_inference_server.DefaultInferenceServer(predict)

    env = detect_run_environment()
    _ensure_writable_workdir(env)
    data_root = str(get_data_paths(env).train_data.parent)

    if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        inference_server.serve()
    else:
        inference_server.run_local_gateway((data_root,))

    submission_path = Path.cwd() / "submission.parquet"
    if submission_path.exists():
        try:
            df = pd.read_parquet(submission_path)
            df.to_csv(submission_path.with_suffix(".csv"), index=False)
            print(f"✅ submission.csv written next to {submission_path.name}")
        except Exception as exc:  # pragma: no cover - best effort helper
            print(f"⚠️ Unable to export submission.csv: {exc}")


if __name__ == "__main__":
    main()
