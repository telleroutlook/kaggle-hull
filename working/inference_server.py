#!/usr/bin/env python3
"""Inference server entrypoint for the Hull Tactical competition."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Set, cast

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

from lib.data import load_train_data, get_feature_columns  # noqa: E402
from lib.env import detect_run_environment, get_data_paths  # noqa: E402
from lib.models import HullModel  # noqa: E402


STATE: Dict[str, object] = {
    "model": None,
    "feature_columns": None,
    "fill_values": None,
    "numeric_features": None,
}


def _ensure_model_initialized() -> None:
    """Train the baseline model once and cache everything needed for inference."""

    if STATE["model"] is not None:
        return

    env = detect_run_environment()
    data_paths = get_data_paths(env)
    train_df = load_train_data(data_paths)
    feature_columns = list(get_feature_columns(train_df))

    features = train_df[feature_columns].copy()
    numeric_cols = set(features.select_dtypes(include=[np.number]).columns)

    fill_values: Dict[str, object] = {}
    for col in feature_columns:
        series = features[col]
        if col in numeric_cols:
            value = float(series.median(skipna=True)) if series.notnull().any() else 0.0
        else:
            modes = series.mode(dropna=True)
            value = modes.iloc[0] if not modes.empty else "missing"
        features[col] = pd.to_numeric(series, errors="coerce") if col in numeric_cols else series
        features[col] = features[col].fillna(value)
        fill_values[col] = value

    features = features.fillna(0)
    target = train_df["forward_returns"].fillna(train_df["forward_returns"].median())

    model = HullModel(model_type="baseline")
    model.fit(features, target)

    STATE["model"] = model
    STATE["feature_columns"] = feature_columns
    STATE["fill_values"] = fill_values
    STATE["numeric_features"] = numeric_cols

    print(f"✅ Trained baseline model on {len(train_df):,} rows using {len(feature_columns)} features.")


def _ensure_pandas(df) -> pd.DataFrame:
    """Convert Polars/Pandas batches to a pandas.DataFrame copy."""

    if isinstance(df, pd.DataFrame):
        return df.copy()

    to_pandas = getattr(df, "to_pandas", None)
    if callable(to_pandas):
        return to_pandas()

    raise TypeError(f"Unsupported batch type: {type(df)!r}")


def _prepare_features(batch_df: pd.DataFrame) -> pd.DataFrame:
    feature_columns = cast(List[str], STATE["feature_columns"])
    fill_values = cast(Dict[str, object], STATE["fill_values"])
    numeric_features = cast(Set[str], STATE["numeric_features"])

    # Align to the training feature order and inject any missing columns.
    features = batch_df.reindex(columns=feature_columns, fill_value=np.nan)

    for col in feature_columns:
        fill_value = fill_values.get(col, 0.0)
        if col in numeric_features:
            features[col] = pd.to_numeric(features[col], errors="coerce").fillna(fill_value)
        else:
            features[col] = features[col].fillna(fill_value)

    return features.fillna(0)


def predict(test_batch):
    """Return allocations for each row in the incoming batch."""

    _ensure_model_initialized()

    batch_df = _ensure_pandas(test_batch)
    if batch_df.empty:
        return pd.DataFrame({"prediction": []})

    features = _prepare_features(batch_df)
    model: HullModel = STATE["model"]  # type: ignore[assignment]
    predictions = model.predict(features)
    predictions = np.clip(predictions, 0.0, 2.0)

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
