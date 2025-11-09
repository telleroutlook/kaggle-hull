#!/usr/bin/env python3
"""Inference server entrypoint for the Hull Tactical competition."""

from __future__ import annotations

import os
import shutil
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, cast

# 配置警告处理 - 在所有其他导入之前
def configure_warnings_early():
    """配置警告处理以避免pandas比较警告"""
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', 
                          message='.*invalid value encountered in greater.*',
                          category=RuntimeWarning)
    warnings.filterwarnings('ignore',
                          message='.*invalid value encountered in less.*', 
                          category=RuntimeWarning)

# 立即配置警告处理
configure_warnings_early()

import numpy as np
import pandas as pd

# 导入警告处理器
try:
    from warnings_handler import ensure_warnings_configured
    ensure_warnings_configured()
except ImportError:
    pass

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

from lib.artifacts import (  # noqa: E402
    load_first_available_oof,
    oof_artifact_candidates,
    update_oof_artifact,
)
from lib.data import load_training_frame  # noqa: E402
from lib.env import (  # noqa: E402
    PROJECT_ROOT,
    detect_run_environment,
    get_data_paths,
    get_log_paths,
)
from lib.features import FeaturePipeline, build_feature_pipeline, pipeline_config_hash  # noqa: E402
from lib.model_registry import get_model_params, resolve_model_type  # noqa: E402
from lib.models import HullModel  # noqa: E402
from lib.strategy import (  # noqa: E402
    VolatilityOverlay,
    optimize_scale_with_rolling_cv,
    scale_to_allocation,
    tune_allocation_scale,
)


TRUE_STRINGS = {"1", "true", "yes", "on"}
ALLOW_MISSING_OOF = os.getenv("HULL_ALLOW_MISSING_OOF", "0").lower() in TRUE_STRINGS
FORCE_RECALIBRATE = os.getenv("HULL_FORCE_RECALIBRATE", "0").lower() in TRUE_STRINGS

STATE: Dict[str, object] = {
    "model": None,
    "pipeline": None,
    "allocation_scale": 20.0,
    "overlay": None,
    "overlay_config": None,
    "pipeline_config_hash": None,
    "training_metadata": None,
}


def _normalize_for_json(value: Any) -> Any:
    """Convert numpy/scalar containers into JSON-friendly primitives."""

    import numpy as _np

    if isinstance(value, _np.generic):
        return value.item()
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, dict):
        return {k: _normalize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_json(v) for v in value]
    return value


def _overlay_target_from_env() -> float | None:
    raw = os.getenv("HULL_OVERLAY_TARGET_QUANTILE")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


OVERLAY_TARGET_QUANTILE = _overlay_target_from_env()


def _resolve_overlay_config(artifact_entry: dict | None) -> dict[str, Any]:
    """Return the overlay configuration stored in the artefact or the defaults."""

    overlay_cfg = dict(artifact_entry.get("overlay_config", {})) if artifact_entry else {}
    if not overlay_cfg:
        overlay_cfg = {
            "lookback": 63,
            "min_periods": 21,
            "volatility_cap": 1.2,
        }
    if OVERLAY_TARGET_QUANTILE is not None:
        overlay_cfg.setdefault("target_volatility_quantile", OVERLAY_TARGET_QUANTILE)
    return overlay_cfg


def _mirror_oof_to_repo(source: Path) -> None:
    """Copy refreshed artefacts into the project tree so they ship with the build."""

    if not source.exists():
        return
    packaged = PROJECT_ROOT / "working" / "artifacts" / source.name
    try:
        packaged.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve(strict=False) == packaged.resolve(strict=False):
            return
    except OSError:
        return

    try:
        shutil.copy2(source, packaged)
        print(f"📦 Mirrored recalibrated artefact to {packaged}")
    except OSError as exc:
        print(f"⚠️ Unable to mirror artefact to {packaged}: {exc}")


def _persist_recalibrated_oof(
    *,
    model_type: str,
    allocation_scale: float,
    pipeline_config: dict[str, Any],
    pipeline_hash: str,
    training_meta: dict,
    overlay_config: dict[str, Any],
    tuning_result: dict[str, Any],
    recalibration_reasons: list[str],
    log_paths,
    feature_count: int,
    n_rows: int,
) -> None:
    """Write the freshly tuned scale/overlay back into the artefact JSON."""

    payload = {
        "model_type": model_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_rows": n_rows,
        "feature_count": feature_count,
        "preferred_scale": float(allocation_scale),
        "pipeline_config": pipeline_config,
        "pipeline_config_hash": pipeline_hash,
        "augment_data": bool(training_meta.get("augment_data", False)),
        "augmentation_factor": float(training_meta.get("augmentation_factor", 0.0)),
        "overlay_config": overlay_config,
        "calibration_reasons": recalibration_reasons,
        "tuning_result": _normalize_for_json(tuning_result),
    }
    sharpe = tuning_result.get("strategy_sharpe") or tuning_result.get("cv_sharpe")
    if sharpe is not None:
        payload["oof_metrics"] = {"sharpe": float(sharpe)}

    update_oof_artifact(log_paths.oof_metrics, model_type, payload)
    print(f"💾 Recalibrated artefact saved to {log_paths.oof_metrics} (scale={allocation_scale:.2f})")
    _mirror_oof_to_repo(log_paths.oof_metrics)


def _ensure_model_initialized() -> None:
    """Train the baseline model once and cache everything needed for inference."""

    if STATE["model"] is not None:
        return

    env = detect_run_environment()
    data_paths = get_data_paths(env)
    log_paths = get_log_paths(env)
    train_df, training_meta = load_training_frame(data_paths, return_metadata=True)
    pipeline = build_feature_pipeline(stateful=True)
    pipeline_config = pipeline.to_config()
    pipeline_hash = pipeline_config_hash(
        pipeline_config,
        augment_flag=training_meta["augment_data"],
        extra={"augmentation_factor": training_meta["augmentation_factor"]},
    )
    features = pipeline.fit_transform(train_df)
    target = train_df["forward_returns"].fillna(train_df["forward_returns"].median())

    requested_env = os.getenv("HULL_MODEL_TYPE")
    model_type = resolve_model_type(None)
    if requested_env:
        print(f"ℹ️ HULL_MODEL_TYPE={requested_env} -> resolved '{model_type}' for inference server")
    else:
        print(f"ℹ️ HULL_MODEL_TYPE not set; defaulting to '{model_type}' for inference server")
    model_params = get_model_params(model_type)
    model = HullModel(model_type=model_type, model_params=model_params)
    model.fit(features, target)

    artifact_entry, artifact_path = load_first_available_oof(
        model_type, oof_artifact_candidates(log_paths)
    )
    recalibration_reasons: list[str] = []
    if artifact_entry:
        print(f"🧾 Loaded OOF artefact from {artifact_path}")
    else:
        if not ALLOW_MISSING_OOF:
            raise RuntimeError(
                "OOF artefact missing. Set HULL_ALLOW_MISSING_OOF=1 if you intentionally want to "
                "recalibrate inside the inference server."
            )
        recalibration_reasons.append("missing artefact")

    overlay_config = _resolve_overlay_config(artifact_entry)
    stored_scale = artifact_entry.get("preferred_scale") if artifact_entry else None
    stored_hash = artifact_entry.get("pipeline_config_hash") if artifact_entry else None
    stored_aug = artifact_entry.get("augment_data") if artifact_entry else None

    if artifact_entry:
        if stored_hash is None:
            recalibration_reasons.append("artefact missing pipeline_config_hash")
        elif stored_hash != pipeline_hash:
            recalibration_reasons.append("pipeline_config_hash mismatch")
        if stored_aug is not None and bool(stored_aug) != bool(training_meta["augment_data"]):
            recalibration_reasons.append("augment flag mismatch")
        if stored_scale is None:
            recalibration_reasons.append("missing preferred_scale")
    if FORCE_RECALIBRATE:
        recalibration_reasons.append("HULL_FORCE_RECALIBRATE=1")

    raw_predictions = model.predict(features, clip=False)
    tuning_result: Dict[str, float] = {}
    allocation_scale: float
    if artifact_entry and not recalibration_reasons:
        allocation_scale = float(stored_scale)  # type: ignore[arg-type]
        print(
            f"♻️ Using OOF allocation scale {allocation_scale:.2f} "
            f"from artefact timestamp {artifact_entry.get('timestamp')}"
        )
    else:
        if recalibration_reasons:
            print(
                "⚖️ Recalibrating allocation scale because "
                + "; ".join(recalibration_reasons)
            )
        tuning_result = optimize_scale_with_rolling_cv(raw_predictions, target.to_numpy())
        allocation_scale = tuning_result.get("scale", 20.0)
        if allocation_scale is None:
            tuning_fallback = tune_allocation_scale(raw_predictions, target.to_numpy())
            allocation_scale = tuning_fallback.get("scale", 20.0)
            tuning_result = tuning_fallback
        if not artifact_entry:
            print("⚠️ OOF artefact missing; using locally calibrated allocation scale.")
        _persist_recalibrated_oof(
            model_type=model_type,
            allocation_scale=allocation_scale,
            pipeline_config=pipeline_config,
            pipeline_hash=pipeline_hash,
            training_meta=training_meta,
            overlay_config=overlay_config,
            tuning_result=tuning_result,
            recalibration_reasons=recalibration_reasons or ["missing artefact"],
            log_paths=log_paths,
            feature_count=features.shape[1],
            n_rows=len(train_df),
        )

    STATE["model"] = model
    STATE["pipeline"] = pipeline
    STATE["allocation_scale"] = allocation_scale
    STATE["overlay_config"] = overlay_config
    STATE["overlay"] = None
    STATE["pipeline_config_hash"] = pipeline_hash
    STATE["training_metadata"] = training_meta

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
            overlay_cfg = cast(dict | None, STATE.get("overlay_config"))
            overlay_params = overlay_cfg or {}
            overlay_state = VolatilityOverlay(
                lookback=overlay_params.get("lookback", 63),
                min_periods=overlay_params.get("min_periods", 21),
                volatility_cap=overlay_params.get("volatility_cap", 1.2),
                reference_is_lagged=True,
                target_volatility_quantile=overlay_params.get(
                    "target_volatility_quantile", OVERLAY_TARGET_QUANTILE
                ),
            )
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
