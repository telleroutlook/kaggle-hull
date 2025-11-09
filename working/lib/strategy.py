"""Strategy utilities for converting raw forecasts into risk-aware allocations."""

from __future__ import annotations

from collections import deque
from typing import Iterable, Dict, Any, Deque, List

import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from .evaluation import backtest_strategy

DEFAULT_SCALES = np.linspace(5.0, 40.0, 15)


def scale_to_allocation(predictions: np.ndarray, *, scale: float, midpoint: float = 1.0) -> np.ndarray:
    """Map unconstrained predictions to [0, 2] leverage weights."""

    allocations = midpoint + predictions * scale
    return np.clip(allocations, 0.0, 2.0)


def tune_allocation_scale(
    predictions: np.ndarray,
    forward_returns: np.ndarray,
    *,
    candidate_scales: Iterable[float] | None = None,
) -> Dict[str, Any]:
    """Grid-search the leverage scale that maximizes Sharpe on the training set."""

    if candidate_scales is None:
        candidate_scales = DEFAULT_SCALES

    best_scale = 20.0
    best_metrics: Dict[str, float] = {"strategy_sharpe": -np.inf}

    for scale in candidate_scales:
        allocations = scale_to_allocation(predictions, scale=scale)
        metrics = backtest_strategy(allocations, forward_returns)
        if metrics["strategy_sharpe"] > best_metrics["strategy_sharpe"]:
            best_scale = scale
            best_metrics = metrics
            best_metrics["scale"] = scale

    return best_metrics


def optimize_scale_with_rolling_cv(
    predictions: np.ndarray,
    forward_returns: np.ndarray,
    *,
    candidate_scales: Iterable[float] | None = None,
    min_splits: int = 3,
    max_splits: int = 6,
) -> Dict[str, Any]:
    """Choose a leverage scale via rolling (time-series) cross validation.

    Unlike :func:`tune_allocation_scale`, this routine evaluates candidate scales
    on walk-forward validation windows that live *inside* the provided training
    sample. Only the training data for the outer fold is used, which helps
    prevent optimistic bias when later evaluating on held-out folds.
    """

    preds = np.asarray(predictions, dtype=float)
    returns = np.asarray(forward_returns, dtype=float)
    n_samples = len(preds)

    if candidate_scales is None:
        candidate_scales = DEFAULT_SCALES

    # Guard against degenerate splits on very small samples.
    max_allowed_splits = max(1, n_samples - 1)
    n_splits = min(max_splits, max(min_splits, 2))
    n_splits = min(n_splits, max_allowed_splits)

    if n_splits < 2:
        fallback = tune_allocation_scale(preds, returns, candidate_scales=candidate_scales)
        fallback.setdefault("cv_sharpe", fallback.get("strategy_sharpe", 0.0))
        fallback["n_splits"] = 1
        return fallback

    splitter = TimeSeriesSplit(n_splits=n_splits)
    scale_scores: Dict[float, list[float]] = {float(scale): [] for scale in candidate_scales}

    for _, val_idx in splitter.split(preds):
        val_preds = preds[val_idx]
        val_returns = returns[val_idx]
        for scale in scale_scores:
            allocations = scale_to_allocation(val_preds, scale=scale)
            metrics = backtest_strategy(allocations, val_returns)
            scale_scores[scale].append(metrics["strategy_sharpe"])

    averaged_scores = {
        scale: float(np.mean(scores)) if scores else float("-inf")
        for scale, scores in scale_scores.items()
    }
    best_scale = max(averaged_scores, key=averaged_scores.get)
    return {
        "scale": best_scale,
        "cv_sharpe": averaged_scores[best_scale],
        "scale_scores": averaged_scores,
        "n_splits": n_splits,
    }


class VolatilityOverlay:
    """Rolling volatility cap that enforces the 120% constraint."""

    def __init__(
        self,
        *,
        lookback: int = 63,
        min_periods: int | None = None,
        volatility_cap: float = 1.2,
        clip_bounds: tuple[float, float] = (0.0, 2.0),
        reference_is_lagged: bool = False,
        target_volatility_quantile: float | None = None,
    ) -> None:
        self.lookback = max(2, lookback)
        self.min_periods = min_periods or max(10, self.lookback // 3)
        self.volatility_cap = volatility_cap
        self.clip_bounds = clip_bounds
        self.reference_is_lagged = reference_is_lagged
        self.target_volatility_quantile = target_volatility_quantile

        self.market_returns: Deque[float] = deque()
        self.strategy_returns: Deque[float] = deque()
        self.prev_allocation: float | None = None
        self.breaches: int = 0
        self.scaling_history: List[float] = []
        self.target_history: List[float] = []

    def _append_realized(self, realized_return: float) -> None:
        # Check for finite values before processing
        if not np.isfinite(realized_return) or self.prev_allocation is None or not np.isfinite(self.prev_allocation):
            return
            
        # Additional safety check for reasonable values
        if abs(realized_return) > 10.0:  # Sanity check for extreme returns
            return
            
        try:
            self.market_returns.append(float(realized_return))
            strategy_return = float(self.prev_allocation * realized_return)
            # Check for reasonable strategy return
            if abs(strategy_return) < 100.0:  # Sanity check
                self.strategy_returns.append(strategy_return)
            if len(self.market_returns) > self.lookback:
                self.market_returns.popleft()
                self.strategy_returns.popleft()
        except (OverflowError, ValueError):
            # Silently skip problematic values
            pass

    def _calculate_target_vol(self) -> float:
        if not self.market_returns:
            return 0.0
        market_series = np.asarray(self.market_returns, dtype=float)
        
        # Ensure no NaN/inf values in market series
        market_series = np.where(np.isfinite(market_series), market_series, 0.0)
        
        if len(market_series) == 0 or not np.any(np.isfinite(market_series)):
            return 1e-6
            
        base_target = float(np.std(market_series)) * self.volatility_cap
        if (
            self.target_volatility_quantile is not None
            and 0 < self.target_volatility_quantile < 1
            and len(market_series) >= self.min_periods
        ):
            adaptive = float(np.quantile(np.abs(market_series), self.target_volatility_quantile))
            base_target = max(adaptive, 1e-6)
        return max(base_target, 1e-6)

    def _compute_scale(self) -> tuple[float, float]:
        if len(self.market_returns) < self.min_periods or len(self.strategy_returns) < self.min_periods:
            return 1.0, self._calculate_target_vol()

        # Safe computation with NaN/inf protection
        strategy_returns = np.array(self.strategy_returns, dtype=float)
        market_returns = np.array(self.market_returns, dtype=float)
        
        # Replace any remaining NaN/inf values
        strategy_returns = np.where(np.isfinite(strategy_returns), strategy_returns, 0.0)
        market_returns = np.where(np.isfinite(market_returns), market_returns, 0.0)
        
        strat_vol = float(np.std(strategy_returns))
        target_vol = self._calculate_target_vol()
        
        if strat_vol > target_vol and strat_vol > 1e-6:
            self.breaches += 1
            scale = target_vol / max(strat_vol, 1e-6)
            return scale, target_vol
        return 1.0, target_vol

    def transform(self, allocations: np.ndarray, reference_returns: np.ndarray) -> Dict[str, np.ndarray]:
        """Apply the overlay to a sequence of allocations.

        Args:
            allocations: raw leverage weights in chronological order.
            reference_returns: either realized returns (if ``reference_is_lagged``
                is False) or the lagged returns made available in the test feed.
        Returns:
            Dict with ``allocations`` and ``scaling_factors`` arrays.
        """

        # Validate and clean input data to prevent warnings
        allocations = np.asarray(allocations, dtype=float)
        reference_returns = np.asarray(reference_returns, dtype=float)
        
        # Replace NaN and inf values with safe defaults
        allocations = np.where(np.isfinite(allocations), allocations, 1.0)
        reference_returns = np.where(np.isfinite(reference_returns), reference_returns, 0.0)
        
        scaled = np.zeros_like(allocations, dtype=float)
        scaling_factors = np.ones_like(allocations, dtype=float)
        refs = reference_returns

        for idx, raw_allocation in enumerate(np.asarray(allocations, dtype=float)):
            if self.reference_is_lagged:
                realized = refs[idx]
                if np.isfinite(realized):
                    self._append_realized(realized)
            else:
                # Use returns up to idx-1 only by appending after scaling.
                if idx > 0 and np.isfinite(refs[idx - 1]):
                    self._append_realized(refs[idx - 1])

            scale_factor, target_vol = self._compute_scale()
            scaling_factors[idx] = scale_factor
            scaled_value = float(np.clip(raw_allocation * scale_factor, *self.clip_bounds))
            scaled[idx] = scaled_value
            self.prev_allocation = scaled_value
            self.scaling_history.append(scale_factor)
            self.target_history.append(target_vol)

            if not self.reference_is_lagged and np.isfinite(refs[idx]):
                # Realized return for the same timestamp becomes available after
                # the allocation is placed; append it for the next step.
                self._append_realized(refs[idx])

        target_series = np.array(self.target_history[-len(scaled) :], dtype=float)
        return {
            "allocations": scaled,
            "scaling_factors": scaling_factors,
            "target_volatility": target_series,
        }


def apply_volatility_overlay(
    allocations: np.ndarray,
    realized_returns: np.ndarray,
    *,
    lookback: int = 63,
    min_periods: int | None = None,
    volatility_cap: float = 1.2,
    reference_is_lagged: bool = False,
    target_volatility_quantile: float | None = None,
) -> Dict[str, Any]:
    """Convenience wrapper that instantiates :class:`VolatilityOverlay`."""

    overlay = VolatilityOverlay(
        lookback=lookback,
        min_periods=min_periods,
        volatility_cap=volatility_cap,
        reference_is_lagged=reference_is_lagged,
        target_volatility_quantile=target_volatility_quantile,
    )
    result = overlay.transform(allocations, realized_returns)
    return {**result, "breaches": overlay.breaches}


__all__ = [
    "DEFAULT_SCALES",
    "scale_to_allocation",
    "tune_allocation_scale",
    "optimize_scale_with_rolling_cv",
    "VolatilityOverlay",
    "apply_volatility_overlay",
]
